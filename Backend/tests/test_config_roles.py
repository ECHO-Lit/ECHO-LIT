"""Configuration -- each image role runs with exactly what that image installs.

Test Plan Section 3.1.8.  See tests/plans/3.1.8-configuration-testing.md.

Module CC.  One Dockerfile builds two images: `ROLE=api` installs
`requirements-api.txt` only; `ROLE=worker` adds torch and the rest of
`requirements-worker.txt`.  The scheduler (Celery beat) runs in the API
image.  Whether each process can start is therefore a question about its
import closure, and it is answered in a fresh interpreter, where nothing
already imported can hide an import: a meta-path blocker refuses the ML
packages the API image does not have, and records who asked for them.

SRS DC-2: "The control plane shall carry no machine learning dependency and
shall perform no inference."
README: "the API image does not contain or import the ML runtime."
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

from tests import _config as cfg

pytestmark = [pytest.mark.configuration, pytest.mark.critical]

# What the worker image adds over the API image.
ML_PACKAGES = (
    "torch", "torchaudio", "transformers", "librosa", "sklearn", "umap", "captum", "lime",
    "shap", "accelerate", "jiwer", "pyloudnorm", "numba", "pandas", "hdbscan", "scipy",
)

# Packages an application module may import through the package that declares
# them, because they are that package's own public surface: FastAPI is built
# on Starlette and Pydantic; Celery ships kombu and billiard; boto3 is the
# client for botocore's exception types.  Anything else must be declared.
PUBLIC_SURFACE = {
    "starlette": "fastapi",
    "pydantic": "fastapi",
    "pydantic_core": "fastapi",
    "kombu": "celery",
    "billiard": "celery",
    "botocore": "boto3",
}

PROBE = r'''
import importlib.abc, json, sys
code, blocked = sys.argv[1], set(filter(None, sys.argv[2].split(",")))
attempts = []

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.partition(".")[0] not in blocked:
            return None
        frame, importer = sys._getframe(1), None
        while frame is not None:
            filename = frame.f_code.co_filename
            if not filename.startswith("<frozen") and "importlib" not in filename:
                importer = filename
                break
            frame = frame.f_back
        attempts.append([name, importer])
        raise ImportError(f"{name} is not installed in this image role")

if blocked:
    sys.meta_path.insert(0, Blocker())
result = {"ok": True, "error": None}
try:
    exec(code, {})
except BaseException as exc:
    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
result["attempts"] = attempts
result["app_modules"] = sorted(n for n in sys.modules if n == "app" or n.startswith("app."))
print("PROBE-RESULT " + json.dumps(result))
'''


def _probe(code: str, *, blocked=ML_PACKAGES, extra_env: dict[str, str] | None = None) -> dict:
    env = {key: value for key, value in os.environ.items() if key.upper() not in cfg.SETTINGS_KEYS}
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        STORAGE_LOCAL_ROOT=str(Path(os.environ.get("TEMP", "/tmp")) / "echo-config-probe"),
    )
    env.update(extra_env or {})
    completed = subprocess.run(
        [sys.executable, "-B", "-c", PROBE, code, ",".join(blocked)],
        cwd=cfg.BACKEND,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    line = next((l for l in completed.stdout.splitlines() if l.startswith("PROBE-RESULT ")), None)
    assert line, f"probe produced no result\nstdout:\n{completed.stdout[-2000:]}\nstderr:\n{completed.stderr[-4000:]}"
    return json.loads(line.removeprefix("PROBE-RESULT "))


def _from_app(attempts: list[list[str]]) -> list[list[str]]:
    app_root = os.path.normcase(str(cfg.BACKEND / "app"))
    return [
        attempt for attempt in attempts
        if attempt[1] and os.path.normcase(attempt[1]).startswith(app_root)
    ]


@pytest.fixture(scope="module")
def api_role():
    return _probe("import app.main")


@pytest.fixture(scope="module")
def scheduler_role():
    # What `celery beat` does at startup: load the app, then import its task
    # modules (`include=`), in the ROLE=api image the scheduler is built from.
    return _probe(
        "from app.core.celery_app import celery_app\n"
        "celery_app.loader.import_default_modules()"
    )


# -- requirement files -------------------------------------------------------

def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared(path: Path) -> set[str]:
    """Distribution names a requirements file declares, following `-r`."""
    names = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r"):
            names |= _declared(path.parent / line[2:].strip())
            continue
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", line)
        names.add(_canonical(match.group(0)))
    return names


def _module_file(module: str) -> Path | None:
    base = cfg.BACKEND / Path(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _imports(path: Path, *, import_time_only: bool) -> set[str]:
    """Top-level names a module imports -- at import time, or anywhere."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()

    def visit(nodes):
        for node in nodes:
            if isinstance(node, ast.Import):
                names.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.partition(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and import_time_only:
                continue
            elif isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
                continue
            else:
                visit(ast.iter_child_nodes(node))

    visit(tree.body)
    return names


def _undeclared(imports: set[str], declared: set[str]) -> dict[str, list[str]]:
    distributions = packages_distributions()
    first_party = {"app", "tests", "conftest", "__future__"}
    missing = {}
    for name in sorted(imports - first_party - set(sys.stdlib_module_names)):
        providers = [_canonical(d) for d in distributions.get(name, [name])]
        if any(provider in declared for provider in providers):
            continue
        if PUBLIC_SURFACE.get(name) and _canonical(PUBLIC_SURFACE[name]) in declared:
            continue
        missing[name] = providers
    return missing


class TestRoleClosure:
    def test_the_api_role_imports_without_the_ml_runtime(self, api_role):
        """CF-60: DC-2 -- the control plane starts in an image with no ML packages."""
        assert api_role["ok"], api_role["error"]
        assert _from_app(api_role["attempts"]) == []

    def test_the_scheduler_role_loads_every_task_without_the_ml_runtime(self, scheduler_role):
        """CF-61: the scheduler is built as ROLE=api but loads the worker's task modules."""
        assert scheduler_role["ok"], scheduler_role["error"]
        assert _from_app(scheduler_role["attempts"]) == []
        assert "app.worker.tasks" in scheduler_role["app_modules"]

    def test_the_worker_role_imports_its_runtime(self):
        """CF-62: the worker image's own entry points import with its packages."""
        result = _probe(
            "import app.worker.tasks, app.worker.executor, app.worker.model_adapters, "
            "app.worker.model_registry, app.core.device",
            blocked=(),
        )
        assert result["ok"], result["error"]


class TestRequirementCoverage:
    def test_every_api_import_is_declared_for_the_api_image(self, api_role):
        """CF-63: guards BUG-74.

        The API imports numpy (`dataset_eda_service`, via the datasets
        routes) but its image never asked for it: it arrived unpinned through
        soundfile, while the worker image pins `numpy<2.0`.
        """
        imports: set[str] = set()
        for module in api_role["app_modules"]:
            path = _module_file(module)
            if path:
                imports |= _imports(path, import_time_only=True)
        assert _undeclared(imports, _declared(cfg.BACKEND / "requirements-api.txt")) == {}

    def test_every_worker_import_is_declared_for_the_worker_image(self):
        """CF-64: every package any application module imports, at any depth."""
        imports: set[str] = set()
        for path in (cfg.BACKEND / "app").rglob("*.py"):
            imports |= _imports(path, import_time_only=False)
        declared = _declared(cfg.BACKEND / "requirements-worker.txt")
        assert _undeclared(imports, declared) == {}

    def test_every_test_import_is_declared_for_the_dev_environment(self):
        """CF-65: guards BUG-74 -- the suite installs from `requirements-dev.txt`.

        FO-16 and this section read the compose file with PyYAML, which no
        requirement file declared.
        """
        imports: set[str] = set()
        for path in (cfg.BACKEND / "tests").glob("*.py"):
            imports |= _imports(path, import_time_only=False)
        declared = _declared(cfg.BACKEND / "requirements-dev.txt")
        assert _undeclared(imports, declared) == {}

    def test_the_legacy_routes_need_the_ml_runtime(self):
        """CF-66: OBS-47 -- why the API pins ENABLE_LEGACY_SYNC_INFERENCE off.

        The legacy synchronous routes import the ML stack, which the API
        image does not install: enabling them there stops the API starting.
        """
        result = _probe("import app.main", extra_env={"ENABLE_LEGACY_SYNC_INFERENCE": "true"})
        assert not result["ok"]
        assert "ImportError" in result["error"]
        assert any(name.partition(".")[0] in ML_PACKAGES for name, _ in _from_app(result["attempts"]))
