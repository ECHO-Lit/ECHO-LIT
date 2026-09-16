"""Risks, Dependencies, Assumptions and Constraints -- the register's guards.

Test Plan Section 5.  See tests/plans/5-risks-dependencies-assumptions-constraints.md.

Section 5 of the Master Test Plan is a register: risks with their mitigation
and contingency, dependencies, assumptions, constraints.  A register is only as
good as the safeguards it names, and the research for this section found three
that were claimed but not in force -- a clean checkout failed instead of
skipping (TEST-06), a wall-clock budget ran under the coverage tracer (TEST-07),
and offline mode was a property of one command rather than of the suite
(TEST-08).  So every mitigation the register calls in force, and every
assumption it calls proven, is checked here; RD-14 keeps the register and this
module in step.

No conftest fixture is requested.  Two cases (RD-02, RD-04) run pytest in a
child interpreter, because what they guard is what a fresh process sees.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests import _corpora, _report, run_tests

pytestmark = pytest.mark.important

TESTS_DIR = Path(__file__).resolve().parent
BACKEND = TESTS_DIR.parent
REPO = BACKEND.parent
FRONTEND = REPO / "Frontend"
APP = BACKEND / "app"
DATA = BACKEND / "data"
README = TESTS_DIR / "README.md"
SECTION_DOC = TESTS_DIR / "plans" / "5-risks-dependencies-assumptions-constraints.md"

TRUTHY = {"1", "ON", "YES", "TRUE"}

# Variables a child run must not inherit: offline mode has to come from the
# suite itself, and a coverage run's subprocess patch would trace the child.
_INHERITED = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
              "ECHO_MODEL_TESTS", "ECHO_KEEP_CORPORA", "PYTEST_ADDOPTS")


def _child_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k not in _INHERITED and not k.startswith("COVERAGE_PROCESS")}
    env.update(extra)
    return env


def _pytest(args: list[str], env: dict[str, str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=BACKEND, env=env, capture_output=True, text=True, timeout=timeout,
    )


# --------------------------------------------------------------------------- A. offline

class TestOffline:
    def test_offline_mode_is_active_in_process(self):
        """RD-01 -- guards TEST-08: no run may reach the Hugging Face Hub.

        The opt-out for the real-model cases (`ECHO_MODEL_TESTS=1
        HF_HUB_OFFLINE=0`) is the only way past it.
        """
        if os.environ.get("ECHO_MODEL_TESTS") and os.environ.get("HF_HUB_OFFLINE") == "0":
            pytest.skip("offline mode opted out for the real-model cases")
        from huggingface_hub import constants, is_offline_mode

        assert os.environ.get("HF_HUB_OFFLINE", "").upper() in TRUTHY
        assert os.environ.get("TRANSFORMERS_OFFLINE", "").upper() in TRUTHY
        assert constants.HF_HUB_OFFLINE is True
        assert is_offline_mode() is True

    def test_offline_mode_does_not_depend_on_the_caller(self):
        """RD-02 -- guards TEST-08: a bare `pytest` from a clean shell is offline too.

        Before the fix only `run_tests.py report` set the variables, so plain
        `pytest tests/` and every other runner command were online.
        """
        node = f"{Path(__file__).relative_to(BACKEND).as_posix()}::TestOffline::test_offline_mode_is_active_in_process"
        result = _pytest(["-q", node], _child_env(), timeout=300)

        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
        assert "1 passed" in result.stdout

    def test_the_conftest_sets_offline_mode_before_any_hub_capable_import(self):
        """RD-03 -- guards TEST-08: huggingface_hub reads the variables once, at import.

        So both `setdefault`s must run before conftest.py first imports the app
        or anything that loads the hub, and `tests/__init__.py` must import none
        of it.
        """
        hub_capable = ("app", "transformers", "huggingface_hub", "torch")

        def imports_hub(node: ast.AST) -> bool:
            if isinstance(node, ast.Import):
                return any(alias.name.split(".")[0] in hub_capable for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                return (node.module or "").split(".")[0] in hub_capable
            return False

        tree = ast.parse((TESTS_DIR / "conftest.py").read_text(encoding="utf-8"))
        first_import = min(node.lineno for node in ast.walk(tree) if imports_hub(node))
        defaults = {
            node.args[0].value: (node.lineno, node.args[1].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault" and ast.unparse(node.func.value) == "os.environ"
            and len(node.args) == 2 and all(isinstance(a, ast.Constant) for a in node.args)
        }
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            assert name in defaults, f"conftest.py does not default {name}"
            line, value = defaults[name]
            assert value == "1" and line < first_import, (name, line, first_import)

        init = ast.parse((TESTS_DIR / "__init__.py").read_text(encoding="utf-8"))
        assert not any(imports_hub(node) for node in ast.walk(init))


# --------------------------------------------------------------------------- B. corpora

# A module that reads a bundled corpus names it: through the guard, the registry,
# a literal dataset parameter or build_index call, or a path into Backend/data.
CORPUS_READER = re.compile(
    r'requires_corpora\(|DATASET_(?:PATHS|BASE_DIRS)|build_index\(\s*"'
    r'|parents\[\d\]\s*/\s*"data"|"dataset":\s*"(?:saa|common-voice|cv-valid-dev)"'
)

# The cases TEST-06 found reading a corpus without a guard, as (module, test).
TEST_06 = {
    ("test_fr10_orchestration.py", "test_prepare_analysis_partitions_saa_and_advances_progress"),
    ("test_fr10_orchestration.py", "test_prepare_analysis_insufficient_groups_fails_job_not_raises_uncaught"),
    ("test_fr10_api_contract.py", "test_groupable_columns_saa"),
    ("test_fr10_cancellation_and_failures.py", "test_aggregate_is_noop_when_cancel_requested_between_stages"),
    ("test_fr10_cancellation_and_failures.py",
     "test_aggregate_shrinks_group_on_partial_failure_and_excludes_below_threshold"),
    ("test_failover_corrupt_data.py", "test_corrupt_fr10_caches_are_recomputed"),
    ("test_failover_worker_recovery.py", "test_fr10_items_do_not_swallow_the_soft_limit"),
    ("test_dataset_serving_api.py", "test_every_bundled_dataset_loads_with_normalised_keys"),
    ("test_dataset_serving_api.py", "test_an_unlabelled_dataset_reports_zero_classes_without_failing"),
    ("test_whisper_transcript_consistency.py", "test_real_whisper_base_agrees_on_clean_and_noisy_speech"),
}


def _corpus_modules() -> list[str]:
    return sorted(
        p.name for p in TESTS_DIR.glob("test_*.py")
        if p.name != Path(__file__).name and CORPUS_READER.search(p.read_text(encoding="utf-8"))
    )


def _paths_in(value) -> list[Path]:
    if isinstance(value, Path):
        return [value]
    if isinstance(value, dict):
        return [v for v in value.values() if isinstance(v, Path)]
    return []


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class TestCorpora:
    @pytest.mark.slow
    def test_a_clean_checkout_skips_and_never_fails(self, tmp_path):
        """RD-04 -- guards TEST-06: with every corpus hidden, nothing fails.

        A child run over every module that reads a corpus, with
        `-p tests._corpora` repointing the app's corpus paths at a directory
        that does not exist -- what a fresh clone sees.
        """
        modules = _corpus_modules()
        assert len(modules) >= 9, modules
        junit = tmp_path / "junit.xml"
        result = _pytest(
            ["-q", "-p", "tests._corpora", "-m", "not performance", f"--junitxml={junit}",
             *(f"tests/{m}" for m in modules)],
            _child_env(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"), timeout=900,
        )
        run = _report.read_junit(junit, "hidden corpora", "backend")

        assert run.status == "ok", result.stdout[-2000:] + result.stderr[-2000:]
        broken = [f"{c.nodeid}: {c.message}" for c in run.cases if c.outcome in ("failed", "error")]
        assert broken == []
        skipped = {(c.file, c.nodeid.split("::")[-1].split("[")[0]) for c in run.cases if c.outcome == "skipped"}
        assert TEST_06 <= skipped, sorted(TEST_06 - skipped)
        reasons = {c.message for c in run.cases if c.outcome == "skipped"}
        assert all("Backend/data" in reason for reason in reasons), reasons
        assert result.returncode == 0

    def test_the_hiding_plugin_covers_every_corpus_path_in_app(self, monkeypatch, tmp_path):
        """RD-05 -- guards TEST-06: RD-04 means nothing if a corpus path escapes the plugin.

        Every app/ file that builds a path into Backend/data/ has a disposition
        in `_corpora.HIDDEN`, and hiding moves every Path-valued global of the
        repointed modules out of Backend/data/.
        """
        pattern = re.compile(r'parents\[\d+\]\s*/\s*"data"|Path\(\s*"data"\s*\)')
        found = {
            p.relative_to(BACKEND).as_posix() for p in APP.rglob("*.py")
            if pattern.search(p.read_text(encoding="utf-8"))
        }
        assert found == set(_corpora.HIDDEN)

        modules = {d.module: importlib.import_module(d.module) for d in _corpora.HIDDEN.values()}
        from app.services import dataset_service

        root = tmp_path / "hidden"
        monkeypatch.setattr(dataset_service, "_metadata_cache", {})
        _corpora.hide(root, setattr_=monkeypatch.setattr, setitem=monkeypatch.setitem)

        for disposition in _corpora.HIDDEN.values():
            if disposition.exempt:
                continue
            module = modules[disposition.module]
            for attr in disposition.repointed:
                paths = _paths_in(getattr(module, attr))
                assert paths and all(_under(p, root) for p in paths), (disposition.module, attr)
            leaked = [name for name, value in vars(module).items() for p in _paths_in(value) if _under(p, DATA)]
            assert leaked == [], (disposition.module, leaked)
        assert not root.exists()

    def test_corpora_are_never_committed(self):
        """RD-06 -- C-02: the corpus licences forbid redistribution (SAVEE outright)."""
        lines = (BACKEND / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert "/data" in (line.strip() for line in lines)
        if shutil.which("git") is None:
            pytest.skip("git is not installed")
        listed = subprocess.run(["git", "ls-files", "--", "data"], cwd=BACKEND, capture_output=True, text=True)
        if listed.returncode != 0:
            pytest.skip("not a git checkout")
        assert listed.stdout.strip() == ""


# --------------------------------------------------------------------------- C. timing

CLOCKS = {"perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns", "time", "time_ns"}
BARE_CLOCKS = CLOCKS - {"time", "time_ns"}
STAMPS = {"started", "ended"}
TIMING_HELPERS = {"loop_lag_probe", "sample", "run_profile", "poll_latencies", "by_endpoint"}
ORDERING = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)
COLLECTORS = {"append", "extend", "add", "insert"}

# Unmarked clock-derived assertions accepted with a recorded reason.  Empty.
ALLOWED_UNMARKED: set[str] = set()


@dataclass(frozen=True)
class ClockOracle:
    file: str
    test: str
    marked: bool
    line: int


def _is_loop(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return "loop" in node.id
    if isinstance(node, ast.Call):
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        return name in ("get_running_loop", "get_event_loop")
    return False


def _is_source(node: ast.AST, helpers: set[str]) -> bool:
    if isinstance(node, ast.Attribute) and node.attr in STAMPS and isinstance(node.ctx, ast.Load):
        return True
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        if func.attr in CLOCKS and isinstance(func.value, ast.Name) and func.value.id == "time":
            return True
        return func.attr == "time" and not node.args and _is_loop(func.value)
    return isinstance(func, ast.Name) and (func.id in helpers or func.id in BARE_CLOCKS)


def _tainted(node: ast.AST, names: set[str], helpers: set[str]) -> bool:
    return any(
        _is_source(sub, helpers) or (isinstance(sub, ast.Name) and sub.id in names)
        for sub in ast.walk(node)
    )


def _names(target: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def _taint(func: ast.AST, helpers: set[str]) -> set[str]:
    """Names in `func` (nested defs included) that hold a clock-derived value."""
    names: set[str] = set()
    while True:
        before = len(names)
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                pairs = [(t, node.value) for t in node.targets]
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
                pairs = [(node.target, node.value)]
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                pairs = [(node.target, node.iter)]
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                pairs = [(i.optional_vars, i.context_expr) for i in node.items if i.optional_vars is not None]
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr in COLLECTORS and isinstance(node.func.value, ast.Name)):
                pairs = [(node.func.value, arg) for arg in node.args]
            else:
                continue
            for target, value in pairs:
                if _tainted(value, names, helpers):
                    names |= _names(target)
        if len(names) == before:
            return names


def _clock_assertion(func: ast.AST, helpers: set[str]) -> int | None:
    """Line of the first assert comparing a clock-derived value, if any."""
    names = _taint(func, helpers)
    for node in ast.walk(func):
        if not isinstance(node, ast.Assert):
            continue
        for cmp in (n for n in ast.walk(node.test) if isinstance(n, ast.Compare)):
            operands = [cmp.left, *cmp.comparators]
            approx = any("approx" in ast.unparse(o) for o in operands)
            if (any(isinstance(op, ORDERING) for op in cmp.ops) or approx) and any(
                _tainted(o, names, helpers) for o in operands
            ):
                return node.lineno
    return None


def _marks_performance(nodes) -> bool:
    return any(
        isinstance(n, ast.Attribute) and n.attr == "performance" and ast.unparse(n.value).endswith("mark")
        for node in nodes for n in ast.walk(node)
    )


def _pytestmark(body) -> list[ast.AST]:
    return [
        node.value for node in body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "pytestmark" for t in node.targets)
    ]


def _clock_oracles() -> list[ClockOracle]:
    oracles = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        helpers = {
            alias.asname or alias.name
            for node in tree.body if isinstance(node, ast.ImportFrom) and node.module in ("tests._perf", "tests._load")
            for alias in node.names if alias.name in TIMING_HELPERS
        }
        module_marked = _marks_performance(_pytestmark(tree.body))
        scopes = [(tree.body, module_marked, "")]
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef) and cls.name.startswith("Test"):
                marked = module_marked or _marks_performance([*cls.decorator_list, *_pytestmark(cls.body)])
                scopes.append((cls.body, marked, f"{cls.name}::"))
        for body, scope_marked, prefix in scopes:
            for func in body:
                if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) and func.name.startswith("test_"):
                    line = _clock_assertion(func, helpers)
                    if line is not None:
                        marked = scope_marked or _marks_performance(func.decorator_list)
                        oracles.append(ClockOracle(path.name, prefix + func.name, marked, line))
    return oracles


class TestTiming:
    def test_every_clock_derived_assertion_is_a_performance_case(self):
        """RD-07 -- guards TEST-07: a wall-clock oracle under the tracer measures the tracer.

        Data flow within each test function, nested defs included: values from
        `time.*`, `loop.time()`, the `_perf`/`_load` timing helpers or an
        observation's `.started`/`.ended` are followed through assignment,
        loops, `with` and `list.append`; an assert that orders one (or compares
        it to `pytest.approx`) is a clock oracle, and must carry `performance`
        at module, class or function level.  Flow through other functions is
        not followed -- the positive controls show the analysis sees the
        suite's own idioms.
        """
        oracles = _clock_oracles()
        unmarked = sorted(f"{o.file}::{o.test} (line {o.line})" for o in oracles if not o.marked)
        assert [u for u in unmarked if u.split(" (")[0] not in ALLOWED_UNMARKED] == []

        marked = {(o.file, o.test.split("::")[-1]) for o in oracles if o.marked}
        # Positive controls: a direct `.ended - .started` flow and a helper flow.
        assert ("test_failover_under_load.py", "test_half_the_workers_die_holding_jobs") in marked
        assert any(file == "test_perf_event_loop.py" for file, _ in marked)


# --------------------------------------------------------------------------- D. redis

class TestRedis:
    def test_the_app_uses_nothing_fakeredis_cannot_model(self):
        """RD-08 -- A-02: fakeredis without lupa cannot run Lua, and the fake must be Redis 7.

        No script, function or lock call anywhere in app/ (torch's `.eval()`
        takes no arguments; Redis's takes at least two), and the fake server
        speaks the version SRS §3.10 deploys.
        """
        import fakeredis

        banned = {"register_script", "script_load", "evalsha", "fcall", "fcall_ro", "lock"}
        calls = []
        for path in APP.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    attr = node.func.attr
                    if attr in banned or (attr == "eval" and len(node.args) >= 2):
                        calls.append(f"{path.relative_to(BACKEND).as_posix()}:{node.lineno} {attr}")
        assert calls == []
        assert tuple(fakeredis.FakeServer().version) >= (7,)


# --------------------------------------------------------------------------- E. environment

HARNESS_PINS = ("pytest", "pytest-asyncio", "pytest-cov", "fakeredis", "httpx", "anyio")
RECORDED_DRIFT = {"transformers", "boto3"}  # OBS-43


def _requirements(path: Path, seen: list) -> list:
    from packaging.requirements import Requirement

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if line.startswith("-r"):
            _requirements(path.parent / line[2:].strip(), seen)
        else:
            seen.append(Requirement(line))
    return seen


def _major(spec: str) -> int:
    return int(re.search(r"\d+", spec).group(0))


class TestEnvironment:
    def test_the_harness_pins_are_exact_and_installed(self):
        """RD-09 -- R-03: the harness depends on behaviour later releases remove.

        httpx 0.28 drops `AsyncClient(app=)`, which every API case uses;
        coverage below 7.10 has no `patch = subprocess`; RD-12 relies on
        pytest's parser, pinned here.
        """
        from importlib.metadata import version

        from packaging.version import Version

        pins = {r.name.lower(): r for r in _requirements(BACKEND / "requirements-dev.txt", [])
                if r.name.lower() in HARNESS_PINS}
        for name in HARNESS_PINS:
            specs = list(pins[name].specifier)
            assert len(specs) == 1 and specs[0].operator == "==", (name, str(pins[name].specifier))
            assert version(name) == specs[0].version, name
        assert Version(version("httpx")) < Version("0.28")
        assert Version(version("coverage")) >= Version("7.10")

    def test_drift_is_no_worse_than_recorded(self):
        """RD-10 -- OBS-43: the venv breaks the requirements in exactly the recorded ways.

        transformers is 5.x against `==4.46.3` and boto3 is absent; both are
        reached only through stand-ins.  Any new violation fails here.
        """
        from importlib.metadata import PackageNotFoundError, version

        from packaging.utils import canonicalize_name

        requirements = _requirements(BACKEND / "requirements.txt", [])
        assert len(requirements) >= 25
        violations = set()
        for requirement in requirements:
            if requirement.marker and not requirement.marker.evaluate():
                continue
            try:
                installed = version(requirement.name)
            except PackageNotFoundError:
                violations.add(canonicalize_name(requirement.name))
                continue
            if not requirement.specifier.contains(installed, prereleases=True):
                violations.add(canonicalize_name(requirement.name))
        assert violations <= RECORDED_DRIFT, sorted(violations - RECORDED_DRIFT)

    def test_the_frontend_tooling_ceilings_hold(self):
        """RD-11 -- C-07: vitest 4 needs Vite 6, jsdom 30 needs Node 22.22."""
        package = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))
        lock = json.loads((FRONTEND / "package-lock.json").read_text(encoding="utf-8"))
        ranges = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
        locked = {name: lock["packages"][f"node_modules/{name}"]["version"] for name in ranges
                  if f"node_modules/{name}" in lock["packages"]}
        ceilings = {"vitest": 3, "@vitest/coverage-v8": 3, "jsdom": 26, "vite": 5, "@testing-library/jest-dom": 6}
        for name, major in ceilings.items():
            assert _major(ranges[name]) == major, (name, ranges[name])
            assert _major(locked[name]) == major, (name, locked[name])
        assert locked["@vitest/coverage-v8"] == locked["vitest"]

    def test_the_callers_shell_sets_no_settings_key(self):
        """RD-15 -- A-06: pydantic-settings reads the environment, case-insensitively.

        A `REDIS_URL` or `STORAGE_BACKEND` left in the developer's shell would
        silently reconfigure every case that reads `settings`.
        """
        from tests._config import SETTINGS_KEYS

        keys = {key.upper() for key in SETTINGS_KEYS}
        assert sorted(k for k in os.environ if k.upper() in keys) == []


# --------------------------------------------------------------------------- F. docs

RUNNER_COMMANDS = {"critical", "performance", "report", "summary", "all"}


def _readme_segments() -> list[list[str]]:
    """Every command in the README's bash fences, split on `&&`, env prefixes dropped."""
    segments = []
    for block in re.findall(r"```bash\n(.*?)```", README.read_text(encoding="utf-8"), re.S):
        for line in block.splitlines():
            tokens = shlex.split(line, comments=True)
            current: list[str] = []
            for token in [*tokens, "&&"]:
                if token != "&&":
                    current.append(token)
                    continue
                while current and re.match(r"^[A-Z_][A-Z0-9_]*=", current[0]):
                    current.pop(0)
                if current:
                    segments.append(current)
                current = []
    return segments


def _pytest_args(segment: list[str]) -> list[str] | None:
    if Path(segment[0]).name in ("pytest", "pytest.exe"):
        return segment[1:]
    if "python" in Path(segment[0]).name and segment[1:3] == ["-m", "pytest"]:
        return segment[3:]
    return None


class TestDocs:
    def test_the_readme_pytest_commands_parse(self, request):
        """RD-12 -- guards OBS-58: the README advised `--ignore-missing-models`, which does not exist.

        Each pytest command must parse with no unknown option, select only
        registered markers, and name test paths that exist.
        """
        parser = request.config._parser
        markers = {line.split(":")[0].split("(")[0].strip() for line in request.config.getini("markers")}
        commands = [args for s in _readme_segments() if (args := _pytest_args(s)) is not None]
        assert len(commands) >= 3, commands
        for args in commands:
            namespace, unknown = parser.parse_known_and_unknown_args(args)
            assert unknown == [], (args, unknown)
            if namespace.markexpr:
                words = set(re.findall(r"[A-Za-z_]\w*", namespace.markexpr)) - {"and", "or", "not"}
                assert words <= markers, (args, words - markers)
            for target in namespace.file_or_dir:
                assert (BACKEND / target.split("::")[0]).exists(), (args, target)

    def test_the_readme_runner_and_npm_commands_exist(self):
        """RD-13 -- guards OBS-58: every `run_tests.py` and `npm run` command is real."""
        scripts = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))["scripts"]
        allowed = RUNNER_COMMANDS | set(run_tests.TEST_CATEGORIES)
        runner, npm = 0, 0
        for segment in _readme_segments():
            for i, token in enumerate(segment):
                if token.endswith("run_tests.py"):
                    runner += 1
                    if i + 1 < len(segment):
                        assert segment[i + 1] in allowed, segment
            if segment[0] == "npm" and len(segment) > 1 and segment[1] in ("run", "test"):
                npm += 1
                script = segment[2] if segment[1] == "run" else "test"
                assert script in scripts, segment
        assert runner >= 3 and npm >= 1

    def test_the_register_and_its_guards_agree(self):
        """RD-14 -- the register's claims and this module cannot drift apart.

        The Section 5 matrix lists exactly the RD cases defined here, it cites
        no RD id that does not exist, and every risk and assumption names the
        case that checks it or says it is procedural.
        """
        assert SECTION_DOC.exists(), f"{SECTION_DOC.name} is missing"
        text = SECTION_DOC.read_text(encoding="utf-8")

        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        defined = {
            match.group(0) for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
            and (match := re.match(r"RD-\d{2}", ast.get_docstring(node) or ""))
        }
        in_matrix = set(re.findall(r"^\|\s*\**(RD-\d{2})\b", text, re.M))
        assert in_matrix == defined
        assert set(re.findall(r"\bRD-\d{2}\b", text)) <= defined

        # A risk's id has its own cell; an assumption's shares the first cell with its text.
        rows = re.findall(r"^\|\s*\**([RA]-\d{2})\b(.*)$", text, re.M)
        assert sum(r.startswith("R-") for r, _ in rows) >= 10 and sum(r.startswith("A-") for r, _ in rows) >= 5
        uncited = [
            row for row, rest in rows
            if not re.search(r"\b(?:RD|TD|CF|FO|FT|DI|PP|LT|UI)-\d+", rest) and "Procedural" not in rest
        ]
        assert uncited == []
