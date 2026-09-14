"""Configuration instruments for the Section 3.1.8 configuration modules.

Test Plan Section 3.1.8.  See tests/plans/3.1.8-configuration-testing.md.

Configuration testing asks one question of many configurations: does the
system behave as specified under each of them?  The configurations ECHO ships
are written down in files -- `docker-compose.yml`, `Backend/.env.example`, the
README's run recipes -- so this module turns each of those into the
environment a process would actually receive, and loads `Settings` from it.

Compose itself is never executed.  Its semantics are modelled where they
matter here, and only those:

* YAML merge keys (`<<: *anchor`) -- PyYAML's safe loader resolves them;
* precedence -- a service's `env_file` is read first and its `environment:`
  overrides it, key by key;
* `env_file` entries in the long form `{path, required}` (Compose >= 2.24),
  where `required: false` tolerates an absent file;
* interpolation -- `${VAR}`, `${VAR:-default}`, and `$$` as a literal `$`.

Nothing here reads a developer's own `Backend/.env` or `Frontend/.env`.  A
case that needs a `.env` supplies a synthetic one, or uses the committed
`.env.example`.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from app.core.settings import Settings


BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
FRONTEND = REPO / "Frontend"
COMPOSE = REPO / "docker-compose.yml"
ENV_EXAMPLE = BACKEND / ".env.example"
FRONTEND_ENV_EXAMPLE = FRONTEND / ".env.example"
README = REPO / "README.md"
LIFECYCLE = BACKEND / "s3-lifecycle.json"

DAY = 24 * 60 * 60
MIB = 1024 * 1024

# Settings the 3.1.8 fixes add.  Listed so the environment can be cleared of
# them on both sides of the fix, and so a pre-fix run fails on an assertion
# rather than on a KeyError.
ADDED_SETTINGS = {
    "ALLOWED_ORIGINS",
    "MODEL_REGISTRY_MAX_ENTRIES",
    "MODEL_REGISTRY_IDLE_SECONDS",
    "MAX_SALIENCY_SECONDS",
    "MAX_SALIENCY_SECONDS_SHAP",
    "SALIENCY_SHAP_SAMPLES",
}
SETTINGS_KEYS = set(Settings.model_fields) | ADDED_SETTINGS

# Where each Redis role lives: SAD section 9 and SRS DC-2 give each its own
# logical database, in this order.
REDIS_ROLES = ("REDIS_URL", "JOB_REDIS_URL", "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND")

# Keys a container needs because of WHERE it runs, not how it is tuned: its
# network, its mounts.  Compose pins these; `Backend/.env` must not move them.
TOPOLOGY_KEYS = {*REDIS_ROLES, "STORAGE_BACKEND", "STORAGE_LOCAL_ROOT"}

# Read by libraries and the interpreter, never by `Settings`.
LIBRARY_VARS = {
    "HF_HOME",
    "PYTHONUNBUFFERED",
    "PYTHONFAULTHANDLER",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMBA_THREADING_LAYER",
    "NUMBA_CPU_NAME",
    "NUMBA_CPU_FEATURES",
    "TOKENIZERS_PARALLELISM",
    # Set per pool child by app/worker/tasks.py (BUG-78), never by an operator.
    "NUMBA_CACHE_DIR",
}

BACKEND_ENV_FILE = "./Backend/.env"


# -- compose ---------------------------------------------------------------

def load_compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def python_services(compose: dict | None = None) -> list[str]:
    """Every service built from the backend image: the API, workers, beat."""
    compose = compose or load_compose()
    return [
        name
        for name, service in compose["services"].items()
        if (service.get("build") or {}).get("context") == "./Backend"
    ]


def worker_services(compose: dict | None = None) -> list[str]:
    compose = compose or load_compose()
    return [name for name in compose["services"] if name.startswith("worker-")]


_INTERPOLATION = re.compile(r"\$\$|\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?-([^}]*))?\}")


def interpolate(value: str, env: dict[str, str] | None = None) -> str:
    """Compose's variable substitution, as far as this repo uses it."""
    env = env or {}

    def replace(match: re.Match) -> str:
        if match.group(0) == "$$":
            return "$"
        name, default = match.group(1), match.group(2)
        return env.get(name) or (default if default is not None else "")

    return _INTERPOLATION.sub(replace, value)


def env_file_entries(service: dict) -> list[dict]:
    """A service's `env_file`, normalised to the long form."""
    raw = service.get("env_file") or []
    if isinstance(raw, (str, dict)):
        raw = [raw]
    return [entry if isinstance(entry, dict) else {"path": entry, "required": True} for entry in raw]


def service_env(service: dict, dotenv: dict[str, str] | None) -> dict[str, str]:
    """The environment Compose gives a container of `service`.

    `dotenv` is the content of `Backend/.env`, or None when the file does not
    exist.  `env_file` is applied first; `environment:` overrides it.
    """
    env: dict[str, str] = {}
    for entry in env_file_entries(service):
        if entry["path"] != BACKEND_ENV_FILE:
            continue
        if dotenv is None:
            if entry.get("required", True):
                raise FileNotFoundError(f"{BACKEND_ENV_FILE} is required by this service")
            continue
        env.update(dotenv)
    environment = service.get("environment") or {}
    if isinstance(environment, list):
        environment = dict(item.split("=", 1) for item in environment)
    env.update({key: "" if value is None else str(value) for key, value in environment.items()})
    return {key: interpolate(value) for key, value in env.items()}


def mount_targets(service: dict) -> dict[str, str]:
    """`{target: source}` for a service's volumes."""
    targets = {}
    for volume in service.get("volumes") or []:
        parts = volume.split(":")
        targets[parts[1]] = parts[0]
    return targets


def command_flags(command: list[str] | str) -> dict[str, str]:
    """`--name=value` flags of a command line."""
    tokens = shlex.split(command) if isinstance(command, str) else list(command)
    flags = {}
    for token in tokens:
        if token.startswith("--") and "=" in token:
            name, value = token[2:].split("=", 1)
            flags[name] = value
    return flags


def queues_of(command: list[str] | str) -> set[str]:
    value = command_flags(command).get("queues", "")
    return {queue for queue in value.split(",") if queue}


# -- env files ---------------------------------------------------------------

_ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")


@dataclass
class EnvFile:
    active: dict[str, str] = field(default_factory=dict)
    commented: dict[str, str] = field(default_factory=dict)
    production: dict[str, str] = field(default_factory=dict)
    raw_active_lines: list[str] = field(default_factory=list)

    @property
    def documented(self) -> set[str]:
        return set(self.active) | set(self.commented)


def read_env_file(path: Path) -> EnvFile:
    """Parse KEY=VALUE lines, `# KEY=VALUE` lines, and the production block.

    The production block is the documented production configuration, written
    as commented assignments between `# >>> production` and `# <<< production`.
    """
    parsed = EnvFile()
    in_production = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("# >>> production"):
            in_production = True
            continue
        if stripped.startswith("# <<< production"):
            in_production = False
            continue
        if stripped.startswith("#"):
            body = stripped.lstrip("#").strip()
            match = _ASSIGNMENT.match(body)
            if match:
                value = re.split(r"\s+#", match.group(2), maxsplit=1)[0].strip()
                parsed.commented[match.group(1)] = value
                if in_production:
                    parsed.production[match.group(1)] = value
            continue
        match = _ASSIGNMENT.match(stripped)
        if match:
            parsed.active[match.group(1)] = match.group(2)
            parsed.raw_active_lines.append(stripped)
    return parsed


# -- settings ----------------------------------------------------------------

def clear_settings_env(monkeypatch) -> None:
    for key in SETTINGS_KEYS:
        monkeypatch.delenv(key, raising=False)


def load_settings(monkeypatch, env: dict[str, str] | None = None) -> Settings:
    """A fresh `Settings` built from exactly `env` -- the module singleton is untouched."""
    clear_settings_env(monkeypatch)
    for key, value in (env or {}).items():
        if key.upper() in SETTINGS_KEYS:
            monkeypatch.setenv(key, value)
    return Settings()


def redis_target(url: str) -> tuple[str | None, int, int]:
    parts = urlsplit(url)
    database = parts.path.lstrip("/") or "0"
    return parts.hostname, parts.port or 6379, int(database)


def assert_srs_invariants(settings: Settings, label: str, *, role_indexes: bool = True) -> None:
    """What every deployed configuration owes the SRS, whatever else it varies."""
    # PE-3, SRS 2.4 and 3.10: a 24-hour expiry on sessions and job artifacts.
    assert settings.SESSION_TTL_SECONDS == DAY, (label, "SESSION_TTL_SECONDS", settings.SESSION_TTL_SECONDS)
    assert settings.JOB_TTL_SECONDS == DAY, (label, "JOB_TTL_SECONDS", settings.JOB_TTL_SECONDS)
    # PE-3: audio up to 100 MB and 10 minutes per file.
    assert settings.MAX_UPLOAD_BYTES == 100 * MIB, (label, settings.MAX_UPLOAD_BYTES)
    assert settings.MAX_AUDIO_DURATION_SECONDS == 600, (label, settings.MAX_AUDIO_DURATION_SECONDS)
    # DC-2: separate logical databases for sessions/cache, jobs, broker, results.
    targets = [redis_target(getattr(settings, role)) for role in REDIS_ROLES]
    assert len(set(targets)) == 4, (label, targets)
    if role_indexes:
        assert [target[2] for target in targets] == [0, 1, 2, 3], (label, targets)
    # The recovery timings 3.1.7 ordered.
    assert (
        settings.TASK_SOFT_TIME_LIMIT_SECONDS
        < settings.TASK_TIME_LIMIT_SECONDS
        < settings.BROKER_VISIBILITY_TIMEOUT_SECONDS
        < settings.STALE_JOB_SECONDS
    ), label


# -- README run recipes ------------------------------------------------------

@dataclass
class Topology:
    """The processes one documented way of running ECHO starts."""

    name: str
    services: set[str]
    native_commands: list[list[str]] = field(default_factory=list)
    native_env: dict[str, str] = field(default_factory=dict)


def _code_blocks(text: str) -> list[list[str]]:
    """Each fenced block as logical lines, with `\\` continuations joined."""
    blocks = []
    for body in re.findall(r"```[a-z]*\n(.*?)```", text, flags=re.S):
        lines, pending = [], ""
        for line in body.splitlines():
            if line.rstrip().endswith("\\"):
                pending += line.rstrip()[:-1] + " "
                continue
            lines.append((pending + line).strip())
            pending = ""
        blocks.append([line for line in lines if line and not line.startswith("#")])
    return blocks


def _compose_services(line: str, compose: dict) -> set[str]:
    """The services `docker compose ... up ...` starts, dependencies included."""
    tokens = shlex.split(line)[2:]
    profiles, scales, named = set(), {}, []
    iterator = iter(tokens)
    for token in iterator:
        if token == "--profile":
            profiles.add(next(iterator))
        elif token == "--scale":
            name, count = next(iterator).split("=")
            scales[name] = int(count)
        elif token.startswith("-") or token == "up":
            continue
        else:
            named.append(token)
    services = compose["services"]
    if named:
        selected = set(named)
        frontier = list(named)
        while frontier:
            depends = services[frontier.pop()].get("depends_on") or {}
            for dependency in depends:
                if dependency not in selected:
                    selected.add(dependency)
                    frontier.append(dependency)
    else:
        selected = {
            name for name, service in services.items()
            if not service.get("profiles") or set(service["profiles"]) & profiles
        }
    return {name for name in selected if scales.get(name, 1) > 0}


def readme_topologies(compose: dict | None = None) -> dict[str, Topology]:
    compose = compose or load_compose()
    blocks = _code_blocks(README.read_text(encoding="utf-8"))
    topologies: dict[str, Topology] = {}

    def compose_lines(block):
        return [line for line in block if line.startswith("docker compose") and " up" in f" {line} "]

    quickstart = next(block for block in blocks if "docker compose up --build" in block)
    topologies["default"] = Topology(
        "default", _compose_services(next(line for line in quickstart if line.startswith("docker compose up")), compose)
    )
    for profile in ("gpu", "amd"):
        line = next(
            line for block in blocks for line in compose_lines(block) if f"--profile {profile}" in line
        )
        topologies[profile] = Topology(profile, _compose_services(line, compose))

    mps_block = next(block for block in blocks if any("ML_DEVICE=mps" in line for line in block))
    mps_compose = compose_lines(mps_block)
    services = (
        _compose_services(mps_compose[0], compose) if mps_compose else topologies["default"].services
    )
    native_line = next(line for line in mps_block if "ML_DEVICE=mps" in line)
    tokens = shlex.split(native_line)
    native_env = {}
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        key, value = tokens.pop(0).split("=", 1)
        native_env[key] = value
    topologies["mps"] = Topology("mps", services, [tokens], native_env)
    return topologies
