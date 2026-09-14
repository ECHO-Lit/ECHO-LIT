"""Configuration -- the deployment files agree with each other and with the SRS.

Test Plan Section 3.1.8.  See tests/plans/3.1.8-configuration-testing.md.

Module CB.  A static reading of `docker-compose.yml`, both Dockerfiles, the
env files, the README's run recipes, the operator scripts and the ignore
files.  It builds on FO-16 (3.1.7), which already pins append-only
persistence, `noeviction`, the Redis volume, the restart policies and the
shared mounts, and does not repeat those checks.

SRS 3.5: "configuration shall be environment-driven so the same images run
across environments."
SRS DC-2: "The control plane shall carry no machine learning dependency ...
GPU workers shall run at concurrency one per device."
SRS RE-1: "The failure of a single worker shall not affect the availability
of the API or the interface. Queued work shall continue on remaining workers."
"""
from __future__ import annotations

import json
import re
from fnmatch import fnmatch

import pytest

from app.core.celery_app import celery_app, queue_for
from app.core.model_catalog import MODEL_DEFINITIONS
from app.core.settings import Settings
from app.schemas.jobs import JobOperation
from tests import _config as cfg

pytestmark = [pytest.mark.configuration, pytest.mark.important]

COMPOSE = cfg.load_compose()
SERVICES = COMPOSE["services"]
PYTHON_SERVICES = cfg.python_services(COMPOSE)
WORKERS = cfg.worker_services(COMPOSE)
EXAMPLE = cfg.read_env_file(cfg.ENV_EXAMPLE)

# Every queue work is routed to: by operation and model, and by the beat schedule.
QUEUES = {queue_for(operation.value, model) for operation in JobOperation for model in [*MODEL_DEFINITIONS, None]}
QUEUES |= {entry["options"]["queue"] for entry in celery_app.conf.beat_schedule.values()}
GPU_QUEUES = {"gpu-fast", "gpu-large"}

# Settings no operator is meant to change: the API's URL layout and the
# result schema are part of the client contract.
INTERNAL_SETTINGS = {"API_V1_PREFIX", "RESULT_SCHEMA_VERSION"}


def _pinned_extra(service: str) -> set[str]:
    """What a service may pin beyond topology: its device, or the legacy guard."""
    if service == "api":
        return {"ENABLE_LEGACY_SYNC_INFERENCE"}
    return {"ML_DEVICE"}


def _size_bytes(text: str) -> int:
    match = re.fullmatch(r"(\d+)\s*([kmg]?)b?", text.strip().lower())
    assert match, text
    return int(match.group(1)) * {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[match.group(2)]


class TestComposeEnvironment:
    @pytest.mark.parametrize("service", PYTHON_SERVICES)
    def test_every_python_service_reads_backend_env_optionally(self, service):
        """CF-30: guards BUG-62 -- `Backend/.env` reaches every backend process.

        `required: false` keeps a fresh clone, with no `.env` yet, bootable.
        """
        assert cfg.env_file_entries(SERVICES[service]) == [{"path": cfg.BACKEND_ENV_FILE, "required": False}]

    @pytest.mark.parametrize("service", PYTHON_SERVICES)
    def test_compose_pins_only_container_topology(self, service):
        """CF-31: guards BUG-61 and BUG-62.

        `environment:` overrides `env_file`, so every tunable Compose pinned
        was a tunable `.env` could never change -- the seven-day TTLs among
        them.
        """
        keys = set(SERVICES[service].get("environment") or {})
        stray = keys - cfg.TOPOLOGY_KEYS - cfg.LIBRARY_VARS - _pinned_extra(service)
        assert stray == set(), service

    @pytest.mark.parametrize("service", PYTHON_SERVICES)
    def test_topology_values_match_the_compose_network_and_mounts(self, service):
        """CF-32: DC-2 -- four databases on the `redis` service; storage is a mount."""
        definition = SERVICES[service]
        env = cfg.service_env(definition, None)
        targets = [cfg.redis_target(env[role]) for role in cfg.REDIS_ROLES]
        assert targets == [("redis", 6379, index) for index in range(4)]
        mounts = cfg.mount_targets(definition)
        assert env["STORAGE_LOCAL_ROOT"] in mounts
        if any(source == "hf-cache" for source in mounts.values()):
            assert mounts[env["HF_HOME"]] == "hf-cache"

    def test_the_api_role_keeps_the_legacy_ml_routes_disabled(self):
        """CF-33: DC-2 -- the control plane image carries no ML runtime.

        The legacy synchronous routes import torch; the API image does not
        install it (CF-66), so the API pins the flag off whatever `.env` says.
        """
        api = SERVICES["api"]
        assert api["build"]["args"]["ROLE"] == "api"
        assert api["environment"]["ENABLE_LEGACY_SYNC_INFERENCE"] == "false"


class TestEnvInventory:
    def test_every_configured_key_is_consumed(self):
        """CF-34: guards BUG-64 -- no configuration key is dead.

        `ALLOWED_ORIGINS` was set by Compose and documented in `.env.example`
        but was not a setting: it was read with `os.getenv`, outside every
        check `Settings` makes.
        """
        configured = set(EXAMPLE.documented)
        for service in PYTHON_SERVICES:
            configured |= set(SERVICES[service].get("environment") or {})
        consumed = set(Settings.model_fields) | cfg.LIBRARY_VARS
        assert configured - consumed == set()

    def test_every_setting_is_documented_in_env_example(self):
        """CF-35: guards BUG-62 -- an operator can find every knob in one file."""
        undocumented = set(Settings.model_fields) - EXAMPLE.documented - INTERNAL_SETTINGS
        assert undocumented == set()

    def test_no_active_env_example_key_is_overridden_by_compose(self):
        """CF-36: guards BUG-62.

        An active line in `.env.example` that Compose also pins is a line whose
        edit does nothing.  Topology keys are documented, commented, with the
        values a native run uses.
        """
        for service in PYTHON_SERVICES:
            pinned = set(SERVICES[service].get("environment") or {})
            assert set(EXAMPLE.active) & pinned == set(), service
        assert set(EXAMPLE.active) & cfg.TOPOLOGY_KEYS == set()
        assert cfg.TOPOLOGY_KEYS <= set(EXAMPLE.commented)

    def test_env_files_and_compose_values_parse_unambiguously(self):
        """CF-37: env-file values mean the same to Compose, Vite and pydantic.

        Compose keeps an inline `# comment` and surrounding quotes as part of
        the value; a YAML bare `true` or `1` is not a string.
        """
        for path in (cfg.ENV_EXAMPLE, cfg.FRONTEND_ENV_EXAMPLE):
            for line in cfg.read_env_file(path).raw_active_lines:
                value = line.split("=", 1)[1]
                assert " #" not in value and not value.startswith(("'", '"')), (path.name, line)
        for service in PYTHON_SERVICES + ["frontend"]:
            for key, value in (SERVICES[service].get("environment") or {}).items():
                assert isinstance(value, str), (service, key, value)


class TestTopology:
    @pytest.mark.parametrize("topology", ["default", "gpu", "amd", "mps"])
    def test_every_routed_queue_is_consumed_in_every_documented_topology(self, topology):
        """CF-38: every queue has a consumer; accelerated runs keep GPU work off the CPU.

        The mps case guards BUG-77: the README started the full stack and a
        native MPS worker together, so the CPU `worker-model-local` kept
        taking GPU-queue work from the accelerator it was meant to leave it to.
        """
        run = cfg.readme_topologies(COMPOSE)[topology]
        consumers: dict[str, set[str]] = {}
        for name in run.services:
            for queue in cfg.queues_of(SERVICES[name].get("command") or []):
                consumers.setdefault(queue, set()).add(name)
        for index, command in enumerate(run.native_commands):
            for queue in cfg.queues_of(command):
                consumers.setdefault(queue, set()).add(f"native-{index}")

        assert QUEUES <= set(consumers), (topology, QUEUES - set(consumers))
        if topology != "default":
            accelerated = {"gpu": {"worker-gpu"}, "amd": {"worker-amd"}, "mps": {"native-0"}}[topology]
            for queue in GPU_QUEUES:
                assert consumers[queue] == accelerated, (topology, queue, consumers[queue])

    @pytest.mark.parametrize("worker", ["worker-model-local", "worker-gpu", "worker-amd", "readme-mps"])
    def test_gpu_queue_workers_run_one_task_per_device(self, worker):
        """CF-39: PE-3, DC-2 -- concurrency one per device, prefetch one."""
        if worker == "readme-mps":
            command = cfg.readme_topologies(COMPOSE)["mps"].native_commands[0]
        else:
            command = SERVICES[worker]["command"]
        flags = cfg.command_flags(command)
        assert flags["concurrency"] == "1" and flags["prefetch-multiplier"] == "1", worker
        if worker == "worker-gpu":
            devices = SERVICES[worker]["deploy"]["resources"]["reservations"]["devices"]
            assert devices == [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]
        if worker == "worker-amd":
            assert {"/dev/kfd:/dev/kfd", "/dev/dri:/dev/dri"} <= set(SERVICES[worker]["devices"])

    @pytest.mark.parametrize("worker", WORKERS)
    def test_worker_services_can_be_scaled(self, worker):
        """CF-40: guards BUG-69 -- PE-2 scaling and RE-1's "remaining workers".

        A fixed `container_name` makes `docker compose up --scale worker-cpu=2`
        fail outright: two containers cannot share one name.
        """
        assert "container_name" not in SERVICES[worker]
        assert "ports" not in SERVICES[worker]

    @pytest.mark.parametrize("worker", WORKERS)
    def test_every_worker_has_a_liveness_healthcheck(self, worker):
        """CF-41: guards BUG-69 -- a hung worker is visible, not just a dead one.

        `$$HOSTNAME` must reach the container as `$HOSTNAME`: a single `$`
        would be interpolated by Compose, on the host, to nothing.
        """
        check = SERVICES[worker].get("healthcheck") or {}
        test = check.get("test") or []
        assert test[:1] == ["CMD-SHELL"], worker
        assert "inspect ping -d celery@$$HOSTNAME" in test[1]
        assert {"interval", "timeout", "retries", "start_period"} <= set(check)

    def test_beat_runs_in_exactly_one_singleton_service(self):
        """CF-42: one scheduler -- two beats would publish every periodic task twice."""
        beats = [name for name, service in SERVICES.items() if "beat" in (service.get("command") or [])]
        assert beats == ["scheduler"]
        assert SERVICES["scheduler"].get("container_name")
        for worker in WORKERS:
            command = SERVICES[worker]["command"]
            assert "-B" not in command and "--beat" not in command
        assert "inspect ping" not in str(SERVICES["scheduler"].get("healthcheck", ""))

    def test_operator_scripts_do_not_address_fixed_worker_containers(self):
        """CF-43: guards BUG-69.

        `queue-status.sh` pinged `echo-worker-model-local`, which the GPU
        profiles scale to zero and which scaling would rename.
        """
        for script in sorted((cfg.REPO / "scripts").glob("*.sh")):
            assert "echo-worker-" not in script.read_text(encoding="utf-8"), script.name

    @pytest.mark.parametrize("worker", WORKERS)
    def test_prefork_workers_pin_native_thread_pools(self, worker):
        """CF-44: other software in the same process -- native thread pools.

        Celery's prefork pool forks children that inherit OpenMP/BLAS/numba
        pool state; the compose file records the SIGSEGVs that caused.
        """
        env = cfg.service_env(SERVICES[worker], None)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
            assert env[key] == "1", (worker, key)
        assert env["NUMBA_THREADING_LAYER"] == "workqueue"
        assert env["NUMBA_CPU_NAME"] == "generic"
        assert env["TOKENIZERS_PARALLELISM"] == "false"
        assert SERVICES[worker]["shm_size"]


class TestRedis:
    def test_redis_bounds_memory_without_evicting(self):
        """CF-45: guards BUG-70 -- RE-2 under memory pressure.

        `noeviction` without `maxmemory` never refuses a write: Redis grows
        until the host kills it, which loses exactly what RE-2 protects.
        With a bound it refuses writes, and the API reports that as a 503.
        """
        command = SERVICES["redis"]["command"]
        assert "--maxmemory" in command
        limit = cfg.interpolate(command[command.index("--maxmemory") + 1])
        assert _size_bytes(limit) >= 256 * cfg.MIB
        assert command[command.index("--maxmemory-policy") + 1] == "noeviction"

    def test_redis_meets_the_srs_version_floor(self):
        """CF-46: SRS 3.10 -- "Redis (version 7 or later)"."""
        tag = SERVICES["redis"]["image"].split(":", 1)[1]
        assert int(re.match(r"\d+", tag).group(0)) >= 7


class TestRetention:
    @pytest.mark.parametrize("configuration", ["code-defaults", "env-example"])
    def test_the_s3_lifecycle_matches_the_documented_job_ttl(self, monkeypatch, configuration):
        """CF-47: SRS 3.10 -- "a matching S3 lifecycle policy"."""
        env = {} if configuration == "code-defaults" else EXAMPLE.active
        loaded = cfg.load_settings(monkeypatch, env)
        days = json.loads(cfg.LIFECYCLE.read_text())["Rules"][0]["Expiration"]["Days"]
        assert days * cfg.DAY == loaded.JOB_TTL_SECONDS


class TestImages:
    def test_the_backend_build_context_is_versioned_and_excludes_local_state(self):
        """CF-48: guards BUG-75.

        `Backend/.dockerignore` was itself gitignored, so a fresh clone built
        with the whole backend as context: the virtualenv, datasets, uploads
        and the developer's `.env`.
        """
        patterns = [
            line.strip()
            for line in (cfg.BACKEND / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert not [p for p in patterns if fnmatch(".dockerignore", p.lstrip("/"))]
        ignored = {line.strip() for line in (cfg.BACKEND / ".dockerignore").read_text().splitlines()}
        assert {".env", ".venv/", "data/", "uploads/", "shared-storage/", "tests/"} <= ignored

    def test_image_roles_install_the_matching_requirements(self):
        """CF-49: DC-1 Python 3.11; each role installs its own requirement set."""
        dockerfile = (cfg.BACKEND / "Dockerfile").read_text(encoding="utf-8")
        bases = re.findall(r"^FROM\s+(\S+)", dockerfile, flags=re.M)
        assert "python:3.11-slim" in bases
        assert "python3.11" in dockerfile
        api_branch = re.search(r'if \[ "\$\{ROLE\}" = "api" \]; then(.*?)else', dockerfile, flags=re.S).group(1)
        assert "requirements-api.txt" in api_branch and "requirements-worker.txt" not in api_branch
        for name in ("api", "scheduler"):
            assert SERVICES[name]["build"]["args"]["ROLE"] == "api", name
        for worker in WORKERS:
            assert SERVICES[worker]["build"]["args"]["ROLE"] == "worker", worker


class TestDocumentation:
    def test_no_document_offers_an_evicting_redis(self):
        """CF-50: guards BUG-77 -- SRS 3.10's non-evicting policy, as documented.

        ARCHITECTURE.md allowed "an evicting cache" for REDIS_URL, which holds
        the sessions; PROJECT.md described `allkeys-lru` at 256 MB.
        """
        for name in ("README.md", "ARCHITECTURE.md", "PROJECT.md"):
            text = " ".join((cfg.REPO / name).read_text(encoding="utf-8").split())
            assert "allkeys-lru" not in text, name
            assert not re.search(r"(?<!non-)\bevicting cache", text), name

    def test_documented_python_version_matches_the_images(self):
        """CF-51: guards BUG-77 -- DC-1's Python 3.11 in the sample CI too."""
        text = (cfg.BACKEND / "tests" / "README.md").read_text(encoding="utf-8")
        versions = re.findall(r"python-version:\s*['\"]?(\d+\.\d+)", text)
        assert versions and set(versions) == {"3.11"}

    def test_the_compose_frontend_leaves_the_api_base_to_the_client(self):
        """CF-52: guards BUG-65.

        Pinning `http://localhost:8000` made the client call a different site
        from `http://127.0.0.1:8080`, where the Lax session cookie is never
        sent.  Unset, the client derives the API from the page's own host, on
        the port the API publishes.
        """
        assert "VITE_API_BASE_URL" not in (SERVICES["frontend"].get("environment") or {})
        assert "8000:8000" in SERVICES["api"]["ports"]
