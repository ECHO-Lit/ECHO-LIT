"""Configuration -- Settings under every configuration ECHO ships.

Test Plan Section 3.1.8.  See tests/plans/3.1.8-configuration-testing.md.

Module CA.  Each documented configuration -- the code defaults, every compose
service with and without `Backend/.env`, the native `.env.example`, the
README's macOS worker, and the documented production configuration -- is
turned into the environment its process receives, and `Settings` is loaded
from it.  The SRS invariants must hold under all of them, and a configuration
that would break the system must be refused when the process starts, not
discovered by a user.

SRS SE-2: "The API shall ... restrict CORS to a configured allowed list of
origins. ... Session cookies shall be HttpOnly, with Secure and SameSite
enabled in production, and HTTPS shall be enforced."
SRS 3.5: "configuration shall be environment-driven so the same images run
across environments."
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from httpx import AsyncClient
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from app.core.settings import Settings, settings
from app.main import app
from tests import _config as cfg
from tests._faults import tolerant_client

pytestmark = [pytest.mark.configuration, pytest.mark.critical]


@pytest.fixture(autouse=True)
def _clean_settings_env(monkeypatch):
    """No case sees the environment the test process happened to start with."""
    cfg.clear_settings_env(monkeypatch)


def _native_example_env() -> dict[str, str]:
    """`.env.example` as a native run reads it: the active lines only."""
    return cfg.read_env_file(cfg.ENV_EXAMPLE).active


def _production_valid_env(**overrides: str) -> dict[str, str]:
    env = {
        "ENVIRONMENT": "production",
        "COOKIE_SECURE": "true",
        "COOKIE_SAMESITE": "none",
        "ALLOWED_ORIGINS": "https://echo.example.org",
    }
    env.update(overrides)
    return env


def _origins(loaded: Settings) -> list[str]:
    # Read through getattr so that, before the fix, the case fails on its
    # assertion rather than on an AttributeError.
    return list(getattr(loaded, "allowed_origins", None) or [])


class TestDeploymentConfigurations:
    @pytest.mark.parametrize("name", ["code-defaults", "native-env-example", "readme-mps-worker"])
    def test_a_non_compose_configuration_meets_the_srs_invariants(self, monkeypatch, name):
        """CF-01: PE-3, DC-2 -- the configurations that run outside Compose."""
        compose = cfg.load_compose()
        if name == "code-defaults":
            env = {}
        elif name == "native-env-example":
            env = _native_example_env()
        else:
            env = cfg.readme_topologies(compose)["mps"].native_env

        loaded = cfg.load_settings(monkeypatch, env)
        cfg.assert_srs_invariants(loaded, name)

        if name == "readme-mps-worker":
            # The native macOS worker must meet the Compose API on the same
            # data: the same Redis databases (Compose publishes 6379 on the
            # host) and the same object store (the API's bind-mount source,
            # since the README runs the worker from Backend/).
            api = compose["services"]["api"]
            api_env = cfg.service_env(api, None)
            for role in cfg.REDIS_ROLES:
                assert cfg.redis_target(getattr(loaded, role))[1:] == cfg.redis_target(api_env[role])[1:], role
            assert "6379:6379" in compose["services"]["redis"]["ports"]
            source = cfg.mount_targets(api)[api_env["STORAGE_LOCAL_ROOT"]]
            assert (cfg.BACKEND / loaded.STORAGE_LOCAL_ROOT).resolve() == (cfg.REPO / source).resolve()

    @pytest.mark.parametrize("dotenv", ["no-env-file", "env-example"])
    @pytest.mark.parametrize("service", cfg.python_services())
    def test_every_compose_service_configuration_meets_the_srs_invariants(self, monkeypatch, service, dotenv):
        """CF-02: guards BUG-61.

        The compose file set both TTLs to 604 800 s -- seven days -- so the
        deployed stack kept sessions, jobs and uploaded voice audio a week
        against the SRS's 24 hours, whatever the code defaulted to.
        """
        compose = cfg.load_compose()
        definition = compose["services"][service]
        env = cfg.service_env(definition, None if dotenv == "no-env-file" else _native_example_env())
        loaded = cfg.load_settings(monkeypatch, env)

        cfg.assert_srs_invariants(loaded, f"{service}/{dotenv}")
        assert loaded.STORAGE_LOCAL_ROOT in cfg.mount_targets(definition), service
        if service == "api":
            assert "http://localhost:8080" in _origins(loaded)

    @pytest.mark.parametrize("service", cfg.python_services())
    def test_compose_topology_overrides_a_native_style_env_file(self, monkeypatch, service):
        """CF-03: guards BUG-62.

        A developer's `Backend/.env` is written for native runs: Redis on
        localhost, the object store relative to Backend/.  Once Compose reads
        that file, the container's own network and mounts must still win.
        """
        definition = cfg.load_compose()["services"][service]
        assert {"path": cfg.BACKEND_ENV_FILE, "required": False} in cfg.env_file_entries(definition)

        native = {role: f"redis://localhost:6379/{index}" for index, role in enumerate(cfg.REDIS_ROLES)}
        native["STORAGE_LOCAL_ROOT"] = "shared-storage"
        loaded = cfg.load_settings(monkeypatch, cfg.service_env(definition, native))

        for index, role in enumerate(cfg.REDIS_ROLES):
            assert getattr(loaded, role) == f"redis://redis:6379/{index}", role
        assert loaded.STORAGE_LOCAL_ROOT == "/app/shared-storage"

    @pytest.mark.parametrize("service", cfg.python_services())
    def test_tunables_in_backend_env_reach_every_python_service(self, monkeypatch, service):
        """CF-04: guards BUG-62.

        The README told users to copy `.env.example` to `Backend/.env` and
        "edit if needed", but no process read it: every edit was silently
        ignored.
        """
        definition = cfg.load_compose()["services"][service]
        tuned = {
            "SESSION_TTL_SECONDS": "3600",
            "COOKIE_SAMESITE": "strict",
            "ALLOWED_ORIGINS": "http://example.test:8080",
        }
        loaded = cfg.load_settings(monkeypatch, cfg.service_env(definition, tuned))

        assert loaded.SESSION_TTL_SECONDS == 3600
        assert loaded.COOKIE_SAMESITE == "strict"
        if service == "api":
            assert _origins(loaded) == ["http://example.test:8080"]

    def test_the_documented_production_configuration_is_secure_and_complete(self, monkeypatch):
        """CF-05: guards BUG-77 and BUG-63.

        `.env.example`'s production advice swapped only REDIS_URL, leaving the
        job store, broker and result backend pointing at a host called
        `redis`, and gave no complete production configuration to load.
        """
        import json

        from celery import Celery

        production = cfg.read_env_file(cfg.ENV_EXAMPLE).production
        assert production, "no `# >>> production` block in Backend/.env.example"
        loaded = cfg.load_settings(monkeypatch, production)

        assert getattr(loaded, "is_production", False)
        assert loaded.COOKIE_SECURE is True
        assert loaded.COOKIE_SAMESITE == "none"
        assert _origins(loaded) and all(origin.startswith("https://") for origin in _origins(loaded))
        targets = [cfg.redis_target(getattr(loaded, role)) for role in cfg.REDIS_ROLES]
        assert len(set(targets)) == 4, targets
        assert all(getattr(loaded, role).startswith("rediss://") for role in cfg.REDIS_ROLES)
        assert loaded.STORAGE_BACKEND == "s3" and loaded.S3_BUCKET
        assert loaded.ENABLE_LEGACY_SYNC_INFERENCE is False
        lifecycle = json.loads(cfg.LIFECYCLE.read_text())["Rules"][0]["Expiration"]["Days"]
        assert loaded.JOB_TTL_SECONDS == lifecycle * cfg.DAY
        # Celery's Redis result backend refuses a rediss:// URL that does not
        # say how to verify the certificate; building it is the check.
        Celery("echo-config-check", broker=loaded.CELERY_BROKER_URL, backend=loaded.CELERY_RESULT_BACKEND).backend


class TestCookieSecurity:
    @pytest.mark.parametrize("environment", ["production", " Prod ", "PRODUCTION"])
    def test_production_refuses_insecure_cookies(self, monkeypatch, environment):
        """CF-06: guards BUG-63 -- SE-2's Secure cookie in production."""
        env = _production_valid_env(ENVIRONMENT=environment, COOKIE_SECURE="false", COOKIE_SAMESITE="lax")
        with pytest.raises(ValidationError, match="COOKIE_SECURE"):
            cfg.load_settings(monkeypatch, env)

    def test_samesite_none_requires_secure(self, monkeypatch):
        """CF-07: guards BUG-63.

        Browsers reject a `SameSite=None` cookie that is not also `Secure`, so
        the session would silently restart on every request.
        """
        with pytest.raises(ValidationError, match="SameSite"):
            cfg.load_settings(monkeypatch, {"COOKIE_SAMESITE": "none", "COOKIE_SECURE": "false"})

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("Strict", "strict"), (" LAX ", "lax"), ("strictly", None), ("", "lax")],
    )
    def test_samesite_is_normalised_and_unknown_values_are_refused(self, monkeypatch, value, expected):
        """CF-08: guards BUG-63.

        Starlette asserts the value on every response it sets a cookie on, so
        an unknown one surfaced as a 500 on every new session (CF-11) rather
        than at startup.  An empty value means "unset".
        """
        if expected is None:
            with pytest.raises(ValidationError):
                cfg.load_settings(monkeypatch, {"COOKIE_SAMESITE": value})
        else:
            assert cfg.load_settings(monkeypatch, {"COOKIE_SAMESITE": value}).COOKIE_SAMESITE == expected

    def test_production_origins_must_use_https(self, monkeypatch):
        """CF-09: guards BUG-63 -- SE-2 "HTTPS shall be enforced"."""
        env = _production_valid_env(ALLOWED_ORIGINS="http://echo.example.org")
        with pytest.raises(ValidationError, match="https"):
            cfg.load_settings(monkeypatch, env)


class TestCookieAttributes:
    @pytest.mark.parametrize("profile", ["development", "production"])
    async def test_the_session_cookie_carries_the_configured_attributes(self, monkeypatch, client, profile):
        """CF-10: SE-2 -- the cookie a browser receives, attribute by attribute.

        `test_session_cookie.py` checks only that a `sid` cookie exists.
        """
        if profile == "production":
            monkeypatch.setattr(settings, "COOKIE_SECURE", True)
            monkeypatch.setattr(settings, "COOKIE_SAMESITE", "none")
            monkeypatch.setattr(settings, "COOKIE_DOMAIN", "echo.example.org")

        response = await client.get("/session")
        header = response.headers["set-cookie"]
        attributes = {part.strip().split("=")[0].lower(): part.strip() for part in header.split(";")}

        assert "httponly" in attributes
        assert attributes["path"] == "Path=/"
        assert attributes["max-age"] == f"Max-Age={settings.SESSION_TTL_SECONDS}"
        if profile == "production":
            assert "secure" in attributes
            assert attributes["samesite"].lower() == "samesite=none"
            assert attributes["domain"] == "Domain=echo.example.org"
        else:
            assert "secure" not in attributes
            assert attributes["samesite"].lower() == "samesite=lax"
            assert "domain" not in attributes

    async def test_an_unvalidated_samesite_value_fails_every_new_session(self, monkeypatch):
        """CF-11: the failure CF-08 prevents at startup.

        Set past validation, an unknown SameSite value reaches Starlette's
        per-response assertion and every new visitor gets a 500.
        """
        monkeypatch.setattr(settings, "COOKIE_SAMESITE", "strictly")
        async with tolerant_client() as client:
            response = await client.get("/session")
        assert response.status_code == 500


class TestCorsAllowList:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (" http://a.test:8080 , http://b.test:8080 ", ["http://a.test:8080", "http://b.test:8080"]),
            ("http://a.test:8080/", ["http://a.test:8080"]),
            ("http://a.test:8080,http://a.test:8080/", ["http://a.test:8080"]),
        ],
    )
    def test_origins_are_parsed_and_normalised_by_settings(self, monkeypatch, raw, expected):
        """CF-12: guards BUG-64.

        A browser's Origin header never ends in a slash, so a configured
        `http://host:8080/` never matched and the site was silently blocked.
        """
        assert _origins(cfg.load_settings(monkeypatch, {"ALLOWED_ORIGINS": raw})) == expected

    @pytest.mark.parametrize("raw", ["*", "http://a.test:1,*", ",", "localhost:8080"])
    def test_a_wildcard_or_empty_allow_list_is_refused_at_startup(self, monkeypatch, raw):
        """CF-13: guards BUG-64.

        With credentials on, Starlette answers a `*` list by echoing any
        origin (CF-16) -- every site on the web could read a user's session.
        A list of only commas became an empty list that blocked every browser.
        """
        with pytest.raises(ValidationError, match="ALLOWED_ORIGINS"):
            cfg.load_settings(monkeypatch, {"ALLOWED_ORIGINS": raw})

    def test_the_app_is_built_from_the_validated_allow_list(self):
        """CF-14: guards BUG-64 -- one source of truth for the allow-list."""
        cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
        assert cors.kwargs["allow_origins"] == _origins(settings)
        assert cors.kwargs["allow_credentials"] is True

    @pytest.mark.parametrize(("origin", "admitted"), [("http://localhost:8080", True), ("http://evil.example", False)])
    async def test_a_listed_origin_is_admitted_and_an_unlisted_one_is_not(self, client, origin, admitted):
        """CF-15: SE-2 -- the allow-list as a browser's preflight experiences it."""
        response = await client.options(
            "/session",
            headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
        )
        if admitted:
            assert response.headers.get("access-control-allow-origin") == origin
        else:
            assert "access-control-allow-origin" not in response.headers

    async def test_a_wildcard_with_credentials_reflects_any_origin(self):
        """CF-16: why CF-13 refuses `*` -- measured on the installed Starlette."""
        bare = Starlette(
            routes=[Route("/", lambda request: PlainTextResponse("ok"))],
            middleware=[Middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True)],
        )
        async with AsyncClient(app=bare, base_url="http://test") as client:
            response = await client.get("/", headers={"Origin": "http://evil.example", "Cookie": "sid=abc"})
        assert response.headers["access-control-allow-origin"] == "http://evil.example"
        assert response.headers["access-control-allow-credentials"] == "true"


class TestValueRanges:
    @pytest.mark.parametrize(
        "env",
        [
            {"SESSION_TTL_SECONDS": "0"},
            {"JOB_TTL_SECONDS": "-1"},
            {"MAX_UPLOAD_BYTES": "0"},
            {"MAX_AUDIO_DURATION_SECONDS": "0"},
            {"TASK_SOFT_TIME_LIMIT_SECONDS": "-10", "TASK_TIME_LIMIT_SECONDS": "0"},
            {"STALE_JOB_SWEEP_SECONDS": "0"},
            {"FR10_MIN_GROUP_SIZE": "0"},
            {"FR10_MIN_SPEAKERS_PER_GROUP": "0"},
        ],
        ids=lambda env: ",".join(f"{k}={v}" for k, v in env.items()),
    )
    def test_non_positive_values_are_refused_at_startup(self, monkeypatch, env):
        """CF-17: guards BUG-71.

        A TTL of 0 made every Redis write fail with "invalid expire time",
        which the API reported as a misleading "temporarily unavailable" 503
        on every job; a zero limit refused every upload.  Negative time
        limits passed the ordering check.
        """
        with pytest.raises(ValidationError):
            cfg.load_settings(monkeypatch, env)


class TestNormalisation:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [("S3_ENDPOINT_URL", None), ("S3_ACCESS_KEY_ID", None), ("SESSION_TTL_SECONDS", cfg.DAY)],
    )
    def test_empty_environment_values_mean_unset(self, monkeypatch, key, expected):
        """CF-18: guards BUG-71.

        `KEY=` in an env file is how an operator leaves a key unset; it was
        read as the empty string (an S3 endpoint of "") or refused outright.
        """
        assert getattr(cfg.load_settings(monkeypatch, {key: ""}), key) == expected

    @pytest.mark.parametrize(("value", "expected"), [(" Local ", "local"), ("S3", "s3"), ("gcs", None)])
    def test_storage_backend_is_normalised_at_startup(self, monkeypatch, value, expected):
        """CF-19: guards BUG-71.

        `get_storage` stripped the value but the retention sweep did not, so
        " local" stored objects and then never cleaned them up.
        """
        if expected is None:
            with pytest.raises(ValidationError):
                cfg.load_settings(monkeypatch, {"STORAGE_BACKEND": value})
        else:
            assert cfg.load_settings(monkeypatch, {"STORAGE_BACKEND": value}).STORAGE_BACKEND == expected


class TestHermeticity:
    def test_settings_never_read_a_dotenv_file(self, monkeypatch, tmp_path):
        """CF-20: Settings come from the process environment only.

        Compose injects `Backend/.env`; the code must not read it itself, or
        every pytest run would take its configuration from whichever `.env`
        the developer happens to have.
        """
        assert Settings.model_config.get("env_file") is None
        (tmp_path / ".env").write_text("SESSION_TTL_SECONDS=5\n")
        monkeypatch.chdir(tmp_path)
        assert cfg.load_settings(monkeypatch).SESSION_TTL_SECONDS == cfg.DAY


class TestSingleSource:
    def test_no_application_module_reads_the_environment_directly(self):
        """CF-21: guards BUG-64 and BUG-72 -- SRS 3.5, one validated source.

        `os.getenv` reads bypass every check `Settings` makes, and the keys
        they read appear in no configuration file.  The one exception is a
        third-party library's own variable, named literally (BUG-78 sets
        numba's cache directory per pool child); anything else, including
        passing `os.environ` around wholesale, is an offender.
        """

        def is_os(node, attr):
            return (
                isinstance(node, ast.Attribute) and node.attr == attr
                and isinstance(node.value, ast.Name) and node.value.id == "os"
            )

        def library_key(node) -> bool:
            return isinstance(node, ast.Constant) and node.value in cfg.LIBRARY_VARS

        offenders = []
        for path in sorted((cfg.BACKEND / "app").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            where = str(path.relative_to(cfg.BACKEND))
            vetted: set[int] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and is_os(node.func, "getenv"):
                    vetted.add(id(node.func))
                    if not (node.args and library_key(node.args[0])):
                        offenders.append(f"{where}:{node.lineno}")
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and is_os(node.func.value, "environ"):
                    vetted.add(id(node.func.value))
                    if not (node.args and library_key(node.args[0])):
                        offenders.append(f"{where}:{node.lineno}")
                elif isinstance(node, ast.Subscript) and is_os(node.value, "environ"):
                    vetted.add(id(node.value))
                    if not library_key(node.slice):
                        offenders.append(f"{where}:{node.lineno}")
            for node in ast.walk(tree):
                if (is_os(node, "environ") or is_os(node, "getenv")) and id(node) not in vetted:
                    offenders.append(f"{where}:{node.lineno}")
        assert offenders == []
