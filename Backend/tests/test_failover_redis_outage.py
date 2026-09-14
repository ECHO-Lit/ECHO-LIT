"""Failover and Recovery -- Redis loss, refusal and restart at the API tier.

Test Plan Section 3.1.7.  See tests/plans/3.1.7-failover-and-recovery.md.

Every request the API serves touches Redis first: `SessionMiddleware` refreshes
the session before any route runs.  So "communication interruption" to Redis is
the failure every user meets at once, and the questions are the ones the RUP
template asks of a recovery: does the system fail cleanly while the dependency
is gone, and does it resume -- with no data lost and no manual step -- when the
dependency returns?

SRS RE-2: "Job metadata and results shall be held in a durable, non-evicting
Redis configuration with add-only persistence, so that ongoing and completed
work is not silently lost under memory pressure."
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from redis.exceptions import ConnectionError as RedisConnectionError

from app.api.routes import jobs as jobs_routes
from app.core import redis as redis_module
from app.core.settings import settings
from app.core.storage import get_storage
from app.services import custom_dataset_service
from app.worker import executor, tasks
from tests._faults import ORIGIN, FaultHarness, FaultyRedis, oom_refusal, tolerant_client
from tests._fixtures import wav_bytes

pytestmark = [pytest.mark.failover, pytest.mark.critical]

COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.yml"


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    monkeypatch.setattr(custom_dataset_service, "SESSIONS_BASE_DIR", tmp_path / "sessions")
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


@pytest.fixture(autouse=True)
def harness(monkeypatch):
    faults = FaultHarness()
    faults.install(monkeypatch)
    yield faults
    faults.reconnect()


@pytest.fixture(autouse=True)
def broker(monkeypatch):
    calls: list[dict] = []

    def _send_task(name, args=None, kwargs=None, queue=None, **rest):
        calls.append({"name": name, "args": args, "queue": queue})
        return SimpleNamespace(id=f"celery-{len(calls)}")

    monkeypatch.setattr(jobs_routes.celery_app, "send_task", _send_task)
    monkeypatch.setattr(jobs_routes.celery_app.control, "revoke", lambda *a, **k: None)
    return SimpleNamespace(calls=calls)


@pytest.fixture
def model(monkeypatch):
    calls: list[str] = []

    def fake_model(operation, model, audio_path, parameters, model_spec=None):
        calls.append(audio_path.name)
        return {"text": "the quick brown fox"}

    monkeypatch.setattr(executor, "_execute_one", fake_model)
    return calls


async def _upload(client) -> str:
    response = await client.post("/upload", files={"file": ("clip.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 201, response.text
    return response.json()["audio_id"]


async def _submit(client, audio_id) -> str:
    response = await client.post(
        "/jobs", json={"operation": "prediction", "model": "whisper-base", "audio_ids": [audio_id]}
    )
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def _assert_clean_unavailable(response) -> None:
    """What a browser needs to recover from a 503: a readable, CORS-visible body."""
    assert response.status_code == 503, f"{response.status_code}: {response.text[:200]}"
    assert response.headers.get("access-control-allow-origin") == ORIGIN
    assert response.headers.get("retry-after")
    assert "temporarily unavailable" in response.json()["detail"]


def _objects(prefix: str) -> list[Path]:
    root = Path(settings.STORAGE_LOCAL_ROOT) / prefix
    return [path for path in root.rglob("*") if path.is_file()] if root.exists() else []


# --------------------------------------------------------------------------
# The instrument
# --------------------------------------------------------------------------


class TestHarness:
    async def test_an_outage_refuses_every_client_kind_and_all_recover(self, harness):
        """FO-01: the outage switch fails every access path the product uses.

        Plain commands, pipelines, a WATCH transaction and the worker's
        synchronous client all fail with ConnectionError while the server is
        down -- and the *same* client objects work again afterwards, as a
        redis-py pool reconnects after a real outage.
        """
        session_db, job_db, sync_db = redis_module.redis, redis_module.job_redis, tasks._heartbeat_redis
        await session_db.set("k", "v")

        with harness.outage():
            with pytest.raises(RedisConnectionError):
                await session_db.get("k")
            with pytest.raises(RedisConnectionError):
                pipe = job_db.pipeline()
                pipe.set("a", 1)
                await pipe.execute()
            with pytest.raises(RedisConnectionError):
                pipe = job_db.pipeline(transaction=True)
                await pipe.watch("a")
            with pytest.raises(RedisConnectionError):
                sync_db.get("k")

        assert await session_db.get("k") == "v"
        pipe = job_db.pipeline()
        pipe.set("a", 1)
        assert await pipe.execute() == [True]
        assert sync_db.set("b", 2)


# --------------------------------------------------------------------------
# Failing cleanly while Redis is gone
# --------------------------------------------------------------------------


class TestOutage:
    @pytest.mark.parametrize(("method", "path", "body"), [
        ("GET", "/session", None),
        ("GET", "/jobs/0123", None),
        ("POST", "/jobs", {"operation": "prediction", "model": "whisper-base", "audio_ids": ["a"]}),
        ("GET", "/upload/list", None),
        ("GET", "/queue", None),
    ])
    async def test_a_request_during_an_outage_gets_a_clean_503(self, harness, method, path, body):
        """FO-02: guards BUG-50.

        A browser can only act on a response it can read.  An unhandled
        exception becomes Starlette's plain-text 500 from outside the CORS
        middleware, which a cross-origin browser reports as "Failed to fetch".
        """
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            await client.get("/session")
            with harness.outage():
                response = await client.request(method, path, json=body)

        _assert_clean_unavailable(response)

    async def test_a_cors_preflight_is_answered_during_an_outage_and_creates_no_session(self, harness):
        """FO-03: guards BUG-50.

        A preflight carries no cookie and needs no session.  Before the fix it
        passed through the session middleware -- failing during an outage, and
        writing a fresh session key for every preflight when Redis was up.
        """
        headers = {
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        }
        async with tolerant_client() as client:
            with harness.outage():
                during = await client.options("/jobs", headers=headers)
            after = await client.options("/jobs", headers=headers)

        assert during.status_code == 200, during.text[:200]
        assert during.headers.get("access-control-allow-origin") == ORIGIN
        assert after.status_code == 200
        assert "set-cookie" not in after.headers
        assert harness.raw(0).keys("sess:*") == []

    @pytest.mark.parametrize(("method", "path", "body"), [
        ("GET", "/jobs/0123", None),
        ("POST", "/jobs", {"operation": "prediction", "model": "whisper-base", "audio_ids": ["a"]}),
        ("DELETE", "/jobs/0123", None),
    ])
    async def test_a_route_level_redis_failure_is_a_clean_503(self, harness, monkeypatch, method, path, body):
        """FO-04: guards BUG-50 -- the job database alone is unreachable.

        The session middleware succeeds, so the failure surfaces inside a
        route: it too must reach the browser as a readable 503.
        """
        down = FaultyRedis(redis_module.job_redis, lambda name: RedisConnectionError("job database down"))
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            await client.get("/session")
            monkeypatch.setattr(redis_module, "job_redis", down)
            response = await client.request(method, path, json=body)

        _assert_clean_unavailable(response)

    async def test_health_and_metrics_degrade_to_a_structured_503(self, harness):
        """FO-05: guards BUG-50 for `/metrics`.

        The monitoring endpoints are what an operator reads during an outage.
        `/metrics` already caught its own dependency errors (3.1.2 BUG-08) but
        the session middleware failed first, so that handler never ran.
        """
        async with tolerant_client() as client:
            with harness.outage():
                health = await client.get("/health")
                metrics = await client.get("/metrics")

        assert health.status_code == 503
        assert health.json()["status"] == "degraded"
        assert health.json()["session_redis"] is False
        assert metrics.status_code == 503, metrics.text[:200]
        assert metrics.json()["status"] == "degraded"

    async def test_no_job_is_published_when_its_record_cannot_be_written(self, harness, monkeypatch, broker):
        """FO-06: guards BUG-50 -- a job the API cannot track is never queued.

        `jobs.create` runs before the publish, so a refused record write must
        stop the request before a worker could run an untracked job.
        """
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            audio_id = await _upload(client)
            writes_refused = FaultyRedis(
                redis_module.job_redis,
                lambda name: RedisConnectionError("refused") if name in {"set", "zadd", "expire"} else None,
            )
            monkeypatch.setattr(redis_module, "job_redis", writes_refused)
            response = await client.post(
                "/jobs", json={"operation": "prediction", "model": "whisper-base", "audio_ids": [audio_id]}
            )

        _assert_clean_unavailable(response)
        assert broker.calls == []
        assert harness.raw(1).keys("job:*") == []

    async def test_an_upload_interrupted_after_the_copy_leaves_no_object(self, harness, monkeypatch):
        """FO-07: the upload's compensating delete, under a refused record write."""
        refuse_records = FaultyRedis(
            redis_module.job_redis,
            lambda name: RedisConnectionError("refused") if name in {"set", "sadd", "expire"} else None,
        )
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            await client.get("/session")
            monkeypatch.setattr(redis_module, "job_redis", refuse_records)
            response = await client.post("/upload", files={"file": ("clip.wav", wav_bytes(), "audio/wav")})

        _assert_clean_unavailable(response)
        assert _objects("uploads") == []

    async def test_a_materialise_interrupted_after_the_copy_leaves_no_object(self, harness, monkeypatch, tmp_path):
        """FO-08: guards BUG-60.

        `/audio/materialize` copies a dataset file into the object store and
        then writes its record.  Upload deletes the copy when the record write
        fails; materialise did not, leaving an object nothing references.
        """
        from app.services import dataset_service

        source = tmp_path / "speaker01.wav"
        source.write_bytes(wav_bytes())
        monkeypatch.setattr(dataset_service, "resolve_file", lambda dataset, filename, sid: source)
        refuse_records = FaultyRedis(
            redis_module.job_redis,
            lambda name: RedisConnectionError("refused") if name in {"set", "sadd", "expire"} else None,
        )
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            await client.get("/session")
            monkeypatch.setattr(redis_module, "job_redis", refuse_records)
            response = await client.post("/audio/materialize", json={"dataset": "ravdess", "filename": "speaker01.wav"})

        _assert_clean_unavailable(response)
        assert _objects("datasets") == []

    async def test_an_oom_refusal_is_a_503_and_leaves_existing_keys_unchanged(self, harness, monkeypatch):
        """FO-11: guards BUG-50 -- RE-2's `noeviction` refusal, surfaced cleanly.

        Under `maxmemory-policy noeviction` a full Redis refuses writes rather
        than evicting job state.  The refusal must reach the user as a 503, and
        nothing already stored may be lost to make room.
        """
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            audio_id = await _upload(client)
            job_id = await _submit(client, audio_id)
            before = {db: {key: harness.raw(db).dump(key) for key in harness.raw(db).keys()} for db in (0, 1)}
            monkeypatch.setattr(redis_module, "redis", FaultyRedis(redis_module.redis, oom_refusal))
            response = await client.get(f"/jobs/{job_id}")

        _assert_clean_unavailable(response)
        after = {db: {key: harness.raw(db).dump(key) for key in harness.raw(db).keys()} for db in (0, 1)}
        assert after == before

    async def test_a_wrong_type_session_key_gets_a_new_session(self, harness):
        """FO-12: guards BUG-51 -- a corrupted session key must not brick the cookie.

        `ensure_session` issues HSETNX on `sess:{sid}:meta`.  A value of the
        wrong type made that fail with WRONGTYPE on every request, and the same
        pipeline refreshed the key's TTL each time, so the session could never
        expire its way out.
        """
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            old_sid = (await client.get("/session")).json()["sid"]
            harness.raw(0).set(f"sess:{old_sid}:meta", "not a hash")

            response = await client.get("/session")
            again = await client.get("/session")

        assert response.status_code == 200, response.text[:200]
        new_sid = response.json()["sid"]
        assert new_sid != old_sid
        assert f"sid={new_sid}" in response.headers.get("set-cookie", "")
        assert again.json()["sid"] == new_sid


# --------------------------------------------------------------------------
# Resuming when Redis returns
# --------------------------------------------------------------------------


class TestRecovery:
    async def test_the_same_session_continues_after_an_outage(self, harness):
        """FO-09: RE-2 -- recovery needs no restart and loses nothing.

        After the outage the same cookie reaches the same session, audio and
        job, through the same client objects the API held throughout.
        """
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            sid = (await client.get("/session")).json()["sid"]
            audio_id = await _upload(client)
            job_id = await _submit(client, audio_id)
            with harness.outage():
                for path in ("/session", f"/jobs/{job_id}", "/upload/list"):
                    await client.get(path)

            assert (await client.get("/session")).json()["sid"] == sid
            listed = (await client.get("/upload/list")).json()["files"]
            job = await client.get(f"/jobs/{job_id}")

        assert [item["audio_id"] for item in listed] == [audio_id]
        assert job.status_code == 200
        assert job.json()["status"] == "queued"

    async def test_an_in_flight_job_completes_after_the_outage(self, harness, broker, monkeypatch):
        """FO-10: RE-2 / SAD UC-1 "a worker failure triggers a retry".

        Redis goes away in the middle of a real `execute_job`, after the model
        has run and before the result is recorded.  The task's retry picks the
        job up once Redis is back, and the user fetches a complete result.
        """
        from celery.signals import task_retry

        model_calls = 0

        def fail_after_the_model(*args, **kwargs):
            nonlocal model_calls
            model_calls += 1
            if model_calls == 1:
                harness.disconnect()
            return {"text": "the quick brown fox"}

        monkeypatch.setattr(executor, "_execute_one", fail_after_the_model)

        def redis_is_back(sender=None, **kwargs):
            harness.reconnect()

        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            audio_id = await _upload(client)
            job_id = await _submit(client, audio_id)
            envelope = broker.calls[-1]["args"][0]

            task_retry.connect(redis_is_back, weak=False)
            try:
                outcome = await asyncio.to_thread(tasks.execute_job.apply, args=[envelope], task_id="fo-10")
            finally:
                task_retry.disconnect(redis_is_back)

            status = (await client.get(f"/jobs/{job_id}")).json()
            result = await client.get(f"/jobs/{job_id}/result")

        assert outcome.successful(), outcome.result
        assert model_calls == 2
        assert status["status"] == "success"
        assert status["error"] is None
        assert result.status_code == 200
        assert result.json()["items"][0]["result"]["text"] == "the quick brown fox"


class TestRestart:
    async def test_state_survives_an_append_only_replay_restart(self, harness, broker, model):
        """FO-13: RE-2 -- a Redis restart with append-only persistence loses nothing.

        Sessions, audio records, job records and their TTLs come back, and a
        job that was queued before the restart runs to completion after it.
        """
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            sid = (await client.get("/session")).json()["sid"]
            audio_id = await _upload(client)
            job_id = await _submit(client, audio_id)
            envelope = broker.calls[-1]["args"][0]
            ttl_before = harness.raw(1).ttl(f"job:{job_id}")

            harness.restart(persist=True)

            ttl_after = harness.raw(1).ttl(f"job:{job_id}")
            assert (await client.get("/session")).json()["sid"] == sid
            assert [f["audio_id"] for f in (await client.get("/upload/list")).json()["files"]] == [audio_id]
            outcome = await asyncio.to_thread(tasks.execute_job.apply, args=[envelope], task_id="fo-13")
            status = (await client.get(f"/jobs/{job_id}")).json()
            result = await client.get(f"/jobs/{job_id}/result")

        assert 0 < ttl_after <= ttl_before
        assert ttl_before - ttl_after <= 2
        assert outcome.successful(), outcome.result
        assert status["status"] == "success"
        assert result.status_code == 200

    async def test_a_restart_without_persistence_reports_lost_jobs_as_not_found(self, harness):
        """FO-14: FR-14 "Polling an unknown or expired job returns not found".

        If Redis comes back empty, a job the user was polling is gone.  The API
        must say so -- 404, which the client understands -- rather than fail.
        """
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            audio_id = await _upload(client)
            job_id = await _submit(client, audio_id)

            harness.restart(persist=False)

            job = await client.get(f"/jobs/{job_id}")
            listed = await client.get("/upload/list")

        assert job.status_code == 404
        assert listed.status_code == 200
        assert listed.json()["files"] == []

    async def test_the_dataset_sweep_keeps_every_dataset_after_a_replayed_restart(self, harness, tmp_path):
        """FO-15: RE-2 -- the hourly dataset sweep reads session keys after a restart.

        The sweep deletes the datasets of sessions whose key is gone, so a
        restart that lost session keys would take every dataset with it.  With
        append-only replay, every live session's datasets survive.
        """
        async with tolerant_client() as client:
            sid = (await client.get("/session")).json()["sid"]
        dataset_dir = custom_dataset_service.SESSIONS_BASE_DIR / sid / "datasets" / "mine"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "metadata.json").write_text("{}")

        harness.restart(persist=True)
        removed = tasks.cleanup_expired_session_datasets_task()

        assert removed == 0
        assert dataset_dir.exists()

    def test_the_deployment_configures_durable_non_evicting_redis(self):
        """FO-16: RE-2 as deployed -- the static half of the durability claim.

        The in-process cases above prove the application recovers when Redis
        keeps its data; this pins the configuration that makes Redis keep it:
        append-only persistence, no eviction, a named volume, restart policies,
        and the object store shared by the API and every worker.
        """
        compose = yaml.safe_load(COMPOSE.read_text())
        services = compose["services"]
        redis_service = services["redis"]
        command = redis_service["command"]

        assert command[command.index("--appendonly") + 1] == "yes"
        assert command[command.index("--maxmemory-policy") + 1] == "noeviction"
        assert any(volume.startswith("redis-data:") and volume.endswith(":/data") for volume in redis_service["volumes"])
        assert "redis-data" in compose["volumes"]

        runtime = [name for name in services if name in {"redis", "api", "scheduler"} or name.startswith("worker")]
        for name in runtime:
            assert services[name]["restart"] == "unless-stopped", name
        for name in [n for n in runtime if n != "redis"]:
            mounts = services[name]["volumes"]
            assert any(m.endswith(":/app/shared-storage") for m in mounts), name
            assert any(m.endswith(":/app/uploads") for m in mounts), name
