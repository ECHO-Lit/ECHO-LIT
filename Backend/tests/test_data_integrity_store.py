"""Data and Database Integrity Testing -- Redis and repository layer.

Test Plan Section 3.1.1.  See tests/plans/3.1.1-data-and-database-integrity-testing.md.

ECHO has no SQL database: the persistence tier is Redis (three logical DBs) plus
an object store.  This module covers the Redis half -- key families, TTLs,
serialisation, validate-on-read and the one piece of optimistic-concurrency code
in the codebase (`JobRepository.update`).

It differs from `test_data_integrity.py` in one load-bearing way: every case here
goes through the *real* access methods (`ensure_session`, `put_queue`, the four
repositories) and the *real* key helpers, rather than inventing key names.  A
test that writes `"session_1"` and reads it back proves fakeredis works; it
proves nothing about ECHO.

Nothing here touches the filesystem, so no storage isolation fixture is needed --
see `test_data_integrity_artifacts.py` for the cases that do.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.core import redis as redis_module
from app.core.redis import (
    cache_result,
    ensure_session,
    get_queue,
    get_result,
    k_meta,
    k_queue,
    k_result,
    k_sess,
    put_queue,
)
from app.core.settings import settings
from app.repositories.audio import AudioRepository
from app.repositories.jacobian_lenses import JacobianLensRepository
from app.repositories.jobs import JobRepository
from app.repositories.models import CustomModelRepository
from app.schemas.jacobian_lens import JacobianLensRecord
from app.schemas.jobs import AudioAsset, JobError, JobOperation, JobProgress, JobRecord, JobStatus
from app.schemas.models import CustomModelRecord
from app.services.queue_service import add_item, set_progress

# A fixed epoch keeps ordering assertions deterministic.  Calling datetime.now()
# three times in a row can collide at float resolution and make DI-06 arbitrary.
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# TTL assertions are bounds, not observed expiry: fakeredis re-stamps its clock
# on every command dispatch, so there is nothing to fast-forward.  The lower
# bound is loose enough to survive a slow test run, tight enough to catch a key
# written with no expiry at all (ttl == -1) or a missing key (ttl == -2).
DAY_TTL_FLOOR = 86_000
DAY_TTL_CEIL = 86_400


def audio_asset(**overrides) -> AudioAsset:
    fields = {
        "audio_id": "a1",
        "session_id": "s1",
        "object_key": "uploads/s1/a1.wav",
        "filename": "clip.wav",
        "media_type": "audio/wav",
        "size_bytes": 1024,
        "duration_seconds": 1.5,
        "sample_rate": 16000,
        "channels": 1,
        "sha256": "a" * 64,
        "created_at": T0,
    }
    fields.update(overrides)
    return AudioAsset(**fields)


def job_record(**overrides) -> JobRecord:
    fields = {
        "job_id": "j1",
        "session_id": "s1",
        "operation": JobOperation.prediction,
        "model": "whisper-base",
        "audio_ids": ["a1"],
        "created_at": T0,
        "updated_at": T0,
    }
    fields.update(overrides)
    return JobRecord(**fields)


def custom_model_record(**overrides) -> CustomModelRecord:
    fields = {
        "model_id": "m1",
        "session_id": "s1",
        "hf_repo": "openai/whisper-tiny",
        "created_at": T0,
        "updated_at": T0,
    }
    fields.update(overrides)
    return CustomModelRecord(**fields)


def lens_record(**overrides) -> JacobianLensRecord:
    fields = {
        "lens_id": "l1",
        "session_id": "s1",
        "model_id": "whisper-base",
        "model_revision": "rev-1",
        "fit_job_id": "j1",
        "created_at": T0,
        "updated_at": T0,
        "sample_count": 2,
    }
    fields.update(overrides)
    return JacobianLensRecord(**fields)


# --------------------------------------------------------------------------
# A. Real key helpers and repositories, seeded with valid and invalid data
# --------------------------------------------------------------------------


@pytest.mark.critical
class TestSessionKeyFamily:
    """`ensure_session` is the only writer of the `sess:*` family."""

    async def test_ensure_session_populates_only_meta(self):
        """DI-01: a fresh session writes `sess:{sid}:meta` and nothing else."""
        sid = await ensure_session(None)

        assert len(sid) == 32
        assert await redis_module.redis.hget(k_meta(sid), "created") == "1"
        # `k_sess` is a namespace prefix, never a key in its own right, and the
        # queue key does not exist until something is actually queued.
        assert await redis_module.redis.exists(k_sess(sid)) == 0
        assert await redis_module.redis.exists(k_queue(sid)) == 0
        assert sorted(await redis_module.redis.keys("sess:*")) == [k_meta(sid)]

    async def test_ensure_session_is_idempotent_and_refreshes_meta_ttl(self):
        """DI-02: re-entry does not clobber `created` but does slide the TTL."""
        sid = await ensure_session(None)
        await redis_module.redis.hset(k_meta(sid), "created", "tampered")
        await redis_module.redis.expire(k_meta(sid), 10)

        await ensure_session(sid)

        # HSETNX must not overwrite an existing field.
        assert await redis_module.redis.hget(k_meta(sid), "created") == "tampered"
        # An active session must not expire out from under its owner.
        assert await redis_module.redis.ttl(k_meta(sid)) > DAY_TTL_FLOOR


@pytest.mark.critical
class TestQueueRoundTrip:
    async def test_put_queue_get_queue_preserves_shape_and_order(self):
        """DI-03: JSON round-trip preserves order and scalar types."""
        sid = await ensure_session(None)

        # An absent key reads as the canonical empty state, not None.
        assert await get_queue(sid) == {"items": [], "processing": None, "completed": []}

        await put_queue(
            sid,
            {
                "items": [{"id": "a", "n": 1}, {"id": "b", "n": 2}],
                "processing": "a",
                "completed": [],
            },
        )
        state = await get_queue(sid)

        assert [item["id"] for item in state["items"]] == ["a", "b"]
        assert isinstance(state["items"][0]["n"], int)
        assert state["processing"] == "a"

    async def test_queue_service_mutations_are_read_modify_write(self):
        """DI-04: `add_item` then `set_progress` leave one coherent document."""
        sid = await ensure_session(None)

        await add_item(sid, {"id": "a"})
        await set_progress(sid, {"processing": "a"})

        # set_progress merges into the document rather than replacing it.
        assert await get_queue(sid) == {
            "items": [{"id": "a"}],
            "processing": "a",
            "completed": [],
        }


@pytest.mark.critical
class TestRepositoryRoundTrip:
    async def test_audio_repository_create_populates_record_and_index(self):
        """DI-05: `create` writes both the record and its session index entry."""
        asset = audio_asset()
        repo = AudioRepository()

        await repo.create(asset)

        # Inspect the store out-of-band: the record must be valid AudioAsset JSON.
        raw = await redis_module.job_redis.get("audio:a1")
        assert AudioAsset.model_validate_json(raw) == asset
        assert await redis_module.job_redis.smembers("session:s1:audio") == {"a1"}

        # ...and all three read paths must agree with it.
        assert (await repo.get("a1")).sha256 == asset.sha256
        assert await repo.get_owned("a1", "s1") is not None
        assert [a.audio_id for a in await repo.list_owned("s1")] == ["a1"]

    async def test_index_orderings_differ_by_repository_as_designed(self):
        """DI-06: jobs list oldest-first; models and lenses list newest-first."""
        stamps = [T0, T0 + timedelta(seconds=10), T0 + timedelta(seconds=20)]

        for index, created in enumerate(stamps):
            await JobRepository().create(
                job_record(job_id=f"j{index}", created_at=created, updated_at=created)
            )
            await CustomModelRepository().create(
                custom_model_record(model_id=f"m{index}", created_at=created, updated_at=created)
            )
            await JacobianLensRepository().create(
                lens_record(lens_id=f"l{index}", created_at=created, updated_at=created)
            )

        # ZRANGE -- ascending.
        assert await JobRepository().list_session_job_ids("s1") == ["j0", "j1", "j2"]
        # ZREVRANGE -- descending.  The asymmetry is deliberate, not a bug.
        assert [r.model_id for r in await CustomModelRepository().list_owned("s1")] == [
            "m2",
            "m1",
            "m0",
        ]
        assert [r.lens_id for r in await JacobianLensRepository().list_owned("s1")] == [
            "l2",
            "l1",
            "l0",
        ]


@pytest.mark.critical
class TestInvalidSeeds:
    @pytest.mark.parametrize(
        ("key", "read"),
        [
            ("audio:x", lambda: AudioRepository().get("x")),
            ("job:x", lambda: JobRepository().get("x")),
            ("custom-model:x", lambda: CustomModelRepository().get("x")),
            ("jacobian-lens:x", lambda: JacobianLensRepository().get("x")),
        ],
        ids=["audio", "job", "custom_model", "lens"],
    )
    async def test_repository_get_rejects_wrong_shape_json(self, key, read):
        """DI-07: structurally wrong JSON raises rather than half-populating."""
        # Syntactically valid, semantically nonsense: every required field but one
        # is missing.  Validate-on-read must refuse it outright.
        await redis_module.job_redis.set(key, json.dumps({"audio_id": "x"}))

        with pytest.raises(ValidationError):
            await read()


@pytest.mark.critical
class TestLogicalDatabaseSeparation:
    """Session state (DB0) and control-plane records (DB1) are separate stores.

    The `fake_redis` fixture in conftest mirrors the production split across
    three databases on one server, so this asserts the property in the same
    environment every other test runs in.
    """

    async def test_job_records_are_not_visible_on_the_session_database(self):
        """DI-08: session state and job state live on separate databases."""
        await JobRepository().create(job_record())
        await put_queue("s1", {"items": [], "processing": None, "completed": []})

        # Each write landed on its own database...
        assert await redis_module.job_redis.exists("job:j1") == 1
        assert await redis_module.redis.exists(k_queue("s1")) == 1
        # ...and is invisible from the other.
        assert await redis_module.redis.exists("job:j1") == 0
        assert await redis_module.job_redis.exists(k_queue("s1")) == 0


# --------------------------------------------------------------------------
# B. TTL coverage and expiry mid-job
# --------------------------------------------------------------------------


@pytest.mark.critical
class TestTtlCoverage:
    async def test_every_repository_key_family_carries_a_bounded_ttl(self):
        """DI-09: no repository leaks an immortal key."""
        await AudioRepository().create(audio_asset())
        await JobRepository().create(job_record())
        await JobRepository().request_cancel("j1")
        await CustomModelRepository().create(custom_model_record())
        await JacobianLensRepository().create(lens_record())

        keys = [
            "audio:a1",
            "session:s1:audio",
            "job:j1",
            "session:s1:jobs",
            "job:j1:cancel",
            "custom-model:m1",
            "session:s1:custom-models",
            "jacobian-lens:l1",
            "session:s1:jacobian-lenses",
        ]
        ttls = {key: await redis_module.job_redis.ttl(key) for key in keys}

        # Assert over the whole dict so a failure names the offending key rather
        # than stopping at the first one.  ttl == -1 means "no expiry set" and
        # ttl == -2 means "key missing"; the bound excludes both.
        offenders = {
            key: ttl for key, ttl in ttls.items() if not DAY_TTL_FLOOR < ttl <= DAY_TTL_CEIL
        }
        assert offenders == {}

    async def test_session_queue_ttl_is_only_set_by_put_queue(self):
        """DI-10: the queue key does not exist until the first write."""
        sid = await ensure_session(None)

        # ensure_session EXPIREs the queue key before anything creates it, which
        # is a no-op.  Documented here so a future eager-create is noticed.
        assert await redis_module.redis.ttl(k_queue(sid)) == -2

        await put_queue(sid, {"items": [], "processing": None, "completed": []})
        assert DAY_TTL_FLOOR < await redis_module.redis.ttl(k_queue(sid)) <= DAY_TTL_CEIL

    async def test_result_cache_ttl_is_shorter_than_the_session_ttl(self):
        """DI-11: the legacy result cache defaults to six hours, not a day."""
        await cache_result("whisper-base", "h1", {"prediction": "hello"})
        default_ttl = await redis_module.redis.ttl(k_result("whisper-base", "h1"))
        assert 21_400 < default_ttl <= 21_600
        assert default_ttl < settings.SESSION_TTL_SECONDS

        await cache_result("whisper-base", "h2", {"prediction": "hello"}, ttl=60)
        assert 0 < await redis_module.redis.ttl(k_result("whisper-base", "h2")) <= 60

    async def test_update_and_save_refresh_the_job_ttl(self):
        """DI-12: an active job's key slides forward instead of expiring."""
        record = job_record()
        await JobRepository().create(record)
        # Simulate a job that has been running for most of a day.
        await redis_module.job_redis.expire("job:j1", 100)

        await JobRepository().update("j1", task_id="celery-1")
        assert DAY_TTL_FLOOR < await redis_module.job_redis.ttl("job:j1") <= DAY_TTL_CEIL

        await redis_module.job_redis.expire("job:j1", 100)
        await JobRepository().save(record)
        assert DAY_TTL_FLOOR < await redis_module.job_redis.ttl("job:j1") <= DAY_TTL_CEIL


@pytest.mark.critical
class TestExpiryMidJob:
    async def test_update_returns_none_when_the_record_expired(self):
        """DI-13: a vanished job is reported, not resurrected."""
        await JobRepository().create(job_record())
        # Explicit delete rather than expire(key, 0): fakeredis compares the
        # expiry against a clock it re-stamps per command, so a zero TTL is not
        # reliably expired on the very next call.
        await redis_module.job_redis.delete("job:j1")

        result = await JobRepository().update("j1", status=JobStatus.success)

        assert result is None
        # The WATCH/GET path must not write a partial record back.
        assert await redis_module.job_redis.exists("job:j1") == 0

    @pytest.mark.important
    async def test_expired_audio_leaves_a_dangling_session_index_entry(self):
        """DI-14: the index over-reports once a record expires."""
        await AudioRepository().create(audio_asset(audio_id="a1"))
        await AudioRepository().create(audio_asset(audio_id="a2"))

        await redis_module.job_redis.delete("audio:a1")

        # The index still claims two members...
        assert await redis_module.job_redis.smembers("session:s1:audio") == {"a1", "a2"}
        # ...but the read path drops the one that no longer resolves.
        assert [a.audio_id for a in await AudioRepository().list_owned("s1")] == ["a2"]
        assert await AudioRepository().get_owned("a1", "s1") is None


# --------------------------------------------------------------------------
# C. Malformed and corrupt stored values
# --------------------------------------------------------------------------


@pytest.mark.critical
class TestCorruptValues:
    """Two deliberately different failure modes, depending on which layer reads.

    `app/core/redis.py` degrades: an unreadable value is logged and treated as
    absent, because both queue mutators read before writing, so raising would
    strand the session for the whole 24h TTL with no way back.

    Repositories instead refuse: `model_validate_json` raises ValidationError for
    bad syntax and bad shape alike, since substituting an empty record would
    quietly lose a job rather than surface a fault.

    An empty string is falsy at both layers, so a truncated-to-zero write is
    indistinguishable from a missing key either way.
    """

    async def test_get_queue_recovers_from_a_non_json_payload(self):
        """DI-15: a corrupt queue resets rather than stranding the session."""
        sid = await ensure_session(None)
        await redis_module.redis.set(k_queue(sid), "{truncated")

        assert await get_queue(sid) == {"items": [], "processing": None, "completed": []}

        # The session must be usable again immediately: add_item reads before it
        # writes, so a raising read would brick /queue/add for the full TTL.
        await add_item(sid, {"id": "a"})
        assert (await get_queue(sid))["items"] == [{"id": "a"}]

    async def test_get_queue_recovers_from_a_non_object_payload(self):
        """DI-15: valid JSON of the wrong type is refused just as firmly."""
        sid = await ensure_session(None)
        await redis_module.redis.set(k_queue(sid), json.dumps([1, 2, 3]))

        # A list survives json.loads but breaks every consumer downstream.
        assert await get_queue(sid) == {"items": [], "processing": None, "completed": []}

    async def test_get_queue_returns_default_for_empty_string(self):
        """DI-16: a zero-length value reads as absent."""
        sid = await ensure_session(None)
        await redis_module.redis.set(k_queue(sid), "")

        assert await get_queue(sid) == {"items": [], "processing": None, "completed": []}

    async def test_get_result_treats_a_corrupt_cache_entry_as_a_miss(self):
        """DI-17: a corrupt cache entry is a miss, so the caller recomputes."""
        await redis_module.redis.set(k_result("whisper-base", "h1"), "{truncated")
        assert await get_result("whisper-base", "h1") is None

        await redis_module.redis.set(k_result("whisper-base", "h2"), "")
        assert await get_result("whisper-base", "h2") is None

    @pytest.mark.parametrize(
        ("key", "read"),
        [
            ("audio:x", lambda: AudioRepository().get("x")),
            ("job:x", lambda: JobRepository().get("x")),
            ("custom-model:x", lambda: CustomModelRepository().get("x")),
            ("jacobian-lens:x", lambda: JacobianLensRepository().get("x")),
        ],
        ids=["audio", "job", "custom_model", "lens"],
    )
    async def test_repository_get_raises_validation_error_on_non_json(self, key, read):
        """DI-18: Pydantic wraps the decode failure -- not a JSONDecodeError."""
        await redis_module.job_redis.set(key, "not-json")

        with pytest.raises(ValidationError) as exc:
            await read()
        assert exc.value.errors()[0]["type"] == "json_invalid"

    async def test_repository_get_returns_none_for_empty_string(self):
        """DI-19: a truncated write is reported as "not found", not as an error."""
        await redis_module.job_redis.set("audio:x", "")
        await redis_module.job_redis.set("job:x", "")

        assert await AudioRepository().get("x") is None
        assert await JobRepository().get("x") is None

    async def test_update_propagates_validation_error_and_releases_the_watch(self):
        """DI-20: `update` only catches WatchError; the connection stays usable."""
        await redis_module.job_redis.set("job:j1", "{}")

        with pytest.raises(ValidationError):
            await JobRepository().update("j1", status=JobStatus.success)

        # The `finally: pipe.reset()` must have run -- a leaked WATCH would
        # poison every later command on this connection.
        await redis_module.job_redis.set("probe", "1")
        assert await redis_module.job_redis.get("probe") == "1"
        # The corrupt value is left exactly as found, not partially rewritten.
        assert await redis_module.job_redis.get("job:j1") == "{}"

    @pytest.mark.integration
    async def test_corrupt_queue_does_not_break_the_queue_endpoint(self, client):
        """DI-21: blast radius -- a corrupt queue degrades, it does not 500."""
        sid = (await client.get("/session")).json()["sid"]
        await redis_module.redis.set(k_queue(sid), "{truncated")

        response = await client.get("/queue")

        assert response.status_code == 200
        assert response.json() == {"items": [], "processing": None, "completed": []}

        # ...and the session can still be written to afterwards.
        added = await client.post("/queue/add", json={"id": "a"})
        assert added.status_code in (200, 201)


# --------------------------------------------------------------------------
# D. JobRepository.update invariants
# --------------------------------------------------------------------------


@pytest.mark.critical
class TestJobUpdateInvariants:
    async def test_terminal_status_blocks_a_different_status(self):
        """DI-22: a finished job cannot be moved, and is not rewritten."""
        await JobRepository().create(job_record(status=JobStatus.success))
        stored_before = await redis_module.job_redis.get("job:j1")

        returned = await JobRepository().update("j1", status=JobStatus.queued)

        assert returned.status == JobStatus.success
        # Assert on the *stored* bytes: a return-value-only assertion would pass
        # even if the early return had been removed and a write had happened.
        assert await redis_module.job_redis.get("job:j1") == stored_before

    async def test_terminal_lock_does_not_apply_when_status_matches(self):
        """DI-23: the guard is `status != record.status`, so a re-assert writes."""
        await JobRepository().create(
            job_record(status=JobStatus.failure, error=JobError(code="boom", message="fell over"))
        )

        returned = await JobRepository().update(
            "j1", status=JobStatus.failure, result_key="results/s1/j1/result.json"
        )

        assert returned.result_key == "results/s1/j1/result.json"
        # The write must not take the error down with it.
        assert returned.error is not None
        assert returned.error.code == "boom"

    async def test_terminal_lock_is_skipped_entirely_when_status_is_omitted(self):
        """DI-24: a task_id-only update must not erase why the job failed."""
        await JobRepository().create(
            job_record(status=JobStatus.failure, error=JobError(code="boom", message="fell over"))
        )

        returned = await JobRepository().update("j1", task_id="celery-123")

        assert returned.status == JobStatus.failure
        assert returned.task_id == "celery-123"
        # Real call sites do exactly this immediately after dispatch --
        # app/api/routes/jobs.py:159, app/worker/tasks.py:145 and :224 -- so a
        # job that failed first would otherwise surface as failure/error=null.
        assert returned.error.code == "boom"
        assert (await JobRepository().get("j1")).error.code == "boom"

    async def test_an_explicit_error_still_overwrites_the_stored_one(self):
        """DI-24b: guarding the assignment must not make errors unsettable."""
        await JobRepository().create(
            job_record(status=JobStatus.processing, error=JobError(code="first", message="one"))
        )

        returned = await JobRepository().update(
            "j1", error=JobError(code="second", message="two")
        )

        assert returned.error.code == "second"

    async def test_success_clears_a_stale_retryable_error(self):
        """DI-24c: a job that succeeded carries no error from its failed attempt."""
        # This is the path worker/executor.py takes on a transient failure: it
        # records a retryable error, re-raises for Celery, and the retry succeeds.
        await JobRepository().create(
            job_record(
                status=JobStatus.processing,
                error=JobError(code="transient_failure", message="Retrying", retryable=True),
            )
        )

        returned = await JobRepository().update("j1", status=JobStatus.success)

        assert returned.status == JobStatus.success
        assert returned.error is None

    async def test_progress_is_monotonic(self):
        """DI-25: stale progress is discarded, the rest of the call still applies."""
        await JobRepository().create(
            job_record(progress=JobProgress(current=5, total=10, message="Analysing"))
        )

        returned = await JobRepository().update(
            "j1",
            progress=JobProgress(current=2, total=10, message="stale"),
            result_key="results/s1/j1/result.json",
        )

        assert returned.progress.current == 5
        assert returned.progress.message == "Analysing"
        # Rejecting the progress must not reject the whole update.
        assert returned.result_key == "results/s1/j1/result.json"

    async def test_equal_progress_is_applied(self):
        """DI-26: the comparison is >=, so same-index progress replaces."""
        await JobRepository().create(
            job_record(progress=JobProgress(current=5, total=10, message="Analysing"))
        )

        returned = await JobRepository().update(
            "j1", progress=JobProgress(current=5, total=10, message="Queued")
        )

        assert returned.progress.message == "Queued"


class _RacingPipeline:
    """Injects a competing write immediately before EXEC.

    The competing write goes out on a separate pooled connection, so it notifies
    the watch that this pipeline is holding and EXEC comes back empty -- which
    redis-py surfaces as WatchError.  That is the real mechanism, not a
    simulated exception.
    """

    def __init__(self, pipe, client, key, payload, remaining):
        self._pipe = pipe
        self._client = client
        self._key = key
        self._payload = payload
        self.state = remaining

    def __getattr__(self, name):
        return getattr(self._pipe, name)

    async def execute(self, *args, **kwargs):
        if self.state["remaining"] > 0:
            self.state["remaining"] -= 1
            self.state["injected"] += 1
            await self._client.set(self._key, self._payload)
        return await self._pipe.execute(*args, **kwargs)


class _RacingClient:
    def __init__(self, inner, key, payload, remaining):
        self._inner = inner
        self._key = key
        self._payload = payload
        self.state = {"remaining": remaining, "injected": 0}

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def pipeline(self, *args, **kwargs):
        return _RacingPipeline(
            self._inner.pipeline(*args, **kwargs),
            self._inner,
            self._key,
            self._payload,
            self.state,
        )


@pytest.mark.critical
class TestJobUpdateConcurrency:
    """`JobRepository.update` is the only optimistic-concurrency code in ECHO."""

    @pytest.fixture
    def racing(self, monkeypatch):
        def _install(remaining: int, payload: str):
            client = _RacingClient(redis_module.job_redis, "job:j1", payload, remaining)
            # `update` reads `redis_module.job_redis` at call time, so patching
            # the module attribute is enough -- no need to touch the repository.
            monkeypatch.setattr(redis_module, "job_redis", client)
            return client

        return _install

    async def test_update_retries_after_a_concurrent_write(self, racing):
        """DI-28: a lost race is retried and re-reads the newer record."""
        await JobRepository().create(job_record())
        competitor = job_record(task_id="racer").model_dump_json()
        client = racing(remaining=2, payload=competitor)

        returned = await JobRepository().update("j1", status=JobStatus.processing)

        assert returned.status == JobStatus.processing
        # The competitor's field survived, proving the retry re-read the record
        # rather than replaying its own stale snapshot.
        assert returned.task_id == "racer"
        assert client.state["injected"] == 2

    async def test_five_failed_attempts_raise_runtime_error(self, racing):
        """DI-29: the repository fails loudly rather than dropping a transition."""
        await JobRepository().create(job_record())
        competitor = job_record(task_id="racer").model_dump_json()
        client = racing(remaining=99, payload=competitor)

        with pytest.raises(RuntimeError, match="Concurrent updates prevented job transition: j1"):
            await JobRepository().update("j1", status=JobStatus.processing)

        assert client.state["injected"] == 5
        assert await JobRepository().get("j1") is not None


# --------------------------------------------------------------------------
# E. Ownership and index hygiene
# --------------------------------------------------------------------------


@pytest.mark.important
@pytest.mark.security
class TestOwnership:
    @pytest.mark.parametrize(
        ("create", "get", "get_owned"),
        [
            (
                lambda: AudioRepository().create(audio_asset(audio_id="x")),
                lambda: AudioRepository().get("x"),
                lambda sid: AudioRepository().get_owned("x", sid),
            ),
            (
                lambda: JobRepository().create(job_record(job_id="x")),
                lambda: JobRepository().get("x"),
                lambda sid: JobRepository().get_owned("x", sid),
            ),
            (
                lambda: CustomModelRepository().create(custom_model_record(model_id="x")),
                lambda: CustomModelRepository().get("x"),
                lambda sid: CustomModelRepository().get_owned("x", sid),
            ),
            (
                lambda: JacobianLensRepository().create(lens_record(lens_id="x")),
                lambda: JacobianLensRepository().get("x"),
                lambda sid: JacobianLensRepository().get_owned("x", sid),
            ),
        ],
        ids=["audio", "job", "custom_model", "lens"],
    )
    async def test_get_owned_rejects_a_foreign_session(self, create, get, get_owned):
        """DI-30: cross-session reads are denied at the data layer."""
        await create()

        # Both halves matter: the second alone would also pass if the record
        # were simply missing.
        assert await get() is not None
        assert await get_owned("s1") is not None
        assert await get_owned("s2") is None

    async def test_delete_requires_ownership_and_leaves_the_record_intact(self):
        """DI-31: a foreign delete is a no-op, not a partial delete."""
        await AudioRepository().create(audio_asset())

        assert await AudioRepository().delete("a1", "s2") is None

        assert await redis_module.job_redis.exists("audio:a1") == 1
        assert await redis_module.job_redis.smembers("session:s1:audio") == {"a1"}

    async def test_list_owned_refilters_a_poisoned_index(self):
        """DI-32: a forged index entry cannot leak another session's record."""
        await AudioRepository().create(audio_asset(audio_id="a1", session_id="s1"))
        await AudioRepository().create(audio_asset(audio_id="b1", session_id="s2"))
        await CustomModelRepository().create(custom_model_record(model_id="m1", session_id="s1"))
        await CustomModelRepository().create(custom_model_record(model_id="m2", session_id="s2"))

        # Forge index entries pointing at records owned by s2.
        await redis_module.job_redis.sadd("session:s1:audio", "b1")
        await redis_module.job_redis.zadd("session:s1:custom-models", {"m2": T0.timestamp()})

        assert [a.audio_id for a in await AudioRepository().list_owned("s1")] == ["a1"]
        assert [r.model_id for r in await CustomModelRepository().list_owned("s1")] == ["m1"]

    async def test_deleted_job_removes_record_cancel_flag_and_index_entry(self):
        """DI-33: `delete` cleans up all four keys the job owns."""
        record = job_record(job_id="j1")
        await JobRepository().create(record)
        await JobRepository().create(job_record(job_id="j2"))
        await JobRepository().request_cancel("j1")
        await redis_module.job_redis.set("job:j1:completed-items", "x")

        # Note the signature: delete takes a JobRecord, not an id.
        await JobRepository().delete(record)

        assert await redis_module.job_redis.exists("job:j1") == 0
        assert await redis_module.job_redis.exists("job:j1:cancel") == 0
        assert await redis_module.job_redis.exists("job:j1:completed-items") == 0
        assert await JobRepository().list_session_job_ids("s1") == ["j2"]

    async def test_expired_job_leaves_a_dangling_zset_member(self):
        """DI-34: callers must tolerate index entries that no longer resolve."""
        await JobRepository().create(job_record(job_id="j1"))
        await redis_module.job_redis.delete("job:j1")

        # app/api/routes/analyses.py:281-284 handles exactly this hole.
        assert "j1" in await JobRepository().list_session_job_ids("s1")
        assert await JobRepository().get("j1") is None
