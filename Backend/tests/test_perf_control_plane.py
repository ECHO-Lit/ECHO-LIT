"""Performance Profiling -- control-plane response time and per-request cost.

Test Plan Section 3.1.4.  See tests/plans/3.1.4-performance-profiling.md.

SRS PE-1: "Control plane API calls that do not perform inference (upload
acknowledgment, job creation, status polling, result retrieval) shall respond
within 500 ms on average under normal load.  Job submission shall return within
this budget regardless of the size of the underlying computation."

Three kinds of assertion, deliberately kept apart:

* **Budgets** (TestResponseTimeBudget) -- the SRS number, asserted on the mean
  of repeated samples against the real ASGI application.  In-process the
  measured means sit one to two orders of magnitude below 500 ms, so a failure
  here is a real regression, not runner noise.
* **Independence from computation size** -- the PE-1 clause that makes a
  control plane a control plane.  Asserted structurally where possible (the
  dataset is never read at submission) so the check does not depend on timing.
* **Round-trip and memory accounting** -- an N+1 pattern costs microseconds on
  fakeredis and a network hop per item in production, so it is asserted as a
  command count.  Memory is asserted as the largest single read a route makes
  from an upload, which is what bounds its footprint.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from httpx import AsyncClient
from starlette.datastructures import UploadFile

from app.api.routes import analyses as analyses_routes
from app.api.routes import jobs as jobs_routes
from app.core import redis as redis_module
from app.core.settings import settings
from app.core.storage import get_storage
from app.main import app
from app.repositories.audio import AudioRepository
from app.repositories.jacobian_lenses import JacobianLensRepository
from app.repositories.jobs import JobRepository
from app.repositories.models import CustomModelRepository
from app.schemas.jacobian_lens import JacobianLensRecord
from app.schemas.jobs import AudioAsset, JobOperation, JobProgress, JobRecord, JobStatus
from app.schemas.models import CustomModelRecord, CustomModelStatus
from app.services import custom_dataset_service, dataset_service
from tests._perf import CountingRedis, sample

pytestmark = [pytest.mark.performance, pytest.mark.critical]

BUDGET_SECONDS = 0.5  # SRS PE-1
MAX_BATCH = 200  # JobCreateRequest.audio_ids max_length
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


@pytest.fixture(autouse=True)
def isolated_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr(custom_dataset_service, "SESSIONS_BASE_DIR", tmp_path / "sessions")


@pytest.fixture(autouse=True)
def fake_broker(monkeypatch):
    calls: list[dict] = []

    def _send_task(name, args=None, kwargs=None, queue=None, **rest):
        calls.append({"name": name, "queue": queue})
        return SimpleNamespace(id=f"celery-{len(calls)}")

    monkeypatch.setattr(jobs_routes.celery_app, "send_task", _send_task)
    monkeypatch.setattr(jobs_routes.celery_app.control, "revoke", lambda *a, **k: None)
    return calls


async def _sid(client) -> str:
    return (await client.get("/session")).json()["sid"]


async def _seed_assets(session_id: str, count: int) -> list[str]:
    """Register `count` assets directly.

    Going through POST /upload would spend ~80 ms per file in ffprobe on this
    host -- the cost PP-02 measures on its own -- and would make a 200-file
    fixture take longer than the case it sets up.
    """
    repository = AudioRepository()
    ids = []
    for index in range(count):
        audio_id = f"{session_id[:8]}{index:06d}"
        await repository.create(AudioAsset(
            audio_id=audio_id, session_id=session_id,
            object_key=f"uploads/{session_id}/{audio_id}.wav", filename=f"clip-{index}.wav",
            media_type="audio/wav", size_bytes=32_000, duration_seconds=1.0,
            sample_rate=16_000, channels=1, sha256=f"{index:064x}", created_at=T0,
        ))
        ids.append(audio_id)
    return ids


def _wav_of(seconds: float) -> bytes:
    rng = np.random.default_rng(0)
    buffer = io.BytesIO()
    sf.write(buffer, (0.1 * rng.standard_normal(int(16_000 * seconds))).astype("float32"),
             16_000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _count_job_redis(monkeypatch) -> CountingRedis:
    counter = CountingRedis(redis_module.job_redis)
    monkeypatch.setattr(redis_module, "job_redis", counter)
    return counter


# --------------------------------------------------------------------------
# PE-1 budgets
# --------------------------------------------------------------------------


class TestResponseTimeBudget:
    async def test_session_establishment_meets_the_budget(self):
        """PP-01: every first contact creates a session; it must be cheap."""

        async def first_contact():
            async with AsyncClient(app=app, base_url="http://test") as fresh:
                response = await fresh.get("/session")
                assert response.status_code == 200

        stats = await sample(20, first_contact)
        print(stats.ms())
        assert stats.mean < BUDGET_SECONDS, stats.ms()

    @pytest.mark.parametrize("seconds", [60, 300], ids=["1-minute clip", "5-minute clip"])
    async def test_upload_acknowledgment_meets_the_budget(self, client, seconds):
        """PP-02: PE-1 names "upload acknowledgment" explicitly.

        A 5-minute 16 kHz clip is ~9.6 MB -- the size of a typical research
        recording.  The 100 MB boundary is profiled separately in PP-13, where
        the cost is dominated by bytes rather than by the control plane.
        """
        payload = _wav_of(seconds)

        async def upload():
            response = await client.post("/upload", files={"file": ("clip.wav", payload, "audio/wav")})
            assert response.status_code == 201, response.text

        stats = await sample(5, upload)
        print(stats.ms())
        assert stats.mean < BUDGET_SECONDS, stats.ms()

    async def test_job_submission_meets_the_budget(self, client):
        """PP-03: FR-4 AC-1 -- "returns a job_id within the control plane latency budget"."""
        [audio_id] = await _seed_assets(await _sid(client), 1)

        async def submit():
            response = await client.post(
                "/jobs", json={"operation": "prediction", "model": "whisper-base", "audio_ids": [audio_id]}
            )
            assert response.status_code == 202

        stats = await sample(20, submit)
        print(stats.ms())
        assert stats.mean < BUDGET_SECONDS, stats.ms()

    async def test_status_polling_meets_the_budget(self, client):
        """PP-04: the call a running job makes every one to five seconds for its whole life."""
        ids = await _seed_assets(await _sid(client), MAX_BATCH)
        created = await client.post(
            "/jobs", json={"operation": "prediction", "model": "whisper-base", "audio_ids": ids}
        )
        job_id = created.json()["job_id"]

        async def poll():
            response = await client.get(f"/jobs/{job_id}")
            assert response.status_code == 200

        stats = await sample(50, poll)
        print(stats.ms())
        assert stats.mean < BUDGET_SECONDS, stats.ms()

    async def test_result_retrieval_meets_the_budget(self, client):
        """PP-05: a maximal batch's embedding result -- 200 items x 768 floats (~3 MB JSON).

        Result retrieval is the only PE-1 call whose payload grows with the
        computation, so it is profiled at the batch cap.
        """
        sid = await _sid(client)
        rng = np.random.default_rng(0)
        result = {
            "job_id": "j-big", "operation": "embedding", "model": "whisper-base",
            "items": [
                {"audio_id": f"a{i}", "result": {"embedding": rng.standard_normal(768).round(6).tolist()}}
                for i in range(MAX_BATCH)
            ],
            "metadata": {"cache_hit": False},
        }
        get_storage().put_json(f"results/{sid}/j-big/result.json", result)
        await JobRepository().create(JobRecord(
            job_id="j-big", session_id=sid, operation=JobOperation.embedding, model="whisper-base",
            audio_ids=[f"a{i}" for i in range(MAX_BATCH)], status=JobStatus.success,
            result_key=f"results/{sid}/j-big/result.json",
            progress=JobProgress(current=MAX_BATCH, total=MAX_BATCH, message="Completed"),
            created_at=T0, updated_at=T0,
        ))

        async def fetch():
            response = await client.get("/jobs/j-big/result")
            assert response.status_code == 200
            assert len(response.json()["items"]) == MAX_BATCH

        stats = await sample(5, fetch)
        print(stats.ms())
        assert stats.mean < BUDGET_SECONDS, stats.ms()

    async def test_session_listing_meets_the_budget(self, client):
        """PP-06: the audio list the dashboard loads on mount, at 200 assets."""
        await _seed_assets(await _sid(client), MAX_BATCH)

        async def listing():
            response = await client.get("/upload/list")
            assert len(response.json()["files"]) == MAX_BATCH

        stats = await sample(10, listing)
        print(stats.ms())
        assert stats.mean < BUDGET_SECONDS, stats.ms()


# --------------------------------------------------------------------------
# PE-1: "regardless of the size of the underlying computation"
# --------------------------------------------------------------------------


class TestSubmissionIndependentOfComputationSize:
    async def test_a_maximum_batch_submits_within_half_the_budget(self, client):
        """PP-07: the largest batch the API accepts, with headroom to spare.

        Half the budget, because PE-1 is an *average under load* and a batch
        submission that already spends the whole budget unloaded cannot meet it
        once other users are active (see 3.1.5).
        """
        ids = await _seed_assets(await _sid(client), MAX_BATCH)

        async def submit():
            response = await client.post(
                "/jobs", json={"operation": "prediction", "model": "whisper-base", "audio_ids": ids}
            )
            assert response.status_code == 202

        stats = await sample(10, submit)
        print(stats.ms())
        assert stats.mean < BUDGET_SECONDS / 2, stats.ms()

    async def test_fairness_submission_never_reads_the_dataset(self, client, monkeypatch):
        """PP-08: FR-10 at its largest configuration does no dataset work in the request.

        Asserted structurally: the dataset loaders are replaced with a tripwire.
        A timing assertion would pass on a small bundled dataset and still hide
        an O(dataset) scan that only hurts on a large one.
        """

        def tripwire(*args, **kwargs):
            raise AssertionError("fairness submission read the dataset inside the request")

        monkeypatch.setattr(dataset_service, "load_metadata", tripwire)
        monkeypatch.setattr(analyses_routes, "groupable_columns", tripwire)

        async def submit():
            response = await client.post("/api/v1/analyses/fairness", json={
                "dataset": "l2-arctic", "grouping_key": ["accent"], "model": "whisper-base",
                "max_items_per_group": 2000, "n_bootstrap": 20000,
                "metrics": ["wer", "cer", "silhouette", "leakage"],
            })
            assert response.status_code == 202, response.text

        stats = await sample(10, submit)
        print(stats.ms())
        assert stats.mean < BUDGET_SECONDS, stats.ms()


# --------------------------------------------------------------------------
# Round trips -- the N+1 oracle
# --------------------------------------------------------------------------


async def _seed_models(session_id: str, count: int) -> None:
    repository = CustomModelRepository()
    for index in range(count):
        await repository.create(CustomModelRecord(
            model_id=f"custom-{session_id}-{index}", session_id=session_id, hf_repo=f"owner/model-{index}",
            status=CustomModelStatus.VALIDATING, created_at=T0, updated_at=T0,
        ))


async def _seed_lenses(session_id: str, count: int) -> None:
    repository = JacobianLensRepository()
    for index in range(count):
        await repository.create(JacobianLensRecord(
            lens_id=f"jlens-{session_id}-{index}", session_id=session_id, model_id="whisper-base",
            model_revision="rev", fit_job_id=f"job-{index}", created_at=T0, updated_at=T0,
            sample_count=2,
        ))


class TestRoundTrips:
    @pytest.mark.parametrize(
        ("seed", "lister"),
        [
            (_seed_assets, lambda sid: AudioRepository().list_owned(sid)),
            (_seed_models, lambda sid: CustomModelRepository().list_owned(sid)),
            (_seed_lenses, lambda sid: JacobianLensRepository().list_owned(sid)),
        ],
        ids=["audio", "custom-models", "jacobian-lenses"],
    )
    async def test_listing_costs_constant_round_trips(self, monkeypatch, seed, lister):
        """PP-09: a session listing costs the same Redis round trips at 5 items as at 200.

        Guards BUG-40.  Before the fix each listing issued one index read plus
        one GET per member -- 201 round trips for 200 assets, each a network hop
        against a real Redis.
        """
        await seed("s-small", 5)
        await seed("s-large", MAX_BATCH)
        counter = _count_job_redis(monkeypatch)

        small = await lister("s-small")
        small_trips = counter.round_trips
        counter.reset()
        large = await lister("s-large")
        large_trips = counter.round_trips

        assert len(small) == 5 and len(large) == MAX_BATCH
        assert large_trips == small_trips, dict(counter.calls)
        assert large_trips <= 2, dict(counter.calls)

    @pytest.mark.parametrize("route", ["jobs", "linguistic-vs-acoustic"])
    async def test_submission_costs_constant_round_trips(self, client, monkeypatch, route):
        """PP-10: resolving the submitted audio is one round trip, not one per item.

        Guards BUG-40 on the submission path, which PE-1 budgets most tightly.
        """
        sid = await _sid(client)
        # 20 items x (baseline + 2 pitch steps) = 60, FR-7's MAX_GRID_VARIANTS.
        ids = await _seed_assets(sid, 20)

        def body(audio_ids):
            if route == "jobs":
                return "/jobs", {"operation": "prediction", "model": "whisper-base", "audio_ids": audio_ids}
            return "/api/v1/analyses/linguistic-vs-acoustic", {
                "audio_ids": audio_ids, "model": "whisper-base",
                "sweeps": [{"property": "pitch", "steps": 2}], "include_lexical_control": False,
            }

        counter = _count_job_redis(monkeypatch)
        url, payload = body(ids[:1])
        assert (await client.post(url, json=payload)).status_code == 202
        one = counter.round_trips
        counter.reset()
        url, payload = body(ids)
        assert (await client.post(url, json=payload)).status_code == 202
        many = counter.round_trips

        assert many == one, dict(counter.calls)


# --------------------------------------------------------------------------
# /health -- the endpoint monitoring calls every few seconds
# --------------------------------------------------------------------------


class TestHealthCost:
    async def test_health_never_scans_the_keyspace(self, client, monkeypatch):
        """PP-11: guards BUG-39 -- `/health` issued `KEYS worker-heartbeat:*`.

        KEYS is O(every key in the database) and blocks the Redis server while
        it runs.  The job database holds every job, audio and model record, so
        the probe a monitor calls every few seconds grew with total usage.
        """
        counter = _count_job_redis(monkeypatch)

        response = await client.get("/health")

        assert response.status_code == 200
        assert "keys" not in counter.calls and "scan" not in counter.calls, dict(counter.calls)

    async def test_health_latency_is_flat_in_keyspace_size(self, client):
        """PP-12: the same guard observed as time -- 50 000 unrelated job keys."""
        empty = await sample(20, lambda: client.get("/health"))
        pipe = redis_module.job_redis.pipeline()
        for index in range(50_000):
            pipe.set(f"job:filler-{index}", "{}", ex=60)
        await pipe.execute()
        loaded = await sample(20, lambda: client.get("/health"))

        assert loaded.p50 < 2 * empty.p50 + 0.005, f"empty {empty.ms()} | loaded {loaded.ms()}"

    async def test_health_still_counts_live_workers(self, client):
        """PP-12b: the replacement keeps the metric -- fresh heartbeats count, stale do not."""
        from app.core.heartbeat import record_worker_heartbeat

        now = datetime.now(timezone.utc).timestamp()
        await record_worker_heartbeat(redis_module.job_redis, "worker-a", now)
        await record_worker_heartbeat(redis_module.job_redis, "worker-b", now - 10)
        await record_worker_heartbeat(redis_module.job_redis, "worker-dead", now - 3600)

        response = await client.get("/health")

        assert response.json()["workers"] == 2


# --------------------------------------------------------------------------
# Probe cost -- the dominant term of an upload acknowledgment
# --------------------------------------------------------------------------


class TestProbeCost:
    @pytest.mark.parametrize("fmt", ["WAV", "FLAC"])
    def test_common_formats_are_probed_without_spawning_a_process(self, tmp_path, monkeypatch, fmt):
        """PP-31: guards BUG-47 -- ffprobe ran first for every upload.

        A process spawn measured ~78 ms per probe on the development host
        against ~0.8 ms for libsndfile's in-process header read, and under a
        25-way burst of uploads it was the largest single term of LT-02's upload
        latency.  Asserted structurally: a spawn attempt fails the case.
        """
        from app.core import audio_probe

        def no_spawn(*args, **kwargs):
            raise AssertionError("probe_audio spawned a process for a format libsndfile reads")

        monkeypatch.setattr(audio_probe.subprocess, "run", no_spawn)
        path = tmp_path / f"clip.{fmt.lower()}"
        sf.write(path, np.zeros(16_000, dtype="float32"), 16_000, format=fmt)

        duration, sample_rate, channels = audio_probe.probe_audio(path)

        assert (round(duration, 3), sample_rate, channels) == (1.0, 16_000, 1)

    def test_containers_libsndfile_cannot_open_still_reach_ffprobe(self, tmp_path, monkeypatch):
        """PP-32: the fallback is kept -- M4A and anything else libsndfile rejects."""
        from app.core import audio_probe

        spawned = []

        def fake_ffprobe(command, **kwargs):
            spawned.append(command[0])
            return SimpleNamespace(stdout='{"streams": [{"sample_rate": "44100", "channels": 2}],'
                                          ' "format": {"duration": "3.5"}}')

        monkeypatch.setattr(audio_probe.subprocess, "run", fake_ffprobe)
        path = tmp_path / "clip.m4a"
        path.write_bytes(b"\x00\x00\x00\x20ftypM4A not a libsndfile container")

        assert audio_probe.probe_audio(path) == (3.5, 44_100, 2)
        assert spawned == ["ffprobe"]


# --------------------------------------------------------------------------
# Memory -- how much of an upload a route holds at once
# --------------------------------------------------------------------------


@pytest.fixture
def read_sizes(monkeypatch):
    """Record the size of every chunk a route reads from an UploadFile."""
    sizes: list[int] = []
    original = UploadFile.read

    async def recording_read(self, size: int = -1):
        data = await original(self, size)
        sizes.append(len(data))
        return data

    monkeypatch.setattr(UploadFile, "read", recording_read)
    return sizes


class TestUploadMemory:
    async def test_audio_upload_streams_in_bounded_chunks(self, client, read_sizes):
        """PP-13: `/upload` never holds more than 1 MiB of a file at once.

        This is what lets the 100 MB boundary file of PE-3 be accepted without
        the API's footprint growing by 100 MB per concurrent upload.  Positive
        control for PP-14.
        """
        payload = _wav_of(300)  # ~9.6 MB

        response = await client.post("/upload", files={"file": ("clip.wav", io.BytesIO(payload), "audio/wav")})

        assert response.status_code == 201
        assert sum(read_sizes) == len(payload)
        assert max(read_sizes) <= 1024 * 1024

    async def test_dataset_upload_streams_in_bounded_chunks(self, client, read_sizes):
        """PP-14: guards BUG-37 -- the dataset route read each file whole.

        `upload_files_to_dataset` called `await file.read()` with no size, so a
        20-file batch of 100 MB recordings held up to 100 MB per file in the API
        process, with no cap at all (see LT-31 for the missing limit itself).
        """
        await client.post("/upload/dataset/create", data={"dataset_name": "speech"})
        payload = _wav_of(300)

        response = await client.post(
            "/upload/dataset/speech/files",
            files=[("files", ("a.wav", io.BytesIO(payload), "audio/wav"))],
        )

        assert response.status_code == 200, response.text
        assert max(read_sizes) <= 1024 * 1024, f"largest single read: {max(read_sizes)} bytes"
