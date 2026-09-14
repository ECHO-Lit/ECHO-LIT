"""Performance Profiling -- work that runs on the event loop.

Test Plan Section 3.1.4.  See tests/plans/3.1.4-performance-profiling.md.

The control plane is one asyncio process: every user's requests share one event
loop.  A synchronous call inside an `async def` route does not slow down only
the request that made it -- it freezes every other request on that worker until
it returns.  That is the mechanism behind two SRS clauses:

  PE-1  "...within 500 ms on average under normal load."
  PE-2  "Submitting a batch of many files shall not block the interface."

Each case runs one route while `loop_lag_probe` ticks on the same loop, with the
route's expensive dependency replaced by `blocking_cost(0.25)` -- a stand-in of
known duration for a broker publish, an ffprobe spawn or a dataset scan.  The
oracle is the loop's worst wake-up delay: under 100 ms if the cost ran off the
loop, at least 250 ms if it ran on it.  The stand-in's duration is fixed on
purpose, so the verdict depends on *where* the work runs, not on how fast this
host is.

Where the blocking behaviour is the product's own code with no stand-in needed
-- `resolve_file`'s retry sleep -- the real code runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.api.routes import analyses as analyses_routes
from app.api.routes import datasets as datasets_routes
from app.api.routes import jobs as jobs_routes
from app.core.settings import settings
from app.core.storage import get_storage
from app.repositories.audio import AudioRepository
from app.repositories.jobs import JobRepository
from app.schemas.jobs import AudioAsset, JobOperation, JobProgress, JobRecord, JobStatus
from app.services import custom_dataset_service
from tests._fixtures import upload_files
from tests._perf import blocking_cost, loop_lag_probe

pytestmark = [pytest.mark.performance, pytest.mark.critical]

COST = 0.25       # the stand-in's blocking duration
LAG_LIMIT = 0.10  # well above Windows' ~16 ms timer floor, well below COST
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


@pytest.fixture
def slow_broker(monkeypatch):
    """A broker whose publish and revoke each take COST seconds, as under a
    congested or reconnecting Redis."""
    counter = SimpleNamespace(n=0)

    def _send_task(*args, **kwargs):
        counter.n += 1
        return blocking_cost(COST, SimpleNamespace(id=f"celery-{counter.n}"))()

    monkeypatch.setattr(jobs_routes.celery_app, "send_task", _send_task)
    monkeypatch.setattr(jobs_routes.celery_app.control, "revoke", blocking_cost(COST))


async def _sid(client) -> str:
    return (await client.get("/session")).json()["sid"]


async def _asset(session_id: str) -> str:
    await AudioRepository().create(AudioAsset(
        audio_id="a1", session_id=session_id, object_key=f"uploads/{session_id}/a1.wav",
        filename="clip.wav", media_type="audio/wav", size_bytes=32_000, duration_seconds=2.0,
        sample_rate=16_000, channels=1, sha256="0" * 64, created_at=T0,
    ))
    return "a1"


async def _running_job(session_id: str, operation=JobOperation.prediction) -> str:
    await JobRepository().create(JobRecord(
        job_id="running", session_id=session_id, operation=operation, model="whisper-base",
        audio_ids=["a1"], status=JobStatus.processing, task_id="celery-running",
        progress=JobProgress(current=0, total=1, message="Processing"),
        created_at=T0, updated_at=T0,
    ))
    return "running"


class TestProbeCalibration:
    async def test_a_non_blocking_route_reads_below_the_limit(self, client):
        """PP-15: positive control -- the probe does not alarm on an ordinary request.

        Without this, every other case in the module could be failing for a
        reason unrelated to the route under test.
        """
        await _sid(client)
        async with loop_lag_probe() as probe:
            for _ in range(10):
                await client.get("/health")
        assert probe.max < LAG_LIMIT, f"idle lag {probe.max * 1000:.0f} ms"

    async def test_the_probe_detects_a_blocking_call(self):
        """PP-15b: negative control -- a 250 ms synchronous sleep is seen as >= 250 ms."""
        async with loop_lag_probe() as probe:
            blocking_cost(COST)()
        assert probe.max >= COST * 0.9


class TestBrokerCallsLeaveTheLoop:
    async def test_job_submission_does_not_block_on_the_broker(self, client, slow_broker):
        """PP-16: guards BUG-35 -- `celery_app.send_task` ran on the loop in POST /jobs."""
        audio_id = await _asset(await _sid(client))

        async with loop_lag_probe() as probe:
            response = await client.post(
                "/jobs", json={"operation": "prediction", "model": "whisper-base", "audio_ids": [audio_id]}
            )

        assert response.status_code == 202
        assert probe.max < LAG_LIMIT, f"loop blocked {probe.max * 1000:.0f} ms"

    async def test_linguistic_acoustic_submission_does_not_block_on_the_broker(self, client, slow_broker):
        """PP-17: guards BUG-35 on the FR-7 endpoint."""
        audio_id = await _asset(await _sid(client))

        async with loop_lag_probe() as probe:
            response = await client.post("/api/v1/analyses/linguistic-vs-acoustic", json={
                "audio_ids": [audio_id], "model": "whisper-base",
                "sweeps": [{"property": "pitch", "steps": 2}],
            })

        assert response.status_code == 202, response.text
        assert probe.max < LAG_LIMIT, f"loop blocked {probe.max * 1000:.0f} ms"

    async def test_fairness_submission_does_not_block_on_the_broker(self, client, slow_broker):
        """PP-18: guards BUG-35 on the FR-10 endpoint."""
        await _sid(client)

        async with loop_lag_probe() as probe:
            response = await client.post("/api/v1/analyses/fairness", json={
                "dataset": "l2-arctic", "grouping_key": ["accent"], "model": "whisper-base",
            })

        assert response.status_code == 202, response.text
        assert probe.max < LAG_LIMIT, f"loop blocked {probe.max * 1000:.0f} ms"

    async def test_cancelling_a_job_does_not_block_on_the_broker(self, client, slow_broker):
        """PP-19: guards BUG-35 -- `control.revoke` is a broadcast over the broker."""
        job_id = await _running_job(await _sid(client))

        async with loop_lag_probe() as probe:
            response = await client.delete(f"/jobs/{job_id}")

        assert response.status_code == 202
        assert probe.max < LAG_LIMIT, f"loop blocked {probe.max * 1000:.0f} ms"

    async def test_cancel_all_does_not_block_on_the_broker(self, client, slow_broker):
        """PP-20: guards BUG-35 on `/analyses/fairness/cancel-all`."""
        await _running_job(await _sid(client), JobOperation.fairness)

        async with loop_lag_probe() as probe:
            response = await client.post("/api/v1/analyses/fairness/cancel-all")

        assert response.json()["count"] == 1
        assert probe.max < LAG_LIMIT, f"loop blocked {probe.max * 1000:.0f} ms"


class TestDatasetWorkLeavesTheLoop:
    async def test_dataset_upload_does_not_block_the_interface(self, client, monkeypatch):
        """PP-21: guards BUG-36 -- the literal PE-2 clause.

        "Submitting a batch of many files shall not block the interface."  Each
        file was written, probed and its dataset's metadata rewritten on the
        loop.  ffprobe alone measured ~78 ms per file on the development host,
        so a 20-file batch froze every user of the API for ~1.6 s.  Five files at
        50 ms reproduce COST exactly.
        """
        monkeypatch.setattr(
            custom_dataset_service, "probe_audio", blocking_cost(COST / 5, (1.0, 16_000, 1))
        )
        await client.post("/upload/dataset/create", data={"dataset_name": "speech"})

        async with loop_lag_probe() as probe:
            response = await client.post(
                "/upload/dataset/speech/files", files=upload_files([f"c{i}.wav" for i in range(5)])
            )

        assert response.status_code == 200, response.text
        assert response.json()["total_files"] == 5
        assert probe.max < LAG_LIMIT, f"loop blocked {probe.max * 1000:.0f} ms"

    @pytest.mark.parametrize(
        ("attribute", "path"),
        [("load_metadata", "/ravdess/metadata"), ("compute_metadata_eda", "/ravdess/eda")],
        ids=["metadata", "eda"],
    )
    async def test_dataset_reads_do_not_block_the_interface(self, client, monkeypatch, attribute, path):
        """PP-22: guards BUG-38 -- a cold RAVDESS metadata load probes ~144 files.

        `load_metadata` uses a thread pool for the probes, but the route called
        it synchronously, so the loop waited on `pool.map` for the whole scan.
        """
        monkeypatch.setattr(datasets_routes, attribute, blocking_cost(COST, []))
        await _sid(client)

        async with loop_lag_probe() as probe:
            response = await client.get(path)

        assert response.status_code == 200
        assert probe.max < LAG_LIMIT, f"loop blocked {probe.max * 1000:.0f} ms"

    async def test_groupable_columns_do_not_block_the_interface(self, client, monkeypatch):
        """PP-23: guards BUG-38 on the FR-10 grouping dropdown's data source."""
        monkeypatch.setattr(
            analyses_routes, "groupable_columns",
            blocking_cost(COST, {
                "dataset": "l2-arctic", "n_rows": 0, "speaker_column": None,
                "content_column": None, "columns": [],
            }),
        )
        await _sid(client)

        async with loop_lag_probe() as probe:
            response = await client.get("/api/v1/analyses/fairness/groupable", params={"dataset": "l2-arctic"})

        assert response.status_code == 200, response.text
        assert probe.max < LAG_LIMIT, f"loop blocked {probe.max * 1000:.0f} ms"

    @pytest.mark.parametrize("route", ["serve", "materialize"])
    async def test_a_missing_dataset_file_does_not_freeze_the_api(self, client, route):
        """PP-24: guards BUG-38 -- the real `resolve_file`, no stand-in.

        `resolve_file` retries a missing built-in file three times with
        `time.sleep(0.1)` to ride out a Docker Desktop bind-mount race.  The
        retry is legitimate; running it on the loop is not: every 404 for a
        missing file froze the whole API for 300 ms.
        """
        await _sid(client)

        async with loop_lag_probe() as probe:
            if route == "serve":
                response = await client.get("/ravdess/file/does-not-exist.wav")
            else:
                response = await client.post(
                    "/audio/materialize", json={"dataset": "ravdess", "filename": "does-not-exist.wav"}
                )

        assert response.status_code == 404
        assert probe.max < LAG_LIMIT, f"loop blocked {probe.max * 1000:.0f} ms"
