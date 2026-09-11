"""Load Testing -- interactive latency while heavy work runs in the background.

Test Plan Section 3.1.5.  See tests/plans/3.1.5-load-testing.md.

SRS PE-2: "Submitting a batch of many files shall not block the interface."
SRS US-3: "...the interface shall remain fully interactive while computation
proceeds in the background."

The RUP template's Special Considerations call for a background workload on the
server while the measured transactions run.  Here the measured transaction is
the one a researcher's open dashboard repeats for as long as a job runs --
`GET /jobs/{id}` -- issued by ten concurrent pollers, and the background
workload is one *other* user doing the heaviest thing the API accepts.  The
oracle compares the pollers against their own unloaded baseline from the same
case, so host speed cancels out, and checks the loop never stalled.

TestConcurrentWriters covers the other load hazard the fixes for BUG-36 created:
once dataset writes leave the event loop, two requests can genuinely overlap.
"""

from __future__ import annotations

import asyncio
import io
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from httpx import AsyncClient

from app.api.routes import datasets as datasets_routes
from app.api.routes import jobs as jobs_routes
from app.core.settings import settings
from app.core.storage import get_storage
from app.main import app
from app.repositories.audio import AudioRepository
from app.repositories.jobs import JobRepository
from app.schemas.jobs import AudioAsset, JobOperation, JobProgress, JobRecord, JobStatus
from app.services import custom_dataset_service
from tests._fixtures import upload_files
from tests._perf import Stats, blocking_cost, loop_lag_probe, sample

pytestmark = [pytest.mark.performance, pytest.mark.slow]

LAG_LIMIT = 0.10
POLLERS = 10
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
    monkeypatch.setattr(jobs_routes.celery_app, "send_task", lambda *a, **k: SimpleNamespace(id="t"))


@pytest.fixture
async def dashboard():
    """A session with POLLERS running jobs, as a researcher's open dashboard."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        sid = (await client.get("/session")).json()["sid"]
        job_ids = []
        for index in range(POLLERS):
            job_id = f"dash-{index}"
            await JobRepository().create(JobRecord(
                job_id=job_id, session_id=sid, operation=JobOperation.prediction, model="whisper-base",
                audio_ids=["a"], status=JobStatus.processing,
                progress=JobProgress(current=0, total=1, message="Processing"),
                created_at=T0, updated_at=T0,
            ))
            job_ids.append(job_id)
        yield SimpleNamespace(client=client, job_ids=job_ids)


async def _poll_until(client, job_ids, done: asyncio.Event, minimum: float) -> Stats:
    samples: list[float] = []
    started = time.perf_counter()

    async def poller(job_id):
        while not done.is_set() or time.perf_counter() - started < minimum:
            t = time.perf_counter()
            response = await client.get(f"/jobs/{job_id}")
            samples.append(time.perf_counter() - t)
            assert response.status_code == 200
            await asyncio.sleep(0.02)

    await asyncio.gather(*(poller(job_id) for job_id in job_ids))
    return Stats(samples)


async def _measure(dashboard, background) -> tuple[Stats, Stats, float, object]:
    """Baseline the pollers alone, then again while `background()` runs."""
    idle = asyncio.Event()
    idle.set()
    baseline = await _poll_until(dashboard.client, dashboard.job_ids, idle, minimum=1.0)

    done = asyncio.Event()
    outcome = {}

    async def run_background():
        try:
            outcome["value"] = await background()
        finally:
            done.set()

    async with loop_lag_probe() as probe:
        task = asyncio.create_task(run_background())
        loaded = await _poll_until(dashboard.client, dashboard.job_ids, done, minimum=0.5)
        await task
    return baseline, loaded, probe.max, outcome.get("value")


def _assert_unblocked(baseline: Stats, loaded: Stats, lag: float) -> None:
    detail = f"baseline {baseline.ms()} | loaded {loaded.ms()} | lag {lag * 1000:.0f} ms"
    assert lag < LAG_LIMIT, detail
    assert loaded.p95 < 2 * baseline.p95 + 0.02, detail


def _long_wav(seconds: float, sr: int = 44_100, channels: int = 2) -> bytes:
    buffer = io.BytesIO()
    frames = (0.05 * np.random.default_rng(0).standard_normal((int(sr * seconds), channels))).astype("float32")
    sf.write(buffer, frames, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class TestBackgroundWorkload:
    async def test_a_dataset_batch_upload_does_not_block_other_users(self, dashboard, monkeypatch):
        """LT-10: guards BUG-36 -- the literal PE-2 clause, under load.

        Another user uploads a 20-file batch.  The probe stand-in costs 50 ms per
        file, below the ~78 ms ffprobe measured on the development host; before
        the fix the pollers stalled for the whole second of it.
        """
        monkeypatch.setattr(custom_dataset_service, "probe_audio", blocking_cost(0.05, (1.0, 16_000, 1)))

        async def curator():
            async with AsyncClient(app=app, base_url="http://test") as other:
                await other.post("/upload/dataset/create", data={"dataset_name": "corpus"})
                response = await other.post(
                    "/upload/dataset/corpus/files", files=upload_files([f"c{i}.wav" for i in range(20)])
                )
                return response.status_code, response.json()["total_files"]

        baseline, loaded, lag, outcome = await _measure(dashboard, curator)

        assert outcome == (200, 20)
        _assert_unblocked(baseline, loaded, lag)

    async def test_maximum_batch_submissions_do_not_block_other_users(self, dashboard):
        """LT-11: another user submits five 200-file batches back to back (PE-2)."""

        async def batch_user():
            async with AsyncClient(app=app, base_url="http://test") as other:
                sid = (await other.get("/session")).json()["sid"]
                repository = AudioRepository()
                ids = []
                for index in range(200):
                    audio_id = f"bulk{index:04d}"
                    await repository.create(AudioAsset(
                        audio_id=audio_id, session_id=sid, object_key=f"uploads/{sid}/{audio_id}.wav",
                        filename=f"{audio_id}.wav", media_type="audio/wav", size_bytes=1, duration_seconds=1.0,
                        sample_rate=16_000, channels=1, sha256=f"{index:064x}", created_at=T0,
                    ))
                    ids.append(audio_id)
                statuses = []
                for _ in range(5):
                    response = await other.post(
                        "/jobs", json={"operation": "prediction", "model": "whisper-base", "audio_ids": ids}
                    )
                    statuses.append(response.status_code)
                return statuses

        baseline, loaded, lag, outcome = await _measure(dashboard, batch_user)

        assert outcome == [202] * 5
        _assert_unblocked(baseline, loaded, lag)

    async def test_a_cold_dataset_scan_does_not_block_other_users(self, dashboard, monkeypatch):
        """LT-12: guards BUG-38 -- a cold built-in metadata load, three times over."""
        monkeypatch.setattr(datasets_routes, "load_metadata", blocking_cost(0.3, []))

        async def explorer():
            async with AsyncClient(app=app, base_url="http://test") as other:
                return [(await other.get("/ravdess/metadata")).status_code for _ in range(3)]

        baseline, loaded, lag, outcome = await _measure(dashboard, explorer)

        assert outcome == [200, 200, 200]
        _assert_unblocked(baseline, loaded, lag)

    async def test_a_boundary_size_upload_does_not_block_other_users(self, dashboard):
        """LT-13: PE-3's largest accepted file (~96 MB, 9.5 minutes) alongside the pollers."""
        payload = _long_wav(570)
        assert 90 * 2**20 < len(payload) <= settings.MAX_UPLOAD_BYTES

        async def uploader():
            async with AsyncClient(app=app, base_url="http://test", timeout=120) as other:
                response = await other.post(
                    "/upload", files={"file": ("long.wav", io.BytesIO(payload), "audio/wav")}
                )
                return response.status_code, round(response.json()["duration_seconds"])

        baseline, loaded, lag, outcome = await _measure(dashboard, uploader)

        assert outcome == (201, 570)
        _assert_unblocked(baseline, loaded, lag)


class TestConcurrentWriters:
    async def test_concurrent_uploads_to_one_dataset_keep_every_file(self, client, monkeypatch):
        """LT-14: four overlapping batch uploads into one dataset lose nothing.

        Guards the per-dataset lock added with the BUG-36 fix.  Each upload
        reads the dataset's metadata, adds its files and writes it back; off the
        event loop and unlocked, two of those interleave and the later write
        silently drops the earlier batch's entries while its audio stays on disk.
        """
        monkeypatch.setattr(custom_dataset_service, "probe_audio", blocking_cost(0.02, (1.0, 16_000, 1)))
        await client.post("/upload/dataset/create", data={"dataset_name": "shared"})

        responses = await asyncio.gather(*(
            client.post("/upload/dataset/shared/files",
                        files=upload_files([f"b{batch}-{i}.wav" for i in range(5)]))
            for batch in range(4)
        ))

        assert [r.status_code for r in responses] == [200] * 4
        listing = (await client.get("/upload/dataset/shared/files")).json()
        assert listing["total_files"] == 20
        assert sorted(f["filename"] for f in listing["files"]) == sorted(
            f"b{batch}-{i}.wav" for batch in range(4) for i in range(5)
        )

    async def test_concurrent_uploads_of_one_name_get_distinct_files(self, client, monkeypatch):
        """LT-15: the collision-renaming check-then-write is also inside the lock."""
        monkeypatch.setattr(custom_dataset_service, "probe_audio", blocking_cost(0.02, (1.0, 16_000, 1)))
        await client.post("/upload/dataset/create", data={"dataset_name": "dupes"})

        await asyncio.gather(*(
            client.post("/upload/dataset/dupes/files", files=upload_files(["clip.wav"])) for _ in range(4)
        ))

        names = sorted(f["filename"] for f in (await client.get("/upload/dataset/dupes/files")).json()["files"])
        assert names == ["clip.wav", "clip_1.wav", "clip_2.wav", "clip_3.wav"]
