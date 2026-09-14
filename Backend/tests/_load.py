"""Workload driver for the Section 3.1.5 load modules.

Test Plan Section 3.1.5.  See tests/plans/3.1.5-load-testing.md.

A *virtual user* is its own `AsyncClient`, so it holds its own session cookie
and is a distinct session to the API -- the unit PE-1's "under normal load" is
about.  Each one runs the researcher's core loop against the real ASGI
application: establish a session, upload a clip, then repeatedly submit a job,
poll it to completion, fetch the result, and think.

Behind the broker boundary a `SimulatedWorker` stands in for Celery: it
receives exactly the envelope `send_task` would have published and advances the
job through the real `JobRepository` and the real object store, taking a fixed
service time.  That keeps the whole control-plane lifecycle -- submission,
status polling, result retrieval -- under load, while the model's own cost,
which PE-1 explicitly excludes, is a parameter rather than a GPU.

Everything a run observes lands in a `Report`: latency per endpoint class with
timestamps (so a sustained run can be checked for drift), every error, loop lag,
process RSS, and throughput.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace

import psutil
from httpx import AsyncClient

from app.core.storage import get_storage
from app.main import app
from app.repositories.jobs import JobRepository
from app.schemas.jobs import JobProgress, JobStatus
from tests._fixtures import wav_bytes
from tests._perf import LagProbe, Stats, loop_lag_probe


@dataclass(frozen=True)
class Profile:
    name: str
    users: int
    duration: float       # seconds the users keep starting new jobs
    think: float          # pause between one job's result and the next submission
    poll_interval: float  # the client's polling period while a job runs
    service_time: float   # simulated worker time per job
    workers: int          # simulated worker concurrency


# Three shapes, after the RUP template's "average vs peak vs sustained-peak".
# The Frontend polls every 1-2 s and a researcher pauses between analyses, so
# AVERAGE is a handful of users at that cadence.  PEAK removes the think time and
# polls ten times faster than the real client -- deliberately harsher than any
# human cohort.  SUSTAINED holds PEAK three times as long, to expose drift.
AVERAGE = Profile("average", users=5, duration=5.0, think=0.5, poll_interval=0.2, service_time=0.1, workers=4)
PEAK = Profile("peak", users=25, duration=5.0, think=0.0, poll_interval=0.05, service_time=0.05, workers=8)
SUSTAINED = Profile("sustained-peak", users=25, duration=15.0, think=0.0, poll_interval=0.05, service_time=0.05, workers=8)


@dataclass
class Report:
    profile: Profile
    timeline: list[tuple[float, str, float]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    completed_jobs: int = 0
    elapsed: float = 0.0
    lag: LagProbe | None = None
    rss_start: int = 0
    rss_end: int = 0

    def stats(self, endpoint: str | None = None, *, start: float = 0.0, end: float = float("inf")) -> Stats:
        samples = [
            latency for at, name, latency in self.timeline
            if (endpoint is None or name == endpoint) and start <= at < end
        ]
        return Stats(samples)

    @property
    def endpoints(self) -> list[str]:
        return sorted({name for _, name, _ in self.timeline})

    @property
    def throughput(self) -> float:
        return self.completed_jobs / self.elapsed if self.elapsed else 0.0

    def summary(self) -> str:
        lines = [
            f"{self.profile.name}: {self.completed_jobs} jobs in {self.elapsed:.1f}s "
            f"({self.throughput:.1f} jobs/s), errors={len(self.errors)}, "
            f"lag max={(self.lag.max if self.lag else 0) * 1000:.0f}ms, "
            f"rss +{(self.rss_end - self.rss_start) / 2**20:.1f}MiB"
        ]
        lines += [f"  {name:8s} {self.stats(name).ms()}" for name in self.endpoints]
        return "\n".join(lines)


class SimulatedWorker:
    """Consumes published envelopes and completes their jobs after `service_time`.

    `send_task` is called from a worker thread (the control plane publishes off
    the loop), so envelopes cross back onto the loop with call_soon_threadsafe.
    """

    def __init__(self, service_time: float, concurrency: int) -> None:
        self.service_time = service_time
        self.concurrency = concurrency
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self.published = 0
        self._loop = asyncio.get_running_loop()
        self._tasks: list[asyncio.Task] = []

    def send_task(self, name, args=None, kwargs=None, queue=None, **options):
        self.published += 1
        envelope = args[0]
        self._loop.call_soon_threadsafe(self.queue.put_nowait, envelope)
        return SimpleNamespace(id=f"sim-{self.published}")

    async def _consume(self) -> None:
        jobs = JobRepository()
        storage = get_storage()
        while True:
            envelope = await self.queue.get()
            job_id = envelope["job_id"]
            total = len(envelope["audio"])
            await jobs.update(job_id, status=JobStatus.processing,
                              progress=JobProgress(current=0, total=total, message="Processing"))
            await asyncio.sleep(self.service_time)
            result_key = f"results/{envelope['session_id']}/{job_id}/result.json"
            result = {
                "job_id": job_id, "operation": envelope["operation"], "model": envelope["model"],
                "items": [{"audio_id": a["audio_id"], "result": {"text": "the quick brown fox"}}
                          for a in envelope["audio"]],
                "metadata": {"cache_hit": False},
            }
            await asyncio.to_thread(storage.put_json, result_key, result)
            await jobs.update(job_id, status=JobStatus.success, result_key=result_key,
                              progress=JobProgress(current=total, total=total, message="Completed"))

    def start(self) -> None:
        self._tasks = [asyncio.create_task(self._consume()) for _ in range(self.concurrency)]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


async def _virtual_user(index: int, profile: Profile, report: Report, origin: float, deadline: float,
                        audio: bytes) -> None:
    async def call(endpoint: str, request, expected: set[int]):
        started = time.perf_counter()
        response = await request
        report.timeline.append((started - origin, endpoint, time.perf_counter() - started))
        if response.status_code not in expected:
            report.errors.append(f"user {index} {endpoint}: {response.status_code} {response.text[:200]}")
            return None
        return response

    async with AsyncClient(app=app, base_url="http://test") as client:
        if not await call("session", client.get("/session"), {200}):
            return
        uploaded = await call("upload", client.post(
            "/upload", files={"file": (f"user-{index}.wav", audio, "audio/wav")}), {201})
        if not uploaded:
            return
        audio_id = uploaded.json()["audio_id"]

        while time.perf_counter() < deadline:
            submitted = await call("submit", client.post("/jobs", json={
                "operation": "prediction", "model": "whisper-base", "audio_ids": [audio_id],
            }), {202})
            if not submitted:
                return
            job_id = submitted.json()["job_id"]
            # A job may outlive the submission window, never the run.
            while True:
                polled = await call("poll", client.get(f"/jobs/{job_id}"), {200})
                if not polled:
                    return
                if polled.json()["status"] == "success":
                    break
                if time.perf_counter() > deadline + 10:
                    report.errors.append(f"user {index}: job {job_id} never completed")
                    return
                await asyncio.sleep(profile.poll_interval)
            if not await call("result", client.get(f"/jobs/{job_id}/result"), {200}):
                return
            report.completed_jobs += 1
            await asyncio.sleep(profile.think)


async def run_profile(profile: Profile, monkeypatch) -> Report:
    """Drive `profile` against the in-process API and return what was observed.

    `monkeypatch` routes the broker to the simulated worker for the run.
    """
    from app.api.routes import jobs as jobs_routes

    worker = SimulatedWorker(profile.service_time, profile.workers)
    monkeypatch.setattr(jobs_routes.celery_app, "send_task", worker.send_task)
    worker.start()

    report = Report(profile)
    process = psutil.Process()
    report.rss_start = process.memory_info().rss
    audio = wav_bytes(seconds=1.0)
    origin = time.perf_counter()
    deadline = origin + profile.duration
    try:
        async with loop_lag_probe() as probe:
            await asyncio.gather(*(
                _virtual_user(index, profile, report, origin, deadline, audio)
                for index in range(profile.users)
            ))
    finally:
        await worker.stop()
    report.lag = probe
    report.elapsed = time.perf_counter() - origin
    report.rss_end = process.memory_info().rss
    return report


async def poll_latencies(client: AsyncClient, job_ids: list[str], duration: float) -> Stats:
    """Poll `job_ids` round-robin for `duration` seconds, as N open dashboards do."""
    samples: list[float] = []
    deadline = time.perf_counter() + duration

    async def poller(job_id: str) -> None:
        while time.perf_counter() < deadline:
            started = time.perf_counter()
            response = await client.get(f"/jobs/{job_id}")
            samples.append(time.perf_counter() - started)
            assert response.status_code == 200
            await asyncio.sleep(0.02)

    await asyncio.gather(*(poller(job_id) for job_id in job_ids))
    return Stats(samples)


def by_endpoint(report: Report) -> dict[str, Stats]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for _, name, latency in report.timeline:
        grouped[name].append(latency)
    return {name: Stats(values) for name, values in grouped.items()}
