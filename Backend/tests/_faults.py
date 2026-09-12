"""Fault injection for the Section 3.1.7 failover and recovery modules.

Test Plan Section 3.1.7.  See tests/plans/3.1.7-failover-and-recovery.md.

The unit of failure here is a *component*, not a request: Redis going away and
coming back, Redis restarting from its append-only file, a worker process dying
with a task in hand.  Everything runs in-process against the real ASGI
application and the real worker code, so a case exercises exactly the code that
would run in production when that component fails.

`FaultHarness` owns one `fakeredis.FakeServer` that plays the role of the Redis
process.  Every client the application uses -- the API's asyncio clients, the
worker's synchronous clients, and one asyncio client per (thread, event loop) so
a worker thread and the API loop can share data -- is resolved through the
harness on every call.  That makes three failure modes cheap and exact:

  * `outage()`   -- the server refuses connections (`FakeServer.connected`),
                    every client raises `redis.exceptions.ConnectionError`, and
                    the same clients work again once it returns.
  * `restart()`  -- a new server process, repopulated key by key with
                    DUMP/PTTL/RESTORE as Redis replays its append-only file;
                    connections to the old process die.
  * `FaultyRedis` -- one client refuses chosen commands (an OOM refusal under
                    `noeviction`, one database unreachable).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import fakeredis
from fakeredis.aioredis import FakeRedis as AsyncFakeRedis
from httpx import ASGITransport, AsyncClient

from app.core import redis as redis_module
from app.main import app
from tests._load import Profile, SimulatedWorker

ORIGIN = "http://localhost:8080"  # an allowed CORS origin (app/main.py defaults)


class FaultHarness:
    """One swappable Redis server behind every client the application holds."""

    DATABASES = (0, 1, 2, 3)

    def __init__(self) -> None:
        self.server = fakeredis.FakeServer()
        self.generation = 0
        self._lock = threading.Lock()

    # -- clients ------------------------------------------------------------

    def async_client(self, db: int) -> "LoopLocalRedis":
        return LoopLocalRedis(self, db)

    def sync_client(self, db: int) -> "GenerationRedis":
        return GenerationRedis(self, db)

    def install(self, monkeypatch) -> None:
        """Point the API's and the worker's Redis clients at this harness."""
        from app.worker import tasks

        monkeypatch.setattr(redis_module, "redis", self.async_client(0))
        monkeypatch.setattr(redis_module, "job_redis", self.async_client(1))
        monkeypatch.setattr(redis_module, "broker_redis", self.async_client(2))
        monkeypatch.setattr(tasks, "_heartbeat_redis", self.sync_client(1))
        monkeypatch.setattr(tasks, "_session_redis", self.sync_client(0))

    def raw(self, db: int) -> fakeredis.FakeRedis:
        """A synchronous client on the current server, for seeding and inspection."""
        return fakeredis.FakeRedis(server=self.server, db=db, decode_responses=True)

    # -- failures -----------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self.server.connected

    def disconnect(self) -> None:
        self.server.connected = False

    def reconnect(self) -> None:
        self.server.connected = True

    @contextlib.contextmanager
    def outage(self):
        self.disconnect()
        try:
            yield self
        finally:
            self.reconnect()

    def restart(self, *, persist: bool = True) -> None:
        """Replace the Redis process.

        With `persist`, the new process is repopulated the way Redis replays an
        append-only file: every key with its value and remaining TTL.  Without
        it, the new process starts empty -- a restart with persistence off, or a
        lost volume.  Either way the old process's connections die, so a client
        still holding one fails and reconnects to the new process.
        """
        old = self.server
        new = fakeredis.FakeServer()
        if persist:
            for db in self.DATABASES:
                source = fakeredis.FakeRedis(server=old, db=db)
                target = fakeredis.FakeRedis(server=new, db=db)
                for key in source.scan_iter(count=1000):
                    ttl = source.pttl(key)
                    target.restore(key, ttl if ttl and ttl > 0 else 0, source.dump(key), replace=True)
        with self._lock:
            self.server = new
            self.generation += 1
        old.connected = False


class LoopLocalRedis:
    """An asyncio FakeRedis per (thread, event loop, server generation).

    redis.asyncio clients are bound to the loop that first used them, and each
    Celery worker process runs its own loop (`tasks._loop`), so the API and a
    worker need separate clients over the same data -- the pattern 3.1.5's
    worker-scaling module established.  The generation makes a restart visible:
    after it, every caller gets a client on the new server.
    """

    def __init__(self, harness: FaultHarness, db: int) -> None:
        self._harness = harness
        self._db = db
        self._clients: dict[tuple, AsyncFakeRedis] = {}
        self._lock = threading.Lock()

    def _client(self) -> AsyncFakeRedis:
        try:
            loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            loop_id = None
        key = (threading.get_ident(), loop_id, self._harness.generation)
        with self._lock:
            client = self._clients.get(key)
            if client is None:
                client = AsyncFakeRedis(server=self._harness.server, db=self._db, decode_responses=True)
                self._clients[key] = client
        return client

    def __getattr__(self, name):
        return getattr(self._client(), name)


class GenerationRedis:
    """A synchronous FakeRedis that follows the harness across restarts."""

    def __init__(self, harness: FaultHarness, db: int) -> None:
        self._harness = harness
        self._db = db
        self._clients: dict[int, fakeredis.FakeRedis] = {}

    def _client(self) -> fakeredis.FakeRedis:
        generation = self._harness.generation
        if generation not in self._clients:
            self._clients[generation] = fakeredis.FakeRedis(
                server=self._harness.server, db=self._db, decode_responses=True
            )
        return self._clients[generation]

    def __getattr__(self, name):
        return getattr(self._client(), name)


class FaultyRedis:
    """Delegating proxy that refuses chosen commands, in the style of `CountingRedis`.

    `fail(name)` returns the exception to raise for command `name`, or None to
    let it through.  Pipelines are refused when they are executed or watched,
    the points at which a real pipeline first talks to the server.
    """

    def __init__(self, inner, fail) -> None:
        self._inner = inner
        self._fail = fail

    def _check(self, name: str) -> None:
        exc = self._fail(name)
        if exc is not None:
            raise exc

    def pipeline(self, *args, **kwargs):
        return _FaultyPipeline(self._inner.pipeline(*args, **kwargs), self._check)

    def __getattr__(self, name):
        attribute = getattr(self._inner, name)
        if not callable(attribute):
            return attribute

        def _guarded(*args, **kwargs):
            self._check(name)
            return attribute(*args, **kwargs)

        return _guarded


class _FaultyPipeline:
    def __init__(self, inner, check) -> None:
        self._inner = inner
        self._check = check
        self._queued: list[str] = []

    async def execute(self, *args, **kwargs):
        for name in self._queued or ["execute"]:
            self._check(name)
        return await self._inner.execute(*args, **kwargs)

    async def watch(self, *keys):
        self._check("watch")
        return await self._inner.watch(*keys)

    def __getattr__(self, name):
        attribute = getattr(self._inner, name)
        if not callable(attribute) or name in {"reset", "multi"}:
            return attribute

        def _queue(*args, **kwargs):
            self._queued.append(name)
            result = attribute(*args, **kwargs)
            return self if result is self._inner else result

        return _queue


def oom_refusal(name: str):
    """What Redis answers a write under `maxmemory-policy noeviction` when full."""
    from redis.exceptions import ResponseError

    writes = {"set", "setex", "hset", "hsetnx", "expire", "sadd", "zadd", "incr", "hincrby", "execute"}
    if name in writes:
        return ResponseError("OOM command not allowed when used memory > 'maxmemory'.")
    return None


def tolerant_client(**kwargs) -> AsyncClient:
    """An API client that sees the response the ASGI server would send.

    httpx re-raises an application exception by default, which hides what a
    browser actually receives; with `raise_app_exceptions=False` an unhandled
    error arrives as Starlette's plain-text 500.
    """
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    return AsyncClient(transport=transport, base_url="http://test", **kwargs)


# --------------------------------------------------------------------------
# A worker that behaves like the Celery task under dependency failure
# --------------------------------------------------------------------------


class ResilientWorker(SimulatedWorker):
    """3.1.5's simulated worker, with the recovery behaviour of the real task.

    `execute_job` retries a Redis or storage failure with back-off (tasks.py);
    this worker does the same, so a job caught by an outage finishes once the
    outage ends.  `kill(n)` stops `n` consumers while each holds an envelope and
    puts the envelope back on the queue, which is what `acks_late` with
    `reject_on_worker_lost` does when a worker process dies mid-task.
    """

    def __init__(self, service_time: float, concurrency: int) -> None:
        super().__init__(service_time, concurrency)
        self.holding: dict[asyncio.Task, dict] = {}
        self.retries = 0
        self.redelivered = 0

    async def _run_one(self, envelope: dict) -> None:
        from app.core.storage import get_storage
        from app.repositories.jobs import JobRepository
        from app.schemas.jobs import JobProgress, JobStatus

        jobs = JobRepository()
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
        await asyncio.to_thread(get_storage().put_json, result_key, result)
        await jobs.update(job_id, status=JobStatus.success, result_key=result_key,
                          progress=JobProgress(current=total, total=total, message="Completed"))

    async def _consume(self) -> None:
        from redis.exceptions import RedisError

        me = asyncio.current_task()
        while True:
            envelope = await self.queue.get()
            self.holding[me] = envelope
            delay = 0.02
            while True:
                try:
                    await self._run_one(envelope)
                    break
                except RedisError:
                    self.retries += 1
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 0.2)
            self.holding.pop(me, None)

    def kill(self, n: int) -> int:
        """Kill up to `n` consumers that are mid-job; requeue what they held."""
        killed = 0
        for task, envelope in list(self.holding.items()):
            if killed == n:
                break
            if task in self._tasks and not task.done():
                task.cancel()
                self._tasks.remove(task)
                self.holding.pop(task, None)
                self.queue.put_nowait(envelope)
                self.redelivered += 1
                killed += 1
        return killed


# --------------------------------------------------------------------------
# Load under a fault schedule
# --------------------------------------------------------------------------


@dataclass
class Observation:
    started: float
    ended: float
    endpoint: str
    status: int
    retry_after: str | None
    allow_origin: str | None
    json_detail: bool


@dataclass
class FaultReport:
    profile: Profile
    observations: list[Observation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    accepted: dict[str, str] = field(default_factory=dict)   # job_id -> final status seen
    completed_after: dict[int, int] = field(default_factory=dict)  # user -> jobs completed after the fault
    windows: list[tuple[float, float]] = field(default_factory=list)
    elapsed: float = 0.0

    def statuses(self) -> set[int]:
        return {o.status for o in self.observations}

    def unavailable(self) -> list[Observation]:
        return [o for o in self.observations if o.status == 503]

    def outside_windows(self, observations, slack: float = 0.25) -> list[Observation]:
        def overlaps(o):
            return any(o.started <= end + slack and o.ended >= start - slack for start, end in self.windows)
        return [o for o in observations if not overlaps(o)]


async def run_with_faults(profile: Profile, monkeypatch, schedule) -> tuple[FaultReport, ResilientWorker]:
    """Drive `profile` while `schedule(report, worker, origin)` injects faults.

    Each virtual user retries a step that was refused with 503 -- the behaviour
    the Retry-After header asks for -- and records every response.  A status
    other than the expected one or 503 is an error.
    """
    from app.api.routes import jobs as jobs_routes
    from tests._fixtures import wav_bytes

    worker = ResilientWorker(profile.service_time, profile.workers)
    monkeypatch.setattr(jobs_routes.celery_app, "send_task", worker.send_task)
    worker.start()
    report = FaultReport(profile)
    audio = wav_bytes(seconds=1.0)
    origin = time.perf_counter()
    deadline = origin + profile.duration
    fault_over = asyncio.Event()

    async def call(index, endpoint, make, expected):
        while True:
            started = time.perf_counter() - origin
            response = await make()
            ended = time.perf_counter() - origin
            try:
                json_detail = isinstance(response.json().get("detail"), str)
            except Exception:
                json_detail = False
            report.observations.append(Observation(
                started, ended, endpoint, response.status_code, response.headers.get("retry-after"),
                response.headers.get("access-control-allow-origin"), json_detail,
            ))
            if response.status_code in expected:
                return response
            if response.status_code != 503:
                report.errors.append(f"user {index} {endpoint}: {response.status_code} {response.text[:160]}")
                return None
            if time.perf_counter() > deadline + 15:
                report.errors.append(f"user {index} {endpoint}: still unavailable after the run")
                return None
            await asyncio.sleep(profile.poll_interval)

    async def user(index: int) -> None:
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            if not await call(index, "session", lambda: client.get("/session"), {200}):
                return
            uploaded = await call(index, "upload", lambda: client.post(
                "/upload", files={"file": (f"user-{index}.wav", audio, "audio/wav")}), {201})
            if not uploaded:
                return
            audio_id = uploaded.json()["audio_id"]
            while time.perf_counter() < deadline or not fault_over.is_set():
                submitted = await call(index, "submit", lambda: client.post("/jobs", json={
                    "operation": "prediction", "model": "whisper-base", "audio_ids": [audio_id],
                }), {202})
                if not submitted:
                    return
                job_id = submitted.json()["job_id"]
                report.accepted[job_id] = "queued"
                while True:
                    polled = await call(index, "poll", lambda: client.get(f"/jobs/{job_id}"), {200})
                    if not polled:
                        return
                    report.accepted[job_id] = polled.json()["status"]
                    if polled.json()["status"] in {"success", "failure", "cancelled"}:
                        break
                    if time.perf_counter() > deadline + 15:
                        report.errors.append(f"user {index}: job {job_id} never finished")
                        return
                    await asyncio.sleep(profile.poll_interval)
                if report.accepted[job_id] != "success":
                    report.errors.append(f"user {index}: job {job_id} ended {report.accepted[job_id]}")
                    return
                if not await call(index, "result", lambda: client.get(f"/jobs/{job_id}/result"), {200}):
                    return
                if fault_over.is_set():
                    report.completed_after[index] = report.completed_after.get(index, 0) + 1
                await asyncio.sleep(profile.think)

    async def run_schedule():
        try:
            await schedule(report, worker, origin)
        finally:
            fault_over.set()

    try:
        await asyncio.gather(run_schedule(), *(user(i) for i in range(profile.users)))
    finally:
        await worker.stop()
    report.elapsed = time.perf_counter() - origin
    return report, worker


# --------------------------------------------------------------------------
# Seeding jobs and envelopes without going through HTTP
# --------------------------------------------------------------------------


async def seed_job(job_id: str, *, session_id: str = "s1", status=None, updated_at=None,
                   operation=None, audio_ids=None, error=None, total: int = 1):
    """Write a job record exactly as stored, including a back-dated `updated_at`.

    `JobRepository.create` keeps the record's timestamps (`save`/`update`
    would stamp the current time), which is what lets a case put a job in the
    state a crash left it in an hour ago.
    """
    from datetime import datetime, timezone

    from app.repositories.jobs import JobRepository
    from app.schemas.jobs import JobOperation, JobProgress, JobRecord, JobStatus

    now = datetime.now(timezone.utc)
    stamp = updated_at or now
    record = JobRecord(
        job_id=job_id, session_id=session_id, operation=operation or JobOperation.prediction,
        model="whisper-base", audio_ids=audio_ids or [f"{job_id}-a0"],
        status=status or JobStatus.queued,
        progress=JobProgress(current=0, total=total, message="Running"),
        created_at=stamp, updated_at=stamp, error=error,
    )
    await JobRepository().create(record)
    return record


def envelope_for(job_id: str, *, items: int = 1, session_id: str = "s1", operation: str = "prediction",
                 parameters: dict | None = None) -> dict:
    """A task envelope whose audio objects exist in the (isolated) object store."""
    import hashlib

    from app.core.settings import settings
    from app.core.storage import get_storage
    from app.schemas.jobs import TaskAudio, TaskEnvelope
    from tests._fixtures import wav_bytes

    storage = get_storage()
    audio = []
    for index in range(items):
        key = f"uploads/{session_id}/{job_id}-{index}.wav"
        path = storage.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wav_bytes())
        audio.append(TaskAudio(
            audio_id=f"{job_id}-a{index}", object_key=key, filename=f"clip-{index}.wav",
            media_type="audio/wav", sha256=hashlib.sha256(f"{job_id}-{index}".encode()).hexdigest(),
        ))
    return TaskEnvelope(
        job_id=job_id, session_id=session_id, operation=operation, model="whisper-base",
        audio=audio, parameters=parameters or {},
        result_schema_version=settings.RESULT_SCHEMA_VERSION, code_version=settings.CODE_VERSION,
    ).model_dump(mode="json")


def celery_request(task_name: str, args: list, task_id: str = "task-1"):
    """A real `celery.worker.request.Request` for `task_name`, with recorded ack/reject.

    `celery.contrib.testing.mocks.TaskMessage` builds a protocol-2 message; the
    Request is the worker's own object, so `on_failure` / `on_timeout` make the
    same decisions they make in a running worker, under ECHO's configuration.
    """
    from celery.contrib.testing.mocks import TaskMessage
    from celery.worker.request import Request

    from app.core.celery_app import celery_app

    outcome = SimpleNamespace(acked=0, rejected=[])
    message = TaskMessage(task_name, id=task_id, args=args)
    request = Request(
        message, app=celery_app, task=celery_app.tasks[task_name],
        on_ack=lambda *a: setattr(outcome, "acked", outcome.acked + 1),
        on_reject=lambda logger, errors, requeue: outcome.rejected.append(requeue),
    )
    return request, outcome


def exception_info(exc: BaseException):
    """A billiard ExceptionInfo for `exc`, as the pool hands to `on_failure`."""
    from billiard.einfo import ExceptionInfo

    try:
        raise exc
    except BaseException:
        return ExceptionInfo()
