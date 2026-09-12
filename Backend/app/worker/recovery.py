"""Recovery for jobs whose worker stopped without recording an outcome.

A job reaches `success`, `failure` or `cancelled` only because some task wrote
it there.  When every task that could have done so is gone -- the pool child hit
the hard time limit, the worker container was killed, the final failure write
could not reach Redis -- the record stays unfinished and the client polls it
until its TTL.  The pieces here close each of those gaps:

  * `reap_stale_jobs`, run by beat, fails jobs nothing has written to for
    longer than any task may run plus the broker's redelivery window;
  * `finish_failed_sync` records a hard time limit from the worker's main
    process, where the asyncio clients must not be used;
  * `describe_failure` turns an exception into the error the user reads.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
import logging

from celery.exceptions import SoftTimeLimitExceeded
from pydantic import ValidationError
from redis.exceptions import WatchError

from app.core import redis as redis_module
from app.core.settings import settings
from app.repositories.jobs import TERMINAL_STATES, JobRepository, is_stale
from app.schemas.jobs import JobError, JobRecord, JobStatus


logger = logging.getLogger(__name__)

WORKER_LOST_MESSAGE = "The worker running this job stopped responding. Submit it again to retry."
_SCAN_BATCH = 500


def describe_failure(exc: BaseException, code: str) -> JobError:
    """The error a user reads for `exc`.

    `SoftTimeLimitExceeded` is raised with no arguments, so its text is empty;
    it gets a message of its own rather than a blank "Analysis failed".
    """
    if isinstance(exc, SoftTimeLimitExceeded):
        minutes = settings.TASK_SOFT_TIME_LIMIT_SECONDS // 60
        return JobError(
            code="time_limit_exceeded",
            message=f"The job exceeded its {minutes}-minute time limit",
            retryable=False,
        )
    return JobError(code=code, message=(str(exc) or type(exc).__name__)[:500], retryable=False)


def hard_time_limit_error() -> JobError:
    minutes = settings.TASK_TIME_LIMIT_SECONDS // 60
    return JobError(
        code="time_limit_exceeded",
        message=f"The job exceeded its {minutes}-minute time limit and was stopped",
        retryable=False,
    )


def job_id_from_args(args: Iterable) -> str | None:
    """The job a task was working for: every job task takes its envelope as a positional argument."""
    for value in args or ():
        if isinstance(value, dict) and "job_id" in value and "session_id" in value:
            return value["job_id"]
    return None


def finish_failed_sync(client, job_id: str, error: JobError) -> bool:
    """Fail an unfinished job and ask its other tasks to stop, on a synchronous client.

    The same transition as `JobRepository.finish`, for the worker's main
    process: it runs Celery's timeout and failure callbacks, has no event loop
    of its own, and must not touch the asyncio clients that belong to the pool
    children's loops.  Returns whether the job was moved.
    """
    key = JobRepository._key(job_id)
    with client.pipeline(transaction=True) as pipe:
        for _ in range(5):
            try:
                pipe.watch(key)
                raw = pipe.get(key)
                if not raw:
                    return False
                record = JobRecord.model_validate_json(raw)
                if record.status in TERMINAL_STATES:
                    return False
                record.status = JobStatus.failure
                record.error = error
                record.updated_at = datetime.now(timezone.utc)
                pipe.multi()
                pipe.set(key, record.model_dump_json(), ex=settings.JOB_TTL_SECONDS)
                pipe.set(JobRepository._cancel_key(job_id), "1", ex=settings.JOB_TTL_SECONDS)
                pipe.execute()
                return True
            except WatchError:
                continue
    return False


async def _job_keys() -> list[str]:
    # The job database also holds `job:{id}:cancel`, `job:{id}:completed-items`
    # and per-stage counters; only `job:{id}` itself is a record.
    return [
        key async for key in redis_module.job_redis.scan_iter(match="job:*", count=_SCAN_BATCH)
        if key.count(":") == 1
    ]


async def reap_stale_jobs(now: datetime | None = None) -> int:
    """Fail every job whose worker has stopped responding; return how many.

    A job is stale when it is unfinished and nothing has written to it for
    `STALE_JOB_SECONDS` -- longer than a task may run plus the window in which
    the broker would have redelivered it.  The transition re-checks staleness
    inside its transaction, so a job a worker writes to meanwhile is left alone.
    A Redis error propagates: the run fails having changed nothing, and an
    outage can never be mistaken for every job having stopped.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=settings.STALE_JOB_SECONDS)
    keys = await _job_keys()
    stale: list[str] = []
    for start in range(0, len(keys), _SCAN_BATCH):
        batch = keys[start:start + _SCAN_BATCH]
        for key, raw in zip(batch, await redis_module.job_redis.mget(batch)):
            if not raw:
                continue
            try:
                record = JobRecord.model_validate_json(raw)
            except ValidationError:
                logger.warning("reaper skipping unreadable job record %s", key)
                continue
            if is_stale(record, cutoff):
                stale.append(record.job_id)

    jobs = JobRepository()
    error = JobError(code="worker_lost", message=WORKER_LOST_MESSAGE, retryable=True)
    reaped = 0
    for job_id in stale:
        record = await jobs.finish(job_id, JobStatus.failure, error=error, when=lambda r: is_stale(r, cutoff))
        if record is None:
            continue
        # Any of its tasks still alive somewhere stop at their next cancel check.
        await jobs.request_cancel(job_id)
        reaped += 1
        logger.warning("reaped_stale_job job_id=%s cutoff=%s", job_id, cutoff.isoformat())
    return reaped
