from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from redis.exceptions import WatchError

from app.core import redis as redis_module
from app.core.settings import settings
from app.schemas.jobs import JobError, JobProgress, JobRecord, JobStatus


TERMINAL_STATES = {JobStatus.success, JobStatus.failure, JobStatus.cancelled}


def is_stale(record: JobRecord, cutoff: datetime) -> bool:
    """Unfinished, untouched since `cutoff`, and not merely waiting for a worker.

    A `queued` job that never started is left alone however old it is: its
    message is durable in the broker and a worker will take it.  A queued job
    carrying an error has started before -- a transient failure put it back --
    so it is judged like a running one.
    """
    if record.status in TERMINAL_STATES or record.updated_at >= cutoff:
        return False
    return record.status != JobStatus.queued or record.error is not None


def unresponsive(record: JobRecord, now: datetime, seconds: int) -> bool:
    return is_stale(record, now - timedelta(seconds=seconds))


class JobRepository:
    @staticmethod
    def _key(job_id: str) -> str:
        return f"job:{job_id}"

    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"session:{session_id}:jobs"

    @staticmethod
    def _cancel_key(job_id: str) -> str:
        return f"job:{job_id}:cancel"

    async def create(self, record: JobRecord) -> None:
        client = redis_module.job_redis
        pipe = client.pipeline()
        pipe.set(self._key(record.job_id), record.model_dump_json(), ex=settings.JOB_TTL_SECONDS)
        pipe.zadd(self._session_key(record.session_id), {record.job_id: record.created_at.timestamp()})
        pipe.expire(self._session_key(record.session_id), settings.JOB_TTL_SECONDS)
        await pipe.execute()

    async def get(self, job_id: str) -> JobRecord | None:
        raw = await redis_module.job_redis.get(self._key(job_id))
        return JobRecord.model_validate_json(raw) if raw else None

    async def get_owned(self, job_id: str, session_id: str) -> JobRecord | None:
        record = await self.get(job_id)
        return record if record and record.session_id == session_id else None

    async def list_session_job_ids(self, session_id: str) -> list[str]:
        return await redis_module.job_redis.zrange(self._session_key(session_id), 0, -1)

    async def save(self, record: JobRecord) -> JobRecord:
        record.updated_at = datetime.now(timezone.utc)
        await redis_module.job_redis.set(
            self._key(record.job_id), record.model_dump_json(), ex=settings.JOB_TTL_SECONDS
        )
        return record

    async def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        progress: JobProgress | None = None,
        task_id: str | None = None,
        result_key: str | None = None,
        cache_hit: bool | None = None,
        error: JobError | None = None,
        child_task_ids: list[str] | None = None,
    ) -> JobRecord | None:
        client = redis_module.job_redis
        key = self._key(job_id)
        for _ in range(5):
            pipe = client.pipeline(transaction=True)
            try:
                await pipe.watch(key)
                raw = await pipe.get(key)
                if not raw:
                    await pipe.reset()
                    return None
                record = JobRecord.model_validate_json(raw)
                if record.status in TERMINAL_STATES and status and status != record.status:
                    await pipe.reset()
                    return record
                if status is not None:
                    record.status = status
                if progress is not None:
                    if progress.current >= record.progress.current:
                        record.progress = progress
                if task_id is not None:
                    record.task_id = task_id
                if result_key is not None:
                    record.result_key = result_key
                if cache_hit is not None:
                    record.cache_hit = cache_hit
                if child_task_ids is not None:
                    record.child_task_ids = child_task_ids
                # Only an explicit error overwrites the stored one. Updates that
                # say nothing about failure -- a task_id stamp, a progress tick --
                # must not silently erase why a job failed.
                if error is not None:
                    record.error = error
                # A job that reached success carries no error, even if an earlier
                # attempt recorded a retryable one (see the transient-failure path
                # in worker/executor.py, which sets an error and then retries).
                if record.status == JobStatus.success:
                    record.error = None
                record.updated_at = datetime.now(timezone.utc)
                pipe.multi()
                pipe.set(key, record.model_dump_json(), ex=settings.JOB_TTL_SECONDS)
                await pipe.execute()
                return record
            except WatchError:
                continue
            finally:
                await pipe.reset()
        raise RuntimeError(f"Concurrent updates prevented job transition: {job_id}")

    async def finish(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: JobError | None = None,
        message: str | None = None,
        when: Callable[[JobRecord], bool] | None = None,
    ) -> JobRecord | None:
        """Move an unfinished job to the terminal `status`; leave a finished one alone.

        Unlike `update`, which keeps a terminal record's status but would still
        overwrite its error, this changes nothing about a job that has already
        finished -- the first recorded reason is the one the user sees.  `when`
        is re-checked against the record read inside the transaction, so a job
        written to in the meantime is not moved.  Returns the new record, or
        None if the job was missing, finished, or no longer satisfied `when`.
        """
        client = redis_module.job_redis
        key = self._key(job_id)
        for _ in range(5):
            pipe = client.pipeline(transaction=True)
            try:
                await pipe.watch(key)
                raw = await pipe.get(key)
                if not raw:
                    return None
                record = JobRecord.model_validate_json(raw)
                if record.status in TERMINAL_STATES or (when is not None and not when(record)):
                    return None
                record.status = status
                if error is not None:
                    record.error = error
                if message is not None:
                    record.progress = JobProgress(
                        current=record.progress.current, total=record.progress.total, message=message
                    )
                record.updated_at = datetime.now(timezone.utc)
                pipe.multi()
                pipe.set(key, record.model_dump_json(), ex=settings.JOB_TTL_SECONDS)
                await pipe.execute()
                return record
            except WatchError:
                continue
            finally:
                await pipe.reset()
        raise RuntimeError(f"Concurrent updates prevented job transition: {job_id}")

    async def cancel(self, record: JobRecord, now: datetime | None = None) -> JobRecord:
        """Request cancellation, and apply it at once when no worker will act on it.

        A queued job has no worker yet.  A running job that has been silent for
        longer than the hard time limit cannot still be running the step it
        last reported: its worker is gone, and nothing would ever see the flag.
        """
        await self.request_cancel(record.job_id)
        now = now or datetime.now(timezone.utc)
        if record.status == JobStatus.queued or unresponsive(record, now, settings.TASK_TIME_LIMIT_SECONDS):
            return await self.finish(record.job_id, JobStatus.cancelled, message="Cancelled") or record
        return record

    async def request_cancel(self, job_id: str) -> None:
        await redis_module.job_redis.set(
            self._cancel_key(job_id), "1", ex=settings.JOB_TTL_SECONDS
        )

    async def cancellation_requested(self, job_id: str) -> bool:
        return bool(await redis_module.job_redis.exists(self._cancel_key(job_id)))

    async def delete(self, record: JobRecord) -> None:
        client = redis_module.job_redis
        pipe = client.pipeline()
        pipe.delete(self._key(record.job_id), self._cancel_key(record.job_id))
        pipe.delete(f"job:{record.job_id}:completed-items")
        pipe.zrem(self._session_key(record.session_id), record.job_id)
        await pipe.execute()
