from __future__ import annotations

from datetime import datetime, timezone
from redis.exceptions import WatchError

from app.core import redis as redis_module
from app.core.settings import settings
from app.schemas.jobs import JobError, JobProgress, JobRecord, JobStatus


TERMINAL_STATES = {JobStatus.success, JobStatus.failure, JobStatus.cancelled}


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
                record.error = error
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
