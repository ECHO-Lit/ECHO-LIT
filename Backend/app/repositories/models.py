from __future__ import annotations

from datetime import datetime, timezone

from app.core import redis as redis_module
from app.core.settings import settings
from app.schemas.models import CustomModelRecord


class CustomModelRepository:
    """Transient, session-owned custom model records stored with job metadata."""

    @staticmethod
    def _key(model_id: str) -> str:
        return f"custom-model:{model_id}"

    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"session:{session_id}:custom-models"

    async def create(self, record: CustomModelRecord) -> None:
        client = redis_module.job_redis
        pipe = client.pipeline()
        pipe.set(self._key(record.model_id), record.model_dump_json(), ex=settings.JOB_TTL_SECONDS)
        pipe.zadd(self._session_key(record.session_id), {record.model_id: record.created_at.timestamp()})
        pipe.expire(self._session_key(record.session_id), settings.JOB_TTL_SECONDS)
        await pipe.execute()

    async def get(self, model_id: str) -> CustomModelRecord | None:
        raw = await redis_module.job_redis.get(self._key(model_id))
        return CustomModelRecord.model_validate_json(raw) if raw else None

    async def get_owned(self, model_id: str, session_id: str) -> CustomModelRecord | None:
        record = await self.get(model_id)
        return record if record and record.session_id == session_id else None

    async def list_owned(self, session_id: str) -> list[CustomModelRecord]:
        ids = await redis_module.job_redis.zrevrange(self._session_key(session_id), 0, -1)
        records = [await self.get(model_id) for model_id in ids]
        return [record for record in records if record and record.session_id == session_id]

    async def save(self, record: CustomModelRecord) -> None:
        record.updated_at = datetime.now(timezone.utc)
        await redis_module.job_redis.set(
            self._key(record.model_id), record.model_dump_json(), ex=settings.JOB_TTL_SECONDS
        )

    async def delete(self, record: CustomModelRecord) -> None:
        client = redis_module.job_redis
        pipe = client.pipeline()
        pipe.delete(self._key(record.model_id))
        pipe.zrem(self._session_key(record.session_id), record.model_id)
        await pipe.execute()
