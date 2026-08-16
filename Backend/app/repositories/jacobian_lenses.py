"""Session-scoped index of fitted Jacobian-lens artifacts."""

from __future__ import annotations

from datetime import datetime, timezone

from app.core import redis as redis_module
from app.core.settings import settings
from app.schemas.jacobian_lens import JacobianLensRecord


class JacobianLensRepository:
    @staticmethod
    def _key(lens_id: str) -> str:
        return f"jacobian-lens:{lens_id}"

    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"session:{session_id}:jacobian-lenses"

    async def create(self, record: JacobianLensRecord) -> None:
        client = redis_module.job_redis
        pipe = client.pipeline()
        pipe.set(self._key(record.lens_id), record.model_dump_json(), ex=settings.JOB_TTL_SECONDS)
        pipe.zadd(self._session_key(record.session_id), {record.lens_id: record.created_at.timestamp()})
        pipe.expire(self._session_key(record.session_id), settings.JOB_TTL_SECONDS)
        await pipe.execute()

    async def get(self, lens_id: str) -> JacobianLensRecord | None:
        raw = await redis_module.job_redis.get(self._key(lens_id))
        return JacobianLensRecord.model_validate_json(raw) if raw else None

    async def get_owned(self, lens_id: str, session_id: str) -> JacobianLensRecord | None:
        record = await self.get(lens_id)
        return record if record and record.session_id == session_id else None

    async def list_owned(self, session_id: str) -> list[JacobianLensRecord]:
        ids = await redis_module.job_redis.zrevrange(self._session_key(session_id), 0, -1)
        records = [await self.get(lens_id) for lens_id in ids]
        return [record for record in records if record and record.session_id == session_id]

    async def save(self, record: JacobianLensRecord) -> None:
        record.updated_at = datetime.now(timezone.utc)
        await redis_module.job_redis.set(
            self._key(record.lens_id), record.model_dump_json(), ex=settings.JOB_TTL_SECONDS
        )
