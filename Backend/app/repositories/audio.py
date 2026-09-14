from __future__ import annotations

import json

from app.core import redis as redis_module
from app.core.settings import settings
from app.schemas.jobs import AudioAsset


class AudioRepository:
    @staticmethod
    def _key(audio_id: str) -> str:
        return f"audio:{audio_id}"

    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"session:{session_id}:audio"

    async def create(self, asset: AudioAsset) -> None:
        client = redis_module.job_redis
        pipe = client.pipeline()
        pipe.set(self._key(asset.audio_id), asset.model_dump_json(), ex=settings.JOB_TTL_SECONDS)
        pipe.sadd(self._session_key(asset.session_id), asset.audio_id)
        pipe.expire(self._session_key(asset.session_id), settings.JOB_TTL_SECONDS)
        await pipe.execute()

    async def get(self, audio_id: str) -> AudioAsset | None:
        raw = await redis_module.job_redis.get(self._key(audio_id))
        return AudioAsset.model_validate_json(raw) if raw else None

    async def get_owned(self, audio_id: str, session_id: str) -> AudioAsset | None:
        asset = await self.get(audio_id)
        return asset if asset and asset.session_id == session_id else None

    async def get_many(self, audio_ids: list[str]) -> list[AudioAsset | None]:
        """One MGET for many records, in the order asked; a missing one is None.

        A loop of GETs costs one network round trip per item -- 200 for a
        maximal batch -- on the request paths PE-1 budgets at 500 ms.
        """
        if not audio_ids:
            return []
        raws = await redis_module.job_redis.mget([self._key(audio_id) for audio_id in audio_ids])
        return [AudioAsset.model_validate_json(raw) if raw else None for raw in raws]

    async def get_owned_many(self, audio_ids: list[str], session_id: str) -> list[AudioAsset | None]:
        return [
            asset if asset and asset.session_id == session_id else None
            for asset in await self.get_many(audio_ids)
        ]

    async def delete(self, audio_id: str, session_id: str) -> AudioAsset | None:
        asset = await self.get_owned(audio_id, session_id)
        if not asset:
            return None
        client = redis_module.job_redis
        pipe = client.pipeline()
        pipe.delete(self._key(audio_id))
        pipe.srem(self._session_key(session_id), audio_id)
        await pipe.execute()
        return asset

    async def list_owned(self, session_id: str) -> list[AudioAsset]:
        ids = await redis_module.job_redis.smembers(self._session_key(session_id))
        return [asset for asset in await self.get_owned_many(list(ids), session_id) if asset]
