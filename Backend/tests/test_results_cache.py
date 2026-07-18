import pytest

from app.core.redis import cache_result, get_result
from app.schemas.jobs import TaskEnvelope
from app.worker.executor import analysis_cache_key


@pytest.mark.asyncio
async def test_internal_cache_roundtrip(fake_redis):
    await cache_result("whisper-base", "content-key", {"prediction": "hello"})
    assert await get_result("whisper-base", "content-key") == {"prediction": "hello"}


@pytest.mark.asyncio
async def test_public_cache_api_is_gone(client):
    put = await client.post("/results/whisper-base/key", json={"prediction": "unsafe"})
    get = await client.get("/results/whisper-base/key")
    assert put.status_code == 410
    assert get.status_code == 410


def test_content_cache_key_is_stable_and_parameter_sensitive():
    base = {
        "job_id": "job",
        "session_id": "session",
        "operation": "prediction",
        "model": "whisper-base",
        "audio": [{
            "audio_id": "audio", "object_key": "uploads/session/audio.wav",
            "filename": "audio.wav", "media_type": "audio/wav", "sha256": "abc",
        }],
        "parameters": {},
        "execution_profile": "mock",
        "result_schema_version": "v1",
        "code_version": "test",
    }
    first = analysis_cache_key(TaskEnvelope.model_validate(base))
    assert first == analysis_cache_key(TaskEnvelope.model_validate(base))
    changed = {**base, "parameters": {"timestamps": True}}
    assert first != analysis_cache_key(TaskEnvelope.model_validate(changed))
    real = {**base, "execution_profile": "mps"}
    assert first != analysis_cache_key(TaskEnvelope.model_validate(real))
