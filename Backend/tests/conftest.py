import asyncio
import os
import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock, patch
from httpx import AsyncClient
from fakeredis.aioredis import FakeRedis

# Make app importable
from app.main import app
from app.core import redis as redis_module

@pytest.fixture(autouse=True, scope="function")
async def fake_redis(monkeypatch):
    """
    Replace the global redis client with fakeredis for each test.
    """
    client = FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_module, "redis", client)
    monkeypatch.setattr(redis_module, "job_redis", client)
    monkeypatch.setattr(redis_module, "broker_redis", client)
    yield
    await client.flushall()
    await client.aclose()

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test", follow_redirects=True) as ac:
        yield ac

# Additional fixtures for comprehensive testing based on Master Test Plan

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def sample_audio_data() -> np.ndarray:
    """Generate sample audio data for testing."""
    # 5 seconds of synthetic audio at 16kHz
    duration = 5.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Mix of sine waves to simulate speech
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 note
        0.2 * np.sin(2 * np.pi * 880 * t) +  # A5 note
        0.1 * np.random.normal(0, 0.05, len(t))  # Noise
    )
    return audio

@pytest.fixture
def sample_audio_file(temp_dir: Path, sample_audio_data: np.ndarray) -> Path:
    """Create a sample WAV file for testing."""
    file_path = temp_dir / "test_sample.wav"
    import soundfile as sf
    sf.write(file_path, sample_audio_data, 16000)
    return file_path

@pytest.fixture
def mock_model_outputs() -> Dict[str, Any]:
    """Standard mock outputs for model inference testing."""
    return {
        'whisper': {
            'predicted_transcript': 'This is a test transcription',
            'segments': [
                {'start': 0.0, 'end': 2.5, 'text': 'This is a test'},
                {'start': 2.5, 'end': 5.0, 'text': 'transcription'}
            ],
            'language': 'en',
            'attention': np.random.rand(6, 1500, 80).tolist()
        },
        'wav2vec2': {
            'predicted_emotion': 'neutral',
            'confidence': 0.85,
            'all_predictions': {
                'neutral': 0.85, 'happy': 0.08, 'sad': 0.04, 
                'angry': 0.02, 'fear': 0.01
            },
            'attention': np.random.rand(12, 12, 100, 100).tolist()
        }
    }

@pytest.fixture
def performance_thresholds() -> Dict[str, float]:
    """Performance testing thresholds from test plan."""
    return {
        'model_inference_max_time': 10.0,  # seconds
        'audio_upload_max_time': 5.0,
        'cache_retrieval_max_time': 0.05,
        'ui_response_max_time': 0.1,
        'max_memory_usage_mb': 2048
    }

@pytest.fixture
def test_session_data() -> Dict[str, Any]:
    """Standard session data for testing."""
    return {
        'session_id': 'test_session_12345',
        'user_id': 'test_user',
        'created_at': '2025-10-01T12:00:00Z',
        'uploaded_files': [],
        'analysis_history': []
    }
