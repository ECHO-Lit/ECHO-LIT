"""Regression tests for the CPU worker's native-crash loop (SIGSEGV in librosa's localmax).

Two halves: the audio-features path must no longer reach librosa's numba stencil
kernel, and a task whose worker child keeps dying must eventually fail its job instead
of being redelivered forever under task_reject_on_worker_lost.
"""

import fakeredis
import numpy as np
import pytest
import soundfile as sf
from redis.exceptions import ConnectionError as RedisConnectionError

import librosa
import librosa.util.utils as librosa_utils

# Resolve librosa's own localmax before audio_features_service replaces it.
LIBROSA_LOCALMAX = librosa_utils.localmax

from app.services import audio_features_service  # noqa: E402
from app.worker import tasks  # noqa: E402


class TestLocalmaxReplacement:
    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64, np.float32, np.float64])
    @pytest.mark.parametrize("shape", [(2,), (50,), (7, 40), (3, 5, 11), (1025, 60)])
    def test_matches_librosa_on_every_axis(self, dtype, shape):
        rng = np.random.default_rng(0)
        if np.issubdtype(dtype, np.integer):
            x = rng.integers(-3, 4, size=shape).astype(dtype)  # plenty of ties
        else:
            x = rng.standard_normal(shape).astype(dtype)
            x[rng.random(shape) < 0.05] = np.nan
            x[rng.random(shape) < 0.1] = 0.0

        for axis in range(-x.ndim, x.ndim):
            if x.shape[axis] < 2:
                continue
            for candidate in (x, np.swapaxes(x, 0, -1)):  # contiguous and strided
                expected = LIBROSA_LOCALMAX(candidate, axis=axis)
                actual = audio_features_service._localmax(candidate, axis=axis)
                assert actual.dtype == expected.dtype
                assert np.array_equal(actual, expected)

    def test_is_installed_where_librosa_looks_it_up(self):
        assert librosa.util.localmax is audio_features_service._localmax

    def test_feature_extraction_never_reaches_the_numba_kernel(self, tmp_path, monkeypatch):
        def crash(*args, **kwargs):
            raise AssertionError("librosa's numba _localmax kernel was called")

        monkeypatch.setattr(librosa_utils, "_localmax", crash)
        sr = 22050
        t = np.arange(sr * 3) / sr
        y = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        y[:: sr // 2] += 0.8
        path = tmp_path / "tone.wav"
        sf.write(path, y, sr)

        features = audio_features_service.extract_audio_frequency_features(str(path))

        assert features  # chroma_stft, tonnetz and beat_track all ran


class _Request:
    def __init__(self, task_id="task-1", retries=0):
        self.id = task_id
        self.retries = retries


class _Task:
    def __init__(self, **request):
        self.request = _Request(**request)


class TestRedeliveryGuard:
    @pytest.fixture
    def updates(self, monkeypatch):
        recorded = []

        class Jobs:
            def update(self, job_id, **fields):
                recorded.append((job_id, fields))

        monkeypatch.setattr(tasks, "_heartbeat_redis", fakeredis.FakeRedis(decode_responses=True))
        monkeypatch.setattr(tasks, "JobRepository", Jobs)
        monkeypatch.setattr(tasks, "_run", lambda value: value)
        return recorded

    def test_fails_the_job_once_deliveries_pass_the_cap(self, updates):
        task = _Task()
        for _ in range(tasks.MAX_DELIVERIES_PER_ATTEMPT):
            tasks._fail_if_redelivered_too_often(task, "job-1")
        assert updates == []

        with pytest.raises(tasks.RepeatedWorkerCrash):
            tasks._fail_if_redelivered_too_often(task, "job-1")

        [(job_id, fields)] = updates
        assert job_id == "job-1"
        assert fields["status"].value == "failure"
        assert fields["error"].code == "worker_crashed"
        assert fields["error"].retryable is False

    def test_a_celery_retry_starts_a_fresh_count(self, updates):
        for _ in range(tasks.MAX_DELIVERIES_PER_ATTEMPT):
            tasks._fail_if_redelivered_too_often(_Task(retries=0), "job-1")

        tasks._fail_if_redelivered_too_often(_Task(retries=1), "job-1")

        assert updates == []

    def test_an_unreachable_counter_lets_the_job_run(self, updates, monkeypatch):
        class DownRedis:
            def pipeline(self):
                raise RedisConnectionError("job redis is unreachable")

        monkeypatch.setattr(tasks, "_heartbeat_redis", DownRedis())

        for _ in range(tasks.MAX_DELIVERIES_PER_ATTEMPT + 2):
            tasks._fail_if_redelivered_too_often(_Task(), "job-1")

        assert updates == []

    @pytest.mark.parametrize(
        ("task", "args", "worker_fn"),
        [
            (tasks.execute_job, ({"job_id": "job-1"},), "execute"),
            (tasks.execute_job_item, ({"job_id": "job-1"}, 0), "execute_batch_item"),
        ],
    )
    def test_job_tasks_check_before_doing_any_work(self, updates, monkeypatch, task, args, worker_fn):
        def must_not_run(*args, **kwargs):
            raise AssertionError("the crashing work ran again")

        monkeypatch.setattr(tasks, worker_fn, must_not_run)
        tasks._heartbeat_redis.set("task-deliveries:crashy:0", tasks.MAX_DELIVERIES_PER_ATTEMPT)

        result = task.apply(args=args, task_id="crashy")

        assert isinstance(result.result, tasks.RepeatedWorkerCrash)
        assert updates[0][1]["error"].code == "worker_crashed"
