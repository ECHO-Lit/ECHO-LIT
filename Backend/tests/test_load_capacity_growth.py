"""Load Testing -- capacity limits and bounded growth of stored state.

Test Plan Section 3.1.5.  See tests/plans/3.1.5-load-testing.md.

SRS PE-3: "The system shall accept audio up to 100 MB and 10 minutes per file.
... A 24-hour expiry on transient objects shall bound cumulative storage growth."

Load is not only concurrency.  A service that stays fast under a day's traffic
but keeps every byte it was ever sent fails on the second week, and a limit
enforced on one upload path but not another is a capacity limit in name only.
Two concerns here:

* **Per-file capacity** is enforced on the custom-dataset upload path, not just
  on POST /upload.  Before BUG-37 was fixed the dataset path had neither the
  size nor the duration limit.
* **Cumulative growth** is bounded on every tier that holds user data: Redis by
  TTL (asserted under sustained load in LT-03), the object store by the hourly
  mtime sweep, and custom datasets -- which live on disk outside the object
  store -- by the new sweep keyed on session expiry (BUG-43).
"""

from __future__ import annotations

import io
import os
import time

import numpy as np
import pytest
import soundfile as sf

from app.core import redis as redis_module
from app.core.celery_app import celery_app
from app.core.redis import k_meta
from app.core.settings import settings
from app.core.storage import get_storage
from app.services import custom_dataset_service
from app.services.custom_dataset_service import cleanup_expired_session_datasets
from tests._fixtures import upload_files

pytestmark = [pytest.mark.performance, pytest.mark.critical]


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


@pytest.fixture(autouse=True)
def sessions_root(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    monkeypatch.setattr(custom_dataset_service, "SESSIONS_BASE_DIR", root)
    return root


def _wav(seconds: float) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(16_000 * seconds), dtype="float32"), 16_000, format="WAV")
    return buffer.getvalue()


async def _dataset_with_a_file(client, name: str) -> str:
    sid = (await client.get("/session")).json()["sid"]
    await client.post("/upload/dataset/create", data={"dataset_name": name})
    response = await client.post(f"/upload/dataset/{name}/files", files=upload_files(["a.wav"]))
    assert response.status_code == 200, response.text
    return sid


# --------------------------------------------------------------------------
# Per-file capacity on the dataset path
# --------------------------------------------------------------------------


class TestDatasetUploadCapacity:
    async def test_an_oversize_file_is_refused_and_the_rest_of_the_batch_lands(
        self, client, monkeypatch, sessions_root
    ):
        """LT-30: guards BUG-37 -- PE-3's 100 MB limit on the dataset path.

        The limit is lowered to 64 KB for the case so the boundary costs
        kilobytes, not a 100 MB fixture; the route reads the same setting POST
        /upload does, whose deployed value 3.1.2 pinned in FT-07c.
        """
        monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 64 * 1024)
        await client.post("/upload/dataset/create", data={"dataset_name": "corpus"})

        response = await client.post("/upload/dataset/corpus/files", files=[
            ("files", ("small.wav", io.BytesIO(_wav(0.5)), "audio/wav")),   # ~16 KB
            ("files", ("large.wav", io.BytesIO(_wav(5.0)), "audio/wav")),   # ~160 KB
        ])

        assert response.status_code == 207, response.text
        body = response.json()
        assert body["errors"] == ["Failed to upload large.wav: Audio exceeds the 100 MB upload limit"]
        assert [f["filename"] for f in body["uploaded_files"]] == ["small.wav"]
        stored = {p.name for p in sessions_root.rglob("*.wav")}
        assert stored == {"small.wav"}

    async def test_an_overlong_file_is_refused_and_leaves_nothing_on_disk(
        self, client, monkeypatch, sessions_root
    ):
        """LT-31: guards BUG-37 -- PE-3's 10 minute limit on the dataset path."""
        monkeypatch.setattr(settings, "MAX_AUDIO_DURATION_SECONDS", 1.0)
        await client.post("/upload/dataset/create", data={"dataset_name": "corpus"})

        response = await client.post("/upload/dataset/corpus/files", files=[
            ("files", ("short.wav", io.BytesIO(_wav(0.5)), "audio/wav")),
            ("files", ("long.wav", io.BytesIO(_wav(2.0)), "audio/wav")),
        ])

        assert response.status_code == 207, response.text
        assert response.json()["errors"] == ["Failed to upload long.wav: Audio exceeds the 10 minute duration limit"]
        assert response.json()["dataset_metadata"]["total_files"] == 1
        assert {p.name for p in sessions_root.rglob("*.wav")} == {"short.wav"}


# --------------------------------------------------------------------------
# Cumulative growth: custom datasets
# --------------------------------------------------------------------------


class TestSessionDatasetExpiry:
    async def test_datasets_of_expired_sessions_are_removed_and_live_ones_kept(
        self, client, sessions_root
    ):
        """LT-32: guards BUG-43 -- custom datasets outlived their session forever.

        Two researchers upload datasets; one session then expires (its meta key
        leaves Redis on TTL, simulated by deleting it).  The sweep removes
        exactly that session's tree.
        """
        from httpx import AsyncClient

        from app.main import app

        live_sid = await _dataset_with_a_file(client, "mine")
        async with AsyncClient(app=app, base_url="http://test") as other:
            expired_sid = await _dataset_with_a_file(other, "theirs")
        await redis_module.redis.delete(k_meta(expired_sid))

        alive = {sid for sid in (live_sid, expired_sid) if await redis_module.redis.exists(k_meta(sid))}
        removed = cleanup_expired_session_datasets(lambda sid: sid in alive)

        assert removed == 1
        assert not (sessions_root / expired_sid).exists()
        assert (sessions_root / live_sid / "datasets" / "mine" / "a.wav").is_file()

    def test_the_beat_task_asks_the_session_database(self, sessions_root, monkeypatch):
        """LT-33: the scheduled task's wiring -- sessions are on DB0, keyed by k_meta."""
        from app.worker import tasks

        for sid in ("alive", "gone"):
            (sessions_root / sid / "datasets").mkdir(parents=True)
        asked = []

        class SessionDb:
            def exists(self, key):
                asked.append(key)
                return key == k_meta("alive")

        monkeypatch.setattr(tasks, "_session_redis", SessionDb())

        assert tasks.cleanup_expired_session_datasets_task() == 1
        assert sorted(asked) == sorted([k_meta("alive"), k_meta("gone")])
        assert [p.name for p in sessions_root.iterdir()] == ["alive"]

    def test_an_unreachable_session_database_deletes_nothing(self, sessions_root, monkeypatch):
        """LT-34: fail safe -- a Redis outage must never read as "every session expired"."""
        from app.worker import tasks

        (sessions_root / "someone" / "datasets").mkdir(parents=True)

        class DownDb:
            def exists(self, key):
                raise ConnectionError("session redis is unreachable")

        monkeypatch.setattr(tasks, "_session_redis", DownDb())

        with pytest.raises(ConnectionError):
            tasks.cleanup_expired_session_datasets_task()
        assert (sessions_root / "someone").is_dir()

    def test_both_sweeps_are_scheduled_hourly(self):
        """LT-35: the bounds only hold if something runs them."""
        schedule = celery_app.conf.beat_schedule

        for name, task in [
            ("cleanup-expired-local-objects", "app.worker.tasks.cleanup_expired_local_objects"),
            ("cleanup-expired-session-datasets", "app.worker.tasks.cleanup_expired_session_datasets"),
        ]:
            assert schedule[name]["task"] == task
            assert schedule[name]["schedule"] == 60 * 60
            assert schedule[name]["options"] == {"queue": "cpu"}


# --------------------------------------------------------------------------
# Cumulative growth: the object store
# --------------------------------------------------------------------------


class TestObjectStoreExpiry:
    def test_objects_older_than_the_ttl_are_swept_and_fresh_ones_kept(self, tmp_path):
        """LT-36: the pre-existing mtime sweep, which had no test at all."""
        from app.worker.tasks import cleanup_expired_local_objects

        storage = get_storage()
        storage.put_json("results/s1/old/result.json", {"v": 1})
        storage.put_json("results/s1/new/result.json", {"v": 2})
        old = storage.path_for("results/s1/old/result.json")
        stale = time.time() - settings.JOB_TTL_SECONDS - 60
        os.utime(old, (stale, stale))

        assert cleanup_expired_local_objects() == 1
        assert not old.exists()
        assert not old.parent.exists(), "an emptied directory is pruned too"
        assert storage.exists("results/s1/new/result.json")
