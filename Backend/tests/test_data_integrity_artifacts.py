"""Data and Database Integrity Testing -- object storage, caches, dataset metadata.

Test Plan Section 3.1.1.  See tests/plans/3.1.1-data-and-database-integrity-testing.md.

The companion module `test_data_integrity_store.py` covers the Redis half of the
persistence tier.  This one covers everything that lands on a filesystem: the
object-storage abstraction, the content-addressed analysis cache (whose pointer
lives in Redis but whose payload lives in the store), and the custom-dataset
metadata tree.

Every case here is isolated by the two autouse fixtures below.  Both defaults
are *relative* paths -- STORAGE_LOCAL_ROOT is "shared-storage" and
SESSIONS_BASE_DIR is "uploads/sessions" -- so a missed fixture writes into the
developer's real working tree.  `get_storage` is additionally lru_cached, which
turns a missed cache_clear() into an order-dependent failure that only shows up
under some -k selections.  Hence: autouse, module-wide, no exceptions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.core import redis as redis_module
from app.core import storage as storage_module
from app.core.settings import settings
from app.core.storage import LocalObjectStorage, StorageError, _safe_key, get_storage
from app.repositories.jobs import JobRepository
from app.schemas.jobs import JobOperation, JobRecord, JobStatus, TaskEnvelope
from app.services import custom_dataset_service
from app.services.custom_dataset_service import CustomDatasetManager
from app.worker import executor
from app.worker.cache_policy import item_cache_identity
from app.worker.executor import (
    _cached_item_result,
    analysis_cache_key,
    complete_batch_from_cache,
    item_cache_key,
)

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


@pytest.fixture(autouse=True)
def isolated_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr(custom_dataset_service, "SESSIONS_BASE_DIR", tmp_path / "sessions")
    yield


def envelope_dict(**overrides) -> dict:
    base = {
        "job_id": "j1",
        "session_id": "s1",
        "operation": "prediction",
        "model": "whisper-base",
        "audio": [
            {
                "audio_id": "a1",
                "object_key": "uploads/s1/a1.wav",
                "filename": "a1.wav",
                "media_type": "audio/wav",
                "sha256": "aaa",
            }
        ],
        "parameters": {},
        "result_schema_version": "v1",
        "code_version": "test",
    }
    base.update(overrides)
    return base


def audio_entry(sha256: str, audio_id: str = "a1") -> dict:
    return {
        "audio_id": audio_id,
        "object_key": f"uploads/s1/{audio_id}.wav",
        "filename": f"{audio_id}.wav",
        "media_type": "audio/wav",
        "sha256": sha256,
    }


def job_record(**overrides) -> JobRecord:
    fields = {
        "job_id": "j1",
        "session_id": "s1",
        "operation": JobOperation.prediction,
        "model": "whisper-base",
        "audio_ids": ["a1"],
        "created_at": T0,
        "updated_at": T0,
    }
    fields.update(overrides)
    return JobRecord(**fields)


# --------------------------------------------------------------------------
# F. Object storage
# --------------------------------------------------------------------------


@pytest.mark.critical
@pytest.mark.security
class TestObjectKeySafety:
    @pytest.mark.parametrize(
        "key",
        [
            "../secrets.json",
            "cache/../../etc/passwd",
            "/etc/passwd",
            "",
            "a\\..\\..\\b",
            "C:\\Windows\\system32",
        ],
        ids=["parent", "nested_parent", "absolute", "empty", "backslash", "windows_absolute"],
    )
    def test_safe_key_rejects_traversal_absolute_and_empty(self, key):
        """DI-35: keys that could escape the root are refused at the door.

        The backslash cases matter for portability: PurePosixPath treats "\\" as
        an ordinary character, so without an explicit check these would be
        accepted here and then mean different things on Windows and POSIX.
        """
        with pytest.raises(StorageError, match="Invalid object key"):
            _safe_key(key)

    def test_safe_key_accepts_and_normalises_legitimate_keys(self):
        """DI-35: ordinary nested keys pass through; `.` segments collapse."""
        assert _safe_key("cache/abc/result.json") == "cache/abc/result.json"
        assert _safe_key("cache/./abc.json") == "cache/abc.json"

    def test_path_for_confines_keys_to_the_storage_root(self, tmp_path):
        """DI-36: resolved paths stay under the configured root."""
        storage = LocalObjectStorage(tmp_path / "root")

        resolved = storage.path_for("cache/a/b.json")

        assert storage.root in resolved.parents
        assert resolved.name == "b.json"
        # The "escapes storage root" branch is defence-in-depth behind
        # _safe_key, which already rejects every portable way of reaching it.


@pytest.mark.critical
class TestAtomicWrites:
    def test_put_file_leaves_no_tmp_artifact(self, tmp_path):
        """DI-37: a successful write publishes one file and cleans up after itself."""
        source = tmp_path / "src.json"
        source.write_bytes(b'{"ok":true}')
        storage = get_storage()

        storage.put_file("cache/d/result.json", source)

        destination = storage.path_for("cache/d/result.json")
        assert destination.read_bytes() == b'{"ok":true}'
        assert list(destination.parent.glob("*.tmp")) == []
        assert len(list(destination.parent.iterdir())) == 1

    def test_failed_copy_never_publishes_a_partial_destination(self, tmp_path, monkeypatch):
        """DI-38: readers never observe a truncated object."""
        source = tmp_path / "src.json"
        source.write_bytes(b'{"ok":true}')
        storage = get_storage()

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(storage_module.shutil, "copyfile", boom)

        with pytest.raises(OSError):
            storage.put_file("cache/d/result.json", source)

        # A leftover .tmp is permitted -- that is the designed partial-write
        # signal.  What must never exist is the destination key itself.
        assert storage.exists("cache/d/result.json") is False


@pytest.mark.critical
class TestJsonObjectWrites:
    @pytest.mark.parametrize(
        "bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
    )
    def test_put_json_rejects_non_finite_floats_without_writing(self, bad):
        """DI-39: a NaN in a result fails the job rather than caching garbage."""
        storage = get_storage()

        # Non-finite floats are a live failure mode here, not a hypothetical:
        # a zero-variance probe or an empty audio window produces them.
        with pytest.raises(ValueError):
            storage.put_json("cache/d/result.json", {"items": [{"score": bad}]})

        assert storage.exists("cache/d/result.json") is False

    def test_put_json_does_not_leak_a_temp_file_when_the_payload_is_rejected(self, monkeypatch):
        """DI-39b: the rejected-payload path cleans up after itself."""
        import tempfile

        scratch = tempfile.mkdtemp()
        monkeypatch.setattr(tempfile, "tempdir", scratch)
        before = set(Path(scratch).iterdir())

        with pytest.raises(ValueError):
            get_storage().put_json("cache/d/result.json", {"score": float("nan")})

        # The temp file is created before the payload is serialised, so the
        # cleanup has to survive the serialisation itself raising.
        assert set(Path(scratch).iterdir()) == before

    def test_reads_raise_for_a_missing_key(self, tmp_path):
        """DI-40: a missing object raises -- it never reads back as empty bytes."""
        storage = get_storage()

        assert storage.exists("cache/missing.json") is False
        with pytest.raises(StorageError, match="Object not found"):
            storage.get_bytes("cache/missing.json")
        with pytest.raises(StorageError, match="Object not found"):
            storage.get_json("cache/missing.json")
        with pytest.raises(StorageError, match="Object not found"):
            storage.download_file("cache/missing.json", tmp_path / "out.json")


# --------------------------------------------------------------------------
# G. Cache pointer/blob consistency and key material
# --------------------------------------------------------------------------


@pytest.mark.important
class TestBatchCachePointer:
    """The batch cache is split: pointer in Redis (DB0), payload in the store.

    Pointers are always seeded and read through `redis_module.redis` -- the same
    module attribute production code uses -- rather than a locally captured
    client, so these stay correct if conftest's DB collapse is ever fixed.
    """

    async def test_returns_false_when_the_blob_is_missing(self):
        """DI-41: a dangling pointer never produces a false cache hit."""
        data = envelope_dict()
        digest = analysis_cache_key(TaskEnvelope.model_validate(data))
        await JobRepository().create(job_record())
        # Pointer only -- no payload was ever written to the store.
        await redis_module.redis.set(f"analysis-cache:{digest}", "cache/deadbeef/result.json")

        assert await complete_batch_from_cache(data) is False

        record = await JobRepository().get("j1")
        assert record.status == JobStatus.queued
        assert record.cache_hit is False
        assert record.result_key is None
        assert get_storage().exists("results/s1/j1/result.json") is False

    async def test_missing_pointer_returns_false_even_when_a_blob_exists(self):
        """DI-43: the pointer is the authority, not the conventional blob path."""
        data = envelope_dict()
        digest = analysis_cache_key(TaskEnvelope.model_validate(data))
        await JobRepository().create(job_record())
        get_storage().put_json(f"cache/{digest}/result.json", {"items": []})

        assert await complete_batch_from_cache(data) is False

    async def test_pointer_and_blob_together_complete_the_job(self):
        """DI-42: a real hit rewrites the payload for the *new* job."""
        data = envelope_dict()
        digest = analysis_cache_key(TaskEnvelope.model_validate(data))
        await JobRepository().create(job_record(job_id="j2"))
        cache_key = f"cache/{digest}/result.json"
        # Stored as the cold run that produced it recorded itself: nothing cached.
        get_storage().put_json(cache_key, {
            "job_id": "j1",
            "items": [{"result": 1, "cache_hit": False}, {"result": 2, "cache_hit": False}],
            "cache_info": {"cached_count": 0, "missing_count": 2, "cache_hit_rate": 0},
        })
        await redis_module.redis.set(f"analysis-cache:{digest}", cache_key)

        assert await complete_batch_from_cache({**data, "job_id": "j2"}) is True

        payload = get_storage().get_json("results/s1/j2/result.json")
        # The rewrite is the whole point: a cached payload carries the producing
        # job's id and must be restamped for the consumer.
        assert payload["job_id"] == "j2"
        assert payload["metadata"]["cache_hit"] is True
        # Its counts too: replaying the cold run's 0/2 told the user nothing
        # came from cache when everything did.
        assert payload["cache_info"] == {"cached_count": 2, "missing_count": 0, "cache_hit_rate": 1.0}
        assert all(item["cache_hit"] for item in payload["items"])

        record = await JobRepository().get("j2")
        assert record.status == JobStatus.success
        assert record.cache_hit is True
        assert record.result_key == "results/s1/j2/result.json"

    async def test_item_cache_read_ignores_a_pointer_whose_blob_is_gone(self):
        """DI-48: `_cached_item_result` honours its exists() guard."""
        envelope = TaskEnvelope.model_validate(envelope_dict())
        digest = item_cache_key(envelope, "aaa")
        await redis_module.redis.set(f"analysis-item-cache:{digest}", "cache-items/gone.json")

        # Recompute, not raise.
        assert await _cached_item_result(envelope, "aaa", get_storage()) is None

    async def test_perturbation_results_are_never_item_cached(self):
        """DI-48: perturbation payloads embed session-owned ids, so caching is off."""
        envelope = TaskEnvelope.model_validate(
            envelope_dict(operation="perturbation", parameters={"kind": "noise", "snr_db": 10})
        )
        digest = item_cache_key(envelope, "aaa")
        get_storage().put_json("cache-items/present.json", {"result": "leaked"})
        await redis_module.redis.set(f"analysis-item-cache:{digest}", "cache-items/present.json")

        # Even with a valid pointer *and* a valid blob, the read is skipped.
        assert await _cached_item_result(envelope, "aaa", get_storage()) is None


@pytest.mark.important
class TestCacheKeyMaterial:
    def test_analysis_cache_key_is_audio_order_sensitive(self):
        """DI-44: only sha256 is material, and its ordering matters."""
        two = envelope_dict(audio=[audio_entry("aaa", "a1"), audio_entry("bbb", "a2")])
        swapped = envelope_dict(audio=[audio_entry("bbb", "a2"), audio_entry("aaa", "a1")])

        key = lambda d: analysis_cache_key(TaskEnvelope.model_validate(d))  # noqa: E731

        # Items are positionally aligned downstream, so a reordering is a
        # different result, not the same one.
        assert key(two) != key(swapped)
        # Content identity is the sha256 alone: renaming a file must reuse cache.
        renamed = envelope_dict(
            audio=[{**audio_entry("aaa"), "audio_id": "zzz", "filename": "other.wav",
                    "object_key": "uploads/s1/other.wav"}]
        )
        assert key(envelope_dict()) == key(renamed)
        # ...but different bytes must not.
        assert key(envelope_dict()) != key(envelope_dict(audio=[audio_entry("ccc")]))

    def test_analysis_cache_key_ignores_session_and_job_identity(self):
        """DI-45: cache entries are shared across sessions by design."""
        key = lambda d: analysis_cache_key(TaskEnvelope.model_validate(d))  # noqa: E731

        assert key(envelope_dict()) == key(
            envelope_dict(job_id="other-job", session_id="other-session")
        )
        # This sharing is *why* the repository ownership checks in
        # test_data_integrity_store.py::TestOwnership are load-bearing: the
        # cache cannot be relied on to keep sessions apart.

    def test_analysis_cache_key_is_sensitive_to_model_revision_schema_and_code(self, monkeypatch):
        """DI-46: a stale cache must not survive a model or schema upgrade."""
        key = lambda d: analysis_cache_key(TaskEnvelope.model_validate(d))  # noqa: E731
        baseline = key(envelope_dict())

        assert baseline != key(envelope_dict(model="wav2vec2"))
        assert baseline != key(envelope_dict(result_schema_version="v2"))
        assert baseline != key(envelope_dict(code_version="other"))

        # The revision is not carried on the envelope -- it is looked up from the
        # catalogue at key time, so a pinned-revision bump invalidates silently.
        monkeypatch.setitem(executor.MODEL_REVISIONS, "whisper-base", "rev-99")
        assert baseline != key(envelope_dict())

    def test_item_cache_key_tracks_only_extraction_parameters(self):
        """DI-47: per-file keys move with the params that fed the model."""
        def key(**params):
            data = envelope_dict(operation="hidden_states", parameters=params)
            return item_cache_key(TaskEnvelope.model_validate(data), "aaa")

        baseline = key(pooling="mean")
        assert baseline != key(pooling="max")
        assert baseline != key(pooling="mean", noise_snr_db=10)
        assert baseline != key(pooling="mean", seed=7)

        # Operations outside the shared extraction identity pass through
        # untouched.  (The hidden_states <-> layer_probe collapse itself is
        # already covered by test_layer_probe_job.py / test_layer_probe_schema.py.)
        assert item_cache_identity("prediction", {"timestamps": True}) == (
            "prediction",
            {"timestamps": True},
        )


# --------------------------------------------------------------------------
# H. Custom-dataset metadata corruption
# --------------------------------------------------------------------------


@pytest.mark.important
class TestDatasetMetadataCorruption:
    """A torn mid-`json.dump` write cannot be produced without killing the
    process, so every case here pre-truncates the file instead.  That is the
    correct proxy: it reproduces exactly the on-disk state a crash would leave.
    """

    @pytest.fixture
    def corrupt_dataset(self):
        manager = CustomDatasetManager("s1")
        manager.create_dataset("speech")
        metadata = manager.datasets_dir / "speech" / "dataset_metadata.json"
        metadata.write_text('{"dataset_name": "spe', encoding="utf-8")
        return manager

    def test_read_paths_degrade_but_do_not_raise(self, corrupt_dataset):
        """DI-49: a corrupt dataset disappears from the UI rather than erroring."""
        assert corrupt_dataset.get_dataset_metadata("speech") is None
        assert corrupt_dataset.list_datasets() == []
        # The audio is orphaned, not deleted -- it still occupies disk.
        assert (corrupt_dataset.datasets_dir / "speech").is_dir()

    def test_write_paths_raise_value_error(self, corrupt_dataset):
        """DI-50: corruption reaches the routes as the error they translate.

        The upload routes turn ValueError into a 4xx.  A raw JSONDecodeError
        would escape as an unhandled 500 instead.
        """
        with pytest.raises(ValueError, match="unreadable metadata"):
            corrupt_dataset.add_file_to_dataset("speech", "clip.wav", b"RIFFdata")
        with pytest.raises(ValueError, match="unreadable metadata"):
            corrupt_dataset.add_manifest_to_dataset("speech", "m.csv", b"path,text\na.wav,hi\n")

        # Metadata is loaded before any bytes are written, so a failed add must
        # not leave an untracked orphan file behind.
        assert not (corrupt_dataset.datasets_dir / "speech" / "clip.wav").exists()

    def test_corrupt_labels_read_as_absent_and_can_be_cleared(self):
        """DI-51: a user can recover a corrupt label table without shell access."""
        manager = CustomDatasetManager("s1")
        manager.create_dataset("speech")
        manager.set_labels("speech", {"a.wav": {"accent": "us"}}, source="csv")
        labels_path = manager._labels_path("speech")
        labels_path.write_text('{"table": {"a.wav"', encoding="utf-8")

        # The probe sees "unlabelled", never a half-parsed answer key.
        assert manager.get_labels("speech") is None
        assert manager.clear_labels("speech") is True
        assert not labels_path.exists()
        assert manager.clear_labels("speech") is False

    def test_set_labels_replaces_rather_than_merging(self):
        """DI-52: derived fields match the table actually written."""
        manager = CustomDatasetManager("s1")
        manager.create_dataset("speech")

        manager.set_labels(
            "speech",
            {
                "a.wav": {"accent": "us", "gender": "f"},
                "b.wav": {"accent": "uk", "gender": "m"},
                "c.wav": {"accent": "in", "gender": "f"},
            },
            source="csv",
        )
        manager.set_labels("speech", {"z.wav": {"accent": "au"}}, source="manual")

        stored = json.loads(manager._labels_path("speech").read_text(encoding="utf-8"))
        # A silent union of two answer keys would be impossible to reason about.
        assert stored["table"] == {"z.wav": {"accent": "au"}}
        assert stored["columns"] == ["accent"]
        assert stored["n_labelled"] == 1
