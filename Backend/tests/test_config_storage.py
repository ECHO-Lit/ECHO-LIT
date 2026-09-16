"""Configuration -- the object store under its local and S3 configurations.

Test Plan Section 3.1.8.  See tests/plans/3.1.8-configuration-testing.md.

Module CE.  `STORAGE_BACKEND` selects a shared directory (development) or an
S3-compatible bucket (production, SAD section 7).  Every caller is written
against one contract -- a missing object is a `StorageError`, a transient
failure is retried, `exists` answers the question it is asked -- so the same
behaviour is required of both configurations.

boto3 is not installed in the test environment (OBS-43), and no case may
reach a network: the S3 configuration runs against a stand-in `boto3` /
`botocore` placed in `sys.modules`, raising the exception types botocore
raises.

SRS 3.5: "A health endpoint shall report the status of Redis, storage and
worker heartbeats to support monitoring."
SRS RE-2: "Failed tasks shall be retryable."
"""
from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import pytest

from app.core import storage as storage_module
from app.core.settings import settings
from app.core.storage import LocalObjectStorage, StorageError, get_storage
from tests._fixtures import wav_bytes
from tests._faults import tolerant_client

pytestmark = [pytest.mark.configuration, pytest.mark.critical]


# -- stand-in botocore / boto3 ---------------------------------------------

class BotoCoreError(Exception):
    pass


class EndpointConnectionError(BotoCoreError):
    def __init__(self, endpoint_url: str):
        super().__init__(f"Could not connect to the endpoint URL: {endpoint_url!r}")


class ClientError(Exception):
    def __init__(self, code: str, operation: str):
        self.response = {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {}}
        super().__init__(f"An error occurred ({code}) when calling the {operation} operation")


class FakeS3Client:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.objects: dict[tuple[str, str], bytes] = {}
        self.failure: Exception | None = None

    def _call(self, operation, bucket, key):
        if self.failure is not None:
            raise self.failure
        if (bucket, key) not in self.objects:
            raise ClientError("404" if operation == "HeadObject" else "NoSuchKey", operation)
        return self.objects[(bucket, key)]

    def head_object(self, Bucket, Key):
        self._call("HeadObject", Bucket, Key)
        return {}

    def get_object(self, Bucket, Key):
        data = self._call("GetObject", Bucket, Key)
        return {"Body": types.SimpleNamespace(read=lambda: data)}

    def download_file(self, bucket, key, filename):
        if self.failure is None and (bucket, key) not in self.objects:
            raise ClientError("404", "HeadObject")
        Path(filename).write_bytes(self._call("GetObject", bucket, key))

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        if self.failure is not None:
            raise self.failure
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def delete_object(self, Bucket, Key):
        if self.failure is not None:
            raise self.failure
        self.objects.pop((Bucket, Key), None)


@pytest.fixture
def fake_boto3(monkeypatch):
    clients: list[FakeS3Client] = []
    boto3 = types.ModuleType("boto3")

    def client(service, **kwargs):
        assert service == "s3"
        clients.append(FakeS3Client(**kwargs))
        return clients[-1]

    boto3.client = client
    botocore = types.ModuleType("botocore")
    exceptions = types.ModuleType("botocore.exceptions")
    exceptions.BotoCoreError = BotoCoreError
    exceptions.ClientError = ClientError
    exceptions.EndpointConnectionError = EndpointConnectionError
    botocore.exceptions = exceptions
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", exceptions)
    return clients


@pytest.fixture(autouse=True)
def _fresh_storage(tmp_path, monkeypatch):
    """Each case builds its own store; none leaks the S3 one into later modules."""
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


def _use_s3(monkeypatch, **overrides):
    values = {
        "STORAGE_BACKEND": "s3",
        "S3_BUCKET": "echo-test",
        "S3_ENDPOINT_URL": "https://s3.example.test",
        "S3_REGION": "eu-west-1",
        "S3_ACCESS_KEY_ID": "AKIDTEST",
        "S3_SECRET_ACCESS_KEY": "secret",
    }
    values.update(overrides)
    for key, value in values.items():
        monkeypatch.setattr(settings, key, value)
    get_storage.cache_clear()


def _store(backend, monkeypatch, fake_boto3):
    if backend == "s3":
        _use_s3(monkeypatch)
    return get_storage()


def _unavailable():
    # Read through getattr so the pre-fix run fails on an assertion.
    return getattr(storage_module, "StorageUnavailable", None)


class TestBackendSelection:
    @pytest.mark.parametrize("backend", ["local", " LOCAL ", "s3", "gcs"])
    def test_the_configured_backend_is_constructed(self, monkeypatch, fake_boto3, backend):
        """CF-85: STORAGE_BACKEND selects the store; an unknown one is refused."""
        if backend == "s3":
            _use_s3(monkeypatch)
            assert isinstance(get_storage(), storage_module.S3ObjectStorage)
            return
        monkeypatch.setattr(settings, "STORAGE_BACKEND", backend)
        get_storage.cache_clear()
        if backend == "gcs":
            with pytest.raises(StorageError, match="STORAGE_BACKEND"):
                get_storage()
        else:
            assert isinstance(get_storage(), LocalObjectStorage)

    @pytest.mark.parametrize("gap", ["no-bucket", "no-boto3"])
    def test_an_incomplete_s3_configuration_fails_with_a_storage_error(self, monkeypatch, fake_boto3, gap):
        """CF-86: a half-configured S3 deployment says what is missing."""
        if gap == "no-bucket":
            _use_s3(monkeypatch, S3_BUCKET=None)
            with pytest.raises(StorageError, match="S3_BUCKET"):
                get_storage()
        else:
            _use_s3(monkeypatch)
            monkeypatch.setitem(sys.modules, "boto3", None)
            with pytest.raises(StorageError, match="boto3"):
                get_storage()

    def test_the_s3_client_is_built_from_settings(self, monkeypatch, fake_boto3):
        """CF-87: the endpoint, region and credentials come from the environment (SE-2)."""
        _use_s3(monkeypatch)
        get_storage()
        assert fake_boto3[-1].kwargs == {
            "endpoint_url": "https://s3.example.test",
            "region_name": "eu-west-1",
            "aws_access_key_id": "AKIDTEST",
            "aws_secret_access_key": "secret",
        }


class TestStorageContract:
    @pytest.mark.parametrize("backend", ["local", "s3"])
    def test_a_missing_object_does_not_exist(self, monkeypatch, fake_boto3, backend):
        """CF-88: `exists` answers False for an object that is not there."""
        assert _store(backend, monkeypatch, fake_boto3).exists("uploads/nobody/none.wav") is False

    @pytest.mark.parametrize("operation", ["get_bytes", "download_file"])
    @pytest.mark.parametrize("backend", ["local", "s3"])
    def test_reading_a_missing_object_is_a_storage_error(self, monkeypatch, tmp_path, fake_boto3, backend, operation):
        """CF-89: guards BUG-67 (the s3 instances).

        The local store raised `StorageError("Object not found")`; the S3
        store let botocore's `ClientError` escape, which no caller catches.
        """
        store = _store(backend, monkeypatch, fake_boto3)
        with pytest.raises(StorageError, match="not found"):
            if operation == "get_bytes":
                store.get_bytes("uploads/nobody/none.wav")
            else:
                store.download_file("uploads/nobody/none.wav", tmp_path / "out.wav")

    @pytest.mark.parametrize(
        "failure",
        [ClientError("403", "HeadObject"), EndpointConnectionError("https://s3.example.test")],
        ids=["forbidden", "unreachable"],
    )
    def test_an_unreachable_or_forbidden_bucket_is_a_storage_failure(self, monkeypatch, fake_boto3, failure):
        """CF-90: guards BUG-67.

        `exists` caught every exception and returned False: wrong credentials,
        a wrong bucket and an unreachable endpoint all read as "no such object".
        """
        _use_s3(monkeypatch)
        store = get_storage()
        fake_boto3[-1].failure = failure
        unavailable = _unavailable()
        assert unavailable is not None and issubclass(unavailable, StorageError)
        with pytest.raises(unavailable):
            store.exists("healthcheck")

    @pytest.mark.parametrize(
        "failure",
        [ClientError("403", "HeadObject"), EndpointConnectionError("https://s3.example.test")],
        ids=["forbidden", "unreachable"],
    )
    async def test_health_reports_s3_down(self, monkeypatch, client, fake_boto3, failure):
        """CF-91: guards BUG-67 -- SRS 3.5's storage status, under the S3 configuration.

        With `exists` swallowing errors, `/health` reported `storage: true`
        for a bucket the deployment could not reach.
        """
        _use_s3(monkeypatch)
        get_storage()
        fake_boto3[-1].failure = failure
        response = await client.get("/health")
        assert response.status_code == 503
        body = response.json()
        assert body["storage"] is False
        assert "storage" in body["detail"]

    def test_an_s3_outage_is_transient_for_workers(self, monkeypatch, fake_boto3):
        """CF-92: guards BUG-67 -- RE-2's retry, under the S3 configuration.

        The workers retry `StorageError` as transient (`tasks.TRANSIENT`); a
        raw botocore error failed the job outright instead.
        """
        from app.worker import tasks

        _use_s3(monkeypatch)
        store = get_storage()
        fake_boto3[-1].failure = ClientError("503", "GetObject")
        with pytest.raises(Exception) as raised:
            store.get_bytes("results/some/key.json")
        assert isinstance(raised.value, tasks.TRANSIENT)

    async def test_an_upload_during_an_s3_outage_is_a_503(self, monkeypatch, fake_boto3):
        """CF-93: guards BUG-67 -- US-4, a readable refusal with a retry hint.

        The botocore error escaped the route's handler as a bare 500.
        """
        _use_s3(monkeypatch)
        get_storage()
        fake_boto3[-1].failure = EndpointConnectionError("https://s3.example.test")
        async with tolerant_client() as client:
            response = await client.post("/upload", files={"file": ("tone.wav", wav_bytes(), "audio/wav")})
        assert response.status_code == 503
        assert response.headers.get("retry-after")


class TestLocalRetention:
    def _age(self, path: Path, seconds: float) -> None:
        stamp = time.time() - seconds
        os.utime(path, (stamp, stamp))

    def test_cleanup_runs_for_a_non_canonical_backend_value(self, monkeypatch, tmp_path):
        """CF-94: guards BUG-71.

        `get_storage` stripped STORAGE_BACKEND but the hourly sweep did not,
        so " Local " stored objects and never removed them: PE-3's growth
        bound silently off.
        """
        from app.worker.tasks import cleanup_expired_local_objects

        root = tmp_path / "objects"
        old = root / "uploads" / "s" / "old.wav"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"x")
        self._age(old, settings.JOB_TTL_SECONDS + 60)
        monkeypatch.setattr(settings, "STORAGE_BACKEND", " Local ")

        assert cleanup_expired_local_objects() == 1
        assert not old.exists()

    def test_cleanup_removes_exactly_the_objects_older_than_the_job_ttl(self, tmp_path):
        """CF-95: PE-3, SRS 3.10 -- the local half of the 24-hour expiry."""
        from app.worker.tasks import cleanup_expired_local_objects

        root = tmp_path / "objects"
        (root / "a").mkdir(parents=True)
        old, fresh = root / "a" / "old.json", root / "a" / "fresh.json"
        old.write_text("{}")
        fresh.write_text("{}")
        self._age(old, settings.JOB_TTL_SECONDS + 60)
        self._age(fresh, settings.JOB_TTL_SECONDS - 3600)

        assert cleanup_expired_local_objects() == 1
        assert not old.exists() and fresh.exists()

    def test_cleanup_leaves_s3_retention_to_the_lifecycle_policy(self, monkeypatch, tmp_path):
        """CF-96: SRS 3.10 -- under S3 the bucket's lifecycle rule expires objects."""
        from app.worker.tasks import cleanup_expired_local_objects

        old = tmp_path / "objects" / "old.json"
        old.parent.mkdir(parents=True, exist_ok=True)
        old.write_text("{}")
        self._age(old, settings.JOB_TTL_SECONDS + 60)
        monkeypatch.setattr(settings, "STORAGE_BACKEND", "s3")

        assert cleanup_expired_local_objects() == 0
        assert old.exists()
