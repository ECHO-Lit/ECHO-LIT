from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
import json
import logging
from pathlib import Path, PurePosixPath
import shutil
from typing import Any
import uuid

from .settings import settings


logger = logging.getLogger(__name__)


class StorageError(RuntimeError):
    pass


class StorageUnavailable(StorageError):
    """The backend could not be reached or refused the request.

    Transient from the caller's point of view: the workers retry it (it is a
    `StorageError`) and the API answers 503, as it does for Redis.
    """


def configured_backend() -> str:
    """STORAGE_BACKEND, canonical.

    Settings normalise it at startup; this also covers a value set past
    validation, so `get_storage` and the retention sweep can never disagree.
    """
    return settings.STORAGE_BACKEND.strip().lower()


def _safe_key(key: str) -> str:
    # Backslashes are rejected before parsing: PurePosixPath treats "\" as an
    # ordinary character, so "a\..\..\b" would survive the traversal check below
    # and then mean different things on Windows and POSIX. Object keys are always
    # built with forward slashes, so a backslash is never legitimate here.
    if "\\" in key:
        raise StorageError("Invalid object key")
    path = PurePosixPath(key)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise StorageError("Invalid object key")
    return path.as_posix()


def _without_extended_prefix(path: Path) -> Path:
    r"""Drop Windows' `\\?\` extended-length prefix from a resolved path.

    On Windows, `Path.resolve()` intermittently returns the `\\?\C:\...` form
    for a path while another thread is creating files in the same directory.
    That path has no `C:\...` ancestors, so the containment check below read a
    legitimate key as escaping the root -- and concurrent workers writing the
    shared `cache-items/` directory failed items with StorageError. POSIX paths
    and UNC (`\\?\UNC\...`) paths pass through unchanged.
    """
    text = str(path)
    if text.startswith("\\\\?\\") and not text.startswith("\\\\?\\UNC\\"):
        return Path(text[4:])
    return path


class ObjectStorage(ABC):
    @abstractmethod
    def put_file(self, key: str, source: Path, content_type: str | None = None) -> None: ...

    @abstractmethod
    def download_file(self, key: str, destination: Path) -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    def put_json(self, key: str, value: Any) -> None:
        import tempfile

        # temp_path is bound before anything can raise, so a rejected payload
        # (allow_nan=False turns a NaN into a ValueError) still gets cleaned up
        # instead of leaking the temp file into the system temp directory.
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        temp_path = Path(handle.name)
        try:
            with handle:
                json.dump(value, handle, allow_nan=False, separators=(",", ":"))
            self.put_file(key, temp_path, "application/json")
        finally:
            temp_path.unlink(missing_ok=True)

    def get_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key))


class LocalObjectStorage(ObjectStorage):
    def __init__(self, root: str | Path):
        self.root = _without_extended_prefix(Path(root).resolve())
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        candidate = _without_extended_prefix((self.root / _safe_key(key)).resolve())
        if self.root not in candidate.parents:
            raise StorageError("Object key escapes storage root")
        return candidate

    def put_file(self, key: str, source: Path, content_type: str | None = None) -> None:
        destination = self.path_for(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A temporary name per write, not per key: two writers of one key (the
        # per-file cache, when two users analyse the same clip) must not share
        # a staging file, or the second rename finds the first already moved.
        temporary = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def download_file(self, key: str, destination: Path) -> None:
        source = self.path_for(key)
        if not source.is_file():
            raise StorageError("Object not found")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def get_bytes(self, key: str) -> bytes:
        path = self.path_for(key)
        if not path.is_file():
            raise StorageError("Object not found")
        return path.read_bytes()

    def delete(self, key: str) -> None:
        self.path_for(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self.path_for(key).is_file()


_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}


class S3ObjectStorage(ObjectStorage):
    """The S3 configuration of the store, held to the same contract as the local one.

    Callers are written against `StorageError`: a missing object is one, and
    the workers retry it as transient.  botocore raises its own exception
    types, so every call translates them -- a missing object into
    `StorageError("Object not found")`, anything else (bad credentials, a
    wrong bucket, an unreachable endpoint) into `StorageUnavailable`.
    """

    def __init__(self) -> None:
        if not settings.S3_BUCKET:
            raise StorageError("S3_BUCKET is required for S3 storage")
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as exc:
            raise StorageError("boto3 is required for S3 storage") from exc
        self._client_errors = (BotoCoreError, ClientError)
        self.bucket = settings.S3_BUCKET
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT_URL,
            region_name=settings.S3_REGION,
            aws_access_key_id=settings.S3_ACCESS_KEY_ID,
            aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
        )

    @staticmethod
    def _is_not_found(exc: BaseException) -> bool:
        response = getattr(exc, "response", None) or {}
        return str(response.get("Error", {}).get("Code")) in _NOT_FOUND_CODES

    def _translate(self, exc: BaseException) -> StorageError:
        if self._is_not_found(exc):
            return StorageError("Object not found")
        return StorageUnavailable(f"S3 storage unavailable: {exc}")

    def _call(self, operation: Callable[[], Any]) -> Any:
        try:
            return operation()
        except self._client_errors as exc:
            raise self._translate(exc) from exc

    def put_file(self, key: str, source: Path, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else None
        self._call(lambda: self.client.upload_file(str(source), self.bucket, _safe_key(key), ExtraArgs=extra or {}))

    def download_file(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._call(lambda: self.client.download_file(self.bucket, _safe_key(key), str(destination)))

    def get_bytes(self, key: str) -> bytes:
        return self._call(lambda: self.client.get_object(Bucket=self.bucket, Key=_safe_key(key))["Body"].read())

    def delete(self, key: str) -> None:
        self._call(lambda: self.client.delete_object(Bucket=self.bucket, Key=_safe_key(key)))

    def exists(self, key: str) -> bool:
        # Only "no such object" is an answer; a failure to ask is not.  Until
        # this raised, wrong credentials made /health report storage healthy.
        try:
            self.client.head_object(Bucket=self.bucket, Key=_safe_key(key))
            return True
        except self._client_errors as exc:
            if self._is_not_found(exc):
                return False
            raise self._translate(exc) from exc


def read_cache_entry(
    storage: ObjectStorage, key: str, *, valid: Callable[[Any], bool] | None = None
) -> Any | None:
    """Read a cache object, treating a damaged one as a miss.

    A cache is an optimisation: an entry that no longer parses, or parses to
    the wrong shape, must cost a recomputation rather than the job.  The entry
    is deleted so the recomputation writes a sound one in its place -- every
    reader checks `exists` first, so its pointer then reads as a miss too.
    """
    try:
        value = storage.get_json(key)
    except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError are both ValueErrors
        logger.warning("Discarding unreadable cache entry %s: %s", key, exc)
        storage.delete(key)
        return None
    if valid is not None and not valid(value):
        logger.warning("Discarding malformed cache entry %s", key)
        storage.delete(key)
        return None
    return value


def is_item_entry(value: Any) -> bool:
    """The shape of a per-file cache entry: `{"result": ...}`."""
    return isinstance(value, dict) and "result" in value


@lru_cache(maxsize=1)
def get_storage() -> ObjectStorage:
    backend = configured_backend()
    if backend == "local":
        return LocalObjectStorage(settings.STORAGE_LOCAL_ROOT)
    if backend == "s3":
        return S3ObjectStorage()
    raise StorageError("STORAGE_BACKEND must be 'local' or 's3'")
