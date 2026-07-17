from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Any

from .settings import settings


class StorageError(RuntimeError):
    pass


def _safe_key(key: str) -> str:
    path = PurePosixPath(key)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise StorageError("Invalid object key")
    return path.as_posix()


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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump(value, handle, allow_nan=False, separators=(",", ":"))
            temp_path = Path(handle.name)
        try:
            self.put_file(key, temp_path, "application/json")
        finally:
            temp_path.unlink(missing_ok=True)

    def get_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key))


class LocalObjectStorage(ObjectStorage):
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        candidate = (self.root / _safe_key(key)).resolve()
        if self.root not in candidate.parents:
            raise StorageError("Object key escapes storage root")
        return candidate

    def put_file(self, key: str, source: Path, content_type: str | None = None) -> None:
        destination = self.path_for(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)

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


class S3ObjectStorage(ObjectStorage):
    def __init__(self) -> None:
        if not settings.S3_BUCKET:
            raise StorageError("S3_BUCKET is required for S3 storage")
        try:
            import boto3
        except ImportError as exc:
            raise StorageError("boto3 is required for S3 storage") from exc
        self.bucket = settings.S3_BUCKET
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT_URL,
            region_name=settings.S3_REGION,
            aws_access_key_id=settings.S3_ACCESS_KEY_ID,
            aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
        )

    def put_file(self, key: str, source: Path, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else None
        self.client.upload_file(str(source), self.bucket, _safe_key(key), ExtraArgs=extra or {})

    def download_file(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, _safe_key(key), str(destination))

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=_safe_key(key))["Body"].read()

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=_safe_key(key))

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=_safe_key(key))
            return True
        except Exception:
            return False


@lru_cache(maxsize=1)
def get_storage() -> ObjectStorage:
    backend = settings.STORAGE_BACKEND.strip().lower()
    if backend == "local":
        return LocalObjectStorage(settings.STORAGE_LOCAL_ROOT)
    if backend == "s3":
        return S3ObjectStorage()
    raise StorageError("STORAGE_BACKEND must be 'local' or 's3'")
