from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
import time
from typing import Any

from app.worker.model_adapters import AudioModelAdapter, get_model_adapter


@dataclass
class RegistryEntry:
    key: tuple[str, str, str, str]
    resource: Any
    last_used: float


class ModelRegistry:
    """Per-worker lazy model registry with bounded LRU/idle eviction."""

    def __init__(self) -> None:
        self.max_entries = int(os.getenv("MODEL_REGISTRY_MAX_ENTRIES", "3"))
        self.idle_seconds = int(os.getenv("MODEL_REGISTRY_IDLE_SECONDS", "1800"))
        self._entries: OrderedDict[tuple[str, str, str, str], RegistryEntry] = OrderedDict()

    def prepare(self, model: str | AudioModelAdapter, purpose: str) -> Any:
        from app.core.device import INFERENCE_DEVICE
        from app.services import model_loader_service as models

        adapter = model if isinstance(model, AudioModelAdapter) else get_model_adapter(model)
        variant = adapter.resource_variant(purpose)
        key = (adapter.model_id, variant, adapter.revision, str(INFERENCE_DEVICE))
        now = time.monotonic()
        self.evict_idle(now)
        if key in self._entries:
            entry = self._entries.pop(key)
            entry.last_used = now
            self._entries[key] = entry
            return entry.resource

        resource = adapter.load_resource(variant)
        self._entries[key] = RegistryEntry(key=key, resource=resource, last_used=now)
        while len(self._entries) > self.max_entries:
            old_key, _ = self._entries.popitem(last=False)
            if old_key[0] in {"whisper-base", "whisper-large", "wav2vec2"}:
                models.unload_model_resources(old_key[0], old_key[1])
        return resource

    def evict_idle(self, now: float | None = None) -> None:
        from app.services import model_loader_service as models

        current = now or time.monotonic()
        stale = [key for key, entry in self._entries.items() if current - entry.last_used > self.idle_seconds]
        for key in stale:
            self._entries.pop(key, None)
            if key[0] in {"whisper-base", "whisper-large", "wav2vec2"}:
                models.unload_model_resources(key[0], key[1])


model_registry = ModelRegistry()
