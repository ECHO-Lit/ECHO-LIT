from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import time
from typing import Any
import uuid

from app.core import redis as redis_module
from app.core.model_catalog import MODEL_REVISIONS
from app.core.settings import settings
from app.core.storage import ObjectStorage, get_storage
from app.core.storage import StorageError
from redis.exceptions import RedisError
from app.repositories.audio import AudioRepository
from app.repositories.jobs import JobRepository
from app.schemas.jobs import AudioAsset, JobError, JobProgress, JobStatus, TaskEnvelope
from app.worker.cache_policy import item_cache_identity


logger = logging.getLogger(__name__)


class JobCancelled(Exception):
    pass


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _attach_embedding_analysis(
    result: dict[str, Any], vectors: list[Any], parameters: dict[str, Any]
) -> None:
    """Add the 2D/3D projection and (optionally) HDBSCAN clustering to an embedding result.

    Shared by the single-task and batched paths so both stay in step.
    """
    if len(vectors) < 2:
        return
    from app.services.model_loader_service import reduce_dimensions

    result["projection"] = _jsonable(
        reduce_dimensions(
            vectors,
            method=parameters.get("reduction", "pca"),
            n_components=int(parameters.get("n_components", 2)),
        )
    )
    if not parameters.get("cluster"):
        return
    # Clustering is additive: never fail an otherwise-successful embedding job over it.
    try:
        from app.services.clustering_service import cluster_embeddings

        result["clustering"] = _jsonable(
            cluster_embeddings(
                vectors,
                min_cluster_size=int(parameters.get("min_cluster_size", 5)),
            )
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
        logger.exception("clustering_failed job_id=%s", result.get("job_id"))
        result["clustering"] = {"error": str(exc)}


# Synthetic property added when a noise pass was requested: does the encoder
# represent *that the signal is degraded*, and at what depth?
NOISE_PROPERTY = "noise"


def _probe_matrix(activations: list[dict[str, Any]], key: str) -> list[list[list[float]]]:
    """Collect one [layer][dim] block per file, rejecting inconsistent shapes."""
    matrix: list[list[list[float]]] = []
    expected: tuple[int, int] | None = None
    for index, payload in enumerate(activations):
        layers = (payload or {}).get(key)
        if not layers:
            raise ValueError(f"file {index} has no '{key}' activations")
        shape = (len(layers), len(layers[0]))
        if expected is None:
            expected = shape
        elif shape != expected:
            raise ValueError(
                f"file {index} has shape {shape}, expected {expected} -- "
                "all files in a probe must come from the same model"
            )
        matrix.append(layers)
    return matrix


def _attach_probe_analysis(
    result: dict[str, Any], activations: list[dict[str, Any]], parameters: dict[str, Any]
) -> None:
    """Train per-layer probes over the batch, then drop the raw activations.

    `activations` holds one `hidden_states` payload per file, in the same order
    as `result["items"]` -- which is the same order as the request's `audio_ids`,
    and therefore the order the labels in `parameters["properties"]` are aligned
    to.

    Shared by the single-task and batched paths so both stay in step.
    """
    from app.services.probing_service import train_layer_probes

    properties = {
        name: list(labels) for name, labels in (parameters.get("properties") or {}).items()
    }
    n_files = len(activations)
    for name, labels in properties.items():
        # The API validates this against `audio_ids`; re-checking against the
        # activations the worker actually assembled is what catches a reordered
        # or short batch, which would otherwise silently mislabel every row.
        if len(labels) != n_files:
            raise ValueError(f"property '{name}' has {len(labels)} labels for {n_files} files")

    clean = _probe_matrix(activations, "layers")
    layer_names = (activations[0] or {}).get("layer_names")
    options = {
        "probe": parameters.get("probe", "logreg"),
        "cv_folds": int(parameters.get("cv_folds", 5)),
        "project_dims": int(parameters.get("project_dims", 256)),
        "min_class_count": int(parameters.get("min_class_count", 5)),
        "include_control": bool(parameters.get("include_control", True)),
        "seed": int(parameters.get("seed", 42)),
        "layer_names": layer_names,
    }
    probes = train_layer_probes(clean, properties, **options)

    if parameters.get("noise_snr_db") is not None:
        noisy = _probe_matrix(activations, "noisy_layers")
        # The noise probe is trained on the clean and noisy passes stacked, but
        # the *other* properties are not: stacking puts a near-duplicate of every
        # clip in the batch, so a speaker or emotion probe would meet the same
        # utterance in both its training and its test fold and report inflated
        # accuracy. Only the clean/noisy contrast is legitimate over that set.
        stacked = clean + noisy
        noise_labels = ["clean"] * n_files + ["noisy"] * n_files
        noise_probe = train_layer_probes(stacked, {NOISE_PROPERTY: noise_labels}, **options)
        probes["properties"][NOISE_PROPERTY] = noise_probe["properties"][NOISE_PROPERTY]
        probes["noise_snr_db"] = float(parameters["noise_snr_db"])

    result["probes"] = _jsonable(probes)
    # Raw activations are what make this job expensive to move: 144 whisper-large
    # files is ~50 MB of JSON. They are already content-addressed per file, so the
    # batch response keeps only their shape.
    result["items"] = [
        {
            "audio_id": item["audio_id"],
            "layers": probes["num_layers"],
            "dim": probes["hidden_dim"],
            **({"cache_hit": item["cache_hit"]} if "cache_hit" in item else {}),
        }
        for item in result["items"]
    ]


def analysis_cache_key(envelope: TaskEnvelope) -> str:
    material = {
        "operation": envelope.operation.value,
        "model": envelope.model,
        "revision": MODEL_REVISIONS.get(envelope.model or ""),
        "audio": [asset.sha256 for asset in envelope.audio],
        "parameters": envelope.parameters,
        "schema": envelope.result_schema_version,
        "code": envelope.code_version,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def item_cache_key(envelope: TaskEnvelope, sha256: str) -> str:
    # Per-file entries are keyed on what actually produced them, not on the whole
    # job configuration -- see `cache_policy` for why that distinction matters.
    operation, parameters = item_cache_identity(envelope.operation.value, envelope.parameters)
    material = {
        "operation": operation,
        "model": envelope.model,
        "revision": MODEL_REVISIONS.get(envelope.model or ""),
        "audio": sha256,
        "parameters": parameters,
        "schema": envelope.result_schema_version,
        "code": envelope.code_version,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


async def _cached_item_result(envelope: TaskEnvelope, sha256: str, storage: ObjectStorage) -> Any | None:
    """Per-file result cache so duplicate or re-submitted jobs skip model execution."""
    if envelope.operation.value == "perturbation":
        return None
    pointer = f"analysis-item-cache:{item_cache_key(envelope, sha256)}"
    cached_key = await redis_module.redis.get(pointer)
    if cached_key and storage.exists(cached_key):
        return storage.get_json(cached_key)["result"]
    return None


async def _store_item_result(envelope: TaskEnvelope, sha256: str, storage: ObjectStorage, output: Any) -> None:
    if envelope.operation.value == "perturbation":
        return
    digest = item_cache_key(envelope, sha256)
    result_key = f"cache-items/{digest}.json"
    storage.put_json(result_key, {"result": output})
    await redis_module.redis.set(
        f"analysis-item-cache:{digest}", result_key, ex=settings.JOB_TTL_SECONDS
    )


def _aggregate_batch(result: dict[str, Any], filenames: list[str]) -> None:
    items = result["items"]
    if result["operation"] == "prediction" and result["model"] == "wav2vec2":
        predictions = []
        counts: dict[str, int] = {}
        for filename, item in zip(filenames, items):
            value = item["result"]
            emotion = value["predicted_emotion"]
            counts[emotion] = counts.get(emotion, 0) + 1
            predictions.append({"filename": filename, **value})
        total = len(predictions)
        dominant, dominant_count = max(counts.items(), key=lambda entry: entry[1])
        result.update({
            "emotion_distribution": {key: count / total for key, count in counts.items()},
            "emotion_counts": counts,
            "individual_predictions": predictions,
            "summary": {
                "total_files": total,
                "dominant_emotion": dominant,
                "dominant_count": dominant_count,
                "dominant_percentage": dominant_count / total,
            },
        })
    elif result["operation"] == "prediction" and (result["model"] or "").startswith("whisper"):
        from collections import Counter
        import re

        transcripts = []
        words: list[str] = []
        for filename, item in zip(filenames, items):
            value = item["result"]
            transcript = value if isinstance(value, str) else value.get("text") or value.get("transcript") or ""
            tokens = re.findall(r"[a-zA-Z']{3,}", transcript.lower())
            words.extend(tokens)
            transcripts.append({"filename": filename, "transcript": transcript, "word_count": len(tokens)})
        counts = Counter(words)
        total_words = len(words)
        result.update({
            "common_terms": [
                {"term": term, "count": count, "percentage": count / total_words * 100 if total_words else 0}
                for term, count in counts.most_common(10)
            ],
            "individual_transcripts": transcripts,
            "summary": {
                "total_files": len(transcripts),
                "total_words": total_words,
                "unique_words": len(counts),
                "avg_words_per_file": total_words / len(transcripts) if transcripts else 0,
            },
        })
    elif result["operation"] == "audio_features":
        import numpy as np

        analyses = [
            {"filename": filename, "features": item["result"]}
            for filename, item in zip(filenames, items)
        ]
        numeric_keys = sorted({
            key for analysis in analyses for key, value in analysis["features"].items()
            if isinstance(value, (int, float))
        })
        statistics = {}
        distributions = {}
        for key in numeric_keys:
            values = np.array([
                analysis["features"][key] for analysis in analyses if key in analysis["features"]
            ], dtype=float)
            statistics[key] = {
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "min": float(np.min(values)), "max": float(np.max(values)),
                "median": float(np.median(values)),
                "q1": float(np.percentile(values, 25)), "q3": float(np.percentile(values, 75)),
            }
            hist, edges = np.histogram(values, bins=min(20, max(1, len(values))))
            distributions[key] = {"histogram": hist.tolist(), "bins": edges.tolist()}
        ranked = sorted(statistics.items(), key=lambda entry: abs(entry[1]["mean"]), reverse=True)
        result.update({
            "model_context": result.get("model") or "audio_features",
            "individual_analyses": analyses,
            "aggregate_statistics": statistics,
            "feature_distributions": distributions,
            "most_common_features": [
                {"feature": key, "normalized_mean": value["mean"], "stability_score": 1 / (1 + value["std"]),
                 "prevalence_score": 1.0, "mean": value["mean"], "std": value["std"]}
                for key, value in ranked
            ],
            "feature_categories": {"Audio features": numeric_keys},
            "summary": {
                "total_files": len(analyses), "total_features_extracted": len(numeric_keys),
                "avg_duration": statistics.get("duration", {}).get("mean", 0),
                "avg_tempo": statistics.get("tempo", {}).get("mean", 0),
            },
        })
    if len(items) > 1:
        cached_count = sum(1 for item in items if item.get("cache_hit"))
        result["cache_info"] = {
            "cached_count": cached_count,
            "missing_count": len(items) - cached_count,
            "cache_hit_rate": cached_count / len(items) if items else 0,
        }


async def _check_cancel(job_id: str, repository: JobRepository) -> None:
    if await repository.cancellation_requested(job_id):
        raise JobCancelled()


def _execute_one(
    operation: str,
    model: str | None,
    audio_path: Path,
    parameters: dict[str, Any],
    model_spec=None,
) -> Any:
    if operation == "audio_features":
        # Import the librosa-only module directly, NOT via model_loader_service --
        # that would pull torch/transformers/umap into the forked CPU worker and
        # reintroduce the OpenMP/LLVM clash that segfaults it.
        from app.services.audio_features_service import extract_audio_frequency_features

        return extract_audio_frequency_features(str(audio_path))
    if not model:
        raise ValueError(f"{operation} requires a model")
    from app.worker.model_adapters import get_model_adapter
    from app.worker.model_registry import model_registry

    adapter = get_model_adapter(model, model_spec)
    if not adapter.supports(operation):
        raise ValueError(f"{model} does not support {operation}")
    resource = model_registry.prepare(adapter, operation)
    return adapter.execute(operation, str(audio_path), parameters, resource)


def _load_torch_artifact(storage: ObjectStorage, key: str, destination: Path) -> dict[str, Any]:
    """Load an artifact ECHO created itself; user-controlled pickle data is never read."""
    import torch

    storage.download_file(key, destination)
    try:
        return torch.load(destination, map_location="cpu", weights_only=True)
    except TypeError:  # Torch before weights_only support.
        return torch.load(destination, map_location="cpu")


async def _execute_jacobian_lens_fit(
    envelope: TaskEnvelope, storage: ObjectStorage, temp_root: Path
) -> dict[str, Any]:
    import torch

    from app.repositories.jacobian_lenses import JacobianLensRepository
    from app.schemas.jacobian_lens import JacobianLensStatus
    from app.services.jacobian_lens_service import fit_encoder_jacobian_lens
    from app.worker.model_adapters import get_model_adapter
    from app.worker.model_registry import model_registry

    lens_id = envelope.parameters["lens_id"]
    repository = JacobianLensRepository()
    record = await repository.get_owned(lens_id, envelope.session_id)
    if not record:
        raise ValueError("Jacobian lens record no longer exists")
    transcripts = {sample["audio_id"]: sample["transcript"] for sample in envelope.parameters["samples"]}
    samples = []
    for asset in envelope.audio:
        transcript = transcripts.get(asset.audio_id)
        if not transcript:
            raise ValueError(f"Missing transcript for lens sample {asset.audio_id}")
        local_path = temp_root / f"{asset.audio_id}{Path(asset.filename).suffix}"
        storage.download_file(asset.object_key, local_path)
        samples.append((str(local_path), transcript))
    adapter = get_model_adapter(envelope.model or "", envelope.model_spec)
    resource = model_registry.prepare(adapter, "jacobian_lens_fit")
    job_repository = JobRepository()
    event_loop = asyncio.get_running_loop()

    def on_sample(completed: int, total: int) -> None:
        """Persist sample progress from the CPU-bound fitting thread."""
        update = job_repository.update(
            envelope.job_id,
            progress=JobProgress(
                current=completed,
                total=total,
                message=f"Fitting encoder Jacobian lenses ({completed}/{total})",
            ),
        )
        asyncio.run_coroutine_threadsafe(update, event_loop).result()

    # Fitting is CPU-bound. Keeping it in a worker thread lets the event loop
    # publish progress after each fitted or held-out sample.
    artifact = await asyncio.to_thread(
        fit_encoder_jacobian_lens,
        adapter,
        resource,
        samples,
        probe_count=int(envelope.parameters["probe_count"]),
        max_audio_seconds=float(envelope.parameters["max_audio_seconds"]),
        on_sample=on_sample,
    )
    artifact_key = f"jacobian-lenses/{envelope.session_id}/{lens_id}/lens.pt"
    metadata_key = f"jacobian-lenses/{envelope.session_id}/{lens_id}/metadata.json"
    artifact_path = temp_root / "lens.pt"
    torch.save(artifact, artifact_path)
    storage.put_file(artifact_key, artifact_path, "application/octet-stream")
    metadata = {
        key: value
        for key, value in artifact.items()
        if key not in {"matrices", "baselines"}
    }
    metadata.update({"lens_id": lens_id, "layer_count": len(artifact["matrices"])})
    storage.put_json(metadata_key, metadata)
    record.status = JacobianLensStatus.READY
    record.architecture = artifact["architecture"]
    record.artifact_key = artifact_key
    record.metadata_key = metadata_key
    record.format_version = int(artifact["format_version"])
    record.layer_count = len(artifact["matrices"])
    record.error = None
    await repository.save(record)
    return metadata


async def _execute_jacobian_lens_apply(
    envelope: TaskEnvelope, storage: ObjectStorage, audio_path: Path, temp_root: Path
) -> dict[str, Any]:
    from app.repositories.jacobian_lenses import JacobianLensRepository
    from app.schemas.jacobian_lens import JacobianLensStatus
    from app.services.jacobian_lens_service import apply_encoder_jacobian_lens
    from app.worker.model_adapters import get_model_adapter
    from app.worker.model_registry import model_registry

    lens_id = envelope.parameters["lens_id"]
    record = await JacobianLensRepository().get_owned(lens_id, envelope.session_id)
    if not record or record.status != JacobianLensStatus.READY or not record.artifact_key:
        raise ValueError("Jacobian lens is not available")
    artifact = _load_torch_artifact(storage, record.artifact_key, temp_root / "lens.pt")
    adapter = get_model_adapter(envelope.model or "", envelope.model_spec)
    resource = model_registry.prepare(adapter, "jacobian_lens_apply")
    output = apply_encoder_jacobian_lens(
        adapter, resource, artifact, str(audio_path),
        top_k=int(envelope.parameters["top_k"]),
        max_frames=int(envelope.parameters["max_frames"]),
    )
    return {"lens_id": lens_id, **output}


async def _execute_perturbation(
    envelope: TaskEnvelope,
    input_path: Path,
    storage: ObjectStorage,
    temp_root: Path,
) -> dict[str, Any]:
    from app.services.pertubation_service import perturb_and_save

    result = perturb_and_save(
        file_path=str(input_path),
        perturbations=envelope.parameters["perturbations"],
        output_dir=str(temp_root),
    )
    if not result.get("success"):
        raise ValueError(result.get("error") or "Perturbation failed")
    output = Path(result["perturbed_file"])
    audio_id = uuid.uuid4().hex
    suffix = output.suffix.lower() or ".wav"
    object_key = f"generated/{envelope.session_id}/{audio_id}{suffix}"
    storage.put_file(object_key, output, "audio/wav")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    asset = AudioAsset(
        audio_id=audio_id,
        session_id=envelope.session_id,
        object_key=object_key,
        filename=result.get("filename", output.name),
        media_type="audio/wav",
        size_bytes=output.stat().st_size,
        duration_seconds=float(result.get("duration_ms", 0)) / 1000,
        sample_rate=int(result.get("sample_rate", 0)) or None,
        channels=None,
        sha256=digest,
        created_at=datetime.now(timezone.utc),
    )
    await AudioRepository().create(asset)
    return {
        "audio_id": asset.audio_id,
        "filename": asset.filename,
        "playback_url": f"/audio/{asset.audio_id}",
        "duration_ms": result.get("duration_ms"),
        "sample_rate": result.get("sample_rate"),
        "applied_perturbations": result.get("applied_perturbations", []),
        "success": True,
    }


async def execute(envelope_data: dict[str, Any], celery_task_id: str) -> None:
    started_at = time.monotonic()
    envelope = TaskEnvelope.model_validate(envelope_data)
    jobs = JobRepository()
    initial_record = await jobs.get(envelope.job_id)
    queue_latency = (
        max(0.0, (datetime.now(timezone.utc) - initial_record.created_at).total_seconds())
        if initial_record else 0.0
    )
    storage = get_storage()
    cache_digest = analysis_cache_key(envelope)
    cache_pointer_key = f"analysis-cache:{cache_digest}"
    # Generated audio is session-owned. Reusing a cached perturbation payload
    # would leak an audio_id owned by the session that originally created it.
    cacheable = envelope.operation.value not in {"perturbation", "jacobian_lens_fit"}
    await jobs.update(
        envelope.job_id,
        status=JobStatus.started,
        task_id=celery_task_id,
        progress=JobProgress(current=0, total=len(envelope.audio), message="Worker started"),
    )
    logger.info(
        "job_started job_id=%s operation=%s model=%s queue_latency_seconds=%.3f",
        envelope.job_id, envelope.operation.value, envelope.model, queue_latency,
    )
    try:
        await _check_cancel(envelope.job_id, jobs)
        if envelope.operation.value == "jacobian_lens_fit":
            await jobs.update(
                envelope.job_id,
                status=JobStatus.processing,
                progress=JobProgress(
                    current=0,
                    total=len(envelope.audio),
                    message="Fitting encoder Jacobian lenses",
                ),
            )
            with tempfile.TemporaryDirectory(prefix=f"echo-{envelope.job_id}-") as temp_dir:
                output = await _execute_jacobian_lens_fit(envelope, storage, Path(temp_dir))
            result = {
                "job_id": envelope.job_id,
                "operation": envelope.operation.value,
                "model": envelope.model,
                "items": [],
                "lens": output,
                "metadata": {"parameters": envelope.parameters, "cache_hit": False},
            }
            job_result_key = f"results/{envelope.session_id}/{envelope.job_id}/result.json"
            storage.put_json(job_result_key, result)
            await jobs.update(
                envelope.job_id,
                status=JobStatus.success,
                progress=JobProgress(
                    current=len(envelope.audio),
                    total=len(envelope.audio),
                    message="Jacobian lens fitted",
                ),
                result_key=job_result_key,
            )
            await redis_module.job_redis.hincrby("metrics:jobs", "success", 1)
            return
        cached_key = await redis_module.redis.get(cache_pointer_key) if cacheable else None
        if cached_key and storage.exists(cached_key):
            job_result_key = f"results/{envelope.session_id}/{envelope.job_id}/result.json"
            cached_payload = storage.get_json(cached_key)
            cached_payload["job_id"] = envelope.job_id
            cached_payload.setdefault("metadata", {})["cache_hit"] = True
            cached_payload["metadata"]["queue_latency_seconds"] = queue_latency
            cached_payload["metadata"]["execution_seconds"] = time.monotonic() - started_at
            storage.put_json(job_result_key, cached_payload)
            await jobs.update(
                envelope.job_id,
                status=JobStatus.success,
                progress=JobProgress(
                    current=len(envelope.audio), total=len(envelope.audio), message="Completed from cache"
                ),
                result_key=job_result_key,
                cache_hit=True,
            )
            await redis_module.job_redis.hincrby("metrics:jobs", "cache_hits", 1)
            await redis_module.job_redis.hincrby("metrics:jobs", "success", 1)
            return

        items: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix=f"echo-{envelope.job_id}-") as temp_dir:
            temp_root = Path(temp_dir)
            for index, asset in enumerate(envelope.audio, start=1):
                await _check_cancel(envelope.job_id, jobs)
                await jobs.update(
                    envelope.job_id,
                    status=JobStatus.processing,
                    progress=JobProgress(
                        current=index - 1,
                        total=len(envelope.audio),
                        message=f"Processing {asset.filename}",
                    ),
                )
                if envelope.operation.value == "perturbation":
                    local_path = temp_root / f"{asset.audio_id}{Path(asset.filename).suffix}"
                    storage.download_file(asset.object_key, local_path)
                    output = await _execute_perturbation(envelope, local_path, storage, temp_root)
                    item_cache_hit = False
                else:
                    output = await _cached_item_result(envelope, asset.sha256, storage)
                    item_cache_hit = output is not None
                    if output is None:
                        local_path = temp_root / f"{asset.audio_id}{Path(asset.filename).suffix}"
                        storage.download_file(asset.object_key, local_path)
                        if envelope.operation.value == "jacobian_lens_apply":
                            output = await _execute_jacobian_lens_apply(
                                envelope, storage, local_path, temp_root
                            )
                        else:
                            output = _jsonable(_execute_one(
                                envelope.operation.value, envelope.model, local_path,
                                envelope.parameters, envelope.model_spec,
                            ))
                        await _store_item_result(envelope, asset.sha256, storage, output)
                items.append({"audio_id": asset.audio_id, "result": _jsonable(output), "cache_hit": item_cache_hit})
                await jobs.update(
                    envelope.job_id,
                    status=JobStatus.processing,
                    progress=JobProgress(
                        current=index,
                        total=len(envelope.audio),
                        message=f"Processed {index} of {len(envelope.audio)}",
                    ),
                )

            result: dict[str, Any] = {
                "job_id": envelope.job_id,
                "operation": envelope.operation.value,
                "model": envelope.model,
                "items": items,
                "metadata": {
                    "model_revision": MODEL_REVISIONS.get(envelope.model or ""),
                    "parameters": envelope.parameters,
                    "result_schema_version": envelope.result_schema_version,
                    "code_version": envelope.code_version,
                    "cache_key": cache_digest,
                    "cache_hit": False,
                    "queue_latency_seconds": queue_latency,
                    "execution_seconds": time.monotonic() - started_at,
                },
            }
            try:
                from app.core.device import accelerator_memory_allocated_mb

                result["metadata"]["accelerator_memory_mb"] = accelerator_memory_allocated_mb()
            except Exception:
                result["metadata"]["accelerator_memory_mb"] = None
            _aggregate_batch(result, [asset.filename for asset in envelope.audio])
            if envelope.operation.value == "embedding":
                _attach_embedding_analysis(
                    result, [item["result"] for item in items], envelope.parameters
                )
            if envelope.operation.value == "layer_probe":
                _attach_probe_analysis(
                    result, [item["result"] for item in items], envelope.parameters
                )

            if cacheable:
                result_key = f"cache/{cache_digest}/result.json"
                storage.put_json(result_key, _jsonable(result))
                await redis_module.redis.set(
                    cache_pointer_key, result_key, ex=settings.JOB_TTL_SECONDS
                )
            job_result_key = f"results/{envelope.session_id}/{envelope.job_id}/result.json"
            storage.put_json(job_result_key, _jsonable(result))
            await jobs.update(
                envelope.job_id,
                status=JobStatus.success,
                progress=JobProgress(
                    current=len(envelope.audio), total=len(envelope.audio), message="Completed"
                ),
                result_key=job_result_key,
                cache_hit=False,
            )
            await redis_module.job_redis.hincrby("metrics:jobs", "success", 1)
            logger.info(
                "job_completed job_id=%s execution_seconds=%.3f",
                envelope.job_id, time.monotonic() - started_at,
            )
    except JobCancelled:
        await jobs.update(
            envelope.job_id,
            status=JobStatus.cancelled,
            progress=JobProgress(current=0, total=len(envelope.audio), message="Cancelled"),
        )
        await redis_module.job_redis.hincrby("metrics:jobs", "cancelled", 1)
    except (StorageError, RedisError):
        await jobs.update(
            envelope.job_id,
            status=JobStatus.queued,
            progress=JobProgress(current=0, total=len(envelope.audio), message="Retrying transient failure"),
            error=JobError(code="transient_failure", message="A transient dependency failed", retryable=True),
        )
        raise
    except Exception as exc:
        if envelope.operation.value == "jacobian_lens_fit":
            from app.repositories.jacobian_lenses import JacobianLensRepository
            from app.schemas.jacobian_lens import JacobianLensStatus

            repository = JacobianLensRepository()
            lens = await repository.get_owned(envelope.parameters.get("lens_id", ""), envelope.session_id)
            if lens:
                lens.status = JacobianLensStatus.FAILED
                lens.error = str(exc)[:500]
                await repository.save(lens)
        await jobs.update(
            envelope.job_id,
            status=JobStatus.failure,
            error=JobError(code="execution_failed", message=str(exc)[:500], retryable=False),
        )
        await redis_module.job_redis.hincrby("metrics:jobs", "failure", 1)
        logger.exception("job_failed job_id=%s", envelope.job_id)
        raise


async def execute_batch_item(
    envelope_data: dict[str, Any], asset_index: int, celery_task_id: str
) -> dict[str, Any]:
    envelope = TaskEnvelope.model_validate(envelope_data)
    jobs = JobRepository()
    asset = envelope.audio[asset_index]
    try:
        await _check_cancel(envelope.job_id, jobs)
        await jobs.update(
            envelope.job_id,
            status=JobStatus.processing,
            progress=JobProgress(
                current=0,
                total=len(envelope.audio),
                message=f"Processing {asset.filename}",
            ),
        )
        storage = get_storage()
        output = await _cached_item_result(envelope, asset.sha256, storage)
        if output is None:
            with tempfile.TemporaryDirectory(prefix=f"echo-{envelope.job_id}-{asset_index}-") as temp_dir:
                local_path = Path(temp_dir) / f"{asset.audio_id}{Path(asset.filename).suffix}"
                storage.download_file(asset.object_key, local_path)
                output = _jsonable(_execute_one(
                    envelope.operation.value, envelope.model, local_path, envelope.parameters, envelope.model_spec
                ))
            await _store_item_result(envelope, asset.sha256, storage, output)
        completed_key = f"job:{envelope.job_id}:completed-items"
        client = redis_module.job_redis
        pipe = client.pipeline()
        pipe.sadd(completed_key, asset.audio_id)
        pipe.expire(completed_key, settings.JOB_TTL_SECONDS)
        results = await pipe.execute()
        completed = await client.scard(completed_key)
        await jobs.update(
            envelope.job_id,
            status=JobStatus.processing,
            progress=JobProgress(
                current=completed,
                total=len(envelope.audio),
                message=f"Processed {completed} of {len(envelope.audio)}",
            ),
        )
        return {
            "index": asset_index,
            "audio_id": asset.audio_id,
            "filename": asset.filename,
            "result": _jsonable(output),
            "task_id": celery_task_id,
        }
    except JobCancelled:
        await jobs.update(envelope.job_id, status=JobStatus.cancelled)
        raise
    except (StorageError, RedisError):
        await jobs.update(
            envelope.job_id,
            status=JobStatus.processing,
            error=JobError(code="transient_failure", message="Retrying batch item", retryable=True),
        )
        raise
    except Exception as exc:
        await jobs.update(
            envelope.job_id,
            status=JobStatus.failure,
            error=JobError(code="batch_item_failed", message=str(exc)[:500], retryable=False),
        )
        raise


async def finalize_batch(
    child_results: list[dict[str, Any]], envelope_data: dict[str, Any]
) -> None:
    started_at = time.monotonic()
    envelope = TaskEnvelope.model_validate(envelope_data)
    jobs = JobRepository()
    try:
        await _check_cancel(envelope.job_id, jobs)
    except JobCancelled:
        await jobs.update(envelope.job_id, status=JobStatus.cancelled)
        return
    ordered = sorted(child_results, key=lambda item: item["index"])
    result: dict[str, Any] = {
        "job_id": envelope.job_id,
        "operation": envelope.operation.value,
        "model": envelope.model,
        "items": [
            {"audio_id": item["audio_id"], "result": item["result"]}
            for item in ordered
        ],
        "metadata": {
            "model_revision": MODEL_REVISIONS.get(envelope.model or ""),
            "parameters": envelope.parameters,
            "result_schema_version": envelope.result_schema_version,
            "code_version": envelope.code_version,
            "cache_key": analysis_cache_key(envelope),
            "cache_hit": False,
            "execution_seconds": time.monotonic() - started_at,
        },
    }
    _aggregate_batch(result, [item["filename"] for item in ordered])
    if envelope.operation.value == "embedding":
        _attach_embedding_analysis(
            result, [item["result"] for item in ordered], envelope.parameters
        )
    if envelope.operation.value == "layer_probe":
        _attach_probe_analysis(
            result, [item["result"] for item in ordered], envelope.parameters
        )
    storage = get_storage()
    cache_digest = analysis_cache_key(envelope)
    cache_result_key = f"cache/{cache_digest}/result.json"
    job_result_key = f"results/{envelope.session_id}/{envelope.job_id}/result.json"
    storage.put_json(cache_result_key, result)
    storage.put_json(job_result_key, result)
    await redis_module.redis.set(
        f"analysis-cache:{cache_digest}", cache_result_key, ex=settings.JOB_TTL_SECONDS
    )
    await jobs.update(
        envelope.job_id,
        status=JobStatus.success,
        progress=JobProgress(
            current=len(envelope.audio), total=len(envelope.audio), message="Completed"
        ),
        result_key=job_result_key,
    )
    await redis_module.job_redis.delete(f"job:{envelope.job_id}:completed-items")
    await redis_module.job_redis.hincrby("metrics:jobs", "success", 1)


async def complete_batch_from_cache(envelope_data: dict[str, Any]) -> bool:
    envelope = TaskEnvelope.model_validate(envelope_data)
    cache_digest = analysis_cache_key(envelope)
    cached_key = await redis_module.redis.get(f"analysis-cache:{cache_digest}")
    storage = get_storage()
    if not cached_key or not storage.exists(cached_key):
        return False
    payload = storage.get_json(cached_key)
    payload["job_id"] = envelope.job_id
    payload.setdefault("metadata", {})["cache_hit"] = True
    job_result_key = f"results/{envelope.session_id}/{envelope.job_id}/result.json"
    storage.put_json(job_result_key, payload)
    await JobRepository().update(
        envelope.job_id,
        status=JobStatus.success,
        progress=JobProgress(
            current=len(envelope.audio), total=len(envelope.audio), message="Completed from cache"
        ),
        result_key=job_result_key,
        cache_hit=True,
    )
    await redis_module.job_redis.hincrby("metrics:jobs", "cache_hits", 1)
    await redis_module.job_redis.hincrby("metrics:jobs", "success", 1)
    return True
