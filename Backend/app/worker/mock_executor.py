"""Deterministic, model-free execution used by the local mock worker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import wave
from typing import Any


def _random_for(
    operation: str,
    model: str | None,
    audio_path: Path,
    parameters: dict[str, Any],
) -> random.Random:
    digest = hashlib.sha256()
    digest.update(audio_path.read_bytes())
    digest.update(operation.encode())
    digest.update((model or "").encode())
    digest.update(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode())
    return random.Random(int.from_bytes(digest.digest()[:8], "big"))


def _wav_duration_ms(path: Path) -> tuple[int, int]:
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            duration_ms = round(handle.getnframes() / max(1, sample_rate) * 1000)
            return duration_ms, sample_rate
    except (wave.Error, OSError):
        return 1000, 16000


def execute_one(
    operation: str,
    model: str | None,
    audio_path: Path,
    parameters: dict[str, Any],
) -> Any:
    rng = _random_for(operation, model, audio_path, parameters)
    if operation == "prediction":
        if model == "wav2vec2":
            labels = ["neutral", "happy", "sad", "angry", "fear"]
            selected = labels[rng.randrange(len(labels))]
            raw = {label: rng.uniform(0.05, 0.35) for label in labels}
            raw[selected] += 0.8
            total = sum(raw.values())
            probabilities = {label: round(value / total, 6) for label, value in raw.items()}
            return {
                "predicted_emotion": selected,
                "probabilities": probabilities,
                "confidence": probabilities[selected],
            }
        return "This is a deterministic local mock transcription."

    if operation == "saliency":
        words = ["This", "is", "a", "local", "mock", "explanation"]
        values = [round(rng.uniform(0.15, 0.95), 6) for _ in words]
        segments = [
            {
                "start_time": index * 0.5,
                "end_time": (index + 1) * 0.5,
                "saliency": value,
                "intensity": value,
                "word": word,
            }
            for index, (word, value) in enumerate(zip(words, values))
        ]
        return {
            "model": model,
            "method": parameters.get("method", "gradcam"),
            "segments": segments,
            "total_duration": len(words) * 0.5,
            "series": [round(rng.uniform(0.05, 1.0), 6) for _ in range(96)],
        }

    if operation == "attention":
        words = ["this", "is", "local", "mock", "attention"]
        pairs = []
        for source, from_word in enumerate(words):
            for target, to_word in enumerate(words):
                pairs.append({
                    "from_word": from_word,
                    "to_word": to_word,
                    "from_time": [source * 0.5, (source + 1) * 0.5],
                    "to_time": [target * 0.5, (target + 1) * 0.5],
                    "attention_weight": round(
                        0.75 if source == target else rng.uniform(0.02, 0.45), 6
                    ),
                    "from_index": source,
                    "to_index": target,
                })
        return {
            "model": model,
            "layer": int(parameters.get("layer_idx", 6)),
            "head": int(parameters.get("head_idx", 0)),
            "sequence_length": len(words),
            "word_chunks": [
                {"text": word, "timestamp": [index * 0.5, (index + 1) * 0.5]}
                for index, word in enumerate(words)
            ],
            "attention_pairs": pairs,
            "timestamp_attention": [
                {"time": index * 0.1, "attention": round(rng.uniform(0.1, 0.95), 6)}
                for index in range(26)
            ],
            "total_duration": 2.5,
        }

    if operation == "embedding":
        return [round(rng.uniform(-1.0, 1.0), 7) for _ in range(32)]

    if operation == "audio_features":
        return {
            "duration": round(rng.uniform(1.0, 8.0), 4),
            "tempo": round(rng.uniform(80.0, 150.0), 4),
            "rms_energy": round(rng.uniform(0.02, 0.4), 6),
            "zero_crossing_rate": round(rng.uniform(0.01, 0.25), 6),
            "spectral_centroid": round(rng.uniform(700.0, 3600.0), 4),
            "spectral_bandwidth": round(rng.uniform(500.0, 2800.0), 4),
        }

    raise ValueError(f"Unsupported mock operation/model combination: {operation}/{model}")


def create_perturbation(
    input_path: Path,
    temp_root: Path,
    perturbations: list[dict[str, Any]],
) -> dict[str, Any]:
    output = temp_root / "mock-perturbed.wav"
    try:
        with wave.open(str(input_path), "rb"):
            pass
        shutil.copyfile(input_path, output)
    except (wave.Error, OSError):
        subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-y", "-i", str(input_path),
                "-acodec", "pcm_s16le", str(output),
            ],
            check=True,
            capture_output=True,
        )
    duration_ms, sample_rate = _wav_duration_ms(output)
    return {
        "success": True,
        "perturbed_file": str(output),
        "filename": f"mock-perturbed-{input_path.stem}.wav",
        "duration_ms": duration_ms,
        "sample_rate": sample_rate,
        "applied_perturbations": [
            {"type": spec["type"], "status": "applied", "params": spec.get("params", {})}
            for spec in perturbations
        ],
    }


def reduce_dimensions(embeddings: list[list[float]], n_components: int) -> list[list[float]]:
    if not embeddings:
        return []
    width = min(n_components, len(embeddings[0]))
    means = [sum(row[index] for row in embeddings) / len(embeddings) for index in range(width)]
    return [
        [round(row[index] - means[index], 7) for index in range(width)]
        for row in embeddings
    ]
