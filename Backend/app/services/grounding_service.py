"""FR-10: explanation-fairness / saliency grounding metrics
(docs/FR10plan.md Part 1 S4.4).

Worker-only (numpy + librosa at module level) -- see
app/services/fairness_service.py's docstring for the control-plane split.

Consumes the `series` (frame-level saliency scores) and `total_duration`
fields already produced by app/services/saliency_service.generate_saliency,
so this module never touches the model itself; it only scores an existing
saliency output against the source waveform.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_speech_mask(waveform: np.ndarray, sr: int, n_frames: int, *, frame_ms: float = 25.0) -> np.ndarray:
    """Public wrapper so callers that need the mask for a SECOND purpose
    (e.g. l2_arctic_annotations.phone_error_grounding's auroc_within_speech
    control) can reuse the exact same VAD grounding_metrics uses internally,
    instead of silently diverging with their own threshold."""
    return _speech_mask(waveform, sr, n_frames, frame_ms=frame_ms)


def _speech_mask(waveform: np.ndarray, sr: int, n_frames: int, *, frame_ms: float = 25.0) -> np.ndarray:
    """Energy-based VAD resampled to n_frames booleans. Threshold is relative
    (top of a log-energy histogram split), not absolute, so it is robust to
    per-clip loudness differences across groups."""
    if waveform.size == 0 or n_frames == 0:
        return np.zeros(n_frames, dtype=bool)
    hop = max(1, int(sr * frame_ms / 1000))
    n_native = max(1, waveform.size // hop)
    energies = np.array([
        np.sqrt(np.mean(np.square(waveform[i * hop:(i + 1) * hop])) + 1e-12)
        for i in range(n_native)
    ])
    log_energy = np.log(energies + 1e-9)
    # Otsu-style split on log-energy: threshold at the midpoint between the
    # means of the two halves of a sorted split, iterated once -- cheap and
    # stable enough for a silence/speech binary without a full VAD model.
    threshold = np.median(log_energy)
    for _ in range(3):
        low = log_energy[log_energy <= threshold]
        high = log_energy[log_energy > threshold]
        if len(low) == 0 or len(high) == 0:
            break
        threshold = 0.5 * (low.mean() + high.mean())
    native_mask = log_energy > threshold
    # Resample the native-resolution mask to n_frames by nearest-index lookup.
    idx = np.clip((np.arange(n_frames) * n_native / n_frames).astype(int), 0, n_native - 1)
    return native_mask[idx]


def _gini(p: np.ndarray) -> float:
    sorted_p = np.sort(p)
    n = len(sorted_p)
    if n == 0 or sorted_p.sum() <= 0:
        return 0.0
    cum = np.cumsum(sorted_p)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def grounding_metrics(
    series: list[float], total_duration: float, waveform: np.ndarray | None, sr: int | None,
) -> dict[str, Any]:
    s = np.abs(np.asarray(series, dtype=float))
    if s.sum() <= 0 or len(s) == 0:
        return {"status": "degenerate"}
    p = s / s.sum()
    T = len(s)

    if waveform is not None and sr and waveform.size > 0:
        speech = _speech_mask(waveform, int(sr), T)
    else:
        speech = np.ones(T, dtype=bool)  # no waveform available -> treat as all-speech (neutral)

    duty = float(speech.mean())
    grounding_lift = float(p[speech].sum() / max(duty, 1e-6)) if duty > 0 else None

    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(max(T, 2)))
    gini = _gini(p)
    k = max(1, T // 10)
    top_idx = np.argsort(p)[::-1][:k]
    top10_mass = float(p[top_idx].sum())
    dt = total_duration / T if T else 0.0
    top_times = top_idx.astype(float) * dt
    top10_dispersion_s = float(np.mean(np.abs(top_times - np.median(top_times)))) if len(top_times) else 0.0

    return {
        "status": "ok",
        "grounding_lift": grounding_lift,
        "speech_duty_cycle": duty,
        "attribution_entropy": entropy,
        "attribution_gini": gini,
        "top10_mass": top10_mass,
        "top10_dispersion_s": top10_dispersion_s,
    }


def group_grounding_summary(per_item: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [m for m in per_item if m.get("status") == "ok"]
    if not ok:
        return {"status": "unavailable", "n_explained": len(per_item)}
    return {
        "status": "ok", "n_explained": len(ok),
        "grounding_lift": float(np.mean([m["grounding_lift"] for m in ok if m["grounding_lift"] is not None])),
        "attribution_entropy": float(np.mean([m["attribution_entropy"] for m in ok])),
        "attribution_gini": float(np.mean([m["attribution_gini"] for m in ok])),
        "top10_mass": float(np.mean([m["top10_mass"] for m in ok])),
    }
