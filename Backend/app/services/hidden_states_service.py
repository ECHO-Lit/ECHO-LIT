"""Per-layer encoder activation extraction.

Turns one audio file into one pooled vector *per encoder layer*, which is the
input the layer-wise probing feature is built on (see `probing_service`).

The service deliberately knows nothing about Whisper, Wav2Vec2 or Transformers.
It is handed a ``run_encoder`` callable that maps a waveform to a sequence of
per-layer hidden states, and owns everything around that: audio loading, noise
injection at a requested SNR, pooling, sanitising and the payload contract.
Adding support for another architecture is therefore a five-line callable in the
relevant adapter rather than a change here.

Module-level imports stay limited to numpy so that importing this contract from
the pure-computation side costs nothing; librosa and torch are pulled in lazily.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Encoders in this project all consume 16 kHz mono.
DEFAULT_SAMPLE_RATE = 16000

# `hidden_states` from a Transformers encoder includes the embedding output
# before the first block. It is kept as layer 0 and named "input": dropping it
# would hide the "raw features" end of the emergence curve, which is precisely
# the baseline the curve is read against.
INPUT_LAYER_NAME = "input"

RunEncoder = Callable[[np.ndarray], Sequence[Any]]


def layer_names(num_layers: int) -> list[str]:
    """Stable display names for a hidden-states tuple of length `num_layers`."""
    return [INPUT_LAYER_NAME] + [f"layer_{index}" for index in range(1, num_layers)]


def add_noise_at_snr(
    waveform: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add white Gaussian noise scaled to hit a target signal-to-noise ratio.

    `pertubation_service.add_gaussian_noise` takes an absolute noise level, which
    makes the effective SNR depend on how loud the clip happens to be -- unusable
    as a probe target, because "noisy" would then partly mean "quiet". Scaling to
    the measured signal power makes the clean/noisy contrast mean the same thing
    on every file.
    """
    signal_power = float(np.mean(np.square(waveform, dtype=np.float64)))
    if signal_power <= 0:
        # Digital silence has no SNR to hit; leave it alone rather than turning
        # it into pure noise.
        return waveform
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(scale=float(np.sqrt(noise_power)), size=waveform.shape)
    return (waveform + noise).astype(np.float32, copy=False)


def load_waveform(audio_path: str, sample_rate: int) -> np.ndarray:
    """Load mono audio at `sample_rate`, guaranteed finite."""
    import librosa

    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    if not np.all(np.isfinite(waveform)):
        # Mirrors the guard in model_loader_service: a single NaN sample
        # propagates through the encoder and produces an all-NaN activation set
        # that fails much later and much less legibly.
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    return waveform


def _to_numpy(state: Any) -> np.ndarray:
    """Accept a torch tensor or an array-like and return [time, dim] float32."""
    if hasattr(state, "detach"):
        state = state.detach().float().cpu().numpy()
    array = np.asarray(state, dtype=np.float32)
    if array.ndim == 3:
        # [batch, time, dim] -- a single file per call, so batch is 1.
        array = array[0]
    elif array.ndim != 2:
        raise ValueError(f"Unsupported hidden-state shape: {array.shape}")
    return array


def pool_layers(hidden_states: Sequence[Any]) -> tuple[list[list[float]], int]:
    """Mean-pool each layer over time. Returns (layers, hidden_dim)."""
    if not hidden_states:
        raise ValueError("Model returned no hidden states; output_hidden_states was not honoured")

    pooled: list[list[float]] = []
    hidden_dim: int | None = None
    for index, state in enumerate(hidden_states):
        array = _to_numpy(state)
        if hidden_dim is None:
            hidden_dim = int(array.shape[1])
        elif int(array.shape[1]) != hidden_dim:
            raise ValueError(
                f"Layer {index} has width {array.shape[1]}, expected {hidden_dim}"
            )
        vector = array.mean(axis=0)
        if not np.all(np.isfinite(vector)):
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        pooled.append([float(value) for value in vector])
    return pooled, int(hidden_dim or 0)


def extract_layer_activations(
    run_encoder: RunEncoder,
    audio_path: str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    pooling: str = "mean",
    noise_snr_db: float | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Extract one mean-pooled vector per encoder layer for one audio file.

    Args:
        run_encoder: Maps a mono waveform to the model's per-layer hidden states,
            i.e. what `output_hidden_states=True` returns. Length must be
            ``encoder blocks + 1``.
        audio_path: Local path to the audio file.
        sample_rate: Rate the model's processor expects.
        pooling: Only ``"mean"`` is implemented. Frame-level pooling belongs to
            the deferred phonetic-probing phase and is rejected rather than
            silently downgraded.
        noise_snr_db: When set, a second pass over the same clip with white noise
            added at this SNR, returned as ``noisy_layers``. Drives the
            clean-vs-noisy robustness probe.
        seed: Seeds the noise so a re-run of the same file is bit-identical.

    Returns:
        A payload shared verbatim by the ``hidden_states`` and ``layer_probe``
        operations -- they store one item-cache entry between them, so this shape
        must not vary by operation.
    """
    if pooling != "mean":
        raise ValueError(f"Unsupported pooling: {pooling}")

    waveform = load_waveform(audio_path, sample_rate)
    layers, hidden_dim = pool_layers(run_encoder(waveform))

    noisy_layers: list[list[float]] | None = None
    if noise_snr_db is not None:
        noisy_waveform = add_noise_at_snr(
            waveform, float(noise_snr_db), np.random.default_rng(seed)
        )
        noisy_layers, noisy_dim = pool_layers(run_encoder(noisy_waveform))
        if noisy_dim != hidden_dim or len(noisy_layers) != len(layers):
            raise ValueError("Clean and noisy passes disagree on activation shape")

    return {
        "num_layers": len(layers),
        "hidden_dim": hidden_dim,
        "layer_names": layer_names(len(layers)),
        "pooling": pooling,
        "sample_rate": int(sample_rate),
        "noise_snr_db": None if noise_snr_db is None else float(noise_snr_db),
        "layers": layers,
        "noisy_layers": noisy_layers,
    }
