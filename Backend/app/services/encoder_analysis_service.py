"""Encoder-layer structural analysis: attention distance profiles + layer CKA.

Two label-free views of what the Whisper encoder does with a clip:

* Distance profiles -- for every encoder layer and head, how attention mass is
  distributed over frame offsets. A diagonal band means local acoustic
  processing; a long tail means the head integrates utterance-level context.
* Linear CKA -- similarity between every pair of layers' representations,
  invariant to rotation and scaling, plus each layer's participation ratio
  (effective dimensionality). Together they show where along depth the
  representation stops changing.

Both are computed from a single encoder forward pass with
``output_attentions=True, output_hidden_states=True``. No decoder, no prompts,
no special tokens: encoder attention runs over mel frames only, so the
attention-sink problem that plagues decoder self-attention does not exist here.

Module-level imports stay limited to numpy so the pure functions can be unit
tested without torch; librosa/torch/transformers are pulled in lazily, mirroring
`hidden_states_service`.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Whisper's two stride-2 convolutions put encoder positions on a 20 ms grid.
POSITION_STEP_MS = 20.0
WAVEFORM_SAMPLES_PER_POSITION = SAMPLE_RATE // int(1000 / POSITION_STEP_MS)  # 320
DEFAULT_MAX_ENCODER_FRAMES = 512
DEFAULT_N_BINS = 64
# "Local" attention = within +-2 positions (40 ms) of the diagonal.
DIAGONAL_HALF_WIDTH = 2
_EPS = 1e-12


def _to_numpy_2d(state: Any) -> np.ndarray:
    """Accept a torch tensor or array-like and return [time, dim] float64."""
    if hasattr(state, "detach"):
        state = state.detach().float().cpu().numpy()
    array = np.asarray(state, dtype=np.float64)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Unsupported hidden-state shape: {array.shape}")
    return array


def _to_numpy_attention(state: Any) -> np.ndarray:
    """Accept one layer's attention tensor and return [heads, N, N] float64."""
    if hasattr(state, "detach"):
        state = state.detach().float().cpu().numpy()
    array = np.asarray(state, dtype=np.float64)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Unsupported attention shape: {array.shape}")
    return array


def offset_bin_edges(n_positions: int, n_bins: int) -> np.ndarray:
    """Log-spaced integer offset cutoffs (exclusive upper bounds) plus the end.

    Offsets are |i - j| in [0, n_positions - 1]. Log spacing puts resolution
    where speech structure lives (the first few hundred ms) instead of wasting
    bins on the 20+ second tail that carries almost no mass. Enough candidates
    are drawn to survive integer dedup, so the returned edge count (and the
    profile width) is stable for a given `n_bins`.
    """
    top = max(n_positions - 1, 2)
    interior = np.unique(np.floor(np.geomspace(1, top, n_bins * 4)).astype(np.int64))
    if interior.size > n_bins:
        picks = np.unique(np.linspace(0, interior.size - 1, n_bins).astype(np.int64))
        interior = interior[picks]
    return np.concatenate([[0], interior, [n_positions]])


def attention_distance_profiles(
    layer_attention: np.ndarray, bin_edges: np.ndarray
) -> dict[str, Any]:
    """Distance profile for one layer's attention.

    Args:
        layer_attention: [heads, N, N] attention matrix (rows sum to 1).
        bin_edges: exclusive upper bounds from `offset_bin_edges`.

    Returns:
        head_profiles: [heads, n_bins] mass per offset bin, rows summing to 1.
        diagonal_mass: [heads] fraction of mass within +-`DIAGONAL_HALF_WIDTH`.
        profile_entropy: [heads] Shannon entropy of the profile normalised to
            [0, 1] against a uniform distribution over populated bins.
    """
    heads, n, width = layer_attention.shape
    if width != n:
        raise ValueError(f"Attention must be square, got {n}x{width}")

    offsets = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    # Bin b covers offsets [bin_edges[b], bin_edges[b+1]); digitize only needs
    # the interior bounds, and passing edge 0 would push offset 0 into bin 1.
    interior = bin_edges[1:-1]
    bin_idx = np.digitize(offsets.ravel(), interior, right=False)
    n_bins = max(len(bin_edges) - 1, 1)

    flat = layer_attention.reshape(heads, -1)
    total = flat.sum(axis=1)
    profiles = np.zeros((heads, n_bins), dtype=np.float64)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            profiles[:, b] = flat[:, mask].sum(axis=1)
    profiles = profiles / np.maximum(total[:, None], _EPS)

    diag_mask = offsets <= DIAGONAL_HALF_WIDTH
    diagonal_mass = flat[:, diag_mask.ravel()].sum(axis=1) / np.maximum(total, _EPS)

    entropies = np.zeros(heads, dtype=np.float64)
    for head in range(heads):
        row = profiles[head]
        populated = row[row > _EPS]
        if populated.size > 1:
            entropies[head] = float(
                -(populated * np.log(populated)).sum() / np.log(populated.size)
            )

    return {
        "head_profiles": profiles,
        "diagonal_mass": diagonal_mass,
        "profile_entropy": entropies,
    }


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two [N, dim] representations.

    1.0 means the two layers encode the same information up to an orthogonal
    transform and isotropic scaling; 0.0 means unrelated geometry. Column means
    are removed first so a per-layer mean offset cannot inflate the score.
    """
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, "fro") ** 2
    xx = np.linalg.norm(x.T @ x, "fro")
    yy = np.linalg.norm(y.T @ y, "fro")
    denominator = xx * yy
    if denominator <= _EPS:
        return 0.0
    return float(cross / denominator)


def participation_ratio(x: np.ndarray) -> float:
    """Participation ratio of a representation's spectrum, normalised to [0, 1].

    PR = (sum lambda)^2 / sum(lambda^2) over the covariance eigenvalues; computed
    from singular values of the centered [N, dim] matrix. 1.0 means variance is
    spread evenly across every available direction; near 0 means the layer's
    output lives in a handful of directions.
    """
    x = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False)
    power = singular**2
    denominator = (power**2).sum()
    if denominator <= _EPS:
        return 0.0
    effective = float((power.sum() ** 2) / denominator)
    return float(effective / max(min(x.shape), 1))


def cka_layer_matrix(
    hidden_states: Sequence[Any], max_frames: int
) -> dict[str, Any]:
    """Pairwise linear CKA and participation ratio across all layers.

    `hidden_states` is what ``output_hidden_states=True`` returns: the embedding
    output followed by every block, each [batch, N, dim]. Frames are strided to
    at most `max_frames` rows before the maths, which bounds cost for
    whisper-large (32 layers x 1280 dims) without changing the picture.
    """
    arrays = []
    stride = 1
    for state in hidden_states:
        array = _to_numpy_2d(state)
        stride = max(stride, int(np.ceil(array.shape[0] / max(max_frames, 1))))
        arrays.append(array)
    if stride > 1:
        arrays = [array[::stride] for array in arrays]

    count = len(arrays)
    matrix = [[1.0] * count for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            value = linear_cka(arrays[i], arrays[j])
            matrix[i][j] = value
            matrix[j][i] = value
    adjacent = [matrix[i][i + 1] for i in range(count - 1)]
    ratios = [participation_ratio(array) for array in arrays]
    return {
        "matrix": matrix,
        "adjacent_cka": adjacent,
        "participation_ratio": ratios,
    }


def run_encoder_analysis(
    audio_path: str,
    model_size: str = "base",
    max_encoder_frames: int = DEFAULT_MAX_ENCODER_FRAMES,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, Any]:
    """One forward pass -> distance profiles + CKA for the Whisper encoder.

    The waveform is truncated to `max_encoder_frames` positions (~20 ms each) by
    slicing the mel input, which keeps whisper-large's attention memory bounded:
    32 layers x 16 heads x N^2 grows fast past ~10 s of audio.
    """
    import torch

    from app.services import model_loader_service as models
    from app.services.hidden_states_service import layer_names, load_waveform

    if model_size == "large":
        processor, model = models.get_whisper_large_models()
    else:
        processor, model = models.get_whisper_base_models()
    device = next(model.parameters()).device

    waveform = load_waveform(audio_path, SAMPLE_RATE)
    waveform = waveform[: max_encoder_frames * WAVEFORM_SAMPLES_PER_POSITION]
    if waveform.size == 0:
        raise ValueError(f"Loaded audio is empty for file: {audio_path}")

    input_features = processor(
        waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features
    # The feature extractor always emits the full 30 s window (3000 mel frames);
    # slice to 2x positions because the conv stem downsamples mel frames 2x.
    input_features = input_features[..., : 2 * max_encoder_frames].to(device)

    with torch.no_grad():
        output = model.encoder(
            input_features,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    attentions = getattr(output, "attentions", None)
    hidden_states = getattr(output, "hidden_states", None)
    if not attentions or not hidden_states:
        raise ValueError("Encoder did not return attentions/hidden states")

    n_positions = int(output.last_hidden_state.shape[1])
    bin_edges = offset_bin_edges(n_positions, n_bins)
    bin_edges_ms = [float(edge * POSITION_STEP_MS) for edge in bin_edges]

    profile_layers = []
    head_count = int(_to_numpy_attention(attentions[0]).shape[0])
    for layer_attention in attentions:
        array = _to_numpy_attention(layer_attention)
        stats = attention_distance_profiles(array, bin_edges)
        profiles = stats["head_profiles"]
        profile_layers.append(
            {
                "head_profiles": [[float(v) for v in row] for row in profiles],
                "mean_profile": [float(v) for v in profiles.mean(axis=0)],
                "diagonal_mass": [float(v) for v in stats["diagonal_mass"]],
                "profile_entropy": [float(v) for v in stats["profile_entropy"]],
            }
        )
        logger.info(
            "encoder_analysis: profiled attention layer (%d heads, %d positions)",
            head_count,
            n_positions,
        )

    cka = cka_layer_matrix(hidden_states, max_encoder_frames)

    return {
        "model": f"whisper-{model_size}",
        "encoder_positions": n_positions,
        "position_step_ms": POSITION_STEP_MS,
        "audio_seconds_analyzed": n_positions * POSITION_STEP_MS / 1000.0,
        "attention_profiles": {
            "n_layers": len(profile_layers),
            "n_heads": head_count,
            "bin_edges_ms": bin_edges_ms,
            "layers": profile_layers,
        },
        "cka": {
            "layer_names": layer_names(len(hidden_states)),
            "matrix": cka["matrix"],
            "adjacent_cka": cka["adjacent_cka"],
            "participation_ratio": cka["participation_ratio"],
        },
    }
