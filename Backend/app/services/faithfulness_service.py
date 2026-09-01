"""Faithfulness evaluation for saliency maps.

A saliency map makes a claim: *"the model used these regions."*  Nothing checks
that claim.  This module does, by deleting the regions a map calls important and
re-measuring the model.

Pure computation: no torch, no Redis, no model loading.  The model enters only
through a ``score_fn`` callable, which is what makes the whole thing testable
against a synthetic model whose ground truth is known -- see
``tests/test_faithfulness_service.py``.  That test file is where a broken metric
gets caught; on real audio there is no way to distinguish "this map is
unfaithful" from "the evaluation is subtly wrong".

Reading the numbers honestly
----------------------------
``aopc_deletion`` on its own means nothing.  Masking audio degrades a model
whichever regions you pick, so a large deletion score is partly just damage.
The comparison that carries information is against the random baseline:

``faithfulness_gain = aopc_deletion - aopc_random``
    The headline.  At or below zero, the map located nothing the model was not
    equally hurt by losing at random.  The baseline is an estimate, so read the
    gain against ``aopc_random_stderr``: anything smaller than the standard error
    on that estimate is noise, not a finding.  ``verdict`` already applies that
    rule.

``occlusion_spearman``
    Rank correlation between what the map claims each segment is worth and what
    removing that segment actually costs.  The most directly interpretable
    number here, and the one to show a reader who does not want a curve.

This mirrors the ``selectivity`` convention in ``probing_service``: a score is
reported against the baseline that would arise with no real signal, never alone.

The score_fn contract
---------------------
``score_fn(spans) -> float in [0, 1]``, where ``spans`` is a list of
``(start_seconds, end_seconds)`` regions to mask and the return value is *higher
when the model more strongly produces the output it produced on clean audio*.
Callers are responsible for making that a bounded, probability-like quantity:

* classification -- softmax probability of the clean predicted class
* seq2seq ASR -- geometric-mean per-token probability of the clean transcript
  (``exp`` of the mean token log-probability) under teacher forcing
* CTC ASR -- ``exp`` of the mean log-softmax along the clean greedy path

Keeping every model kind on the same [0, 1] scale is what makes an AOPC
comparable between a Whisper run and a wav2vec2 run.

The output contract is consumed by the frontend
(``SaliencyFaithfulness`` in ``Frontend/src/lib/faithfulness.ts``), so every key
below is present even in degenerate cases -- the UI must never branch on a
partial shape.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Resolution the ranking runs at, in frames per second.
#
# This is not a performance knob, it is a correctness one.  wav2vec2 saliency
# comes back at *one value per audio sample* (55k values for a 3.4 s clip).
# Ranking at that resolution and masking the top 20% does not remove audio --
# it punches tens of thousands of sample-sized holes spread across the whole
# clip, which is broadband noise injection wearing a deletion curve's clothes.
# The measured effect then has nothing to do with *where* the map pointed.
#
# 50 Hz (20 ms frames) is Whisper's own encoder frame rate and wav2vec2's
# transformer frame rate, so pooling to it costs no real localisation and makes
# every model's map rank in the same units.
EVAL_FRAME_RATE = 50.0

# Below this, the tracked score has no room to fall and every metric collapses to
# zero.  That is indistinguishable in the output from "the map found nothing", so
# it is reported as a skip instead: a confidently wrong verdict is worse than an
# admitted one.  A baseline this low means the scorer is misconfigured or the
# model produced nothing on the clean clip.
MIN_BASELINE_SCORE = 1e-4

# A map cannot be evaluated below this many frames: the mask granularity gets
# coarser than the fractions being swept and every curve point collapses onto
# the same handful of spans.
MIN_FRAMES = 4

# Verdict thresholds.  Deliberately here rather than in the UI, so the rule is
# stated once and every consumer reads the same result.
GAIN_UNINFORMATIVE = 0.02
GAIN_FAITHFUL = 0.10
SPEARMAN_FAITHFUL = 0.30

# The random baseline is a sample, and one draw of it is noisy enough that an
# uninformative map can post a gain around 0.1 by luck.  Averaging a few
# placements per fraction costs `n_steps` forward passes each and buys both a
# steadier baseline and an estimate of its spread, which the verdict then
# requires the gain to clear.
DEFAULT_RANDOM_REPEATS = 3

Spans = list[tuple[float, float]]
ScoreFn = Callable[[Spans], float]


def _pool(values: np.ndarray, target: int) -> np.ndarray:
    """Mean-pool a timeline down to `target` frames, preserving its shape.

    Returns the input untouched when it is already at or below the target;
    nothing is ever upsampled, because inventing resolution the map does not
    have would put mask boundaries where no attribution was computed.
    """
    if target <= 0 or values.size <= target:
        return values
    edges = np.linspace(0, values.size, target + 1).astype(int)
    return np.array(
        [values[start:end].mean() if end > start else values[min(start, values.size - 1)]
         for start, end in zip(edges[:-1], edges[1:])],
        dtype=np.float64,
    )


def _merge_runs(indices: Sequence[int]) -> list[tuple[int, int]]:
    """Collapse sorted frame indices into ``[start, end)`` blocks."""
    blocks: list[tuple[int, int]] = []
    for index in sorted(indices):
        if blocks and index == blocks[-1][1]:
            blocks[-1] = (blocks[-1][0], index + 1)
        else:
            blocks.append((index, index + 1))
    return blocks


def _to_spans(blocks: Sequence[tuple[int, int]], frame_seconds: float) -> Spans:
    return [(start * frame_seconds, end * frame_seconds) for start, end in blocks]


def _random_blocks_like(
    blocks: Sequence[tuple[int, int]], n_frames: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Randomly reposition blocks, preserving their number and lengths.

    The block *structure* is preserved on purpose.  Masking the same total
    duration as scattered single frames removes far less coherent information
    than masking it as a few contiguous stretches, which would make the random
    baseline artificially easy to beat and every map look faithful.
    """
    lengths = [end - start for start, end in blocks]
    total = sum(lengths)
    if total <= 0:
        return []
    if total >= n_frames:
        return [(0, n_frames)]
    rng.shuffle(lengths)
    # Uniform composition of the unmasked frames into the gaps around the blocks
    # (sorted uniform cut points, not `multinomial` -- multinomial concentrates
    # each gap around its mean, which would pin every random mask near the middle
    # of the clip and quietly bias the baseline toward whatever lives there).
    free = n_frames - total
    cuts = np.sort(rng.integers(0, free + 1, size=len(lengths)))
    gaps = np.diff(np.concatenate(([0], cuts, [free])))
    placed: list[tuple[int, int]] = []
    cursor = int(gaps[0])
    for index, length in enumerate(lengths):
        placed.append((cursor, cursor + length))
        cursor += length + int(gaps[index + 1])
    return placed


def _complement(blocks: Sequence[tuple[int, int]], n_frames: int) -> list[tuple[int, int]]:
    kept = np.zeros(n_frames, dtype=bool)
    for start, end in blocks:
        kept[start:end] = True
    return _merge_runs(np.flatnonzero(~kept).tolist())


def _aopc(baseline: float, points: Sequence[dict[str, float]]) -> float:
    """Mean drop from baseline across the swept fractions.

    The area-over-the-perturbation-curve convention: one number summarising how
    fast the model falls apart as the ranked regions are removed.
    """
    if not points:
        return 0.0
    return float(np.mean([baseline - point["score"] for point in points]))


def _auc(points: Sequence[dict[str, float]]) -> float:
    """Trapezoidal area under a score-versus-fraction curve."""
    if len(points) < 2:
        return float(points[0]["score"]) if points else 0.0
    fractions = np.array([point["fraction"] for point in points], dtype=float)
    scores = np.array([point["score"] for point in points], dtype=float)
    span = float(fractions[-1] - fractions[0])
    if span <= 0:
        return float(np.mean(scores))
    # `trapezoid` is the NumPy 2 name; `trapz` is the 1.x one this project pins.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(scores, fractions) / span)


def _spearman(a: Sequence[float], b: Sequence[float]) -> tuple[float | None, float | None]:
    """Spearman rho with a p-value when SciPy is importable.

    Falls back to a rank-Pearson computed with NumPy so the metric survives in a
    stripped worker image; the p-value is simply omitted there rather than
    approximated.
    """
    if len(a) < 3 or len(b) != len(a):
        return None, None
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None, None
    try:
        import warnings

        from scipy.stats import spearmanr

        with warnings.catch_warnings():
            # A degenerate input is already handled by the nan check below; the
            # warning would just be noise in worker logs.
            warnings.simplefilter("ignore")
            result = spearmanr(a, b)
        rho, p_value = float(result.statistic), float(result.pvalue)
        return (None, None) if math.isnan(rho) else (rho, p_value)
    except Exception:
        rank_a = np.argsort(np.argsort(np.asarray(a, dtype=float)))
        rank_b = np.argsort(np.argsort(np.asarray(b, dtype=float)))
        rho = float(np.corrcoef(rank_a, rank_b)[0, 1])
        return (None, None) if math.isnan(rho) else (rho, None)


def _verdict(gain: float, spearman: float | None, noise: float) -> str:
    """Classify a gain, refusing to call anything inside the baseline's error bar.

    `noise` is the standard error on the *mean* random AOPC -- the uncertainty on
    the quantity `gain` is actually measured against. Using the raw per-draw
    standard deviation instead would over-reject: it is the spread of a single
    draw, not of the average, and at two or three draws the two differ enough to
    flip a verdict.
    """
    if gain <= max(GAIN_UNINFORMATIVE, noise):
        return "uninformative"
    if gain >= GAIN_FAITHFUL and (spearman is None or spearman >= SPEARMAN_FAITHFUL):
        return "faithful"
    return "weak"


def _empty_result(
    reason: str,
    *,
    target: dict[str, Any],
    baseline_score: float,
    attribution_source: str | None,
    n_steps: int,
    seed: int,
    audio_seconds: float,
) -> dict[str, Any]:
    return {
        "target": target,
        "baseline_score": float(baseline_score),
        "attribution_source": attribution_source,
        "masked_fractions": [],
        "curves": {
            "deletion_saliency": [],
            "deletion_random": [],
            "deletion_inverse": [],
            "insertion_saliency": [],
        },
        "comparison": {
            "fraction": 0.0,
            "clean_score": float(baseline_score),
            "masked_score": float(baseline_score),
            "random_score": float(baseline_score),
            "removed_spans": [],
            "random_spans": [],
        },
        "occlusion": [],
        "metrics": {
            "aopc_deletion": 0.0,
            "aopc_random": 0.0,
            "aopc_random_stderr": 0.0,
            "faithfulness_gain": 0.0,
            "aopc_inverse": 0.0,
            "comprehensiveness": 0.0,
            "sufficiency": 0.0,
            "auc_deletion": 0.0,
            "auc_insertion": 0.0,
            "occlusion_spearman": None,
            "occlusion_p_value": None,
        },
        "n_steps": int(n_steps),
        "eval_frames": 0,
        "random_repeats": 0,
        "seed": int(seed),
        "audio_seconds": float(audio_seconds),
        "verdict": "uninformative",
        "skipped_reason": reason,
    }


def evaluate_faithfulness(
    series: Sequence[float],
    segments: Sequence[dict[str, Any]],
    total_duration: float,
    score_fn: ScoreFn,
    *,
    target: dict[str, Any] | None = None,
    attribution_source: str | None = None,
    n_steps: int = 9,
    top_fraction: float = 0.2,
    seed: int = 42,
    include_occlusion: bool = True,
    random_repeats: int = DEFAULT_RANDOM_REPEATS,
    frame_rate: float = EVAL_FRAME_RATE,
) -> dict[str, Any]:
    """Measure whether a saliency map points at what the model actually uses.

    ``series`` is the saliency timeline over ``total_duration`` seconds and
    ``segments`` the same map at segment granularity (used only for the
    occlusion correlation).  ``score_fn`` re-runs the model with the given time
    spans masked; see the module docstring for its contract.

    Costs ``(3 + random_repeats) * n_steps + len(segments) + 1`` forward passes
    -- the caller is responsible for keeping the audio short enough that this is
    affordable.
    """
    target = target or {"kind": "unknown", "label": None}
    fractions = [round((index + 1) / (n_steps + 1), 4) for index in range(n_steps)]

    values = np.asarray(list(series), dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    # Pool before ranking, never after: see `EVAL_FRAME_RATE`.
    if frame_rate > 0 and total_duration > 0:
        values = _pool(values, int(round(total_duration * frame_rate)))
    n_frames = values.size

    baseline_score = float(score_fn([]))

    if baseline_score < MIN_BASELINE_SCORE:
        return _empty_result(
            "the model produced no measurable score on the unmasked clip, so there is "
            "nothing for masking to reduce",
            target=target,
            baseline_score=baseline_score,
            attribution_source=attribution_source,
            n_steps=n_steps,
            seed=seed,
            audio_seconds=max(0.0, float(total_duration)),
        )

    if n_frames < MIN_FRAMES or total_duration <= 0:
        return _empty_result(
            f"saliency timeline too short to evaluate ({n_frames} frames)",
            target=target,
            baseline_score=baseline_score,
            attribution_source=attribution_source,
            n_steps=n_steps,
            seed=seed,
            audio_seconds=max(0.0, float(total_duration)),
        )

    frame_seconds = float(total_duration) / n_frames
    rng = np.random.default_rng(seed)

    # Descending saliency, with ties broken at random (seeded, so still
    # reproducible).  Breaking ties by index instead would turn a flat or
    # plateaued map into a left-to-right scan -- a structured mask that beats the
    # random baseline whenever the model happens to depend on early audio, which
    # would report a confident result for a map carrying no ordering at all.
    # Smoothing in the saliency service makes plateaus common, so this matters.
    tie_break = rng.permutation(n_frames)
    order = np.lexsort((tie_break, -values))
    inverse_order = order[::-1]

    repeats = max(1, int(random_repeats))
    deletion_saliency: list[dict[str, float]] = []
    deletion_random: list[dict[str, float]] = []
    deletion_inverse: list[dict[str, float]] = []
    insertion_saliency: list[dict[str, float]] = []
    # [repeat][fraction], so an AOPC can be formed per draw and its spread read off.
    random_draws: list[list[float]] = [[] for _ in range(repeats)]

    for fraction in fractions:
        count = max(1, int(round(fraction * n_frames)))

        top_blocks = _merge_runs(order[:count].tolist())
        deletion_saliency.append({
            "fraction": fraction,
            "score": float(score_fn(_to_spans(top_blocks, frame_seconds))),
        })

        draws = []
        for repeat in range(repeats):
            random_blocks = _random_blocks_like(top_blocks, n_frames, rng)
            score = float(score_fn(_to_spans(random_blocks, frame_seconds)))
            random_draws[repeat].append(score)
            draws.append(score)
        deletion_random.append({
            "fraction": fraction,
            "score": float(np.mean(draws)),
            "std": float(np.std(draws, ddof=1)) if repeats > 1 else 0.0,
        })

        bottom_blocks = _merge_runs(inverse_order[:count].tolist())
        deletion_inverse.append({
            "fraction": fraction,
            "score": float(score_fn(_to_spans(bottom_blocks, frame_seconds))),
        })

        # Insertion keeps only the most salient fraction: everything else masked.
        insertion_saliency.append({
            "fraction": fraction,
            "score": float(score_fn(_to_spans(_complement(top_blocks, n_frames), frame_seconds))),
        })

    # Comprehensiveness / sufficiency at the configured fraction, read off the
    # sweep when it lands on a swept point and measured directly otherwise.
    def _score_at(points: Sequence[dict[str, float]], fraction: float, blocks_fn) -> float:
        for point in points:
            if abs(point["fraction"] - fraction) < 1e-9:
                return point["score"]
        return float(score_fn(_to_spans(blocks_fn(), frame_seconds)))

    top_count = max(1, int(round(top_fraction * n_frames)))
    top_blocks_at_k = _merge_runs(order[:top_count].tolist())
    masked_score = _score_at(deletion_saliency, round(top_fraction, 4), lambda: top_blocks_at_k)
    comprehensiveness = baseline_score - masked_score
    sufficiency = baseline_score - _score_at(
        insertion_saliency, round(top_fraction, 4), lambda: _complement(top_blocks_at_k, n_frames)
    )

    # The before/after view needs the exact regions that were removed, not a
    # reconstruction: the ranking breaks ties at random, so a UI re-deriving it
    # would show a mask the model was never actually given. One extra pass buys a
    # matching random control to display beside it.
    comparison_random_blocks = _random_blocks_like(top_blocks_at_k, n_frames, rng)
    comparison = {
        "fraction": round(top_fraction, 4),
        "clean_score": baseline_score,
        "masked_score": float(masked_score),
        "random_score": float(score_fn(_to_spans(comparison_random_blocks, frame_seconds))),
        "removed_spans": _to_spans(top_blocks_at_k, frame_seconds),
        "random_spans": _to_spans(comparison_random_blocks, frame_seconds),
    }

    occlusion: list[dict[str, float]] = []
    if include_occlusion and segments:
        for segment in segments:
            start = float(segment.get("start_time", 0.0))
            end = float(segment.get("end_time", 0.0))
            if end <= start:
                continue
            occlusion.append({
                "start_time": start,
                "end_time": end,
                "word": segment.get("word"),
                "saliency": float(segment.get("saliency", 0.0)),
                "drop": baseline_score - float(score_fn([(start, end)])),
            })

    spearman, p_value = _spearman(
        [point["saliency"] for point in occlusion], [point["drop"] for point in occlusion]
    )

    aopc_deletion = _aopc(baseline_score, deletion_saliency)
    per_draw_aopc = [
        float(np.mean([baseline_score - score for score in draw])) for draw in random_draws
    ]
    aopc_random = float(np.mean(per_draw_aopc))
    aopc_random_stderr = (
        float(np.std(per_draw_aopc, ddof=1) / np.sqrt(repeats)) if repeats > 1 else 0.0
    )
    gain = aopc_deletion - aopc_random

    result = {
        "target": target,
        "baseline_score": baseline_score,
        "attribution_source": attribution_source,
        "masked_fractions": fractions,
        "curves": {
            "deletion_saliency": deletion_saliency,
            "deletion_random": deletion_random,
            "deletion_inverse": deletion_inverse,
            "insertion_saliency": insertion_saliency,
        },
        "comparison": comparison,
        "occlusion": occlusion,
        "metrics": {
            "aopc_deletion": aopc_deletion,
            "aopc_random": aopc_random,
            "aopc_random_stderr": aopc_random_stderr,
            "faithfulness_gain": gain,
            "aopc_inverse": _aopc(baseline_score, deletion_inverse),
            "comprehensiveness": float(comprehensiveness),
            "sufficiency": float(sufficiency),
            "auc_deletion": _auc(deletion_saliency),
            "auc_insertion": _auc(insertion_saliency),
            "occlusion_spearman": spearman,
            "occlusion_p_value": p_value,
        },
        "n_steps": int(n_steps),
        "eval_frames": int(n_frames),
        "random_repeats": repeats,
        "seed": int(seed),
        "audio_seconds": float(total_duration),
        "verdict": _verdict(gain, spearman, aopc_random_stderr),
        "skipped_reason": None,
    }
    if attribution_source == "energy_fallback":
        # The map being scored is not an attribution at all: `generate_saliency`
        # fell back to an encoder energy map. The numbers below are real, but
        # they describe that fallback, not the requested method.
        logger.warning("Faithfulness evaluated over an energy-map fallback, not a real attribution")
    return result
