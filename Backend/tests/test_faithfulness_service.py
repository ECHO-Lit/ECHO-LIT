"""
Saliency faithfulness tests.

This is the highest-value test file in the faithfulness feature, for the same
reason `test_probing_service.py` is for probing: it is the only place the ground
truth is knowable.  On real audio there is no way to tell "this saliency map is
unfaithful" from "the evaluation pipeline is subtly broken and reports that".
Here the model's dependence on the audio is planted, so the answer is known in
advance.

If `test_planted_region_is_recovered` fails, every number the faithfulness UI
shows is meaningless -- fix it before looking at anything else.
"""

import numpy as np
import pytest

from app.services.faithfulness_service import evaluate_faithfulness


SEED = 42
DURATION = 10.0
N_FRAMES = 100
PLANTED = (3.0, 5.0)


def _overlap(spans, window):
    """Seconds of `window` covered by `spans`."""
    total = 0.0
    for start, end in spans:
        total += max(0.0, min(end, window[1]) - max(start, window[0]))
    return total


def _synthetic_model(
    planted=PLANTED, baseline=0.95, floor=0.05, collateral=0.0
):
    """A model whose output depends only on the planted window.

    `collateral` adds a per-second penalty for masking *anything*, which is what
    real audio models do -- damage is not free even in irrelevant regions.  It is
    the reason the random baseline exists: without subtracting it, collateral
    damage alone would make any map look faithful.
    """
    width = planted[1] - planted[0]

    def score_fn(spans):
        covered = _overlap(spans, planted) / width
        masked = sum(end - start for start, end in spans)
        score = baseline - covered * (baseline - floor) - collateral * masked
        return float(max(0.0, min(1.0, score)))

    return score_fn


def _series(peak=PLANTED, background=0.05, height=1.0, n_frames=N_FRAMES):
    """A saliency timeline that is high inside `peak` and low elsewhere."""
    frame_seconds = DURATION / n_frames
    values = np.full(n_frames, background, dtype=float)
    for index in range(n_frames):
        centre = (index + 0.5) * frame_seconds
        if peak[0] <= centre < peak[1]:
            values[index] = height
    return values.tolist()


def _segments(series, n_segments=20, n_frames=N_FRAMES):
    """Segment-level view of a timeline, matching what the saliency service emits."""
    values = np.asarray(series, dtype=float)
    width = DURATION / n_segments
    frames_per = n_frames / n_segments
    segments = []
    for index in range(n_segments):
        lo = int(index * frames_per)
        hi = max(lo + 1, int((index + 1) * frames_per))
        segments.append({
            "start_time": index * width,
            "end_time": (index + 1) * width,
            "word": f"segment_{index + 1}",
            "saliency": float(values[lo:hi].mean()),
            "intensity": float(values[lo:hi].mean()),
        })
    return segments


def _run(series, score_fn, **kwargs):
    return evaluate_faithfulness(
        series,
        _segments(series),
        DURATION,
        score_fn,
        seed=SEED,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The core claim
# ---------------------------------------------------------------------------


def test_planted_region_is_recovered():
    """A map pointing at the region the model actually uses must score well."""
    result = _run(_series(), _synthetic_model())

    metrics = result["metrics"]
    assert metrics["faithfulness_gain"] > 0.2
    assert metrics["aopc_deletion"] > metrics["aopc_random"]
    assert metrics["occlusion_spearman"] > 0.5
    assert result["verdict"] == "faithful"


def test_flat_map_is_not_credited():
    """A map that says nothing must not be credited for collateral damage.

    Checked across seeds rather than on one: a flat map has no ordering, so its
    deletion mask is itself a random draw and a single evaluation lands anywhere
    in the baseline's spread.  The property that must hold is that nothing is
    credited *systematically* -- mean gain at zero, and never a "faithful"
    verdict.  (The verdict rule enforces the per-run half of this by refusing to
    call a gain smaller than `aopc_random_std`.)
    """
    flat = [0.5] * N_FRAMES
    gains = []
    for seed in range(6):
        result = evaluate_faithfulness(
            flat, _segments(flat), DURATION, _synthetic_model(collateral=0.05), seed=seed
        )
        gains.append(result["metrics"]["faithfulness_gain"])
        assert result["verdict"] != "faithful"

    assert float(np.mean(gains)) == pytest.approx(0.0, abs=0.05)


def test_inverted_map_scores_negative_gain():
    """Pointing away from what the model uses is worse than pointing randomly."""
    inverted = [1.0 - value for value in _series()]
    result = _run(inverted, _synthetic_model())

    assert result["metrics"]["faithfulness_gain"] < 0
    assert result["verdict"] == "uninformative"


def test_collateral_damage_alone_does_not_look_faithful():
    """A model hurt equally by any masking must produce no gain for any map.

    This is the failure mode the random baseline exists to catch: without it,
    `aopc_deletion` would be large here and the feature would report a confident
    "faithful" for a map carrying no information at all.
    """
    only_damage = lambda spans: max(0.0, 0.95 - 0.08 * sum(e - s for s, e in spans))
    result = _run(_series(), only_damage)

    assert result["metrics"]["aopc_deletion"] > 0.1  # damage is real
    assert result["metrics"]["faithfulness_gain"] == pytest.approx(0.0, abs=0.05)
    assert result["verdict"] == "uninformative"


def test_inverse_curve_sits_below_deletion_curve():
    """Deleting the least salient audio must hurt less than deleting the most."""
    result = _run(_series(), _synthetic_model())

    assert result["metrics"]["aopc_inverse"] < result["metrics"]["aopc_deletion"]


def test_comprehensiveness_and_sufficiency():
    """Removing the top fraction hurts; keeping only the top fraction does not."""
    result = _run(_series(), _synthetic_model(), top_fraction=0.2)

    metrics = result["metrics"]
    # The planted window is 20% of the audio and the map ranks it first, so the
    # top 20% is exactly what the model depends on.
    assert metrics["comprehensiveness"] > 0.5
    assert metrics["sufficiency"] < 0.2


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------


def test_noise_estimate_is_the_error_on_the_mean():
    """The verdict gate must shrink as the baseline is pinned down, not stay put.

    `faithfulness_gain` is measured against the *mean* random AOPC, so the
    uncertainty that matters is the standard error of that mean. Reporting the
    per-draw standard deviation instead would keep the gate wide however many
    draws were paid for, and at two or three draws that flips verdicts.
    """
    series, score_fn = _series(), _synthetic_model(collateral=0.05)
    few = evaluate_faithfulness(series, [], DURATION, score_fn, seed=SEED, random_repeats=2)
    many = evaluate_faithfulness(series, [], DURATION, score_fn, seed=SEED, random_repeats=8)

    assert many["metrics"]["aopc_random_stderr"] < few["metrics"]["aopc_random_stderr"]
    assert few["metrics"]["aopc_random_stderr"] > 0


def test_single_draw_reports_no_error_bar():
    result = evaluate_faithfulness(
        _series(), [], DURATION, _synthetic_model(), seed=SEED, random_repeats=1
    )
    assert result["metrics"]["aopc_random_stderr"] == 0.0


def test_random_baseline_masks_the_same_duration():
    """The baseline is only fair if it removes as much audio as the map does."""
    masked_totals = []

    def recording_score_fn(spans):
        masked_totals.append(sum(end - start for start, end in spans))
        return 0.9

    repeats = 3
    evaluate_faithfulness(
        _series(), [], DURATION, recording_score_fn,
        n_steps=3, seed=SEED, random_repeats=repeats,
    )

    # Call order: baseline, then per fraction -- saliency, `repeats` random
    # placements, inverse, insertion.
    per_fraction = 3 + repeats
    for index in range(3):
        window = masked_totals[1 + index * per_fraction: 1 + (index + 1) * per_fraction]
        saliency, randoms, inverse, insertion = window[0], window[1:-2], window[-2], window[-1]
        assert len(randoms) == repeats
        for random_ in randoms:
            assert random_ == pytest.approx(saliency)
        assert inverse == pytest.approx(saliency)
        assert insertion == pytest.approx(DURATION - saliency)


def test_deterministic_under_seed():
    series, score_fn = _series(), _synthetic_model(collateral=0.03)
    first = _run(series, score_fn)
    second = _run(series, score_fn)

    assert first["curves"]["deletion_random"] == second["curves"]["deletion_random"]


def test_random_baseline_actually_varies_with_seed():
    """Guards against a "random" baseline that silently degenerated to a constant."""
    series, score_fn = _series(), _synthetic_model()
    a = evaluate_faithfulness(series, [], DURATION, score_fn, seed=1)
    b = evaluate_faithfulness(series, [], DURATION, score_fn, seed=2)

    assert a["curves"]["deletion_random"] != b["curves"]["deletion_random"]


def test_random_blocks_are_placed_uniformly():
    """The baseline must be able to land anywhere, not just mid-clip.

    A gap distribution concentrated on its mean (the trap `multinomial` sets)
    pins every random mask near the centre of the audio, which silently biases
    the baseline toward whatever the model happens to keep there.
    """
    from app.services.faithfulness_service import _random_blocks_like

    rng = np.random.default_rng(SEED)
    starts = [_random_blocks_like([(30, 50)], 100, rng)[0][0] for _ in range(300)]

    assert min(starts) < 10
    assert max(starts) > 70
    assert float(np.std(starts)) > 15


# ---------------------------------------------------------------------------
# Contract: the UI must never meet a partial shape
# ---------------------------------------------------------------------------


EXPECTED_METRICS = {
    "aopc_deletion", "aopc_random", "aopc_random_stderr", "faithfulness_gain", "aopc_inverse",
    "comprehensiveness", "sufficiency", "auc_deletion", "auc_insertion",
    "occlusion_spearman", "occlusion_p_value",
}


def test_dead_baseline_is_reported_as_a_skip():
    """A scorer returning nothing must not read as "the map found nothing".

    Both produce all-zero metrics, but one is a finding and the other is a
    misconfiguration. Conflating them means shipping a confident verdict about a
    measurement that never happened.
    """
    result = _run(_series(), lambda spans: 0.0)

    assert result["skipped_reason"] is not None
    assert "no measurable score" in result["skipped_reason"]
    assert result["metrics"]["faithfulness_gain"] == 0.0


@pytest.mark.parametrize(
    "series,duration",
    [
        ([], 10.0),            # no timeline at all
        ([0.5] * 100, 0.0),    # no duration to map it onto
        ([0.5, 0.9], 10.0),    # below MIN_FRAMES
    ],
)
def test_degenerate_inputs_return_the_full_contract(series, duration):
    result = evaluate_faithfulness(series, [], duration, _synthetic_model(), seed=SEED)

    assert set(result["metrics"]) == EXPECTED_METRICS
    assert set(result["curves"]) == {
        "deletion_saliency", "deletion_random", "deletion_inverse", "insertion_saliency"
    }
    assert result["skipped_reason"] is not None
    assert result["verdict"] == "uninformative"


def test_sample_resolution_map_is_pooled_before_ranking():
    """A per-sample map must still produce contiguous, audio-sized masks.

    wav2vec2 saliency arrives at one value per audio sample. Ranked at that
    resolution the "top 20%" is tens of thousands of scattered sample-sized
    holes -- broadband noise, not a deletion. Pooling to the evaluation frame
    rate is what keeps the mask a set of real regions.
    """
    per_sample = np.repeat(np.asarray(_series()), 160).tolist()  # 16 kHz-ish
    assert len(per_sample) == N_FRAMES * 160

    result = _run(per_sample, _synthetic_model())

    assert result["eval_frames"] == pytest.approx(DURATION * 50, abs=1)
    # Masks must be few and contiguous, not shredded across the clip.
    assert len(result["comparison"]["removed_spans"]) <= 4
    assert result["metrics"]["faithfulness_gain"] > 0.2
    assert result["verdict"] == "faithful"


def test_pooling_never_upsamples():
    """A map coarser than the frame rate keeps its own resolution."""
    coarse = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
    result = evaluate_faithfulness(coarse, [], DURATION, _synthetic_model(), seed=SEED)

    assert result["eval_frames"] == len(coarse)


def test_comparison_reports_the_regions_actually_removed():
    """The before/after view must be handed the real mask, not a reconstruction."""
    result = _run(_series(), _synthetic_model(), top_fraction=0.2)

    comparison = result["comparison"]
    covered = _overlap(comparison["removed_spans"], PLANTED)
    assert covered == pytest.approx(PLANTED[1] - PLANTED[0], abs=0.15)
    assert comparison["clean_score"] > comparison["masked_score"]
    # Removing the same duration at random must cost less than removing the map's pick.
    assert comparison["random_score"] > comparison["masked_score"]
    removed = sum(end - start for start, end in comparison["removed_spans"])
    random_removed = sum(end - start for start, end in comparison["random_spans"])
    assert random_removed == pytest.approx(removed)


def test_healthy_result_has_the_same_contract():
    result = _run(_series(), _synthetic_model())

    assert set(result["metrics"]) == EXPECTED_METRICS
    assert result["skipped_reason"] is None
    assert len(result["curves"]["deletion_saliency"]) == result["n_steps"]
    assert len(result["occlusion"]) == 20
    assert 0.0 <= result["baseline_score"] <= 1.0


def test_non_finite_saliency_is_survived():
    """NaN in a map must not propagate into the ranking or the metrics."""
    series = _series()
    series[10] = float("nan")
    series[20] = float("inf")
    result = _run(series, _synthetic_model())

    assert np.isfinite(result["metrics"]["faithfulness_gain"])
    assert result["skipped_reason"] is None


def test_energy_fallback_source_is_carried_through():
    """A caller must be able to see the map was not a real attribution."""
    result = _run(_series(), _synthetic_model(), attribution_source="energy_fallback")

    assert result["attribution_source"] == "energy_fallback"


def test_occlusion_can_be_skipped():
    result = _run(_series(), _synthetic_model(), include_occlusion=False)

    assert result["occlusion"] == []
    assert result["metrics"]["occlusion_spearman"] is None
    # A verdict is still reachable without the correlation.
    assert result["verdict"] == "faithful"
