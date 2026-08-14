"""FR-10 explanation-fairness / saliency grounding metrics
(docs/FR10plan.md Part 1 S4.4, S9)."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.grounding_service import compute_speech_mask, grounding_metrics, group_grounding_summary


def _waveform_half_silent(n=16000, seed=0):
    rng = np.random.default_rng(seed)
    wave = rng.normal(0, 0.15, n).astype(np.float32)
    wave[: n // 2] = 0.0  # first half silence, second half speech-like energy
    return wave


def test_grounding_lift_is_one_for_uniform_saliency_regardless_of_duty_cycle():
    # Uniform attribution carries no information about where the mass falls,
    # so lift should be ~1.0 (chance) whether duty cycle is 0.5 or 0.2.
    series = list(np.ones(100))
    for duty in (0.2, 0.5, 0.8):
        n = 16000
        wave = np.random.default_rng(1).normal(0, 0.15, n).astype(np.float32)
        silent_len = int(n * (1 - duty))
        wave[:silent_len] = 0.0
        result = grounding_metrics(series, 1.0, wave, 16000)
        assert result["status"] == "ok"
        assert result["grounding_lift"] == pytest.approx(1.0, abs=0.15), f"duty={duty}: {result}"


def test_grounding_lift_exceeds_one_when_saliency_concentrates_on_speech():
    wave = _waveform_half_silent()
    series = [0.0] * 50 + [1.0] * 50  # concentrated on the (non-silent) second half
    result = grounding_metrics(series, 1.0, wave, 16000)
    assert result["status"] == "ok"
    assert result["grounding_lift"] > 1.5


def test_grounding_lift_below_one_when_saliency_concentrates_on_silence():
    wave = _waveform_half_silent()
    series = [1.0] * 50 + [0.0] * 50  # concentrated on the silent first half
    result = grounding_metrics(series, 1.0, wave, 16000)
    assert result["status"] == "ok"
    assert result["grounding_lift"] < 0.7


def test_attribution_entropy_bounds():
    n = 100
    uniform = grounding_metrics(list(np.ones(n)), 1.0, None, None)
    peaked = grounding_metrics([1.0] + [0.0] * (n - 1), 1.0, None, None)
    assert uniform["attribution_entropy"] == pytest.approx(1.0, abs=0.02)
    assert peaked["attribution_entropy"] < 0.2
    assert peaked["attribution_entropy"] < uniform["attribution_entropy"]


def test_degenerate_all_zero_series_reported_not_crashed():
    result = grounding_metrics([0.0] * 50, 1.0, None, None)
    assert result["status"] == "degenerate"


def test_empty_series_reported_not_crashed():
    result = grounding_metrics([], 1.0, None, None)
    assert result["status"] == "degenerate"


def test_missing_waveform_treats_all_frames_as_speech():
    # No waveform available -> grounding_lift is neutral (all-speech mask),
    # not a crash or a silently wrong number.
    result = grounding_metrics(list(np.ones(50)), 1.0, None, None)
    assert result["status"] == "ok"
    assert result["speech_duty_cycle"] == 1.0
    assert result["grounding_lift"] == pytest.approx(1.0, abs=1e-6)


def test_speech_mask_flags_silence_and_speech_correctly():
    wave = _waveform_half_silent(n=8000)
    mask = compute_speech_mask(wave, 16000, 100)
    # First half of frames should be mostly NOT speech, second half mostly speech.
    assert mask[:40].mean() < 0.3
    assert mask[60:].mean() > 0.7


def test_group_grounding_summary_averages_only_ok_items():
    items = [
        {"status": "ok", "grounding_lift": 1.5, "attribution_entropy": 0.5, "attribution_gini": 0.3, "top10_mass": 0.2},
        {"status": "ok", "grounding_lift": 2.5, "attribution_entropy": 0.7, "attribution_gini": 0.5, "top10_mass": 0.4},
        {"status": "degenerate"},
    ]
    summary = group_grounding_summary(items)
    assert summary["status"] == "ok"
    assert summary["n_explained"] == 2  # degenerate item excluded from the average
    assert summary["grounding_lift"] == pytest.approx(2.0)


def test_group_grounding_summary_all_degenerate_reports_unavailable():
    summary = group_grounding_summary([{"status": "degenerate"}, {"status": "unavailable"}])
    assert summary["status"] == "unavailable"
