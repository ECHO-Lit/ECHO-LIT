"""FR-7 metrics + sensitivity-profile aggregation (docs/FR7plan.md Part 1 S4, S8)."""

import jiwer
import numpy as np
import pytest

from app.services.sensitivity_metrics_service import (
    asr_metrics,
    classification_metrics,
    get_normalizer,
    set_level_metrics,
)
from app.services.linguistic_acoustic_service import (
    build_property_profile,
    build_sensitivity_profile,
)


def test_wer_matches_jiwer_directly():
    normalize = get_normalizer("en")
    hyp, ref = "the quick brown socks", "the quick brown fox"
    expected = jiwer.process_words(normalize(ref), normalize(hyp)).wer
    result = asr_metrics(hyp, ref, "en")
    assert result["wer"] == pytest.approx(expected)
    assert result["degenerate_reference"] is False


def test_wer_identical_is_zero():
    result = asr_metrics("the quick brown fox", "the quick brown fox", "en")
    assert result["wer"] == 0.0
    assert result["cer"] == 0.0


def test_wer_degenerate_reference():
    result = asr_metrics("hello", "", "en")
    assert result["degenerate_reference"] is True
    assert result["wer"] == 1.0

    silent = asr_metrics("", "", "en")
    assert silent["wer"] == 0.0


def test_insertion_ratio_flags_hallucination_shape():
    # Reference has 4 words, hypothesis inserts 3 extra -> insertion-heavy.
    result = asr_metrics("the quick brown fox ran away fast", "the quick brown fox", "en")
    assert result["insertions"] >= 3
    assert result["insertion_ratio"] > 0.5


def test_classification_metrics_flip_and_js():
    baseline = {"happy": 0.7, "sad": 0.2, "neutral": 0.1}
    perturbed = {"happy": 0.1, "sad": 0.8, "neutral": 0.1}
    result = classification_metrics(perturbed, baseline)
    assert result["label_flipped"] == 1
    assert result["predicted_label"] == "sad"
    assert result["baseline_label"] == "happy"
    assert 0.0 <= result["js_divergence"] <= np.log(2) + 1e-9


def test_classification_metrics_identical_posteriors():
    baseline = {"happy": 0.7, "sad": 0.3}
    result = classification_metrics(baseline, baseline)
    assert result["label_flipped"] == 0
    assert result["js_divergence"] == pytest.approx(0.0, abs=1e-9)
    assert result["confidence_delta"] == pytest.approx(0.0, abs=1e-9)


def test_set_level_metrics_macro_f1():
    y_true = ["happy", "sad", "happy", "neutral"]
    y_pred = ["happy", "sad", "sad", "neutral"]
    result = set_level_metrics(y_true, y_pred)
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert result["accuracy"] == pytest.approx(0.75)


def _points(thetas, degradations):
    return [
        {"theta": t, "degradation": d, "applicable": True, "raw": {}}
        for t, d in zip(thetas, degradations)
    ]


def test_breakdown_interpolation():
    thetas = [40, 30, 20, 10, 5, 0]
    degradations = [0.0, 0.05, 0.15, 0.35, 0.60, 0.90]
    profile = build_property_profile(_points(thetas, degradations), "noise")
    assert profile["applicable"] is True
    assert 5 < profile["breakdown_theta"] < 10


def test_sensitivity_index_ranks_properties():
    flat = build_property_profile(
        _points([-6, -3, 0, 3, 6], [0.02, 0.02, 0.0, 0.03, 0.02]), "pitch"
    )
    steep = build_property_profile(_points([40, 20, 0], [0.0, 0.4, 0.95]), "noise")
    assert steep["sensitivity_index"] > flat["sensitivity_index"]


def test_not_applicable_property_profile():
    profile = build_property_profile(_points([1.0], [0.5]), "speed")
    assert profile["applicable"] is False


def test_sensitivity_profile_verdicts():
    linguistic = build_sensitivity_profile(
        [build_property_profile(_points([-6, 0, 6], [0.02, 0.0, 0.03]), "pitch")],
        controls={"lexical_destruction": {"degradation": 0.8}},
        task="transcription",
    )
    assert linguistic["verdict"] == "linguistically_driven"

    dominated = build_sensitivity_profile(
        [build_property_profile(_points([40, 20, 0], [0.0, 0.4, 0.95]), "noise")],
        controls={"lexical_destruction": {"degradation": 0.9}},
        task="transcription",
    )
    assert dominated["verdict"] in {"acoustically_dominated", "mixed"}


def test_sensitivity_profile_inconclusive_when_nothing_applicable():
    profile = build_sensitivity_profile(
        [build_property_profile(_points([1.0], [0.5]), "speed")],
        controls={"lexical_destruction": {"degradation": 0.8}},
        task="transcription",
    )
    assert profile["verdict"] == "inconclusive"
