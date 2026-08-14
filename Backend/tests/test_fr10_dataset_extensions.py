"""FR-10 S6: SAA reference-position heatmap / age-onset dose-response and
L2-ARCTIC phone-error grounding / equal-accentedness regression, checked
against the real bundled datasets (docs/FR10plan.md Part 1 S6, S9)."""

import numpy as np
import pytest

from app.services import dataset_service, l2_arctic_annotations as l2, saa_reference_analysis as saa


@pytest.fixture(scope="module")
def saa_rows():
    return dataset_service.load_metadata("saa", None)


@pytest.fixture(scope="module")
def l2_rows():
    return dataset_service.load_metadata("l2-arctic", None)


def test_saa_reference_text_is_shared_across_all_rows(saa_rows):
    passages = {r["reading_passage"] for r in saa_rows}
    assert len(passages) == 1, "SAA's matched design depends on one shared passage"


def test_saa_position_heatmap_isolates_planted_error(saa_rows):
    reference_text = saa_rows[0]["reading_passage"]
    words = saa.canonical_words(reference_text)
    planted_index = 5

    per_group = {"clean": [], "botched": []}
    for row in saa_rows[:20]:
        per_group["clean"].append(reference_text)
    for row in saa_rows[20:40]:
        per_group["botched"].append(
            " ".join(w if i != planted_index else "zzz" for i, w in enumerate(words))
        )

    heatmap = saa.position_error_heatmap(per_group, reference_text)
    assert heatmap["error_rate_by_group"]["clean"][planted_index] == 0.0
    assert heatmap["error_rate_by_group"]["botched"][planted_index] == 1.0
    assert sum(heatmap["error_rate_by_group"]["botched"]) == pytest.approx(1.0, abs=1e-9)


def test_saa_onset_age_dose_response_recovers_planted_slope(saa_rows):
    rng = np.random.RandomState(0)
    wer_values, onset_ages, ages = [], [], []
    for row in saa_rows:
        onset = float(row["age_english_onset"])
        wer_values.append(max(0.0, 0.05 + 0.01 * onset + rng.normal(0, 0.02)))
        onset_ages.append(onset)
        ages.append(float(row["age"]))

    result = saa.onset_age_dose_response(wer_values, onset_ages, ages)
    assert result["status"] == "ok"
    assert result["beta_onset_age"] > 0.005
    assert result["spearman_wer_vs_onset_age"] > 0.5


def test_l2_arctic_accentedness_uses_true_total_and_phone_count(l2_rows):
    row = next(r for r in l2_rows if r["error_type"] == "substitution")
    acc = l2.accentedness(row)
    assert acc == pytest.approx(float(row["true_total"]) / len(row["g2p_canonical"]))


def test_l2_arctic_phone_grounding_discriminates_peaked_vs_inverted_saliency(l2_rows):
    row = next(r for r in l2_rows if r["error_type"] == "substitution")
    intervals = l2.annotations_for(row["filename"])
    assert intervals, "fixture row must have at least one annotated phone error"

    duration = max(iv["xmax"] for iv in intervals) + 0.5
    n_frames = 200
    dt = duration / n_frames
    rng = np.random.RandomState(0)
    series = rng.uniform(0, 0.1, n_frames)
    for interval in intervals:
        lo = int(interval["xmin"] / dt)
        hi = max(lo + 1, int(np.ceil(interval["xmax"] / dt)))
        series[lo:hi] = 5.0

    speech_mask = np.ones(n_frames, dtype=bool)
    peaked = l2.phone_error_grounding(series.tolist(), duration, intervals, speech_mask)
    inverted = l2.phone_error_grounding((5.0 - series).tolist(), duration, intervals, speech_mask)

    assert peaked["status"] == "ok"
    assert peaked["auroc"] > 0.9
    assert inverted["auroc"] < 0.3


def test_l2_arctic_ols_recovers_planted_intercept_gap():
    rng = np.random.RandomState(0)
    y, accentedness_values, labels = [], [], []
    for label, intercept in [("groupA", 0.10), ("groupB", 0.30)]:
        for _ in range(40):
            a = rng.uniform(0, 1)
            y.append(intercept + 0.1 * a + rng.normal(0, 0.02))
            accentedness_values.append(a)
            labels.append(label)

    result = l2.ols_with_group_dummies(y, accentedness_values, labels)
    assert result["status"] == "ok"
    assert result["group_intercepts"]["groupB"] == pytest.approx(0.20, abs=0.03)


def test_l2_arctic_bundled_subset_is_one_speaker_per_language(l2_rows):
    """Documents the confound the speaker_confounded gate exists to catch:
    this is a corpus property, not a bug -- if it ever stops being true the
    gate should stop firing, and this test should be revisited."""
    by_language = {}
    for row in l2_rows:
        by_language.setdefault(row["native_language"], set()).add(row["speaker_code"])
    assert all(len(speakers) == 1 for speakers in by_language.values())
