"""FR-10 S6: SAA reference-position heatmap / age-onset dose-response and
L2-ARCTIC phone-error grounding / equal-accentedness regression, checked
against the real bundled datasets (docs/FR10plan.md Part 1 S6, S9)."""

import numpy as np
import pytest

from app.services import dataset_service, l2_arctic_annotations as l2, saa_reference_analysis as saa
from app.services.fairness_service import build_index


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


# ---------------------------------------------------------------------------
# Phone confusion matrix (FR-10 S6.2 extension)
# ---------------------------------------------------------------------------

def _all_l2_entries(l2_rows):
    """One entry per row, grouped by native_language -- mirrors what
    _build_dataset_extensions passes in (filename + group label per indexed
    item), covering every row so counts describe the whole corpus."""
    return [{"filename": row["filename"], "group_label": row["native_language"]} for row in l2_rows]


def test_phone_confusion_matrix_dedups_the_sampling_stratum(l2_rows):
    """sub_/add_/del_ filename prefixes are a balanced sampling stratum over
    the SAME underlying recordings (sub_0_ABA.wav and add_0_ABA.wav are
    byte-identical audio) -- annotation rows must be deduped on
    (speaker, utt_id, xmin, xmax, canonical, perceived, error_type), not
    counted once per filename, or the interval count is inflated ~65%."""
    result = l2.phone_confusion_matrix(_all_l2_entries(l2_rows))
    assert result["status"] == "ok"
    assert result["n_intervals"] == 539
    assert result["n_intervals_raw"] == 891


def test_phone_confusion_matrix_per_group_totals(l2_rows):
    result = l2.phone_confusion_matrix(_all_l2_entries(l2_rows))
    totals = {}
    for cell in result["cells"]:
        totals[cell["group"]] = totals.get(cell["group"], 0) + cell["n"]
    assert totals == {"Arabic": 115, "Spanish": 184, "Chinese": 240}


def test_phone_confusion_matrix_stress_stripping_changes_shared_pair_count(l2_rows):
    stripped = l2.phone_confusion_matrix(_all_l2_entries(l2_rows), strip_stress=True)
    raw = l2.phone_confusion_matrix(_all_l2_entries(l2_rows), strip_stress=False)
    n3_stripped = sum(1 for p in stripped["shared_pairs"] if p["n_groups"] == 3)
    n3_raw = sum(1 for p in raw["shared_pairs"] if p["n_groups"] == 3)
    assert n3_stripped == 5
    assert n3_raw == 4


def test_phone_confusion_matrix_z_to_s_is_the_dominant_shared_pair(l2_rows):
    result = l2.phone_confusion_matrix(_all_l2_entries(l2_rows))
    all_three = [p for p in result["shared_pairs"] if p["n_groups"] == 3]
    assert all_three, "at least one pair must occur in all three L1 groups"
    top = max(all_three, key=lambda p: sum(p["n_by_group"].values()))
    assert (top["canonical"], top["perceived"]) == ("Z", "S")
    assert sum(top["n_by_group"].values()) == 65


def test_phone_confusion_matrix_saliency_weighting_flows_through(l2_rows):
    """A synthetic saliency_by_interval keyed by interval_key must land on
    the matching cell's mean_saliency_ratio -- the join key connecting
    interval_saliency() records (built per explain-sample item) to
    phone_confusion_matrix() cells (built per corpus item)."""
    entries = _all_l2_entries(l2_rows)
    baseline = l2.phone_confusion_matrix(entries)
    some_cell = next(c for c in baseline["cells"] if c["n"] >= 1)
    # Recover an interval_key landing in this exact cell by re-reading the
    # annotations for one filename in this group.
    row = next(r for r in l2_rows if r["native_language"] == some_cell["group"])
    intervals = l2.annotations_for(row["filename"])
    target = next(
        (iv for iv in intervals
         if l2.normalize_phone(iv["canonical_phone"]) == some_cell["canonical"]
         and l2.normalize_phone(iv["perceived_phone"]) == some_cell["perceived"]),
        None,
    )
    if target is None:
        pytest.skip("no interval in the fixture row lands in the sampled cell")
    weighted = l2.phone_confusion_matrix(entries, saliency_by_interval={target["interval_key"]: [2.5, 3.5]})
    cell = next(
        c for c in weighted["cells"]
        if c["group"] == some_cell["group"] and c["canonical"] == some_cell["canonical"]
        and c["perceived"] == some_cell["perceived"]
    )
    assert cell["n_saliency"] >= 1
    assert cell["mean_saliency_ratio"] is not None


def test_interval_saliency_ratio_discriminates_peaked_vs_flat(l2_rows):
    row = next(r for r in l2_rows if r["error_type"] == "substitution")
    intervals = l2.annotations_for(row["filename"])
    assert intervals

    duration = max(iv["xmax"] for iv in intervals) + 0.5
    n_frames = 200
    dt = duration / n_frames
    series = np.full(n_frames, 0.1)
    for interval in intervals:
        lo = int(interval["xmin"] / dt)
        hi = max(lo + 1, int(np.ceil(interval["xmax"] / dt)))
        series[lo:hi] = 5.0
    speech_mask = np.ones(n_frames, dtype=bool)

    peaked = l2.interval_saliency(series.tolist(), duration, intervals, speech_mask)
    flat = l2.interval_saliency(np.full(n_frames, 0.1).tolist(), duration, intervals, speech_mask)

    assert all(r["saliency_ratio"] is not None and r["saliency_ratio"] > 1 for r in peaked)
    assert all(r["saliency_ratio"] is not None and abs(r["saliency_ratio"] - 1) < 1e-6 for r in flat)


def test_accentedness_survives_build_index_covariates():
    """Regression test for the covariate-stripping bug: g2p_canonical used to
    be stripped from item.covariates by _RESERVED_COLUMNS (it was only in
    _NON_GROUPABLE, meant to block it as a grouping KEY, not as covariate
    data), which silently zeroed accentedness() for every real request and
    made equal_accentedness_regression always report insufficient_data. This
    must go through build_index()'s actual covariate construction, not a raw
    metadata row, or it passes for the wrong reason."""
    index = build_index("l2-arctic", ["native_language"], None, task="transcription", min_group_size=1,
                         min_speakers_per_group=1)
    item = next(item for group in index.groups for item in group.items)
    assert item.covariates.get("g2p_canonical"), "g2p_canonical must survive the covariate filter"
    assert l2.accentedness(item.covariates) is not None
