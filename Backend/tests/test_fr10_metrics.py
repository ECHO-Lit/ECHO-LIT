"""FR-10 metrics math: macro/micro WER agreement, label-set-explicit macro-F1,
blockwise vs naive bootstrap width, paired-vs-unpaired estimator selection,
Holm family-wise error rate, MDE reporting (docs/FR10plan.md Part 1 S4, S9)."""

from __future__ import annotations

import random

import numpy as np
import pytest
from sklearn.metrics import f1_score

from app.services.fairness_metrics_service import (
    ItemMetric,
    asr_group_metrics,
    estimate_gap,
    holm_bonferroni,
)


def _items(n_speakers: int, per_speaker: int, base: float, noise: float, seed: int,
           content_prefix: str = "content") -> list[ItemMetric]:
    rng = random.Random(seed)
    out = []
    for s in range(n_speakers):
        for u in range(per_speaker):
            out.append(ItemMetric(
                f"spk{s}-u{u}", f"spk{s}", f"{content_prefix}{u}",
                max(0.0, base + rng.gauss(0, noise)),
            ))
    return out


def test_macro_and_micro_wer_agree_on_equal_length_references():
    # Every reference the same length -> micro (pooled) and macro (per-utterance
    # mean) WER must coincide, since weighting by reference length is a no-op.
    items = [
        {"item_id": "a", "speaker_id": "s1", "content_id": "c1",
         "hypothesis": "the quick brown fox", "reference": "the quick brown dog"},
        {"item_id": "b", "speaker_id": "s2", "content_id": "c2",
         "hypothesis": "a lazy cat sleeps", "reference": "a lazy dog sleeps"},
    ]
    result = asr_group_metrics(items, language="en")
    assert result["micro_wer"] == pytest.approx(result["macro_wer"], abs=1e-9)


def test_macro_f1_uses_explicit_label_set_not_just_present_labels():
    # A group missing a class entirely must still be scored against the FULL
    # label set, or two groups' macro-F1 become non-comparable (Part 1 S4.2).
    y_true = ["happy", "happy", "sad"]
    y_pred = ["happy", "sad", "sad"]
    all_labels = ["happy", "sad", "angry"]  # "angry" never appears in this group
    explicit = f1_score(y_true, y_pred, average="macro", labels=all_labels, zero_division=0)
    implicit = f1_score(y_true, y_pred, average="macro", zero_division=0)  # only present labels
    assert explicit != implicit, "test fixture must actually exercise the missing-class case"

    from app.services.fairness_metrics_service import classification_group_metrics
    items = [
        {"item_id": "1", "speaker_id": "s1", "content_id": "c1", "reference_label": "happy", "predicted_label": "happy", "confidence": 0.9},
        {"item_id": "2", "speaker_id": "s1", "content_id": "c2", "reference_label": "happy", "predicted_label": "sad", "confidence": 0.6},
        {"item_id": "3", "speaker_id": "s2", "content_id": "c3", "reference_label": "sad", "predicted_label": "sad", "confidence": 0.8},
    ]
    result = classification_group_metrics(items, all_labels)
    assert result["macro_f1"] == pytest.approx(explicit)
    assert result["labels"] == all_labels


def test_blockwise_bootstrap_ci_is_wider_than_naive_utterance_level():
    """Utterances from one speaker are correlated -- a blockwise (speaker)
    bootstrap must produce a WIDER CI than treating every utterance as an
    independent draw would. We can't call a "naive" estimator directly (the
    module only implements the correct one), so this test constructs the
    naive comparison by hand and asserts the module's CI is at least that wide."""
    rng = np.random.default_rng(0)
    n_speakers, per_speaker = 6, 20
    speaker_offsets = rng.normal(0, 0.08, n_speakers)  # between-speaker variance
    ref, other = [], []
    for s in range(n_speakers):
        for u in range(per_speaker):
            val_ref = max(0.0, 0.15 + speaker_offsets[s] + rng.normal(0, 0.01))
            val_other = max(0.0, 0.15 + speaker_offsets[s] + rng.normal(0, 0.01))  # no true gap
            ref.append(ItemMetric(f"r{s}-{u}", f"s{s}", f"c{u}", val_ref))
            other.append(ItemMetric(f"o{s}-{u}", f"s{s}", f"c{u}", val_other))

    gap = estimate_gap(ref, other, metric="wer", design="unmatched", n_boot=2000, seed=0)
    blockwise_width = gap["ci"][1] - gap["ci"][0]

    # Naive: pool ALL utterances as if independent (ignores the shared
    # per-speaker offset that correlates utterances within a speaker).
    naive_boot = np.empty(2000)
    ref_vals = np.array([i.value for i in ref])
    other_vals = np.array([i.value for i in other])
    rng2 = np.random.default_rng(1)
    for k in range(2000):
        naive_boot[k] = (
            rng2.choice(other_vals, size=len(other_vals), replace=True).mean()
            - rng2.choice(ref_vals, size=len(ref_vals), replace=True).mean()
        )
    naive_width = float(np.percentile(naive_boot, 97.5) - np.percentile(naive_boot, 2.5))

    assert blockwise_width > naive_width, (
        f"blockwise CI ({blockwise_width:.4f}) should be wider than the naive "
        f"utterance-level CI ({naive_width:.4f}) when utterances are speaker-correlated"
    )


def test_paired_estimator_removes_content_difficulty_confound():
    """Synthetic corpus: acoustics are IDENTICAL between groups, but group B
    happens to read harder sentences. The unpaired estimator should show a
    gap (confounded by difficulty); the paired-by-content estimator, which
    matches on content_id, should show ~zero (the true acoustic effect)."""
    rng = np.random.default_rng(0)
    n_content = 20
    difficulty = {f"c{i}": rng.uniform(0, 0.3) for i in range(n_content)}  # per-sentence difficulty

    ref, other = [], []
    for i in range(n_content):
        cid = f"c{i}"
        # Same acoustic noise process on both sides -- no true acoustic effect.
        ref.append(ItemMetric(f"r{i}", f"spkA{i % 3}", cid, difficulty[cid] + rng.normal(0, 0.01)))
        other.append(ItemMetric(f"o{i}", f"spkB{i % 3}", cid, difficulty[cid] + rng.normal(0, 0.01)))
    # Group B disproportionately reads the harder half of the content set.
    hard_ids = sorted(difficulty, key=difficulty.get)[n_content // 2:]
    other_unbalanced = [im for im in other if im.content_id in hard_ids] * 2  # oversample hard content

    unpaired_like = estimate_gap(ref, other_unbalanced, metric="wer", design="unmatched", n_boot=1000, seed=0)
    paired = estimate_gap(ref, other, metric="wer", design="matched", n_boot=1000, seed=0)

    assert paired["method"] == "paired_by_content"
    assert abs(paired["point"]) < 0.02, f"paired estimate should be ~0 (no true acoustic gap), got {paired['point']}"
    assert abs(unpaired_like["point"]) > abs(paired["point"]), (
        "unpaired estimate on content-imbalanced groups should show a larger "
        "(difficulty-confounded) gap than the paired estimate"
    )


def test_holm_bonferroni_controls_family_wise_error_under_simulation():
    """1000 simulated families of 24 null comparisons (no true effect). At
    alpha=0.05, the fraction of families with at least one falsely 'significant'
    Holm-adjusted p-value should stay near or below 5%, unlike raw p-values,
    which would falsely flag far more often."""
    rng = np.random.default_rng(0)
    n_families, n_tests, alpha = 1000, 24, 0.05

    raw_false_positive_families = 0
    holm_false_positive_families = 0
    for _ in range(n_families):
        pvals = list(rng.uniform(0, 1, n_tests))  # all null
        if min(pvals) < alpha:
            raw_false_positive_families += 1
        adjusted = holm_bonferroni(pvals)
        if min(adjusted) < alpha:
            holm_false_positive_families += 1

    raw_rate = raw_false_positive_families / n_families
    holm_rate = holm_false_positive_families / n_families
    assert holm_rate <= alpha * 1.5, f"Holm family-wise error rate too high: {holm_rate}"
    assert holm_rate < raw_rate, "Holm correction must reduce false-positive families vs. raw p-values"


def test_mde_reported_for_every_inconclusive_style_comparison():
    ref = _items(20, 3, 0.10, 0.05, seed=1)
    other = _items(20, 3, 0.11, 0.05, seed=2)  # tiny true gap, likely non-significant
    gap = estimate_gap(ref, other, metric="wer", design="unmatched", n_boot=1000, seed=0)
    assert gap["mde"] is not None
    assert gap["mde"] > 0
