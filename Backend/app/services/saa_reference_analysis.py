"""FR-10 S6.1: Speech Accent Archive (SAA)-specific analyses.

Every one of SAA's 150 speakers reads the IDENTICAL elicitation paragraph
("Please call Stella...") -- the property Part 2's voice-conversion roadmap
spends a whole pipeline trying to synthesise, except here it's real human
speech with zero synthesis artifacts. That fixed reference lets two things
happen that are not possible on a variable-content corpus:

1. A reference-POSITION word error heatmap: since every hypothesis aligns to
   the SAME canonical word sequence, per-group error rate can be reported per
   word position, not just as one aggregate WER number.
2. An age-of-English-onset dose-response regression: age_english_onset is a
   continuous covariate (0-53 in the bundled subset), so WER can be regressed
   on it directly instead of forcing a categorical L1 comparison.

Worker-only (numpy/jiwer at module level) -- see
app/services/fairness_service.py's docstring for the control-plane split.
"""

from __future__ import annotations

import re
from typing import Any

import jiwer
import numpy as np


def is_saa(dataset: str) -> bool:
    return dataset.lower() == "saa"


def canonical_words(reference_text: str) -> list[str]:
    return [w for w in re.sub(r"[^\w\s']", " ", (reference_text or "").lower()).split() if w]


def reference_position_errors(hypothesis: str, reference_text: str) -> list[int]:
    """Per reference-word-position error code: 0=hit, 1=substitution,
    2=deletion. Insertions have no canonical position and are not attributed
    to a slot (docs/FR10plan.md Part 1 S6.1(b))."""
    words = canonical_words(reference_text)
    n = len(words)
    codes = [0] * n
    if n == 0:
        return codes
    try:
        output = jiwer.process_words(" ".join(words), (hypothesis or "").lower())
    except ValueError:
        return codes
    if not output.alignments:
        return codes
    for chunk in output.alignments[0]:
        if chunk.type == "substitute":
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                if i < n:
                    codes[i] = 1
        elif chunk.type == "delete":
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                if i < n:
                    codes[i] = 2
    return codes


def position_error_heatmap(per_group_hypotheses: dict[str, list[str]], reference_text: str) -> dict[str, Any] | None:
    """[group -> per-reference-word-position error rate], averaged over that
    group's speakers. Converts "group X has higher WER" into "the model
    systematically mishandles reference words 7, 12, 19 for group X"."""
    words = canonical_words(reference_text)
    if not words:
        return None
    heatmap: dict[str, list[float]] = {}
    for label, hypotheses in per_group_hypotheses.items():
        if not hypotheses:
            continue
        codes = np.array([reference_position_errors(h, reference_text) for h in hypotheses])
        heatmap[label] = (codes > 0).mean(axis=0).tolist()
    if not heatmap:
        return None
    return {"words": words, "error_rate_by_group": heatmap}


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks


def onset_age_dose_response(wer_values: list[float], onset_ages: list[float], ages: list[float]) -> dict[str, Any]:
    """WER ~ beta0 + beta1*age_english_onset + beta2*chronological_age, OLS
    via numpy.linalg.lstsq, plus the Spearman correlation between WER and
    onset age (docs/FR10plan.md Part 1 S6.1(c)). Descriptive, not a
    significance test -- report alongside `n` and let the reader judge power."""
    n = len(wer_values)
    if n < 5:
        return {"status": "insufficient_data", "n": n}
    y = np.asarray(wer_values, dtype=float)
    X = np.column_stack([np.ones(n), onset_ages, ages])
    coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coefs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    rank_onset, rank_wer = _rank(np.asarray(onset_ages, dtype=float)), _rank(y)
    spearman = float(np.corrcoef(rank_onset, rank_wer)[0, 1]) if n > 1 else None

    return {
        "status": "ok", "n": n, "intercept": float(coefs[0]),
        "beta_onset_age": float(coefs[1]), "beta_chronological_age": float(coefs[2]),
        "r2": r2, "spearman_wer_vs_onset_age": spearman,
    }
