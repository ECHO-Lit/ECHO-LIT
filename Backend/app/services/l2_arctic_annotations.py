"""FR-10 S6.2: L2-ARCTIC phone-error-annotation-driven analyses.

The bundled L2_ARCTIC_dataset/l2_phone_error_annotations.csv gives 891
time-boundaried (xmin, xmax, canonical_phone, perceived_phone, error_type)
intervals -- a human-labelled reference attribution mask, which almost no
audio interpretability benchmark has. This module scores a model's saliency
map against that mask (phone_error_grounding), and provides a dose-response
regression of WER on human-annotated "accentedness" (l2_metadata.csv's
true_total column) -- the analysis that answers "does the model penalise this
accent beyond what its own phonetic deviation warrants?" (docs/FR10plan.md
Part 1 S6.2).

Worker-only (numpy/scikit-learn at module level) -- see
app/services/fairness_service.py's docstring for the control-plane split.

CAVEAT the caller must always carry alongside any output of this module: the
bundled L2-ARCTIC subset has exactly ONE speaker per native_language (ABA/
Arabic, LXC/Chinese, EBVS/Spanish). Cross-L1 comparisons built from this
module's per-item numbers are therefore NOT separable from a speaker-identity
effect -- see the `min_speakers_per_group` gate in fairness_service.build_index
and Part 1 S2.3/S6.2's closing caveat.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

_ANNOTATIONS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "L2_ARCTIC_dataset" / "l2_phone_error_annotations.csv"
)


def is_l2_arctic(dataset: str) -> bool:
    return dataset.lower() == "l2-arctic"


@lru_cache(maxsize=1)
def _load_annotations() -> dict[str, list[dict[str, Any]]]:
    if not _ANNOTATIONS_PATH.exists():
        return {}
    by_file: dict[str, list[dict[str, Any]]] = {}
    with _ANNOTATIONS_PATH.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                xmin, xmax = float(row["xmin"]), float(row["xmax"])
            except (KeyError, ValueError):
                continue
            by_file.setdefault(row["filename"], []).append({
                "xmin": xmin, "xmax": xmax,
                "canonical_phone": row.get("canonical_phone", ""),
                "perceived_phone": row.get("perceived_phone", ""),
                "error_type": row.get("error_type", "unknown"),
            })
    return by_file


def annotations_for(filename: str) -> list[dict[str, Any]]:
    return _load_annotations().get(filename, [])


def phone_error_grounding(
    series: list[float], total_duration: float, intervals: list[dict[str, Any]], speech_mask: np.ndarray | None,
) -> dict[str, Any]:
    """docs/FR10plan.md Part 1 S6.2(a): does the model's saliency land on the
    time region a human annotator flagged as a phone-level deviation?
    `auroc_within_speech` is the control that makes this trustworthy -- without
    it, any saliency map that merely prefers speech over silence scores well."""
    s = np.abs(np.asarray(series, dtype=float))
    T = len(s)
    if T == 0 or not intervals or total_duration <= 0:
        return {"status": "unavailable"}

    dt = total_duration / T
    labels = np.zeros(T, dtype=bool)
    for interval in intervals:
        lo = max(0, int(interval["xmin"] / dt))
        hi = max(lo + 1, min(T, int(np.ceil(interval["xmax"] / dt))))
        labels[lo:hi] = True
    if not labels.any() or labels.all():
        return {"status": "degenerate"}

    inside_fraction = float(labels.mean())
    total_mass = float(s.sum())
    p = s / total_mass if total_mass > 0 else s

    result: dict[str, Any] = {
        "status": "ok",
        "grounding_score": float(p[labels].sum() / max(inside_fraction, 1e-9)),
        "auroc": float(roc_auc_score(labels, s)),
        "average_precision": float(average_precision_score(labels, s)),
        "n_intervals": len(intervals), "annotated_fraction": inside_fraction,
        "auroc_within_speech": None,
    }
    if speech_mask is not None:
        within_labels, within_scores = labels[speech_mask], s[speech_mask]
        if len(set(within_labels.tolist())) > 1:
            result["auroc_within_speech"] = float(roc_auc_score(within_labels, within_scores))
    return result


def approx_phone_count(g2p_canonical: str | None) -> int:
    """Character count of the IPA canonical transcription as a phone-count
    proxy -- NOT a true phoneme tokenisation (some IPA symbols are
    multi-character), but monotonic in utterance length, which is enough for
    the dose-response ranking in accentedness() below."""
    return len((g2p_canonical or "").strip())


def accentedness(covariates: dict[str, Any]) -> float | None:
    """true_total (human-annotated phone deviations for this utterance)
    divided by an approximate canonical phone count -- a continuous,
    human-annotated, model-independent measure of how far the utterance
    departs from canonical English pronunciation (docs/FR10plan.md Part 1
    S6.2(d)). None if the required columns are missing or unparsable."""
    try:
        true_total = float(covariates.get("true_total", ""))
    except (TypeError, ValueError):
        return None
    phone_count = approx_phone_count(covariates.get("g2p_canonical"))
    if phone_count <= 0:
        return None
    return true_total / phone_count


def ols_with_group_dummies(
    y: list[float], accentedness_values: list[float], group_labels: list[str],
) -> dict[str, Any]:
    """WER ~ accentedness + group dummies (first group alphabetically = the
    baseline), fit by ordinary least squares via numpy.linalg.lstsq (no
    statsmodels dependency). A non-zero group intercept at fixed accentedness
    is the strongest available "equal-accentedness disparity" claim -- but
    see the module-level CAVEAT: with this corpus's 1-speaker-per-L1 subset,
    a group intercept is NOT separable from a speaker-identity effect. This
    function is deliberately descriptive, not a significance test."""
    labels = sorted(set(group_labels))
    n = len(y)
    if n < len(labels) + 2:
        return {"status": "insufficient_data", "n": n}

    n_dummies = len(labels) - 1
    X = np.ones((n, 2 + n_dummies))
    X[:, 1] = accentedness_values
    for j, label in enumerate(labels[1:]):
        X[:, 2 + j] = [1.0 if g == label else 0.0 for g in group_labels]
    y_arr = np.asarray(y, dtype=float)
    coefs, _, _, _ = np.linalg.lstsq(X, y_arr, rcond=None)
    y_hat = X @ coefs
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "status": "ok", "n": n, "baseline_group": labels[0],
        "intercept": float(coefs[0]), "beta_accentedness": float(coefs[1]),
        "group_intercepts": {label: float(coefs[2 + j]) for j, label in enumerate(labels[1:])},
        "r2": r2,
    }
