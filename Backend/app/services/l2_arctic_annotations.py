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
import re
from collections import Counter, defaultdict
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
            speaker = row.get("speaker", "")
            utt_id = row.get("utt_id", "")
            canonical = row.get("canonical_phone", "")
            perceived = row.get("perceived_phone", "")
            error_type = row.get("error_type", "unknown")
            # Dedup key: the sub_/add_/del_ filename prefixes are a balanced
            # SAMPLING STRATUM over the *same* underlying recordings (verified
            # by md5: sub_0_ABA.wav and add_0_ABA.wav are byte-identical), not
            # distinct utterances, so `filename` alone over-counts intervals
            # by ~65%. This key is what phone_confusion_matrix() dedups on.
            interval_key = f"{speaker}|{utt_id}|{xmin}|{xmax}|{canonical}|{perceived}|{error_type}"
            by_file.setdefault(row["filename"], []).append({
                "xmin": xmin, "xmax": xmax,
                "canonical_phone": canonical, "perceived_phone": perceived,
                "error_type": error_type,
                "speaker": speaker, "utt_id": utt_id, "interval_key": interval_key,
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


# ---------------------------------------------------------------------------
# Phone confusion matrix (canonical_phone -> perceived_phone) per L1 group,
# optionally weighted by the model's saliency inside each annotated interval.
#
# Both axes are human annotations -- there is no phoneme recogniser anywhere
# in this system (ASR adapters decode straight to text and discard offsets),
# so a genuine model-side "perceived phone" cannot be derived. The model
# signal here is instead "how much does the model's saliency attend to this
# annotated interval", via interval_saliency() below.
#
# Measured on the bundled CSV: only 4-5 confusion PAIRS occur in all three L1
# groups (Z->S is the one with usable count; the rest are n<=6), so per-cell
# cross-group comparison is not statistically viable. phone_confusion_matrix()
# therefore also reports `shared_pairs` (the pairs that ARE comparable) and
# `attribution_by_phone_class` (6 broad classes x 3 groups), which is where
# any real cross-group claim should be read from.
# ---------------------------------------------------------------------------

_STRESS_RE = re.compile(r"[0-9]+$")


def normalize_phone(phone: str, strip_stress: bool = True) -> str:
    """AA1 -> AA when strip_stress. Sentinels `sil`/`err` and the `*`
    distortion-marker suffix (L2-ARCTIC's "approximate/distorted" flag, e.g.
    `HH*`, `R*`) pass through untouched -- they are meaningful categories,
    not noise to normalize away."""
    phone = (phone or "").strip()
    if not strip_stress:
        return phone
    return _STRESS_RE.sub("", phone)


# ARPABET (stress-stripped) -> broad phone class. Sentinels are handled
# separately in _phone_class() below, not listed here.
_PHONE_CLASS: dict[str, str] = {
    # vowels
    "AA": "vowel", "AE": "vowel", "AH": "vowel", "AO": "vowel", "AW": "vowel",
    "AX": "vowel", "AY": "vowel", "EH": "vowel", "ER": "vowel", "EY": "vowel",
    "IH": "vowel", "IY": "vowel", "OW": "vowel", "OY": "vowel", "UH": "vowel",
    "UW": "vowel",
    # stops
    "B": "stop", "D": "stop", "G": "stop", "K": "stop", "P": "stop", "T": "stop",
    # affricates
    "CH": "affricate", "JH": "affricate",
    # fricatives
    "DH": "fricative", "F": "fricative", "HH": "fricative", "S": "fricative",
    "SH": "fricative", "TH": "fricative", "V": "fricative", "Z": "fricative",
    "ZH": "fricative",
    # nasals
    "M": "nasal", "N": "nasal", "NG": "nasal",
    # approximants / liquids / glides
    "L": "approximant", "R": "approximant", "W": "approximant", "Y": "approximant",
}


def _phone_class(phone: str) -> str:
    if phone == "sil":
        return "silence"
    if phone == "err":
        return "unintelligible"
    base = phone.rstrip("*")
    return _PHONE_CLASS.get(base, "unintelligible")


def interval_saliency(
    series: list[float], total_duration: float, intervals: list[dict[str, Any]],
    speech_mask: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Per annotated interval: mean |saliency| inside [xmin, xmax] relative to
    the mean over speech-masked frames (falls back to all frames when no
    speech_mask). The ratio, not the raw mean, is what makes values
    comparable across items and saliency methods -- same reasoning as
    `auroc_within_speech` above. Same frame-index arithmetic as
    phone_error_grounding() so both discretise identically."""
    s = np.abs(np.asarray(series, dtype=float))
    T = len(s)
    if T == 0 or not intervals or total_duration <= 0:
        return []

    dt = total_duration / T
    baseline_frames = s[speech_mask] if speech_mask is not None and speech_mask.any() else s
    baseline = float(baseline_frames.mean()) if len(baseline_frames) else 0.0

    records: list[dict[str, Any]] = []
    for interval in intervals:
        lo = max(0, int(interval["xmin"] / dt))
        hi = max(lo + 1, min(T, int(np.ceil(interval["xmax"] / dt))))
        segment = s[lo:hi]
        mean_val = float(segment.mean()) if len(segment) else None
        ratio = (mean_val / baseline) if (mean_val is not None and baseline > 1e-12) else None
        records.append({
            "interval_key": interval.get("interval_key"),
            "canonical_phone": interval["canonical_phone"],
            "perceived_phone": interval["perceived_phone"],
            "error_type": interval["error_type"],
            "saliency_ratio": ratio,
            "n_frames": hi - lo,
        })
    return records


def phone_confusion_matrix(
    entries: list[dict[str, Any]],
    saliency_by_interval: dict[str, list[float]] | None = None,
    strip_stress: bool = True,
) -> dict[str, Any]:
    """Builds the per-L1-group canonical->perceived phone confusion matrix
    from the bundled human annotations.

    entries: one dict per indexed item, `{"filename": str, "group_label": str}`.
    saliency_by_interval: `{interval_key: [saliency_ratio, ...]}`, typically
    collected from interval_saliency() records across the explain sample --
    absent keys simply yield n_saliency=0 on that cell (the explain sample is
    smaller than the full indexed set).
    """
    saliency_by_interval = saliency_by_interval or {}
    annotations = _load_annotations()
    if not annotations:
        return {"status": "unavailable"}

    seen_keys: set[str] = set()
    # cell key -> accumulated stats
    cells: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    canonical_marginals: dict[str, Counter] = defaultdict(Counter)
    perceived_marginals: dict[str, Counter] = defaultdict(Counter)
    groups_seen: set[str] = set()
    n_intervals_raw = 0

    for entry in entries:
        filename = entry["filename"]
        group_label = entry["group_label"]
        for interval in annotations.get(filename, []):
            n_intervals_raw += 1
            key = interval["interval_key"]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            groups_seen.add(group_label)

            canonical = normalize_phone(interval["canonical_phone"], strip_stress)
            perceived = normalize_phone(interval["perceived_phone"], strip_stress)
            error_type = interval["error_type"]
            cell_key = (canonical, perceived, group_label, error_type)
            cell = cells.setdefault(cell_key, {
                "canonical": canonical, "perceived": perceived, "group": group_label,
                "error_type": error_type, "n": 0, "_ratios": [],
            })
            cell["n"] += 1
            ratios = saliency_by_interval.get(key, [])
            cell["_ratios"].extend(r for r in ratios if r is not None)

            canonical_marginals[group_label][canonical] += 1
            perceived_marginals[group_label][perceived] += 1

    if not cells:
        return {"status": "unavailable"}

    cell_list = []
    for cell in cells.values():
        ratios = cell.pop("_ratios")
        cell["mean_saliency_ratio"] = float(np.mean(ratios)) if ratios else None
        cell["n_saliency"] = len(ratios)
        cell_list.append(cell)

    # shared_pairs: canonical->perceived pairs (ignoring error_type/group) that
    # occur in >=2 groups -- the only cells with enough cross-group support to
    # compare directly (measured: only 4-5 pairs hit all 3 groups).
    pair_groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for cell in cell_list:
        pair_groups[(cell["canonical"], cell["perceived"])][cell["group"]].append(cell)

    shared_pairs = []
    for (canonical, perceived), by_group in pair_groups.items():
        if len(by_group) < 2:
            continue
        n_by_group = {g: sum(c["n"] for c in cs) for g, cs in by_group.items()}
        ratio_by_group: dict[str, float | None] = {}
        for g, cs in by_group.items():
            ratios = [c["mean_saliency_ratio"] for c in cs if c["mean_saliency_ratio"] is not None]
            ratio_by_group[g] = float(np.mean(ratios)) if ratios else None
        shared_pairs.append({
            "canonical": canonical, "perceived": perceived, "n_groups": len(by_group),
            "n_by_group": n_by_group, "mean_saliency_ratio_by_group": ratio_by_group,
        })
    shared_pairs.sort(key=lambda p: (-p["n_groups"], -sum(p["n_by_group"].values())))

    # attribution_by_phone_class: the statistically viable cross-group
    # comparison -- 6 broad classes x N groups instead of near-empty cells.
    class_ratios: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    class_n: dict[str, Counter] = defaultdict(Counter)
    class_n_saliency: dict[str, Counter] = defaultdict(Counter)
    for cell in cell_list:
        phone_class = _phone_class(cell["canonical"])
        group = cell["group"]
        class_n[group][phone_class] += cell["n"]
        if cell["mean_saliency_ratio"] is not None:
            class_ratios[group][phone_class].append(cell["mean_saliency_ratio"])
            class_n_saliency[group][phone_class] += cell["n_saliency"]

    attribution_by_phone_class: dict[str, dict[str, Any]] = {}
    for group in groups_seen:
        by_class: dict[str, Any] = {}
        classes = set(class_n[group]) | set(class_ratios[group])
        for phone_class in classes:
            ratios = class_ratios[group].get(phone_class, [])
            by_class[phone_class] = {
                "mean_saliency_ratio": float(np.mean(ratios)) if ratios else None,
                "n": class_n[group].get(phone_class, 0),
                "n_saliency": class_n_saliency[group].get(phone_class, 0),
            }
        attribution_by_phone_class[group] = by_class

    return {
        "status": "ok",
        "strip_stress": strip_stress,
        "n_intervals": len(seen_keys),
        "n_intervals_raw": n_intervals_raw,
        "groups": sorted(groups_seen),
        "cells": cell_list,
        "canonical_marginals": {g: dict(c) for g, c in canonical_marginals.items()},
        "perceived_marginals": {g: dict(c) for g, c in perceived_marginals.items()},
        "shared_pairs": shared_pairs,
        "attribution_by_phone_class": attribution_by_phone_class,
        "caveat": (
            "This bundled subset has exactly 1 speaker per native_language; "
            "cross-group differences are NOT separable from a speaker-identity "
            "effect. Per-cell counts are small outside a handful of pairs "
            "(see `shared_pairs` for the cells with cross-group support)."
        ),
    }
