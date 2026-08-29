"""FR-10: grouped performance metrics, representational comparison, and
disparity inference (docs/FR10plan.md Part 1 S4).

Worker-only: imports numpy/scikit-learn/jiwer at module level, so this module
must never be imported from the FastAPI control plane (see
app/services/fairness_service.py's module docstring for the split rationale).

Confidence intervals use the PERCENTILE bootstrap method (2.5/97.5 quantiles
of the resampled statistic), not full BCa -- BCa's bias-correction and
acceleration terms need a jackknife pass this module does not implement.
Percentile intervals are still valid under the standard bootstrap regularity
conditions and are what is actually computed here; call sites and the result
schema should not claim BCa.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    recall_score,
    roc_auc_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize as l2_normalize

from app.services.sensitivity_metrics_service import asr_metrics

# Two-sided z_(1-alpha/2) + z_power for alpha=0.05, power=0.80.
_MDE_Z = 1.9600 + 0.8416
MIN_SHARED_CONTENT_FOR_PAIRING = 5


# ---------------------------------------------------------------------------
# Per-item metric container -- the generic unit every estimator operates on,
# so the SAME bootstrap/pairing code serves WER, accuracy, grounding_lift,
# attribution_entropy, etc. without per-metric special-casing.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ItemMetric:
    item_id: str
    speaker_id: str | None
    content_id: str | None
    value: float


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


# ---------------------------------------------------------------------------
# Grouped performance metrics (Part 1 S4.2)
# ---------------------------------------------------------------------------

def asr_group_metrics(items: list[dict[str, Any]], language: str | None = "en") -> dict[str, Any]:
    """items: [{item_id, speaker_id, content_id, hypothesis, reference}]"""
    per_item = []
    for it in items:
        metrics = asr_metrics(it.get("hypothesis") or "", it.get("reference") or "", language)
        per_item.append({
            "item_id": it["item_id"], "speaker_id": it.get("speaker_id"),
            "content_id": it.get("content_id"), **metrics,
        })
    edits = sum(m["substitutions"] + m["deletions"] + m["insertions"] for m in per_item)
    ref_words = sum(m["ref_words"] for m in per_item)
    speakers = {m["speaker_id"] for m in per_item if m["speaker_id"]}
    return {
        "n_items": len(items), "n_speakers": len(speakers),
        "micro_wer": edits / max(ref_words, 1),
        "macro_wer": _safe_mean([m["wer"] for m in per_item]),
        "micro_cer": (sum(m["cer"] * m["ref_words"] for m in per_item) / max(ref_words, 1)),
        "macro_cer": _safe_mean([m["cer"] for m in per_item]),
        "sub_rate": sum(m["substitutions"] for m in per_item) / max(ref_words, 1),
        "del_rate": sum(m["deletions"] for m in per_item) / max(ref_words, 1),
        "ins_rate": sum(m["insertions"] for m in per_item) / max(ref_words, 1),
        "insertion_ratio": _safe_mean([m["insertion_ratio"] for m in per_item]),
        "per_item": per_item,
    }


def classification_group_metrics(items: list[dict[str, Any]], all_labels: list[str]) -> dict[str, Any]:
    """items: [{item_id, speaker_id, content_id, predicted_label, reference_label, confidence}]"""
    y_true = [it["reference_label"] for it in items]
    y_pred = [it["predicted_label"] for it in items]
    confidences = [float(it.get("confidence", 0.0)) for it in items]
    correct = [t == p for t, p in zip(y_true, y_pred)]
    return {
        "n_items": len(items),
        "n_speakers": len({it.get("speaker_id") for it in items if it.get("speaker_id")}),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=all_labels, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=all_labels, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(set(y_true)) > 1 else float(accuracy_score(y_true, y_pred)),
        "ece": expected_calibration_error(confidences, correct, n_bins=10),
        "mean_confidence": _safe_mean(confidences),
        "confusion": confusion_matrix(y_true, y_pred, labels=all_labels).tolist(),
        "labels": all_labels,
        "per_item": [
            {"item_id": it["item_id"], "speaker_id": it.get("speaker_id"), "content_id": it.get("content_id"),
             "correct": bool(t == p), "confidence": c}
            for it, t, p, c in zip(items, y_true, y_pred, confidences)
        ],
    }


def expected_calibration_error(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Representational comparison (Part 1 S4.3)
# ---------------------------------------------------------------------------

def separability(vectors: list[list[float]], group_labels: list[str],
                  speaker_ids: list[str | None], *, seed: int = 0, n_permutations: int = 200) -> dict[str, Any]:
    if len(vectors) < len(set(group_labels)) * 2 or len(set(group_labels)) < 2:
        return {"status": "insufficient_data"}
    X = l2_normalize(np.asarray(vectors, dtype=float))
    labels = np.asarray(group_labels)
    try:
        overall = float(silhouette_score(X, labels, metric="cosine"))
        per_sample = silhouette_samples(X, labels, metric="cosine")
    except ValueError:
        return {"status": "insufficient_data"}

    per_group = {g: float(per_sample[labels == g].mean()) for g in sorted(set(group_labels))}

    rng = np.random.default_rng(seed)
    null_scores = np.empty(n_permutations)
    shuffled = labels.copy()
    for k in range(n_permutations):
        rng.shuffle(shuffled)
        try:
            null_scores[k] = silhouette_score(X, shuffled, metric="cosine")
        except ValueError:
            null_scores[k] = np.nan
    null_scores = null_scores[~np.isnan(null_scores)]
    null_mean = float(null_scores.mean()) if len(null_scores) else 0.0
    null_sd = float(null_scores.std()) if len(null_scores) else 1e-9

    out: dict[str, Any] = {
        "status": "ok",
        "silhouette_by_group_label": overall,
        "per_group_silhouette": per_group,
        "permutation_null": {"mean": null_mean, "sd": null_sd, "n": int(len(null_scores))},
        "silhouette_z": float((overall - null_mean) / max(null_sd, 1e-9)),
    }
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        out["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except ValueError:
        pass

    speaker_arr = [s if s else f"__item_{i}" for i, s in enumerate(speaker_ids)]
    if len(set(speaker_arr)) >= 2:
        try:
            out["silhouette_by_speaker"] = float(silhouette_score(X, np.asarray(speaker_arr), metric="cosine"))
        except ValueError:
            out["silhouette_by_speaker"] = None
    else:
        out["silhouette_by_speaker"] = None
    return out


def group_leakage(vectors: list[list[float]], group_labels: list[str],
                   speaker_ids: list[str | None], *, seed: int = 0) -> dict[str, Any]:
    """Balanced-accuracy of a speaker-disjoint k-NN classifier predicting
    group from embedding. StratifiedGroupKFold(groups=speaker) so the split
    never lets the same speaker appear in both train and test -- otherwise
    the score measures speaker identification, not accent (Part 1 S4.3)."""
    labels = np.asarray(group_labels)
    n_groups = len(set(group_labels))
    speakers = [s if s else f"__item_{i}" for i, s in enumerate(speaker_ids)]
    speakers_per_group: dict[str, set[str]] = defaultdict(set)
    for g, s in zip(group_labels, speakers):
        speakers_per_group[g].add(s)
    min_speakers = min(len(v) for v in speakers_per_group.values())
    if min_speakers < 2:
        return {"status": "skipped", "reason": "too_few_speakers_for_disjoint_cv"}

    n_splits = max(2, min(5, min_speakers))
    X = l2_normalize(np.asarray(vectors, dtype=float))
    try:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_val_score(
            KNeighborsClassifier(n_neighbors=min(5, min_speakers), metric="cosine"),
            X, labels, groups=np.asarray(speakers), cv=cv, scoring="balanced_accuracy",
        )
    except ValueError as exc:
        return {"status": "skipped", "reason": str(exc)[:200]}

    chance = 1.0 / n_groups
    mean_score = float(scores.mean())
    return {
        "status": "ok", "balanced_accuracy": mean_score, "std": float(scores.std()),
        "chance": chance, "leakage_lift": mean_score / max(chance, 1e-9),
        "n_folds": n_splits, "split": "StratifiedGroupKFold(speaker)",
    }


def representation_interpretation(separability_result: dict[str, Any], has_disparity: bool) -> str:
    if separability_result.get("status") != "ok":
        return "unavailable"
    separable = separability_result.get("silhouette_z", 0.0) > 3.0
    if separable and has_disparity:
        return "representation_linked_disparity"
    if separable and not has_disparity:
        return "representation_only"
    if not separable and has_disparity:
        return "disparity_not_representation_linked"
    return "no_structure_no_disparity"


# ---------------------------------------------------------------------------
# Disparity estimation (Part 1 S4.5)
# ---------------------------------------------------------------------------

def _pair_by_content(ref: list[ItemMetric], other: list[ItemMetric]) -> list[tuple[str, float, float]]:
    ref_by_c: dict[str, list[float]] = defaultdict(list)
    other_by_c: dict[str, list[float]] = defaultdict(list)
    for im in ref:
        if im.content_id:
            ref_by_c[im.content_id].append(im.value)
    for im in other:
        if im.content_id:
            other_by_c[im.content_id].append(im.value)
    shared = sorted(set(ref_by_c) & set(other_by_c))
    return [(c, _safe_mean(ref_by_c[c]), _safe_mean(other_by_c[c])) for c in shared]


def _block_map(items: list[ItemMetric], use_speaker: bool) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for im in items:
        block = im.speaker_id if (use_speaker and im.speaker_id) else im.item_id
        out[block].append(im.value)
    return out


def _percentile_ci(boot: np.ndarray, alpha: float = 0.05) -> list[float]:
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def _two_sided_p(boot: np.ndarray, point: float) -> float:
    centered = boot - boot.mean()
    if point == 0:
        return 1.0
    n = len(boot)
    p = float(np.mean(np.abs(centered) >= abs(point)))
    return float(min(max(p, 1.0 / n), 1.0))


def estimate_gap(
    ref_items: list[ItemMetric], other_items: list[ItemMetric], *,
    metric: str, design: str, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
) -> dict[str, Any]:
    """Returns a Gap dict: point, ci, p_value, method, block_unit, ratio,
    effect_size, mde. See docs/FR10plan.md Part 1 S4.5.

    Estimator selection is content-count-driven, not design-label-driven:
    `matched` designs with a SINGLE shared content id (e.g. SAA's one shared
    reading passage) cannot be paired per-content -- there is only one pair.
    Content is still perfectly controlled in that case (every group reads
    identical text), so the blockwise-by-speaker estimator is fully valid and
    is used instead; pairing is used only when there are enough distinct
    shared content ids to bootstrap over.
    """
    rng_seed = seed
    ref_vals = np.array([i.value for i in ref_items], dtype=float)
    other_vals = np.array([i.value for i in other_items], dtype=float)
    point = float(other_vals.mean() - ref_vals.mean())

    use_paired = False
    pairs: list[tuple[str, float, float]] = []
    if design in ("matched", "partially_matched"):
        pairs = _pair_by_content(ref_items, other_items)
        use_paired = len(pairs) >= MIN_SHARED_CONTENT_FOR_PAIRING

    rng = np.random.default_rng(rng_seed)
    if use_paired:
        diffs = np.array([o - r for _, r, o in pairs])
        point = float(diffs.mean())
        idx_pool = np.arange(len(diffs))
        boot = np.array([diffs[rng.choice(idx_pool, size=len(diffs), replace=True)].mean()
                          for _ in range(n_boot)])
        method, block_unit, n_blocks = "paired_by_content", "content_id", len(pairs)
    else:
        use_speaker = any(i.speaker_id for i in ref_items) and any(i.speaker_id for i in other_items)
        ref_blocks = _block_map(ref_items, use_speaker)
        other_blocks = _block_map(other_items, use_speaker)
        ref_keys, other_keys = list(ref_blocks.keys()), list(other_blocks.keys())
        boot = np.empty(n_boot)
        for k in range(n_boot):
            rk = [ref_keys[i] for i in rng.integers(0, len(ref_keys), size=len(ref_keys))]
            ok = [other_keys[i] for i in rng.integers(0, len(other_keys), size=len(other_keys))]
            rv = np.concatenate([ref_blocks[b] for b in rk])
            ov = np.concatenate([other_blocks[b] for b in ok])
            boot[k] = ov.mean() - rv.mean()
        method = "unpaired"
        block_unit = "speaker" if use_speaker else "utterance"
        n_blocks = min(len(ref_keys), len(other_keys))

    ci = _percentile_ci(boot, alpha)
    p_value = _two_sided_p(boot, point)
    ref_mean = ref_vals.mean() if len(ref_vals) else 0.0
    ratio = float(other_vals.mean() / ref_mean) if ref_mean not in (0.0,) else float("inf")
    pooled_sd = float(np.sqrt((ref_vals.var(ddof=1) if len(ref_vals) > 1 else 0.0) / 2
                               + (other_vals.var(ddof=1) if len(other_vals) > 1 else 0.0) / 2))
    effect_size = point / pooled_sd if pooled_sd > 1e-9 else None

    mde = minimum_detectable_effect(ref_items, other_items, block_unit=block_unit, alpha=alpha)

    return {
        "metric": metric, "point": point, "ci": ci, "p_value": p_value,
        "method": method, "block_unit": block_unit, "n_blocks": n_blocks,
        "ratio": ratio, "effect_size": effect_size, "mde": mde, "n_boot": n_boot,
    }


def minimum_detectable_effect(
    ref_items: list[ItemMetric], other_items: list[ItemMetric], *,
    block_unit: str, alpha: float = 0.05,
) -> float | None:
    """Smallest true gap this corpus could detect at the given alpha and 80%
    power, from the observed block-level variance and block counts. None
    means undefined (fewer than 2 blocks on one side -- e.g. a single-speaker
    group), rendered by the UI as 'undefined: too few independent blocks'."""
    use_speaker = block_unit == "speaker"
    ref_means = [np.mean(v) for v in _block_map(ref_items, use_speaker).values()]
    other_means = [np.mean(v) for v in _block_map(other_items, use_speaker).values()]
    if len(ref_means) < 2 or len(other_means) < 2:
        return None
    se = float(np.sqrt(np.var(ref_means, ddof=1) / len(ref_means) + np.var(other_means, ddof=1) / len(other_means)))
    return round(_MDE_Z * se, 6)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (n - rank) * p_values[i])
        running_max = max(running_max, adj)
        adjusted[i] = running_max
    return adjusted


def disparity_summary(group_values: dict[str, float], group_weights: dict[str, int]) -> dict[str, Any]:
    """Corpus-level disparity statistics for one metric across all groups
    (Part 1 S4.5): max_gap, disparity_ratio, gini, weighted_std."""
    if len(group_values) < 2:
        return {}
    labels = list(group_values.keys())
    values = np.array([group_values[l] for l in labels], dtype=float)
    weights = np.array([max(group_weights.get(l, 1), 1) for l in labels], dtype=float)
    best_idx, worst_idx = int(values.argmin()), int(values.argmax())
    weighted_mean = float(np.average(values, weights=weights))
    weighted_var = float(np.average((values - weighted_mean) ** 2, weights=weights))
    sorted_vals = np.sort(np.clip(values, 0, None))
    n = len(sorted_vals)
    gini = (
        float((2 * np.sum((np.arange(1, n + 1)) * sorted_vals) / (n * sorted_vals.sum())) - (n + 1) / n)
        if sorted_vals.sum() > 0 else 0.0
    )
    return {
        "max_gap": float(values[worst_idx] - values[best_idx]),
        "disparity_ratio": float(values[worst_idx] / values[best_idx]) if values[best_idx] != 0 else float("inf"),
        "gini": gini,
        "weighted_std": float(np.sqrt(weighted_var)),
        "best_group": labels[best_idx], "worst_group": labels[worst_idx],
    }


# ---------------------------------------------------------------------------
# Verdict engine (Part 1 S4.6)
# ---------------------------------------------------------------------------

Severity = Literal["significant", "content_confounded", "underpowered", "speaker_confounded", "ok"]


def verdict_for(
    gap: dict[str, Any], *, group_label: str, reference_label: str, n_speakers: int,
    design: str, min_speakers: int, absolute_threshold: float, ratio_threshold: float,
) -> dict[str, Any]:
    metric = gap["metric"]
    if n_speakers < min_speakers:
        return {
            "group": group_label, "metric": metric, "verdict": "inconclusive",
            "severity": "speaker_confounded",
            "message": (f"{group_label} has {n_speakers} speaker(s); an effect on {metric} cannot "
                        f"be separated from a speaker-identity effect."),
        }

    severity: str = "significant"
    causal_note = ""
    if design == "unmatched":
        severity = "content_confounded"
        causal_note = " Groups did not read matched content; this gap may partly reflect content difficulty."

    p_adj = gap.get("p_adjusted", gap["p_value"])
    practically_meaningful = abs(gap["point"]) >= absolute_threshold or (
        gap["ratio"] != float("inf") and gap["ratio"] >= ratio_threshold
    )
    if p_adj < 0.05 and practically_meaningful:
        ratio_str = f"{gap['ratio']:.2f}x" if gap["ratio"] != float("inf") else "n/a"
        message = (
            f"{group_label} shows {metric} {gap['point']:+.3f} vs {reference_label} "
            f"(95% CI [{gap['ci'][0]:+.3f}, {gap['ci'][1]:+.3f}], p_adj={p_adj:.4f}, {ratio_str}). "
            f"Estimator: {gap['method']}, blocks: {gap['block_unit']} (n={gap['n_blocks']})."
            + causal_note
        )
        return {"group": group_label, "metric": metric, "verdict": "disparity_detected",
                "severity": severity, "message": message}

    if gap.get("mde") is not None and abs(gap["point"]) < gap["mde"]:
        return {
            "group": group_label, "metric": metric, "verdict": "inconclusive", "severity": "underpowered",
            "message": (f"No significant difference in {metric} for {group_label}, but this corpus can "
                        f"only detect gaps >= {gap['mde']:.3f} at this sample size (n_blocks={gap['n_blocks']}). "
                        f"Absence of evidence is not evidence of absence."),
        }
    if gap.get("mde") is None:
        return {
            "group": group_label, "metric": metric, "verdict": "inconclusive", "severity": "underpowered",
            "message": (f"{group_label} has too few independent blocks to estimate a detectable effect "
                        f"for {metric}."),
        }

    return {
        "group": group_label, "metric": metric, "verdict": "no_evidence_of_disparity", "severity": "ok",
        "message": (f"No statistically significant {metric} gap detected for {group_label} vs "
                    f"{reference_label} (p_adj={p_adj:.4f})."),
    }
