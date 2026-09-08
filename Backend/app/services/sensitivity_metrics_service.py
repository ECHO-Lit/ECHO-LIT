"""FR-7 evaluation metrics: WER/CER for ASR, confidence/JS-divergence deltas
for classification, and set-level Macro-F1/recall (docs/FR7plan.md Part 1 S4).
"""

from __future__ import annotations

from typing import Any

import jiwer
import numpy as np
from sklearn.metrics import f1_score, recall_score


def get_normalizer(language: str | None):
    """Whisper's own text normaliser, so WER/CER are comparable with published
    Whisper numbers instead of penalising punctuation/casing differences."""
    from transformers.models.whisper.english_normalizer import (
        BasicTextNormalizer,
        EnglishTextNormalizer,
    )

    return EnglishTextNormalizer({}) if (language or "en").startswith("en") else BasicTextNormalizer()


def asr_metrics(hypothesis: str, reference: str, language: str | None = "en") -> dict[str, Any]:
    """WER/CER between a variant's transcript and a reference transcript.

    The caller decides what "reference" means: the dataset ground truth
    (WER_gt) or the baseline (theta_0) transcript (WER_self). Both use this
    same function; only the argument differs.
    """
    normalize = get_normalizer(language)
    hyp, ref = normalize(hypothesis or ""), normalize(reference or "")
    if not ref:
        return {
            "wer": 1.0 if hyp else 0.0,
            "cer": 1.0 if hyp else 0.0,
            "degenerate_reference": True,
        }
    words = jiwer.process_words(ref, hyp)
    chars = jiwer.process_characters(ref, hyp)
    ref_word_count = len(ref.split())
    return {
        "wer": min(words.wer, 1.0),
        "cer": min(chars.cer, 1.0),
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "hits": words.hits,
        "ref_words": ref_word_count,
        "hyp_words": len(hyp.split()),
        # Insertion-heavy degradation with few deletions is the hallucination
        # signature (cross-reference FR-8); surfaced here so FR-7 and FR-8
        # agree on the same underlying counts.
        "insertion_ratio": words.insertions / max(ref_word_count, 1),
        "degenerate_reference": False,
    }


def classification_metrics(posterior: dict[str, float], baseline: dict[str, float]) -> dict[str, Any]:
    """Per-item classification deltas: confidence shift, label flip, and
    Jensen-Shannon divergence between the perturbed and baseline posteriors."""
    labels = sorted(set(posterior) | set(baseline))
    p = np.array([posterior.get(k, 0.0) for k in labels]) + 1e-12
    q = np.array([baseline.get(k, 0.0) for k in labels]) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    # JS divergence in nats, bounded [0, ln2]; symmetric and finite, unlike
    # KL, which explodes when the perturbed posterior zeroes a baseline class.
    js = 0.5 * float(np.sum(p * np.log(p / m))) + 0.5 * float(np.sum(q * np.log(q / m)))
    top_idx, base_idx = int(np.argmax(p)), int(np.argmax(q))
    top, base_top = labels[top_idx], labels[base_idx]
    return {
        "predicted_label": top,
        "confidence": float(np.max(p)),
        "baseline_label": base_top,
        "confidence_delta": float(np.max(p) - np.max(q)),
        "label_flipped": int(top != base_top),
        "js_divergence": js,
        "true_class_confidence_delta": float(p[base_idx] - q[base_idx]),
    }


def set_level_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Macro-F1 and macro-recall across the AUDIO SET at one grid point.

    Undefined for a single file -- the caller must gate on len(y_true) >= 2
    (see docs/FR7plan.md Part 1 S4.3: the SRS's single-audio_id input makes
    this an optional superset, not the primary FR-7 signal).
    """
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(np.mean([a == b for a, b in zip(y_true, y_pred)])) if y_true else 0.0,
    }
