"""Wiring between a real model and the pure faithfulness evaluator.

`faithfulness_service` deliberately knows nothing about audio or torch: it asks
a `score_fn` what the model does when given time spans are masked.  This module
is what turns a model into such a callable -- it owns audio loading, masking,
and the alignment between a saliency timeline and the waveform.

Two alignment rules are load-bearing:

1. **The waveform is cropped to the saliency map's own `total_duration`.**
   `saliency_service` truncates to `MAX_SALIENCY_SECONDS` (less for SHAP), so a
   map covers only the start of a long clip.  Deriving the crop from the map
   rather than re-deriving it from the environment means the two cannot drift
   apart; a mismatch would silently place every mask in the wrong location and
   the whole evaluation would measure nothing.
2. **Masking happens in the waveform, not the feature domain**, so the model
   sees a genuinely altered input through its own front end rather than a
   doctored spectrogram.

The masked region is filled with a very low noise floor rather than silence, for
the reason spelled out in `pertubation_service.apply_time_masking`: a perfectly
silent stretch makes Whisper emit an empty generation for that chunk and crashes
its timestamp extraction. The dither is inaudible and keeps the mel-spectrogram
non-degenerate.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from app.services.faithfulness_service import DEFAULT_RANDOM_REPEATS, evaluate_faithfulness
from app.services.saliency_service import generate_saliency

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# ~ -80 dBFS. Matches `pertubation_service.apply_time_masking`.
MASK_NOISE_FLOOR = 1e-4

# Scorers take a (possibly masked) waveform and return a [0, 1] score; see the
# `score_fn` contract in `faithfulness_service`.
Scorer = Callable[[np.ndarray], float]
# Given the clean cropped waveform, produce that scorer plus a description of
# what it tracks.
ScorerFactory = Callable[[np.ndarray, int], tuple[Scorer, dict[str, Any]]]


def mask_spans(
    audio: np.ndarray,
    spans: list[tuple[float, float]],
    sample_rate: int = SAMPLE_RATE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Replace `spans` (in seconds) with a noise floor, leaving the rest intact."""
    if not spans:
        return audio
    generator = rng or np.random.default_rng(0)
    masked = np.array(audio, dtype=np.float32, copy=True)
    for start, end in spans:
        lo = max(0, int(round(start * sample_rate)))
        hi = min(masked.size, int(round(end * sample_rate)))
        if hi > lo:
            masked[lo:hi] = generator.normal(0.0, MASK_NOISE_FLOOR, hi - lo).astype(np.float32)
    return masked


def run_faithfulness(
    audio_path: str,
    model_id: str,
    parameters: dict[str, Any],
    scorer_factory: ScorerFactory,
    saliency_fn: Callable[[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate a saliency map for one clip and measure whether it is faithful.

    `saliency_fn` overrides how the map is produced. Built-in models go through
    `saliency_service.generate_saliency`, which dispatches on model name and has
    no branch for a custom repository -- the generic adapter passes its own.

    Returns the evaluation merged with the saliency map it was computed over, so
    a single job gives the UI everything it needs to show the map, the curves,
    and the before/after comparison without a second round trip.
    """
    import librosa

    method = parameters.get("method", "gradcam")
    produce = saliency_fn or (lambda path, how: generate_saliency(path, model_id, how))
    saliency = produce(audio_path, method)

    series = saliency.get("series") or []
    segments = saliency.get("segments") or []
    total_duration = float(saliency.get("total_duration") or 0.0)

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if not np.all(np.isfinite(audio)):
        logger.warning("run_faithfulness: non-finite samples in %s; zeroing them", audio_path)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    # Rule 1 above: the map defines the window, not the other way round.
    if total_duration > 0:
        audio = audio[: int(round(total_duration * SAMPLE_RATE))]
    audio = audio.astype(np.float32, copy=False)

    seed = int(parameters.get("seed", 42))
    scorer, target = scorer_factory(audio, SAMPLE_RATE)
    mask_rng = np.random.default_rng(seed)

    def score_fn(spans: list[tuple[float, float]]) -> float:
        return scorer(mask_spans(audio, spans, SAMPLE_RATE, mask_rng))

    evaluation = evaluate_faithfulness(
        series,
        segments,
        total_duration,
        score_fn,
        target=target,
        attribution_source=saliency.get("attribution_source"),
        n_steps=int(parameters.get("n_steps", 9)),
        top_fraction=float(parameters.get("top_fraction", 0.2)),
        seed=seed,
        include_occlusion=bool(parameters.get("include_occlusion", True)),
        random_repeats=int(parameters.get("random_repeats", DEFAULT_RANDOM_REPEATS)),
    )

    return {
        "model": saliency.get("model", model_id),
        "method": method,
        **evaluation,
        # The map under test travels with its verdict: the UI draws both, and a
        # stored result stays interpretable without re-running saliency.
        "saliency": {
            "series": series,
            "segments": segments,
            "total_duration": total_duration,
            "emotion": saliency.get("emotion"),
        },
    }


# ---------------------------------------------------------------------------
# Scorer factories, one per model architecture
# ---------------------------------------------------------------------------


def classification_scorer(model: Any, feature_extractor: Any) -> ScorerFactory:
    """Track the softmax probability of the class predicted on clean audio.

    Already in [0, 1] and directly interpretable: "how much confidence in its own
    answer does the model lose when this audio goes away".
    """
    import torch

    def factory(audio: np.ndarray, sample_rate: int) -> tuple[Scorer, dict[str, Any]]:
        device = next(model.parameters()).device

        def probabilities(waveform: np.ndarray) -> Any:
            inputs = feature_extractor(
                waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True
            )
            values = inputs.input_values.to(device)
            mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None
            with torch.no_grad():
                logits = model(input_values=values, attention_mask=mask).logits
            return torch.nn.functional.softmax(logits, dim=-1)[0]

        clean = probabilities(audio)
        index = int(torch.argmax(clean).item())
        id2label = getattr(model.config, "id2label", {}) or {}
        label = id2label.get(index, str(index))

        def scorer(waveform: np.ndarray) -> float:
            return float(probabilities(waveform)[index].item())

        return scorer, {"kind": "class_prob", "label": label}

    return factory


def _decoder_prompt_candidates(model: Any, processor: Any, features: Any) -> list[list[int]]:
    """Plausible decoder prompts for this Whisper build, best guesses first.

    `generate()` returns only the text tokens in current Transformers -- the
    `<|startoftranscript|><|en|><|transcribe|><|notimestamps|>` prefix it ran with
    is stripped from the output. Teacher forcing without that prefix asks the
    model to continue from a bare word, which is far off-distribution: measured on
    whisper-base it gave a mean token log-prob of -11.4 (the model's own argmax
    wanted to emit `<|startoftranscript|>` mid-sentence) instead of the ~-0.3 a
    correctly prompted pass gives.

    The prefix has to be rebuilt, and every route to it is version-dependent:
    `tokenizer.prefix_tokens` omits the language unless the tokenizer was told
    one, `generation_config.forced_decoder_ids` leaves the language slot `None`,
    and `detect_language` is not present on every build. So candidates are
    generated here and `seq2seq_transcript_scorer` picks between them by
    measurement: the right prompt is the one under which the model's own greedy
    output is most likely.
    """
    tokenizer = processor.tokenizer
    start = model.config.decoder_start_token_id

    def token_id(token: str) -> int | None:
        value = tokenizer.convert_tokens_to_ids(token)
        return value if isinstance(value, int) and value >= 0 else None

    detected = None
    try:
        detected = int(model.detect_language(features)[0, -1])
    except Exception:
        logger.debug("Whisper language detection unavailable", exc_info=True)

    transcribe = token_id("<|transcribe|>")
    no_timestamps = token_id("<|notimestamps|>")
    candidates: list[list[int]] = []

    for language in (detected, token_id("<|en|>")):
        if language is None:
            continue
        built = [start, language]
        if transcribe is not None:
            built.append(transcribe)
        if no_timestamps is not None:
            built.append(no_timestamps)
        candidates.append(built)

    prefix = [int(value) for value in (getattr(tokenizer, "prefix_tokens", None) or [])]
    if prefix:
        candidates.append(prefix)
    candidates.append([start])

    unique: list[list[int]] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def seq2seq_transcript_scorer(model: Any, processor: Any, max_new_tokens: int = 128) -> ScorerFactory:
    """Track how strongly the model still produces its *clean* transcript.

    The score is the geometric-mean per-token probability of that transcript
    under teacher forcing -- `exp` of the mean token log-probability.  Not WER:
    WER is discrete, jumps around under small perturbations, and needs a full
    re-decode for every curve point, where this needs one forward pass and moves
    smoothly.  `exp` puts it on the same [0, 1] scale as a class probability, so
    an ASR AOPC and a classifier AOPC can be read side by side.
    """
    import torch

    def factory(audio: np.ndarray, sample_rate: int) -> tuple[Scorer, dict[str, Any]]:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        def features(waveform: np.ndarray) -> Any:
            return processor(
                waveform, sampling_rate=sample_rate, return_tensors="pt"
            ).input_features.to(device=device, dtype=dtype)

        clean_features = features(audio)
        with torch.no_grad():
            generated = model.generate(clean_features, max_new_tokens=max_new_tokens)

        eos = getattr(processor.tokenizer, "eos_token_id", None)
        produced = generated[0].tolist()
        # Older Transformers returns the special prefix inline; current versions
        # strip it. Filtering to text tokens makes both behave the same.
        text_ids = [token for token in produced if eos is None or token < eos]
        if not text_ids:
            # Nothing was transcribed; there is no target to track.
            return (lambda waveform: 0.0), {"kind": "transcript_logprob", "label": ""}

        transcript = processor.tokenizer.decode(text_ids, skip_special_tokens=True).strip()

        def mean_text_logprob(prompt: list[int], input_features: Any) -> float:
            sequence = torch.tensor([prompt + text_ids], device=device)
            with torch.no_grad():
                logits = model(
                    input_features=input_features, decoder_input_ids=sequence[:, :-1]
                ).logits
            targets = sequence[0, 1:]
            log_probs = torch.log_softmax(logits.float()[0], dim=-1)
            picked = log_probs[torch.arange(targets.numel(), device=device), targets]
            # Only the text is scored. The prefix is forced by construction, so
            # including it would pull every score toward 1.0 and flatten the curves.
            return float(picked[-len(text_ids):].mean().item())

        # Pick the prompt the model itself finds most plausible for its own
        # output. Correctness here is measurable rather than assumed, which is
        # what keeps this working across Transformers versions.
        prompt = max(
            _decoder_prompt_candidates(model, processor, clean_features),
            key=lambda candidate: mean_text_logprob(candidate, clean_features),
        )

        def scorer(waveform: np.ndarray) -> float:
            return float(np.exp(mean_text_logprob(prompt, features(waveform))))

        return scorer, {"kind": "transcript_logprob", "label": transcript}

    return factory


def ctc_transcript_scorer(model: Any, processor: Any) -> ScorerFactory:
    """Track the likelihood of the clean greedy CTC path.

    `exp` of the mean log-softmax along that path, which keeps CTC models on the
    same [0, 1] scale as the other two kinds.
    """
    import torch

    def factory(audio: np.ndarray, sample_rate: int) -> tuple[Scorer, dict[str, Any]]:
        device = next(model.parameters()).device

        def logits_for(waveform: np.ndarray) -> Any:
            inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
            tensors = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}
            with torch.no_grad():
                return model(**tensors).logits[0]

        clean = logits_for(audio)
        path = clean.argmax(dim=-1)
        transcript = ""
        try:
            transcript = processor.batch_decode(path.unsqueeze(0))[0].strip()
        except Exception:
            logger.debug("CTC decode unavailable for the faithfulness target label", exc_info=True)

        def scorer(waveform: np.ndarray) -> float:
            logits = logits_for(waveform)
            # A mask can change the frame count; compare over the overlap only.
            frames = min(logits.shape[0], path.shape[0])
            log_probs = torch.log_softmax(logits[:frames].float(), dim=-1)
            picked = log_probs[torch.arange(frames, device=device), path[:frames]]
            return float(torch.exp(picked.mean()).item())

        return scorer, {"kind": "ctc_logprob", "label": transcript}

    return factory
