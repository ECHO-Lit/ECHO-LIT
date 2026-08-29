"""Adapter-native, calibrated encoder readouts for speech Jacobian lenses.

The original stochastic-Jacobian readout could surface vocabulary tokens with
no evidence that they reflected the model's final representation.  This
implementation instead distils a frozen model's final verbal state into a
regularised linear readout for every encoder layer, then reports held-out
calibration.  Seq2seq targets are aligned to encoder time with the model's own
cross-attention; CTC targets already share an encoder-time axis.
"""

from __future__ import annotations

from typing import Any, Callable


class JacobianLensError(ValueError):
    pass


def _device_of(model: Any):
    return next(model.parameters()).device


def _prepare_audio(processor: Any, audio_path: str, max_audio_seconds: float) -> tuple[dict[str, Any], float]:
    import librosa
    import numpy as np

    sample_rate = int(getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000))
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True, duration=max_audio_seconds)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    batch = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    return dict(batch), float(len(audio)) / sample_rate if sample_rate else 0.0


def _move_to_device(values: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in values.items()}


def _tokenizer(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise JacobianLensError("Model processor does not expose a tokenizer")
    return tokenizer


def _output_projection(model: Any) -> Any:
    projection = getattr(model, "get_output_embeddings", lambda: None)()
    if projection is None:
        projection = getattr(model, "lm_head", None) or getattr(model, "proj_out", None)
    if projection is None or not hasattr(projection, "weight"):
        raise JacobianLensError("Model does not expose a standard vocabulary output projection")
    return projection


def _token_prior(processor: Any, transcripts: list[str], smoothing: float = 1e-5) -> Any:
    """Estimate a log-unigram prior over the vocabulary from the training
    transcripts using the same processor tokenizer that aligns the targets.

    This is the `log P(token)` used to frequency-correct the readout during
    apply: `log P(token | encoder) - log P(token)`.  Tokens never seen in the
    transcripts get a floor set by a Laplace factor so they remain possible
    but never dominate.
    """
    import torch

    tokenizer = _tokenizer(processor)
    counts: dict[int, int] = {}
    total = 0
    for transcript in transcripts:
        ids = tokenizer(transcript, add_special_tokens=False).input_ids
        for token_id in ids:
            counts[token_id] = counts.get(token_id, 0) + 1
            total += 1
    vocab_size = len(tokenizer)
    if total == 0 or vocab_size == 0:
        raise JacobianLensError("Could not estimate a token prior from the transcripts")
    prior = torch.full((vocab_size,), smoothing, dtype=torch.float64)
    for token_id, count in counts.items():
        prior[token_id] = (count + smoothing) / (total + smoothing * vocab_size)
    prior = prior / prior.sum()
    return prior


def _seq2seq_states(model: Any, inputs: dict[str, Any], transcript: str, processor: Any):
    tokenizer = _tokenizer(processor)
    labels = tokenizer(transcript, return_tensors="pt", truncation=True).input_ids.to(_device_of(model))
    if labels.shape[-1] < 1:
        raise JacobianLensError("Transcript produced no tokens")
    outputs = model(
        **inputs,
        labels=labels,
        output_hidden_states=True,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )
    encoder_states = getattr(outputs, "encoder_hidden_states", None)
    decoder_states = getattr(outputs, "decoder_hidden_states", None)
    cross_attentions = getattr(outputs, "cross_attentions", None)
    logits = getattr(outputs, "logits", None)
    if not encoder_states or not decoder_states or not cross_attentions or cross_attentions[-1] is None:
        raise JacobianLensError(
            "Seq2seq model did not return encoder/decoder states and cross-attention needed for a calibrated lens"
        )
    if logits is None:
        raise JacobianLensError("Seq2seq model did not return output logits")
    # Hugging Face includes the pre-transformer embedding state at index zero.
    # A lens is fitted for each actual encoder block, not that input embedding.
    if len(encoder_states) < 2:
        raise JacobianLensError("Seq2seq encoder did not expose transformer-layer states")
    # [batch, heads, decoder tokens, encoder frames].  The last decoder layer
    # is the closest available alignment to the vocabulary projection.
    alignment = cross_attentions[-1].mean(dim=1)
    return tuple(encoder_states[1:]), logits, alignment


def _ctc_states(model: Any, inputs: dict[str, Any]):
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        raise JacobianLensError("CTC model did not return hidden states")
    if len(hidden_states) < 2:
        raise JacobianLensError("CTC encoder did not expose transformer-layer states")
    return tuple(hidden_states[1:]), hidden_states[-1]


def _frame_indices(frame_count: int, maximum: int, device: Any):
    import torch

    count = min(frame_count, maximum)
    if count < 1:
        raise JacobianLensError("Encoder returned no audio frames")
    if count == frame_count:
        return torch.arange(frame_count, device=device)
    return torch.linspace(0, frame_count - 1, steps=count, device=device).round().long()


def _aligned_fit_pairs(
    architecture: str,
    model: Any,
    inputs: dict[str, Any],
    transcript: str,
    processor: Any,
    frame_samples: int,
) -> tuple[list[Any], Any]:
    """Return equally timed encoder representations and frozen verbal targets."""
    if architecture == "seq2seq":
        encoder_states, logits, alignment = _seq2seq_states(model, inputs, transcript, processor)
        indices = _frame_indices(encoder_states[0].shape[1], frame_samples, encoder_states[0].device)
        sources = [state.index_select(1, indices).squeeze(0) for state in encoder_states]
        # Logits are [batch, decoder_tokens, vocab].  Distribute per-decoder-token
        # logits over the encoder frames each token attends to, then renormalise.
        selected_attention = alignment.index_select(2, indices).transpose(1, 2)
        targets = selected_attention @ logits
        targets = targets / selected_attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # Logits live in ℝ^vocab.  The lens predicts logits directly in vocab space.
        return sources, targets.squeeze(0)
    if architecture == "ctc":
        encoder_states, final_states = _ctc_states(model, inputs)
        indices = _frame_indices(encoder_states[0].shape[1], frame_samples, encoder_states[0].device)
        return (
            [state.index_select(1, indices).squeeze(0) for state in encoder_states],
            final_states.index_select(1, indices).squeeze(0),
        )
    raise JacobianLensError(f"Unsupported Jacobian-lens architecture: {architecture}")


def _encoder_states_for_apply(architecture: str, model: Any, inputs: dict[str, Any]):
    if architecture == "ctc":
        states, _ = _ctc_states(model, inputs)
        return states
    if architecture == "seq2seq":
        encoder = getattr(model, "get_encoder", lambda: None)()
        if encoder is None:
            raise JacobianLensError("Seq2seq model does not expose get_encoder()")
        try:
            outputs = encoder(**inputs, output_hidden_states=True, return_dict=True)
        except TypeError:
            primary = next(value for value in inputs.values() if hasattr(value, "ndim"))
            outputs = encoder(primary, output_hidden_states=True, return_dict=True)
        states = getattr(outputs, "hidden_states", None)
        if not states:
            raise JacobianLensError("Seq2seq encoder did not return hidden states")
        if len(states) < 2:
            raise JacobianLensError("Seq2seq encoder did not expose transformer-layer states")
        return tuple(states[1:])
    raise JacobianLensError(f"Unsupported Jacobian-lens architecture: {architecture}")


def fit_encoder_jacobian_lens(
    adapter: Any,
    resource: Any,
    samples: list[tuple[str, str]],
    max_audio_seconds: float,
    frame_samples: int = 32,
    ridge_regularization: float = 1e-3,
    on_sample: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Fit calibrated, teacher-aligned linear encoder readouts.

    The model remains frozen.  A ridge decoder is fitted from every layer's
    encoder frames to the final verbal representation.  Twenty percent of a
    sufficiently large set is held out and never contributes to the weights.
    """
    import torch

    architecture = adapter.jacobian_lens_architecture()
    if not architecture:
        raise JacobianLensError(f"{adapter.model_id} does not support a speech Jacobian lens")
    processor, model = adapter.jacobian_lens_components(resource)
    model.eval()
    if len(samples) >= 10:
        validation_indices = set(range(0, len(samples), 5))
        train_samples = [sample for index, sample in enumerate(samples) if index not in validation_indices]
        validation_samples = [sample for index, sample in enumerate(samples) if index in validation_indices]
    else:
        train_samples, validation_samples = samples, []
    if len(train_samples) < 2:
        raise JacobianLensError("At least two training samples are required after validation split")

    token_prior = _token_prior(processor, [t for _, t in train_samples])
    token_prior_log_probs = token_prior.log().float().cpu()

    statistics: list[dict[str, Any]] | None = None
    projection = _output_projection(model)

    def fit_pairs(audio_path: str, transcript: str) -> tuple[list[Any], Any]:
        inputs, _ = _prepare_audio(processor, audio_path, max_audio_seconds)
        inputs = _move_to_device(inputs, _device_of(model))
        with torch.no_grad():
            return _aligned_fit_pairs(
                architecture, model, inputs, transcript, processor, frame_samples
            )

    total_steps = len(train_samples) + len(validation_samples)
    for index, (audio_path, transcript) in enumerate(train_samples, start=1):
        sources, targets = fit_pairs(audio_path, transcript)
        targets = targets.detach().float().cpu()
        if targets.shape[-1] != projection.weight.shape[-1]:
            raise JacobianLensError("Final verbal state does not match the model vocabulary projection")
        if statistics is None:
            statistics = [
                {
                    "count": 0,
                    "x_sum": torch.zeros(source.shape[-1], dtype=torch.float64),
                    "y_sum": torch.zeros(targets.shape[-1], dtype=torch.float64),
                    "xx": torch.zeros((source.shape[-1], source.shape[-1]), dtype=torch.float64),
                    "xy": torch.zeros((source.shape[-1], targets.shape[-1]), dtype=torch.float64),
                }
                for source in sources
            ]
        if len(sources) != len(statistics):
            raise JacobianLensError("Model returned an inconsistent number of encoder layers")
        for layer, source in enumerate(sources):
            values = source.detach().float().cpu().double()
            targets64 = targets.double()
            stats = statistics[layer]
            stats["count"] += values.shape[0]
            stats["x_sum"].add_(values.sum(dim=0))
            stats["y_sum"].add_(targets64.sum(dim=0))
            stats["xx"].add_(values.T @ values)
            stats["xy"].add_(values.T @ targets64)
        if on_sample:
            on_sample(index, total_steps)

    if not statistics:
        raise JacobianLensError("No lens samples were fitted")
    weights, source_means, target_means = [], [], []
    for stats in statistics:
        count = stats["count"]
        source_mean = stats["x_sum"] / count
        target_mean = stats["y_sum"] / count
        covariance = stats["xx"] - count * torch.outer(source_mean, source_mean)
        cross_covariance = stats["xy"] - count * torch.outer(source_mean, target_mean)
        scale = covariance.diagonal().mean().clamp_min(1e-8)
        system = covariance + torch.eye(covariance.shape[0], dtype=torch.float64) * (ridge_regularization * scale)
        try:
            weight = torch.linalg.solve(system, cross_covariance)
        except RuntimeError as exc:
            raise JacobianLensError("Could not solve the regularised encoder readout") from exc
        weights.append(weight.float())
        source_means.append(source_mean.float())
        target_means.append(target_mean.float())

    quality = [{"cosine_similarity_sum": 0.0, "top1_agreement_sum": 0.0, "frame_count": 0} for _ in weights]
    for index, (audio_path, transcript) in enumerate(validation_samples, start=len(train_samples) + 1):
        sources, targets = fit_pairs(audio_path, transcript)
        targets = targets.detach().float().cpu()
        for layer, source in enumerate(sources):
            values = source.detach().float().cpu()
            predicted = (values - source_means[layer]) @ weights[layer] + target_means[layer]
            cosine = torch.nn.functional.cosine_similarity(predicted, targets, dim=-1)
            predicted_ids = predicted.argmax(dim=-1)
            teacher_ids = targets.argmax(dim=-1)
            quality[layer]["cosine_similarity_sum"] += float(cosine.sum())
            quality[layer]["top1_agreement_sum"] += float((predicted_ids == teacher_ids).sum())
            quality[layer]["frame_count"] += int(values.shape[0])
        if on_sample:
            on_sample(index, total_steps)

    layer_quality = [
        {
            "layer": layer,
            "validation_frames": values["frame_count"],
            "cosine_similarity": (
                values["cosine_similarity_sum"] / values["frame_count"]
                if values["frame_count"] else None
            ),
            "top1_agreement": (
                values["top1_agreement_sum"] / values["frame_count"]
                if values["frame_count"] else None
            ),
        }
        for layer, values in enumerate(quality)
    ]
    return {
        "format_version": 3,
        "method": "teacher_aligned_logit_readout",
        "architecture": architecture,
        "model_id": adapter.model_id,
        "model_revision": adapter.jacobian_lens_revision(),
        "weights": weights,
        "source_means": source_means,
        "target_means": target_means,
        "token_prior_log_probs": token_prior_log_probs,
        "sample_count": len(samples),
        "training_sample_count": len(train_samples),
        "validation_sample_count": len(validation_samples),
        "frame_samples": frame_samples,
        "ridge_regularization": ridge_regularization,
        "quality": {"layers": layer_quality},
    }


def _pool_frames(values: Any, max_frames: int) -> list[tuple[Any, int, int]]:
    frame_count = values.shape[0]
    bucket_count = min(frame_count, max_frames)
    return [
        (
            values[start:end].mean(dim=0),
            start,
            end,
        )
        for index in range(bucket_count)
        for start, end in [
            (int(index * frame_count / bucket_count), max(int((index + 1) * frame_count / bucket_count), int(index * frame_count / bucket_count) + 1))
        ]
    ]


def apply_encoder_jacobian_lens(
    adapter: Any,
    resource: Any,
    artifact: dict[str, Any],
    audio_path: str,
    top_k: int,
    max_frames: int,
) -> dict[str, Any]:
    if artifact.get("format_version") not in (2, 3) or artifact.get("method") not in ("teacher_aligned_ridge_readout", "teacher_aligned_logit_readout"):
        raise JacobianLensError("This is a legacy uncalibrated J-Lens. Refit it before interpreting its readout.")
    import torch

    architecture = adapter.jacobian_lens_architecture()
    if artifact.get("model_id") != adapter.model_id or artifact.get("model_revision") != adapter.jacobian_lens_revision():
        raise JacobianLensError("Lens was fitted for a different model revision")
    if artifact.get("architecture") != architecture:
        raise JacobianLensError("Lens architecture does not match this model")
    format_version = artifact.get("format_version", 1)
    processor, model = adapter.jacobian_lens_components(resource)
    inputs, duration = _prepare_audio(processor, audio_path, max_audio_seconds=60.0)
    inputs = _move_to_device(inputs, _device_of(model))
    with torch.no_grad():
        states = _encoder_states_for_apply(architecture, model, inputs)
    weights = artifact["weights"]
    source_means = artifact["source_means"]
    target_means = artifact["target_means"]
    token_prior_log_probs = artifact.get("token_prior_log_probs")
    if token_prior_log_probs is not None:
        token_prior_log_probs = torch.tensor(token_prior_log_probs, dtype=torch.float32)
    if len(states) != len(weights) or len(states) != len(source_means) or len(states) != len(target_means):
        raise JacobianLensError("Lens layer count does not match the loaded model")
    tokenizer = _tokenizer(processor)
    decode = getattr(tokenizer, "decode", None)
    quality_by_layer = {item["layer"]: item for item in artifact.get("quality", {}).get("layers", [])}
    layers = []
    for layer_index, (state, weight, source_mean, target_mean) in enumerate(
        zip(states, weights, source_means, target_means)
    ):
        # Fitted readouts are stored in float32; retain that precision even
        # when the inference model itself runs in fp16/bf16.
        weight = weight.to(_device_of(model), dtype=torch.float32)
        source_mean = source_mean.to(_device_of(model), dtype=torch.float32)
        target_mean = target_mean.to(_device_of(model), dtype=torch.float32)
        frames = _pool_frames(state.squeeze(0), max_frames)
        frame_values = torch.stack([value for value, _, _ in frames]).float()
        logits = (frame_values - source_mean) @ weight + target_mean
        raw_logits = logits.clone()
        if token_prior_log_probs is not None:
            logits = logits - token_prior_log_probs.to(logits.device, dtype=torch.float32)
        top = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
        raw_top = raw_logits.topk(min(top_k, raw_logits.shape[-1]), dim=-1)
        probabilities = logits.softmax(dim=-1).gather(dim=-1, index=top.indices)
        raw_probabilities = raw_logits.softmax(dim=-1).gather(dim=-1, index=raw_top.indices)
        layer_frames = []
        source_frames = max(state.shape[1], 1)
        for frame_index, (_, start, end) in enumerate(frames):
            ids = top.indices[frame_index].tolist()
            raw_ids = raw_top.indices[frame_index].tolist()
            tokens = [
                str(decode([token_id], skip_special_tokens=False)).strip() if callable(decode) else str(token_id)
                for token_id in ids
            ]
            raw_tokens_list = [
                str(decode([token_id], skip_special_tokens=False)).strip() if callable(decode) else str(token_id)
                for token_id in raw_ids
            ]
            layer_frames.append({
                "start_time": duration * start / source_frames,
                "end_time": duration * end / source_frames,
                "tokens": [
                    {
                        "token_id": token_id,
                        "token": token or "␠",
                        "score": float(score),
                        "probability": float(probability),
                    }
                    for token_id, token, score, probability in zip(
                        ids, tokens, top.values[frame_index].float().tolist(), probabilities[frame_index].float().tolist()
                    )
                ],
                "raw_tokens": [
                    {
                        "token_id": token_id,
                        "token": token or "␠",
                        "score": float(score),
                        "probability": float(probability),
                    }
                    for token_id, token, score, probability in zip(
                        raw_ids, raw_tokens_list, raw_top.values[frame_index].float().tolist(), raw_probabilities[frame_index].float().tolist()
                    )
                ],
            })
        layers.append({"layer": layer_index, "quality": quality_by_layer.get(layer_index), "frames": layer_frames})
    return {
        "model": adapter.model_id,
        "architecture": architecture,
        "duration_seconds": duration,
        "method": artifact["method"],
        "prior_correction": token_prior_log_probs is not None,
        "quality": artifact.get("quality", {}),
        "layers": layers,
    }
