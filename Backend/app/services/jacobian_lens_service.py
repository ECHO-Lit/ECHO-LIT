"""Adapter-native fitting and application of encoder Jacobian lenses.

The implementation deliberately has no model-family imports.  Architecture
differences are limited to the standard Hugging Face seq2seq and CTC output
contracts, selected by ``AudioModelAdapter.jacobian_lens_architecture``.
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


def _seq2seq_states(model: Any, inputs: dict[str, Any], transcript: str, processor: Any):
    tokenizer = _tokenizer(processor)
    labels = tokenizer(transcript, return_tensors="pt", truncation=True).input_ids.to(_device_of(model))
    if labels.shape[-1] < 1:
        raise JacobianLensError("Transcript produced no tokens")
    outputs = model(
        **inputs,
        labels=labels,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    encoder_states = getattr(outputs, "encoder_hidden_states", None)
    decoder_states = getattr(outputs, "decoder_hidden_states", None)
    if not encoder_states or not decoder_states:
        raise JacobianLensError("Seq2seq model did not return encoder and decoder hidden states")
    # Hugging Face includes the pre-transformer embedding state at index zero.
    # A lens is fitted for each actual encoder block, not that input embedding.
    if len(encoder_states) < 2:
        raise JacobianLensError("Seq2seq encoder did not expose transformer-layer states")
    return tuple(encoder_states[1:]), decoder_states[-1]


def _ctc_states(model: Any, inputs: dict[str, Any]):
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        raise JacobianLensError("CTC model did not return hidden states")
    if len(hidden_states) < 2:
        raise JacobianLensError("CTC encoder did not expose transformer-layer states")
    return tuple(hidden_states[1:]), hidden_states[-1]


def _states_for_fit(architecture: str, model: Any, inputs: dict[str, Any], transcript: str, processor: Any):
    if architecture == "seq2seq":
        return _seq2seq_states(model, inputs, transcript, processor)
    if architecture == "ctc":
        return _ctc_states(model, inputs)
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
    probe_count: int,
    max_audio_seconds: float,
    on_sample: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Fit average encoder-to-verbal-state Jacobians with Hutchinson VJPs.

    For a pooled encoder state ``h_l`` and pooled final verbal state ``z``, a
    Rademacher probe ``r`` gives ``grad(z·r, h_l) = J_l.T @ r``.  Averaging
    ``outer(r, grad)`` estimates the transport matrix ``J_l`` without forming
    an exact Jacobian for every audio clip.
    """
    import torch

    architecture = adapter.jacobian_lens_architecture()
    if not architecture:
        raise JacobianLensError(f"{adapter.model_id} does not support a speech Jacobian lens")
    processor, model = adapter.jacobian_lens_components(resource)
    model.eval()
    matrices: list[Any] | None = None
    baselines: list[Any] | None = None
    counts: list[int] | None = None

    for index, (audio_path, transcript) in enumerate(samples, start=1):
        inputs, _ = _prepare_audio(processor, audio_path, max_audio_seconds)
        inputs = _move_to_device(inputs, _device_of(model))
        encoder_states, target_states = _states_for_fit(
            architecture, model, inputs, transcript, processor
        )
        target = target_states.mean(dim=1).squeeze(0)
        # ``target`` is computed *from* the raw encoder activations during the
        # model forward pass.  Do not ask autograd for a derivative with
        # respect to a pooled tensor created after that forward pass: it is a
        # child of the activation, rather than an ancestor of ``target``.  We
        # differentiate with respect to each raw layer activation, then pool
        # the VJP across frames below.
        raw_sources = list(encoder_states)
        sources = [state.mean(dim=1).squeeze(0) for state in raw_sources]
        if matrices is None:
            matrices = [torch.zeros((target.numel(), source.numel()), dtype=torch.float32, device="cpu") for source in sources]
            baselines = [torch.zeros(source.numel(), dtype=torch.float32, device="cpu") for source in sources]
            counts = [0 for _ in sources]
        if len(sources) != len(matrices):
            raise JacobianLensError("Model returned an inconsistent number of encoder layers")

        for probe_index in range(probe_count):
            # Rademacher probes provide an unbiased estimate of the full
            # transport matrix while requiring one VJP per probe.
            probe = torch.empty_like(target).bernoulli_(0.5).mul_(2).sub_(1)
            scalar = torch.dot(target, probe)
            gradients = torch.autograd.grad(
                scalar,
                raw_sources,
                retain_graph=probe_index < probe_count - 1,
                allow_unused=True,
            )
            for layer, gradient in enumerate(gradients):
                if gradient is None:
                    raise JacobianLensError(f"Encoder layer {layer} is disconnected from the verbal output")
                # A perturbation of the pooled activation is broadcast to all
                # encoder frames, so its VJP is the sum over frame gradients.
                pooled_gradient = gradient.sum(dim=1).squeeze(0)
                matrices[layer].add_(
                    torch.outer(probe.detach().float().cpu(), pooled_gradient.detach().float().cpu())
                )

        for layer, source in enumerate(sources):
            baselines[layer].add_(source.detach().float().cpu())
            counts[layer] += 1
        if on_sample:
            on_sample(index, len(samples))

    if not matrices or not baselines or not counts:
        raise JacobianLensError("No lens samples were fitted")
    normalizer = float(len(samples) * probe_count)
    return {
        "format_version": 1,
        "architecture": architecture,
        "model_id": adapter.model_id,
        "model_revision": adapter.jacobian_lens_revision(),
        "matrices": [matrix / normalizer for matrix in matrices],
        "baselines": [baseline / count for baseline, count in zip(baselines, counts)],
        "sample_count": len(samples),
        "probe_count": probe_count,
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
    import torch

    architecture = adapter.jacobian_lens_architecture()
    if artifact.get("model_id") != adapter.model_id or artifact.get("model_revision") != adapter.jacobian_lens_revision():
        raise JacobianLensError("Lens was fitted for a different model revision")
    if artifact.get("architecture") != architecture:
        raise JacobianLensError("Lens architecture does not match this model")
    processor, model = adapter.jacobian_lens_components(resource)
    projection = _output_projection(model)
    inputs, duration = _prepare_audio(processor, audio_path, max_audio_seconds=60.0)
    inputs = _move_to_device(inputs, _device_of(model))
    with torch.no_grad():
        states = _encoder_states_for_apply(architecture, model, inputs)
    matrices, baselines = artifact["matrices"], artifact["baselines"]
    if len(states) != len(matrices) or len(states) != len(baselines):
        raise JacobianLensError("Lens layer count does not match the loaded model")
    tokenizer = _tokenizer(processor)
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    layers = []
    for layer_index, (state, matrix, baseline) in enumerate(zip(states, matrices, baselines)):
        # Fitted transports are stored in float32; retain that precision even
        # when the inference model itself runs in fp16/bf16.
        matrix = matrix.to(_device_of(model), dtype=torch.float32)
        baseline = baseline.to(_device_of(model), dtype=torch.float32)
        frames = _pool_frames(state.squeeze(0), max_frames)
        frame_values = torch.stack([value for value, _, _ in frames]).float()
        transported = (frame_values - baseline) @ matrix.T
        logits = transported @ projection.weight.to(transported.device, dtype=torch.float32).T
        top = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
        layer_frames = []
        source_frames = max(state.shape[1], 1)
        for frame_index, (_, start, end) in enumerate(frames):
            ids = top.indices[frame_index].tolist()
            tokens = convert(ids) if callable(convert) else [str(token_id) for token_id in ids]
            layer_frames.append({
                "start_time": duration * start / source_frames,
                "end_time": duration * end / source_frames,
                "tokens": [
                    {"token_id": token_id, "token": token, "score": float(score)}
                    for token_id, token, score in zip(ids, tokens, top.values[frame_index].float().tolist())
                ],
            })
        layers.append({"layer": layer_index, "frames": layer_frames})
    return {
        "model": adapter.model_id,
        "architecture": architecture,
        "duration_seconds": duration,
        "layers": layers,
    }
