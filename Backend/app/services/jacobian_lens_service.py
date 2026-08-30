"""Decoder-side fitting and application of Jacobian lenses for speech-to-text.

The lens follows the LLM "Jacobian lens" construction (Gurnee et al., 2026,
"A Global Workspace in Language Models"): for every decoder layer it estimates
the average, position-resolved causal map from that layer's residual stream to
the model's final pre-logit state, averaged over source positions, future
positions, and fit transcripts.  Reading out replaces everything downstream of
a layer with that single linear map followed by the model's own output
projection, which yields ranked vocabulary tokens per (position, layer).

The implementation deliberately has no model-family imports.  Only the standard
Hugging Face encoder-decoder contract is assumed, selected by
``AudioModelAdapter.jacobian_lens_architecture`` (``"decoder"`` for seq2seq
speech-to-text models).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


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


def _encoder_features(model: Any, inputs: dict[str, Any]) -> Any:
    """Run the frozen encoder outside the autograd graph.

    The encoder is only the substrate that feeds cross-attention keys/values;
    gradients for decoder probes never flow into it, so it can be computed once
    under ``no_grad`` for every sample.
    """
    import torch

    encoder = getattr(model, "get_encoder", lambda: None)()
    if encoder is None:
        raise JacobianLensError("Model does not expose get_encoder()")
    with torch.no_grad():
        try:
            encoded = encoder(**inputs)
        except TypeError:
            primary = next(value for value in inputs.values() if hasattr(value, "ndim"))
            encoded = encoder(primary)
    states = getattr(encoded, "last_hidden_state", None)
    if states is None:
        states = encoded[0]
    return states


def _decoder_input_ids(model: Any, labels: Any) -> Any:
    """Teacher-forcing inputs: shift labels right past the start token."""
    prepare = getattr(model, "prepare_decoder_input_ids_from_labels", None)
    if callable(prepare):
        return prepare(labels=labels)
    start_id = getattr(getattr(model, "config", None), "decoder_start_token_id", None)
    if start_id is None:
        raise JacobianLensError("Model cannot derive decoder inputs from a transcript")
    import torch

    start = torch.full(
        (labels.shape[0], 1), int(start_id), dtype=labels.dtype, device=labels.device
    )
    return torch.cat([start, labels[:, :-1]], dim=1)


def _transcript_labels(model: Any, processor: Any, transcript: str, max_tokens: int | None = None) -> Any:
    tokenizer = _tokenizer(processor)
    if max_tokens is not None:
        labels = tokenizer(transcript, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids
    else:
        labels = tokenizer(transcript, return_tensors="pt", truncation=True).input_ids
    if labels.shape[-1] < 1:
        raise JacobianLensError("Transcript produced no tokens")
    return labels.to(_device_of(model))


def _decoder_states(model: Any, encoder_states: Any, decoder_input_ids: Any):
    """Teacher-forced decoder pass returning per-position residual streams.

    Sources are every recorded decoder hidden state except the final one; the
    target is ``last_hidden_state``, the exact tensor the model's own output
    projection consumes (post final LayerNorm where the architecture has one).
    Gradients w.r.t. any source therefore flow through the ordinary unembedding
    path, which is what makes the fitted map a same-space lens readout.
    """
    decoder = getattr(model, "get_decoder", lambda: None)()
    if decoder is None:
        raise JacobianLensError("Model does not expose get_decoder()")
    outputs = decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_states,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = getattr(outputs, "hidden_states", None)
    target = getattr(outputs, "last_hidden_state", None)
    if not hidden_states or target is None:
        raise JacobianLensError("Decoder did not return hidden states")
    sources = tuple(state for state in hidden_states if state is not target)
    if not sources:
        raise JacobianLensError("Decoder did not expose intermediate hidden states")
    return sources, target


def _decoder_states_for_fit(model: Any, inputs: dict[str, Any], transcript: str, processor: Any):
    import torch

    labels = _transcript_labels(model, processor, transcript)
    with torch.no_grad():
        encoder_states = _encoder_features(model, inputs)
    decoder_input_ids = _decoder_input_ids(model, labels)
    with torch.enable_grad():
        return _decoder_states(model, encoder_states, decoder_input_ids)


def fit_decoder_jacobian_lens(
    adapter: Any,
    resource: Any,
    samples: list[tuple[str, str]],
    probe_count: int,
    max_audio_seconds: float,
    on_sample: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Fit average decoder-layer Jacobians with Hutchinson VJPs.

    For source position ``t`` at decoder layer ``l`` and a future position
    ``t' >= t``, autograd gives the vector-Jacobian product of a Rademacher
    probe ``r`` against ``J = d h_final,t' / d h_l,t``.  One backward pass per
    probe yields, for every source position simultaneously, the sum over
    causally reachable future positions, because the causal mask zeroes the
    pairs with ``t > t'``.  Averaging ``outer(r, sum_t grad_t)`` over probes,
    source positions, future positions and fit transcripts estimates

        J_l = E[t, t'>=t, transcript, probe] [ d h_final,t' / d h_l,t ]

    without ever forming an exact Jacobian -- the same averaged, position-
    resolved estimator the LLM Jacobian lens uses, read from the decoder's own
    residual stream instead of across model components.
    """
    import torch

    architecture = adapter.jacobian_lens_architecture()
    if architecture != "decoder":
        raise JacobianLensError(
            f"{adapter.model_id} does not support a decoder Jacobian lens "
            "(only seq2seq speech-to-text models have a decoder to lens)"
        )
    processor, model = adapter.jacobian_lens_components(resource)
    model.eval()
    matrices: list[Any] | None = None

    for index, (audio_path, transcript) in enumerate(samples, start=1):
        inputs, _ = _prepare_audio(processor, audio_path, max_audio_seconds)
        inputs = _move_to_device(inputs, _device_of(model))
        sources, target = _decoder_states_for_fit(model, inputs, transcript, processor)
        if matrices is None:
            matrices = [
                torch.zeros((target.shape[-1], state.shape[-1]), dtype=torch.float32, device="cpu")
                for state in sources
            ]
        if len(sources) != len(matrices):
            raise JacobianLensError("Model returned an inconsistent number of decoder layers")
        # Every (source, future) pair with t <= t' contributes; the causal mask
        # removes the rest, so the triangular mean divides by T(T+1)/2.
        positions = target.shape[1]
        triangular = positions * (positions + 1) / 2.0

        for probe_index in range(probe_count):
            # Rademacher probes give an unbiased estimate of the full averaged
            # Jacobian while requiring one VJP per probe.
            probe = torch.empty(target.shape[-1], device=target.device).bernoulli_(0.5).mul_(2).sub_(1)
            scalar = torch.einsum("btd,d->", target.float(), probe)
            gradients = torch.autograd.grad(
                scalar,
                sources,
                retain_graph=probe_index < probe_count - 1,
                allow_unused=True,
            )
            for layer, gradient in enumerate(gradients):
                if gradient is None:
                    raise JacobianLensError(
                        f"Decoder layer {layer} is disconnected from the final verbal state"
                    )
                # Each source position's VJP already sums its reachable future
                # positions; summing again over positions covers the triangle.
                summed = gradient.sum(dim=1).squeeze(0)
                matrices[layer].add_(
                    torch.outer(probe.detach().float().cpu(), summed.detach().float().cpu()),
                    alpha=1.0 / triangular,
                )

        if on_sample:
            on_sample(index, len(samples))

    if not matrices:
        raise JacobianLensError("No lens samples were fitted")
    normalizer = float(len(samples) * probe_count)
    return {
        "format_version": 2,
        "architecture": "decoder",
        "model_id": adapter.model_id,
        "model_revision": adapter.jacobian_lens_revision(),
        "method": "hutchinson-decoder-vjp",
        "matrices": [matrix / normalizer for matrix in matrices],
        "sample_count": len(samples),
        "probe_count": probe_count,
    }


def _display_token(token: Any, index: int) -> str:
    text = str(token).replace("Ġ", " ").replace("▁", " ")
    return text if text.strip() else f"token_{index + 1}"


def _decode_transcript(processor: Any, token_ids: Any) -> str:
    decode = getattr(processor, "batch_decode", None)
    if callable(decode):
        return decode(token_ids, skip_special_tokens=True)[0]
    return ""


def apply_decoder_jacobian_lens(
    adapter: Any,
    resource: Any,
    artifact: dict[str, Any],
    audio_path: str,
    top_k: int,
    transcript: str | None = None,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Read vocabulary evidence from every decoder layer at every position.

    The decoder actually runs: greedy generation (or a provided reference
    transcript) produces the positions, then a teacher-forced pass collects the
    residual stream at each layer and position.  Each cell applies the fitted
    averaged Jacobian for that layer and reads out through the model's own
    output projection, mirroring lens(h) = softmax(E (J_l h)) from the LLM
    Jacobian lens.
    """
    import torch

    architecture = adapter.jacobian_lens_architecture()
    if artifact.get("model_id") != adapter.model_id or artifact.get("model_revision") != adapter.jacobian_lens_revision():
        raise JacobianLensError("Lens was fitted for a different model revision")
    if artifact.get("architecture") != architecture:
        raise JacobianLensError("Lens architecture does not match this model")
    processor, model = adapter.jacobian_lens_components(resource)
    model.eval()
    projection = _output_projection(model)
    inputs, duration = _prepare_audio(processor, audio_path, max_audio_seconds=60.0)
    inputs = _move_to_device(inputs, _device_of(model))
    tokenizer = _tokenizer(processor)
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    device = _device_of(model)

    with torch.no_grad():
        encoder_states = _encoder_features(model, inputs)
        if transcript and transcript.strip():
            labels = _transcript_labels(model, processor, transcript, max_tokens=448)
            decoder_input_ids = _decoder_input_ids(model, labels)
            transcript_source = "provided"
        else:
            generated = model.generate(**inputs, max_new_tokens=max(1, int(max_new_tokens)))
            decoder_input_ids = generated if generated.ndim == 2 else generated[:1]
            transcript_source = "generated"
        sources, _target = _decoder_states(model, encoder_states, decoder_input_ids)

    matrices = artifact["matrices"]
    if len(sources) != len(matrices):
        raise JacobianLensError("Lens layer count does not match the loaded model")
    weight = projection.weight.to(device=device, dtype=torch.float32)
    position_ids = decoder_input_ids[0].tolist()
    positions = [
        {"position": index, "token_id": token_id, "token": _display_token(convert(token_id) if callable(convert) else token_id, index)}
        for index, token_id in enumerate(position_ids)
    ]
    if transcript and transcript.strip():
        transcript_text = transcript
    else:
        transcript_text = _decode_transcript(processor, decoder_input_ids)

    layers = []
    for layer_index, (state, matrix) in enumerate(zip(sources, matrices)):
        matrix = matrix.to(device=device, dtype=torch.float32)
        values = state.squeeze(0).float() @ matrix.T
        logits = values @ weight.T
        probabilities = torch.softmax(logits, dim=-1)
        top = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
        layer_positions = []
        for position_index in range(logits.shape[0]):
            ids = top.indices[position_index].tolist()
            tokens = convert(ids) if callable(convert) else [str(token_id) for token_id in ids]
            layer_positions.append({
                "position": position_index,
                "tokens": [
                    {
                        "token_id": token_id,
                        "token": _display_token(token, token_id),
                        "score": float(score),
                        "probability": float(probability),
                    }
                    for token_id, token, score, probability in zip(
                        ids, tokens, top.values[position_index].float().tolist(),
                        probabilities[position_index, ids].tolist(),
                    )
                ],
            })
        layers.append({"layer": layer_index, "positions": layer_positions})
    return {
        "model": adapter.model_id,
        "architecture": architecture,
        "duration_seconds": duration,
        "transcript": transcript_text,
        "transcript_source": transcript_source,
        "positions": positions,
        "layers": layers,
    }
