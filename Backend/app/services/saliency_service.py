import logging
import os
import torch
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from captum.attr import IntegratedGradients, GradientShap, Lime
from captum.attr._utils.lrp_rules import EpsilonRule
from captum.attr._core.lrp import LRP
from app.core.device import accelerator_memory_allocated_mb, clear_accelerator_cache
from app.services.model_loader_service import (
    transcribe_whisper_base,
    transcribe_whisper_large,
    transcribe_whisper_with_timestamps,
    predict_emotion_wave2vec,
    get_whisper_saliency_models,
    get_emotion_models,
)

logger = logging.getLogger(__name__)
MAX_SALIENCY_SECONDS = int(os.getenv("MAX_SALIENCY_SECONDS", "12"))  # cap analysis window
MAX_SALIENCY_SECONDS_SHAP = int(os.getenv("MAX_SALIENCY_SECONDS_SHAP", "6"))  # stricter for SHAP
SALIENCY_SHAP_SAMPLES = int(os.getenv("SALIENCY_SHAP_SAMPLES", "8"))

def detect_model_type(model: str) -> str:
    if "whisper" in model.lower():
        return "whisper"
    elif "wav2vec" in model.lower():
        return "wav2vec2"
    return "unknown"


#################################################################################################################
def generate_whisper_saliency(audio_file_path: str, model_size: str = "base", method: str = "gradcam", existing_prediction: Dict = None) -> Dict:
    logger.info(f"Generating Whisper saliency for {audio_file_path} using {method} method")
    
    if existing_prediction and "chunks" in existing_prediction:
        data = existing_prediction
        audio = data["audio"]
        chunks = data["chunks"]
        logger.info(f"Using existing prediction with {len(chunks)} chunks")
    else:
        logger.info("Transcribing audio with timestamps for saliency analysis")
        data = transcribe_whisper_with_timestamps(audio_file_path, model_size)
        audio = data["audio"]
        chunks = data["chunks"]
        logger.info(f"Transcription completed with {len(chunks) if chunks else 0} chunks")
    
    # Crop to a safe max duration to avoid OOM
    if isinstance(audio, (list, tuple)):
        audio = np.asarray(audio)
    if hasattr(audio, "shape") and audio is not None:
        # Sanitize before anything touches CUDA — NaN/inf here would cause
        # the encoder to emit garbage and can trigger a CUDA device-side
        # assert that poisons the whole process's CUDA context.
        if not np.all(np.isfinite(audio)):
            logger.warning(f"generate_whisper_saliency: non-finite samples in {audio_file_path}; zeroing them")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        max_seconds = MAX_SALIENCY_SECONDS_SHAP if method == "shap" else MAX_SALIENCY_SECONDS
        max_len = int(max_seconds * 16000)
        if len(audio) > max_len:
            audio = audio[:max_len]
            # Keep only chunks inside the window
            chunks = [c for c in chunks if c.get("timestamp", [0, 0])[0] < max_seconds]
    
    if model_size == "base":
        processor, model = get_whisper_saliency_models("openai/whisper-base")
    else:
        processor, model = get_whisper_saliency_models("openai/whisper-large-v3")

    # Defensive reset: previous saliency runs enable gradient checkpointing.
    # A finally block restores it, but if the process was interrupted or hit an
    # unexpected error path, the flag may still be set — leading to inconsistent
    # forward behaviour on later runs. Ensure a clean slate.
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass

    device = next(model.parameters()).device
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    input_features.requires_grad_(True)
    
    def model_forward(inputs):
        # Reduce to a scalar per batch: energy of encoder activations
        enc = model.encoder(inputs).last_hidden_state  # [B, T, H]
        return enc.pow(2).mean(dim=(1, 2))             # [B]
    
    if method == "gradcam":
        clear_accelerator_cache(device)

        # NOTE: gradient checkpointing intentionally NOT enabled here.
        # With Captum IG's interpolated inputs (n_steps forward+backward passes)
        # the re-run of the encoder under checkpointing can produce shape
        # mismatches inside Whisper's attention (e.g. "tensor a (2) must match
        # tensor b (0) at non-singleton dim 1"). The memory saving is not worth
        # the correctness risk, and this path already uses internal_batch_size=1.
        n_steps = 16
        internal_batch_size = 1

        allocated_mb = accelerator_memory_allocated_mb(device)
        if allocated_mb is not None:
            logger.info(f"GPU memory before saliency: {allocated_mb:.2f} MB")

        attributions = None
        try:
            ig = IntegratedGradients(model_forward)
            attributions = ig.attribute(
                input_features,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
            )
        except RuntimeError as e:
            msg = str(e)
            if "CUDA out of memory" in msg or "out of memory" in msg.lower():
                clear_accelerator_cache(device)
                logger.warning("Whisper IG OOM; retrying with fewer steps")
                try:
                    ig = IntegratedGradients(model_forward)
                    attributions = ig.attribute(
                        input_features,
                        n_steps=8,
                        internal_batch_size=internal_batch_size,
                    )
                except Exception as e2:
                    logger.error(f"Whisper IG retry failed: {e2}; falling back to energy map")
                    attributions = None
            else:
                # Any other runtime error (e.g. transient shape mismatch inside
                # the encoder) falls back to the deterministic energy map
                # rather than surfacing a 500. This keeps saliency available
                # even when IG's autograd path hits an edge case.
                logger.exception("Whisper IG failed; falling back to energy map")
                attributions = None
    elif method == "lime":
        lime = Lime(model_forward)
        attributions = lime.attribute(input_features)
    elif method == "shap":
        # Use Captum GradientShap on the model's current device with small n_samples
        gs = GradientShap(model_forward)
        baseline = torch.zeros_like(input_features)
        try:
            clear_accelerator_cache(device)
            attributions = gs.attribute(
                input_features,
                baselines=baseline,
                n_samples=max(2, min(16, SALIENCY_SHAP_SAMPLES)),
                stdevs=0.09,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("Whisper SHAP OOM; retrying with fewer samples")
                clear_accelerator_cache(device)
                attributions = gs.attribute(
                    input_features,
                    baselines=baseline,
                    n_samples=max(2, min(8, SALIENCY_SHAP_SAMPLES // 2 if SALIENCY_SHAP_SAMPLES > 2 else 2)),
                    stdevs=0.07,
                )
            else:
                logger.exception("Whisper SHAP failed; falling back to energy map")
                attributions = None
        except Exception:
            logger.exception("Whisper SHAP failed; falling back to energy map")
            attributions = None
    else:
        attributions = torch.zeros_like(input_features)
    
    # Reduce to 1D timeline and normalize to [0,1] for visible intensities
    if attributions is not None:
        saliency_np = attributions.detach().cpu().numpy().squeeze()
        if saliency_np.ndim == 2:
            if saliency_np.shape[0] in (64, 80, 128):
                agg = np.mean(np.abs(saliency_np), axis=0)
            else:
                agg = np.mean(np.abs(saliency_np), axis=1)
        elif saliency_np.ndim == 1:
            agg = np.abs(saliency_np)
        else:
            while saliency_np.ndim > 1:
                saliency_np = saliency_np.mean(axis=0)
            agg = np.abs(saliency_np)
        max_abs = float(np.max(agg)) if agg.size > 0 else 0.0
        saliency_scores = (agg / max_abs) if max_abs > 0 else np.zeros_like(agg)
    else:
        saliency_scores = np.array([])

    # Fallback: if scores are empty or nearly constant, use encoder energy map
    use_energy_fallback = (
        saliency_scores.size == 0 or
        (np.max(saliency_scores) - np.min(saliency_scores) if saliency_scores.size > 0 else 0.0) < 1e-6
    )
    if use_energy_fallback:
        logger.info("Using Whisper energy-map fallback for saliency")
        try:
            with torch.no_grad():
                enc = model.encoder(input_features).last_hidden_state  # [B, T, H]
                energy = enc.abs().mean(dim=2).squeeze(0).detach().cpu().numpy()
            if energy.size > 0:
                e_min, e_ptp = float(np.min(energy)), float(np.ptp(energy))
                saliency_scores = (energy - e_min) / (e_ptp + 1e-9)
            else:
                saliency_scores = np.zeros(1, dtype=np.float32)
        except Exception:
            logger.exception("Whisper energy-map fallback failed; returning empty saliency")
            saliency_scores = np.zeros(1, dtype=np.float32)

    # Create dense series with smoothing and percentile clipping
    series = saliency_scores.astype(np.float32)
    if series.size > 0:
        win = max(3, int(series.size / 64))
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=np.float32) / float(win)
        series = np.convolve(series, kernel, mode="same")
        p95 = float(np.percentile(series, 95))
        if p95 > 0:
            series = np.clip(series, 0, p95)
        smin, smax = float(np.min(series)), float(np.max(series))
        series = (series - smin) / (smax - smin + 1e-9)
    
    segments = []
    # Map timestamps to attribution timeline robustly
    total_duration = float(len(audio)) / 16000.0 if hasattr(audio, "__len__") and len(audio) > 0 else 0.0
    T = len(saliency_scores)
    fps = (T / total_duration) if total_duration > 0 else 1.0
    
    # Process word-level chunks if available with simplified logic
    if chunks and total_duration > 0:
        logger.info(f"Processing {len(chunks)} word-level chunks for saliency segmentation")
        
        # Debug: Log first few chunks to understand structure
        if len(chunks) > 0:
            logger.info(f"First chunk structure: {chunks[0]}")
            if len(chunks) > 5:
                logger.info(f"Sample of chunks: {chunks[:3]} ... {chunks[-2:]}")
        
        for chunk in chunks:
            start_time = chunk.get("timestamp", [0, 0])[0]
            end_time = chunk.get("timestamp", [0, 0])[1]
            word = chunk.get("text", "")
            
            # Skip invalid chunks
            if end_time <= start_time or start_time < 0 or end_time > total_duration:
                continue
            
            # Convert to attribution frames
            start_frame = max(0, min(T - 1, int(start_time * fps)))
            end_frame = max(start_frame + 1, min(T, int(end_time * fps)))
            
            # Calculate segment saliency
            if end_frame > start_frame:
                segment_saliency = float(np.mean(saliency_scores[start_frame:end_frame]))
                segments.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "word": word.strip(),
                    "saliency": segment_saliency,
                    "intensity": float(abs(segment_saliency))
                })
        
        # Sort by start time to ensure proper order
        segments.sort(key=lambda x: x["start_time"])
        
        logger.info(f"Created {len(segments)} segments from word-level timestamps")

    # Fallback: if no segments were created, create uniform time-based segments
    if len(segments) == 0 and T > 0 and total_duration > 0:
        logger.info("No word-level segments found, creating uniform time-based segments")
        # Create 10-20 segments based on audio duration (aim for ~0.3-1 second segments)
        num_segments = max(8, min(32, int(total_duration * 2)))
        
        for i in range(num_segments):
            start_time = (i / num_segments) * total_duration
            end_time = ((i + 1) / num_segments) * total_duration
            
            start_frame = max(0, min(T - 1, int(start_time * fps)))
            end_frame = max(start_frame + 1, min(T, int(end_time * fps)))
            
            segment_saliency = float(np.mean(saliency_scores[start_frame:end_frame]))
            segments.append({
                "start_time": start_time,
                "end_time": end_time,
                "word": f"segment_{i+1}",
                "saliency": segment_saliency,
                "intensity": float(abs(segment_saliency))
            })
        
        logger.info(f"Created {len(segments)} uniform time-based segments")

    # Final normalization across segments for visibility
    if len(segments) > 0:
        # Collect raw saliency values
        raw_saliencies = [s.get("saliency", 0.0) for s in segments]
        
        # Use absolute values for intensity (magnitude of importance)
        abs_vals = np.abs(raw_saliencies)
        
        # Robust normalization to prevent all-zero intensities
        max_abs = float(np.max(abs_vals)) if len(abs_vals) > 0 else 0.0
        if max_abs > 1e-9:
            # Scale to [0,1] based on maximum absolute value
            for i, segment in enumerate(segments):
                segment["intensity"] = float(abs_vals[i] / max_abs)
        else:
            # Fallback: use relative ranking if all values are very small
            sorted_indices = np.argsort(-abs_vals)  # Sort descending by magnitude
            for rank, idx in enumerate(sorted_indices):
                # Assign intensity based on ranking: highest gets 1.0, lowest gets 0.1
                segments[idx]["intensity"] = float(1.0 - (rank / len(segments)) * 0.9)
        
        # Ensure minimum visibility for all segments
        for segment in segments:
            segment["intensity"] = max(0.1, segment["intensity"])  # Minimum 10% intensity
    
    return {
        "model": f"whisper-{model_size}",
        "method": method,
        "segments": segments,
        "total_duration": total_duration,
        "series": series.tolist()
    }

################################################################################################################

def generate_wav2vec2_saliency(audio_file_path: str, method: str = "gradcam", existing_prediction: Dict = None) -> Dict:
    feature_extractor, emo_model = get_emotion_models()
    # Derive the device from the model because an OOM fallback may relocate it.
    runtime_device = next(emo_model.parameters()).device
    # Defensive reset: prior attention-extraction paths may have flipped
    # `output_attentions` on the base config; ensure model is in eval mode.
    emo_model.eval()
    if getattr(emo_model.config, "output_attentions", False):
        emo_model.config.output_attentions = False
    if hasattr(emo_model, "wav2vec2") and getattr(emo_model.wav2vec2.config, "output_attentions", False):
        emo_model.wav2vec2.config.output_attentions = False

    audio, rate = librosa.load(audio_file_path, sr=16000)
    # NaN/inf in the waveform propagate into logits and can trip a CUDA
    # device-side assert (which then poisons the whole process's CUDA
    # context). Sanitize up-front.
    if not np.all(np.isfinite(audio)):
        logger.warning(f"generate_wav2vec2_saliency: non-finite samples in {audio_file_path}; zeroing them")
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    # Wav2Vec2 CNN reduces the waveform through 7 conv layers with total
    # effective downsampling of ~320. Empirically an input < ~1600 samples
    # (~0.1 s) can produce a length that mismatches the attention-mask
    # feature-length calculation (giving the "tensor a (N) vs tensor b (0)
    # at dim 1" broadcast error inside the encoder). Bump the minimum well
    # above the raw CNN-arithmetic floor of 400.
    MIN_WAV2VEC2_SAMPLES = 1600
    if len(audio) < MIN_WAV2VEC2_SAMPLES:
        raise ValueError(
            f"Audio file is too short ({len(audio)} samples at 16 kHz). "
            f"Minimum required for wav2vec2 saliency is {MIN_WAV2VEC2_SAMPLES} samples (~{MIN_WAV2VEC2_SAMPLES/16000:.3f}s)."
        )
    # Crop to safe max duration to bound memory
    max_seconds = MAX_SALIENCY_SECONDS_SHAP if method == "shap" else MAX_SALIENCY_SECONDS
    max_len = int(max_seconds * rate)
    if len(audio) > max_len:
        audio = audio[:max_len]
    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)

    input_values = inputs.input_values.to(runtime_device)
    attention_mask = inputs.attention_mask.to(runtime_device) if "attention_mask" in inputs else None
    
    input_values.requires_grad_(True)
    
    # Determine class to attribute (predicted emotion)
    with torch.no_grad():
        tmp_out = emo_model(input_values=input_values, attention_mask=attention_mask)
        tmp_probs = torch.nn.functional.softmax(tmp_out.logits, dim=-1)
        target_idx = int(torch.argmax(tmp_probs, dim=-1).item())

    def model_forward(inputs, mask=None, cls_idx: int = 0):
        if mask is not None and mask.shape[0] != inputs.shape[0]:
            mask = mask.expand(inputs.shape[0], -1)
        outputs = emo_model(input_values=inputs, attention_mask=mask)
        return outputs.logits[:, cls_idx]
    
    if method == "gradcam":
        ig = IntegratedGradients(model_forward)
        try:
            attributions = ig.attribute(
                input_values,
                additional_forward_args=(attention_mask, target_idx),
                n_steps=32,
                internal_batch_size=1,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("Accelerator OOM during Wav2Vec2 saliency. Falling back to CPU with fewer steps.")
                clear_accelerator_cache(runtime_device)
                cpu_device = torch.device("cpu")
                # Move inputs to CPU and temporarily move model — restore in finally to avoid global corruption
                input_values_cpu = input_values.detach().to(cpu_device)
                input_values_cpu.requires_grad_(True)
                attention_mask_cpu = attention_mask.detach().to(cpu_device) if attention_mask is not None else None
                original_device = next(emo_model.parameters()).device
                try:
                    emo_model.to(cpu_device)
                    ig_cpu = IntegratedGradients(model_forward)
                    attributions = ig_cpu.attribute(
                        input_values_cpu,
                        additional_forward_args=(attention_mask_cpu, target_idx),
                        n_steps=16,
                        internal_batch_size=1,
                    )
                    # replace input_values ref so downstream normalization works on CPU
                    input_values = input_values_cpu
                    attention_mask = attention_mask_cpu
                finally:
                    emo_model.to(original_device)
            else:
                raise
    elif method == "lime":
        lime = Lime(model_forward)
        attributions = lime.attribute(input_values, additional_forward_args=(attention_mask, target_idx))
    elif method == "shap":
        # Use Captum GradientShap on model's current device with small n_samples
        gs = GradientShap(model_forward)
        baseline = torch.zeros_like(input_values)
        try:
            clear_accelerator_cache(runtime_device)
            attributions = gs.attribute(
                input_values,
                baselines=baseline,
                additional_forward_args=(attention_mask, target_idx),
                n_samples=max(2, min(16, SALIENCY_SHAP_SAMPLES)),
                stdevs=0.09,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("Wav2Vec2 SHAP OOM; retrying with fewer samples")
                clear_accelerator_cache(runtime_device)
                attributions = gs.attribute(
                    input_values,
                    baselines=baseline,
                    additional_forward_args=(attention_mask, target_idx),
                    n_samples=max(2, min(8, SALIENCY_SHAP_SAMPLES // 2 if SALIENCY_SHAP_SAMPLES > 2 else 2)),
                    stdevs=0.07,
                )
            else:
                logger.exception("Wav2Vec2 SHAP failed; falling back to energy map")
                attributions = None
        except Exception:
            logger.exception("Wav2Vec2 SHAP failed; falling back to energy map")
            attributions = None
    else:
        attributions = torch.zeros_like(input_values)
    
    # Normalize to [0,1] for visible intensities
    if attributions is not None:
        tmp = attributions.detach().cpu().numpy().squeeze()
        if tmp.ndim > 1:
            tmp = np.mean(np.abs(tmp), axis=0)
        else:
            tmp = np.abs(tmp)
        mx = float(np.max(tmp)) if tmp.size > 0 else 0.0
        saliency_scores = (tmp / mx) if mx > 0 else np.zeros_like(tmp)
    else:
        saliency_scores = np.array([])

    # Fallback: if SHAP produced empty/flat attributions, use encoder energy
    if saliency_scores.size == 0 or (np.max(saliency_scores) - np.min(saliency_scores) if saliency_scores.size > 0 else 0.0) < 1e-6:
        logger.info("Using Wav2Vec2 energy-map fallback for saliency")
        with torch.no_grad():
            hs = emo_model.wav2vec2(input_values=input_values, attention_mask=attention_mask).last_hidden_state  # [B,T,H]
            energy = hs.abs().mean(dim=2).squeeze(0).detach().cpu().numpy()
        if energy.size > 0:
            e_min, e_ptp = float(np.min(energy)), float(np.ptp(energy))
            saliency_scores = (energy - e_min) / (e_ptp + 1e-9)
        else:
            saliency_scores = np.zeros(1, dtype=np.float32)

    # Create dense series with smoothing and percentile clipping
    series = saliency_scores.astype(np.float32)
    if series.size > 0:
        win = max(3, int(series.size / 64))
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=np.float32) / float(win)
        series = np.convolve(series, kernel, mode="same")
        p95 = float(np.percentile(series, 95))
        if p95 > 0:
            series = np.clip(series, 0, p95)
        smin, smax = float(np.min(series)), float(np.max(series))
        series = (series - smin) / (smax - smin + 1e-9)
    
    with torch.no_grad():
        model_device = next(emo_model.parameters()).device
        iv = input_values.to(model_device)
        am = attention_mask.to(model_device) if attention_mask is not None else None
        outputs = emo_model(input_values=iv, attention_mask=am)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_emotion = torch.argmax(probs, dim=-1).item()
        id2label = emo_model.config.id2label
        emotion = id2label.get(predicted_emotion, str(predicted_emotion))
    
    segment_duration = len(audio) / 16000
    num_segments = 32
    segment_length = segment_duration / num_segments if num_segments > 0 else segment_duration
    
    segments = []
    # Map segment times to attribution indices using derived fps from saliency timeline
    T = len(saliency_scores)
    fps = (T / segment_duration) if segment_duration > 0 else 1.0
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = (i + 1) * segment_length
        
        start_frame = max(0, min(T - 1, int(start_time * fps)))
        end_frame = max(start_frame + 1, min(T, int(end_time * fps)))
        segment_saliency = np.mean(saliency_scores[start_frame:end_frame])
        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "saliency": float(segment_saliency),
            "intensity": float(abs(segment_saliency))
        })

    # Final normalization across segments for visibility
    if len(segments) > 0:
        # Use robust intensity calculation
        raw_saliencies = [s.get("saliency", 0.0) for s in segments]
        abs_vals = np.abs(raw_saliencies)
        
        # Robust normalization to prevent all-zero intensities
        max_abs = float(np.max(abs_vals)) if len(abs_vals) > 0 else 0.0
        if max_abs > 1e-9:
            # Scale to [0,1] based on maximum absolute value
            for i, segment in enumerate(segments):
                segment["intensity"] = float(abs_vals[i] / max_abs)
        else:
            # Fallback: use relative ranking if all values are very small
            sorted_indices = np.argsort(-abs_vals)  # Sort descending by magnitude
            for rank, idx in enumerate(sorted_indices):
                # Assign intensity based on ranking: highest gets 1.0, lowest gets 0.1
                segments[idx]["intensity"] = float(1.0 - (rank / len(segments)) * 0.9)
        
        # Ensure minimum visibility for all segments
        for segment in segments:
            segment["intensity"] = max(0.1, segment["intensity"])  # Minimum 10% intensity
    
    return {
        "model": "wav2vec2",
        "method": method,
        "emotion": emotion,
        "segments": segments,
        "total_duration": segment_duration,
        "series": series.tolist()
    }

def generate_saliency(audio_file_path: str, model: str, method: str = "gradcam", existing_prediction: Dict = None) -> Dict:
    model_type = detect_model_type(model)
    
    if model_type == "whisper":
        model_size = "base" if "base" in model else "large"
        return generate_whisper_saliency(audio_file_path, model_size, method, existing_prediction)
    elif model_type == "wav2vec2":
        return generate_wav2vec2_saliency(audio_file_path, method, existing_prediction)
    else:
        raise ValueError(f"Unsupported model: {model}")
