import torch
import torchaudio
import os
import uuid
import librosa
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from .dataset_service import resolve_file

def add_gaussian_noise(waveform, noise_level=0.005):
    """
    waveform: Tensor [channels, time]
    noise_level: Standard deviation of noise
    """
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def apply_time_masking(waveform, mask_start_percent, mask_end_percent):
    """
    Apply time masking to a portion of the waveform
    waveform: Tensor [channels, time]
    mask_start_percent: Start percentage (0-100)
    mask_end_percent: End percentage (0-100)
    """
    channels, length = waveform.shape
    start_idx = int(length * mask_start_percent / 100)
    end_idx = int(length * mask_end_percent / 100)

    # Create a copy to avoid modifying original
    masked_waveform = waveform.clone()
    # Replace with a very low amplitude noise floor (~ -80 dBFS) instead of
    # hard zero. A perfectly silent region causes Whisper's ASR pipeline to
    # emit an empty generation for the corresponding chunk, which then
    # crashes `_extract_token_timestamps` and `generate_with_fallback`
    # (tensor-shape / index-out-of-bounds errors, see the fallback ladder
    # in `transcribe_whisper`). The dither is perceptually inaudible but
    # keeps the mel-spectrogram non-degenerate.
    if end_idx > start_idx:
        masked_waveform[:, start_idx:end_idx] = (
            torch.randn(channels, end_idx - start_idx, dtype=masked_waveform.dtype) * 1e-4
        )

    return masked_waveform

def apply_frequency_masking(waveform, sample_rate, mask_freq_start, mask_freq_end):
    """
    Apply frequency masking to the waveform
    waveform: Tensor [channels, time]
    sample_rate: Sample rate of the audio
    mask_freq_start: Start frequency in Hz
    mask_freq_end: End frequency in Hz
    """
    # Convert to frequency domain
    fft = torch.fft.fft(waveform, dim=-1)
    freqs = torch.fft.fftfreq(waveform.shape[-1], 1/sample_rate)
    
    # Create frequency mask
    freq_mask = (freqs >= mask_freq_start) & (freqs <= mask_freq_end)
    fft[:, freq_mask] = 0
    
    # Convert back to time domain
    masked_waveform = torch.fft.ifft(fft, dim=-1).real
    
    return masked_waveform

def apply_pitch_shift(waveform, sample_rate, pitch_shift_semitones):
    """
    Apply pitch shifting to the waveform
    waveform: Tensor [channels, time]
    sample_rate: Sample rate of the audio
    pitch_shift_semitones: Number of semitones to shift (positive = higher, negative = lower)
    """
    print(f"DEBUG: apply_pitch_shift called with pitch_shift_semitones={pitch_shift_semitones}, sample_rate={sample_rate}")
    print(f"DEBUG: Input waveform shape: {waveform.shape}")
    
    # Limit pitch shift to reasonable range to avoid performance issues
    pitch_shift_semitones = max(-6, min(6, pitch_shift_semitones))
    
    # Skip if no shift needed
    if abs(pitch_shift_semitones) < 0.1:
        print("DEBUG: Skipping pitch shift - too small")
        return waveform
    
    # Limit audio length to prevent infinite processing
    max_length = sample_rate * 30  # 30 seconds max
    if waveform.shape[-1] > max_length:
        print(f"DEBUG: Truncating audio to {max_length} samples to prevent long processing")
        waveform = waveform[..., :max_length]
    
    try:
        # Use librosa directly as it's more reliable and faster
        # Convert to numpy for librosa
        if waveform.dim() > 1:
            # Take first channel if stereo
            audio_np = waveform[0].numpy()
        else:
            audio_np = waveform.numpy()
        
        print(f"DEBUG: Using librosa.effects.pitch_shift with n_steps={pitch_shift_semitones}")
        print(f"DEBUG: Audio length: {len(audio_np)} samples")

        shifted_audio = librosa.effects.pitch_shift(
            y=audio_np,
            sr=sample_rate,
            n_steps=pitch_shift_semitones,
        )
        result = torch.from_numpy(shifted_audio).unsqueeze(0)
        print(f"DEBUG: Librosa pitch shift completed successfully, output shape: {result.shape}")
        return result
            
    except Exception as e:
        print(f"DEBUG: Pitch shift failed with error: {e}")
        print("DEBUG: Returning original waveform")
        return waveform

def apply_time_stretch(waveform, stretch_factor):
    """
    Apply time stretching to the waveform
    waveform: Tensor [channels, time]
    stretch_factor: Factor to stretch time (1.0 = no change, >1.0 = slower, <1.0 = faster)
    """
    print(f"DEBUG: apply_time_stretch called with stretch_factor={stretch_factor}")
    print(f"DEBUG: Input waveform shape: {waveform.shape}")
    
    # Skip if no stretch needed
    if abs(stretch_factor - 1.0) < 0.01:
        print("DEBUG: Skipping time stretch - factor too close to 1.0")
        return waveform
    
    try:
        # Use librosa for time stretching as it's more reliable
        # Convert to numpy for librosa
        if waveform.dim() > 1:
            # Take first channel if stereo
            audio_np = waveform[0].numpy()
        else:
            audio_np = waveform.numpy()
        
        print(f"DEBUG: Using librosa.effects.time_stretch with rate={stretch_factor}")
        # Apply time stretch using librosa
        stretched_audio = librosa.effects.time_stretch(
            y=audio_np, 
            rate=stretch_factor
        )
        
        # Convert back to torch tensor
        result = torch.from_numpy(stretched_audio).unsqueeze(0)  # Add channel dimension
        print(f"DEBUG: Time stretch completed successfully, output shape: {result.shape}")
        return result
        
    except Exception as e:
        print(f"DEBUG: Time stretch failed with error: {e}")
        print("DEBUG: Returning original waveform")
        return waveform

def apply_perturbations(waveform, sample_rate, perturbations: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Apply multiple perturbations to a waveform
    waveform: Tensor [channels, time]
    sample_rate: Sample rate of the audio
    perturbations: List of perturbation dictionaries
    Returns: (perturbed_waveform, applied_perturbations)
    """
    perturbed_waveform = waveform.clone()
    applied_perturbations = []
    
    for perturbation in perturbations:
        perturbation_type = perturbation.get("type")
        params = perturbation.get("params", {})
        
        try:
            if perturbation_type == "noise":
                noise_level = params.get("noise_level", 0.005)
                perturbed_waveform = add_gaussian_noise(perturbed_waveform, noise_level)
                applied_perturbations.append({
                    "type": "noise",
                    "params": {"noise_level": noise_level},
                    "status": "applied"
                })
                
            elif perturbation_type == "time_masking":
                mask_start = params.get("mask_start_percent", 20)
                mask_end = params.get("mask_end_percent", 40)
                perturbed_waveform = apply_time_masking(perturbed_waveform, mask_start, mask_end)
                applied_perturbations.append({
                    "type": "time_masking",
                    "params": {"mask_start_percent": mask_start, "mask_end_percent": mask_end},
                    "status": "applied"
                })
                
            elif perturbation_type == "frequency_masking":
                mask_freq_start = params.get("mask_freq_start", 1000)
                mask_freq_end = params.get("mask_freq_end", 2000)
                perturbed_waveform = apply_frequency_masking(perturbed_waveform, sample_rate, mask_freq_start, mask_freq_end)
                applied_perturbations.append({
                    "type": "frequency_masking",
                    "params": {"mask_freq_start": mask_freq_start, "mask_freq_end": mask_freq_end},
                    "status": "applied"
                })
                
            elif perturbation_type == "pitch_shift":
                pitch_shift_semitones = params.get("pitch_shift_semitones", 2)
                print(f"DEBUG: Processing pitch_shift perturbation with {pitch_shift_semitones} semitones")
                perturbed_waveform = apply_pitch_shift(perturbed_waveform, sample_rate, pitch_shift_semitones)
                applied_perturbations.append({
                    "type": "pitch_shift",
                    "params": {"pitch_shift_semitones": pitch_shift_semitones},
                    "status": "applied"
                })
                print(f"DEBUG: Pitch shift perturbation completed")
                
            elif perturbation_type == "time_stretch":
                stretch_factor = params.get("stretch_factor", 1.1)
                perturbed_waveform = apply_time_stretch(perturbed_waveform, stretch_factor)
                applied_perturbations.append({
                    "type": "time_stretch",
                    "params": {"stretch_factor": stretch_factor},
                    "status": "applied"
                })
                
            else:
                applied_perturbations.append({
                    "type": perturbation_type,
                    "params": params,
                    "status": "unsupported"
                })
                
        except Exception as e:
            applied_perturbations.append({
                "type": perturbation_type,
                "params": params,
                "status": "failed",
                "error": str(e)
            })
    
    return perturbed_waveform, applied_perturbations

def perturb_and_save(file_path: str, perturbations: List[Dict[str, Any]], output_dir: str = "uploads", dataset: str = None, session_id: str = None) -> Dict[str, Any]:
    """
    Apply perturbations to an audio file and save the result
    file_path: Path to the input audio file (can be dataset path or absolute path)
    perturbations: List of perturbation dictionaries
    output_dir: Directory to save the perturbed audio
    dataset: Dataset name if file_path is a dataset file
    session_id: Session ID for custom dataset resolution
    Returns: Dictionary with file info and metadata
    """
    # Resolve the file path - handle both dataset files and uploaded files
    try:
        if dataset and not Path(file_path).is_absolute():
            # This is a dataset file, resolve it using the dataset service
            resolved_path = resolve_file(dataset, file_path, session_id)
        else:
            # This is an uploaded file or absolute path
            resolved_path = Path(file_path)
            if not resolved_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
    except FileNotFoundError as e:
        return {
            "original_file": file_path,
            "perturbed_file": "",
            "filename": "",
            "duration_ms": 0,
            "sample_rate": 0,
            "applied_perturbations": [],
            "success": False,
            "error": str(e)
        }
    
    # Load the audio file
    try:
        audio_np, sample_rate = librosa.load(str(resolved_path), sr=None, mono=False)
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]
        waveform = torch.from_numpy(audio_np).float()
    except Exception as e:
        return {
            "original_file": file_path,
            "perturbed_file": "",
            "filename": "",
            "duration_ms": 0,
            "sample_rate": 0,
            "applied_perturbations": [],
            "success": False,
            "error": f"Failed to load audio file: {str(e)}"
        }
    
    # Apply perturbations
    perturbed_waveform, applied_perturbations = apply_perturbations(waveform, sample_rate, perturbations)

    # Sanitize: frequency masking (FFT/iFFT), pitch_shift, and time_stretch can
    # emit NaN/inf on degenerate inputs. If those samples get saved and later
    # fed into Whisper, `.generate()` produces bad token indices which trigger
    # a CUDA device-side assert. Once that assert fires the CUDA context is
    # poisoned for the process — every subsequent from_pretrained fails with
    # `cudaMemGetInfo`. Fix the numbers at the source instead.
    pw_np = perturbed_waveform.detach().cpu().numpy() if hasattr(perturbed_waveform, "detach") else perturbed_waveform.numpy()
    if not np.all(np.isfinite(pw_np)):
        bad = int(np.sum(~np.isfinite(pw_np)))
        print(f"DEBUG: perturbation produced {bad}/{pw_np.size} non-finite samples; zeroing them")
        pw_np = np.nan_to_num(pw_np, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(pw_np))) if pw_np.size > 0 else 0.0
    if peak > 1.0:
        print(f"DEBUG: perturbation peak amplitude {peak:.3f} > 1.0; normalizing")
        pw_np = pw_np / peak
    perturbed_waveform = torch.from_numpy(pw_np)

    # Generate output filename
    input_path = Path(file_path)
    output_filename = f"{input_path.stem}_perturbed_{uuid.uuid4().hex[:8]}{input_path.suffix}"
    output_path = Path(output_dir) / output_filename

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save the perturbed audio
    import soundfile as sf
    save_np = perturbed_waveform.numpy()
    if save_np.ndim == 2:
        save_np = save_np.T
    sf.write(str(output_path), save_np, sample_rate)
    
    # Calculate duration
    duration_ms = int(perturbed_waveform.shape[-1] / sample_rate * 1000)
    
    # Use forward slashes for web compatibility
    perturbed_file_path = str(output_path).replace("\\", "/")
    
    return {
        "original_file": file_path,
        "perturbed_file": perturbed_file_path,
        "filename": output_filename,
        "duration_ms": duration_ms,
        "sample_rate": sample_rate,
        "applied_perturbations": applied_perturbations,
        "success": True
    }