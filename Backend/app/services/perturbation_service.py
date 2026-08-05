"""Audio perturbation DSP engine.

Two families of functionality live here:

1. The legacy single-shot perturbation pipeline (``apply_perturbations`` /
   ``perturb_and_save``), used by the FR-2 "perturbation" job operation and the
   legacy ``/perturb`` route. These compose a *list* of perturbations onto a
   running waveform and are unrelated to FR-7's isolation requirement.

2. The FR-7 "linguistic vs. acoustic" sweep engine (``OPERATORS``,
   ``render_variant``, ``check_applicable``, ``normalize_loudness``,
   ``variant_seed``). Each FR-7 operator is applied to the untouched canonical
   baseline exactly once per grid point -- never chained -- so that exactly
   one acoustic property varies at a time (see docs/FR7plan.md Part 1 S2.4).
"""

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import librosa
import numpy as np
import torch

from .dataset_service import resolve_file


# ---------------------------------------------------------------------------
# Legacy single-shot perturbation pipeline (FR-2 / legacy /perturb route)
# ---------------------------------------------------------------------------

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
    # Limit pitch shift to reasonable range to avoid performance issues
    pitch_shift_semitones = max(-6, min(6, pitch_shift_semitones))

    # Skip if no shift needed
    if abs(pitch_shift_semitones) < 0.1:
        return waveform

    # Limit audio length to prevent infinite processing
    max_length = sample_rate * 30  # 30 seconds max
    if waveform.shape[-1] > max_length:
        waveform = waveform[..., :max_length]

    try:
        # Use librosa directly as it's more reliable and faster
        # Convert to numpy for librosa
        if waveform.dim() > 1:
            # Take first channel if stereo
            audio_np = waveform[0].numpy()
        else:
            audio_np = waveform.numpy()

        shifted_audio = librosa.effects.pitch_shift(
            y=audio_np,
            sr=sample_rate,
            n_steps=pitch_shift_semitones,
        )
        result = torch.from_numpy(shifted_audio).unsqueeze(0)
        return result

    except Exception:
        return waveform

def apply_time_stretch(waveform, stretch_factor):
    """
    Apply time stretching to the waveform
    waveform: Tensor [channels, time]
    stretch_factor: Factor to stretch time (1.0 = no change, >1.0 = slower, <1.0 = faster)
    """
    # Skip if no stretch needed
    if abs(stretch_factor - 1.0) < 0.01:
        return waveform

    try:
        # Use librosa for time stretching as it's more reliable
        # Convert to numpy for librosa
        if waveform.dim() > 1:
            # Take first channel if stereo
            audio_np = waveform[0].numpy()
        else:
            audio_np = waveform.numpy()

        # Apply time stretch using librosa
        stretched_audio = librosa.effects.time_stretch(
            y=audio_np,
            rate=stretch_factor
        )

        # Convert back to torch tensor
        result = torch.from_numpy(stretched_audio).unsqueeze(0)  # Add channel dimension
        return result

    except Exception:
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
                perturbed_waveform = apply_pitch_shift(perturbed_waveform, sample_rate, pitch_shift_semitones)
                applied_perturbations.append({
                    "type": "pitch_shift",
                    "params": {"pitch_shift_semitones": pitch_shift_semitones},
                    "status": "applied"
                })

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
    # poisoned for the process -- every subsequent from_pretrained fails with
    # `cudaMemGetInfo`. Fix the numbers at the source instead.
    pw_np = perturbed_waveform.detach().cpu().numpy() if hasattr(perturbed_waveform, "detach") else perturbed_waveform.numpy()
    if not np.all(np.isfinite(pw_np)):
        pw_np = np.nan_to_num(pw_np, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(pw_np))) if pw_np.size > 0 else 0.0
    if peak > 1.0:
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


# ---------------------------------------------------------------------------
# FR-7 sweep engine: single-property variant rendering
# ---------------------------------------------------------------------------

TARGET_SR = 16_000        # every FR-7 variant is rendered at the model's native rate
TARGET_LUFS = -23.0       # EBU R128 reference; removes gain as a confound
_EPS = 1e-12

PROPERTY_UNITS: Dict[str, str] = {
    "identity": "n/a",
    "pitch": "semitones",
    "speed": "rate",
    "noise": "dB SNR",
    "time_mask": "%",
    "freq_mask": "Hz",
}

Operator = Callable[[np.ndarray, int, Any, np.random.Generator], np.ndarray]


def load_canonical(path: str) -> Tuple[np.ndarray, int]:
    """Mono, float32, 16 kHz. Every operator sees the same front-end so that
    resampling and channel-downmix are never part of the measured effect."""
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32, copy=False), TARGET_SR


def op_identity(x: np.ndarray, sr: int, param: Any, rng: np.random.Generator) -> np.ndarray:
    del sr, param, rng
    return x.copy()


def op_pitch(x: np.ndarray, sr: int, semitones: float, rng: np.random.Generator) -> np.ndarray:
    """Phase-vocoder pitch shift. Duration is preserved to within one hop, so
    word timing (and the reference transcript alignment) is untouched; only
    F0 and the harmonic structure move."""
    del rng
    if abs(semitones) < 1e-3:
        return x.copy()
    return librosa.effects.pitch_shift(y=x, sr=sr, n_steps=float(semitones)).astype(np.float32)


def op_speed(x: np.ndarray, sr: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Phase-vocoder time stretch: rate > 1 is faster. F0 and formants are
    preserved; only the temporal envelope changes."""
    del sr, rng
    if abs(rate - 1.0) < 1e-3:
        return x.copy()
    return librosa.effects.time_stretch(y=x, rate=float(rate)).astype(np.float32)


def op_noise(x: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Additive white Gaussian noise at a calibrated SNR.

    Parameterising by SNR rather than raw standard deviation is what makes the
    sweep comparable across files: sigma is derived from THIS signal's own
    RMS, so 10 dB means the same perceptual thing for a loud and a quiet clip.
        sigma = rms(x) / 10^(SNR/20)
    """
    del sr
    rms = float(np.sqrt(np.mean(np.square(x)) + _EPS))
    sigma = rms / (10.0 ** (snr_db / 20.0))
    return (x + rng.normal(0.0, sigma, size=x.shape)).astype(np.float32)


def op_time_mask(x: np.ndarray, sr: int, width_pct: float, rng: np.random.Generator) -> np.ndarray:
    """Replace a centred span with an inaudible dither floor (~-80 dBFS).

    NOT hard zero -- a perfectly silent region makes Whisper emit an empty
    generation for that chunk (see the legacy time-masking helper above)."""
    del sr
    n = x.shape[-1]
    width = int(n * max(0.0, min(width_pct, 100.0)) / 100.0)
    if width <= 0:
        return x.copy()
    start = (n - width) // 2
    out = x.copy()
    out[start:start + width] = rng.normal(0.0, 1e-4, size=width).astype(np.float32)
    return out


def op_freq_mask(x: np.ndarray, sr: int, band: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    """STFT-domain band suppression with ISTFT resynthesis.

    Deliberately NOT the full-signal rFFT used by the legacy helper above: a
    global FFT smears the notch across the whole utterance and produces
    pre-echo. An STFT notch is time-local and phase-consistent."""
    del rng
    lo, hi = band
    stft = librosa.stft(x, n_fft=1024, hop_length=256)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    mask = (freqs >= lo) & (freqs <= hi)
    stft[mask, :] = 0.0
    return librosa.istft(stft, hop_length=256, length=x.shape[-1]).astype(np.float32)


OPERATORS: Dict[str, Operator] = {
    "identity": op_identity,
    "pitch": op_pitch,
    "speed": op_speed,
    "noise": op_noise,
    "time_mask": op_time_mask,
    "freq_mask": op_freq_mask,
}


def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """Fixed-loudness normalisation (invariant I3): any amplitude change is
    itself an acoustic perturbation, so gain must not ride along unmeasured."""
    try:
        import pyloudnorm as pyln

        loudness = pyln.Meter(sr).integrated_loudness(x)
        if np.isfinite(loudness):
            x = pyln.normalize.loudness(x, loudness, target_lufs)
    except Exception:
        rms = float(np.sqrt(np.mean(np.square(x)) + _EPS))
        x = x * (10.0 ** (target_lufs / 20.0) / max(rms, _EPS))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return (x / peak * 0.99).astype(np.float32) if peak > 0.99 else x.astype(np.float32)


def variant_seed(baseline_sha256: str, prop: str, theta: float, repeat: int) -> int:
    """Content-derived seed -- excludes job_id/session_id on purpose, so a
    re-submitted sweep renders byte-identical audio and hits the existing
    per-item inference cache (invariant I4)."""
    material = f"{baseline_sha256}|{prop}|{theta:.6f}|{repeat}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _scalar(theta: Any) -> float:
    if isinstance(theta, (tuple, list)):
        return float(theta[0])
    return float(theta)


def _measured_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    noise = noisy - clean
    signal_power = float(np.mean(np.square(clean)) + _EPS)
    noise_power = float(np.mean(np.square(noise)) + _EPS)
    return 10.0 * float(np.log10(signal_power / noise_power))


@dataclass(frozen=True)
class Applicability:
    applicable: bool
    reason: str | None = None


def check_applicable(prop: str, theta: float, audio: np.ndarray, sr: int) -> Applicability:
    """SRS exception flow: "properties that cannot be isolated for a given
    input are reported as not applicable" rather than dropped."""
    duration = audio.shape[-1] / sr
    if duration < 0.5:
        return Applicability(False, "Audio is shorter than the 0.5 s minimum for perturbation analysis")
    if prop == "pitch":
        f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
        voiced_ratio = float(np.mean(np.isfinite(f0) & (f0 > 0)))
        if voiced_ratio < 0.10:
            return Applicability(False, "Signal is <10% voiced; pitch is not defined for this input")
    if prop == "speed":
        rate = max(theta, _EPS)
        if duration * (1.0 / rate) > 30.0:
            return Applicability(False, "Stretched duration would exceed the 30 s Whisper window")
    if prop == "time_mask" and duration * theta / 100.0 > duration - 0.25:
        return Applicability(False, "Mask would leave less than 0.25 s of speech")
    if prop == "freq_mask" and theta >= sr / 2:
        return Applicability(False, "Band exceeds the Nyquist frequency")
    return Applicability(True)


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str          # deterministic: sha256(baseline_sha|prop|theta|repeat)[:16]
    property: str
    theta: Any                # float, or (lo, hi) tuple for band-style properties
    repeat: int
    is_control: bool


def render_variant(
    baseline_path: str, spec: VariantSpec, baseline_sha256: str, out_dir: str
) -> Dict[str, Any]:
    """Render one FR-7 grid point from the untouched canonical baseline.

    Always applied to `baseline`, never to a previously rendered variant --
    this is invariant I2 (single-factor variation): exactly one property
    differs between this output and the identity control.
    """
    audio, sr = load_canonical(baseline_path)
    # For a band tuple (freq_mask), the applicability-relevant edge is the
    # HIGH edge (Nyquist check) -- _scalar() alone would check the low edge.
    gate_theta = spec.theta[1] if isinstance(spec.theta, (tuple, list)) else _scalar(spec.theta)
    gate = check_applicable(spec.property, gate_theta, audio, sr)
    if not gate.applicable:
        return {
            "variant_id": spec.variant_id, "applicable": False, "reason": gate.reason,
            "property": spec.property, "theta": spec.theta, "repeat": spec.repeat,
            "is_control": spec.is_control,
        }

    rng = np.random.default_rng(variant_seed(baseline_sha256, spec.property, _scalar(spec.theta), spec.repeat))
    operator = OPERATORS[spec.property]
    param = spec.theta if spec.property == "freq_mask" else _scalar(spec.theta)
    y = operator(audio, sr, param, rng)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    measured_snr = _measured_snr(audio, y) if spec.property == "noise" else None
    y = normalize_loudness(y, sr)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out_dir) / f"{spec.variant_id}.wav")
    import soundfile as sf

    sf.write(out_path, y, sr, subtype="PCM_16")
    digest = hashlib.sha256(Path(out_path).read_bytes()).hexdigest()

    return {
        "variant_id": spec.variant_id, "applicable": True, "path": out_path,
        "property": spec.property, "theta": spec.theta, "repeat": spec.repeat,
        "is_control": spec.is_control,
        "sha256": digest,
        "duration_seconds": len(y) / sr, "sample_rate": sr,
        "measured_snr_db": measured_snr,
        "unit": PROPERTY_UNITS.get(spec.property, ""),
    }
