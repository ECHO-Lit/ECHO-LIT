"""Pure-librosa acoustic feature extraction.

Deliberately kept out of `model_loader_service`, which imports torch, transformers,
sklearn and umap at module scope. This function needs none of them, and the
`audio_features` job runs on the CPU worker under Celery's *prefork* pool -- loading
torch's OpenMP runtime and numba's LLVM runtime (via umap) into the same forked
child is a known SIGSEGV. Importing only librosa and numpy here keeps that worker
free of both.

Do not add a torch/transformers/umap import to this module.
"""

import librosa
import numpy as np


def _zero_crossing_rate(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Drop-in replacement for `librosa.feature.zero_crossing_rate(y)[0]`.

    librosa's version calls into a numba guvectorize kernel (`_zc_wrapper`) that
    segfaults in this container regardless of the NUMBA_CPU_NAME/NUMBA_THREADING_LAYER
    overrides in docker-compose.yml -- those were verified against beat_track/yin only.
    This reimplements the exact default-argument semantics (threshold=1e-10, zero_pos=True,
    pad=False, center=True) in plain numpy, byte-for-byte matched against librosa's output.
    """
    pad = frame_length // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    frames = librosa.util.frame(padded, frame_length=frame_length, hop_length=hop_length)
    clipped = np.where(np.abs(frames) <= 1e-10, 0.0, frames)
    sign = np.sign(clipped)
    sign[sign == 0] = 1.0
    crossings = sign[1:, :] != sign[:-1, :]
    full = np.vstack([np.zeros((1, crossings.shape[1]), dtype=bool), crossings])
    return np.mean(full, axis=0, keepdims=True)[0]


def extract_audio_frequency_features(audio_file_path: str) -> dict:
    """
    Extract comprehensive frequency-domain audio features using librosa.
    
    Args:
        audio_file_path: Path to audio file
    
    Returns:
        Dictionary containing various audio frequency features
    """
    # Load audio with standard sample rate
    audio, sr = librosa.load(audio_file_path, sr=22050)
    
    # Extract various frequency features
    features = {}
    
    # Basic spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    zero_crossing_rate = _zero_crossing_rate(audio)
    
    # MFCC features (first 13 coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Chroma features (pitch class profiles)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    # Tonnetz (tonal centroid features)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
    
    # Tempo and beat tracking
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    
    # RMS Energy
    rms = librosa.feature.rms(y=audio)[0]
    
    # Calculate statistics for each feature
    features = {
        "spectral_centroid": {
            "mean": float(np.mean(spectral_centroids)),
            "std": float(np.std(spectral_centroids)),
            "min": float(np.min(spectral_centroids)),
            "max": float(np.max(spectral_centroids))
        },
        "spectral_rolloff": {
            "mean": float(np.mean(spectral_rolloff)),
            "std": float(np.std(spectral_rolloff)),
            "min": float(np.min(spectral_rolloff)),
            "max": float(np.max(spectral_rolloff))
        },
        "spectral_bandwidth": {
            "mean": float(np.mean(spectral_bandwidth)),
            "std": float(np.std(spectral_bandwidth)),
            "min": float(np.min(spectral_bandwidth)),
            "max": float(np.max(spectral_bandwidth))
        },
        "zero_crossing_rate": {
            "mean": float(np.mean(zero_crossing_rate)),
            "std": float(np.std(zero_crossing_rate)),
            "min": float(np.min(zero_crossing_rate)),
            "max": float(np.max(zero_crossing_rate))
        },
        "rms_energy": {
            "mean": float(np.mean(rms)),
            "std": float(np.std(rms)),
            "min": float(np.min(rms)),
            "max": float(np.max(rms))
        },
        "mfcc": {
            f"mfcc_{i+1}_mean": float(np.mean(mfccs[i])) for i in range(13)
        },
        "chroma": {
            f"chroma_{i+1}_mean": float(np.mean(chroma[i])) for i in range(12)
        },
        "tonnetz": {
            f"tonnetz_{i+1}_mean": float(np.mean(tonnetz[i])) for i in range(6)
        },
        "tempo": float(tempo),
        "duration": float(len(audio) / sr),
        "sample_rate": int(sr)
    }
    
    # Flatten the nested structure for easier processing
    flattened_features = {}
    for key, value in features.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flattened_features[f"{key}_{subkey}"] = subvalue
        else:
            flattened_features[key] = value
    
    return flattened_features
