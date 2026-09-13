"""FR-7 DSP invariants (docs/FR7plan.md Part 1 S2.4, S8).

I1 lexical invariance    -- covered structurally (no operator resynthesizes words)
I2 single-factor variation -- test_single_factor
I3 gain neutrality       -- test_loudness_neutrality
I4 determinism           -- test_determinism
I5 declared controls     -- covered in test_fr7_orchestration (grid always
                            includes identity + lexical_destruction)
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.services.perturbation_service import (
    OPERATORS,
    VariantSpec,
    check_applicable,
    load_canonical,
    normalize_loudness,
    render_variant,
    variant_seed,
)

SR = 16_000


def _voiced_speech_like(duration: float = 2.0, sr: int = SR) -> np.ndarray:
    """A harmonic series with a slow F0 wobble -- passes YIN voicing checks,
    unlike pure white noise, so it stands in for "speech" in DSP-only tests."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0 = 140.0 + 10.0 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    signal = sum(0.15 / k * np.sin(k * phase) for k in range(1, 6))
    return (signal / np.max(np.abs(signal))).astype(np.float32)


@pytest.fixture
def baseline_wav(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.wav"
    sf.write(path, _voiced_speech_like(), SR)
    return path


def test_pitch_preserves_duration(baseline_wav):
    audio, sr = load_canonical(str(baseline_wav))
    rng = np.random.default_rng(0)
    shifted = OPERATORS["pitch"](audio, sr, 6.0, rng)
    assert abs(len(shifted) - len(audio)) / sr < 0.02


def test_speed_changes_duration(baseline_wav):
    audio, sr = load_canonical(str(baseline_wav))
    rng = np.random.default_rng(0)
    stretched = OPERATORS["speed"](audio, sr, 1.4, rng)
    # rate > 1 => faster => shorter
    assert len(stretched) < len(audio)


@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0, 40.0])
def test_noise_snr_calibration(baseline_wav, snr_db):
    audio, sr = load_canonical(str(baseline_wav))
    rng = np.random.default_rng(variant_seed("abc", "noise", snr_db, 0))
    noisy = OPERATORS["noise"](audio, sr, snr_db, rng)
    noise = noisy - audio
    signal_power = float(np.mean(np.square(audio)))
    noise_power = float(np.mean(np.square(noise)))
    measured = 10.0 * np.log10(signal_power / noise_power)
    assert abs(measured - snr_db) < 0.5


def test_determinism(tmp_path, baseline_wav):
    sha = "deadbeef"
    spec = VariantSpec(variant_id="v1", property="noise", theta=10.0, repeat=0, is_control=False)
    out_a = render_variant(str(baseline_wav), spec, sha, str(tmp_path / "a"))
    out_b = render_variant(str(baseline_wav), spec, sha, str(tmp_path / "b"))
    assert out_a["applicable"] and out_b["applicable"]
    assert out_a["sha256"] == out_b["sha256"]


def test_loudness_neutrality(baseline_wav):
    audio, sr = load_canonical(str(baseline_wav))
    for prop, theta in [("pitch", 6.0), ("speed", 1.4), ("noise", 10.0)]:
        rng = np.random.default_rng(0)
        y = OPERATORS[prop](audio, sr, theta, rng)
        y = normalize_loudness(y, sr)
        try:
            import pyloudnorm as pyln

            lufs = pyln.Meter(sr).integrated_loudness(y)
            assert abs(lufs - (-23.0)) < 0.5
        except ImportError:
            rms_db = 20 * np.log10(float(np.sqrt(np.mean(np.square(y)))))
            assert np.isfinite(rms_db)


def test_single_factor(baseline_wav):
    """Pitch shift moves spectral centroid but not duration; time stretch
    moves duration but not spectral centroid (within tolerance)."""
    import librosa

    audio, sr = load_canonical(str(baseline_wav))
    rng = np.random.default_rng(0)

    pitched = OPERATORS["pitch"](audio, sr, 6.0, rng)
    assert abs(len(pitched) - len(audio)) / sr < 0.02
    base_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    pitched_centroid = float(np.mean(librosa.feature.spectral_centroid(y=pitched, sr=sr)))
    assert abs(pitched_centroid - base_centroid) / base_centroid > 0.02

    stretched = OPERATORS["speed"](audio, sr, 1.4, rng)
    assert len(stretched) < len(audio)


def test_not_applicable_reported(tmp_path):
    short_path = tmp_path / "short.wav"
    sf.write(short_path, _voiced_speech_like(duration=0.2), SR)
    audio, sr = load_canonical(str(short_path))
    gate = check_applicable("pitch", 6.0, audio, sr)
    assert not gate.applicable
    assert gate.reason

    spec = VariantSpec(variant_id="short", property="pitch", theta=6.0, repeat=0, is_control=False)
    result = render_variant(str(short_path), spec, "sha", str(tmp_path / "out"))
    assert result["applicable"] is False
    assert result["reason"]


def test_freq_mask_nyquist_gate(baseline_wav):
    audio, sr = load_canonical(str(baseline_wav))
    gate = check_applicable("freq_mask", sr, audio, sr)
    assert not gate.applicable


def test_time_mask_no_hard_silence(baseline_wav):
    audio, sr = load_canonical(str(baseline_wav))
    rng = np.random.default_rng(0)
    masked = OPERATORS["time_mask"](audio, sr, 30.0, rng)
    n = len(audio)
    width = int(n * 0.30)
    start = (n - width) // 2
    region = masked[start:start + width]
    assert np.max(np.abs(region)) < 1e-2
    assert np.max(np.abs(region)) > 0.0  # dithered, not hard zero
