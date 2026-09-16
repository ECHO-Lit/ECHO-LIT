"""Shared factories for the Section 3.1.2 function-test modules.

Deliberately a plain helper module rather than a conftest: conftest fixtures
apply to the whole suite, and the storage/dataset/broker isolation these modules
need would change the environment for the ~400 tests that predate them.  Modules
import what they need explicitly instead.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf


def wav_bytes(seconds: float = 0.25, sr: int = 16_000, freq: float = 440.0) -> bytes:
    """A small, real, decodable mono WAV.

    conftest's `sample_audio_file` is 5 seconds and is rebuilt per test; the
    dataset modules upload three to five files per case and need neither the
    length nor the cost.  A tone rather than silence, so anything that probes or
    analyses it sees a signal.
    """
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    signal = (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32")
    buffer = io.BytesIO()
    sf.write(buffer, signal, sr, format="WAV")
    return buffer.getvalue()


def upload_files(names, *, seconds: float = 0.25):
    """Build a multipart `files` list for httpx from a sequence of filenames."""
    return [
        ("files", (name, io.BytesIO(wav_bytes(seconds=seconds)), "audio/wav")) for name in names
    ]
