from __future__ import annotations

import json
from pathlib import Path
import subprocess


def _probe_with_soundfile(path: Path) -> tuple[float, int | None, int | None] | None:
    try:
        import soundfile

        info = soundfile.info(str(path))
    except Exception:
        return None
    if info.duration <= 0:
        return None
    return float(info.duration), int(info.samplerate), int(info.channels)


def probe_audio(path: Path) -> tuple[float, int | None, int | None]:
    # libsndfile reads the header in-process (~1 ms) and covers WAV, FLAC, OGG
    # and MP3. ffprobe is a process spawn (~80 ms on the development host) and
    # is only needed for containers libsndfile cannot open, such as M4A. It ran
    # first for every upload, which made probing the dominant cost of an upload
    # acknowledgment under a burst of concurrent uploads. The API image ships
    # without ffmpeg, so production already took the libsndfile path.
    probed = _probe_with_soundfile(path)
    if probed is not None:
        return probed
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        payload = json.loads(completed.stdout)
        streams = payload.get("streams") or []
        if not streams:
            raise ValueError("No audio stream found")
        stream = streams[0]
        duration = float((payload.get("format") or {}).get("duration") or 0)
        if duration <= 0:
            raise ValueError("Audio duration could not be determined")
        sample_rate = int(stream["sample_rate"]) if stream.get("sample_rate") else None
        channels = int(stream["channels"]) if stream.get("channels") else None
        return duration, sample_rate, channels
    except (subprocess.SubprocessError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("The file is not decodable audio") from exc
