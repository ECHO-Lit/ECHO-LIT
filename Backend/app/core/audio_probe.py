from __future__ import annotations

import json
from pathlib import Path
import subprocess


def probe_audio(path: Path) -> tuple[float, int | None, int | None]:
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
        try:
            import soundfile

            info = soundfile.info(str(path))
            if info.duration <= 0:
                raise ValueError("Audio duration could not be determined")
            return float(info.duration), int(info.samplerate), int(info.channels)
        except Exception as fallback_exc:
            raise ValueError("The file is not decodable audio") from fallback_exc
