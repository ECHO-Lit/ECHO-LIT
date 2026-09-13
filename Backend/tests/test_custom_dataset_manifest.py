from __future__ import annotations

import pytest


def test_manifest_exposes_only_matched_transcript_pairs(tmp_path, monkeypatch):
    from app.services import custom_dataset_service as datasets

    monkeypatch.setattr(datasets, "SESSIONS_BASE_DIR", tmp_path / "sessions")
    manager = datasets.CustomDatasetManager("session-a")
    manager.create_dataset("speech")
    manager.add_file_to_dataset("speech", "clip-001.wav", b"not-a-real-wave")

    result = manager.add_manifest_to_dataset(
        "speech",
        "metadata.csv",
        b"filename,transcript\nfolder/clip-001.wav,hello there\nclip-002.wav,not uploaded\n",
    )

    assert result["pair_count"] == 2
    assert result["matched_audio_count"] == 1
    rows = manager.get_dataset_files_as_csv_format("speech")
    assert len(rows) == 1
    assert rows[0]["filename"] == "clip-001.wav"
    assert rows[0]["transcript"] == "hello there"


def test_manifest_requires_supported_pair_columns(tmp_path, monkeypatch):
    from app.services import custom_dataset_service as datasets

    monkeypatch.setattr(datasets, "SESSIONS_BASE_DIR", tmp_path / "sessions")
    manager = datasets.CustomDatasetManager("session-a")
    manager.create_dataset("speech")

    with pytest.raises(ValueError, match="filename column"):
        manager.add_manifest_to_dataset("speech", "metadata.csv", b"audio,label\nclip.wav,hello\n")


async def test_upload_accepts_octet_stream_audio(client):
    """Browsers label .flac/.m4a as application/octet-stream; /upload accepts it
    and the dataset-files route must not reject it (regression: the reference
    all-FLAC dataset could never upload)."""
    payload = b"fake-flac-bytes"
    response = await client.post(
        "/upload/dataset/create",
        data={"dataset_name": "flac-set"},
    )
    assert response.status_code == 201

    response = await client.post(
        "/upload/dataset/flac-set/files",
        files=[("files", ("1673-143396-0000.flac", payload, "application/octet-stream"))],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_files"] == 1
    assert body["uploaded_files"][0]["filename"] == "1673-143396-0000.flac"


async def test_upload_still_rejects_non_audio(client):
    await client.post("/upload/dataset/create", data={"dataset_name": "guarded"})
    response = await client.post(
        "/upload/dataset/guarded/files",
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
    )
    assert response.status_code == 400


async def test_custom_dataset_upload_manifest_metadata_roundtrip(client, tmp_path, monkeypatch):
    """Create -> upload audio -> attach metadata.csv -> read rows back with transcripts."""
    from app.core import settings as settings_module
    from app.services import custom_dataset_service as datasets

    monkeypatch.setattr(datasets, "SESSIONS_BASE_DIR", tmp_path / "sessions")
    monkeypatch.setattr(settings_module.settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "storage"))

    import io

    import numpy as np
    import soundfile as sf

    def tiny_wav(seconds: float = 0.2) -> bytes:
        rate = 16_000
        t = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False)
        signal = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, signal, rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    audio_one, audio_two = tiny_wav(), tiny_wav()

    await client.post("/upload/dataset/create", data={"dataset_name": "roundtrip"})
    await client.post(
        "/upload/dataset/roundtrip/files",
        files=[
            ("files", ("clip-001.wav", audio_one, "application/octet-stream")),
            ("files", ("clip-002.wav", audio_two, "audio/wav")),
        ],
    )
    manifest = (
        b"filename,text\n"
        b"clip-001.wav,first transcript\n"
        b"clip-002.wav,second transcript\n"
        b"missing.wav,never uploaded\n"
    )
    response = await client.post(
        "/upload/dataset/roundtrip/manifest",
        files={"manifest": ("metadata.csv", manifest, "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["manifest"]["matched_audio_count"] == 2

    formatted = response.json()["dataset_name"]
    from urllib.parse import quote

    response = await client.get(f"/{quote(formatted)}/metadata")
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 2
    transcripts = {row["filename"]: row["transcript"] for row in rows}
    assert transcripts["clip-001.wav"] == "first transcript"
    assert transcripts["clip-002.wav"] == "second transcript"

    response = await client.post(
        "/audio/materialize",
        json={"dataset": formatted, "filename": "clip-001.wav"},
    )
    assert response.status_code == 201
    audio_id = response.json()["audio_id"]
    response = await client.get(f"/audio/{audio_id}")
    assert response.status_code == 200
    assert response.content == audio_one
