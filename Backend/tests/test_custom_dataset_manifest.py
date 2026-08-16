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
