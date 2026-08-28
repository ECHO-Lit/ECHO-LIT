#!/usr/bin/env python3
"""Download N LibriSpeech clips (clean validation split, speech + text)
from the public HF datasets-server API into a local dataset directory
plus a metadata.csv that matches the app's bundled dataset schema."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "openslr/librispeech_asr"
CONFIG = "all"
SPLIT = "validation.clean"
USER_AGENT = "echo-dataset-downloader/1.0"


def fetch_rows(offset: int, length: int) -> dict:
    query = urllib.parse.urlencode({
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": str(offset),
        "length": str(length),
    })
    request = urllib.request.Request(f"{ROWS_URL}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def collect_samples(count: int) -> tuple[list[dict], int]:
    samples: list[dict] = []
    page = 100
    offset = 0
    total = None
    while len(samples) < count:
        payload = fetch_rows(offset, page)
        if "rows" not in payload or "num_rows_total" not in payload:
            raise RuntimeError(f"Unexpected datasets-server response: {payload}")
        total = int(payload["num_rows_total"])
        rows = payload["rows"]
        if not rows:
            break
        for entry in rows:
            row = entry.get("row", {})
            audio_list = row.get("audio") or []
            audio = audio_list[0] if audio_list else {}
            src = (audio.get("src") or "").strip()
            text = (row.get("text") or "").strip()
            raw_file = (row.get("file") or "").strip()
            if not src or not text or not raw_file:
                continue
            filename = Path(raw_file.replace("\\", "/")).name
            if not filename:
                continue
            samples.append({"filename": filename, "text": text, "src": src})
            if len(samples) >= count:
                break
        offset += len(rows)
        if total is not None and offset >= total:
            break
        time.sleep(0.05)
    return samples, int(total or 0)


def download_one(sample: dict, out_dir: Path) -> Path:
    request = urllib.request.Request(sample["src"], headers={"User-Agent": USER_AGENT})
    target = out_dir / sample["filename"]
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if target.exists():
                existing = target.read_bytes()
                if existing == data:
                    return target
                target = out_dir / f"{target.stem}-{attempt+1}{target.suffix}"
            target.write_bytes(data)
            return target
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("Backend/data/librispeech_1000"))
    parser.add_argument("--metadata", type=Path, default=Path("Backend/data/librispeech_1000/librispeech_1000_metadata.csv"))
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching up to {args.count} rows from {DATASET} ({CONFIG}/{SPLIT})...", flush=True)
    samples, total = collect_samples(args.count)
    if len(samples) < args.count:
        print(f"Only {len(samples)} usable rows found (total split rows: {total})", flush=True)
    print(f"Downloading {len(samples)} audio clips to {out_dir} ...", flush=True)

    written: list[str] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_one, s, out_dir): s for s in samples}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                target = future.result()
                written.append(sample["filename"])
                print(f"  ok {target.name}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"  failed {sample['filename']}: {exc}", file=sys.stderr, flush=True)

    if not written:
        print("No audio clips downloaded; aborting metadata generation", file=sys.stderr, flush=True)
        return 1

    by_name = {s["filename"]: s["text"] for s in samples}
    metadata_path = args.metadata.resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "text", "age", "gender", "accent", "duration", "up_votes", "down_votes", "client_id", "locale"])
        for filename in sorted(written):
            writer.writerow([filename, by_name[filename], "", "", "", "", "", "", "", ""])

    print(f"Wrote {len(written)} clips + metadata.csv ({failed} failed)", flush=True)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())