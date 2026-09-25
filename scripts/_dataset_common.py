"""Shared helpers for the dataset fetch/prepare scripts.

These scripts never ship audio. Each one takes a selection manifest from
`scripts/manifests/`, gets the clips from the dataset's official source, and
writes them into `Backend/data/<dir>` in the layout the app already reads
(see `Backend/app/services/dataset_service.py`). Files in `Backend/data` are
plain host files mounted read-only into the containers, so they persist until
you delete them -- unlike uploaded datasets and sessions, which expire after
24 hours.
"""
from __future__ import annotations

import csv
import shutil
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "Backend" / "data"
MANIFESTS = Path(__file__).resolve().parent / "manifests"
USER_AGENT = "echo-dataset-downloader/1.0"


def read_manifest(name: str) -> list[dict[str, str]]:
    path = MANIFESTS / name
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def copy_manifest(name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MANIFESTS / name, dest)


def print_notice(dataset: str, licence: str, citation: str, extra: str = "") -> None:
    """Every script prints the licence and citation before touching anything."""
    bar = "-" * 72
    print(bar)
    print(f"{dataset}")
    print(f"  licence : {licence}")
    print(f"  cite    : {citation}")
    if extra:
        print(f"  note    : {extra}")
    print(bar, flush=True)


def download(url: str, target: Path, retries: int = 3, timeout: int = 120, show_progress: bool = False) -> None:
    """Stream `url` to `target` via a .part file so a failed run leaves no half-written clip."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                next_mark = 10
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if show_progress and total and done * 100 // total >= next_mark:
                        print(f"  {next_mark}%  ({done / 1e6:.0f} / {total / 1e6:.0f} MB)", flush=True)
                        next_mark += 10
            part.replace(target)
            return
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            time.sleep(1 + attempt)


def summarize(dataset: str, written: int, skipped: int, failed: int, out: Path) -> int:
    print(f"\n{dataset}: {written} written, {skipped} already present, {failed} failed -> {out}", flush=True)
    return 0 if failed == 0 else 2


def die(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1
