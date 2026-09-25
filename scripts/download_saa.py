#!/usr/bin/env python3
"""Fetch the 150-clip Speech Accent Archive subset used by AudioLens.

No account is needed. Clips come from the archive's own site
(https://accent.gmu.edu/audio/<name>.mp3), one request per clip with a short
delay between them. The selection is `scripts/manifests/saa_metadata.csv`; the
script writes the mp3 files to `Backend/data/SAA_dataset/audio/` and the
metadata CSV to `Backend/data/SAA_dataset/saa_metadata.csv`.

Two clips (russian4, spanish159) have been re-uploaded on the archive since the
original subset was made, so they differ byte-for-byte from older local copies.

The full archive is also on OSF (see https://accent.gmu.edu/download); use that
instead if you want more than this subset.

Usage
-----
    python scripts/download_saa.py --dry-run
    python scripts/download_saa.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from _dataset_common import DATA_DIR, copy_manifest, download, print_notice, read_manifest, summarize

AUDIO_URL = "https://accent.gmu.edu/audio/{name}.mp3"
MANIFEST = "saa_metadata.csv"
DELAY_SECONDS = 0.25


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "SAA_dataset")
    parser.add_argument("--dry-run", action="store_true", help="show the plan; download nothing")
    args = parser.parse_args()

    print_notice(
        "Speech Accent Archive (George Mason University)",
        "CC BY-NC-SA 4.0 -- non-commercial only, attribution, share-alike",
        "Weinberger, S. H. (2015). Speech Accent Archive. George Mason University. https://accent.gmu.edu",
    )

    rows = read_manifest(MANIFEST)
    names = [row["filename"].removesuffix(".mp3") for row in rows]
    audio_dir = args.out.resolve() / "audio"
    todo = [name for name in names if not (audio_dir / f"{name}.mp3").exists()]
    print(f"{len(names)} clips in manifest, {len(todo)} not yet in {audio_dir}")

    if args.dry_run:
        print("Dry run: would fetch e.g.", AUDIO_URL.format(name=todo[0] if todo else names[0]))
        return 0

    copy_manifest(MANIFEST, args.out.resolve() / "saa_metadata.csv")
    failed = 0
    for index, name in enumerate(todo, start=1):
        try:
            download(AUDIO_URL.format(name=name), audio_dir / f"{name}.mp3")
            print(f"  [{index}/{len(todo)}] ok {name}.mp3", flush=True)
        except Exception as exc:
            failed += 1
            print(f"  [{index}/{len(todo)}] failed {name}: {exc}", file=sys.stderr, flush=True)
        time.sleep(DELAY_SECONDS)
    return summarize("Speech Accent Archive", len(todo) - failed, len(names) - len(todo), failed, args.out.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
