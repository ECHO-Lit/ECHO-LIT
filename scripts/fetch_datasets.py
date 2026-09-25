#!/usr/bin/env python3
"""Run the dataset scripts that need no account, in one go.

    python scripts/fetch_datasets.py            # RAVDESS + SAA + LibriSpeech-1000
    python scripts/fetch_datasets.py --dry-run

Not included, because each needs something only you can provide:
  * L2-ARCTIC    -> registration, then `prepare_l2arctic.py --source <folder>`
  * Common Voice -> Kaggle account, then `prepare_common_voice.py --kaggle|--source`
  * SAVEE        -> a copy you already own, then `prepare_savee_subset.py --source`

Everything lands in Backend/data, which persists until you delete it (uploaded
datasets and sessions, by contrast, expire after 24 hours).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ("RAVDESS", "download_ravdess.py", True),
    ("Speech Accent Archive", "download_saa.py", True),
    ("LibriSpeech-1000", "download_librispeech_1000.py", False),  # no --dry-run flag
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    failures = []
    for label, script, supports_dry_run in STEPS:
        if args.dry_run and not supports_dry_run:
            print(f"\n== {label}: skipped in --dry-run ({script} has no dry-run)")
            continue
        print(f"\n== {label}")
        command = [sys.executable, str(HERE / script)] + (["--dry-run"] if args.dry_run else [])
        if subprocess.run(command).returncode != 0:
            failures.append(label)

    print("\nStill manual: L2-ARCTIC, Common Voice, SAVEE (see docs/datasets).")
    if failures:
        print("Failed:", ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
