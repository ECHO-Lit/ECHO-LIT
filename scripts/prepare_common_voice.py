#!/usr/bin/env python3
"""Build the 100-clip Common Voice `cv-valid-dev` subset used by AudioLens.

The subset uses the Kaggle release of Mozilla Common Voice (v1, 500 h,
`mozillaorg/common-voice`), where clips are named `cv-valid-dev/sample-NNNNNN.mp3`.
Kaggle needs a free account, so pick one route:

  A. You already have the data (Kaggle download, extracted):
         python scripts/prepare_common_voice.py --source /path/to/common-voice
  B. Let the script pull just the 100 clips with the Kaggle CLI. Install it
     (`pip install kaggle`), create an API token at kaggle.com > Settings > API,
     save it as ~/.kaggle/kaggle.json, accept the dataset's terms once in the
     browser, then:
         python scripts/prepare_common_voice.py --kaggle

Clips are written flat to `Backend/data/common_voice_valid_dev/` (the app looks
files up by base name) with `common_voice_valid_data_metadata.csv` beside them.

Note: newer Common Voice releases (23.0 onward) are distributed through the
Mozilla Data Collective and use different file names, so they do not match this
manifest; use custom-dataset upload for those.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from _dataset_common import DATA_DIR, copy_manifest, die, print_notice, read_manifest, summarize

MANIFEST = "cv_valid_dev.csv"
METADATA_NAME = "common_voice_valid_data_metadata.csv"
KAGGLE_DATASET = "mozillaorg/common-voice"


def fetch_with_kaggle(name: str, dest: Path) -> bool:
    """Download one clip via the kaggle CLI; returns True when `dest` exists afterwards."""
    with tempfile.TemporaryDirectory(prefix="cv_") as tmp:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-f", f"cv-valid-dev/{name}", "-p", tmp],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"    kaggle: {result.stderr.strip() or result.stdout.strip()}", file=sys.stderr)
            return False
        for zipped in Path(tmp).glob("*.zip"):
            with zipfile.ZipFile(zipped) as archive:
                archive.extractall(tmp)
        hit = next(Path(tmp).rglob(name), None)
        if hit is None:
            return False
        shutil.copy2(hit, dest)
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source", type=Path, help="extracted Kaggle common-voice folder (contains cv-valid-dev/)")
    source.add_argument("--kaggle", action="store_true", help="fetch the 100 clips with the kaggle CLI")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "common_voice_valid_dev")
    parser.add_argument("--dry-run", action="store_true", help="show the plan; nothing needed")
    args = parser.parse_args()

    print_notice(
        "Mozilla Common Voice (Kaggle v1, cv-valid-dev)",
        "CC0 (public domain) per Mozilla; check the dataset page for current terms. Do not try to identify speakers.",
        "Ardila et al. (2020), 'Common Voice: A Massively-Multilingual Speech Corpus', LREC 2020",
        "Kaggle account needed: https://www.kaggle.com/datasets/mozillaorg/common-voice",
    )

    rows = read_manifest(MANIFEST)
    names = [Path(row["filename"]).name for row in rows]
    out = args.out.resolve()
    todo = [name for name in names if not (out / name).exists()]
    print(f"{len(names)} clips in manifest, {len(todo)} not yet in {out}")

    if args.dry_run:
        print("Dry run: re-run with --source <folder> or --kaggle to build the subset.")
        return 0
    if not args.source and not args.kaggle:
        return die("pass --source <folder> or --kaggle (see the top of this script)")

    out.mkdir(parents=True, exist_ok=True)
    written = failed = 0
    if args.source:
        if not args.source.exists():
            return die(f"source not found: {args.source}")
        index = {path.name: path for path in args.source.rglob("sample-*.mp3")}
        for name in todo:
            if name in index:
                shutil.copy2(index[name], out / name)
                written += 1
            else:
                failed += 1
                print(f"  missing {name}", file=sys.stderr)
    else:
        if shutil.which("kaggle") is None:
            return die("kaggle CLI not found; run `pip install kaggle` and add your API token (see top of script)")
        for index, name in enumerate(todo, start=1):
            if fetch_with_kaggle(name, out / name):
                written += 1
                print(f"  [{index}/{len(todo)}] ok {name}", flush=True)
            else:
                failed += 1
                print(f"  [{index}/{len(todo)}] failed {name}", file=sys.stderr, flush=True)

    copy_manifest(MANIFEST, out / METADATA_NAME)
    return summarize("Common Voice", written, len(names) - len(todo), failed, out)


if __name__ == "__main__":
    raise SystemExit(main())
