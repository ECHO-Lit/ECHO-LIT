#!/usr/bin/env python3
"""Fetch the 144-clip RAVDESS subset used by AudioLens from the official Zenodo record.

No account is needed. The script downloads `Audio_Speech_Actors_01-24.zip`
(~208 MB) from https://zenodo.org/records/1188976, extracts only the clips
listed in `scripts/manifests/ravdess_subset.csv`, and writes them plus the
metadata CSV to `Backend/data/ravdess_subset/`.

Usage
-----
    python scripts/download_ravdess.py --dry-run
    python scripts/download_ravdess.py
    python scripts/download_ravdess.py --zip ~/Downloads/Audio_Speech_Actors_01-24.zip   # already downloaded
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

from _dataset_common import DATA_DIR, copy_manifest, die, download, print_notice, read_manifest, summarize

ZIP_URL = "https://zenodo.org/api/records/1188976/files/Audio_Speech_Actors_01-24.zip/content"
MANIFEST = "ravdess_subset.csv"
METADATA_NAME = "ravdess_subset_metadata.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "ravdess_subset")
    parser.add_argument("--zip", type=Path, help="use an already-downloaded Audio_Speech_Actors_01-24.zip")
    parser.add_argument("--dry-run", action="store_true", help="show the plan; download nothing")
    args = parser.parse_args()

    print_notice(
        "RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)",
        "CC BY-NC-SA 4.0 -- non-commercial only, attribution, share-alike",
        "Livingstone & Russo (2018), PLoS ONE 13(5): e0196391. https://zenodo.org/records/1188976",
        "Commercial use needs a purchased licence (ravdess@gmail.com).",
    )

    rows = read_manifest(MANIFEST)
    wanted = {row["filename"] for row in rows}
    out = args.out.resolve()
    missing = sorted(name for name in wanted if not (out / name).exists())
    print(f"{len(wanted)} clips in manifest, {len(missing)} not yet in {out}")

    if args.dry_run:
        print("Dry run: would download", ZIP_URL, "(~208 MB) and extract the missing clips.")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    copy_manifest(MANIFEST, out / METADATA_NAME)
    if not missing:
        return summarize("RAVDESS", 0, len(wanted), 0, out)

    written = 0
    with tempfile.TemporaryDirectory(prefix="ravdess_") as tmp:
        zip_path = args.zip
        if zip_path is None:
            zip_path = Path(tmp) / "Audio_Speech_Actors_01-24.zip"
            print(f"Downloading {ZIP_URL}")
            download(ZIP_URL, zip_path, timeout=300, show_progress=True)
        elif not zip_path.exists():
            return die(f"zip not found: {zip_path}")

        need = set(missing)
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                name = Path(member).name
                if name in need:
                    with archive.open(member) as src, (out / name).open("wb") as dst:
                        dst.write(src.read())
                    need.discard(name)
                    written += 1
        if need:
            print(f"warning: {len(need)} manifest clips not found in the zip, e.g. {sorted(need)[:3]}", file=sys.stderr)
        return summarize("RAVDESS", written, len(wanted) - len(missing), len(need), out)


if __name__ == "__main__":
    raise SystemExit(main())
