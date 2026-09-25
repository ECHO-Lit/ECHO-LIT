#!/usr/bin/env python3
"""Build the 150-clip L2-ARCTIC subset used by AudioLens from a copy you downloaded.

L2-ARCTIC needs a (free) registration, so this script cannot fetch it for you:

  1. Open https://psi.engr.tamu.edu/l2-arctic-corpus/ and fill in the download
     form (name, email, affiliation), agree to CC BY-NC 4.0, pass the reCAPTCHA.
  2. An automated email titled "Access to L2-ARCTIC corpus" arrives with a Google
     Drive link (check spam). Download and extract it.
  3. Run this script with --source pointing at the extracted folder.

For every row of `scripts/manifests/l2_arctic_metadata.csv` it finds
`<speaker_code>/.../wav/<utt_id>.wav` under --source and copies it to
`Backend/data/L2_ARCTIC_dataset/audio/<filename>` (e.g. `sub_0_ABA.wav`), then
writes `l2_metadata.csv` and `l2_phone_error_annotations.csv` beside it. The
files in `audio/` are copies of the corpus recordings; the *error_type* in the
filename only records which annotated error the row was sampled for.

Usage
-----
    python scripts/prepare_l2arctic.py --dry-run
    python scripts/prepare_l2arctic.py --source /path/to/l2arctic_release_v5.0
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from _dataset_common import DATA_DIR, copy_manifest, die, print_notice, read_manifest, summarize

METADATA = "l2_arctic_metadata.csv"
ANNOTATIONS = "l2_arctic_phone_error_annotations.csv"


def index_source(source: Path, speakers: set[str]) -> dict[tuple[str, str], Path]:
    """Map (speaker_code, utt_id) -> wav path, whatever nesting the release uses."""
    found: dict[tuple[str, str], Path] = {}
    for path in source.rglob("*.wav"):
        if path.parent.name.lower() != "wav":
            continue
        speaker = next((part for part in path.parts if part in speakers), None)
        if speaker:
            found.setdefault((speaker, path.stem), path)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="extracted L2-ARCTIC release (folders named by speaker code)")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "L2_ARCTIC_dataset")
    parser.add_argument("--dry-run", action="store_true", help="show the plan; no source needed")
    args = parser.parse_args()

    print_notice(
        "L2-ARCTIC (non-native English speech corpus)",
        "CC BY-NC 4.0 -- non-commercial only, attribution; other use needs approval from Dr. Ricardo Gutierrez-Osuna",
        "Zhao et al. (2018), 'L2-ARCTIC: A Non-native English Speech Corpus', Interspeech 2018",
        "Registration required: https://psi.engr.tamu.edu/l2-arctic-corpus/",
    )

    rows = read_manifest(METADATA)
    speakers = {row["speaker_code"] for row in rows}
    print(f"{len(rows)} clips in manifest from {len(speakers)} speakers: {', '.join(sorted(speakers))}")

    if args.dry_run:
        print("Dry run: re-run with --source <extracted L2-ARCTIC folder> to build the subset.")
        return 0
    if not args.source:
        return die("--source is required unless --dry-run is given (see the steps at the top of this script)")
    if not args.source.exists():
        return die(f"source not found: {args.source}")

    index = index_source(args.source, speakers)
    if not index:
        return die(f"no <speaker>/wav/*.wav files found under {args.source}")
    print(f"Found {len(index)} corpus wav files for the manifest's speakers")

    out = args.out.resolve()
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = failed = 0
    for row in rows:
        target = audio_dir / row["filename"]
        if target.exists():
            skipped += 1
            continue
        source_wav = index.get((row["speaker_code"], row["utt_id"]))
        if source_wav is None:
            failed += 1
            print(f"  missing {row['speaker_code']}/{row['utt_id']}.wav (for {row['filename']})", file=sys.stderr)
            continue
        shutil.copy2(source_wav, target)
        written += 1

    copy_manifest(METADATA, out / "l2_metadata.csv")
    copy_manifest(ANNOTATIONS, out / "l2_phone_error_annotations.csv")
    return summarize("L2-ARCTIC", written, skipped, failed, out)


if __name__ == "__main__":
    raise SystemExit(main())
