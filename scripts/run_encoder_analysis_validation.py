"""Run the encoder structural analysis offline, without Docker, Celery or Redis.

The bench version of the `encoder_analysis` operation. It answers the two
sanity questions the feature's value rests on:

1. Do early encoder layers show the expected diagonal attention band (local
   acoustic processing), with mass spreading outward at greater depth?
2. Does the CKA layer curve show structure -- a drop or plateau -- rather than
   a uniform ~1.0, i.e. does depth actually transform the representation?

Usage, from `Backend/`:

    python ../scripts/run_encoder_analysis_validation.py --dataset ravdess --limit 4
    python ../scripts/run_encoder_analysis_validation.py --audio path/to/clip.wav

Results are cached to a .npz-free JSON beside the output; the computation is a
single encoder forward pass per file, so re-running is cheap.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1] / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

DATASET_DIRS = {
    "ravdess": BACKEND / "data" / "ravdess_subset",
    "common-voice": BACKEND / "data" / "common_voice_valid_dev",
}


def dataset_files(dataset: str, limit: int) -> list[Path]:
    directory = DATASET_DIRS[dataset]
    wav_files = sorted(
        path for path in directory.rglob("*") if path.suffix.lower() == ".wav"
    )
    if not wav_files:
        raise SystemExit(f"No .wav files under {directory}")
    return wav_files[:limit]


def run_file(path: Path, model_size: str, max_frames: int, n_bins: int) -> dict:
    from app.services.encoder_analysis_service import run_encoder_analysis

    start = time.time()
    result = run_encoder_analysis(
        str(path), model_size=model_size, max_encoder_frames=max_frames, n_bins=n_bins
    )
    result["file"] = str(path)
    result["seconds"] = round(time.time() - start, 2)
    return result


def print_sanity_checks(results: list[dict]) -> None:
    """Print the two signatures that decide whether the feature earns a UI."""
    first = results[0]
    profiles = first["attention_profiles"]
    diag = np.array([layer["diagonal_mass"] for layer in profiles["layers"]])
    mean_diag = diag.mean(axis=1)

    print("\n=== Signature 1: attention distance profiles (file 1) ===")
    for index, value in enumerate(mean_diag):
        print(f"  layer {index}: mean diagonal mass (<=+-2 frames) = {value:.3f}")
    early_band = mean_diag[0] > 0.3
    spreads = mean_diag[-1] <= mean_diag[0]
    print(
        f"  early-layer diagonal band: {'PRESENT' if early_band else 'ABSENT'}"
        f" | mass spreads with depth: {'YES' if spreads else 'NO'}"
    )

    cka = np.array(first["cka"]["adjacent_cka"])
    print("\n=== Signature 2: CKA between adjacent layers (file 1) ===")
    names = first["cka"]["layer_names"]
    for index, value in enumerate(cka):
        print(f"  {names[index]} -> {names[index + 1]}: {value:.3f}")
    structured = (cka.max() - cka.min()) > 0.05
    print(
        f"  curve range [{cka.min():.3f}, {cka.max():.3f}] -> "
        f"{'STRUCTURED' if structured else 'FLAT (layers are near-duplicates)'}"
    )

    print("\n=== Cross-file stability (mean diagonal mass, layer 0) ===")
    layer0 = [np.mean(r["attention_profiles"]["layers"][0]["diagonal_mass"]) for r in results]
    for r, value in zip(results, layer0):
        print(f"  {Path(r['file']).name}: {value:.3f}")
    stable = np.std(layer0) < 0.15
    print(f"  std {np.std(layer0):.3f} -> {'STABLE' if stable else 'UNSTABLE across files'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_DIRS), default="ravdess")
    parser.add_argument("--audio", action="append", default=[], help="Explicit wav path(s)")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--model", default="whisper-base", choices=["whisper-base", "whisper-large"])
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--out", default=str(BACKEND / "docs" / "analysis" / "encoder_analysis_validation.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.audio:
        files = [Path(p) for p in args.audio]
    else:
        files = dataset_files(args.dataset, args.limit)

    model_size = "large" if args.model == "whisper-large" else "base"
    results = []
    for path in files:
        print(f"Analyzing {path.name} ...")
        results.append(run_file(path, model_size, args.max_frames, args.bins))

    print_sanity_checks(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump({"model": args.model, "files": results}, handle, indent=1)
    print(f"\nFull payload written to {out_path}")


if __name__ == "__main__":
    main()
