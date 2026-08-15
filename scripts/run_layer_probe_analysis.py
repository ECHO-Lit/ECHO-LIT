"""Run the layer-wise probing analysis offline, without Docker, Celery or Redis.

This is the reproducible bench version of what the "Layer Probes" tab does in the
app.  It exists for two reasons: it is the acceptance test for the feature's
actual scientific claim (speaker and gender peak *earlier* in the encoder than
lexical content and emotion), and it produces the numbers and the JSON a written
write-up needs without standing the stack up.

Usage, from `Backend/`:

    python ../scripts/run_layer_probe_analysis.py --dataset ravdess --model whisper-base

Activations are cached to a .npz beside the output, so re-running with different
probe settings costs no model time -- the same property the item cache gives the
real job.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

BACKEND = Path(__file__).resolve().parents[1] / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Property name -> metadata column supplying its label, per dataset. Mirrors
# PROBE_PROPERTIES in Frontend/src/lib/probes.ts; keep the two in step.
DATASET_PROPERTIES: dict[str, dict[str, str]] = {
    "ravdess": {
        "gender": "gender",
        "speaker": "actor",
        "emotion": "emotion",
        "lexical": "statement",
        "intensity": "intensity",
    },
    "common-voice": {
        "gender": "gender",
        "accent": "accent",
        "age": "age",
    },
}

DATASET_DIRS = {
    "ravdess": BACKEND / "data" / "ravdess_subset",
    "common-voice": BACKEND / "data" / "common_voice_valid_dev",
}


def load_rows(dataset: str) -> list[dict[str, str]]:
    directory = DATASET_DIRS[dataset]
    csv_path = next(directory.glob("*_metadata.csv"))
    with csv_path.open(encoding="utf-8") as handle:
        return [
            {key.strip().lower(): (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def extract(dataset: str, model: str, rows: list[dict[str, str]], noise_snr_db: float | None):
    from app.worker.model_adapters import get_model_adapter

    adapter = get_model_adapter(model)
    directory = DATASET_DIRS[dataset]
    clean: list[list[list[float]]] = []
    noisy: list[list[list[float]]] = []
    kept: list[dict[str, str]] = []
    started = time.monotonic()
    parameters = {} if noise_snr_db is None else {"noise_snr_db": noise_snr_db}
    for index, row in enumerate(rows, start=1):
        audio_path = directory / Path(row["filename"]).name
        if not audio_path.exists():
            logging.warning("missing audio, skipping: %s", audio_path.name)
            continue
        payload = adapter.execute("hidden_states", str(audio_path), parameters)
        clean.append(payload["layers"])
        if payload.get("noisy_layers"):
            noisy.append(payload["noisy_layers"])
        kept.append(row)
        if index % 10 == 0 or index == len(rows):
            elapsed = time.monotonic() - started
            print(f"  {index}/{len(rows)} files  ({elapsed:.0f}s)", flush=True)
    return np.array(clean), (np.array(noisy) if noisy else None), kept, payload["layer_names"]


def build_properties(dataset: str, rows: list[dict[str, str]]) -> dict[str, list[str | None]]:
    mapping = DATASET_PROPERTIES[dataset]
    return {
        name: [row.get(column) or None for row in rows]
        for name, column in mapping.items()
        if any(row.get(column) for row in rows)
    }


def report(result: dict) -> None:
    num_layers = result["num_layers"]
    print()
    print(f"{'property':<10} {'peak':>5} {'depth':>6} {'acc':>6} {'major':>6} {'select':>7} {'F1':>6}  n")
    print("-" * 64)
    ranked = sorted(
        (
            (name, probe) for name, probe in result["properties"].items()
            if probe["best_layer"] is not None
        ),
        key=lambda entry: entry[1]["peak_depth"],
    )
    for name, probe in ranked:
        best = probe["layers"][probe["best_layer"]]
        print(
            f"{name:<10} {probe['best_layer']:>2}/{num_layers - 1:<2} "
            f"{probe['peak_depth']:>6.2f} {probe['best_accuracy']:>6.3f} "
            f"{probe['majority_baseline']:>6.3f} "
            f"{(best['selectivity'] if best['selectivity'] is not None else float('nan')):>7.3f} "
            f"{best['macro_f1']:>6.3f}  {probe['n_samples']}"
        )
    for name, probe in result["properties"].items():
        if probe["skipped_reason"]:
            print(f"{name:<10} skipped: {probe['skipped_reason']}")
        if probe["dropped_classes"]:
            dropped = ", ".join(
                f"{entry['label']} ({entry['count']})" for entry in probe["dropped_classes"]
            )
            print(f"{name:<10} dropped classes below threshold: {dropped}")
    print()
    if ranked:
        print("emergence order (shallow -> deep): " + " < ".join(name for name, _ in ranked))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ravdess", choices=sorted(DATASET_DIRS))
    parser.add_argument("--model", default="whisper-base")
    parser.add_argument("--probe", default="logreg", choices=["logreg", "linear_svm"])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--project-dims", type=int, default=256)
    parser.add_argument("--min-class-count", type=int, default=5)
    parser.add_argument("--noise-snr-db", type=float, default=None)
    parser.add_argument("--no-control", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="probe only the first N files")
    parser.add_argument("--out", type=Path, default=Path("layer_probe_result.json"))
    parser.add_argument("--cache", type=Path, default=None, help="activation .npz path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.dataset)
    if args.limit:
        rows = rows[: args.limit]

    cache_path = args.cache or args.out.with_suffix(".activations.npz")
    tag = f"{args.dataset}:{args.model}:{args.noise_snr_db}:{len(rows)}"
    clean = noisy = None
    if cache_path.exists():
        stored = np.load(cache_path, allow_pickle=True)
        if str(stored["tag"]) == tag:
            clean = stored["clean"]
            noisy = stored["noisy"] if "noisy" in stored.files and stored["noisy"].size else None
            rows = list(stored["rows"])
            layer_names = [str(name) for name in stored["layer_names"]]
            print(f"reusing cached activations from {cache_path}")
    if clean is None:
        print(f"extracting {args.model} activations for {len(rows)} {args.dataset} files...")
        clean, noisy, rows, layer_names = extract(args.dataset, args.model, rows, args.noise_snr_db)
        np.savez_compressed(
            cache_path, tag=tag, clean=clean,
            noisy=noisy if noisy is not None else np.array([]),
            rows=np.array(rows, dtype=object), layer_names=np.array(layer_names),
        )
        print(f"cached activations -> {cache_path}")

    print(f"activations: {clean.shape[0]} files x {clean.shape[1]} layers x {clean.shape[2]} dims")
    properties = build_properties(args.dataset, rows)
    print(f"properties: {', '.join(sorted(properties))}")

    from app.services.probing_service import train_layer_probes

    options = dict(
        probe=args.probe, cv_folds=args.cv_folds, project_dims=args.project_dims,
        min_class_count=args.min_class_count, include_control=not args.no_control,
        seed=args.seed, layer_names=layer_names,
    )
    started = time.monotonic()
    result = train_layer_probes(clean.tolist(), properties, **options)
    if noisy is not None:
        n_files = clean.shape[0]
        stacked = clean.tolist() + noisy.tolist()
        noise_labels = ["clean"] * n_files + ["noisy"] * n_files
        noise_result = train_layer_probes(stacked, {"noise": noise_labels}, **options)
        result["properties"]["noise"] = noise_result["properties"]["noise"]
    print(f"trained probes in {time.monotonic() - started:.1f}s")

    report(result)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nfull result -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
