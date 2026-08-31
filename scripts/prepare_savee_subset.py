"""Build a balanced 100-clip SAVEE subset for layer-wise probing.

SAVEE is licensed and cannot be redistributed, so this script takes a copy you
have already obtained and prepares an upload-ready subset from it.

Why a script rather than "select the first 100 files": SAVEE ships grouped by
speaker, so a naive slice returns 100 clips of a single actor and the speaker
probe is then correctly refused for having one class.  This samples stratified
over emotion and round-robins speakers within each emotion, so both properties
survive.  `tests/test_dataset_labels_service.py::TestSaveeEndToEnd` pins both
behaviours.

Usage
-----
    # See the plan without owning the data -- SAVEE's inventory is deterministic
    python scripts/prepare_savee_subset.py --dry-run

    # Prepare a real subset
    python scripts/prepare_savee_subset.py --source /path/to/SAVEE --count 100

Output is a flat folder of `<SPEAKER>_<code><take>.wav` files plus a
`savee_subset_metadata.csv`.  The flat naming matters: it is what lets the app's
`savee` filename pattern recover the labels after upload, so a user can attach an
answer key with one click and never touch the CSV.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.services.dataset_labels_service import SAVEE_EMOTIONS  # noqa: E402

SPEAKERS = ("DC", "JE", "JK", "KL")

# Clips per emotion per speaker in the full corpus: 15 for each of the six
# emotions, 30 for neutral -- 120 per speaker, 480 in total.
CLIPS_PER_EMOTION = {"a": 15, "d": 15, "f": 15, "h": 15, "n": 30, "sa": 15, "su": 15}


def full_inventory() -> list[tuple[str, str, int]]:
    """Every (speaker, emotion_code, take) SAVEE contains.

    Deterministic, which is what lets `--dry-run` show the exact subset before
    you have the audio.
    """
    return [
        (speaker, code, take)
        for speaker in SPEAKERS
        for code, total in CLIPS_PER_EMOTION.items()
        for take in range(1, total + 1)
    ]


def discover(source: Path) -> list[tuple[str, str, int, Path]]:
    """Find SAVEE clips under `source`, flat or nested by speaker folder."""
    found: list[tuple[str, str, int, Path]] = []
    for path in sorted(source.rglob("*.wav")):
        stem = path.stem
        speaker = code = None
        take = None
        if "_" in stem:  # flat: DC_a01.wav
            head, _, tail = stem.partition("_")
            if head.upper() in SPEAKERS:
                speaker = head.upper()
                code, take = _split_code(tail)
        if speaker is None and path.parent.name.upper() in SPEAKERS:  # nested
            speaker = path.parent.name.upper()
            code, take = _split_code(stem)
        if speaker and code in SAVEE_EMOTIONS and take is not None:
            found.append((speaker, code, take, path))
    return found


def _split_code(text: str) -> tuple[str | None, int | None]:
    """Split `sa04` into ('sa', 4). Longest emotion code wins over its prefix."""
    for code in sorted(SAVEE_EMOTIONS, key=len, reverse=True):
        if text.startswith(code) and text[len(code):].isdigit():
            return code, int(text[len(code):])
    return None, None


def sample(
    inventory: list[tuple[str, str, int]], count: int
) -> list[tuple[str, str, int]]:
    """Stratify over emotion, round-robin speakers inside each emotion.

    Emotion is the primary stratum because it has the most classes (7) and is
    therefore the first to become unprobeable when a subset is unbalanced.
    Speakers are cycled within each emotion so speaker stays balanced as a
    by-product rather than being traded away for emotion balance.
    """
    codes = sorted(CLIPS_PER_EMOTION)
    base, remainder = divmod(count, len(codes))

    by_emotion: dict[str, dict[str, list[tuple[str, str, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in inventory:
        by_emotion[entry[1]][entry[0]].append(entry)

    picked: list[tuple[str, str, int]] = []
    # The cursor carries across emotions rather than resetting. Restarting at
    # speaker 0 every time biases the early speakers whenever a quota is not a
    # multiple of the speaker count -- with 7 emotions that cost ~7 clips of
    # imbalance (28/28/23/21 instead of 25 each).
    cursor = 0
    for index, code in enumerate(codes):
        quota = base + (1 if index < remainder else 0)
        pools = {speaker: sorted(items, key=lambda e: e[2]) for speaker, items in by_emotion[code].items()}
        # Round-robin until the quota is met or every pool is exhausted.
        while quota > 0 and any(pools.values()):
            speaker = SPEAKERS[cursor % len(SPEAKERS)]
            cursor += 1
            pool = pools.get(speaker)
            if pool:
                picked.append(pool.pop(0))
                quota -= 1
    return picked


def clip_name(speaker: str, code: str, take: int) -> str:
    return f"{speaker}_{code}{take:02d}.wav"


def report(rows: list[dict[str, str]]) -> None:
    emotions = Counter(row["emotion"] for row in rows)
    speakers = Counter(row["speaker"] for row in rows)
    print(f"\n{len(rows)} clips selected")
    print("  emotion:", ", ".join(f"{name} ({count})" for name, count in sorted(emotions.items())))
    print("  speaker:", ", ".join(f"{name} ({count})" for name, count in sorted(speakers.items())))
    smallest = min(emotions.values())
    print(f"\n  smallest emotion class: {smallest} clips", end="")
    print("  (>= 5, so no class will be dropped)" if smallest >= 5
          else "  (< 5 -- this class WILL be dropped by the probe)")
    print(f"  majority baseline (emotion): {max(emotions.values()) / len(rows):.3f}")
    print(f"  majority baseline (speaker): {max(speakers.values()) / len(rows):.3f}")
    print(
        "\n  Note: every SAVEE speaker is male, so a `gender` column is a single\n"
        "  class. It is written to the CSV deliberately -- the probe refuses it and\n"
        "  says why, which is the behaviour worth seeing."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="SAVEE root (flat or speaker folders)")
    parser.add_argument("--out", type=Path, default=BACKEND / "data" / "savee_subset")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="plan only; no source needed")
    args = parser.parse_args()

    if not args.dry_run and not args.source:
        parser.error("--source is required unless --dry-run is given")

    if args.dry_run:
        inventory = full_inventory()
        paths: dict[tuple[str, str, int], Path] = {}
        print(f"SAVEE full inventory: {len(inventory)} clips (4 speakers x 120)")
    else:
        if not args.source.exists():
            print(f"error: source not found: {args.source}", file=sys.stderr)
            return 1
        discovered = discover(args.source)
        if not discovered:
            print(
                f"error: no SAVEE clips found under {args.source}.\n"
                "Expected either DC_a01.wav or AudioData/DC/a01.wav.",
                file=sys.stderr,
            )
            return 1
        inventory = [(speaker, code, take) for speaker, code, take, _ in discovered]
        paths = {(speaker, code, take): path for speaker, code, take, path in discovered}
        print(f"Found {len(inventory)} SAVEE clips under {args.source}")

    if args.count > len(inventory):
        print(f"error: asked for {args.count} clips but only {len(inventory)} exist", file=sys.stderr)
        return 1

    picked = sample(inventory, args.count)
    rows = [
        {
            "filename": clip_name(speaker, code, take),
            "speaker": speaker,
            "emotion": SAVEE_EMOTIONS[code],
            "emotion_code": code,
            "take": str(take),
            "gender": "male",
        }
        for speaker, code, take in picked
    ]
    rows.sort(key=lambda row: row["filename"])

    if args.dry_run:
        report(rows)
        print("\nFirst 10 clips that would be produced:")
        for row in rows[:10]:
            print(f"  {row['filename']:<16} speaker={row['speaker']}  emotion={row['emotion']}")
        print("\nRe-run with --source <SAVEE path> to actually build the subset.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for row, key in zip(rows, sorted(picked, key=lambda e: clip_name(*e))):
        shutil.copy2(paths[key], args.out / row["filename"])

    csv_path = args.out / "savee_subset_metadata.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["filename", "speaker", "emotion", "emotion_code", "take", "gender"]
        )
        writer.writeheader()
        writer.writerows(rows)

    report(rows)
    print(f"\naudio  -> {args.out}")
    print(f"labels -> {csv_path}")
    print(
        "\nNext: upload the .wav files through Manage Datasets > Upload Files,\n"
        "then either apply the 'SAVEE' filename pattern (no CSV needed) or upload\n"
        "this CSV in the Labels tab."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
