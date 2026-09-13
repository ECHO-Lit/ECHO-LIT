"""Label ingestion for custom datasets.

A layer probe is a graded guessing game: it reads one layer's activations and
predicts a property, and to grade it we must already know the answer.  The
bundled datasets ship that answer key as a CSV.  An uploaded dataset has audio
and nothing else, so `availableProperties` finds no probeable column and the
Layer Probes tab offers nothing.  This module supplies the missing answer key.

Three sources, in the order a user should reach for them:

``parse_labels_csv``
    The user uploads their own CSV, joined to the audio by ``filename``.  The
    general path, and the only one that can express a property the audio does
    not already contain.
``derive_from_filenames``
    Many speech corpora encode everything in the filename -- SAVEE's
    ``DC_a01.wav`` is speaker DC, anger, take 1.  A named pattern extracts
    several properties at once with no CSV at all.
``band_labels``
    Zero-annotation labels quantised from a measurement the upload already
    recorded (duration) or that can be measured from the audio.  These need
    nothing from the user, so every custom dataset has at least one probeable
    property the moment it exists.

Deliberately dependency-free: this is imported by the FastAPI control plane,
whose container has neither numpy nor scikit-learn (see ``requirements-api``).
Everything here is plain Python over plain dicts.
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# --- Constants mirrored from `app.services.probing_service` ------------------
#
# They cannot be imported: `probing_service` imports numpy at module scope and
# the API container does not ship it.  `tests/test_dataset_labels_service.py`
# asserts these stay equal to the originals, so drift fails a test rather than
# silently making the preview disagree with the probe it is previewing.
MISSING_LABELS = {"", "unknown", "none", "n/a", "na"}
MIN_ROWS_FOR_PROBE = 8
MIN_FOLDS = 2

# Columns that describe the *file* rather than something to probe.  Offering
# `size` or `uploaded_at` as a probe target would be noise; `filename` is the
# join key.  `duration` is excluded because `duration_band` is its probeable
# form -- probing a continuous value directly would make one class per file.
RESERVED_COLUMNS = frozenset({
    "filename",
    "original_filename",
    "size",
    "uploaded_at",
    "duration",
    "sample_rate",
    "text",
    "client_id",
    "locale",
    "up_votes",
    "down_votes",
})

# A column with more distinct values than this is an identifier, not a class.
MAX_DISCOVERED_CLASSES = 50

# Bands need enough members each to survive `min_class_count` downstream.
MIN_ROWS_PER_BAND = 4

DEFAULT_BAND_NAMES = ("low", "mid", "high")


def normalise_label(value: Any) -> str | None:
    """Return a usable label, or None when the value means "not annotated"."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in MISSING_LABELS else text


def _basename(path: str) -> str:
    return re.split(r"[\\/]", str(path))[-1]


# --------------------------------------------------------------------------
# Source 1 -- a user-supplied CSV
# --------------------------------------------------------------------------


def parse_labels_csv(text: str) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Parse an uploaded label CSV into ``{filename: {column: label}}``.

    Requires a ``filename`` column; everything else becomes a candidate probe
    property.  Rows are indexed under both the value as written and its
    basename, because a user's CSV commonly carries a path
    (``AudioData/DC/a01.wav``) where the dataset holds the bare name.

    Returns the table plus human-readable warnings.  Warnings are surfaced, not
    raised: a CSV with one unusable column should still contribute its good
    ones, and the user needs to be told which is which.
    """
    warnings: list[str] = []
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("The CSV is empty or has no header row")

    columns = [str(name).strip().lower() for name in reader.fieldnames if name]
    if "filename" not in columns:
        raise ValueError(
            f"The CSV needs a 'filename' column; found: {', '.join(columns) or 'nothing'}"
        )

    label_columns = [name for name in columns if name not in RESERVED_COLUMNS]
    if not label_columns:
        raise ValueError(
            "The CSV has a 'filename' column but no label columns to go with it"
        )

    table: dict[str, dict[str, str]] = {}
    duplicates = 0
    for row in reader:
        cleaned = {
            str(key).strip().lower(): (str(value).strip() if value is not None else "")
            for key, value in row.items()
            if key
        }
        filename = cleaned.get("filename", "")
        if not filename:
            continue
        labels = {
            column: cleaned.get(column, "")
            for column in label_columns
            if normalise_label(cleaned.get(column)) is not None
        }
        if not labels:
            continue
        base = _basename(filename)
        if base in table:
            duplicates += 1
        table[filename] = labels
        table.setdefault(base, labels)

    if duplicates:
        warnings.append(f"{duplicates} duplicate filename(s) in the CSV; the last one wins")
    if not table:
        raise ValueError("No usable rows: every row was missing a filename or all its labels")
    return table, warnings


# --------------------------------------------------------------------------
# Source 2 -- labels encoded in the filename
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FilenamePattern:
    """A named regex over a filename stem, plus per-group value maps.

    `regex` named groups become property columns directly.  `value_maps` turns a
    corpus's short codes into readable classes -- probing "anger" vs "a" makes no
    difference to the maths but a great deal to the chart.
    """

    pattern_id: str
    label: str
    description: str
    regex: str
    value_maps: dict[str, dict[str, str]] = field(default_factory=dict)
    # Groups captured for matching but not worth probing (take numbers, etc.).
    ignore_groups: frozenset[str] = frozenset()
    example: str = ""

    def properties(self) -> list[str]:
        names = re.compile(self.regex).groupindex.keys()
        return [name for name in names if name not in self.ignore_groups]


# SAVEE: 4 male speakers x {anger, disgust, fear, happiness, neutral, sadness,
# surprise}.  Distributed either flat (`DC_a01.wav`) or nested by speaker
# (`AudioData/DC/a01.wav`); `derive_from_filenames` tries the parent directory
# as the speaker when the stem alone does not carry it.
SAVEE_EMOTIONS = {
    "a": "anger",
    "d": "disgust",
    "f": "fear",
    "h": "happiness",
    "n": "neutral",
    "sa": "sadness",
    "su": "surprise",
}

FILENAME_PATTERNS: dict[str, FilenamePattern] = {
    "savee": FilenamePattern(
        pattern_id="savee",
        label="SAVEE",
        description=(
            "Surrey Audio-Visual Expressed Emotion. Filenames encode speaker and "
            "emotion: DC_a01 is speaker DC, anger, take 1."
        ),
        # The emotion code is greedy over 1-2 letters so `sa`/`su` win over `s`,
        # and the take digits cannot be absorbed because they are not [a-z].
        regex=r"^(?P<speaker>[A-Za-z]{2})[_-](?P<emotion>[a-z]{1,2})(?P<take>\d+)$",
        value_maps={"emotion": SAVEE_EMOTIONS},
        ignore_groups=frozenset({"take"}),
        example="DC_a01.wav",
    ),
    "savee-nested": FilenamePattern(
        pattern_id="savee-nested",
        label="SAVEE (speaker folders)",
        description=(
            "SAVEE as originally distributed, one folder per speaker: "
            "AudioData/DC/a01.wav. The folder name supplies the speaker."
        ),
        regex=r"^(?P<emotion>[a-z]{1,2})(?P<take>\d+)$",
        value_maps={"emotion": SAVEE_EMOTIONS},
        ignore_groups=frozenset({"take"}),
        example="AudioData/DC/a01.wav",
    ),
    "ravdess": FilenamePattern(
        pattern_id="ravdess",
        label="RAVDESS",
        description=(
            "Seven hyphen-separated codes: modality-channel-emotion-intensity-"
            "statement-repetition-actor."
        ),
        regex=(
            r"^(?P<modality>\d{2})-(?P<channel>\d{2})-(?P<emotion>\d{2})-"
            r"(?P<intensity>\d{2})-(?P<statement>\d{2})-(?P<repetition>\d{2})-"
            r"(?P<actor>\d{2})$"
        ),
        value_maps={
            "emotion": {
                "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
                "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
            },
            "intensity": {"01": "normal", "02": "strong"},
            "statement": {
                "01": "Kids are talking by the door",
                "02": "Dogs are sitting by the door",
            },
        },
        ignore_groups=frozenset({"modality", "channel", "repetition"}),
        example="03-01-01-01-01-01-16.wav",
    ),
    "crema-d": FilenamePattern(
        pattern_id="crema-d",
        label="CREMA-D",
        description="Underscore-separated: actor_sentence_emotion_intensity.",
        regex=(
            r"^(?P<actor>\d{4})_(?P<sentence>[A-Z]{3})_(?P<emotion>[A-Z]{3})_"
            r"(?P<intensity>[A-Z]{2})$"
        ),
        value_maps={
            "emotion": {
                "ANG": "anger", "DIS": "disgust", "FEA": "fear",
                "HAP": "happy", "NEU": "neutral", "SAD": "sad",
            },
            "intensity": {"LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified"},
        },
        example="1001_DFA_ANG_XX.wav",
    ),
}


def derive_from_filenames(
    filenames: Sequence[str], pattern_id: str
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Extract labels from filenames using a named pattern.

    Returns ``({filename: {property: label}}, warnings)``.  Files that do not
    match are simply absent from the table -- they become unannotated rows the
    probe drops, which is the correct handling for a mixed-provenance folder.
    """
    pattern = FILENAME_PATTERNS.get(pattern_id)
    if pattern is None:
        raise ValueError(
            f"Unknown filename pattern '{pattern_id}'. "
            f"Available: {', '.join(sorted(FILENAME_PATTERNS))}"
        )
    compiled = re.compile(pattern.regex)
    wanted = pattern.properties()
    nests_speaker = pattern.pattern_id == "savee-nested"

    table: dict[str, dict[str, str]] = {}
    unmapped: Counter[str] = Counter()
    unmatched = 0
    for original in filenames:
        base = _basename(original)
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", base)
        match = compiled.match(stem)
        if not match:
            unmatched += 1
            continue
        labels: dict[str, str] = {}
        for name in wanted:
            raw = match.group(name)
            if raw is None:
                continue
            mapping = pattern.value_maps.get(name)
            if mapping is None:
                labels[name] = raw
                continue
            mapped = mapping.get(raw)
            if mapped is None:
                # Keep the raw code rather than dropping the row: an unmapped
                # value is still a consistent class, and naming it lets the user
                # see what the pattern did not recognise.
                unmapped[f"{name}={raw}"] += 1
                labels[name] = raw
            else:
                labels[name] = mapped
        if nests_speaker:
            parts = re.split(r"[\\/]", str(original))
            if len(parts) >= 2 and parts[-2]:
                labels["speaker"] = parts[-2]
        if labels:
            table[original] = labels
            table.setdefault(base, labels)

    warnings: list[str] = []
    if unmatched:
        warnings.append(
            f"{unmatched} of {len(filenames)} filename(s) did not match the "
            f"{pattern.label} pattern and were left unannotated"
        )
    if unmapped:
        listed = ", ".join(f"{code} ({count})" for code, count in sorted(unmapped.items()))
        warnings.append(f"Unrecognised codes kept as-is: {listed}")
    if not table:
        raise ValueError(
            f"No filename matched the {pattern.label} pattern "
            f"(expected something like {pattern.example})"
        )
    return table, warnings


# --------------------------------------------------------------------------
# Source 3 -- zero-annotation bands from a measurement
# --------------------------------------------------------------------------


def band_labels(
    values: Sequence[float | None], band_names: Sequence[str] = DEFAULT_BAND_NAMES
) -> list[str | None]:
    """Quantise a continuous measurement into equal-count bands.

    Rank-based rather than range-based so a skewed distribution still yields
    usable class sizes -- equal-width bins on clip duration routinely put 95 %
    of a corpus in one bin.

    Returns all-``None`` when the result could not be probed anyway: too few
    values, or too few distinct ones to fill the bands.  Silently emitting a
    single-class column would push the failure downstream to the probe, which
    reports it far less legibly than simply not offering the property.
    """
    present = [(index, value) for index, value in enumerate(values) if value is not None]
    bands = list(band_names)
    if len(present) < MIN_ROWS_PER_BAND * len(bands):
        return [None] * len(values)
    if len({value for _, value in present}) < len(bands):
        return [None] * len(values)

    ordered = sorted(present, key=lambda pair: pair[1])
    total = len(ordered)
    out: list[str | None] = [None] * len(values)
    for rank, (index, _) in enumerate(ordered):
        out[index] = bands[min(rank * len(bands) // total, len(bands) - 1)]
    return out


def attach_duration_bands(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Add a `duration_band` column derived from durations already recorded.

    This is what makes a freshly uploaded, wholly unannotated dataset probeable
    at all.  Read it as a sanity floor rather than a finding: duration is an
    input property, so a flat or declining curve is the expected result and a
    *rising* one should be treated as suspicious before it is treated as
    interesting.
    """
    def as_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    bands = band_labels([as_float(row.get("duration")) for row in rows])
    for row, band in zip(rows, bands):
        if band is not None:
            row["duration_band"] = band
    return rows


# --------------------------------------------------------------------------
# Merging and previewing
# --------------------------------------------------------------------------


def merge_labels(
    rows: list[dict[str, str]], table: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    """Join a label table onto file rows, matching on filename or its basename.

    Mutates and returns `rows`.  A file with no entry keeps whatever columns it
    had; the probe treats the gap as "not annotated" rather than as a class.
    """
    for row in rows:
        filename = row.get("filename", "")
        labels = table.get(filename) or table.get(_basename(filename))
        if labels:
            row.update(labels)
    return rows


def label_columns(rows: Iterable[dict[str, str]]) -> list[str]:
    """Columns that could be probe targets, in first-seen order."""
    seen: list[str] = []
    for row in rows:
        for column in row:
            if column not in RESERVED_COLUMNS and column not in seen:
                seen.append(column)
    return seen


def summarise_property(
    rows: Sequence[dict[str, str]],
    column: str,
    *,
    min_class_count: int = 5,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Predict what the probe will do with this column, without running it.

    Extraction is the expensive step and probe training is not, so a user should
    learn "two of your three classes are about to be dropped" in a second rather
    than after a multi-minute job.  The arithmetic deliberately mirrors
    `probing_service._probe_property` so the preview and the run agree.
    """
    normalised = [normalise_label(row.get(column)) for row in rows]
    n_missing = sum(1 for value in normalised if value is None)
    raw_counts = Counter(value for value in normalised if value is not None)

    kept = {label for label, count in raw_counts.items() if count >= min_class_count}
    dropped = [
        {"label": label, "count": int(count)}
        for label, count in sorted(raw_counts.items())
        if label not in kept
    ]
    counts = {label: int(count) for label, count in sorted(raw_counts.items()) if label in kept}
    n_samples = sum(counts.values())

    # The three branches below mirror `_probe_property` exactly, including which
    # counters it leaves at zero in each degenerate case.  Reporting a healthier
    # number here than the probe will report is the one bug this function must
    # not have: the user would plan a run around a sample size that never
    # existed.  `TestPreviewMatchesProbe` pins the agreement.
    if not raw_counts:
        skipped = "no labelled files"
        n_samples, n_classes = 0, 0
    elif len(counts) < 2:
        skipped = (
            f"only {len(counts)} class remains after dropping classes with "
            f"fewer than {min_class_count} files"
        )
        n_samples, n_classes = 0, 0
    elif n_samples < MIN_ROWS_FOR_PROBE:
        skipped = f"needs at least {MIN_ROWS_FOR_PROBE} labelled files, got {n_samples}"
        n_classes = len(counts)
    else:
        skipped = None
        n_classes = len(counts)

    if skipped is not None:
        folds_used = 0
        majority = None
    else:
        folds_used = max(MIN_FOLDS, min(int(cv_folds), min(counts.values())))
        majority = max(counts.values()) / n_samples

    return {
        "property": column,
        "n_samples": n_samples,
        "n_missing": n_missing,
        "n_classes": n_classes,
        "class_counts": counts,
        "dropped_classes": dropped,
        "majority_baseline": majority,
        "cv_folds_used": folds_used,
        "probeable": skipped is None,
        "skipped_reason": skipped,
    }


def preview_dataset(
    rows: Sequence[dict[str, str]],
    *,
    min_class_count: int = 5,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Per-property preview for every probeable column in a dataset."""
    columns = [
        column
        for column in label_columns(rows)
        # An identifier-like column (one value per file) cannot be probed, and
        # offering it would produce as many classes as there are rows.
        if len({normalise_label(row.get(column)) for row in rows} - {None})
        <= MAX_DISCOVERED_CLASSES
    ]
    properties = [
        summarise_property(rows, column, min_class_count=min_class_count, cv_folds=cv_folds)
        for column in columns
    ]
    return {
        "n_files": len(rows),
        "properties": properties,
        "probeable_count": sum(1 for entry in properties if entry["probeable"]),
    }


def available_patterns() -> list[dict[str, Any]]:
    """Filename patterns offered to the UI."""
    return [
        {
            "pattern_id": pattern.pattern_id,
            "label": pattern.label,
            "description": pattern.description,
            "example": pattern.example,
            "properties": pattern.properties(),
        }
        for pattern in FILENAME_PATTERNS.values()
    ]
