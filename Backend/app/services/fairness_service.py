"""FR-10: grouped dataset engine, design classification, execution planning,
and Celery-stage orchestration for accent/language fairness analysis
(docs/FR10plan.md Part 1 S2, S3).

Indexing/gating/design-classification/plan-building below use only the
standard library + dataclasses, so this module is safe to import from the
FastAPI control plane (it powers GET /analyses/fairness/groupable) as well as
from workers. The orchestration functions at the bottom (prepare_analysis,
infer_shard, explain_shard, aggregate_fairness) run ONLY on workers and do all
numpy/sklearn/jiwer-touching work through LOCAL imports of
fairness_metrics_service/grounding_service, exactly so this file's top-level
import list stays free of the worker-only dependencies (mirrors
app/services/fr7_planning.py's control-plane/worker split).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.core import redis as redis_module
from app.core.settings import settings
from app.core.storage import StorageError, get_storage
from app.repositories.jobs import JobRepository
from app.schemas.jobs import JobError, JobProgress, JobStatus, TaskEnvelope
from app.services import dataset_service

logger = logging.getLogger(__name__)


class JobCancelled(Exception):
    pass


class FairnessInputError(ValueError):
    """Raised when the dataset/grouping combination cannot support a
    comparison (SRS FR-10 exception flow). Mapped to
    JobError(code="fr10_insufficient_groups"/"fr10_invalid_grouping",
    retryable=False) by the calling Celery task."""


async def _check_cancel(job_id: str, jobs: JobRepository) -> None:
    if await jobs.cancellation_requested(job_id):
        raise JobCancelled()


def _jsonable(value: Any) -> Any:
    """storage.put_json uses allow_nan=False (app/core/storage.py), so every
    writer must sanitise float('inf')/float('-inf')/float('nan') -> None
    before serializing. inf ratios are a real case, not a hypothetical: a
    reference group with WER exactly 0.0 (e.g. a perfect run, or a synthetic
    fixture) makes `other/reference` infinite in disparity_summary and
    estimate_gap's `ratio` field."""
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

# Metadata columns that are never valid grouping keys (free text / identifiers
# / already-consumed elsewhere).
_NON_GROUPABLE = {
    "filename", "text", "reading_passage", "g2p_canonical", "ipa_perceived",
    "utt_id", "duration", "client_id", "up_votes", "down_votes",
}

# Columns that are a valid grouping axis but whose per-value counts reflect a
# balanced SAMPLING STRATUM (e.g. L2-ARCTIC's error_type: 50/50/50 by
# construction) rather than a speaker attribute. Grouping WER by these is
# meaningless (docs/FR10plan.md Part 1 S6.2); flagged, not blocked, so a
# researcher who has a real reason can still use them as a `filters` axis.
_STRATUM_ONLY = {"error_type", "modality", "vocal_channel", "repetition", "intensity"}

# Columns that identify a physical speaker, used as the bootstrap block unit
# and for speaker-disjoint sampling/CV. Checked in priority order.
_SPEAKER_COLUMNS = ("speakerid", "speaker_code", "speaker", "client_id", "actor")

# Columns carrying an explicit, stable content identifier (preferred over a
# text hash when present, e.g. L2-ARCTIC's utt_id).
_CONTENT_ID_COLUMNS = ("utt_id",)

# Columns carrying the literal reference text/passage read aloud.
_TEXT_COLUMNS = ("text", "reading_passage", "sentence", "statement", "transcript", "transcription")

# Candidate ground-truth label columns for classification tasks.
_LABEL_COLUMNS = ("emotion", "label", "class", "category")

# Sentinels meaning "unlabelled", never a real group value. Common Voice's
# bundled subset writes literal "unknown" for 67/100 accent/gender rows.
_MISSING = {"", "unknown", "n/a", "na", "none", "null", "-"}

_RESERVED_COLUMNS = _NON_GROUPABLE | set(_SPEAKER_COLUMNS) | set(_LABEL_COLUMNS)

# Below this Jaccard overlap (and per-group coverage), two groups' content
# sets are treated as unrelated even if a handful of ids coincide by chance.
MIN_SHARED_CONTENT = 5


def _norm(value: Any) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip().lower()
    return text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _item_id(dataset: str, filename: str) -> str:
    # Strip directory prefixes ("cv-valid-dev/sample-000775.mp3") and
    # extensions so an item id is stable across dataset re-packaging.
    stem = Path(filename).stem
    material = f"{dataset}|{stem}".encode()
    return hashlib.sha1(material).hexdigest()[:16]


def _first_present(rows: list[dict[str, str]], candidates: tuple[str, ...]) -> str | None:
    if not rows:
        return None
    keys = set(rows[0].keys())
    for candidate in candidates:
        if candidate in keys and any(_norm(r.get(candidate)) not in _MISSING for r in rows):
            return candidate
    return None


def _counts(values: list[str]) -> dict[str, int]:
    return dict(Counter(values))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupItem:
    """One evaluation item, fully resolved and ready for a worker."""
    item_id: str
    dataset: str
    filename: str
    session_id: str | None
    duration_seconds: float
    reference_text: str | None
    reference_label: str | None
    speaker_id: str | None
    content_id: str | None
    covariates: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Group:
    key: tuple[str, ...]
    label: str
    items: list[GroupItem] = field(default_factory=list)
    speaker_confounded: bool = False

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def speaker_ids(self) -> set[str]:
        return {i.speaker_id for i in self.items if i.speaker_id}

    @property
    def n_speakers(self) -> int:
        return len(self.speaker_ids)


@dataclass
class GroupIndex:
    dataset: str
    grouping_key: list[str]
    task: str
    groups: list[Group]
    excluded: list[dict[str, Any]]
    speaker_column: str | None
    content_column: str | None
    design: str
    design_evidence: dict[str, Any]
    reference_group: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset, "grouping_key": self.grouping_key, "task": self.task,
            "groups": [
                {"key": list(g.key), "label": g.label, "items": [i.to_dict() for i in g.items],
                 "speaker_confounded": g.speaker_confounded}
                for g in self.groups
            ],
            "excluded": self.excluded, "speaker_column": self.speaker_column,
            "content_column": self.content_column, "design": self.design,
            "design_evidence": self.design_evidence, "reference_group": self.reference_group,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "GroupIndex":
        groups = [
            Group(key=tuple(g["key"]), label=g["label"],
                  items=[GroupItem(**item) for item in g["items"]],
                  speaker_confounded=g.get("speaker_confounded", False))
            for g in data["groups"]
        ]
        return GroupIndex(
            dataset=data["dataset"], grouping_key=data["grouping_key"], task=data["task"],
            groups=groups, excluded=data["excluded"], speaker_column=data["speaker_column"],
            content_column=data["content_column"], design=data["design"],
            design_evidence=data["design_evidence"], reference_group=data["reference_group"],
        )


@dataclass(frozen=True)
class Shard:
    shard_id: str
    group_label: str
    item_ids: list[str]
    operation: str  # "prediction" | "saliency"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionPlan:
    infer_shards: list[Shard]
    explain_shards: list[Shard]
    explain_sample: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "infer_shards": [s.to_dict() for s in self.infer_shards],
            "explain_shards": [s.to_dict() for s in self.explain_shards],
            "explain_sample": self.explain_sample,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ExecutionPlan":
        return ExecutionPlan(
            infer_shards=[Shard(**s) for s in data["infer_shards"]],
            explain_shards=[Shard(**s) for s in data["explain_shards"]],
            explain_sample=data["explain_sample"],
        )


# ---------------------------------------------------------------------------
# Groupable-columns preview (control plane, GET /analyses/fairness/groupable)
# ---------------------------------------------------------------------------

def groupable_columns(dataset: str, session_id: str | None) -> dict[str, Any]:
    """Powers the UI's grouping dropdown before submission. Cheap: built-in
    CSVs are cached by mtime in dataset_service; custom datasets are small
    per-session listings."""
    rows = dataset_service.load_metadata(dataset, session_id)
    speaker_column = _first_present(rows, _SPEAKER_COLUMNS)
    content_id_column = _first_present(rows, _CONTENT_ID_COLUMNS)
    text_column = _first_present(rows, _TEXT_COLUMNS)

    all_columns: set[str] = set()
    for row in rows:
        all_columns.update(row.keys())
    candidates = all_columns - _NON_GROUPABLE

    out = []
    for column in candidates:
        values = [_norm(r.get(column)) for r in rows]
        labelled = [v for v in values if v not in _MISSING]
        n_distinct = len(set(labelled))
        if not (2 <= n_distinct <= max(2, len(rows) // 3)):
            continue
        out.append({
            "column": column,
            "n_values": n_distinct,
            "n_missing": len(values) - len(labelled),
            "counts": _counts(labelled),
            "is_speaker_column": column in _SPEAKER_COLUMNS,
            "is_stratum_only": column in _STRATUM_ONLY,
        })
    out.sort(key=lambda c: (c["is_stratum_only"], -c["n_values"], c["column"]))

    return {
        "dataset": dataset, "n_rows": len(rows),
        "speaker_column": speaker_column,
        "content_column": content_id_column or text_column,
        "columns": out,
    }


# ---------------------------------------------------------------------------
# Indexing (Part 1 S2.1-S2.2)
# ---------------------------------------------------------------------------

def _passes(row: dict[str, str], filters: dict[str, list[str]]) -> bool:
    for column, allowed in filters.items():
        allowed_norm = {_norm(v) for v in allowed}
        if _norm(row.get(column)) not in allowed_norm:
            return False
    return True


def _to_item(
    dataset: str, row: dict[str, str], *, speaker_column: str | None,
    content_id_column: str | None, text_column: str | None, label_column: str | None,
    task: str, session_id: str | None,
) -> GroupItem:
    filename = row.get("filename", "")
    reference_text = row.get(text_column) if text_column else None
    reference_label = row.get(label_column) if (task == "classification" and label_column) else None
    speaker_id = _norm(row.get(speaker_column)) if speaker_column else None
    speaker_id = speaker_id or None

    if content_id_column and row.get(content_id_column):
        content_id = _norm(row[content_id_column])
    elif reference_text:
        content_id = hashlib.sha1(_norm(reference_text).encode()).hexdigest()[:12]
    else:
        content_id = None

    covariates = {k: v for k, v in row.items() if k not in _RESERVED_COLUMNS and k != content_id_column}

    return GroupItem(
        item_id=_item_id(dataset, filename), dataset=dataset, filename=filename,
        session_id=session_id, duration_seconds=_safe_float(row.get("duration")),
        reference_text=reference_text, reference_label=reference_label,
        speaker_id=speaker_id, content_id=content_id, covariates=covariates,
    )


def build_index(
    dataset: str,
    grouping_key: list[str],
    session_id: str | None,
    *,
    task: str,
    min_group_size: int = 8,
    min_speakers_per_group: int = 2,
    filters: dict[str, list[str]] | None = None,
    max_items_per_group: int | None = None,
    reference_group: str | None = None,
    seed: int = 0,
) -> GroupIndex:
    rows = dataset_service.load_metadata(dataset, session_id)
    if not rows:
        raise FairnessInputError(f"Dataset '{dataset}' has no rows.")

    for column in grouping_key:
        if column not in rows[0]:
            raise FairnessInputError(f"Column '{column}' is not present in dataset '{dataset}'.")

    speaker_column = _first_present(rows, _SPEAKER_COLUMNS)
    content_id_column = _first_present(rows, _CONTENT_ID_COLUMNS)
    text_column = _first_present(rows, _TEXT_COLUMNS)
    label_column = _first_present(rows, _LABEL_COLUMNS)

    buckets: dict[tuple[str, ...], list[GroupItem]] = {}
    unlabelled = 0
    for row in rows:
        if filters and not _passes(row, filters):
            continue
        key = tuple(_norm(row.get(c)) for c in grouping_key)
        if any(k in _MISSING for k in key):
            unlabelled += 1
            continue
        item = _to_item(
            dataset, row, speaker_column=speaker_column, content_id_column=content_id_column,
            text_column=text_column, label_column=label_column, task=task, session_id=session_id,
        )
        buckets.setdefault(key, []).append(item)

    excluded: list[dict[str, Any]] = []
    if unlabelled:
        excluded.append({"label": "<unlabelled>", "n_items": unlabelled,
                          "reason": "missing_group_label"})

    groups: list[Group] = []
    rng = random.Random(seed)
    for key, items in sorted(buckets.items()):
        if len(items) < min_group_size:
            excluded.append({
                "label": " ∩ ".join(key), "n_items": len(items),
                "n_speakers": len({i.speaker_id for i in items if i.speaker_id}),
                "reason": "below_min_group_size", "min_group_size": min_group_size,
            })
            continue
        if max_items_per_group and len(items) > max_items_per_group:
            items = _stratified_subsample(items, max_items_per_group, rng)
        groups.append(Group(key=key, label=" ∩ ".join(key), items=items))

    if len(groups) < 2:
        raise FairnessInputError(
            f"Only {len(groups)} group(s) meet the minimum size of {min_group_size}. "
            f"A fairness comparison needs at least two."
        )

    ref_label = _resolve_reference_group(groups, reference_group)
    design, evidence = classify_design(groups, content_id_column, text_column)

    for group in groups:
        group.speaker_confounded = group.n_speakers < min_speakers_per_group  # type: ignore[attr-defined]

    return GroupIndex(
        dataset=dataset, grouping_key=grouping_key, task=task, groups=groups, excluded=excluded,
        speaker_column=speaker_column, content_column=content_id_column or text_column,
        design=design, design_evidence=evidence, reference_group=ref_label,
    )


def _resolve_reference_group(groups: list[Group], requested: str | None) -> str:
    if requested:
        for g in groups:
            if g.label == requested:
                return g.label
        raise FairnessInputError(f"reference_group '{requested}' is not among the surviving groups.")
    return max(groups, key=lambda g: g.n_items).label


def _stratified_subsample(items: list[GroupItem], cap: int, rng: random.Random) -> list[GroupItem]:
    """Cap a group at `cap` items without dropping whole speakers
    disproportionately: round-robin across speakers, shuffled deterministically."""
    by_speaker: dict[str, list[GroupItem]] = {}
    for item in items:
        by_speaker.setdefault(item.speaker_id or item.item_id, []).append(item)
    for bucket in by_speaker.values():
        rng.shuffle(bucket)
    speakers = list(by_speaker.keys())
    rng.shuffle(speakers)
    out: list[GroupItem] = []
    idx = 0
    while len(out) < cap:
        progressed = False
        for speaker in speakers:
            bucket = by_speaker[speaker]
            if idx < len(bucket):
                out.append(bucket[idx])
                progressed = True
                if len(out) >= cap:
                    break
        if not progressed:
            break
        idx += 1
    return out


# ---------------------------------------------------------------------------
# Design classification (Part 1 S2.4) -- the scientific core
# ---------------------------------------------------------------------------

def classify_design(
    groups: list[Group], content_id_column: str | None, text_column: str | None,
) -> tuple[str, dict[str, Any]]:
    """matched: every group covers the same content -> WER differences are
    attributable to acoustics; use a PAIRED estimator.
    partially_matched: a non-trivial shared content subset exists -> paired
    analysis restricted to that subset.
    unmatched: no shared content -> report the gap but label it
    content-confounded; never a causal wording."""
    content_column = content_id_column or text_column
    sets = [{i.content_id for i in g.items if i.content_id} for g in groups]
    if not content_column or any(len(s) == 0 for s in sets):
        return "unmatched", {"reason": "no_content_identifier", "content_column": content_column}

    shared = set.intersection(*sets)
    union = set.union(*sets)
    coverage = [len(shared) / max(len(s), 1) for s in sets]
    jaccard = len(shared) / max(len(union), 1)

    evidence = {
        "content_column": content_column,
        "n_shared_content": len(shared), "n_total_content": len(union),
        "jaccard": round(jaccard, 4),
        "per_group_coverage": {g.label: round(c, 4) for g, c in zip(groups, coverage)},
        "shared_content_ids": sorted(shared)[:500],
    }
    if jaccard >= 0.95 and min(coverage) >= 0.95:
        return "matched", evidence
    if len(shared) >= MIN_SHARED_CONTENT:
        return "partially_matched", evidence
    return "unmatched", evidence


# ---------------------------------------------------------------------------
# Execution plan (Part 1 S2.5)
# ---------------------------------------------------------------------------

def _chunks(seq: list[str], size: int) -> list[list[str]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)] or [[]]


def _shard_id(group_label: str, item_ids: list[str], stage: str) -> str:
    material = f"{group_label}|{stage}|{'|'.join(item_ids)}".encode()
    return hashlib.sha1(material).hexdigest()[:16]


def _balanced_sample(
    items: list[GroupItem], n: int, shared_ids: set[str] | None, rng: random.Random,
) -> list[GroupItem]:
    """Equal-n-per-group deterministic sample, speaker-balanced, preferring
    shared-content items so the saliency comparison shows the SAME utterance
    across groups when the design allows it."""
    by_speaker: dict[str, list[GroupItem]] = {}
    for item in items:
        by_speaker.setdefault(item.speaker_id or item.item_id, []).append(item)
    for bucket in by_speaker.values():
        bucket.sort(key=lambda i: (0 if shared_ids and i.content_id in shared_ids else 1, i.item_id))
    speakers = sorted(by_speaker.keys())
    rng.shuffle(speakers)
    out: list[GroupItem] = []
    idx = 0
    while len(out) < min(n, len(items)):
        for speaker in speakers:
            bucket = by_speaker[speaker]
            if idx < len(bucket):
                out.append(bucket[idx])
                if len(out) >= min(n, len(items)):
                    break
        idx += 1
    return out


def build_plan(
    index: GroupIndex, *, shard_size: int, include_explanations: bool,
    explain_per_group: int, explain_shard_size: int, seed: int,
) -> ExecutionPlan:
    infer: list[Shard] = []
    explain: list[Shard] = []
    sample_map: dict[str, list[str]] = {}
    rng = random.Random(seed)
    shared_ids = set(index.design_evidence.get("shared_content_ids") or [])

    for group in index.groups:
        ids = [i.item_id for i in group.items]
        for chunk in _chunks(ids, shard_size):
            if chunk:
                infer.append(Shard(_shard_id(group.label, chunk, "infer"), group.label, chunk, "prediction"))
        if include_explanations:
            picked = _balanced_sample(group.items, explain_per_group, shared_ids, rng)
            picked_ids = [i.item_id for i in picked]
            sample_map[group.label] = picked_ids
            for chunk in _chunks(picked_ids, explain_shard_size):
                if chunk:
                    explain.append(Shard(_shard_id(group.label, chunk, "explain"), group.label, chunk, "saliency"))

    return ExecutionPlan(infer_shards=infer, explain_shards=explain, explain_sample=sample_map)


# ---------------------------------------------------------------------------
# Worker-only orchestration (Part 1 S3) -- called from app.worker.tasks.
#
# Everything above this line is stdlib-only. Everything below runs ONLY on a
# Celery worker and does numpy/sklearn-touching work through LOCAL imports of
# fairness_metrics_service / grounding_service, so importing THIS module from
# the FastAPI control plane (as GET /analyses/fairness/groupable does) never
# pulls those worker-only dependencies in.
# ---------------------------------------------------------------------------

def _plan_key(session_id: str, job_id: str) -> str:
    return f"fairness/{session_id}/{job_id}/plan.json"


def fairness_cache_key(envelope: TaskEnvelope) -> str:
    """Session-neutral for built-in datasets; for custom datasets the
    `dataset` string already embeds `custom:{session_id}:{name}`
    (custom_dataset_service.format_custom_dataset_name), so no cross-session
    cache collision is possible without adding session_id here explicitly."""
    params = envelope.parameters
    material = {
        "operation": "fairness", "dataset": params.get("dataset"),
        "grouping_key": sorted(params.get("grouping_key", [])),
        "filters": params.get("filters"),
        "model": envelope.model,
        "revision": envelope.model_spec.hf_repo if envelope.model_spec else None,
        "task": params.get("task"), "metrics": sorted(params.get("metrics", [])),
        "min_group_size": params.get("min_group_size"),
        "min_speakers_per_group": params.get("min_speakers_per_group"),
        "max_items_per_group": params.get("max_items_per_group"),
        "shard_size": params.get("shard_size"),
        "explain_per_group": params.get("explain_per_group"),
        "saliency_method": params.get("saliency_method"),
        "include_representation": params.get("include_representation"),
        "include_explanations": params.get("include_explanations"),
        "reference_group": params.get("reference_group"),
        "seed": params.get("seed"), "n_bootstrap": params.get("n_bootstrap"),
        "thresholds": params.get("thresholds"),
        "schema": envelope.result_schema_version, "code": envelope.code_version,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


async def complete_from_cache(envelope_data: dict[str, Any]) -> bool:
    envelope = TaskEnvelope.model_validate(envelope_data)
    digest = fairness_cache_key(envelope)
    storage = get_storage()
    cache_key = f"fairness/cache/{digest}.json"
    if not storage.exists(cache_key):
        return False
    payload = storage.get_json(cache_key)
    payload["job_id"] = envelope.job_id
    payload.setdefault("provenance", {})["cache_hit"] = True
    job_result_key = f"results/{envelope.session_id}/{envelope.job_id}/result.json"
    storage.put_json(job_result_key, payload)
    total = int(payload.get("provenance", {}).get("total_units", 1))
    await JobRepository().update(
        envelope.job_id, status=JobStatus.success,
        progress=JobProgress(current=total, total=total, message="Completed from cache"),
        result_key=job_result_key, cache_hit=True,
    )
    await redis_module.job_redis.hincrby("metrics:jobs", "fr10_cache_hits", 1)
    await redis_module.job_redis.hincrby("metrics:jobs", "fr10_success", 1)
    return True


async def prepare_analysis(envelope_data: dict[str, Any], celery_task_id: str) -> list[dict[str, Any]]:
    """Stage A: partition the dataset into groups, gate, classify design,
    build the shard plan, persist it, and transition QUEUED -> STARTED ->
    PROCESSING. Returns the infer-shard payloads for fr10_orchestrate to fan
    out as a chord."""
    envelope = TaskEnvelope.model_validate(envelope_data)
    jobs = JobRepository()
    await jobs.update(envelope.job_id, status=JobStatus.started, task_id=celery_task_id)

    params = envelope.parameters
    try:
        index = build_index(
            params["dataset"], params["grouping_key"], envelope.session_id,
            task=params["task"],
            min_group_size=params.get("min_group_size", settings.FR10_MIN_GROUP_SIZE),
            min_speakers_per_group=params.get("min_speakers_per_group", settings.FR10_MIN_SPEAKERS_PER_GROUP),
            filters=params.get("filters"), max_items_per_group=params.get("max_items_per_group"),
            reference_group=params.get("reference_group"), seed=params.get("seed", 0),
        )
    except (FairnessInputError, FileNotFoundError, ValueError) as exc:
        await jobs.update(
            envelope.job_id, status=JobStatus.failure,
            error=JobError(code="fr10_invalid_input", message=str(exc)[:500], retryable=False),
        )
        raise FairnessInputError(str(exc)) from exc

    plan = build_plan(
        index, shard_size=params.get("shard_size", 8),
        include_explanations=bool(params.get("include_explanations", True)),
        explain_per_group=params.get("explain_per_group", 12),
        explain_shard_size=params.get("explain_shard_size", 4),
        seed=params.get("seed", 0),
    )

    storage = get_storage()
    storage.put_json(_plan_key(envelope.session_id, envelope.job_id),
                      {"index": index.to_dict(), "plan": plan.to_dict()})

    total = len(plan.infer_shards) + len(plan.explain_shards) + 2  # +partition, +aggregate
    excluded_preview = ", ".join(e["label"] for e in index.excluded[:3])
    message = (
        f"Partitioned into {len(index.groups)} group(s) ({index.design})"
        + (f"; {len(index.excluded)} excluded ({excluded_preview}{'...' if len(index.excluded) > 3 else ''})"
           if index.excluded else "")
    )
    await jobs.update(
        envelope.job_id, status=JobStatus.processing,
        progress=JobProgress(current=1, total=total, message=message),
    )
    return [s.to_dict() for s in plan.infer_shards]


async def load_plan(envelope_data: dict[str, Any]) -> dict[str, Any]:
    envelope = TaskEnvelope.model_validate(envelope_data)
    return get_storage().get_json(_plan_key(envelope.session_id, envelope.job_id))


async def _advance_fr10_stage(job_id: str, counter: str, member: str, message: str) -> None:
    """Idempotent progress via Redis SADD (not INCR): acks_late redelivery of
    a shard task must never double-count (mirrors linguistic_acoustic_service
    ._advance_stage). Tracks infer/explain counters independently and sums
    them into one monotonic `current`."""
    key = f"job:{job_id}:{counter}"
    pipe = redis_module.job_redis.pipeline()
    pipe.sadd(key, member)
    pipe.expire(key, settings.JOB_TTL_SECONDS)
    await pipe.execute()
    done_infer = await redis_module.job_redis.scard(f"job:{job_id}:fr10:infer")
    done_explain = await redis_module.job_redis.scard(f"job:{job_id}:fr10:explain")
    record = await JobRepository().get(job_id)
    if not record:
        return
    current = min(1 + done_infer + done_explain, record.progress.total)
    await JobRepository().update(
        job_id, status=JobStatus.processing,
        progress=JobProgress(
            current=current, total=record.progress.total,
            message=f"{message} -- {done_infer} inference + {done_explain} explanation shard(s) done",
        ),
    )


def _run_op(model: str | None, model_spec: Any, operation: str, audio_path: str, parameters: dict[str, Any]) -> Any:
    from app.worker.model_adapters import get_model_adapter
    from app.worker.model_registry import model_registry

    if not model:
        return None
    adapter = get_model_adapter(model, model_spec)
    if not adapter.supports(operation):
        return None
    resource = model_registry.prepare(adapter, operation)
    return adapter.execute(operation, audio_path, parameters, resource)


def _extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        return output.get("text") or output.get("transcript") or ""
    return ""


def _extract_label_confidence(output: Any) -> tuple[str | None, float, dict[str, float]]:
    if isinstance(output, dict):
        label = output.get("predicted_emotion") or output.get("predicted_label")
        confidence = float(output.get("confidence", 0.0) or 0.0)
        probabilities = output.get("probabilities") or output.get("scores") or {}
        return label, confidence, probabilities
    return None, 0.0, {}


def _item_pred_cache_key(envelope: TaskEnvelope, item_id: str) -> str:
    material = {
        "operation": "fairness_prediction", "model": envelope.model,
        "revision": envelope.model_spec.hf_repo if envelope.model_spec else None,
        "dataset": envelope.parameters.get("dataset"), "item_id": item_id,
        "include_representation": bool(envelope.parameters.get("include_representation", True)),
        "schema": envelope.result_schema_version, "code": envelope.code_version,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


async def infer_shard(envelope_data: dict[str, Any], shard_data: dict[str, Any], celery_task_id: str) -> dict[str, Any]:
    """Stage B: one model load per shard (not per item). Prediction +
    embedding share the same adapter/registry resource where the model
    exposes both; per-item results are cached under a content-addressed key
    so a repeated fairness run over already-predicted items is nearly free."""
    del celery_task_id
    envelope = TaskEnvelope.model_validate(envelope_data)
    jobs = JobRepository()
    try:
        await _check_cancel(envelope.job_id, jobs)
    except JobCancelled:
        await jobs.update(envelope.job_id, status=JobStatus.cancelled)
        raise

    storage = get_storage()
    payload = storage.get_json(_plan_key(envelope.session_id, envelope.job_id))
    index = GroupIndex.from_dict(payload["index"])
    items_by_id = {item.item_id: item for group in index.groups for item in group.items}

    include_representation = bool(envelope.parameters.get("include_representation", True))
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    n_cached = n_failed = 0

    for item_id in shard_data["item_ids"]:
        item = items_by_id.get(item_id)
        if item is None:
            n_failed += 1
            failures.append({"item_id": item_id, "code": "unknown_item"})
            continue

        cache_key = f"fairness/cache/pred/{_item_pred_cache_key(envelope, item_id)}.json"
        if storage.exists(cache_key):
            cached = storage.get_json(cache_key)
            # `group_label` belongs to THIS job's grouping_key, not to the item.
            # _item_pred_cache_key deliberately omits grouping_key so one
            # prediction is reused across groupings, so a cache entry written by
            # a run grouped on `country` carries "bolivia" where a run grouped on
            # `native_language` needs "spanish". Stamp the current shard's label
            # over whatever wrote the entry -- otherwise aggregate_fairness
            # buckets those rows under a label no group in this job has, the real
            # group aggregates to zero rows, and the run dies with the misleading
            # "Fewer than two groups survived after inference failures."
            cached["group_label"] = shard_data["group_label"]
            records.append(cached)
            n_cached += 1
            continue

        try:
            audio_path = dataset_service.resolve_file(item.dataset, item.filename, item.session_id)
        except (FileNotFoundError, ValueError) as exc:
            n_failed += 1
            failures.append({"item_id": item_id, "code": "audio_not_found", "message": str(exc)[:200]})
            continue

        try:
            prediction = _run_op(envelope.model, envelope.model_spec, "prediction", str(audio_path), {})
            record: dict[str, Any] = {
                "item_id": item_id, "speaker_id": item.speaker_id, "content_id": item.content_id,
                "group_label": shard_data["group_label"],
            }
            if index.task == "classification":
                label, confidence, probabilities = _extract_label_confidence(prediction)
                record.update(predicted_label=label, reference_label=item.reference_label,
                              confidence=confidence, probabilities=probabilities)
            else:
                record.update(hypothesis=_extract_text(prediction), reference=item.reference_text)
            if include_representation:
                embedding = _run_op(envelope.model, envelope.model_spec, "embedding", str(audio_path), {})
                if embedding is not None:
                    vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                    record["embedding"] = [float(x) for x in vector]
            record = _jsonable(record)
            # Persist the cache entry grouping-neutral (see the cache-read branch
            # above): the entry is shared by every grouping of this dataset, so
            # baking one run's group_label into it poisons later runs.
            storage.put_json(cache_key, {k: v for k, v in record.items() if k != "group_label"})
            records.append(record)
        except Exception as exc:  # noqa: BLE001 -- per-item failure must not fail the shard
            n_failed += 1
            failures.append({"item_id": item_id, "code": "inference_failed", "message": str(exc)[:200]})

    artifact_key = f"fairness/{envelope.session_id}/{envelope.job_id}/infer/{shard_data['shard_id']}.json"
    storage.put_json(artifact_key, records)
    await _advance_fr10_stage(envelope.job_id, "fr10:infer", shard_data["shard_id"],
                              f"Inference: {shard_data['group_label']}")

    return {
        "shard_id": shard_data["shard_id"], "group_label": shard_data["group_label"],
        "operation": "prediction", "n_items": len(shard_data["item_ids"]),
        "n_cached": n_cached, "n_failed": n_failed, "artifact_key": artifact_key, "failures": failures,
    }


async def mark_infer_complete(envelope_data: dict[str, Any], infer_results: list[dict[str, Any]]) -> None:
    envelope = TaskEnvelope.model_validate(envelope_data)
    record = await JobRepository().get(envelope.job_id)
    if not record:
        return
    next_phase = "starting explanations" if envelope.parameters.get("include_explanations") else "aggregating"
    await JobRepository().update(
        envelope.job_id, status=JobStatus.processing,
        progress=JobProgress(
            current=record.progress.current, total=record.progress.total,
            message=f"Inference complete ({len(infer_results)} shard(s)); {next_phase}",
        ),
    )


async def explain_shard(envelope_data: dict[str, Any], shard_data: dict[str, Any], celery_task_id: str) -> dict[str, Any]:
    """Stage C: saliency + acoustic-grounding metrics for the balanced
    per-group explanation sample chosen in build_plan."""
    del celery_task_id
    from app.services import grounding_service, l2_arctic_annotations
    from app.services.saliency_service import generate_saliency
    import librosa

    envelope = TaskEnvelope.model_validate(envelope_data)
    jobs = JobRepository()
    try:
        await _check_cancel(envelope.job_id, jobs)
    except JobCancelled:
        await jobs.update(envelope.job_id, status=JobStatus.cancelled)
        raise

    storage = get_storage()
    payload = storage.get_json(_plan_key(envelope.session_id, envelope.job_id))
    index = GroupIndex.from_dict(payload["index"])
    items_by_id = {item.item_id: item for group in index.groups for item in group.items}
    method = envelope.parameters.get("saliency_method", "gradcam")
    is_l2_arctic = l2_arctic_annotations.is_l2_arctic(index.dataset)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    n_failed = 0

    for item_id in shard_data["item_ids"]:
        item = items_by_id.get(item_id)
        if item is None:
            n_failed += 1
            failures.append({"item_id": item_id, "code": "unknown_item"})
            continue
        try:
            audio_path = dataset_service.resolve_file(item.dataset, item.filename, item.session_id)
            saliency = generate_saliency(str(audio_path), envelope.model, method)
            series = saliency.get("series", [])
            total_duration = float(saliency.get("total_duration", 0.0))
            waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
            speech_mask = grounding_service.compute_speech_mask(waveform, int(sample_rate), len(series))
            grounding = grounding_service.grounding_metrics(series, total_duration, waveform, sample_rate)
            record: dict[str, Any] = {
                "item_id": item_id, "speaker_id": item.speaker_id, "content_id": item.content_id,
                "group_label": shard_data["group_label"], "grounding": grounding,
            }
            # FR-10 S6.2(a): L2-ARCTIC ships 891 human phone-error annotations
            # as a ground-truth reference attribution mask -- score this
            # item's saliency against it whenever an annotation exists.
            if is_l2_arctic:
                intervals = l2_arctic_annotations.annotations_for(item.filename)
                if intervals:
                    record["l2_phone_grounding"] = l2_arctic_annotations.phone_error_grounding(
                        series, total_duration, intervals, speech_mask,
                    )
            records.append(_jsonable(record))
        except Exception as exc:  # noqa: BLE001 -- per-item failure must not fail the shard
            n_failed += 1
            failures.append({"item_id": item_id, "code": "saliency_failed", "message": str(exc)[:200]})

    artifact_key = f"fairness/{envelope.session_id}/{envelope.job_id}/explain/{shard_data['shard_id']}.json"
    storage.put_json(artifact_key, records)
    await _advance_fr10_stage(envelope.job_id, "fr10:explain", shard_data["shard_id"],
                              f"Explanations: {shard_data['group_label']}")

    return {
        "shard_id": shard_data["shard_id"], "group_label": shard_data["group_label"],
        "operation": "saliency", "n_items": len(shard_data["item_ids"]),
        "n_failed": n_failed, "artifact_key": artifact_key, "failures": failures,
    }


def _build_caveats(index: GroupIndex) -> list[str]:
    caveats = []
    if index.design == "matched":
        caveats.append("Design is `matched`: content is controlled across groups; channel/recording "
                        "condition is not.")
    elif index.design == "partially_matched":
        caveats.append("Design is `partially_matched`: disparities use the shared-content subset where "
                        "enough shared items exist to pair on; otherwise a blockwise-by-speaker estimator "
                        "is used.")
    else:
        caveats.append("Design is `unmatched`: groups did not read matched content; disparities may "
                        "partly reflect content difficulty, not only group differences.")
    if any(g.speaker_confounded for g in index.groups):
        caveats.append("Groups with fewer speakers than `min_speakers_per_group` are reported but "
                        "excluded from headline verdicts -- a group effect cannot be separated from a "
                        "speaker-identity effect at n=1 speaker.")
    return caveats


def _build_dataset_extensions(
    index: GroupIndex, group_reports: dict[str, dict[str, Any]], grounding_by_group: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Part 1 S6: dataset-specific research value that a generic grouped-WER
    comparison cannot produce. Only fires for the two bundled corpora whose
    metadata supports it; every other dataset gets `None` here. Reads the
    `_valid_records` / `_grounding_records` private keys on group_reports, so
    MUST be called before aggregate_fairness pops them."""
    import numpy as np

    from app.services import l2_arctic_annotations, saa_reference_analysis

    items_by_id = {item.item_id: item for group in index.groups for item in group.items}

    if l2_arctic_annotations.is_l2_arctic(index.dataset):
        by_error_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for records in grounding_by_group.values():
            for record in records:
                phone_grounding = record.get("l2_phone_grounding")
                item = items_by_id.get(record["item_id"])
                if not item or not phone_grounding or phone_grounding.get("status") != "ok":
                    continue
                error_type = item.covariates.get("error_type", "unknown")
                by_error_type[error_type].append(phone_grounding)

        error_type_summary: dict[str, Any] = {}
        for error_type, entries in by_error_type.items():
            within_speech = [e["auroc_within_speech"] for e in entries if e.get("auroc_within_speech") is not None]
            error_type_summary[error_type] = {
                "n": len(entries),
                "mean_auroc": float(np.mean([e["auroc"] for e in entries])),
                "mean_auroc_within_speech": float(np.mean(within_speech)) if within_speech else None,
            }
        h1_sub_gt_add_gt_del = None
        if all(k in error_type_summary for k in ("substitution", "addition", "deletion")):
            h1_sub_gt_add_gt_del = bool(
                error_type_summary["substitution"]["mean_auroc"] > error_type_summary["addition"]["mean_auroc"]
                > error_type_summary["deletion"]["mean_auroc"]
            )

        wer_values: list[float] = []
        accentedness_values: list[float] = []
        group_labels: list[str] = []
        for label, report in group_reports.items():
            per_item_wer = {r["item_id"]: r["wer"] for r in report["performance"].get("per_item", [])}
            for record in report.get("_valid_records", []):
                item = items_by_id.get(record["item_id"])
                acc = l2_arctic_annotations.accentedness(item.covariates) if item else None
                wer_value = per_item_wer.get(record["item_id"])
                if acc is not None and wer_value is not None:
                    wer_values.append(wer_value)
                    accentedness_values.append(acc)
                    group_labels.append(label)
        regression = (
            l2_arctic_annotations.ols_with_group_dummies(wer_values, accentedness_values, group_labels)
            if len(wer_values) >= 5 else {"status": "insufficient_data", "n": len(wer_values)}
        )

        return {
            "kind": "l2_arctic",
            "phone_error_grounding_by_error_type": error_type_summary,
            "h1_substitution_gt_addition_gt_deletion": h1_sub_gt_add_gt_del,
            "equal_accentedness_regression": regression,
            "caveat": (
                "This bundled subset has exactly 1 speaker per native_language; "
                "`group_intercepts` in the regression are descriptive only and are NOT "
                "separable from a speaker-identity effect (see the speaker_confounded "
                "flag on each group)."
            ),
        }

    if saa_reference_analysis.is_saa(index.dataset):
        per_group_hypotheses = {
            label: [r["hypothesis"] for r in report.get("_valid_records", []) if r.get("hypothesis")]
            for label, report in group_reports.items()
        }
        reference_text = next(
            (r.get("reference") for report in group_reports.values()
             for r in report.get("_valid_records", []) if r.get("reference")),
            None,
        )
        heatmap = (
            saa_reference_analysis.position_error_heatmap(per_group_hypotheses, reference_text)
            if reference_text else None
        )

        wer_values, onset_ages, ages = [], [], []
        for report in group_reports.values():
            for pr in report["performance"].get("per_item", []):
                item = items_by_id.get(pr["item_id"])
                if not item:
                    continue
                try:
                    onset = float(item.covariates.get("age_english_onset"))
                    age = float(item.covariates.get("age"))
                except (TypeError, ValueError):
                    continue
                wer_values.append(pr["wer"])
                onset_ages.append(onset)
                ages.append(age)
        dose_response = (
            saa_reference_analysis.onset_age_dose_response(wer_values, onset_ages, ages)
            if len(wer_values) >= 5 else {"status": "insufficient_data", "n": len(wer_values)}
        )

        return {
            "kind": "saa",
            "reference_position_heatmap": heatmap,
            "age_onset_dose_response": dose_response,
        }

    return None


# Metric id -> the field each metric reads from a group's performance/grounding
# dict, and the ItemMetric extractor used to build per-item values for the
# bootstrap. Only metrics whose group-level value is a simple mean of a
# per-item quantity are bootstrapped/disparity-tested here; macro_f1 and ece
# are NOT simple per-item means (macro_f1 must be recomputed from the full
# label set on every resample), so they are reported descriptively only --
# see docs/FR10plan.md Part 1 S4.5 and the note in aggregate_fairness.
_BOOTSTRAPPABLE_METRICS = {
    "wer": ("performance", "macro_wer", "wer"),
    "cer": ("performance", "macro_cer", "cer"),
    "accuracy": ("performance", "accuracy", "correct"),
    "grounding_lift": ("grounding", "grounding_lift", "grounding_lift"),
    "attribution_entropy": ("grounding", "attribution_entropy", "attribution_entropy"),
}


async def aggregate_fairness(
    explain_results: list[dict[str, Any]], infer_results: list[dict[str, Any]], envelope_data: dict[str, Any],
) -> None:
    """Stage D: load every shard artifact, compute grouped performance,
    representation and grounding metrics, run the blockwise bootstrap with
    Holm-Bonferroni correction, build verdicts, and write the final report."""
    from app.services import fairness_metrics_service, grounding_service

    envelope = TaskEnvelope.model_validate(envelope_data)
    jobs = JobRepository()
    storage = get_storage()
    params = envelope.parameters

    # A cancel requested between the infer/explain stages and this one must
    # not be clobbered by a SUCCESS written after the fact -- terminal no-op,
    # mirroring JobRepository.update's own terminal-state guard.
    if await jobs.cancellation_requested(envelope.job_id):
        await jobs.update(envelope.job_id, status=JobStatus.cancelled)
        return

    payload = storage.get_json(_plan_key(envelope.session_id, envelope.job_id))
    index = GroupIndex.from_dict(payload["index"])

    # Bucket by the SHARD's group label, not each record's. The shard label comes
    # straight from this job's plan, so it is authoritative; a record's own
    # group_label can be stale if it was served from the grouping-neutral
    # per-item prediction cache written by a run with a different grouping_key.
    records_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shard_result in infer_results:
        artifact_key = shard_result.get("artifact_key")
        if artifact_key:
            label = shard_result.get("group_label")
            for record in storage.get_json(artifact_key):
                records_by_group[label or record["group_label"]].append(record)

    grounding_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shard_result in explain_results:
        artifact_key = shard_result.get("artifact_key")
        if artifact_key:
            label = shard_result.get("group_label")
            for record in storage.get_json(artifact_key):
                grounding_by_group[label or record["group_label"]].append(record)

    min_group_size = params.get("min_group_size", settings.FR10_MIN_GROUP_SIZE)
    min_speakers = params.get("min_speakers_per_group", settings.FR10_MIN_SPEAKERS_PER_GROUP)
    requested_metrics = params.get("metrics") or ["wer", "cer"]
    # The frontend submits task="auto" and doesn't know the resolved task at
    # request time, so it sends both ASR and classification metric ids. Only
    # per-item fields for the actual task exist on inference records (ASR
    # records have no "correct" key, classification records have no "wer"/
    # "cer"), so cross-task bootstrappable metric ids must be dropped here --
    # else _item_metrics_for raises KeyError reaching into the wrong shape.
    _WRONG_TASK_METRICS = {"transcription": {"accuracy"}, "classification": {"wer", "cer"}}
    requested_metrics = [m for m in requested_metrics if m not in _WRONG_TASK_METRICS.get(index.task, set())]
    include_representation = bool(params.get("include_representation", True))
    n_boot = int(params.get("n_bootstrap", 2000))
    seed = int(params.get("seed", 0))
    thresholds = params.get("thresholds") or {}
    absolute_threshold = float(thresholds.get("absolute_error_gap", 0.05))
    ratio_threshold = float(thresholds.get("disparity_ratio", 1.25))

    excluded: list[dict[str, Any]] = list(index.excluded)
    all_labels: list[str] = []
    if index.task == "classification":
        seen: set[str] = set()
        for recs in records_by_group.values():
            for r in recs:
                if r.get("reference_label"):
                    seen.add(r["reference_label"])
                if r.get("predicted_label"):
                    seen.add(r["predicted_label"])
        all_labels = sorted(seen)

    group_reports: dict[str, dict[str, Any]] = {}
    for group in index.groups:
        raw = records_by_group.get(group.label, [])
        valid = [r for r in raw if r.get("hypothesis") is not None or r.get("predicted_label") is not None]
        n_failed = max(len(group.items) - len(valid), 0)
        if len(valid) < min_group_size:
            excluded.append({"label": group.label, "n_items": len(valid), "n_failed": n_failed,
                              "reason": "insufficient_after_failures", "min_group_size": min_group_size})
            continue

        speakers = {r.get("speaker_id") for r in valid if r.get("speaker_id")}
        speaker_confounded = len(speakers) < min_speakers

        if index.task == "classification":
            performance = fairness_metrics_service.classification_group_metrics(valid, all_labels)
        else:
            performance = fairness_metrics_service.asr_group_metrics(valid, params.get("language", "en"))

        grounding_records = grounding_by_group.get(group.label, [])
        grounding = (
            grounding_service.group_grounding_summary([r["grounding"] for r in grounding_records])
            if grounding_records else {"status": "unavailable", "n_explained": 0}
        )

        group_reports[group.label] = {
            "label": group.label, "key": list(group.key), "n_items": len(valid), "n_speakers": len(speakers),
            "n_failed": n_failed, "speaker_confounded": speaker_confounded,
            "performance": performance, "grounding": grounding,
            "_valid_records": valid, "_grounding_records": grounding_records,
        }

    if len(group_reports) < 2:
        await jobs.update(
            envelope.job_id, status=JobStatus.failure,
            error=JobError(code="fr10_insufficient_groups",
                           message="Fewer than two groups survived after inference failures.", retryable=False),
        )
        return

    reference_label = index.reference_group if index.reference_group in group_reports else next(iter(group_reports))

    representation: dict[str, Any] = {"status": "unavailable"}
    if include_representation:
        all_vectors, all_group_labels, all_speaker_ids = [], [], []
        for label, report in group_reports.items():
            for record in report["_valid_records"]:
                if record.get("embedding"):
                    all_vectors.append(record["embedding"])
                    all_group_labels.append(label)
                    all_speaker_ids.append(record.get("speaker_id"))
        if len(all_vectors) >= 2 * len(group_reports):
            sep = fairness_metrics_service.separability(all_vectors, all_group_labels, all_speaker_ids, seed=seed)
            leak = fairness_metrics_service.group_leakage(all_vectors, all_group_labels, all_speaker_ids, seed=seed)
            representation = {**sep, "leakage": leak}

    def _item_metrics_for(label: str, metric_id: str) -> list[fairness_metrics_service.ItemMetric]:
        section, _, item_key = _BOOTSTRAPPABLE_METRICS[metric_id]
        report = group_reports[label]
        if section == "performance":
            source = report["performance"]["per_item"]
            if metric_id == "accuracy":
                return [fairness_metrics_service.ItemMetric(
                    r["item_id"], r.get("speaker_id"), r.get("content_id"), 1.0 if r["correct"] else 0.0,
                ) for r in source]
            return [fairness_metrics_service.ItemMetric(
                r["item_id"], r.get("speaker_id"), r.get("content_id"), float(r[item_key]),
            ) for r in source]
        out = []
        for record in report["_grounding_records"]:
            grounding = record["grounding"]
            if grounding.get("status") == "ok" and grounding.get(item_key) is not None:
                out.append(fairness_metrics_service.ItemMetric(
                    record["item_id"], record.get("speaker_id"), record.get("content_id"), float(grounding[item_key]),
                ))
        return out

    disparities: list[dict[str, Any]] = []
    pending_pvals: list[tuple[int, float]] = []
    bootstrappable = [m for m in requested_metrics if m in _BOOTSTRAPPABLE_METRICS]
    for metric_id in bootstrappable:
        ref_items = _item_metrics_for(reference_label, metric_id)
        if len(ref_items) < 2:
            continue
        for label in group_reports:
            if label == reference_label:
                continue
            other_items = _item_metrics_for(label, metric_id)
            if len(other_items) < 2:
                continue
            gap = fairness_metrics_service.estimate_gap(
                ref_items, other_items, metric=metric_id, design=index.design, n_boot=n_boot, seed=seed,
            )
            pending_pvals.append((len(disparities), gap["p_value"]))
            disparities.append({"group": label, "vs": reference_label, **gap})

    if pending_pvals:
        idxs, pvals = zip(*pending_pvals)
        for i, p_adj in zip(idxs, fairness_metrics_service.holm_bonferroni(list(pvals))):
            disparities[i]["p_adjusted"] = p_adj

    verdicts: list[dict[str, Any]] = []
    verdicted_groups: set[str] = set()
    for gap in disparities:
        report = group_reports[gap["group"]]
        verdicts.append(fairness_metrics_service.verdict_for(
            gap, group_label=gap["group"], reference_label=reference_label, n_speakers=report["n_speakers"],
            design=index.design, min_speakers=min_speakers,
            absolute_threshold=absolute_threshold, ratio_threshold=ratio_threshold,
        ))
        verdicted_groups.add(gap["group"])
    for label, report in group_reports.items():
        if report["speaker_confounded"] and label != reference_label and label not in verdicted_groups:
            verdicts.append({
                "group": label, "metric": "*", "verdict": "inconclusive", "severity": "speaker_confounded",
                "message": f"{label} has {report['n_speakers']} speaker(s); excluded from headline verdicts.",
            })

    summary: dict[str, Any] = {}
    for metric_id in bootstrappable:
        section, field_name, _ = _BOOTSTRAPPABLE_METRICS[metric_id]
        values, weights = {}, {}
        for label, report in group_reports.items():
            source = report[section]
            if section == "grounding" and source.get("status") != "ok":
                continue
            values[label] = source[field_name]
            weights[label] = report["n_items"] if section == "performance" else source.get("n_explained", 1)
        stat = fairness_metrics_service.disparity_summary(values, weights)
        if stat:
            summary[metric_id] = stat

    dataset_extensions = _build_dataset_extensions(index, group_reports, grounding_by_group)

    for report in group_reports.values():
        report.pop("_valid_records", None)
        report.pop("_grounding_records", None)

    verdict_rank = {"disparity_detected": 0, "inconclusive": 1, "no_evidence_of_disparity": 2}
    record = await jobs.get(envelope.job_id)
    total_units = record.progress.total if record else 1

    result = {
        "schema_version": envelope.result_schema_version, "feature": "FR-10",
        "job_id": envelope.job_id, "model": envelope.model, "task": index.task,
        "dataset": index.dataset, "grouping_key": index.grouping_key, "reference_group": reference_label,
        "design": {**index.design_evidence, "type": index.design},
        "groups": list(group_reports.values()),
        "excluded_groups": excluded,
        "disparities": disparities,
        "summary": summary,
        "representation": representation,
        "verdicts": sorted(verdicts, key=lambda v: verdict_rank.get(v["verdict"], 3)),
        "provenance": {
            "code_version": envelope.code_version, "seed": seed, "n_boot": n_boot,
            "normalizer": "EnglishTextNormalizer" if (params.get("language") or "en").startswith("en") else "BasicTextNormalizer",
            "saliency_method": params.get("saliency_method", "gradcam"),
            "min_group_size": min_group_size, "min_speakers": min_speakers,
            "multiplicity_correction": "holm", "n_comparisons": len(disparities),
            "total_units": total_units, "cache_hit": False,
        },
        "caveats": _build_caveats(index),
        "dataset_extensions": dataset_extensions,
    }

    job_result_key = f"results/{envelope.session_id}/{envelope.job_id}/result.json"
    result = _jsonable(result)
    storage.put_json(job_result_key, result)
    storage.put_json(f"fairness/cache/{fairness_cache_key(envelope)}.json", result)

    await jobs.update(
        envelope.job_id, status=JobStatus.success,
        progress=JobProgress(current=total_units, total=total_units, message="Completed"),
        result_key=job_result_key, cache_hit=False,
    )
    await redis_module.job_redis.hincrby("metrics:jobs", "fr10_success", 1)
