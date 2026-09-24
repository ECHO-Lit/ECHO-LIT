/**
 * Model / dataset differential comparison (UC: "Model and dataset comparison").
 *
 * Two shapes of question, one engine:
 *   - `models`   — two models on ONE dataset. The two runs cover the *same
 *                  files*, so results align per item by filename, and each item
 *                  carries a per-model metric and their delta. This is the
 *                  acceptance path.
 *   - `datasets` — one model on TWO datasets. The runs cover *different* files,
 *                  so there is nothing to align per item; the comparison is the
 *                  aggregate metric on each dataset and their difference, with
 *                  each side's items listed in parallel.
 *
 * The whole thing is built on the existing `prediction` job and the dataset
 * metadata endpoint — no backend change. The alignment and scoring
 * (`buildComparison`) are pure so they can be unit-tested without a server;
 * `runComparison` is the thin async shell that fetches predictions and calls it.
 */

import { API_BASE } from './api';
import { describeHttpError } from './httpError';
import { materializeAll, runJob, type JobStatus } from './jobs';
import { computeTranscriptMetrics } from './textMetrics';
import type { CustomModel } from './models';

export type ModelKind = 'asr' | 'classification';
export type CompareMode = 'models' | 'datasets';

/** One side of the comparison: which model ran over which dataset. */
export interface SideSpec {
  model: string;
  dataset: string;
}

export interface CompareRequest {
  mode: CompareMode;
  a: SideSpec;
  b: SideSpec;
  maxItems: number;
}

/** A model's prediction on one file, already reduced to comparable text. */
export interface SidePrediction {
  prediction: string;
  /** WER (asr, lower better) or 1/0 correct (classification); null without truth. */
  metric: number | null;
}

export interface ItemComparison {
  filename: string;
  groundTruth: string | null;
  a: SidePrediction | null;
  b: SidePrediction | null;
  /** `b.metric − a.metric`, present only in `models` mode with a ground truth. */
  delta: number | null;
  /** Whether the two sides produced the same answer (normalised). */
  agree: boolean | null;
}

export interface SideAggregate {
  label: string;
  model: string;
  dataset: string;
  itemCount: number;
  withTruth: number;
  /** Mean WER (asr) or accuracy 0–1 (classification), over items with truth. */
  primary: number | null;
  /** asr: mean word-accuracy %; classification: primary × 100. */
  accuracyPct: number | null;
}

export interface ComparisonResult {
  kind: ModelKind;
  mode: CompareMode;
  a: SideAggregate;
  b: SideAggregate;
  /** Human label for `primary`: "Word error rate" or "Accuracy". */
  primaryLabel: string;
  /** `b.primary − a.primary`; sign is raw, `betterSide` interprets direction. */
  primaryDelta: number | null;
  betterSide: 'a' | 'b' | 'tie' | null;
  /** Fraction of aligned items where both sides agreed (models mode only). */
  agreementRate: number | null;
  items: ItemComparison[];
}

/** The rows we hand `buildComparison`: raw model output per file, per side. */
export interface SideRun {
  spec: SideSpec;
  label: string;
  items: { filename: string; groundTruth: string | null; prediction: string }[];
}

const ASR_TRUTH_KEYS = ['sentence', 'transcript', 'text', 'statement'] as const;
const LABEL_TRUTH_KEYS = ['emotion', 'label', 'emotion_label', 'class'] as const;
const FILENAME_KEYS = ['path', 'filepath', 'file', 'filename'] as const;

const BUILTIN_KIND: Record<string, ModelKind> = {
  'whisper-base': 'asr',
  'whisper-large': 'asr',
  wav2vec2: 'classification',
};

/** Which kind of task a model performs — decides the metric and comparability. */
export function modelKind(modelId: string, customModels: CustomModel[] = []): ModelKind | null {
  if (modelId in BUILTIN_KIND) return BUILTIN_KIND[modelId];
  const custom = customModels.find((m) => m.model_id === modelId);
  if (!custom?.kind) return null;
  return custom.kind === 'audio_classification' ? 'classification' : 'asr';
}

function baseFilename(row: Record<string, unknown>): string {
  for (const key of FILENAME_KEYS) {
    const raw = row[key];
    if (typeof raw === 'string' && raw) return raw.split('/').pop()?.split('\\').pop() || raw;
  }
  return typeof row['id'] === 'string' ? row['id'] : '';
}

/** Ground truth for a metadata row, or null when the column is absent/blank. */
export function groundTruthFor(kind: ModelKind, row: Record<string, unknown>): string | null {
  const keys = kind === 'asr' ? ASR_TRUTH_KEYS : LABEL_TRUTH_KEYS;
  for (const key of keys) {
    const raw = row[key];
    if (typeof raw === 'string' && raw.trim() && raw.trim().toLowerCase() !== 'unknown') {
      return raw.trim();
    }
  }
  return null;
}

/** Reduce a prediction job's per-item result to comparable text. */
export function predictionText(kind: ModelKind, result: unknown): string {
  if (result == null) return '';
  if (typeof result === 'string') return result;
  const r = result as Record<string, unknown>;
  if (kind === 'asr') {
    const t = r.text ?? r.predicted_transcript ?? r.transcript;
    return typeof t === 'string' ? t : '';
  }
  const label = r.predicted_emotion ?? r.predicted_label ?? r.emotion ?? r.label ?? r.prediction;
  return typeof label === 'string' ? label : label != null ? String(label) : '';
}

function normalise(text: string): string {
  return text.trim().toLowerCase().replace(/\s+/g, ' ');
}

/**
 * Per-item metric against ground truth: WER for ASR (lower is better), a 1/0
 * correctness flag for classification (higher is better). Null without truth.
 */
export function metricForItem(kind: ModelKind, prediction: string, truth: string | null): number | null {
  if (truth == null) return null;
  if (kind === 'asr') return computeTranscriptMetrics(prediction, truth).word_error_rate;
  return normalise(prediction) === normalise(truth) ? 1 : 0;
}

function aggregate(kind: ModelKind, run: SideRun): SideAggregate {
  const withTruth = run.items.filter((i) => i.groundTruth != null);
  let primary: number | null = null;
  let accuracyPct: number | null = null;

  if (withTruth.length > 0) {
    if (kind === 'asr') {
      const wers = withTruth.map((i) => computeTranscriptMetrics(i.prediction, i.groundTruth as string));
      primary = wers.reduce((s, m) => s + m.word_error_rate, 0) / wers.length;
      accuracyPct = wers.reduce((s, m) => s + m.accuracy_percentage, 0) / wers.length;
    } else {
      const correct = withTruth.filter((i) => normalise(i.prediction) === normalise(i.groundTruth as string)).length;
      primary = correct / withTruth.length;
      accuracyPct = primary * 100;
    }
  }

  return {
    label: run.label,
    model: run.spec.model,
    dataset: run.spec.dataset,
    itemCount: run.items.length,
    withTruth: withTruth.length,
    primary,
    accuracyPct,
  };
}

/**
 * The core, pure comparison. In `models` mode the two runs share a dataset, so
 * items are joined by filename; in `datasets` mode they are listed side by side
 * with no join (different files). Both paths compute the aggregate difference,
 * which is the headline number.
 */
export function buildComparison(
  kind: ModelKind,
  mode: CompareMode,
  runA: SideRun,
  runB: SideRun,
): ComparisonResult {
  const primaryLabel = kind === 'asr' ? 'Word error rate' : 'Accuracy';
  const aggA = aggregate(kind, runA);
  const aggB = aggregate(kind, runB);

  const items: ItemComparison[] = [];
  let agree = 0;
  let agreeDenom = 0;

  if (mode === 'models') {
    // Same dataset both sides → align by filename.
    const byNameB = new Map(runB.items.map((i) => [i.filename, i]));
    for (const ia of runA.items) {
      const ib = byNameB.get(ia.filename);
      const truth = ia.groundTruth ?? ib?.groundTruth ?? null;
      const ma = metricForItem(kind, ia.prediction, truth);
      const mb = ib ? metricForItem(kind, ib.prediction, truth) : null;
      const bothAgree = ib ? normalise(ia.prediction) === normalise(ib.prediction) : null;
      if (bothAgree != null) {
        agreeDenom += 1;
        if (bothAgree) agree += 1;
      }
      items.push({
        filename: ia.filename,
        groundTruth: truth,
        a: { prediction: ia.prediction, metric: ma },
        b: ib ? { prediction: ib.prediction, metric: mb } : null,
        delta: ma != null && mb != null ? mb - ma : null,
        agree: bothAgree,
      });
    }
  } else {
    // Different datasets → parallel lists, no per-item join.
    for (const ia of runA.items) {
      items.push({
        filename: ia.filename,
        groundTruth: ia.groundTruth,
        a: { prediction: ia.prediction, metric: metricForItem(kind, ia.prediction, ia.groundTruth) },
        b: null,
        delta: null,
        agree: null,
      });
    }
    for (const ib of runB.items) {
      items.push({
        filename: ib.filename,
        groundTruth: ib.groundTruth,
        a: null,
        b: { prediction: ib.prediction, metric: metricForItem(kind, ib.prediction, ib.groundTruth) },
        delta: null,
        agree: null,
      });
    }
  }

  const primaryDelta = aggA.primary != null && aggB.primary != null ? aggB.primary - aggA.primary : null;
  let betterSide: ComparisonResult['betterSide'] = null;
  if (primaryDelta != null) {
    // ASR: lower WER wins. Classification: higher accuracy wins.
    const lowerIsBetter = kind === 'asr';
    if (Math.abs(primaryDelta) < 1e-9) betterSide = 'tie';
    else if ((primaryDelta < 0) === lowerIsBetter) betterSide = 'b';
    else betterSide = 'a';
  }

  return {
    kind,
    mode,
    a: aggA,
    b: aggB,
    primaryLabel,
    primaryDelta,
    betterSide,
    agreementRate: mode === 'models' && agreeDenom > 0 ? agree / agreeDenom : null,
    items,
  };
}

/** Why a chosen pair cannot be compared, or null if it can. Feeds the UI guard. */
export function incompatibilityReason(
  mode: CompareMode,
  a: SideSpec,
  b: SideSpec,
  customModels: CustomModel[] = [],
): string | null {
  const kindA = modelKind(a.model, customModels);
  const kindB = modelKind(b.model, customModels);
  if (kindA == null) return `The task type of "${a.model}" is unknown; it may still be validating.`;
  if (kindB == null) return `The task type of "${b.model}" is unknown; it may still be validating.`;

  if (mode === 'models') {
    if (a.dataset !== b.dataset) return 'Comparing two models requires the same dataset on both sides.';
    if (kindA !== kindB) {
      const name = (k: ModelKind) => (k === 'asr' ? 'speech-to-text' : 'classification');
      return `These models do different tasks (${name(kindA)} vs ${name(kindB)}), so their outputs are not comparable. Pick two models of the same type.`;
    }
    if (a.model === b.model) return 'Pick two different models to compare.';
  } else {
    if (a.model !== b.model) return 'Comparing two datasets requires the same model on both sides.';
    if (a.dataset === b.dataset) return 'Pick two different datasets to compare.';
  }
  return null;
}

interface DatasetRow extends Record<string, unknown> {}

async function fetchDatasetRows(dataset: string, signal?: AbortSignal): Promise<DatasetRow[]> {
  const response = await fetch(`${API_BASE}/${encodeURIComponent(dataset)}/metadata`, {
    credentials: 'include',
    signal,
  });
  if (!response.ok) throw await describeHttpError(response, `Could not load "${dataset}"`);
  const rows = await response.json();
  return Array.isArray(rows) ? rows : [];
}

export interface RunProgress {
  side: 'a' | 'b';
  phase: 'metadata' | 'materialising' | 'predicting' | 'done';
  job?: JobStatus;
}

/** Run one side end to end: metadata → materialise → predict → reduce to text. */
async function runSide(
  kind: ModelKind,
  spec: SideSpec,
  label: string,
  maxItems: number,
  onProgress: (p: RunProgress) => void,
  side: 'a' | 'b',
  signal?: AbortSignal,
): Promise<SideRun> {
  onProgress({ side, phase: 'metadata' });
  const rows = await fetchDatasetRows(spec.dataset, signal);
  const usable = rows.map((row) => ({ filename: baseFilename(row), row })).filter((r) => r.filename);
  const chosen = usable.slice(0, maxItems);
  if (chosen.length === 0) throw new Error(`"${spec.dataset}" has no usable audio files.`);

  onProgress({ side, phase: 'materialising' });
  const references = await materializeAll(spec.dataset, chosen.map((c) => c.filename), signal);
  const audioIds = references.map((r) => r.audio_id);

  onProgress({ side, phase: 'predicting' });
  const result = await runJob<{ items: { audio_id: string; result: unknown }[] }>(
    { operation: 'prediction', model: spec.model, audio_ids: audioIds },
    { signal, onProgress: (job) => onProgress({ side, phase: 'predicting', job }) },
  );

  const items = result.items.map((item, index) => ({
    filename: chosen[index].filename,
    groundTruth: groundTruthFor(kind, chosen[index].row),
    prediction: predictionText(kind, item.result),
  }));

  onProgress({ side, phase: 'done' });
  return { spec, label, items };
}

/**
 * Full run: validate, execute both sides, align and score. Throws the
 * incompatibility reason before spending any compute if the pair is invalid.
 */
export async function runComparison(
  request: CompareRequest,
  customModels: CustomModel[],
  options: { signal?: AbortSignal; onProgress?: (p: RunProgress) => void } = {},
): Promise<ComparisonResult> {
  const reason = incompatibilityReason(request.mode, request.a, request.b, customModels);
  if (reason) throw new Error(reason);

  const kind = modelKind(request.a.model, customModels) as ModelKind;
  const onProgress = options.onProgress ?? (() => {});

  const runA = await runSide(kind, request.a, sideLabel(request, 'a'), request.maxItems, onProgress, 'a', options.signal);
  const runB = await runSide(kind, request.b, sideLabel(request, 'b'), request.maxItems, onProgress, 'b', options.signal);

  return buildComparison(kind, request.mode, runA, runB);
}

/** The label that identifies a side to the reader — whatever varies between them. */
export function sideLabel(request: CompareRequest, side: 'a' | 'b'): string {
  const spec = request[side];
  return request.mode === 'models' ? spec.model : spec.dataset;
}

/** Present a metric for the reader: WER as a percentage, accuracy as a percentage. */
export function formatPrimary(kind: ModelKind, value: number | null): string {
  if (value == null) return '—';
  return `${(value * 100).toFixed(1)}%`;
}
