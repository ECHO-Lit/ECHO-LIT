import { API_BASE } from './api';

export type FairnessMetricId =
  | 'wer' | 'cer' | 'accuracy' | 'macro_f1' | 'ece'
  | 'silhouette' | 'leakage' | 'grounding_lift' | 'attribution_entropy';

export type FairnessDesign = 'matched' | 'partially_matched' | 'unmatched';
export type FairnessVerdictKind = 'disparity_detected' | 'inconclusive' | 'no_evidence_of_disparity';

export interface GroupableColumn {
  column: string;
  n_values: number;
  n_missing: number;
  counts: Record<string, number>;
  is_speaker_column: boolean;
  is_stratum_only: boolean;
}

export interface GroupableColumnsResponse {
  dataset: string;
  n_rows: number;
  speaker_column: string | null;
  content_column: string | null;
  columns: GroupableColumn[];
}

export interface FairnessRequest {
  dataset: string;
  grouping_key: string[];
  model: string;
  task?: 'transcription' | 'classification' | 'auto';
  reference_group?: string | null;
  filters?: Record<string, string[]> | null;
  language?: string | null;
  metrics?: FairnessMetricId[];
  include_representation?: boolean;
  include_explanations?: boolean;
  include_faithfulness?: boolean;
  saliency_method?: 'gradcam' | 'lime' | 'shap';
  min_group_size?: number;
  min_speakers_per_group?: number;
  max_items_per_group?: number | null;
  shard_size?: number;
  explain_per_group?: number;
  n_bootstrap?: number;
  seed?: number;
}

export interface FairnessAccepted {
  job_id: string;
  status: string;
  status_url: string;
  result_url: string;
  dataset: string;
  grouping_key: string[];
  poll_after_ms: number;
  notes: string[];
}

export interface GroupPerformance {
  n_items: number;
  n_speakers: number;
  // ASR
  micro_wer?: number;
  macro_wer?: number;
  micro_cer?: number;
  macro_cer?: number;
  sub_rate?: number;
  del_rate?: number;
  ins_rate?: number;
  insertion_ratio?: number;
  // Classification
  accuracy?: number;
  macro_f1?: number;
  macro_recall?: number;
  balanced_accuracy?: number;
  ece?: number;
  mean_confidence?: number;
  per_item: Array<Record<string, unknown>>;
}

export interface GroupGrounding {
  status: 'ok' | 'unavailable';
  n_explained: number;
  grounding_lift?: number;
  attribution_entropy?: number;
  attribution_gini?: number;
  top10_mass?: number;
}

export interface FairnessGroupReport {
  label: string;
  key: string[];
  n_items: number;
  n_speakers: number;
  n_failed: number;
  speaker_confounded: boolean;
  performance: GroupPerformance;
  grounding: GroupGrounding;
}

export interface FairnessExcludedGroup {
  label: string;
  n_items: number;
  n_speakers?: number;
  n_failed?: number;
  reason: string;
  min_group_size?: number;
}

export interface FairnessGap {
  group: string;
  vs: string;
  metric: FairnessMetricId;
  point: number;
  ci: [number, number];
  p_value: number;
  p_adjusted?: number;
  method: 'paired_by_content' | 'unpaired';
  block_unit: 'content_id' | 'speaker' | 'utterance';
  n_blocks: number;
  ratio: number;
  effect_size: number | null;
  mde: number | null;
  n_boot: number;
}

export interface FairnessDisparitySummary {
  max_gap: number;
  disparity_ratio: number;
  gini: number;
  weighted_std: number;
  best_group: string;
  worst_group: string;
}

export interface FairnessVerdict {
  group: string;
  metric: string;
  verdict: FairnessVerdictKind;
  severity: string;
  message: string;
}

export interface FairnessRepresentation {
  status: 'ok' | 'unavailable' | 'insufficient_data';
  silhouette_by_group_label?: number;
  per_group_silhouette?: Record<string, number>;
  permutation_null?: { mean: number; sd: number; n: number };
  silhouette_z?: number;
  silhouette_by_speaker?: number | null;
  davies_bouldin?: number;
  calinski_harabasz?: number;
  leakage?: {
    status: 'ok' | 'skipped';
    balanced_accuracy?: number;
    std?: number;
    chance?: number;
    leakage_lift?: number;
    n_folds?: number;
    split?: string;
    reason?: string;
  };
}

export interface FairnessDatasetExtensions {
  kind: 'l2_arctic' | 'saa';
  // l2_arctic
  phone_error_grounding_by_error_type?: Record<string, { n: number; mean_auroc: number; mean_auroc_within_speech: number | null }>;
  h1_substitution_gt_addition_gt_deletion?: boolean | null;
  equal_accentedness_regression?: Record<string, unknown>;
  caveat?: string;
  // saa
  reference_position_heatmap?: { words: string[]; error_rate_by_group: Record<string, number[]> } | null;
  age_onset_dose_response?: Record<string, unknown>;
}

export interface FairnessReport {
  schema_version: string;
  feature: 'FR-10';
  job_id: string;
  model: string;
  task: 'transcription' | 'classification';
  dataset: string;
  grouping_key: string[];
  reference_group: string;
  design: {
    type: FairnessDesign;
    content_column?: string;
    n_shared_content?: number;
    n_total_content?: number;
    jaccard?: number;
    per_group_coverage?: Record<string, number>;
    reason?: string;
  };
  groups: FairnessGroupReport[];
  excluded_groups: FairnessExcludedGroup[];
  disparities: FairnessGap[];
  summary: Record<string, FairnessDisparitySummary>;
  representation: FairnessRepresentation;
  verdicts: FairnessVerdict[];
  provenance: Record<string, unknown>;
  caveats: string[];
  dataset_extensions: FairnessDatasetExtensions | null;
}

async function parseError(response: Response): Promise<Error> {
  try {
    const body = await response.json();
    if (Array.isArray(body.detail)) {
      return new Error(body.detail.map((entry: any) => entry.msg).join('; '));
    }
    return new Error(body.detail || `Request failed (${response.status})`);
  } catch {
    return new Error(`Request failed (${response.status})`);
  }
}

export async function fetchGroupableColumns(dataset: string, signal?: AbortSignal): Promise<GroupableColumnsResponse> {
  const response = await fetch(
    `${API_BASE}/api/v1/analyses/fairness/groupable?dataset=${encodeURIComponent(dataset)}`,
    { credentials: 'include', signal },
  );
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export async function submitFairnessAnalysis(body: FairnessRequest, signal?: AbortSignal): Promise<FairnessAccepted> {
  const response = await fetch(`${API_BASE}/api/v1/analyses/fairness`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw await parseError(response);
  return response.json();
}

/** Cancels every non-terminal fairness job in this session, not just
 * whichever one the UI currently has a jobId for -- see the backend route's
 * docstring for why that distinction matters. */
export async function cancelAllFairnessAnalyses(signal?: AbortSignal): Promise<{ cancelled_job_ids: string[]; count: number }> {
  const response = await fetch(`${API_BASE}/api/v1/analyses/fairness/cancel-all`, {
    method: 'POST',
    credentials: 'include',
    signal,
  });
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export const DESIGN_LABEL: Record<FairnessDesign, string> = {
  matched: 'Matched (content controlled)',
  partially_matched: 'Partially matched',
  unmatched: 'Unmatched (content differs across groups)',
};

export const VERDICT_LABEL: Record<FairnessVerdictKind, string> = {
  disparity_detected: 'Disparity detected',
  inconclusive: 'Inconclusive',
  no_evidence_of_disparity: 'No evidence of disparity',
};

/** Lower-is-better metrics need their sign flipped before rendering on a
 * shared "higher = worse for this group" axis (radar, ranking). */
export const LOWER_IS_BETTER: ReadonlySet<FairnessMetricId> = new Set([
  'wer', 'cer', 'ece', 'attribution_entropy',
]);

export function metricLabel(metric: string): string {
  const labels: Record<string, string> = {
    wer: 'WER', cer: 'CER', accuracy: 'Accuracy', macro_f1: 'Macro-F1', ece: 'ECE',
    silhouette: 'Silhouette', leakage: 'Leakage', grounding_lift: 'Grounding lift',
    attribution_entropy: 'Attribution entropy',
  };
  return labels[metric] ?? metric;
}

export function groupMetricValue(group: FairnessGroupReport, metric: FairnessMetricId): number | null {
  switch (metric) {
    case 'wer': return group.performance.macro_wer ?? null;
    case 'cer': return group.performance.macro_cer ?? null;
    case 'accuracy': return group.performance.accuracy ?? null;
    case 'macro_f1': return group.performance.macro_f1 ?? null;
    case 'ece': return group.performance.ece ?? null;
    case 'grounding_lift': return group.grounding.status === 'ok' ? group.grounding.grounding_lift ?? null : null;
    case 'attribution_entropy': return group.grounding.status === 'ok' ? group.grounding.attribution_entropy ?? null : null;
    default: return null;
  }
}
