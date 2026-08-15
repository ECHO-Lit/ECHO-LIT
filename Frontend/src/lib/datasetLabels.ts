/**
 * Answer-key management for custom datasets.
 *
 * A layer probe is graded against known labels. Uploaded audio arrives with
 * none, so until a dataset has an answer key the Layer Probes tab has nothing
 * to offer. These endpoints supply one from either an uploaded CSV or the
 * filenames themselves.
 *
 * Every call returns the same shape, including a `preview` of what the probe
 * will do with the labels. That preview is the point: extraction is the
 * expensive half of a probe run, so a user should learn "two of your three
 * classes are about to be dropped" here rather than after a multi-minute job.
 */

import { API_BASE } from './api';

export interface PropertyPreview {
  property: string;
  n_samples: number;
  n_missing: number;
  n_classes: number;
  class_counts: Record<string, number>;
  dropped_classes: Array<{ label: string; count: number }>;
  /** Largest class's share. Accuracy at or below this means no information. */
  majority_baseline: number | null;
  cv_folds_used: number;
  probeable: boolean;
  skipped_reason: string | null;
}

export interface DatasetPreview {
  n_files: number;
  properties: PropertyPreview[];
  probeable_count: number;
}

export interface DatasetLabels {
  dataset_name: string;
  /** 'csv', `pattern:<id>`, or null when only derived bands are present. */
  source: string | null;
  columns: string[];
  warnings: string[];
  updated_at: string | null;
  preview: DatasetPreview;
  matched_files?: number;
  stored?: string[];
}

export interface LabelPattern {
  pattern_id: string;
  label: string;
  description: string;
  example: string;
  properties: string[];
}

async function readOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${response.status})`);
  }
  return response.json();
}

export async function fetchLabelPatterns(signal?: AbortSignal): Promise<LabelPattern[]> {
  const response = await fetch(`${API_BASE}/upload/dataset/label-patterns`, {
    credentials: 'include',
    signal,
  });
  const body = await readOrThrow<{ patterns: LabelPattern[] }>(response);
  return body.patterns;
}

export async function fetchDatasetLabels(
  datasetName: string,
  signal?: AbortSignal,
): Promise<DatasetLabels> {
  const response = await fetch(
    `${API_BASE}/upload/dataset/${encodeURIComponent(datasetName)}/labels`,
    { credentials: 'include', signal },
  );
  return readOrThrow<DatasetLabels>(response);
}

export async function uploadLabelCsv(
  datasetName: string,
  file: File,
): Promise<DatasetLabels> {
  const form = new FormData();
  form.append('file', file);
  const response = await fetch(
    `${API_BASE}/upload/dataset/${encodeURIComponent(datasetName)}/labels`,
    { method: 'POST', credentials: 'include', body: form },
  );
  return readOrThrow<DatasetLabels>(response);
}

export async function deriveLabelsFromFilenames(
  datasetName: string,
  patternId: string,
): Promise<DatasetLabels> {
  const form = new FormData();
  form.append('pattern_id', patternId);
  const response = await fetch(
    `${API_BASE}/upload/dataset/${encodeURIComponent(datasetName)}/labels/derive`,
    { method: 'POST', credentials: 'include', body: form },
  );
  return readOrThrow<DatasetLabels>(response);
}

export async function clearDatasetLabels(datasetName: string): Promise<DatasetLabels> {
  const response = await fetch(
    `${API_BASE}/upload/dataset/${encodeURIComponent(datasetName)}/labels`,
    { method: 'DELETE', credentials: 'include' },
  );
  return readOrThrow<DatasetLabels>(response);
}
