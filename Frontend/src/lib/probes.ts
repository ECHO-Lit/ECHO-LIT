/**
 * Layer-wise probing: backend contract, the metadata -> label mapping, and the
 * job wrapper.
 *
 * The mapping from a dataset's metadata columns to probe properties lives HERE
 * rather than in the panel, so the components stay presentational and a new
 * dataset is one registry entry.
 *
 * Reading the numbers this module carries:
 *  - `accuracy` alone means nothing. It must be read against `majority_baseline`
 *    (the largest class's share) -- 0.9 accuracy on a 90/10 split is zero
 *    information.
 *  - `selectivity` = accuracy - control_accuracy, where the control probe saw
 *    shuffled labels. It is the honest headline: a probe that beats the majority
 *    baseline but not its own control is memorising rows, not reading the model.
 */

import { API_BASE } from './api';
import { runJob, type JobStatus } from './jobs';

export interface LayerMetric {
  layer: number;
  layer_name: string;
  accuracy: number | null;
  accuracy_std: number | null;
  macro_f1: number | null;
  control_accuracy: number | null;
  selectivity: number | null;
}

export interface DroppedClass {
  label: string;
  count: number;
}

export interface PropertyProbe {
  n_samples: number;
  n_missing: number;
  n_classes: number;
  class_counts: Record<string, number>;
  dropped_classes: DroppedClass[];
  majority_baseline: number | null;
  cv_folds_used: number;
  layers: LayerMetric[];
  best_layer: number | null;
  best_accuracy: number | null;
  best_selectivity: number | null;
  /** best_layer / (num_layers - 1) — comparable across models of different depth. */
  peak_depth: number | null;
  confusion_matrix: number[][];
  class_labels: string[];
  skipped_reason: string | null;
}

export interface LayerProbeResult {
  num_layers: number;
  layer_names: string[];
  hidden_dim: number;
  projected_dim: number;
  n_files: number;
  properties: Record<string, PropertyProbe>;
  params: {
    probe: string;
    cv_folds: number;
    project_dims: number;
    min_class_count: number;
    include_control: boolean;
    seed: number;
  };
  noise_snr_db?: number;
}

export interface LayerProbeResponse {
  job_id: string;
  operation: string;
  model: string;
  probes: LayerProbeResult;
  items: Array<{ audio_id: string; layers: number; dim: number; cache_hit?: boolean }>;
  metadata: Record<string, unknown>;
}

export interface ProbeProperty {
  key: string;
  label: string;
  /** Metadata column supplying this property's label. */
  column: string;
  /** What a HIGH value means. Users read "low speaker accuracy" as "bad model". */
  meaning: string;
}

/**
 * Known probe targets. A dataset offers whichever of these its metadata has;
 * `availableProperties` decides, so custom datasets work without a code change
 * whenever their columns happen to match.
 */
export const PROBE_PROPERTIES: ProbeProperty[] = [
  {
    key: 'gender',
    label: 'Gender',
    column: 'gender',
    meaning:
      'High = the speaker\'s gender is linearly decodable at this depth. Largely an acoustic property (pitch, formants), so it is usually readable from the very first layer.',
  },
  {
    key: 'speaker',
    label: 'Speaker identity',
    column: 'actor',
    meaning:
      'High = the model still distinguishes *who* is talking. A fall toward the top of the stack is the expected, desirable result for an ASR encoder: speaker identity is being discarded in favour of what was said.',
  },
  {
    key: 'emotion',
    label: 'Emotion',
    column: 'emotion',
    meaning:
      'High = the emotional category is decodable. Expected to peak mid-stack, above the acoustic layers but below the most linguistic ones.',
  },
  {
    key: 'lexical',
    label: 'Lexical content',
    column: 'statement',
    meaning:
      'High = *which sentence* was spoken is decodable. Both RAVDESS sentences are spoken by every actor in every emotion, so speaker and emotion are controlled and this genuinely measures linguistic content. Expected to rise with depth.',
  },
  {
    key: 'intensity',
    label: 'Vocal intensity',
    column: 'intensity',
    meaning:
      'High = normal vs strong delivery is decodable. An energy property, so it is expected to be readable early and not to grow with depth.',
  },
  {
    key: 'accent',
    label: 'Accent',
    column: 'accent',
    meaning:
      'High = the speaker\'s accent is decodable. Only a minority of Common Voice clips are annotated, so read this alongside the sample count.',
  },
  {
    key: 'age',
    label: 'Age band',
    column: 'age',
    meaning:
      'High = the speaker\'s age band is decodable. Sparsely annotated in Common Voice; a selectivity at or below zero means no information was found.',
  },
];

export const NOISE_PROPERTY = 'noise';

export const NOISE_PROPERTY_MEANING =
  'High = the encoder represents *that the signal is degraded*. Trained on each clip twice, clean and with white noise added at the chosen SNR.';

/** Values that mean "not annotated" rather than naming a class. */
const MISSING_LABELS = new Set(['', 'unknown', 'none', 'n/a', 'na']);

export type MetadataRow = Record<string, string>;

function normaliseLabel(value: string | undefined): string | null {
  const text = (value ?? '').trim();
  return MISSING_LABELS.has(text.toLowerCase()) ? null : text;
}

export async function fetchDatasetMetadata(
  dataset: string,
  signal?: AbortSignal,
): Promise<MetadataRow[]> {
  const response = await fetch(`${API_BASE}/${dataset}/metadata`, {
    credentials: 'include',
    signal,
  });
  if (!response.ok) throw new Error(`Failed to load dataset metadata (${response.status})`);
  return response.json();
}

/**
 * Index rows by filename, keyed on both the CSV value and its basename.
 *
 * Common Voice stores `cv-valid-dev/sample-000775.mp3` while the file list uses
 * the bare name; matching on only one of the two silently produces zero labels.
 */
function indexByFilename(rows: MetadataRow[]): Map<string, MetadataRow> {
  const index = new Map<string, MetadataRow>();
  for (const row of rows) {
    const filename = row.filename ?? '';
    if (!filename) continue;
    index.set(filename, row);
    const base = filename.split(/[\\/]/).pop();
    if (base && !index.has(base)) index.set(base, row);
  }
  return index;
}

/** Properties this dataset can actually supply: column present, >=2 distinct labels. */
export function availableProperties(rows: MetadataRow[]): ProbeProperty[] {
  return PROBE_PROPERTIES.filter((property) => {
    const values = new Set<string>();
    for (const row of rows) {
      const label = normaliseLabel(row[property.column]);
      if (label) values.add(label);
      if (values.size >= 2) return true;
    }
    return false;
  });
}

/**
 * Build the label payload, positionally aligned with `filenames`.
 *
 * The backend joins labels to files by position, so this alignment is the whole
 * correctness of the feature — the returned arrays are always exactly as long as
 * `filenames`, with `null` where a file has no annotation.
 */
export function extractProperties(
  rows: MetadataRow[],
  filenames: string[],
  selected: string[],
): Record<string, Array<string | null>> {
  const index = indexByFilename(rows);
  const chosen = PROBE_PROPERTIES.filter((property) => selected.includes(property.key));
  const payload: Record<string, Array<string | null>> = {};
  for (const property of chosen) {
    payload[property.key] = filenames.map((filename) => {
      const row = index.get(filename) ?? index.get(filename.split(/[\\/]/).pop() ?? '');
      return row ? normaliseLabel(row[property.column]) : null;
    });
  }
  return payload;
}

/** How many files carry a usable label for each selected property. */
export function labelledCounts(
  properties: Record<string, Array<string | null>>,
): Record<string, number> {
  return Object.fromEntries(
    Object.entries(properties).map(([key, labels]) => [
      key,
      labels.filter((label) => label !== null).length,
    ]),
  );
}

export interface LayerProbeOptions {
  model: string;
  audioIds: string[];
  properties: Record<string, Array<string | null>>;
  probe?: 'logreg' | 'linear_svm';
  cvFolds?: number;
  projectDims?: number;
  minClassCount?: number;
  includeControl?: boolean;
  noiseSnrDb?: number | null;
  seed?: number;
}

export async function runLayerProbe(
  options: LayerProbeOptions,
  hooks: { signal?: AbortSignal; onProgress?: (status: JobStatus) => void } = {},
): Promise<LayerProbeResponse> {
  return runJob<LayerProbeResponse>(
    {
      operation: 'layer_probe',
      model: options.model,
      audio_ids: options.audioIds,
      parameters: {
        properties: options.properties,
        probe: options.probe ?? 'logreg',
        cv_folds: options.cvFolds ?? 5,
        project_dims: options.projectDims ?? 256,
        min_class_count: options.minClassCount ?? 5,
        include_control: options.includeControl ?? true,
        ...(options.noiseSnrDb === null || options.noiseSnrDb === undefined
          ? {}
          : { noise_snr_db: options.noiseSnrDb }),
        seed: options.seed ?? 42,
      },
    },
    hooks,
  );
}

/** Human-readable label for a property key, including the synthetic noise probe. */
export function propertyLabel(key: string): string {
  if (key === NOISE_PROPERTY) return 'Noise (clean vs noisy)';
  return PROBE_PROPERTIES.find((property) => property.key === key)?.label ?? key;
}

export function propertyMeaning(key: string): string {
  if (key === NOISE_PROPERTY) return NOISE_PROPERTY_MEANING;
  return PROBE_PROPERTIES.find((property) => property.key === key)?.meaning ?? '';
}

/** Properties ordered shallow -> deep by where they peak. The emergence story. */
export function byPeakDepth(result: LayerProbeResult): Array<[string, PropertyProbe]> {
  return Object.entries(result.properties)
    .filter(([, probe]) => probe.best_layer !== null)
    .sort((a, b) => (a[1].peak_depth ?? 0) - (b[1].peak_depth ?? 0));
}
