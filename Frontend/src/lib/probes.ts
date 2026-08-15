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
  /** Metadata column supplying this property's label, resolved for this dataset. */
  column: string;
  /** What a HIGH value means. Users read "low speaker accuracy" as "bad model". */
  meaning: string;
  /** True when discovered from an uploaded dataset rather than curated here. */
  discovered?: boolean;
}

interface CuratedProperty {
  key: string;
  label: string;
  /** Column names this property may appear under, in preference order. */
  columns: string[];
  meaning: string;
}

/**
 * Curated probe targets, with the interpretation text that is most of the value.
 *
 * Each lists several accepted column names because the same property is spelled
 * differently per corpus -- RAVDESS calls the speaker `actor`, SAVEE calls it
 * `speaker`. A dataset offers whichever of these its metadata has; anything else
 * is picked up generically by `discoverProperties`, so an uploaded dataset with
 * unfamiliar column names still works.
 */
export const CURATED_PROPERTIES: CuratedProperty[] = [
  {
    key: 'gender',
    label: 'Gender',
    columns: ['gender', 'sex'],
    meaning:
      'High = the speaker\'s gender is linearly decodable at this depth. Largely an acoustic property (pitch, formants), so it is usually readable from the very first layer.',
  },
  {
    key: 'speaker',
    label: 'Speaker identity',
    columns: ['actor', 'speaker', 'speaker_id'],
    meaning:
      'High = the model still distinguishes *who* is talking. A fall toward the top of the stack is the expected, desirable result for an ASR encoder: speaker identity is being discarded in favour of what was said.',
  },
  {
    key: 'emotion',
    label: 'Emotion',
    columns: ['emotion', 'emotion_label'],
    meaning:
      'High = the emotional category is decodable. Expected to peak mid-stack, above the acoustic layers but below the most linguistic ones.',
  },
  {
    key: 'lexical',
    label: 'Lexical content',
    columns: ['statement', 'sentence', 'utterance'],
    meaning:
      'High = *which sentence* was spoken is decodable. When every speaker reads the same fixed sentences, speaker and emotion are controlled and this genuinely measures linguistic content. Expected to rise with depth.',
  },
  {
    key: 'intensity',
    label: 'Vocal intensity',
    columns: ['intensity'],
    meaning:
      'High = normal vs strong delivery is decodable. An energy property, so it is expected to be readable early and not to grow with depth.',
  },
  {
    key: 'accent',
    label: 'Accent',
    columns: ['accent', 'dialect'],
    meaning:
      'High = the speaker\'s accent is decodable. Only a minority of Common Voice clips are annotated, so read this alongside the sample count.',
  },
  {
    key: 'age',
    label: 'Age band',
    columns: ['age', 'age_band'],
    meaning:
      'High = the speaker\'s age band is decodable. Sparsely annotated in Common Voice; a selectivity at or below zero means no information was found.',
  },
  {
    key: 'duration_band',
    label: 'Clip length',
    columns: ['duration_band'],
    meaning:
      'Derived automatically from clip duration, so it needs no annotation. Read it as a sanity check rather than a finding: length is an input property, so flat or falling is expected and a RISING curve should be treated as suspicious.',
  },
];

/** Columns that describe the file rather than something worth probing. */
const RESERVED_COLUMNS = new Set([
  'filename',
  'original_filename',
  'size',
  'uploaded_at',
  'duration',
  'sample_rate',
  'text',
  'client_id',
  'locale',
  'up_votes',
  'down_votes',
  'modality',
  'vocal_channel',
  'repetition',
]);

/** More distinct values than this and the column is an identifier, not a class. */
const MAX_DISCOVERED_CLASSES = 50;

const DISCOVERED_MEANING =
  'Discovered in this dataset\'s labels. High = this property is linearly decodable at that depth. Read it against the majority baseline and the selectivity, not on accuracy alone.';

function humanise(column: string): string {
  const words = column.replace(/[_-]+/g, ' ').trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

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

/** Distinct usable labels in a column, capped so an identifier column exits early. */
function distinctLabels(rows: MetadataRow[], column: string, cap: number): Set<string> {
  const values = new Set<string>();
  for (const row of rows) {
    const label = normaliseLabel(row[column]);
    if (label) values.add(label);
    if (values.size > cap) break;
  }
  return values;
}

/**
 * Columns present in the data that no curated property already claims.
 *
 * This is what makes an uploaded dataset work without a code change: a CSV with
 * a `speaking_style` column becomes a probeable property, keeping only the
 * generic interpretation text since we cannot know what it means.
 */
function discoverProperties(rows: MetadataRow[], claimed: Set<string>): ProbeProperty[] {
  const seen = new Set<string>();
  const found: ProbeProperty[] = [];
  for (const row of rows) {
    for (const column of Object.keys(row)) {
      if (seen.has(column) || claimed.has(column) || RESERVED_COLUMNS.has(column)) continue;
      seen.add(column);
      const values = distinctLabels(rows, column, MAX_DISCOVERED_CLASSES);
      // Two classes minimum to be separable; above the cap it is an id column.
      if (values.size < 2 || values.size > MAX_DISCOVERED_CLASSES) continue;
      found.push({
        key: column,
        label: humanise(column),
        column,
        meaning: DISCOVERED_MEANING,
        discovered: true,
      });
    }
  }
  return found;
}

/**
 * Properties this dataset can actually supply: column present, >=2 distinct labels.
 *
 * Curated properties come first and keep their interpretation text; anything
 * else the dataset carries follows, discovered generically.
 */
export function availableProperties(rows: MetadataRow[]): ProbeProperty[] {
  const claimed = new Set<string>();
  const curated: ProbeProperty[] = [];
  for (const property of CURATED_PROPERTIES) {
    for (const column of property.columns) {
      claimed.add(column);
    }
    // First column that actually separates wins, so a dataset carrying both
    // `actor` and `speaker` resolves to one property rather than two.
    const column = property.columns.find(
      (candidate) => distinctLabels(rows, candidate, MAX_DISCOVERED_CLASSES).size >= 2,
    );
    if (column) curated.push({ key: property.key, label: property.label, column, meaning: property.meaning });
  }
  return [...curated, ...discoverProperties(rows, claimed)];
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
  // Resolved from the rows rather than a static registry, so a discovered
  // property and a curated one whose column varies by corpus both work.
  const chosen = availableProperties(rows).filter((property) => selected.includes(property.key));
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

/**
 * Human-readable label for a property key, including the synthetic noise probe.
 *
 * Discovered properties are not in the curated list, and their key *is* their
 * column name, so humanising the key is the correct fallback rather than a
 * degraded one.
 */
export function propertyLabel(key: string): string {
  if (key === NOISE_PROPERTY) return 'Noise (clean vs noisy)';
  return CURATED_PROPERTIES.find((property) => property.key === key)?.label ?? humanise(key);
}

export function propertyMeaning(key: string): string {
  if (key === NOISE_PROPERTY) return NOISE_PROPERTY_MEANING;
  return (
    CURATED_PROPERTIES.find((property) => property.key === key)?.meaning ?? DISCOVERED_MEANING
  );
}

/** Properties ordered shallow -> deep by where they peak. The emergence story. */
export function byPeakDepth(result: LayerProbeResult): Array<[string, PropertyProbe]> {
  return Object.entries(result.properties)
    .filter(([, probe]) => probe.best_layer !== null)
    .sort((a, b) => (a[1].peak_depth ?? 0) - (b[1].peak_depth ?? 0));
}
