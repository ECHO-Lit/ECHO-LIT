import { API_BASE } from './api';

export type SweepProperty = 'pitch' | 'speed' | 'noise' | 'time_mask' | 'freq_mask';

export interface SweepConfig {
  property: SweepProperty;
  start: number;
  stop: number;
  steps: number;
  repeats?: number;
  band_low_hz?: number;
}

export interface CurvePoint {
  theta: number;
  degradation: number;
  degradation_std?: number;
  ci95?: [number, number];
  raw: Record<string, number | string | boolean | null>;
  variant_audio_id?: string;
  playback_url?: string;
  measured_snr_db?: number | null;
}

export interface PropertyProfile {
  property: SweepProperty;
  unit: string;
  applicable: boolean;
  reason?: string;
  sensitivity_index?: number;
  local_slope_at_identity?: number;
  breakdown_theta?: number | null;
  asymmetry?: number | null;
  monotonic?: boolean;
  curve?: CurvePoint[];
}

export type SensitivityVerdict =
  | 'linguistically_driven'
  | 'mixed'
  | 'acoustically_dominated'
  | 'inconclusive';

export interface SensitivityProfile {
  verdict: SensitivityVerdict;
  reason?: string;
  acoustic_influence: number;
  linguistic_robustness: number;
  relative_to_lexical_destruction: number;
  dominant_property: string | null;
  ranking: Array<{ property: string; sensitivity_index: number; breakdown_theta: number | null }>;
  evidence: string[];
}

export interface LinguisticAcousticResult {
  job_id: string;
  operation: string;
  model: string;
  task: 'transcription' | 'classification';
  baseline: {
    audio_id: string | null;
    transcript: string | null;
    reference_source: string;
    stable: boolean;
  };
  profile: SensitivityProfile;
  properties: PropertyProfile[];
  controls: Record<string, { degradation: number; theta?: number; transcript?: string }>;
  set_level: Record<string, unknown>;
  metadata: {
    variants_total: number;
    variants_cached: number;
    cache_hit_rate: number;
    execution_seconds: number;
    variant_playback_available: boolean;
    not_applicable: Array<{ property: string; theta: unknown; reason?: string }>;
    [key: string]: unknown;
  };
}

export interface LinguisticAcousticAccepted {
  job_id: string;
  status: string;
  status_url: string;
  result_url: string;
  estimated_variants: number;
  estimated_seconds: number;
  poll_after_ms: number;
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

export async function submitLinguisticAcoustic(
  body: {
    audio_ids: string[];
    model: string;
    task?: 'transcription' | 'classification' | 'auto';
    sweeps: SweepConfig[];
    reference_transcript?: string | null;
    language?: string | null;
    include_lexical_control?: boolean;
  },
  signal?: AbortSignal,
): Promise<LinguisticAcousticAccepted> {
  const response = await fetch(`${API_BASE}/api/v1/analyses/linguistic-vs-acoustic`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw await parseError(response);
  return response.json();
}

const STOCHASTIC_PROPERTIES = new Set<SweepProperty>(['noise']);
const MAX_GRID_VARIANTS = 60;

export function estimateVariants(
  sweeps: SweepConfig[], audioCount: number, includeLexicalControl = true,
): number {
  let perAudio = 1 + (includeLexicalControl ? 1 : 0);
  for (const sweep of sweeps) {
    const repeats = STOCHASTIC_PROPERTIES.has(sweep.property) ? (sweep.repeats ?? 1) : 1;
    perAudio += sweep.steps * repeats;
  }
  return perAudio * Math.max(audioCount, 1);
}

export function isGridOverLimit(
  sweeps: SweepConfig[], audioCount: number, includeLexicalControl = true,
): boolean {
  return estimateVariants(sweeps, audioCount, includeLexicalControl) > MAX_GRID_VARIANTS;
}

export { MAX_GRID_VARIANTS };
