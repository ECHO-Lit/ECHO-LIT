import { API_BASE } from './api';

export type CustomModelStatus = 'validating' | 'ready' | 'failed';

export interface CustomModel {
  model_id: string;
  hf_repo: string;
  revision?: string | null;
  status: CustomModelStatus;
  kind?: 'seq2seq_asr' | 'ctc_asr' | 'audio_classification' | null;
  capabilities: string[];
  processor_type?: string | null;
  error?: string | null;
  created_at: string;
}

export interface JacobianLens {
  lens_id: string;
  model_id: string;
  model_revision: string;
  architecture?: 'seq2seq' | 'ctc';
  status: 'fitting' | 'ready' | 'failed';
  format_version?: number | null;
  method?: string | null;
  layer_count?: number;
  error?: string;
  sample_count: number;
}

export async function listJacobianLenses(modelId: string): Promise<JacobianLens[]> {
  const response = await fetch(`${API_BASE}/models/jacobian-lenses/${encodeURIComponent(modelId)}`, {
    credentials: 'include',
  });
  if (!response.ok) throw new Error(`Could not load Jacobian lenses (${response.status})`);
  return response.json();
}

async function errorFor(response: Response): Promise<Error> {
  try {
    const body = await response.json();
    return new Error(body.detail || `Request failed (${response.status})`);
  } catch {
    return new Error(`Request failed (${response.status})`);
  }
}

export async function listCustomModels(): Promise<CustomModel[]> {
  const response = await fetch(`${API_BASE}/models`, { credentials: 'include' });
  if (!response.ok) throw await errorFor(response);
  return response.json();
}

export async function registerCustomModel(hfRepo: string, revision?: string): Promise<{ model_id: string }> {
  const response = await fetch(`${API_BASE}/models`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ hf_repo: hfRepo, ...(revision?.trim() ? { revision: revision.trim() } : {}) }),
  });
  if (!response.ok) throw await errorFor(response);
  return response.json();
}

export async function deleteCustomModel(modelId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/models/${modelId}`, {
    method: 'DELETE',
    credentials: 'include',
  });
  if (!response.ok && response.status !== 204) throw await errorFor(response);
}
