import { API_BASE } from '@/lib/api';

/** Task families ECHO supports for user-supplied models. */
export type CustomModelTask = 'speech-seq2seq' | 'ctc' | 'audio-classification';

export type CheckStatus = 'pass' | 'fail' | 'warn' | 'skip';

export interface ConstraintCheck {
  id: string;
  constraint: string;
  status: CheckStatus;
  detail: string;
}

export interface CustomModelSpec {
  name: string;
  formatted_name: string;
  model_id: string;
  revision: string | null;
  task: CustomModelTask;
  task_label: string;
  auto_class: string;
  architecture: string;
  model_type: string;
  input_name: string;
  sampling_rate: number;
  num_parameters: number;
  num_labels: number | null;
  id2label: Record<string, string> | null;
  has_tokenizer: boolean;
  capabilities: string[];
  capability_labels: string[];
  registered_at: string;
  probe_seconds: number;
  warnings: string[];
}

export interface ValidationResult {
  model_id: string;
  compatible: boolean;
  deep: boolean;
  error?: string;
  checks: ConstraintCheck[];
  task?: CustomModelTask;
  task_label?: string;
  auto_class?: string;
  architecture?: string;
  input_name?: string;
  sampling_rate?: number;
  num_labels?: number | null;
  capabilities?: string[];
  capability_labels?: string[];
  num_parameters?: number;
  probe_seconds?: number;
}

/** Result envelope returned by /inferences/run for a custom model. */
export interface CustomModelPredictionResult {
  model_name: string;
  model_id: string;
  task: CustomModelTask;
  task_label: string;
  capabilities: string[];
  inference_seconds: number;
  limit_exceeded?: string;
  /** ASR tasks (seq2seq + CTC). */
  text?: string;
  /** Classification. */
  predicted_label?: string;
  probabilities?: Record<string, number>;
  confidence?: number;
  /** CTC only. */
  frame_probabilities?: {
    frames: Array<{ time: number; token: string; probability: number }>;
    total_frames: number;
    duration: number;
  };
}

export const isCustomModel = (model?: string | null): boolean =>
  !!model && model.startsWith('custom:');

/** `custom:<session_id>:<name>` -> `<name>`, for display. */
export const customModelDisplayName = (model: string): string => {
  const parts = model.split(':');
  return parts.length === 3 ? parts[2] : model;
};

const json = async (response: Response) => {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body?.detail || `Request failed: ${response.status}`);
  }
  return body;
};

export const fetchCustomModels = async (): Promise<CustomModelSpec[]> => {
  const res = await fetch(`${API_BASE}/custom-models/list`, { credentials: 'include' });
  const data = await json(res);
  return data.models || [];
};

export const validateCustomModel = async (
  modelId: string,
  deep: boolean,
  revision?: string,
): Promise<ValidationResult> => {
  const res = await fetch(`${API_BASE}/custom-models/validate`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId, deep, revision: revision || undefined }),
  });
  return json(res);
};

export const registerCustomModel = async (
  name: string,
  modelId: string,
  revision?: string,
): Promise<{ model: string; spec: CustomModelSpec }> => {
  const res = await fetch(`${API_BASE}/custom-models/register`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, model_id: modelId, revision: revision || undefined }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    // 422 carries the per-constraint report; surface it alongside the message
    // so the dialog can show exactly which constraint rejected the model.
    const err = new Error(body?.detail || `Registration failed: ${res.status}`) as Error & {
      report?: { checks: ConstraintCheck[] };
    };
    err.report = body?.report;
    throw err;
  }
  return body;
};

export const deleteCustomModel = async (name: string): Promise<void> => {
  const res = await fetch(`${API_BASE}/custom-models/${encodeURIComponent(name)}`, {
    method: 'DELETE',
    credentials: 'include',
  });
  await json(res);
};

export const fetchCustomModelCapabilities = async () => {
  const res = await fetch(`${API_BASE}/custom-models/capabilities`, { credentials: 'include' });
  return json(res);
};
