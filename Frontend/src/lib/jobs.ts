import { API_BASE, AudioReference } from './api';

export type JobOperation =
  | 'prediction'
  | 'saliency'
  | 'attention'
  | 'embedding'
  | 'perturbation'
  | 'audio_features';

export type JobState = 'queued' | 'started' | 'processing' | 'success' | 'failure' | 'cancelled';

export interface JobProgress {
  current: number;
  total: number;
  message: string;
}

export interface JobStatus {
  job_id: string;
  operation: JobOperation;
  model?: string;
  status: JobState;
  progress: JobProgress;
  result_url?: string;
  cache_hit: boolean;
  error?: { code: string; message: string; retryable: boolean };
}

export interface CreateJobInput {
  operation: JobOperation;
  audio_ids: string[];
  model?: string;
  parameters?: Record<string, unknown>;
}

async function parseError(response: Response): Promise<Error> {
  try {
    const body = await response.json();
    return new Error(body.detail || `Request failed (${response.status})`);
  } catch {
    return new Error(`Request failed (${response.status})`);
  }
}

export async function materializeAudio(
  dataset: string,
  filename: string,
  signal?: AbortSignal,
): Promise<AudioReference> {
  const response = await fetch(`${API_BASE}/audio/materialize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ dataset, filename }),
    signal,
  });
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export async function resolveAudioId(
  file: { audio_id?: string; file_id?: string; filename?: string } | string,
  dataset?: string,
  signal?: AbortSignal,
): Promise<string> {
  if (typeof file !== 'string') {
    if (file.audio_id) return file.audio_id;
    // New uploads use the same opaque value for file_id during migration.
    if (file.file_id && /^[a-f0-9]{32}$/i.test(file.file_id)) return file.file_id;
  }
  const filename = typeof file === 'string' ? file : file.filename;
  if (!dataset || !filename) throw new Error('An audio ID or dataset file is required');
  return (await materializeAudio(dataset, filename, signal)).audio_id;
}

export async function createJob(input: CreateJobInput, signal?: AbortSignal): Promise<JobStatus> {
  const response = await fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ ...input, parameters: input.parameters || {} }),
    signal,
  });
  if (!response.ok) throw await parseError(response);
  const created = await response.json();
  return getJob(created.job_id, signal);
}

export async function getJob(jobId: string, signal?: AbortSignal): Promise<JobStatus> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`, { credentials: 'include', signal });
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export async function getJobResult<T = unknown>(jobId: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/result`, {
    credentials: 'include',
    signal,
  });
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export async function cancelJob(jobId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`, {
    method: 'DELETE',
    credentials: 'include',
  });
  if (!response.ok && response.status !== 204) throw await parseError(response);
}

function wait(milliseconds: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(resolve, milliseconds);
    signal?.addEventListener('abort', () => {
      window.clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    }, { once: true });
  });
}

export async function runJob<T = unknown>(
  input: CreateJobInput,
  options: { signal?: AbortSignal; onProgress?: (status: JobStatus) => void } = {},
): Promise<T> {
  let status = await createJob(input, options.signal);
  let delay = 1000;
  while (!['success', 'failure', 'cancelled'].includes(status.status)) {
    options.onProgress?.(status);
    await wait(delay, options.signal);
    status = await getJob(status.job_id, options.signal);
    delay = Math.min(Math.round(delay * 1.5), 5000);
  }
  options.onProgress?.(status);
  if (status.status === 'failure') throw new Error(status.error?.message || 'Analysis failed');
  if (status.status === 'cancelled') throw new Error('Analysis was cancelled');
  return getJobResult<T>(status.job_id, options.signal);
}

export function firstJobResult<T = unknown>(result: any): T {
  if (!result?.items?.length) throw new Error('Job returned no result items');
  return result.items[0].result as T;
}
