/**
 * Minimal hand-built payloads for §3.1.3.
 *
 * Kept minimal on the RUP template's own guidance: a small fixture makes an
 * unacceptable value visible. Every shape here is taken from the client's own
 * declared contract (`lib/api.ts`, `lib/jobs.ts`, `lib/models.ts`) or from the
 * `response.json()` consumer in the component, NOT from the backend — this
 * suite asserts what the client does with a given response, and 3.1.2 already
 * covers whether the backend produces it.
 */
import type { AudioReference } from "@/lib/api";
import type { JobState, JobStatus } from "@/lib/jobs";
import type { CustomModel } from "@/lib/models";
import type { GroupableColumnsResponse } from "@/lib/fairness";

export function audioReference(overrides: Partial<AudioReference> = {}): AudioReference {
  return {
    audio_id: "a1b2c3d4e5f60718293a4b5c6d7e8f90",
    file_id: "a1b2c3d4e5f60718293a4b5c6d7e8f90",
    filename: "sample.wav",
    playback_url: "/audio/a1b2c3d4e5f60718293a4b5c6d7e8f90",
    media_type: "audio/wav",
    size_bytes: 32044,
    duration_seconds: 2.0,
    sample_rate: 16000,
    channels: 1,
    ...overrides,
  };
}

export function jobStatus(
  status: JobState,
  overrides: Partial<JobStatus> = {},
): JobStatus {
  return {
    job_id: "job-0001",
    operation: "prediction",
    model: "whisper-base",
    status,
    progress:
      status === "success"
        ? { current: 1, total: 1, message: "Complete" }
        : { current: 0, total: 1, message: "Queued" },
    cache_hit: false,
    ...(status === "success" ? { result_url: "/jobs/job-0001/result" } : {}),
    ...(status === "failure"
      ? { error: { code: "worker_error", message: "Inference failed", retryable: true } }
      : {}),
    ...overrides,
  };
}

/** `GET /models` returns a BARE ARRAY, not an envelope. */
export function customModel(overrides: Partial<CustomModel> = {}): CustomModel {
  return {
    model_id: "cm-0001",
    hf_repo: "openai/whisper-tiny",
    revision: null,
    status: "ready",
    kind: "seq2seq_asr",
    capabilities: ["prediction", "embedding", "saliency", "attention"],
    processor_type: "WhisperProcessor",
    error: null,
    created_at: "2026-09-08T00:00:00Z",
    ...overrides,
  };
}

/** `GET /upload/dataset/list` returns `{ datasets: [...] }`. */
export function customDataset(overrides: Record<string, unknown> = {}) {
  return {
    dataset_name: "custom:sess:my-set",
    formatted_name: "my-set",
    total_files: 3,
    ...overrides,
  };
}

/** `GET /{dataset}/metadata` rows, as AudioDataTable consumes them. */
export const datasetMetadata = [
  { filename: "clip-001.wav", text: "the quick brown fox", duration: 2.1, accent: "us" },
  { filename: "clip-002.wav", text: "jumps over the lazy dog", duration: 1.8, accent: "england" },
  { filename: "clip-003.wav", text: "pack my box", duration: 2.4, accent: "us" },
];

/** `GET /api/v1/analyses/fairness/groupable` — FairnessPanel mounts eagerly. */
export const groupableColumns: GroupableColumnsResponse = {
  dataset: "common-voice",
  n_rows: 3,
  speaker_column: "client_id",
  content_column: "sentence",
  columns: [
    {
      column: "accent",
      n_values: 2,
      n_missing: 0,
      counts: { us: 2, england: 1 },
      is_speaker_column: false,
      is_stratum_only: false,
    },
  ],
};

/** The two lists the Toolbar fetches on mount. */
export const baseRoutes = {
  "GET /upload/dataset/list": { json: { datasets: [] } },
  "GET /models": { json: [] as CustomModel[] },
};

/**
 * The EXACT four requests the dashboard issues on mount, measured. There is no
 * catch-all: a wildcard route returning `{}` silently satisfies callers that
 * expect an array, which is how a component crash gets mistaken for a passing
 * test. An undeclared request must throw and name itself.
 */
export const dashboardRoutes = {
  ...baseRoutes,
  "GET /api/v1/analyses/fairness/groupable": { json: groupableColumns },
  "GET /common-voice/metadata": { json: datasetMetadata },
};
