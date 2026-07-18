# ECHO Runtime Architecture

ECHO uses FastAPI as a CPU-only control plane and Celery workers as the model
execution plane. The frontend communicates only with FastAPI.

## Request flow

1. `POST /upload` validates audio, stores it through the object-storage adapter,
   registers a session-owned opaque `audio_id`, and returns without inference.
2. `POST /jobs` validates an operation and its audio ownership, stores a transient
   job record, and publishes a small task envelope containing storage keys.
3. A worker downloads audio to a per-task temporary directory, lazily loads the
   required model, executes the operation, and stores the structured result.
4. The frontend polls `GET /jobs/{job_id}` and retrieves completed output from
   `GET /jobs/{job_id}/result`.

Job and audio metadata expire after 24 hours. Local object cleanup is performed
by Celery Beat; production S3 buckets must apply a matching lifecycle rule.

## Runtime boundaries

- `api`: FastAPI, session validation, upload validation, job authorization, and
  result proxying. It has no PyTorch or Transformers dependency.
- `worker-mock`: model-free local worker consuming the `mock` queue and writing
  deterministic contract-compatible results through the real storage and job
  repositories.
- `worker-cpu`: CPU-only perturbation, feature, projection, and maintenance work.
- Native MPS worker: single macOS process consuming `cpu`, `gpu-fast`, and
  `gpu-large` while executing real models on Apple Metal. It cannot run inside
  Docker Desktop.
- `worker-model-local`: opt-in development model worker consuming GPU queues on CPU.
- `worker-gpu` / `worker-amd`: optional CUDA/ROCm workers consuming `gpu-fast`
  and `gpu-large`, with concurrency one per GPU.
- `scheduler`: Celery Beat for transient-object maintenance.
- `redis`: separate logical databases for sessions/cache, job metadata, broker,
  and Celery results, configured with persistence and `noeviction`.
- `ObjectStorage`: shared filesystem locally or an S3-compatible bucket in
  production. Clients and queue messages never receive storage keys.

Apply `Backend/s3-lifecycle.json` to the production bucket so transient
uploads, cache objects, generated audio, and result artifacts expire after
24 hours.

## Operations and routing

| Queue | Operations |
| --- | --- |
| `mock` | All operations when `EXECUTION_PROFILE=mock` |
| `gpu-fast` | Whisper Base, Wav2Vec2, and embeddings |
| `gpu-large` | Whisper Large, attention, and saliency |
| `cpu` | Perturbations, audio features, projections, and cleanup |

The worker-owned model registry keys entries by model, purpose, revision, and
device. Standard, attention, saliency, and embedding variants can therefore be
loaded and evicted independently.

The API writes `mock`, `mps`, `cpu`, or `cloud-gpu` into every task envelope.
Workers reject mismatched profiles, and the profile is part of the cache key so
fixture results cannot be returned by real model execution.

## Production configuration

Set `STORAGE_BACKEND=s3` and configure the `S3_*` settings for both API and
workers. Use private authenticated Redis endpoints for `JOB_REDIS_URL`,
`CELERY_BROKER_URL`, and `CELERY_RESULT_BACKEND`; an evicting cache may use a
separate `REDIS_URL`. Keep `ENABLE_LEGACY_SYNC_INFERENCE=false`.
