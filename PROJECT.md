# ECHO — Full Project Reference

> **Purpose of this file:** Single-source, exhaustive reference for the ECHO codebase. Written for both human developers and AI coding assistants. Every major directory, service, model, endpoint, config knob, and data flow is documented below with clickable file paths.
>
> Read [README.md](README.md) for the user-facing pitch. Read this file when you need to *work in* the code.

---

## 1. What ECHO Is

**ECHO** (Explainable Computation for Hearing Outputs) is a **Learning Interpretability Tool for audio models**. It is the audio-domain counterpart to Google's LIT (which targets text/tabular models).

Given a speech model (Whisper for ASR, Wav2Vec2 for emotion classification), ECHO lets a researcher:

- Upload / browse audio datasets.
- Run inference and inspect predictions + confidence.
- Visualize **attention** patterns across transformer layers/heads.
- Project **embeddings** to 2D/3D via PCA / t-SNE / UMAP and explore clusters.
- Compute **saliency** maps (GradCAM, LIME, SHAP, IntegratedGradients, LRP) to see which audio regions drove a prediction.
- Apply **perturbations** (Gaussian noise, time/frequency masking, pitch shift, time-stretch) and observe model robustness.

The project was renamed from *"LIT for Voice"* to *ECHO* in commit `306ff85`. FastAPI title string still reads `"LIT for Voice – API"` — this is intentional and safe to leave, or rename in [Backend/app/main.py](Backend/app/main.py).

**Authors:** Anas Hussaindeen, Chandupa Ambepitiya, Dewmike Amarasinghe. Mentor: Dr. Uthayasanker Thayasivam (University of Moratuwa).

---

## 2. Repository Layout

```
d:\Projects\ECHO\
├── Frontend/              # React 18 + TS + Vite UI
├── Backend/               # FastAPI + Python 3.11 API
├── README.md              # User-facing overview
├── CHANGELOG.md           # v1.0.0 release notes
├── CONTRIBUTING.md        # Dev setup + PR guidelines
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── LICENSE                # MIT
└── PROJECT.md             # (this file)
```

Two top-level apps: [Frontend/](Frontend/) and [Backend/](Backend/). No monorepo tooling — each has its own dependency manager (`npm` / `pip`). Redis is the only external service, run via [Backend/docker-compose.yml](Backend/docker-compose.yml).

---

## 3. Tech Stack (Summary)

| Layer | Stack |
|-------|-------|
| **Frontend framework** | React 18.3.1, TypeScript 5.5.3, Vite 5.4.1 (SWC) |
| **UI** | Tailwind 3.4.11, shadcn/ui (Radix), 40+ base components |
| **Frontend state** | TanStack Query 5.56, React Context, TanStack Table |
| **Audio (UI)** | wavesurfer.js 7.10, react-dropzone, Web Audio API |
| **Data viz** | Plotly.js 3.0, recharts, react-plotly.js |
| **Forms/validation** | React Hook Form 7.53, Zod |
| **Routing** | React Router DOM 6.26 |
| **Backend framework** | FastAPI 0.111, Uvicorn 0.30 |
| **Python** | 3.11+ |
| **ML** | torch ≥ 2.6, torchaudio, transformers ≥ 4.30 |
| **Audio (server)** | librosa ≥ 0.10, soundfile ≥ 0.12 |
| **Explainability** | captum ≥ 0.6, lime ≥ 0.2, shap ≥ 0.42 |
| **Dim reduction** | scikit-learn ≥ 1.3 (PCA, t-SNE), umap-learn ≥ 0.5 |
| **Cache / session** | Redis 7-alpine (async `redis.asyncio`, `fakeredis` for tests) |
| **Config** | pydantic-settings 2.3, python-dotenv |
| **Testing** | pytest 8.2, pytest-asyncio 0.23 |

Full lists: [Frontend/package.json](Frontend/package.json), [Backend/requirements.txt](Backend/requirements.txt).

---

## 4. Frontend ([Frontend/](Frontend/))

### 4.1 Entry + Layout

- **App entry:** [Frontend/src/pages/Index.tsx](Frontend/src/pages/Index.tsx) — renders `<MainLayout />`.
- **Main container:** [Frontend/src/components/layout/MainLayout.tsx](Frontend/src/components/layout/MainLayout.tsx) (~23 KB). Owns all top-level state:
  - Selected model (`whisper-base` | `whisper-large` | `wav2vec2`).
  - Selected dataset (`common-voice` | `ravdess` | custom-dataset-id).
  - Uploaded files, predictions map, embeddings map, perturbation results.
  - `AbortController` refs for cancelable in-flight requests.
  - Prediction cache (avoid re-running inference on the same file).
  - Resizable panel layout via `react-resizable-panels`.
- **Toolbar:** [Frontend/src/components/layout/Toolbar.tsx](Frontend/src/components/layout/Toolbar.tsx) — model + dataset pickers, batch inference trigger.

### 4.2 Panels ([Frontend/src/components/panels/](Frontend/src/components/panels/))

| Panel | File | Role |
|-------|------|------|
| Audio dataset | [AudioDatasetPanel.tsx](Frontend/src/components/panels/AudioDatasetPanel.tsx) | Browse dataset files, filter, select |
| Datapoint editor | [DatapointEditorPanel.tsx](Frontend/src/components/panels/DatapointEditorPanel.tsx) | Edit / annotate audio metadata |
| Embedding | [EmbeddingPanel.tsx](Frontend/src/components/panels/EmbeddingPanel.tsx) | Trigger embedding extraction + reduction, view plot |
| Prediction | [PredictionPanel.tsx](Frontend/src/components/panels/PredictionPanel.tsx) | Show Whisper transcript / accuracy or Wav2Vec2 emotion |

### 4.3 Visualization ([Frontend/src/components/visualization/](Frontend/src/components/visualization/))

- [AttentionVisualization.tsx](Frontend/src/components/visualization/AttentionVisualization.tsx) — Whisper attention (layer/head selector, timestamp-aligned attention curves, attention pairs).
- [EmbeddingPlot.tsx](Frontend/src/components/visualization/EmbeddingPlot.tsx) — 2D/3D Plotly scatter. Supports box/lasso selection, plane picking (`xy`/`xz`/`yz`) for 3D, and angle-range filtering.
- [SaliencyVisualization.tsx](Frontend/src/components/visualization/SaliencyVisualization.tsx) — segment-level saliency overlays.
- [ScalersVisualization.tsx](Frontend/src/components/visualization/ScalersVisualization.tsx) — scalar metric plotting over time. Per recent commit `0863652`, scalers now feed off the audio-embedding pipeline.
- [ScalarPlot.tsx](Frontend/src/components/visualization/ScalarPlot.tsx) — individual scalar metric plot.

### 4.4 Audio, Analysis, Dataset, Predictions

- Audio: [AudioDataTable.tsx](Frontend/src/components/audio/AudioDataTable.tsx), [AudioPlayer.tsx](Frontend/src/components/audio/AudioPlayer.tsx), [AudioUploader.tsx](Frontend/src/components/audio/AudioUploader.tsx), [WaveformViewer.tsx](Frontend/src/components/audio/WaveformViewer.tsx) (wavesurfer.js).
- Analysis: [PerturbationTools.tsx](Frontend/src/components/analysis/PerturbationTools.tsx) — UI for noise / time-mask / freq-mask / pitch-shift / time-stretch.
- Dataset mgmt: [CustomDatasetManager.tsx](Frontend/src/components/dataset/CustomDatasetManager.tsx) — create + upload custom datasets.
- Predictions display: [PredictionDisplay.tsx](Frontend/src/components/predictions/PredictionDisplay.tsx).

### 4.5 Contexts, Hooks, Utilities

- Context: [EmbeddingContext.tsx](Frontend/src/contexts/EmbeddingContext.tsx) — embedding state + fetch logic shared across panels.
- Hooks: [use-mobile.tsx](Frontend/src/hooks/use-mobile.tsx), [use-toast.ts](Frontend/src/hooks/use-toast.ts) (sonner).
- Lib: [Frontend/src/lib/api.ts](Frontend/src/lib/api.ts) exposes `API_BASE` (default `http://localhost:8000`, overridable via `VITE_API_BASE_URL`). Also [audioFeatures.ts](Frontend/src/lib/audioFeatures.ts), [utils.ts](Frontend/src/lib/utils.ts).

### 4.6 Build + Config

- [Frontend/vite.config.ts](Frontend/vite.config.ts) — dev server on port **8080**, path alias `@` → `./src`, `lovable-tagger` component tagger in dev only.
- Scripts: `npm run dev` | `build` | `build:dev` | `lint` | `preview`.
- TypeScript config is loose ([tsconfig.json](Frontend/tsconfig.json)) — no strict null checks. Do not tighten without checking blast radius across components.

---

## 5. Backend ([Backend/](Backend/))

### 5.1 Entry

- [Backend/app/main.py](Backend/app/main.py) — FastAPI app.
  - Middlewares: `SessionMiddleware` (cookie session), CORS (`ALLOWED_ORIGINS` env, defaults to `localhost:3000`, `localhost:8080`, `127.0.0.1:8080`).
  - Registered routers: `/session`, `/results`, `/inferences`, `/upload`, `/datasets`, `/saliency`, `/perturbations`, `/debug`, `/health`.

### 5.2 API Routes ([Backend/app/api/routes/](Backend/app/api/routes/))

| File | Notable endpoints |
|------|-------------------|
| [inferences.py](Backend/app/api/routes/inferences.py) | `POST /inferences/run`, `POST /inferences/batch-check`, `POST /inferences/embeddings`, `POST /inferences/attention-pairs`, `POST /inferences/whisper-accuracy`, `POST /inferences/wav2vec2-detailed` |
| [saliency.py](Backend/app/api/routes/saliency.py) | `POST /saliency/generate` — GradCAM / LIME / SHAP / IntegratedGradients. Cache schema `v2`. |
| [perturbations.py](Backend/app/api/routes/perturbations.py) | Audio perturbation endpoints (noise, mask, pitch, stretch) |
| [datasets.py](Backend/app/api/routes/datasets.py) | `GET /{dataset}/metadata`, `GET/HEAD/OPTIONS /{dataset}/file/{path:path}` (Range-request streaming) |
| [dataset_management.py](Backend/app/api/routes/dataset_management.py) | `POST /upload/dataset/create`, `POST /upload/dataset/{name}/files`, `GET /upload/dataset` — session-scoped custom datasets. Accepts `.wav .mp3 .m4a .flac` |
| [upload.py](Backend/app/api/routes/upload.py) | `POST /upload` (auto-predict), `DELETE /upload/{file_id}` |
| [session.py](Backend/app/api/routes/session.py) | Session lifecycle |
| [results.py](Backend/app/api/routes/results.py) | Cached result retrieval |
| [health.py](Backend/app/api/routes/health.py) | Liveness |
| [debug.py](Backend/app/api/routes/debug.py) | Dev-only introspection |

### 5.3 Services ([Backend/app/services/](Backend/app/services/))

- **[model_loader_service.py](Backend/app/services/model_loader_service.py)** (~62 KB) — heart of the ML layer.
  - Whisper: `transcribe_whisper_base()`, `transcribe_whisper_large()` — supports `return_attention=True`, timestamps, multiple attention-extraction paths. **Critical:** transformers must load Whisper with `attn_implementation="eager"` for `output_attentions=True` to work. See line ~36.
  - Wav2Vec2: `wave2vec()`, `predict_emotion_wave2vec()` — emotion labels: neutral / happy / sad / angry / fear. Attention extraction supported.
  - Embeddings: `extract_whisper_embeddings()` → 768-D, `extract_wav2vec2_embeddings()` → 256-D.
  - Reduction: `reduce_dimensions(embeddings, method, n_components)` → PCA / t-SNE / UMAP.
  - Audio features: `extract_audio_frequency_features()`.

- **[saliency_service.py](Backend/app/services/saliency_service.py)** (~23 KB) — attribution.
  - `generate_saliency()`, `generate_whisper_saliency()`.
  - Methods: GradCAM, LIME, SHAP, IntegratedGradients, LRP.
  - Length caps: normal ≤ 12 s (`MAX_SALIENCY_SECONDS`), SHAP ≤ 6 s (`MAX_SALIENCY_SECONDS_SHAP`). Enforced to prevent OOM.
  - Returns segment-level scores.

- **[pertubation_service.py](Backend/app/services/pertubation_service.py)** *(sic — filename typo, matches import sites)*.
  - `add_gaussian_noise`, `apply_time_masking`, `apply_frequency_masking` (FFT-based), `apply_pitch_shift` (librosa, ±6 semitones, ≤ 30 s audio).

- **[dataset_service.py](Backend/app/services/dataset_service.py)** — built-in datasets (`common-voice`, `cv-valid-dev`, `ravdess`), custom-dataset resolution, CSV metadata loading, librosa-based duration calc.

- **[custom_dataset_service.py](Backend/app/services/custom_dataset_service.py)** — CRUD for per-session custom datasets. Files stored at `uploads/sessions/{session_id}/{dataset_name}/`, metadata as JSON.

- **[queue_service.py](Backend/app/services/queue_service.py)** — task-queue stub.

### 5.4 Core ([Backend/app/core/](Backend/app/core/))

- **[redis.py](Backend/app/core/redis.py)** — async Redis client, key scheme:
  - `sess:{sid}` — session state.
  - `result:{model}:{hash}` — cached inference results.
  - `saliency_v2_{model}_{method}_{file_hash}` — saliency cache with schema version.
  - Helpers: `ensure_session(sid)`, `get_queue`, `put_queue`, `cache_result`, `get_result`. Default TTL 24 h.
- **[session.py](Backend/app/core/session.py)** — cookie session middleware. Cookie: `sid`, `HttpOnly`, `SameSite=lax`. Auto-issues cookie in response.
- **[settings.py](Backend/app/core/settings.py)** — pydantic settings. Env-driven config (see §7).

---

## 6. ML / AI Reference

### 6.1 Models

| Model | Source | Task | Output |
|-------|--------|------|--------|
| Whisper (base, large) | OpenAI via `transformers` | ASR | Transcript, per-token confidence, decoder + cross attentions, timestamps |
| Wav2Vec2 | Facebook via `transformers` | Emotion classification + embeddings | Class probs (neutral/happy/sad/angry/fear), 256-D embedding, attentions |

### 6.2 Explainability Toolbox

- **Saliency:** GradCAM, IntegratedGradients (captum), LIME (perturb-based), SHAP (KernelSHAP), LRP.
- **Attention:** decoder self-attentions + cross-attentions for Whisper; encoder attentions for Wav2Vec2.
- **Embedding viz:** PCA (linear, fast), t-SNE (local structure), UMAP (fast nonlinear).
- **Perturbation:** noise injection, time/frequency masking, pitch shifting, time-stretching — used to probe robustness.

### 6.3 Datasets

- **common-voice** — Mozilla Common Voice validation subset. Metadata CSV: `common_voice_valid_data_metadata.csv`.
- **ravdess** — RAVDESS emotion subset. Metadata CSV: `ravdess_subset_metadata.csv`.
- **Custom** — session-scoped, user-uploaded. Stored under `uploads/sessions/{session_id}/{dataset_name}/`.

Audio files are **not** committed. The `data/` layout expected by the dataset service:

```
Backend/
├── data/
│   ├── common_voice_valid_dev/     # audio + metadata CSV
│   └── ravdess_subset/             # audio + metadata CSV
└── uploads/
    ├── {one-off single-file uploads}
    └── sessions/{sid}/{dataset_name}/
```

---

## 7. Configuration & Environment

### 7.1 Backend env vars (see [settings.py](Backend/app/core/settings.py))

| Var | Default | Notes |
|-----|---------|-------|
| `REDIS_URL` | `redis://localhost:6379/0` | async Redis connection |
| `SESSION_COOKIE_NAME` | `sid` | session cookie name |
| `SESSION_TTL_SECONDS` | `86400` (24 h) | session + Redis key TTL |
| `COOKIE_SECURE` | `False` | flip to `True` behind HTTPS |
| `COOKIE_SAMESITE` | `lax` | |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:8080,http://127.0.0.1:8080` | CORS origins, comma-separated |
| `MAX_SALIENCY_SECONDS` | `12` | normal saliency length cap |
| `MAX_SALIENCY_SECONDS_SHAP` | `6` | SHAP tighter cap (OOM guard) |
| `SALIENCY_SHAP_SAMPLES` | `8` | SHAP samples per call |

`.env` files are gitignored — do not commit secrets.

### 7.2 Frontend env vars

- `VITE_API_BASE_URL` — backend URL (default `http://localhost:8000`). Used in [Frontend/src/lib/api.ts](Frontend/src/lib/api.ts).
- Dev server binds `::1:8080` (see [vite.config.ts](Frontend/vite.config.ts)).

---

## 8. Data & Caching Architecture

**Redis is the only stateful service** — there is no SQL database.

- **Result cache** keyed by `md5(file_path + size + mtime)` × model × method. Cache invalidates automatically when the audio file changes.
- **Session state** is cookie-driven (`sid`), and every session has a Redis namespace.
- Redis config in [docker-compose.yml](Backend/docker-compose.yml): 256 MB max, `allkeys-lru` eviction, `redis-cli ping` healthcheck.
- **Long-term artifacts** (audio, custom datasets, metadata CSVs) live on disk. No object store.

Cache-key patterns to grep for when debugging:

```
sess:{sid}
result:{model}:{hash}
saliency_v2_{model}_{method}_{file_hash}
```

---

## 9. Build, Run, Deploy

### 9.1 Local dev — full stack

**Redis** (required first):
```
cd Backend
docker compose up -d
```

**Backend:**
```
cd Backend
python -m venv .venv
.\.venv\Scripts\activate            # Windows PowerShell
# source .venv/bin/activate         # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload       # http://localhost:8000
```

**Frontend:**
```
cd Frontend
npm install
npm run dev                         # http://localhost:8080
```

Backend README also documents a Conda + CUDA 12.1 path for GPU inference.

### 9.2 CI/CD

No CI workflows are checked in (`.github/workflows/` absent). Docker Compose covers Redis only — no container for backend or frontend yet.

---

## 10. Testing

### 10.1 Backend

Framework: `pytest` + `pytest-asyncio`. Fixtures in [Backend/tests/conftest.py](Backend/tests/conftest.py):

- `fake_redis` — `fakeredis` instance (no live Redis in tests).
- `sample_audio_data`, `sample_audio_file`, `mock_model_outputs`.
- `AsyncClient` for endpoint hits.
- Performance thresholds for load tests.

Test files under [Backend/tests/](Backend/tests/):

- `test_data_integrity.py` — audio processing correctness.
- `test_function_testing.py` — unit tests over services.
- `test_performance_load.py` — batch inference load.
- `test_queue.py` — queue helpers.
- `test_results_cache.py` — cache hit / miss / TTL.
- `test_security.py` — auth / CORS / cookies.
- `test_session_cookie.py` — session middleware.

### 10.2 Frontend

Minimal — `tests/ui-components.test.tsx` under [Frontend/](Frontend/). No configured test runner in `package.json` scripts; expand if adding coverage.

Run backend tests:
```
cd Backend
pytest
```

---

## 11. Key Architectural Patterns / Gotchas

- **Eager attention required.** Whisper must be instantiated with `attn_implementation="eager"` — the SDPA / FlashAttention paths do not return attentions. See top of [model_loader_service.py](Backend/app/services/model_loader_service.py).
- **Saliency length caps** exist for a reason (OOM on longer audio). Do not raise `MAX_SALIENCY_SECONDS_SHAP` above 6 without profiling.
- **Session isolation.** Custom datasets are scoped by `sid` cookie. Cross-session leakage would be a security bug — see [test_security.py](Backend/tests/test_security.py).
- **AbortController pattern.** [MainLayout.tsx](Frontend/src/components/layout/MainLayout.tsx) holds per-request abort refs to cancel stale inference calls when the user changes model or dataset mid-flight. Reuse this pattern when adding new long-running endpoints.
- **Cache schema version.** Saliency uses `saliency_v2_*` — bump the version when the response shape changes to invalidate cleanly instead of writing migration code.
- **Range-request audio streaming.** [datasets.py](Backend/app/api/routes/datasets.py) supports `Range` headers so `<audio>` seek works without downloading whole files. Preserve this when adding new file-serving endpoints.
- **Filename typo:** `pertubation_service.py` (missing `r`). Import sites match. Rename would require touching many files — leave unless doing a dedicated cleanup pass.
- **Legacy title:** [main.py](Backend/app/main.py) still says `"LIT for Voice – API"`. Rename to ECHO if a user-visible change is wanted; otherwise no functional impact.

---

## 12. Roadmap (from [README.md](README.md))

- More audio model architectures (beyond Whisper / Wav2Vec2).
- Advanced perturbation techniques.
- Real-time streaming audio processing.
- Export functionality (results, visualizations).
- Multi-language support.
- Plugin system for third-party interpretability methods.

---

## 13. Where To Look When…

| Task | Start here |
|------|-----------|
| Add a new model | [model_loader_service.py](Backend/app/services/model_loader_service.py) — add loader + inference fn. Wire into [inferences.py](Backend/app/api/routes/inferences.py). Add option in [Toolbar.tsx](Frontend/src/components/layout/Toolbar.tsx). |
| Add a new saliency method | [saliency_service.py](Backend/app/services/saliency_service.py). Bump cache schema version. Extend [SaliencyVisualization.tsx](Frontend/src/components/visualization/SaliencyVisualization.tsx) if the response shape changes. |
| Add a new perturbation | [pertubation_service.py](Backend/app/services/pertubation_service.py) + [perturbations.py](Backend/app/api/routes/perturbations.py) + [PerturbationTools.tsx](Frontend/src/components/analysis/PerturbationTools.tsx). |
| Change embedding reducer | `reduce_dimensions()` in [model_loader_service.py](Backend/app/services/model_loader_service.py) + selector in [EmbeddingPanel.tsx](Frontend/src/components/panels/EmbeddingPanel.tsx). |
| Add a built-in dataset | Drop CSV + audio into `Backend/data/`, register in [dataset_service.py](Backend/app/services/dataset_service.py), expose in [Toolbar.tsx](Frontend/src/components/layout/Toolbar.tsx). |
| Debug a caching miss | Check key format in [redis.py](Backend/app/core/redis.py); confirm file hash inputs (path + size + mtime); look for schema version mismatch in saliency keys. |
| Fix CORS | `ALLOWED_ORIGINS` env var, then [main.py](Backend/app/main.py) middleware setup. |
| Tune Whisper attention output | Confirm `attn_implementation="eager"` at load time in [model_loader_service.py](Backend/app/services/model_loader_service.py). |

---

## 14. Recent Development Focus (git log context)

Head is on `main` (commit `306ff85`). Recent themes:

1. **Renaming** — project renamed to ECHO (`306ff85`).
2. **Toolbar / tabs UI polish** (`03f7b93`, `ae8e44c`).
3. **Scalers now driven by audio embeddings** (`0863652`) — non-trivial data-flow change touching [ScalersVisualization.tsx](Frontend/src/components/visualization/ScalersVisualization.tsx) and embedding pipeline.
4. **Attention maps for custom datasets** (`6ac14b2`, PR #48) — attention viz no longer restricted to built-in datasets.

Run `git log --oneline -30` for the current tail.

---

## 15. License

MIT — see [LICENSE](LICENSE).
