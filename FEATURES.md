# ECHO — Feature & Pipeline Walkthrough

> **Companion to [PROJECT.md](PROJECT.md).** That file maps *where* code lives. This file explains *what each feature does, what the user sees, why it matters, and the exact call chain from click to response.*
>
> Written for two audiences:
> - **Humans** learning the tool for the first time or onboarding to a feature.
> - **AI coding assistants** answering "how does saliency work here?" or "what happens when I press Run?".
>
> Structure per feature: **Concept → What UI shows → User actions → Pipeline (click → API → service → cache → response) → Extension ideas.**

---

## Table of Contents

1. [Concepts primer (attention, saliency, embeddings)](#0-concepts-primer)
2. [Audio Upload](#1-audio-upload)
3. [Dataset Browsing & File Serving](#2-dataset-browsing--file-serving)
4. [Custom Datasets](#3-custom-datasets)
5. [Model Inference (Whisper ASR)](#4-model-inference--whisper-asr)
6. [Model Inference (Wav2Vec2 Emotion)](#5-model-inference--wav2vec2-emotion)
7. [Attention Visualization](#6-attention-visualization)
8. [Embedding Extraction & Projection](#7-embedding-extraction--projection)
9. [Saliency Maps](#8-saliency-maps)
10. [Perturbation Tools](#9-perturbation-tools)
11. [Scalers / Batch Metrics](#10-scalers--batch-metrics)
12. [Caching & Sessions](#11-caching--sessions)
13. [End-to-end user walkthrough](#12-end-to-end-user-walkthrough)

---

## 0. Concepts Primer

### What is **attention**?

Transformer models decide which parts of an input matter for each part of an output. Attention is a matrix of weights: "when producing token *i*, how much did I look at input frame *j*?" Weight near 1 = strong focus, near 0 = ignored.

- **Self-attention:** tokens attending to tokens in the same sequence (encoder side, or decoder → decoder).
- **Cross-attention:** decoder tokens attending to encoder outputs. In Whisper, this is how each transcribed word aligns to a slice of audio.
- **Layers × heads:** every transformer layer contains multiple attention "heads" (parallel views). Different heads specialize (phoneme boundaries, prosody, silence, etc.). ECHO lets you pick layer + head.

**What ECHO shows for attention:** the word-to-audio-time alignment implied by Whisper's cross-attention — which audio interval the model "listened to" when producing each transcript word.

**What else you could do:** head-role analysis (which head fires on stop consonants?), attention-rollout across layers, attention-entropy (peaked vs. diffuse focus), attention-diff between clean vs. perturbed audio.

### What is **saliency**?

Attention says *where the model looked*. Saliency says *how much each input actually mattered to the final prediction*. They often disagree — the model may look at silence but base its decision on a vowel elsewhere.

Methods (all supported in ECHO):

| Method | How it works | Cost | Property |
|--------|--------------|------|----------|
| **GradCAM** | Gradient × activation of a chosen layer | Low | Fast, coarse, layer-specific |
| **IntegratedGradients** | Average gradient along a straight line from baseline to input | Medium | Axiomatic, respects completeness |
| **LIME** | Perturb input, fit a linear surrogate locally | Medium | Model-agnostic, noisy |
| **SHAP** | Shapley values via sampling | High | Theoretically grounded, slow |
| **LRP** | Layer-wise relevance propagation | Medium | Signed relevance flow |

**What ECHO shows for saliency:** segment-level (per short time window) importance scores overlaid on the waveform. Bright regions = pushed prediction toward the chosen class.

**What else you could do:** class-conditional saliency (which regions push *toward* "angry" vs. *away*), saliency stability under noise, occlusion sensitivity, saliency-vs-attention divergence plots.

### What are **embeddings**?

Every audio clip becomes a high-dimensional vector inside the model (512-D for Whisper base, 768-D for Wav2Vec2). Similar audio → nearby vectors. Because 512-D is unplottable, ECHO projects to 2-D or 3-D with PCA / t-SNE / UMAP.

- **PCA:** linear, fast, preserves global variance. Distances roughly meaningful.
- **t-SNE:** nonlinear, preserves *local* neighborhoods. Clusters look clean but global distances are meaningless.
- **UMAP:** nonlinear, faster than t-SNE, preserves some global structure.

**What ECHO shows for embeddings:** scatter plot; clicking a point selects the file; box/lasso selects groups.

**What else you could do:** distance queries ("nearest 5 clips"), embedding drift under perturbation (does the point move?), cluster labeling, cross-model embedding comparison.

### What is a **perturbation**?

A controlled change to the input audio (noise, mask, pitch shift) to probe robustness. If the prediction flips after tiny noise, the model is fragile.

### What are **scalers**?

Batch-level aggregate metrics across many files: emotion distribution, word-frequency histograms, spectral feature summaries. Post commit `0863652`, scalers derive from the embedding pipeline output.

---

## 1. Audio Upload

**What it does:** Add a one-off audio file (outside built-in datasets) for inspection.

**UI ([AudioUploader.tsx](Frontend/src/components/audio/AudioUploader.tsx)):**
- Drag-and-drop zone or click-to-select.
- Accepts `.wav .mp3 .m4a .flac`.
- Toast on success/failure.

**User actions:**
- Drop file → auto-uploads, auto-runs inference, auto-extracts embedding.

**Pipeline:**

```
User drops file
   │
   ▼
AudioUploader.tsx  →  POST /upload  (multipart, {file, model})
   │                        │
   │                        ▼
   │              Backend/app/api/routes/upload.py::upload_audio_file
   │                        │
   │                        ├─ save to uploads/{uuid}.{ext}
   │                        ├─ librosa loads → duration, sample_rate
   │                        ├─ auto call run_inference()   → cached
   │                        └─ auto call extract_single_embedding()  → cached
   │
   ▼
Response: { filename, file_id, duration, sample_rate, size, prediction }
   │
   ▼
MainLayout state updated → PredictionPanel + EmbeddingPanel refresh
```

Backend service refs: [upload.py](Backend/app/api/routes/upload.py), [model_loader_service.py](Backend/app/services/model_loader_service.py).

**Extension ideas:** batch upload, format transcode, sample-rate normalization on ingest.

---

## 2. Dataset Browsing & File Serving

**What it does:** Browse built-in datasets (`common-voice`, `ravdess`) and stream audio via HTTP Range requests so the waveform can seek without downloading everything.

**UI:** [AudioDatasetPanel.tsx](Frontend/src/components/panels/AudioDatasetPanel.tsx) + [AudioDataTable.tsx](Frontend/src/components/audio/AudioDataTable.tsx).

**User actions:** sort, filter, select a row → becomes the active file for other panels.

**Pipeline (metadata):**

```
Panel mounts
   │
   ▼
GET /{dataset}/metadata
   │
   ▼
datasets.py::get_metadata()  →  reads CSV from Backend/data/{dataset}/
   │
   ▼
Rows returned → rendered in AudioDataTable
```

**Pipeline (audio playback with seeking):**

```
<audio> or wavesurfer.js issues:
   Range: bytes=1024-2047
   │
   ▼
GET /{dataset}/file/{path}   (datasets.py::serve_dataset_file)
   │
   ├─ resolve_file(dataset, path, session_id)  → validated absolute path
   ├─ if Range header:
   │      StreamingResponse 206, headers:
   │        Accept-Ranges: bytes
   │        Content-Range: bytes X-Y/total
   └─ else: FileResponse full file
```

Custom datasets are addressed with `dataset = "custom:{sid}:{name}"` and URL-encoded.

**Extension ideas:** paginated metadata, server-side filtering, waveform thumbnails cached to disk.

---

## 3. Custom Datasets

**What it does:** Users create per-session datasets and upload their own audio into them.

**UI:** [CustomDatasetManager.tsx](Frontend/src/components/dataset/CustomDatasetManager.tsx) — tabs for List / Create / Upload.

**Pipeline:**

```
Create:
   POST /dataset/create  { dataset_name }
      → dataset_management.py::create_custom_dataset
      → CustomDatasetManager(session_id).create_dataset()
      → mkdir uploads/sessions/{sid}/{name}/ + metadata.json

Upload:
   POST /dataset/{name}/files  (multipart, files[])
      → validates each file (extension, librosa-decodable)
      → saves as {uuid}.ext + records {filename, original, duration, sample_rate, size}
      → returns updated metadata

List:
   GET /upload/dataset
      → enumerates uploads/sessions/{sid}/*
```

**Session scoping:** dataset name stored/served as `custom:{session_id}:{dataset_name}` so it can never collide with another user's dataset.

**Extension ideas:** dataset export/import, per-file labels for supervised eval, dataset diff.

---

## 4. Model Inference — Whisper ASR

**Concept:** Whisper transcribes speech to text. ECHO exposes `whisper-base` (fast, small) and `whisper-large-v3` (accurate, slow).

**UI:** [PredictionPanel.tsx](Frontend/src/components/panels/PredictionPanel.tsx) — shows transcript. If the file has ground truth (e.g., `common-voice`), also shows accuracy metrics.

**Endpoints:**
- `POST /inferences/run` — generic entry point, returns `{predicted_transcript}`.
- `POST /inferences/whisper-accuracy` — comparison against ground truth, returns WER / CER / Levenshtein / exact-match / character-similarity.

**Pipeline:**

```
User selects file + model=whisper-*
   │
   ▼
PredictionPanel.tsx → POST /inferences/run  { model, file_path | dataset+dataset_file }
   │
   ▼
inferences.py::run_inference
   │
   ├─ MD5(file_path + size + mtime) → cache key
   ├─ Redis hit? → return cached, done
   ├─ else route via MODEL_FUNCTIONS[model]:
   │      transcribe_whisper_base()  or  transcribe_whisper_large()
   │      → model_loader_service.py::transcribe_whisper() (single impl)
   │        - loads WhisperForConditionalGeneration w/ attn_implementation="eager"
   │        - librosa load, resample to 16 kHz
   │        - processor(audio) → input_features
   │        - model.generate() → token IDs → processor.batch_decode()
   ├─ Cache result 6 h
   └─ Return {predicted_transcript, ...}
```

**Critical implementation note:** `attn_implementation="eager"` is required at model load. SDPA / FlashAttention paths silently discard attentions — later attention endpoints would return empty results.

Service ref: [model_loader_service.py](Backend/app/services/model_loader_service.py) (`transcribe_whisper` at ~line 24, `transcribe_whisper_base` at ~356).

**Extension ideas:** language forcing, chunk-level confidence, streaming transcription.

---

## 5. Model Inference — Wav2Vec2 Emotion

**Concept:** Wav2Vec2 fine-tuned for speech emotion recognition (model: `r-f/wav2vec-english-speech-emotion-recognition`). Labels: `neutral / happy / sad / angry / fear`.

**UI:** [PredictionPanel.tsx](Frontend/src/components/panels/PredictionPanel.tsx) — predicted emotion + per-class probability bars.

**Endpoint:** `POST /inferences/wav2vec2-detailed` with `{file_path | dataset+dataset_file, include_attention?}`.

**Pipeline:**

```
POST /inferences/wav2vec2-detailed
   │
   ▼
inferences.py → predict_emotion_wave2vec  (model_loader_service.py:392)
   │
   ├─ Wav2Vec2ForSequenceClassification, on CUDA if available
   ├─ processor(audio, sr=16k) → input_values
   ├─ model(**inputs) → logits
   ├─ softmax → probabilities dict
   └─ argmax → predicted_emotion, top-p confidence
   │
   ▼
Return { predicted_emotion, probabilities, confidence, attention? }
```

If `include_attention=True`, `predict_emotion_wave2vec_with_attention` sets `config.output_attentions=True` and returns attention layers alongside.

**Extension ideas:** confidence calibration, ordinal emotion axes (arousal/valence), multilingual emotion model.

---

## 6. Attention Visualization

**What ECHO shows:** For a chosen Whisper layer + head, the cross-attention weights aligning each transcript word to a time interval in the audio.

**UI:** [AttentionVisualization.tsx](Frontend/src/components/visualization/AttentionVisualization.tsx)
- Layer dropdown (default 6), head dropdown (default 0).
- Word-to-word attention matrix / heatmap.
- Timeline attention curve (weight over time).
- Word-time pairs list.

**Endpoint:** `POST /inferences/attention-pairs` — `{model, file_path|dataset+dataset_file, layer, head}`.

**Pipeline:**

```
User picks layer / head
   │
   ▼
AttentionVisualization useEffect → POST /inferences/attention-pairs
   │
   ▼
inferences.py::extract_attention_pairs_endpoint  (~line 1252)
   │
   ├─ transcribe_whisper_with_attention()      → transcript + attention[layers][heads]
   ├─ transcribe_whisper_with_timestamps()     → word chunks with [start,end] times
   ├─ process_attention_into_pairs()           → merge:
   │      for each (from_word, to_word):
   │          attention_weight = attention[layer][head][from_idx][to_idx]
   │          from_time, to_time from timestamps
   └─ Cache key: {model}_attention_pairs_{file_hash}_l{layer}_h{head}
   │
   ▼
Response: { attention_pairs[], timeline_attention[], transcript }
   │
   ▼
Frontend renders heatmap + timeline
```

**What you can do beyond current UI:** head-role tagging, attention rollout (multiply across layers), attention entropy per token, side-by-side attention for clean vs. perturbed inputs, attention-vs-saliency divergence view.

**Extension ideas:** Wav2Vec2 encoder-attention view (currently Whisper-only), attention export to CSV, click-a-word-to-jump-audio.

---

## 7. Embedding Extraction & Projection

**What ECHO shows:** 2-D or 3-D scatter of many audio clips. Each point = one file. Nearby points = model thinks they sound similar.

**UI:**
- [EmbeddingPanel.tsx](Frontend/src/components/panels/EmbeddingPanel.tsx) — model / reduction-method / component-count selectors, "Extract" trigger.
- [EmbeddingPlot.tsx](Frontend/src/components/visualization/EmbeddingPlot.tsx) — Plotly scatter with box/lasso selection, plane picker (`xy`/`xz`/`yz`) for 3-D, angle-range filter.
- Shared via [EmbeddingContext.tsx](Frontend/src/contexts/EmbeddingContext.tsx).

**Endpoint:** `POST /inferences/embeddings` — `{model, dataset, files[], reduction_method: pca|tsne|umap, n_components: 2|3}`.

**Pipeline:**

```
User clicks Extract
   │
   ▼
POST /inferences/embeddings
   │
   ▼
inferences.py::extract_embeddings_endpoint
   │
   ├─ For each file:
   │     cache lookup: {model}_embeddings_{hash}
   │     miss → extract_whisper_embeddings()  (encoder.last_hidden_state, mean-pool)
   │           or extract_wav2vec2_embeddings() (768-D)
   │     store 24 h
   ├─ Stack all embeddings → matrix [N × D]
   ├─ reduce_dimensions(matrix, method, n_components)
   │     - pca  → sklearn.decomposition.PCA
   │     - tsne → sklearn.manifold.TSNE (perplexity default)
   │     - umap → umap.UMAP
   └─ Return raw + reduced coords
   │
   ▼
EmbeddingContext stores → EmbeddingPlot renders
```

**Interactive UI actions:**
- Click point → set active file (triggers Prediction, Saliency, Attention refresh).
- Box / lasso select → subset for group inspection.
- Switch reduction method → re-run reduction (raw embeddings stay cached, only projection recomputed).
- 3-D plane picker → collapse to a chosen 2-D face for cleaner reading.

**Extension ideas:** k-NN lookup ("show me clips like this one"), embedding drift under perturbation (arrow from clean → perturbed point), cluster labels, cross-model embedding comparison, export to Vega/Bokeh.

---

## 8. Saliency Maps

**What ECHO shows:** For a chosen model + method, per-segment importance scores displayed as a colored overlay along the waveform + a series plot.

**UI:** [SaliencyVisualization.tsx](Frontend/src/components/visualization/SaliencyVisualization.tsx)
- Method picker: `gradcam | integrated_gradients | lime | shap`.
- Overlay on waveform, timeline series plot.
- Segment tooltips with `start_time`, `end_time`, `saliency`, and (when Whisper) `word`.

**Endpoint:** `POST /saliency/generate` — `{model, method, file_path|dataset+dataset_file, no_cache?}`.

**Pipeline:**

```
POST /saliency/generate
   │
   ▼
saliency.py::generate_saliency_endpoint
   │
   ├─ Resolve file (upload or dataset)
   ├─ Cache key: saliency_v2_{model}_{method}_{file_hash}
   │     (schema version = v2; bump if response shape changes)
   ├─ Length cap enforced:
   │     normal ≤ MAX_SALIENCY_SECONDS  (12 s)
   │     SHAP   ≤ MAX_SALIENCY_SECONDS_SHAP  (6 s)
   ├─ saliency_service.py::generate_saliency
   │     dispatches to method impl:
   │       gradcam              → captum LayerGradCam / grad × activation
   │       integrated_gradients → captum IntegratedGradients (baseline=zeros)
   │       lime                 → lime.lime_audio surrogate
   │       shap                 → shap.KernelExplainer  (SALIENCY_SHAP_SAMPLES=8)
   │       lrp                  → captum LRP
   └─ Response:
        { model, method, segments: [{start_time, end_time, saliency, word?}],
          total_duration, series: [...] }
```

**Length caps are load-bearing.** SHAP with N segments is O(2^N) worst case; raising `MAX_SALIENCY_SECONDS_SHAP` above 6 without profiling will OOM.

**Extension ideas:** class-conditional saliency (positive vs. negative contributions per emotion class), saliency stability sweep across noise levels, occlusion baseline for sanity checks, saliency-vs-attention divergence overlay.

---

## 9. Perturbation Tools

**What ECHO shows:** After applying one or more perturbations, the perturbed audio replaces the current audio in the player; inference auto-reruns so the user sees how the prediction shifted.

**UI:** [PerturbationTools.tsx](Frontend/src/components/analysis/PerturbationTools.tsx) — checkboxes + sliders per perturbation.

**Endpoint:** `POST /perturb` — `{file_path, dataset?, perturbations: [{type, params}]}`.

**Perturbation menu:**

| Type | Params | Effect |
|------|--------|--------|
| `noise` | `noise_level` (σ of Gaussian) | Add white noise |
| `timeMasking` | `mask_start_percent`, `mask_end_percent` | Zero a time slice |
| `frequencyMasking` | `mask_low_hz`, `mask_high_hz` | FFT-mask a frequency band |
| `pitchShift` | `pitch_shift_semitones` (±6 max) | librosa `pitch_shift` |
| `timeStretch` | `time_stretch_factor` | librosa `time_stretch` (no pitch change) |

**Pipeline:**

```
User configures perturbations + clicks Apply
   │
   ▼
POST /perturb
   │
   ▼
perturbations.py::apply_perturbations
   │
   ├─ librosa load audio
   ├─ Sequentially apply each perturbation via pertubation_service.py [sic]
   ├─ soundfile write to uploads/{uuid}.wav
   └─ Return { perturbed_file, duration_ms, sample_rate,
                applied_perturbations[], success }
   │
   ▼
MainLayout swaps active file → PredictionPanel auto re-runs inference
```

**Constraints:** pitch shift limited to ±6 semitones and ≤ 30 s clips inside the service — enforced to keep librosa fast and outputs sensible.

**Extension ideas:** perturbation chain scoring (auto-report Δemotion / ΔWER per step), reverberation / codec / background-mix perturbations, adversarial (gradient-directed) perturbations, batch perturbation over a whole dataset for robustness metrics.

---

## 10. Scalers / Batch Metrics

**Concept:** Aggregate signals across many files. Post commit `0863652`, scalers pull from the audio-embedding pipeline output rather than being a separate stream.

**UI:** [ScalersVisualization.tsx](Frontend/src/components/visualization/ScalersVisualization.tsx) + [ScalarPlot.tsx](Frontend/src/components/visualization/ScalarPlot.tsx).

**Endpoints:**
- `POST /inferences/wav2vec2-batch` — emotion distribution across N files.
- `POST /inferences/whisper-batch` — word-frequency stats.
- `POST /inferences/audio-frequency-batch` — spectral feature aggregates.

**Pipeline (Wav2Vec2 batch):**

```
POST /inferences/wav2vec2-batch  { filenames[], dataset }
   │
   ▼
For each file:
   cache lookup {wav2vec2}_{hash}
   miss → run predict_emotion_wave2vec (populates cache)
   │
Aggregate:
   emotion_distribution = counts / total
   dominant_emotion = argmax
   cache_hit_rate = cached / total
   │
   ▼
Response drives:
   - Pie chart of emotion distribution
   - Cache stats (dev insight)
```

**Audio frequency batch:** uses `extract_audio_frequency_features()` (librosa) — spectral centroid, spectral rolloff, MFCC, chroma, tonnetz, tempo, RMS, zero-crossing rate. Backend returns per-feature `mean / std / min / max / median` + histograms.

**Extension ideas:** time-windowed scalers (per second), scaler correlation matrix, export to CSV.

---

## 11. Caching & Sessions

Applies to every feature.

**Session:**
- Middleware in [Backend/app/core/session.py](Backend/app/core/session.py) issues `sid` cookie (`HttpOnly`, `SameSite=lax`) on first request.
- `sid` scopes custom-dataset paths and cache-key namespaces.

**Redis cache** ([Backend/app/core/redis.py](Backend/app/core/redis.py)):

| Kind | Key pattern | TTL |
|------|-------------|-----|
| Prediction | `{model}_{file_md5}` | 6 h |
| Embedding (raw) | `{model}_embeddings_{file_md5}` | 24 h |
| Saliency | `saliency_v2_{model}_{method}_{file_md5}` | 6 h |
| Attention pairs | `{model}_attention_pairs_{file_md5}_l{layer}_h{head}` | 6 h |
| Audio freq features | `audio_freq_{file_md5}` | 24 h |

**File hash inputs:** `md5(file_path + size + mtime)` — the mtime component means editing/replacing a file at the same path invalidates its cache automatically.

**Schema versioning:** saliency uses `_v2` in the key. When response shape changes, bump to `_v3` — cleaner than writing a migration.

---

## 12. End-to-End User Walkthrough

A typical exploration session:

1. **Pick model + dataset** in the toolbar ([Toolbar.tsx](Frontend/src/components/layout/Toolbar.tsx)). Say `whisper-base` + `common-voice`.
2. **Browse dataset** in the audio table. Click a row.
   - Backend serves metadata CSV + streams audio via Range requests.
3. **Play back** the waveform ([WaveformViewer.tsx](Frontend/src/components/audio/WaveformViewer.tsx) via wavesurfer.js).
4. **Prediction panel** auto-populates: `POST /inferences/run` → transcript + accuracy metrics.
5. **Attention** tab: pick layer 6 / head 0. `POST /inferences/attention-pairs` returns word-to-time alignment. Try other heads to see specialization.
6. **Embeddings**: click Extract for the whole dataset. Backend caches raw 768-D vectors, projects to 2-D via UMAP. Scatter plot renders. Lasso-select a cluster.
7. **Saliency**: pick `gradcam` for the currently selected file. Overlay shows which audio segments drove the transcript.
8. **Perturbation**: add Gaussian noise σ=0.005, apply. Perturbed file becomes active; prediction re-runs. Compare transcript / attention / saliency vs. clean.
9. **Custom dataset**: upload own recordings; every feature above works on them via `dataset = "custom:{sid}:{name}"`.

Everything is cache-warmed after the first pass — repeat interactions are Redis hits (see cache-hit-rate in scaler batch responses for dev insight).

---

## Cross-Reference

| Concept | Backend impl | Frontend view |
|---------|--------------|----------------|
| Attention | [model_loader_service.py](Backend/app/services/model_loader_service.py) `transcribe_whisper_with_attention`, [inferences.py](Backend/app/api/routes/inferences.py) `extract_attention_pairs_endpoint` | [AttentionVisualization.tsx](Frontend/src/components/visualization/AttentionVisualization.tsx) |
| Saliency | [saliency_service.py](Backend/app/services/saliency_service.py) | [SaliencyVisualization.tsx](Frontend/src/components/visualization/SaliencyVisualization.tsx) |
| Embeddings | [model_loader_service.py](Backend/app/services/model_loader_service.py) `extract_whisper_embeddings`, `extract_wav2vec2_embeddings`, `reduce_dimensions` | [EmbeddingPlot.tsx](Frontend/src/components/visualization/EmbeddingPlot.tsx), [EmbeddingContext.tsx](Frontend/src/contexts/EmbeddingContext.tsx) |
| Perturbation | [pertubation_service.py](Backend/app/services/pertubation_service.py) | [PerturbationTools.tsx](Frontend/src/components/analysis/PerturbationTools.tsx) |
| Scalers | batch endpoints in [inferences.py](Backend/app/api/routes/inferences.py) | [ScalersVisualization.tsx](Frontend/src/components/visualization/ScalersVisualization.tsx) |
| Caching | [redis.py](Backend/app/core/redis.py) | (transparent) |
| Sessions | [session.py](Backend/app/core/session.py) | Cookie `sid` |

For directory maps + env vars + build/deploy, see [PROJECT.md](PROJECT.md).
