# AudioLens â€” Feature & Pipeline Walkthrough

> **Companion to [PROJECT.md](PROJECT.md).** That file maps *where* code lives. This file explains *what each feature does, what the user sees, why it matters, and the exact call chain from click to response.*
>
> Written for two audiences:
> - **Humans** learning the tool for the first time or onboarding to a feature.
> - **AI coding assistants** answering "how does saliency work here?" or "what happens when I press Run?".
>
> Structure per feature: **Concept â†’ What UI shows â†’ User actions â†’ Pipeline (click â†’ API â†’ service â†’ cache â†’ response) â†’ Extension ideas.**

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

- **Self-attention:** tokens attending to tokens in the same sequence (encoder side, or decoder â†’ decoder).
- **Cross-attention:** decoder tokens attending to encoder outputs. In Whisper, this is how each transcribed word aligns to a slice of audio.
- **Layers Ã— heads:** every transformer layer contains multiple attention "heads" (parallel views). Different heads specialize (phoneme boundaries, prosody, silence, etc.). AudioLens lets you pick layer + head.

**What AudioLens shows for attention:** the word-to-audio-time alignment implied by Whisper's cross-attention â€” which audio interval the model "listened to" when producing each transcript word.

**What else you could do:** head-role analysis (which head fires on stop consonants?), attention-rollout across layers, attention-entropy (peaked vs. diffuse focus), attention-diff between clean vs. perturbed audio.

### What is **saliency**?

Attention says *where the model looked*. Saliency says *how much each input actually mattered to the final prediction*. They often disagree â€” the model may look at silence but base its decision on a vowel elsewhere.

Methods (all supported in AudioLens):

| Method | How it works | Cost | Property |
|--------|--------------|------|----------|
| **GradCAM** | Gradient Ã— activation of a chosen layer | Low | Fast, coarse, layer-specific |
| **IntegratedGradients** | Average gradient along a straight line from baseline to input | Medium | Axiomatic, respects completeness |
| **LIME** | Perturb input, fit a linear surrogate locally | Medium | Model-agnostic, noisy |
| **SHAP** | Shapley values via sampling | High | Theoretically grounded, slow |
| **LRP** | Layer-wise relevance propagation | Medium | Signed relevance flow |

**What AudioLens shows for saliency:** segment-level (per short time window) importance scores overlaid on the waveform. Bright regions = pushed prediction toward the chosen class.

**What else you could do:** class-conditional saliency (which regions push *toward* "angry" vs. *away*), saliency stability under noise, occlusion sensitivity, saliency-vs-attention divergence plots.

### What are **embeddings**?

Every audio clip becomes a high-dimensional vector inside the model (512-D for Whisper base, 768-D for Wav2Vec2). Similar audio â†’ nearby vectors. Because 512-D is unplottable, AudioLens projects to 2-D or 3-D with PCA / t-SNE / UMAP.

- **PCA:** linear, fast, preserves global variance. Distances roughly meaningful.
- **t-SNE:** nonlinear, preserves *local* neighborhoods. Clusters look clean but global distances are meaningless.
- **UMAP:** nonlinear, faster than t-SNE, preserves some global structure.

**What AudioLens shows for embeddings:** scatter plot; clicking a point selects the file; box/lasso selects groups.

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
- Drop file â†’ auto-uploads, auto-runs inference, auto-extracts embedding.

**Pipeline:**

```
User drops file
   â”‚
   â–¼
AudioUploader.tsx  â†’  POST /upload  (multipart, {file, model})
   â”‚                        â”‚
   â”‚                        â–¼
   â”‚              Backend/app/api/routes/upload.py::upload_audio_file
   â”‚                        â”‚
   â”‚                        â”œâ”€ save to uploads/{uuid}.{ext}
   â”‚                        â”œâ”€ librosa loads â†’ duration, sample_rate
   â”‚                        â”œâ”€ auto call run_inference()   â†’ cached
   â”‚                        â””â”€ auto call extract_single_embedding()  â†’ cached
   â”‚
   â–¼
Response: { filename, file_id, duration, sample_rate, size, prediction }
   â”‚
   â–¼
MainLayout state updated â†’ PredictionPanel + EmbeddingPanel refresh
```

Backend service refs: [upload.py](Backend/app/api/routes/upload.py), [model_loader_service.py](Backend/app/services/model_loader_service.py).

**Extension ideas:** batch upload, format transcode, sample-rate normalization on ingest.

---

## 2. Dataset Browsing & File Serving

**What it does:** Browse built-in datasets (`common-voice`, `ravdess`) and stream audio via HTTP Range requests so the waveform can seek without downloading everything.

**UI:** [AudioDatasetPanel.tsx](Frontend/src/components/panels/AudioDatasetPanel.tsx) + [AudioDataTable.tsx](Frontend/src/components/audio/AudioDataTable.tsx).

**User actions:** sort, filter, select a row â†’ becomes the active file for other panels.

**Pipeline (metadata):**

```
Panel mounts
   â”‚
   â–¼
GET /{dataset}/metadata
   â”‚
   â–¼
datasets.py::get_metadata()  â†’  reads CSV from Backend/data/{dataset}/
   â”‚
   â–¼
Rows returned â†’ rendered in AudioDataTable
```

**Pipeline (audio playback with seeking):**

```
<audio> or wavesurfer.js issues:
   Range: bytes=1024-2047
   â”‚
   â–¼
GET /{dataset}/file/{path}   (datasets.py::serve_dataset_file)
   â”‚
   â”œâ”€ resolve_file(dataset, path, session_id)  â†’ validated absolute path
   â”œâ”€ if Range header:
   â”‚      StreamingResponse 206, headers:
   â”‚        Accept-Ranges: bytes
   â”‚        Content-Range: bytes X-Y/total
   â””â”€ else: FileResponse full file
```

Custom datasets are addressed with `dataset = "custom:{sid}:{name}"` and URL-encoded.

**Extension ideas:** paginated metadata, server-side filtering, waveform thumbnails cached to disk.

---

## 3. Custom Datasets

**What it does:** Users create per-session datasets and upload their own audio into them.

**UI:** [CustomDatasetManager.tsx](Frontend/src/components/dataset/CustomDatasetManager.tsx) â€” tabs for List / Create / Upload.

**Pipeline:**

```
Create:
   POST /dataset/create  { dataset_name }
      â†’ dataset_management.py::create_custom_dataset
      â†’ CustomDatasetManager(session_id).create_dataset()
      â†’ mkdir uploads/sessions/{sid}/{name}/ + metadata.json

Upload:
   POST /dataset/{name}/files  (multipart, files[])
      â†’ validates each file (extension, librosa-decodable)
      â†’ saves as {uuid}.ext + records {filename, original, duration, sample_rate, size}
      â†’ returns updated metadata

List:
   GET /upload/dataset
      â†’ enumerates uploads/sessions/{sid}/*
```

**Session scoping:** dataset name stored/served as `custom:{session_id}:{dataset_name}` so it can never collide with another user's dataset.

**Extension ideas:** dataset export/import, per-file labels for supervised eval, dataset diff.

---

## 4. Model Inference â€” Whisper ASR

**Concept:** Whisper transcribes speech to text. AudioLens exposes `whisper-base` (fast, small) and `whisper-large-v3` (accurate, slow).

**UI:** [PredictionPanel.tsx](Frontend/src/components/panels/PredictionPanel.tsx) â€” shows transcript. If the file has ground truth (e.g., `common-voice`), also shows accuracy metrics.

**Endpoints:**
- `POST /inferences/run` â€” generic entry point, returns `{predicted_transcript}`.
- `POST /inferences/whisper-accuracy` â€” comparison against ground truth, returns WER / CER / Levenshtein / exact-match / character-similarity.

**Pipeline:**

```
User selects file + model=whisper-*
   â”‚
   â–¼
PredictionPanel.tsx â†’ POST /inferences/run  { model, file_path | dataset+dataset_file }
   â”‚
   â–¼
inferences.py::run_inference
   â”‚
   â”œâ”€ MD5(file_path + size + mtime) â†’ cache key
   â”œâ”€ Redis hit? â†’ return cached, done
   â”œâ”€ else route via MODEL_FUNCTIONS[model]:
   â”‚      transcribe_whisper_base()  or  transcribe_whisper_large()
   â”‚      â†’ model_loader_service.py::transcribe_whisper() (single impl)
   â”‚        - loads WhisperForConditionalGeneration w/ attn_implementation="eager"
   â”‚        - librosa load, resample to 16 kHz
   â”‚        - processor(audio) â†’ input_features
   â”‚        - model.generate() â†’ token IDs â†’ processor.batch_decode()
   â”œâ”€ Cache result 6 h
   â””â”€ Return {predicted_transcript, ...}
```

**Critical implementation note:** `attn_implementation="eager"` is required at model load. SDPA / FlashAttention paths silently discard attentions â€” later attention endpoints would return empty results.

Service ref: [model_loader_service.py](Backend/app/services/model_loader_service.py) (`transcribe_whisper` at ~line 24, `transcribe_whisper_base` at ~356).

**Extension ideas:** language forcing, chunk-level confidence, streaming transcription.

---

## 5. Model Inference â€” Wav2Vec2 Emotion

**Concept:** Wav2Vec2 fine-tuned for speech emotion recognition (model: `r-f/wav2vec-english-speech-emotion-recognition`). Labels: `neutral / happy / sad / angry / fear`.

**UI:** [PredictionPanel.tsx](Frontend/src/components/panels/PredictionPanel.tsx) â€” predicted emotion + per-class probability bars.

**Endpoint:** `POST /inferences/wav2vec2-detailed` with `{file_path | dataset+dataset_file, include_attention?}`.

**Pipeline:**

```
POST /inferences/wav2vec2-detailed
   â”‚
   â–¼
inferences.py â†’ predict_emotion_wave2vec  (model_loader_service.py:392)
   â”‚
   â”œâ”€ Wav2Vec2ForSequenceClassification, on CUDA if available
   â”œâ”€ processor(audio, sr=16k) â†’ input_values
   â”œâ”€ model(**inputs) â†’ logits
   â”œâ”€ softmax â†’ probabilities dict
   â””â”€ argmax â†’ predicted_emotion, top-p confidence
   â”‚
   â–¼
Return { predicted_emotion, probabilities, confidence, attention? }
```

If `include_attention=True`, `predict_emotion_wave2vec_with_attention` sets `config.output_attentions=True` and returns attention layers alongside.

**Extension ideas:** confidence calibration, ordinal emotion axes (arousal/valence), multilingual emotion model.

---

## 6. Attention Visualization

**What AudioLens shows:** For a chosen Whisper layer + head, the cross-attention weights aligning each transcript word to a time interval in the audio.

**UI:** [AttentionVisualization.tsx](Frontend/src/components/visualization/AttentionVisualization.tsx)
- Layer dropdown (default 6), head dropdown (default 0).
- Word-to-word attention matrix / heatmap.
- Timeline attention curve (weight over time).
- Word-time pairs list.

**Endpoint:** `POST /inferences/attention-pairs` â€” `{model, file_path|dataset+dataset_file, layer, head}`.

**Pipeline:**

```
User picks layer / head
   â”‚
   â–¼
AttentionVisualization useEffect â†’ POST /inferences/attention-pairs
   â”‚
   â–¼
inferences.py::extract_attention_pairs_endpoint  (~line 1252)
   â”‚
   â”œâ”€ transcribe_whisper_with_attention()      â†’ transcript + attention[layers][heads]
   â”œâ”€ transcribe_whisper_with_timestamps()     â†’ word chunks with [start,end] times
   â”œâ”€ process_attention_into_pairs()           â†’ merge:
   â”‚      for each (from_word, to_word):
   â”‚          attention_weight = attention[layer][head][from_idx][to_idx]
   â”‚          from_time, to_time from timestamps
   â””â”€ Cache key: {model}_attention_pairs_{file_hash}_l{layer}_h{head}
   â”‚
   â–¼
Response: { attention_pairs[], timeline_attention[], transcript }
   â”‚
   â–¼
Frontend renders heatmap + timeline
```

**What you can do beyond current UI:** head-role tagging, attention rollout (multiply across layers), attention entropy per token, side-by-side attention for clean vs. perturbed inputs, attention-vs-saliency divergence view.

**Extension ideas:** Wav2Vec2 encoder-attention view (currently Whisper-only), attention export to CSV, click-a-word-to-jump-audio.

---

## 7. Embedding Extraction & Projection

**What AudioLens shows:** 2-D or 3-D scatter of many audio clips. Each point = one file. Nearby points = model thinks they sound similar.

**UI:**
- [EmbeddingPanel.tsx](Frontend/src/components/panels/EmbeddingPanel.tsx) â€” model / reduction-method / component-count selectors, "Extract" trigger.
- [EmbeddingPlot.tsx](Frontend/src/components/visualization/EmbeddingPlot.tsx) â€” Plotly scatter with box/lasso selection, plane picker (`xy`/`xz`/`yz`) for 3-D, angle-range filter.
- Shared via [EmbeddingContext.tsx](Frontend/src/contexts/EmbeddingContext.tsx).

**Endpoint:** `POST /inferences/embeddings` â€” `{model, dataset, files[], reduction_method: pca|tsne|umap, n_components: 2|3}`.

**Pipeline:**

```
User clicks Extract
   â”‚
   â–¼
POST /inferences/embeddings
   â”‚
   â–¼
inferences.py::extract_embeddings_endpoint
   â”‚
   â”œâ”€ For each file:
   â”‚     cache lookup: {model}_embeddings_{hash}
   â”‚     miss â†’ extract_whisper_embeddings()  (encoder.last_hidden_state, mean-pool)
   â”‚           or extract_wav2vec2_embeddings() (768-D)
   â”‚     store 24 h
   â”œâ”€ Stack all embeddings â†’ matrix [N Ã— D]
   â”œâ”€ reduce_dimensions(matrix, method, n_components)
   â”‚     - pca  â†’ sklearn.decomposition.PCA
   â”‚     - tsne â†’ sklearn.manifold.TSNE (perplexity default)
   â”‚     - umap â†’ umap.UMAP
   â””â”€ Return raw + reduced coords
   â”‚
   â–¼
EmbeddingContext stores â†’ EmbeddingPlot renders
```

**Interactive UI actions:**
- Click point â†’ set active file (triggers Prediction, Saliency, Attention refresh).
- Box / lasso select â†’ subset for group inspection.
- Switch reduction method â†’ re-run reduction (raw embeddings stay cached, only projection recomputed).
- 3-D plane picker â†’ collapse to a chosen 2-D face for cleaner reading.

**Extension ideas:** k-NN lookup ("show me clips like this one"), embedding drift under perturbation (arrow from clean â†’ perturbed point), cluster labels, cross-model embedding comparison, export to Vega/Bokeh.

---

## 8. Saliency Maps

**What AudioLens shows:** For a chosen model + method, per-segment importance scores displayed as a colored overlay along the waveform + a series plot.

**UI:** [SaliencyVisualization.tsx](Frontend/src/components/visualization/SaliencyVisualization.tsx)
- Method picker: `gradcam | integrated_gradients | lime | shap`.
- Overlay on waveform, timeline series plot.
- Segment tooltips with `start_time`, `end_time`, `saliency`, and (when Whisper) `word`.

**Endpoint:** `POST /saliency/generate` â€” `{model, method, file_path|dataset+dataset_file, no_cache?}`.

**Pipeline:**

```
POST /saliency/generate
   â”‚
   â–¼
saliency.py::generate_saliency_endpoint
   â”‚
   â”œâ”€ Resolve file (upload or dataset)
   â”œâ”€ Cache key: saliency_v2_{model}_{method}_{file_hash}
   â”‚     (schema version = v2; bump if response shape changes)
   â”œâ”€ Length cap enforced:
   â”‚     normal â‰¤ MAX_SALIENCY_SECONDS  (12 s)
   â”‚     SHAP   â‰¤ MAX_SALIENCY_SECONDS_SHAP  (6 s)
   â”œâ”€ saliency_service.py::generate_saliency
   â”‚     dispatches to method impl:
   â”‚       gradcam              â†’ captum LayerGradCam / grad Ã— activation
   â”‚       integrated_gradients â†’ captum IntegratedGradients (baseline=zeros)
   â”‚       lime                 â†’ lime.lime_audio surrogate
   â”‚       shap                 â†’ shap.KernelExplainer  (SALIENCY_SHAP_SAMPLES=8)
   â”‚       lrp                  â†’ captum LRP
   â””â”€ Response:
        { model, method, segments: [{start_time, end_time, saliency, word?}],
          total_duration, series: [...] }
```

**Length caps are load-bearing.** SHAP with N segments is O(2^N) worst case; raising `MAX_SALIENCY_SECONDS_SHAP` above 6 without profiling will OOM.

**Extension ideas:** class-conditional saliency (positive vs. negative contributions per emotion class), saliency stability sweep across noise levels, occlusion baseline for sanity checks, saliency-vs-attention divergence overlay.

---

## 9. Perturbation Tools

**What AudioLens shows:** After applying one or more perturbations, the perturbed audio replaces the current audio in the player; inference auto-reruns so the user sees how the prediction shifted.

**UI:** [PerturbationTools.tsx](Frontend/src/components/analysis/PerturbationTools.tsx) â€” checkboxes + sliders per perturbation.

**Endpoint:** `POST /perturb` â€” `{file_path, dataset?, perturbations: [{type, params}]}`.

**Perturbation menu:**

| Type | Params | Effect |
|------|--------|--------|
| `noise` | `noise_level` (Ïƒ of Gaussian) | Add white noise |
| `timeMasking` | `mask_start_percent`, `mask_end_percent` | Zero a time slice |
| `frequencyMasking` | `mask_low_hz`, `mask_high_hz` | FFT-mask a frequency band |
| `pitchShift` | `pitch_shift_semitones` (Â±6 max) | librosa `pitch_shift` |
| `timeStretch` | `time_stretch_factor` | librosa `time_stretch` (no pitch change) |

**Pipeline:**

```
User configures perturbations + clicks Apply
   â”‚
   â–¼
POST /perturb
   â”‚
   â–¼
perturbations.py::apply_perturbations
   â”‚
   â”œâ”€ librosa load audio
   â”œâ”€ Sequentially apply each perturbation via pertubation_service.py [sic]
   â”œâ”€ soundfile write to uploads/{uuid}.wav
   â””â”€ Return { perturbed_file, duration_ms, sample_rate,
                applied_perturbations[], success }
   â”‚
   â–¼
MainLayout swaps active file â†’ PredictionPanel auto re-runs inference
```

**Constraints:** pitch shift limited to Â±6 semitones and â‰¤ 30 s clips inside the service â€” enforced to keep librosa fast and outputs sensible.

**Extension ideas:** perturbation chain scoring (auto-report Î”emotion / Î”WER per step), reverberation / codec / background-mix perturbations, adversarial (gradient-directed) perturbations, batch perturbation over a whole dataset for robustness metrics.

---

## 10. Scalers / Batch Metrics

**Concept:** Aggregate signals across many files. Post commit `0863652`, scalers pull from the audio-embedding pipeline output rather than being a separate stream.

**UI:** [ScalersVisualization.tsx](Frontend/src/components/visualization/ScalersVisualization.tsx) + [ScalarPlot.tsx](Frontend/src/components/visualization/ScalarPlot.tsx).

**Endpoints:**
- `POST /inferences/wav2vec2-batch` â€” emotion distribution across N files.
- `POST /inferences/whisper-batch` â€” word-frequency stats.
- `POST /inferences/audio-frequency-batch` â€” spectral feature aggregates.

**Pipeline (Wav2Vec2 batch):**

```
POST /inferences/wav2vec2-batch  { filenames[], dataset }
   â”‚
   â–¼
For each file:
   cache lookup {wav2vec2}_{hash}
   miss â†’ run predict_emotion_wave2vec (populates cache)
   â”‚
Aggregate:
   emotion_distribution = counts / total
   dominant_emotion = argmax
   cache_hit_rate = cached / total
   â”‚
   â–¼
Response drives:
   - Pie chart of emotion distribution
   - Cache stats (dev insight)
```

**Audio frequency batch:** uses `extract_audio_frequency_features()` (librosa) â€” spectral centroid, spectral rolloff, MFCC, chroma, tonnetz, tempo, RMS, zero-crossing rate. Backend returns per-feature `mean / std / min / max / median` + histograms.

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

**File hash inputs:** `md5(file_path + size + mtime)` â€” the mtime component means editing/replacing a file at the same path invalidates its cache automatically.

**Schema versioning:** saliency uses `_v2` in the key. When response shape changes, bump to `_v3` â€” cleaner than writing a migration.

---

## 12. End-to-End User Walkthrough

A typical exploration session:

1. **Pick model + dataset** in the toolbar ([Toolbar.tsx](Frontend/src/components/layout/Toolbar.tsx)). Say `whisper-base` + `common-voice`.
2. **Browse dataset** in the audio table. Click a row.
   - Backend serves metadata CSV + streams audio via Range requests.
3. **Play back** the waveform ([WaveformViewer.tsx](Frontend/src/components/audio/WaveformViewer.tsx) via wavesurfer.js).
4. **Prediction panel** auto-populates: `POST /inferences/run` â†’ transcript + accuracy metrics.
5. **Attention** tab: pick layer 6 / head 0. `POST /inferences/attention-pairs` returns word-to-time alignment. Try other heads to see specialization.
6. **Embeddings**: click Extract for the whole dataset. Backend caches raw 768-D vectors, projects to 2-D via UMAP. Scatter plot renders. Lasso-select a cluster.
7. **Saliency**: pick `gradcam` for the currently selected file. Overlay shows which audio segments drove the transcript.
8. **Perturbation**: add Gaussian noise Ïƒ=0.005, apply. Perturbed file becomes active; prediction re-runs. Compare transcript / attention / saliency vs. clean.
9. **Custom dataset**: upload own recordings; every feature above works on them via `dataset = "custom:{sid}:{name}"`.

Everything is cache-warmed after the first pass â€” repeat interactions are Redis hits (see cache-hit-rate in scaler batch responses for dev insight).

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
