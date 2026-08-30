# Jacobian Lens — Technical Reference

> **Scope note.** This document describes the *current* implementation
> (`Backend/app/services/jacobian_lens_service.py`, as of commit `1185f2a`
> "refactor: fit Jacobian lenses via Hutchinson-probe transports"). It replaces
> the earlier reference that described the ridge-regression + PMI readout
> (preserved in git history at commit `583704d`).
>
> Part I documents the Whisper substrate (layer/dimension tables). Part II
> documents the J-Lens itself: what is fitted, what is frozen, and exactly how
> encoder representations become token scores.

---

# Part I — Whisper Model Substrate

## 1. Dimension glossary

| Symbol | Name | Meaning |
|---|---|---|
| `n_mels` | Mel bins | Frequency bins of the log-Mel spectrogram (80 for base, 128 for large-v3) |
| `n_audio_ctx` | Audio context | Frames fed to the encoder after 2× downsampling (1500 = 30 s of audio, ~20 ms/frame) |
| `d_model` | Hidden width | Embedding/feature vector size at every layer (identical for encoder and decoder in all Whisper sizes) |
| `n_head` | Attention heads | Parallel attention streams per layer; each head has `head_dim = d_model / n_head` (always 64) |
| `n_layer` | Layers | Number of encoder blocks and decoder blocks |
| FFN | Feed-forward width | Hidden size of the MLP inside each block (always `4 × d_model`) |
| `n_text_ctx` | Text context | Maximum decoder token positions (448) |
| `n_vocab` | Vocab size | Number of output tokens incl. specials (SOT, language, task, timestamps) |

## 2. Whisper size family

| Size | Params | Mel bins | Enc layers | Dec layers | d_model | Heads | head_dim | FFN | Vocab |
|---|---|---|---|---|---|---|---|---|---|
| tiny | 39M | 80 | 4 | 4 | 384 | 6 | 64 | 1536 | 51,865 |
| **base** (default here) | 74M | 80 | **6** | **6** | **512** | **8** | 64 | **2048** | 51,865 |
| small | 244M | 80 | 12 | 12 | 768 | 12 | 64 | 3072 | 51,865 |
| medium | 769M | 80 | 24 | 24 | 1024 | 16 | 64 | 4096 | 51,865 |
| **large-v3** | 1550M | **128** | **32** | **32** | **1280** | **20** | 64 | **5120** | 51,866 |

## 3. Encoder — sequential tensor flow

| # | Stage | whisper-base | whisper-large-v3 |
|---|---|---|---|
| 1 | Raw audio (16 kHz, 30 s) | (480,000,) | (480,000,) |
| 2 | Log-Mel spectrogram | (80, 3000) | (128, 3000) |
| 3 | Conv1d #1 (k=3, s=1, GELU) | 80 → 512, out (512, 3000) | 128 → 1280, out (1280, 3000) |
| 4 | Conv1d #2 (k=3, s=2, GELU) | 512 → 512, out (512, 1500) | 1280 → 1280, out (1280, 1500) |
| 5 | Transpose to sequence | (1500, 512) | (1500, 1280) |
| 6 | + sinusoidal positional embedding | (1500, 512) | (1500, 1280) |
| 7 | Encoder blocks ×6 / ×32 (see §4) | (1500, 512) | (1500, 1280) |
| 8 | Final LayerNorm | (1500, 512) | (1500, 1280) |

## 4. Inside one encoder block

| Sub-layer | whisper-base | whisper-large-v3 |
|---|---|---|
| LayerNorm | (1500, 512) | (1500, 1280) |
| Self-attn Q/K/V projections | 512→512 each | 1280→1280 each |
| Attention matrix (per head) | (8, 1500, 1500) | (20, 1500, 1500) |
| Output projection | 512→512 | 1280→1280 |
| Residual add | (1500, 512) | (1500, 1280) |
| FFN fc1 (GELU) | 512→2048 | 1280→5120 |
| FFN fc2 | 2048→512 | 5120→1280 |
| Residual add | (1500, 512) | (1500, 1280) |

## 5. Decoder — sequential tensor flow

| # | Stage | whisper-base | whisper-large-v3 |
|---|---|---|---|
| 1 | Input tokens (SOT + lang + task + text + timestamps, max) | (T ≤ 448,) | (T ≤ 448,) |
| 2 | Token embedding (vocab × d) | 51,865 → 512 | 51,866 → 1280 |
| 3 | + learned positional embedding | (448, 512) | (448, 1280) |
| 4 | Decoder blocks ×6 / ×32 (see §6) | (T, 512) | (T, 1280) |
| 5 | Final LayerNorm | (T, 512) | (T, 1280) |
| 6 | LM head (tied to token embedding, no bias) | 512 → 51,865 | 1280 → 51,866 |

## 6. Inside one decoder block

| Sub-layer | whisper-base | whisper-large-v3 |
|---|---|---|
| LayerNorm | (T, 512) | (T, 1280) |
| Causal self-attn (Q/K/V, 8/20 heads) | 512→512, masks future tokens | 1280→1280 |
| Residual add | (T, 512) | (T, 1280) |
| Cross-attn: Q from decoder, **K/V from encoder output (1500, d)** | Q/K/V 512→512, scores (8, T, 1500) | Q/K/V 1280→1280, scores (20, T, 1500) |
| Residual add | (T, 512) | (T, 1280) |
| FFN 512→2048→512 / 1280→5120→1280 | (T, 512) | (T, 1280) |

The structural difference vs. the encoder: each decoder block has a **second
attention** (cross-attention) whose key/value length is always **1500** (the
audio tokens), while the query length is the transcript position T.

## 7. Known hardcoded-dimension caveats in this codebase

- `Backend/app/services/model_loader_service.py:360-361` hardcodes
  `num_layers = 6 if base else 12` and `num_heads = 8 if base else 16`. The
  12/16 values match **medium**, not large-v3 (32 layers / 20 heads).
- `extract_whisper_attention_pairs` defaults to `layer_idx=6`
  (model_loader_service.py:1435), which is out of range for whisper-base
  (valid encoder block indices are 0–5).

---

# Part II — The Jacobian Lens

## 8. What a J-Lens is here

A J-Lens is a set of per-encoder-layer linear maps **fitted (estimated, not
trained)** on a frozen speech-to-text model. Each map approximates how the
*pooled final decoder state* would shift if that layer's *pooled encoder
state* shifted — a first-order ("Jacobian") transport from acoustic
representation space to verbal representation space. At apply time, each map
is chained with the frozen LM head to rank candidate vocabulary tokens per
time window.

### What is fitted vs. frozen

| Object | Status |
|---|---|
| Encoder + decoder weights | pretrained, frozen |
| E (`proj_out.weight`, tied to decoder token embedding) | pretrained, frozen, read-only |
| Baselines `h̄` (per layer) | fitted — mean of pooled encoder states over the fit samples |
| J matrices (per layer) | fitted — averaged Hutchinson-probe transports, stored in `lens.pt` |

Nothing is gradient-descent-trained: the model runs in `eval()`, no optimizer,
no weight updates. `torch.autograd.grad` is used only to *measure* VJPs
through the frozen network.

## 9. Which layers the lens applies to

- **Every encoder transformer block** gets a lens — not a selected subset.
  `_seq2seq_states` returns `encoder_hidden_states[1:]`
  (jacobian_lens_service.py:72): HF prepends the embedding-output state at
  index 0, which is dropped. One lens per actual encoder block.
- **whisper-base:** encoder blocks 1–6 (reported as `layer: 0..5`).
- **whisper-large-v3:** all 32 encoder blocks (`layer: 0..31`).
- **Excluded:** index 0 (conv1 + conv2 + positional-embedding frontend) and
  the entire decoder. The decoder is never lensed — its final pooled state is
  only the *target*.

## 10. The target: the final decoder state

`decoder_hidden_states[-1]` is the output of the **last decoder block** —
shape `(1, T, d)`, where T is the number of teacher-forced transcript tokens.

Semantically: position *t* has fused all tokens up to *t* (causal self-attn)
and all 1500 audio frames (cross-attention), so it is the model's "pre-logit
summary of what token comes next at position t+1". Early positions carry more
audio-driven content; later positions are dominated by language-model context.

What comes after that state (in HF `WhisperDecoder.forward`):

```
final decoder block output        (1, T, d)      ← the target lives HERE
        │
final LayerNorm (dec.layer_norm)  (1, T, d)      ← NOT in decoder_hidden_states
        │
LM head proj_out (tied weights)   d → n_vocab
        │
logits                            (1, T, 51,865 / 51,866)
        │
softmax / sampling → next-token id → detokenize
```

Two facts that matter downstream:

1. The target is **pre-final-LayerNorm** (HF collects hidden states inside the
   block loop, before `layer_norm` is applied).
2. The decoder is **not generated token-by-token** during fitting — see §11.

## 11. Fitting: a global-to-global map

### Teacher forcing, not generation

`_seq2seq_states` (jacobian_lens_service.py:52-72) passes the transcript as
`labels=labels`, so the decoder runs **teacher-forced in one parallel pass** —
all T positions computed simultaneously, each attending to the full 1500-frame
encoder output via cross-attention. The autoregressive generation loop never
happens during fitting.

### Both sides collapse to single vectors

```
Encoder layer ℓ:  (1, 1500, d)  --mean over frames-->   h_ℓ ∈ R^d    "acoustic gist at depth ℓ"
                                                              │
                                                    J_ℓ  =  ∂z / ∂h_ℓ   (d × d matrix)
                                                              │
Decoder final:    (1, T, d)     --mean over tokens-->    z ∈ R^d     "verbal gist"
```

- `sources = [state.mean(dim=1)...]` (line 155) — each encoder block's output
  is averaged over all 1500 frames.
- `target = target_states.mean(dim=1)` (line 147) — final decoder states
  averaged over all transcript positions.

Variable sequence lengths (1500 vs. T, different per clip) never reach the
math: both means produce fixed d-dim vectors, so every sample contributes to
the same per-layer matrices regardless of clip or transcript length.

### Hutchinson (Rademacher) probe estimation

For probe `r ∈ {±1}^d` and scalar `⟨z, r⟩`, autograd gives
`∇_{h_ℓ} ⟨z, r⟩ = J_ℓᵀ r` — one VJP through the entire frozen network per
probe. The VJP is summed over frames (`gradient.sum(dim=1)`, line 179),
consistent with a *pooled* (broadcast) perturbation. Averaging
`outer(r, Jᵀr)` over probes and samples estimates J without ever forming an
exact Jacobian:

```
J_ℓ  ≈  (1 / (N_samples · probe_count)) · Σ_samples Σ_probes outer(r, ∇_{h_ℓ}⟨z, r⟩)
```

Defaults: `probe_count=4` (range 1–32), `samples` 2–1000, `max_audio_seconds`
30 (max 60) — see `JacobianLensFitParameters` (schemas/jobs.py:103-106).

### Artifact format (format_version 1)

| Field | Description |
|---|---|
| `format_version` | 1 |
| `architecture` | `"seq2seq"` or `"ctc"` |
| `model_id` / `model_revision` | model identity; apply refuses a mismatch (lines 232-235) |
| `matrices` | `[L × (d_target, d_source)]` float32 transport matrices |
| `baselines` | `[L × d_source]` mean pooled encoder state per layer |
| `sample_count` / `probe_count` | fit statistics |

## 12. Handling variable audio and transcript lengths

| Variable | Mechanism | Result |
|---|---|---|
| Audio too long | `librosa.load(..., duration=max_audio_seconds)` (line 26) | truncated |
| Audio too short | Whisper feature extractor **always pads the log-Mel to 30 s** | silence-padded |
| Encoder frames | conv2 stride-2 halves 3000 | **always (1, 1500, d)** |
| Transcript too long | `tokenizer(transcript, truncation=True)` (line 54) | T capped at 448 |
| Transcript any length | `target_states.mean(dim=1)` (line 147) | **always (d,)** |

Costs to be aware of:

1. **Silence padding dilutes the encoder mean** — a 3 s clip is ~90%
   zero-energy frames, so its pooled `h_ℓ` is pulled toward the "silence"
   representation; baselines and J reflect the fit set's average padding
   fraction.
2. **Per-token weighting scales as 1/T** — tokens in short transcripts
   contribute more per token to the mean target.
3. **Mixed lengths bias the linearization point** — the lens approximates
   around the *average fit sample*; the fit-set length distribution matters.
4. **Apply-time timeline stretching** — all 1500 frames (including the padded
   tail) are bucketed and mapped via `duration · start / 1500`
   (lines 259-265), so for short clips the silent tail is plotted *inside*
   the real clip duration.

## 13. Applying the lens: how a pooled vector becomes token scores

### Two matrices, two jobs

J never produces vocab logits by itself; the vocab dimension appears only at
the last step. Shapes for whisper-base (d = 512, vocab = 51,865):

```
h − h̄          (512,)         encoder-space deviation
   │
   @ Jᵀ        (512, 512)     J changes DIRECTION within the same 512-dim space
   ▼
v              (512,)         decoder-state space
   │
   @ Eᵀ        (51865, 512)   E creates the vocab dimension — not J
   ▼
s              (51865,)       one score per token
```

- **Jᵀ** (512→512): entry `J[j, i] = ∂z_j/∂h_i`. The transport computes
  `v_j = Σᵢ (hᵢ − h̄ᵢ) · J[j, i]` — re-expressing the encoder deviation in the
  directions the decoder state would move.
- **Eᵀ** (512→51,865): `E = proj_out.weight`, shape `(n_vocab, d)`, one
  embedding row per token. `s_i = v · e_i` is a plain dot product with every
  token embedding — the same linear readout the real model uses
  (`logits = z @ Eᵀ`).

### Why the composition is meaningful

The true next-token logits are `logit_i = z · e_i` for the real final decoder
state z. The lens approximates:

```
z ≈ z̄ + J·(h − h̄)
⟹ logit_i ≈ z̄·eᵢ  +  (v · eᵢ)
      └─constant─┘   └── what the code computes (lines 255-257) ──┘
```

The constant term (the average utterance's logits) is deliberately dropped, so
`topk` returns tokens whose embeddings align with this segment's *deviation*
from an average clip.

### Interpretation contract (important)

These are **deviation scores, not probabilities**:

- No softmax; `topk` ranks raw dot products (line 257).
- "Most probable" really means "most *amplified* relative to the average fit
  utterance".
- Chained together, the whole readout is one composite linear map:
  `s = (h − h̄) @ (Jᵀ Eᵀ)` — a frozen, linear shadow of the sequential decoder.

### Why E works on an aggregated vector even though it was trained per-position

E was trained so that each position's state `z_t` predicts the next token via
`z_t @ Eᵀ` — it never saw pooled states during pretraining. It still works, in
a limited sense:

1. **Linearity commutes with the mean:**
   `E·((1/T) Σₜ zₜ) = (1/T) Σₜ (E·zₜ)` — the average next-token logit over the
   transcript. Well-defined, though not any single position's distribution.
2. **The code never evaluates E on the pooled state anyway** — only on the
   first-order *deviation* v. `Δlogit_i ≈ v · eᵢ` is just the chain rule at
   the operating point z̄, valid anywhere in R^d because E is a fixed linear
   functional.
3. **What degrades:** scores lose probabilistic calibration off-manifold; only
   the *ranking of deviations* is interpretable. Known biases: frequent-token
   geometry (partially mitigated by the baseline drop), special-token states
   included in the fit-time mean, and growing linearization error for large
   deviations.

## 14. Buckets: the synthetic time axis

**Buckets do not exist in Whisper.** Normal Whisper goes
3000 mel frames → encoder → 1500 frame vectors → cross-attention → ≤448 token
positions, with every frame participating individually. Buckets are a J-Lens
analysis construct, defined in `_pool_frames` (jacobian_lens_service.py:205-218).

| Term | What it is | Count |
|---|---|---|
| **frame** | One encoder output embedding — a d-dim vector for one ~20 ms slice of audio | 1500 |
| **bucket** | A contiguous group of ~15 consecutive frames, collapsed to one vector | 96 (default) |

### Why buckets exist

1. **Volume** — 1500 frames × top-5 tokens × 32 layers ≈ 240k token scores per
   clip; 96 × 5 × 32 ≈ 15k is renderable.
2. **Signal-to-noise** — a single 20 ms frame is a noisy fragment; averaging
   ~15 frames (≈0.3 s, roughly word-length) stabilizes the linear readout.
3. **Tunable resolution** — `max_frames` (default 96, range 8–256,
   `JacobianLensApplyParameters`, schemas/jobs.py:109-112).

### Aggregation technique: plain mean, and why it is mandatory

Equal-weight arithmetic mean — `values[start:end].mean(dim=0)` (line 210), in
float32 (line 254). No weighting, no learned pooling. Mean pooling at apply
time is **required for consistency with the fit**: J was estimated on
mean-pooled vectors (line 155), so the apply path must pool identically —
swapping in max- or attention-pooling would silently mismatch the statistics J
was linearized around. (The layer-probe pipeline enforces the same convention:
`pooling: "mean"` is hard-coded, schemas/jobs.py:126.)

### There are no decoder buckets to match

- **Fit time:** no buckets anywhere — both sides collapse to single d-dim
  vectors; 1500 ≠ T is irrelevant because the means erase both counts.
- **Apply time:** buckets exist **only on the encoder side**. The decoder
  never runs; it is impersonated by the frozen `Jᵀ Eᵀ` chain. Each output
  bucket's timestamp comes entirely from which encoder frames it pooled
  (lines 264-265).

Invariants that *are* enforced:

| Invariant | Guarantee |
|---|---|
| Encoder frame count always 1500 | mel padding to 30 s — one shared time axis for every layer and clip |
| Identical partition for every layer | `_pool_frames` on the same frame count, so bucket *b* of layer 3 covers the same window as bucket *b* of layer 27 |
| Lens matrix count = encoder layer count | `len(states) != len(matrices)` raise, lines 243-244 |
| Lens width = layer width / revision match | lines 232-235 |

## 15. Apply pipeline walkthrough

Example: 15 s clip, 10-token transcript, whisper-base, defaults
(`max_frames=96`, `top_k=5`).

```
Fit (once, on the fit set):                    global means → h̄_ℓ, J_ℓ per layer

Apply:
15 s clip
  → mel padded to 30 s            (80, 3000)
  → encoder only (no decoder)     6 × (1500, 512)
  → per layer: _pool_frames       96 buckets × ~15.6 frames   (≈0.31 s each)
  → per bucket: (h_b − h̄) @ J_ℓᵀ  v_b ∈ R^512
  → per bucket: v_b @ Eᵀ          s_b ∈ R^51,865
  → topk(5) per bucket            96 × 5 = 480 scored tokens per layer
```

Output: `layers[]`, each with `layer` index and `frames[]`, each frame with
`start_time`, `end_time`, and `tokens[]` (`token_id`, `token`, `score`).

Notes on this example:

- The decoder runs over ~13 positions during fitting (10 text + SOT/lang/task
  + EOT), never 448 — no padding to the context length.
- ~half of the 96 buckets cover the silent padded tail (see §12, cost 4).
- The top-5 within a bucket is a **set, not a phrase** — no ordering, no
  language-model coherence between tokens; the apparent "sequence" across
  buckets is emergent from time order, not generated.

## 16. Fit globally, apply locally

```
FIT (once per model, per fit-set):
  whole clip (1500 frames)  ──mean──►  h̄ ◄── J estimated ──►  z̄ ◄──mean──  whole transcript (T tokens)

APPLY (every clip):
  per bucket b (≈0.3 s):   (h_b − h̄) @ Jᵀ @ Eᵀ  ──►  top-5 tokens @ time window b
```

One globally-fitted map, swept across 96 local windows. Key properties:

1. **The global fit anchors every local read** — each bucket is transported as
   a deviation from the global baseline `(h_b − h̄)`.
2. **Time-homogeneity assumption** — the map is assumed valid at every time
   window. The real decoder violates this (cross-attention aligns different
   frames to different tokens), so the lens is a first-order,
   average-alignment view. This is the price of making J estimable from a
   handful of clips.
3. **Per-layer J, shared across buckets** — J differs per encoder layer
   (6 or 32 matrices) but is shared across all buckets within a layer, so
   per-layer comparisons trace the acoustic→verbal progression while
   per-bucket differences reflect the audio, not the map.

## 17. Limitations summary

| # | Limitation | Consequence |
|---|---|---|
| 1 | Silence padding dilutes pooled encoder means | Short clips' h̄ and J biased toward "silence" representation |
| 2 | Mean over T tokens (incl. special-token positions) | Transcript-average target; per-token weight ∝ 1/T |
| 3 | First-order linearization only | Error grows with distance from h̄ |
| 4 | E applied off-manifold | Scores are deviation rankings, not calibrated probabilities |
| 5 | Frequent-token geometry | Common-word embeddings align with many deviation directions; baseline drop mitigates but does not eliminate |
| 6 | Uniform bucketing ignores word boundaries | A bucket can straddle two words; buckets are "alignment-ish hints", not transcriptions |
| 7 | Timeline stretching for short clips | Padded tail mapped inside the real duration (§12.4) |
| 8 | Model/revision-bound artifact | Lens must be refitted when the model changes (lines 232-235) |

## 18. Code map

| Concern | Location |
|---|---|
| Fit / apply implementation | `Backend/app/services/jacobian_lens_service.py` |
| Architecture selection (`seq2seq`/`ctc`), resource variants | `Backend/app/worker/model_adapters.py` |
| Job parameter schemas (`probe_count`, `max_frames`, `top_k`, …) | `Backend/app/schemas/jobs.py` (lines 96-112) |
| Job execution | `Backend/app/worker/executor.py` (`_execute_jacobian_lens_fit` / `_apply`) |
| Lens repository (session-owned records) | `Backend/app/repositories/jacobian_lenses.py` |
| API routes | `Backend/app/api/routes/jobs.py`, `models.py` |
| Frontend lab / visualization | `Frontend/src/pages/JacobianLensLab.tsx`, `Frontend/src/components/visualization/JacobianLensVisualization.tsx` |

## References

- Hvingelby et al. (2023). *Encoder Jacobian Lenses for Interpreting Speech
  Models.* (Stochastic-Jacobian approach this implementation follows.)
- Rademacher/Hutchinson trace estimation — randomized probe identities used
  for the VJP-based transport estimate.
- Belinkov, Y. (2022). *Probing Classifiers: Promises, Shortcomings, and
  Advances.* Computational Linguistics, 48(1).
- Hewitt & Manning (2019). *A Structural Probe for Finding Syntax in Word
  Representations.* NAACL.
- Alain & Bengio (2017). *Understanding Intermediate Layers Using Linear
  Classifier Probes.* ICLR Workshop.
