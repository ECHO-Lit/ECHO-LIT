# Jacobian Lens — Technical Reference

> **Scope note.** This document describes the *current* implementation
> (`Backend/app/services/jacobian_lens_service.py` on branch
> `jlense-decorderonly-test`). It replaces both earlier references: the
> ridge-regression + PMI readout (commit `583704d`) and the pooled
> encoder-transport design (commit `1185f2a`).
>
> Part I documents the Whisper substrate (layer/dimension tables). Part II
> documents the J-Lens itself: a decoder-only, position-resolved lens built the
> same way as the LLM Jacobian lens (Gurnee et al., 2026, *Verbalizable
> Representations Form a Global Workspace in Language Models*).

## Redesign log — encoder transports → decoder-only lens

Motivation: the pooled encoder-transport design (`1185f2a`) fitted a global-to-
global map between mean-pooled encoder states and the mean-pooled decoder
state. Differentiating *after* pooling erased the (t, t′) alignment structure,
linearized around whole-clip means but was applied to local 0.3 s buckets,
dropped the readout constant, applied E off-manifold, and produced deviation
scores with no probability semantics. The current design fixes each of these
by matching the LLM Jacobian-lens construction.

| Change | Old (encoder transport) | New (decoder-only) |
|---|---|---|
| Lens location | one J per **encoder** block | one J per **decoder** layer (incl. embedding output) |
| Source | mean over all 1500 encoder frames | residual stream at each decoder position |
| Target | mean over all T decoder positions | `last_hidden_state` at every position t′ ≥ t |
| Differentiation | after pooling (Jacobian of means) | full sequence resolution, averaged afterwards |
| Spaces | cross-component (encoder → decoder) | same decoder space both sides |
| Readout | `(h_b − h̄) @ Jᵀ @ Eᵀ`, baseline-subtracted, constant dropped | `softmax(E (J h))` on the model's own logit scale |
| Apply-time units | synthetic 0.3 s buckets (`max_frames`) | decoder token positions (bucketing deleted) |
| Artifact | format v1, `[L × (d_target, d_source)]`, baselines | format v2, `[L × (d, d)]`, no baselines, `method: hutchinson-decoder-vjp` |
| Decoder at apply | never ran (impersonated by `Jᵀ Eᵀ`) | runs (greedy generation or provided transcript) |
| CTC support | encoder readout | removed (no decoder to lens) |
| Fit cost per sample | VJPs through encoder + decoder | encoder outside the autograd graph |

Verification: `Backend/tests/test_jacobian_lens.py` checks the fitted map
against the analytic closed form for a linear decoder (`J = 2M/(T+1)` with the
causal triangular average) and checks the apply readout against an independent
`(J h) @ Eᵀ` recomputation.

Known trade-off: the readout axis is now **transcript positions**, not audio
time. Which audio frame supports which token remains the job of cross-attention
tooling, not the lens.

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
| 7 | Encoder blocks ×6 / ×32 | (1500, 512) | (1500, 1280) |
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
| 4 | Decoder blocks ×6 / ×32 | (T, 512) | (T, 1280) |
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

Both caveats affect other services; the J-Lens reads layer counts from the
returned hidden-state tensors, never from hardcoded tables.

---

# Part II — The Jacobian Lens (decoder-only)

## 8. What a J-Lens is here

A J-Lens is a set of per-**decoder**-layer linear maps **fitted (estimated, not
trained)** on a frozen speech-to-text model. Each map is the average,
position-resolved causal map from that decoder layer's residual stream to the
model's final pre-logit state — exactly the estimator the LLM Jacobian lens
uses (Gurnee et al., 2026). At apply time, everything downstream of a layer is
replaced by the fitted map followed by the model's own output projection,
yielding ranked vocabulary tokens per **(position, layer)** cell.

### What is fitted vs. frozen

| Object | Status |
|---|---|
| Encoder + decoder weights | pretrained, frozen |
| E (`proj_out.weight`, tied to decoder token embedding) | pretrained, frozen, read-only |
| J matrices (per decoder layer) | fitted — averaged Hutchinson-probe VJPs, stored in `lens.pt` |

There are **no baselines** and **no gradient-descent training**: the model runs
in `eval()`, no optimizer, no weight updates. `torch.autograd.grad` is used
only to *measure* VJPs through the frozen decoder.

## 9. Which layers the lens applies to

- **Every recorded decoder hidden state except the final one.** HF appends the
  post-final-LayerNorm state to `decoder_hidden_states`; that state *is* the
  fit target, so it is excluded from the sources (identity check,
  `_decoder_states`, jacobian_lens_service.py:113-140).
- **whisper-base:** 7 lens layers — index 0 (token + positional embedding
  output) and decoder blocks 1–6, reported as `layer: 0..6`.
- **whisper-large-v3:** 33 lens layers (`layer: 0..32`).
- **Excluded:** the encoder entirely. It runs under `torch.no_grad()` and is
  never lensed — it only supplies cross-attention keys/values
  (`_encoder_features`, jacobian_lens_service.py:62-83). This mirrors the LLM
  case, where the lens lives in one shared residual stream rather than spanning
  separate components.

## 10. The target: the final pre-logit state

The fit target is the decoder's `last_hidden_state` — the exact tensor HF feeds
to `proj_out` to produce logits (post final LayerNorm where the architecture
has one; transformers ≥ 4.46 returns this object as both
`last_hidden_state` and the last `hidden_states` entry):

```
decoder block stack output        (1, T, d)
        │
final LayerNorm (dec.layer_norm)  (1, T, d)
        │                          ← the target lives HERE (last_hidden_state)
LM head proj_out (tied weights)   d → n_vocab
        │
logits                            (1, T, 51,865 / 51,866)
```

Semantically, position *t* has fused all tokens up to *t* (causal self-attn)
and all 1500 audio frames (cross-attention), so it is the model's "pre-logit
summary of what token comes next at position t+1".

## 11. Fitting: an averaged, position-resolved causal map

### Teacher forcing, not generation

`_decoder_states_for_fit` (jacobian_lens_service.py:142-150) passes the
transcript through `_decoder_input_ids` (labels shifted right past the start
token), so the decoder runs **teacher-forced in one parallel pass** — all T
positions computed simultaneously, each attending to the full 1500-frame
encoder output via cross-attention. The autoregressive generation loop never
happens during fitting.

### Full position resolution — the pooled design is gone

The earlier implementation pooled both sides to single vectors before
differentiating, which destroyed positional structure and forced a global-to-
global map. The current implementation differentiates at **full sequence
resolution** and averages afterwards, mirroring the LLM J-lens definition:

```
J_ℓ  =  E[ t, t' ≥ t, transcript, probe ]  [ ∂ h_dec,final,t' / ∂ h_dec,ℓ,t ]
```

- **Source:** the residual stream at decoder layer ℓ, position t — per
  position, never pooled.
- **Target:** the final pre-logit state at every future position t' ≥ t.
- **Averaging:** over all causally reachable (source, future) pairs, all fit
  transcripts, and all probes.

### Hutchinson (Rademacher) probe estimation

For probe `r ∈ {±1}^d`, the scalar `Σ_t' ⟨h_final,t', r⟩` backpropagated once
gives, at **every** source position t simultaneously, `Σ_{t'≥t} J_ℓ(t→t')ᵀ r`
— the causal mask zeroes the pairs with t > t', so one VJP covers the whole
triangle (`_decoder_states` returns graph-connected states;
`fit_decoder_jacobian_lens`, lines 153-244). Averaging `outer(r, Σ_t g_t)`
over probes and samples, normalized by the triangular count T(T+1)/2, estimates
J without ever forming an exact Jacobian:

```
J_ℓ  ≈  (1 / (N_samples · probe_count)) · Σ_samples Σ_probes outer(r, Σ_t ∂⟨h_final,·,r⟩/∂h_ℓ,t) / (T(T+1)/2)
```

Defaults: `probe_count=4` (range 1–32), `samples` 2–1000, `max_audio_seconds`
30 (max 60) — see `JacobianLensFitParameters` (schemas/jobs.py:103-106).

### Artifact format (format_version 2)

| Field | Description |
|---|---|
| `format_version` | 2 |
| `architecture` | `"decoder"` (CTC models are excluded — they have no decoder to lens) |
| `model_id` / `model_revision` | model identity; apply refuses a mismatch |
| `method` | `"hutchinson-decoder-vjp"` |
| `matrices` | `[L × (d, d)]` float32 transport matrices (square: same space both sides) |
| `sample_count` / `probe_count` | fit statistics |

There are **no baselines**: same-space readouts need no pooled anchor, and the
earlier baseline subtraction (an artifact of cross-space transport) is gone.

## 12. Handling variable audio and transcript lengths

| Variable | Mechanism | Result |
|---|---|---|
| Audio too long | `librosa.load(..., duration=max_audio_seconds)` | truncated |
| Audio too short | Whisper feature extractor **always pads the log-Mel to 30 s** | silence-padded |
| Encoder frames | conv2 stride-2 halves 3000 | **always (1, 1500, d)** |
| Transcript too long | tokenizer `truncation=True` (448 cap at apply) | T capped |
| Transcript any length | per-position readout | every position reported individually |

Note: silence padding no longer biases the *fit target* (positions, not a
mean over frames, are what J acts on at apply time). It still affects the
decoder states via cross-attention to padded frames — same as it affects real
transcription.

## 13. Applying the lens: the (position, layer) readout

### The decoder actually runs

1. Encoder runs under `no_grad` (substrate only).
2. Positions come from the model's own **greedy generation** (default) or from
   a **provided reference transcript** (`transcript` apply parameter) for a
   teacher-forced reading.
3. A teacher-forced decoder pass collects the residual stream at every layer
   and position.
4. Per layer, per position:

```
h_ℓ,t            (d,)          decoder residual stream at layer ℓ, position t
   │
   @ J_ℓᵀ        (d × d)       same space in, same space out
   ▼
v_ℓ,t            (d,)          linearized pre-logit state
   │
   @ Eᵀ          (d → 51,865)  E = proj_out.weight — the model's own readout
   ▼
logits           (51,865,)     softmax → per-token probability (display only)
   │
   topk(k)                     ranked tokens for this (position, layer) cell
```

This is literally `lens(h) = softmax(E (J_ℓ h))` from the LLM J-lens, with
Whisper's unembedding path (its final LayerNorm is part of the decoder, so it
is absorbed into J).

### Why the composition is meaningful

The true logits are `logits = E · h_final`. The lens approximates, for every
position and on average across contexts:

```
h_final,t' ≈ J_ℓ · h_ℓ,t        (first order, averaged over t' ≥ t)
⟹ logits ≈ E · (J_ℓ · h_ℓ,t)
```

Unlike the earlier cross-component design, **no constant term is dropped and no
baseline is subtracted** — the map lands in the same space the model's own
readout consumes, so scores sit on the real logit scale.

### Interpretation contract

- The `score` is a first-order-approximated logit; `probability` is its
  softmax (normalized for display, not calibrated).
- "Top token at (position, layer)" means "the token whose lens direction this
  activation most aligns with, averaged over contexts" — the verbalizable
  content the LLM J-lens surfaces, here at transcript positions.
- Cells at early layers can be noisy/uninterpretable; the LLM paper observes
  the same low-layer regime.
- The per-position top-k is a *ranking*, not a phrase: there is no language-
  model coherence between neighboring cells.

## 14. Positions: the natural time axis

**There is no bucketing.** The previous design pooled ~15 encoder frames into
synthetic 0.3 s buckets and needed `max_frames` tuning; that construct is
deleted along with `_pool_frames`. Positions are decoder token positions — the
same granularity the LLM J-lens reads at — and the frontend renders the
(position × layer) grid with the token spine on top.

| Invariant | Guarantee |
|---|---|
| Position count = decoder input length | teacher-forced pass on generated or provided ids |
| Identical positions for every layer | one shared forward pass; cells are indexed by the same `position` |
| Lens matrix count = recorded decoder state count | `len(sources) != len(matrices)` raise |
| Lens width = model width / revision match | artifact model check at apply entry |

## 15. Apply pipeline walkthrough

Example: 15 s clip, whisper-base, defaults (`top_k=5`, no transcript).

```
Fit (once, on the fit set):     per decoder layer: J_ℓ = E[∂h_final,t'/∂h_ℓ,t]

Apply:
15 s clip
  → mel padded to 30 s                    (80, 3000)
  → encoder (no grad)                     (1500, 512)
  → greedy generation                     ids: [SOT, t1 … tN]  (T ≤ max_new_tokens)
  → teacher-forced decoder pass           7 × (T, 512) hidden states
  → per layer: (H @ J_ℓᵀ) @ Eᵀ            (T, 51,865) logits
  → softmax + topk(5) per position        7 × T × 5 scored tokens
```

Output: `positions[]` (the decoder input tokens) and `layers[]`, each with
`positions[]`, each cell with `tokens[]` (`token_id`, `token`, `score`,
`probability`), plus `transcript` and `transcript_source`
(`"generated"` / `"provided"`).

## 16. Fit globally, apply per position

```
FIT (once per model, per fit-set):
  teacher-forced transcript ── per (t, t'≥t) causal pair ──►  averaged J_ℓ per decoder layer

APPLY (every clip):
  per (position t, layer ℓ):   J_ℓ · h_ℓ,t  ──►  E  ──►  top-k tokens
```

Key properties:

1. **Position-resolved readout** — every (position, layer) cell is read from
   that position's own residual stream, as in the LLM J-lens visualization.
2. **Causally correct averaging** — J_ℓ averages only pairs (t, t' ≥ t); a
   perturbation at t cannot influence earlier positions.
3. **Per-layer J, shared across positions** — per-layer comparisons trace the
   depth progression of what the decoder is "poised to say"; per-position
   differences reflect the content, not the map.

## 17. Limitations summary

| # | Limitation | Consequence |
|---|---|---|
| 1 | First-order linearization only | Error grows with distance between a position's actual trajectory and the average context |
| 2 | Averaged J erases pairwise alignment structure | The lens shows *what* is verbalizable per position, not *which audio frame* each token reads from (cross-attention remains the tool for that) |
| 3 | Fit targets are teacher-forced reference transcripts | J reflects "given this prefix, what is amplified next" — not autonomous generation dynamics |
| 4 | E applied to linearized states | Softmax probabilities are display-normalized, not calibrated |
| 5 | Frequent-token geometry | Common-word lens directions dominate rankings more often than rare ones |
| 6 | Hutchinson probes are approximate | 4–32 probes per sample leaves residual estimation noise in J |
| 7 | Model/revision-bound artifact | Lens must be refitted when the model changes |
| 8 | Read-only | No steering/ablation/patching yet (the LLM J-lens's "write" mode) |

## 18. Relation to the LLM Jacobian lens

| Design axis | LLM J-lens (Anthropic 2026) | This implementation |
|---|---|---|
| Source space | residual stream per position | decoder residual stream per position (same) |
| Target | final residual stream, positions t' ≥ t | decoder `last_hidden_state`, t' ≥ t (same) |
| Jacobian | exact VJPs averaged over (t, t', ~1000 prompts) | exact VJP structure, Hutchinson-estimated over fit transcripts |
| Readout | softmax(W_U · norm(J h)) | softmax(E · (J h)) — Whisper's LN is inside the decoder |
| Fit/apply granularity | per (position, layer) | per (position, layer) (same) |
| Write mode | steering, ablation, patching | not yet implemented |
| Substrate | one shared residual stream | encoder substrate feeds cross-attention; never lensed |

## 19. Code map

| Concern | Location |
|---|---|
| Fit / apply implementation | `Backend/app/services/jacobian_lens_service.py` (`fit_decoder_jacobian_lens` :153, `apply_decoder_jacobian_lens` :259) |
| Architecture selection (`decoder`) | `Backend/app/worker/model_adapters.py` (`jacobian_lens_architecture`) |
| Job parameter schemas (`probe_count`, `top_k`, `transcript`, `max_new_tokens`) | `Backend/app/schemas/jobs.py` (lines 96-116) |
| Job execution | `Backend/app/worker/executor.py` (`_execute_jacobian_lens_fit` / `_execute_jacobian_lens_apply`) |
| Lens repository (session-owned records) | `Backend/app/repositories/jacobian_lenses.py` |
| API routes | `Backend/app/api/routes/jobs.py`, `models.py` |
| Frontend lab / visualization | `Frontend/src/pages/JacobianLensLab.tsx`, `Frontend/src/components/visualization/JacobianLensVisualization.tsx` |
| Fit/apply unit tests (linear closed form, exact readout) | `Backend/tests/test_jacobian_lens.py` |

## References

- Gurnee, W. et al. (2026). *Verbalizable Representations Form a Global
  Workspace in Language Models.* transformer-circuits.pub — the Jacobian-lens
  construction (`J_ℓ = E[∂h_final,t'/∂h_ℓ,t]` over t' ≥ t and a prompt corpus)
  this implementation follows, applied to a speech decoder.
- Hvingelby et al. (2023). *Encoder Jacobian Lenses for Interpreting Speech
  Models.* — the earlier encoder-transport formulation this redesign supersedes.
- Rademacher/Hutchinson trace estimation — randomized probe identities used for
  the VJP-based transport estimate.
- Belinkov, Y. (2022). *Probing Classifiers: Promises, Shortcomings, and
  Advances.* Computational Linguistics, 48(1).
- Hewitt & Manning (2019). *A Structural Probe for Finding Syntax in Word
  Representations.* NAACL.
- Alain & Bengio (2017). *Understanding Intermediate Layers Using Linear
  Classifier Probes.* ICLR Workshop.
