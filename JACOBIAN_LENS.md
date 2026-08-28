# Jacobian Lens — Technical Reference

## Overview

A **Jacobian Lens** (J-Lens) is a set of linear probes fitted on each encoder layer of a frozen speech-to-text model. Each probe maps the layer's encoder representation to the model's final verbal (decoder) state, measuring *what vocabulary information each layer encodes*. Unlike the original stochastic-Jacobian method that sampled vocabulary tokens with no calibration, this implementation uses a regularised linear readout trained against teacher-forced decoder targets with held-out validation.

The name "Jacobian" comes from the original technique of computing the Jacobian of the output with respect to encoder representations — though this implementation uses a more direct fitted readout rather than numerical differentiation.

---

## Architecture Support

### Seq2Seq (Whisper Base, Whisper Large, etc.)

For encoder-decoder models, the teacher target is the **last decoder hidden state** after teacher-forcing the ground-truth transcript. Because the decoder produces one hidden state per output token, but the encoder produces one per audio frame, the targets must be **aligned to the encoder time axis**.

Alignment uses the model's own **cross-attention** — specifically the last decoder layer's attention weights over encoder frames:

```
For each selected encoder frame index j:
  target[j] = Σ_t attn[t, j] · decoder_state[t]
  target[j] /= max(Σ_t attn[t, j], 1e-6)
```

This means each encoder frame's target is the weighted average of all decoder hidden states, weighted by how much that state attended to that encoder frame. This is a soft alignment — a frame that contributes to multiple output tokens gets a blended target.

### CTC (custom CTC ASR models)

CTC models share an encoder-only architecture where the encoder output is already time-aligned with the input. The teacher target is simply the **final encoder hidden state** at each frame — no cross-attention alignment is needed. These models use `ModelKind.CTC_ASR` and expose `jacobian_lens_architecture() -> "ctc"`.

---

## Fitting Procedure

### 1. Audio Preparation

Each audio file is loaded with `librosa` at the model's native sample rate (typically 16 kHz), trimmed to `max_audio_seconds` (default 30s), and passed through the processor to produce model inputs.

### 2. Frame Subsampling

Encoder outputs can be thousands of frames long. To keep the per-sample feature count tractable, frames are subsampled to a fixed number (`frame_samples`, default 32):

```
frame_count = min(encoder_frames, frame_samples)
indices = linspace(0, encoder_frames - 1, steps=frame_count).round()
```

If the encoder already has ≤ `frame_samples` frames, all are used.

### 3. Feature Accumulation

For each training sample, the encoder-hidden sources `x` and aligned teacher targets `y` are accumulated into per-layer sufficient statistics:

- `count += N` (number of sampled frames)
- `x_sum += Σ x_i` (sum of encoder frame vectors)
- `y_sum += Σ y_i` (sum of target frame vectors)
- `xx += X^T X` (design matrix gram)
- `xy += X^T Y` (cross-covariance)

These are computed in **float64** to avoid numerical precision loss during accumulation across hundreds of samples.

### 4. Ridge Regression

Each layer's linear readout `W` is solved independently:

```
cov = xx - count · outer(x_mean, x_mean)      # centered covariance
cross_cov = xy - count · outer(x_mean, y_mean)  # centered cross-covariance
scale = mean(diag(cov))                        # feature-scale normaliser
system = cov + λ · scale · I                   # ridge-regularised system
W = solve(system, cross_cov)                   # W ∈ R^(d_enc × d_dec)
```

where `λ` is `ridge_regularization` (default 0.001). The `scale` term makes the regularisation agnostic to the absolute magnitude of encoder activations.

The readout for each encoder layer is then:

```
v_target(x) = (x - μ_x)^T W + μ_y
```

### 5. Validation Split

When ≥ 10 samples are provided, every 5th sample (indices 0, 5, 10, ...) is held out from training. This gives ~20% held-out samples. Fewer than 10 samples means no validation, and the lens is marked "exploratory" in the UI.

Validation metrics measure how well the fitted readout predicts held-out targets using two criteria:

#### Cosine Similarity

For each validation frame pair `(y_pred, y_true)`:

```
cosine = (y_pred · y_true) / (||y_pred|| · ||y_true||)
```

Averaged over all frames in all validation samples. Closer to 1.0 means the readout captures the orientation of the verbal state.

#### Top-1 Token Agreement

Both `y_pred` and `y_true` are projected through the vocabulary embedding matrix to get logits, then the argmax token is compared:

```
predicted_token_id = argmax(y_pred · W_proj^T)
teacher_token_id  = argmax(y_true · W_proj^T)
agreement = (predicted_token_id == teacher_token_id)
```

Reported as a fraction of validation frames. Values above chance (≈ 1/vocab_size ≈ 0.004% for Whisper's ~52k vocabulary) indicate that the readout encodes specific lexical information.

---

## Applying the Lens

### 1. Encoder-Only Forward Pass

During application, only the **encoder** runs — no decoder or cross-attention is needed (unlike fitting). This makes application fast. The encoder outputs are obtained via:

- **Seq2Seq**: `model.get_encoder()(input_features)`
- **CTC**: Full forward pass through the encoder-only model

### 2. Temporal Pooling

To reduce per-layer frame counts to a manageable visual grid, frames are **equally pooled** into a fixed number of buckets (`max_frames`, default 96):

```
bucket_size = ceil(encoder_frames / max_frames)
For bucket i:
  pooled_frame = mean(frames[i·bucket_size : (i+1)·bucket_size])
```

Each bucket spans `duration · start / total_frames` to `duration · end / total_frames` seconds.

### 3. Readout → Logits

```
verbal_state = pooled_frame - μ_x         # center
verbal_state = verbal_state · W            # project
verbal_state = verbal_state + μ_y          # uncenter
logits = verbal_state · W_projection^T     # vocab projection
```

### 4. Frequency Correction (PMI Correction)

A major improvement over the raw readout. The **token unigram prior** is estimated from the training transcripts using the model's own tokenizer:

```
count[token_id] = frequency in training transcripts
P(token) = (count[token_id] + smoothing) / (total_tokens + smoothing · vocab_size)
```

During application, the prior is subtracted from logits (in log space):

```
corrected_logits = logits - log P(token)
```

This is equivalent to computing **pointwise mutual information** between the encoder evidence and each token:

```
PMI(token, encoder) = log P(token | encoder) - log P(token)
                     = log P(token, encoder) - log P(token) - log P(encoder)
                     = logits - log P(token) + constant
```

The constant (log P(encoder)) is the same for all tokens and doesn't affect rankings. This correction suppresses high-frequency special tokens (like `<|en|>`, `<|endoftext|>`) that have high prior probability but carry no acoustic evidence, allowing low-frequency content words to surface when they have non-negligible encoder support.

### 5. Top-k Selection

```
top_k = min(parameters.top_k, vocab_size)
top_scores, top_indices = topk(corrected_logits, k=top_k)
probabilities = softmax(corrected_logits).gather(top_indices)
```

Both the corrected and uncorrected (raw) top-k tokens are returned for comparison, letting the UI display both views.

---

## Artifact Format

The fitted lens is saved as a `.pt` file (PyTorch `torch.save`) containing a dictionary:

| Field | Shape | Description |
|---|---|---|
| `format_version` | int | 2 (version 2 = calibrated) |
| `method` | str | `"teacher_aligned_ridge_readout"` |
| `architecture` | str | `"seq2seq"` or `"ctc"` |
| `model_id` | str | e.g. `"whisper-base"` |
| `model_revision` | str | HuggingFace model revision |
| `weights` | list[float32 Tensor] | `[L x (d_enc, d_dec)]`, one per layer |
| `source_means` | list[float32 Tensor] | `[L x d_enc]`, encoder mean per layer |
| `target_means` | list[float32 Tensor] | `[L x d_dec]`, target mean per layer |
| `token_prior_log_probs` | float32 Tensor | `[vocab_size]`, log-unigram prior (PMI correction) |
| `sample_count` | int | Total samples used |
| `training_sample_count` | int | Training samples after validation split |
| `validation_sample_count` | int | Held-out samples |
| `frame_samples` | int | Per-sample frame count |
| `ridge_regularization` | float | Ridge penalty λ |
| `quality` | dict | Per-layer validation metrics |

---

## Quality Metrics Interpretation

### Cosine Similarity per Layer

| Range | Interpretation |
|---|---|
| 0.9–1.0 | Readout nearly perfectly captures the verbal state orientation. The layer's representation is highly aligned with the final output. |
| 0.7–0.9 | Strong alignment. The layer encodes substantial linguistic information accessible via a linear transform. |
| 0.5–0.7 | Moderate alignment. Some vocabulary signal exists but is weak or noisy. |
| < 0.3 | Poor alignment. The layer likely encodes acoustic or low-level features, not vocabulary. |

### Top-1 Token Agreement

| Range | Interpretation |
|---|---|
| > 20% | The readout can reliably predict the exact vocabulary token at each frame. Strong lexical encoding. |
| 5–20% | Some lexical signal exists, well above chance, but predictions are noisy. |
| 1–5% | Weak but detectable lexical signal. |
| < 0.1% | Essentially random — the linear readout captures direction but not token identity. |

**Typical observations:** Early layers show low cosine and near-zero agreement (acoustic encoding). Middle layers show rising cosine as abstract representations form. Late layers (especially the last 2–3) often peak in both metrics. A sharp drop in the final layer can suggest the residual connection "shorts out" the encoder output.

---

## Temporal Alignment in the UI

The frontend visualises layer × time as a grid where each cell shows the rank-N token (configurable via the slider). The grid reveals:

- **Token transitions over time** — which vocabulary items are "active" at which audio interval
- **Layer progression** — whether late layers "agree" on the same content words or diverge
- **Special-token saturation** — PMI correction makes this visible by comparison between corrected and raw panels

Clicking a cell expands it to show all top-k tokens with their corrected probabilities, the raw (uncorrected) tokens if they differ, and the layer's held-out quality metrics.

---

## Limitations

1. **Linearity:** The readout assumes a linear mapping from encoder to verbal state. The true mapping is highly non-linear, so the probes capture only the *linearly accessible* component of vocabulary encoding.
2. **Teacher-forcing:** Targets are conditioned on ground-truth text (teacher forcing), not the model's free-running output. The probes measure "what the encoder knows that the *correct* decoder state needs," not "what the model actually predicts."
3. **Cross-attention averaging:** Seq2seq alignment blurs frames that contribute to multiple output tokens, smoothing fine-grained temporal resolution.
4. **Token frequency bias:** Without PMI correction, the readout is dominated by high-frequency special tokens regardless of encoder evidence. The PMI correction helps but can overcorrect if the training transcript distribution doesn't match inference.
5. **Sample efficiency:** Ridge regression with accumulating statistics is efficient but requires ≥ 2 training samples (≥ 10 for validation) and benefits from hundreds.
6. **Pooling artifacts:** Uniform temporal pooling can split a single phonetic event across two buckets or merge distinct events into one.
7. **Model-specific:** The lens is tied to a specific model revision and must be refitted when the model changes.

---

## References

- Belinkov, Y. (2022). *Probing Classifiers: Promises, Shortcomings, and Advances.* Computational Linguistics, 48(1).
- Hvingelby et al. (2023). *Encoder Jacobian Lenses for Interpreting Speech Models.* (Original stochastic-Jacobian approach.)
- Hewitt & Manning (2019). *A Structural Probe for Finding Syntax in Word Representations.* NAACL.
- Alain & Bengio (2017). *Understanding Intermediate Layers Using Linear Classifier Probes.* ICLR Workshop.