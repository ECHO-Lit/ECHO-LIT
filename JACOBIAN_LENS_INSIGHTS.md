# Jacobian Lens — Insights Guide

> **Companion to** [`JACOBIAN_LENS.md`](./JACOBIAN_LENS.md) (technical reference).
> This document is about *use*: what you can actually learn from the
> `(position × layer)` readout the decoder-only J-Lens produces, with concrete
> examples. Layer numbering follows the implementation: index `0` = the decoder
> embedding output (UI shows "Layer 1"), the **highest index = the last decoder
> block = closest to the output**. For whisper-base there are 7 lens layers
> (`0..6`); the final pre-logit state itself is the fit target and is never a
> readout layer.

---

## What one apply gives you

After fitting a lens and running an apply job, each result contains:

- `positions[]` — the decoder's input tokens (generated, or a provided
  transcript), the "spine" of the grid;
- `layers[]` — one entry per decoder layer, each with a `positions[]` array;
- each cell holds the top-k tokens with `score` (logit scale) and
  `probability` (display softmax).

Read a **column** (one position, all layers) for a depth trajectory; read a
**row** (one layer, all positions) for the model's state along the transcript
at a fixed depth.

---

## 1. Where the transcription decision crystallizes (layer trajectory)

Pick the position immediately before a spoken word and read down the column.
Early layers rank generic frequent tokens ("the", "and"), mid layers surface
syntactic candidates, and the top rows converge on the word that is actually
spoken next.

**Insight:** the row where convergence happens is the depth at which acoustic
evidence gets integrated into the verbal decision. On whisper-base this
typically sits in the upper-middle band and shifts toward earlier layers as
noise increases. A column that *never* converges on the spoken word marks a
position where the model transcribed against its own internal preference.

## 2. The SOT position is an audio-only workspace readout

At the `<|startoftranscript|>` position the decoder has consumed **no**
transcript tokens yet — everything in the residual stream arrived through
cross-attention from the audio. Its top tokens are the model's acoustic summary
of the clip: likely first words, "you" (Whisper's trained silence convention),
or artifacts of background speech.

**Insight:** this is the closest available reading of "what did the audio
alone put into the workspace" — before any language-model context from the
transcript has been written.

## 3. LM-prior vs. acoustic evidence, per position

Compare mid-layer vs. late-layer readouts at the same position:

- Both agree on a common collocation ("…honor of the first…") → the token is
  being driven by the **language-model prior**; the audio could be anything.
- Mid layers are generic but the top rows flip to an unusual word → that word
  came from the **audio**.

Sweeping this disagreement across a transcript quantifies, per word, how much
of the transcription is prior vs. evidence — useful for finding where Whisper
is "just talking" rather than listening.

## 4. Seeing hallucination and repetition before it is spoken

Whisper's best-known failure under silence is the repetition loop ("the the
the…"). At a silence-padded position whose input token is unrelated, mid-layer
readouts often already rank the loop tokens highly **before** generation
commits to them.

**Insight:** the grid is a leading indicator — the loop direction is visible
in the residual stream one or more positions before it appears in the output
transcript. The same logic surfaces confabulated words under noise.

## 5. Homophone and ambiguity competition

At positions over ambiguous audio, the top-5 cell shows the contest directly:
"their" / "there" / "they're" with close scores.

**Insight:** a wide top-k spread means the audio under-determined the choice
and the model fell back to prior; a narrow spread means the acoustics decided
it. Sorting transcript positions by top-k score gap produces an ambiguity map
of the clip.

## 6. Endpoint and special-token behavior

Near the end of real speech content, readouts drift toward `EOT`, `.`, or
"thank you" (Whisper's trained silence close). You can see **when** the model
starts planning to stop, at **which layer** the stop direction first appears,
and how much silence padding accelerates it.

**Insight:** premature stop-direction activation in upper layers while speech
is still ongoing is a measurable "early cutoff" signature; its absence on
run-on audio explains runaway transcriptions.

## 7. Layer-band profiling (the "workspace range")

Aggregate top-1 agreement with the actual next transcript token, per layer,
across a dataset. The result is a curve: near-zero at Layers 1–2, peaking in
the mid-upper band, then shifting as output selection takes over — the Whisper
analog of the LLM J-lens finding that early-layer readouts are noise and the
final layers flip to "motor mode".

**Insight:** the band where the curve peaks is the depth worth reading for any
downstream audit; layers outside it can be ignored for insight work.

---

## How to read a specific result (worked example)

15 s clip, whisper-base, generated transcript `positions = [SOT, "a", "laudable",
"regard", …]`, `layers = 0..6`, `top_k = 5`:

| Reading | Cells to inspect | What to conclude |
|---|---|---|
| Acoustic gist | Layer 7 (index 6) at SOT | top tokens ≈ how the clip opens |
| Decision depth | column at position of "laudable" | the row where "laudable" enters top-5 |
| Prior vs. evidence | mid vs. top rows at each content word | disagreement ⇒ audio-driven |
| Ambiguity | top-5 spread at function words | small gap ⇒ acoustics decided |
| Hallucination risk | any position where input ≠ audio-relevant yet loop tokens rank high | loop is being prepared |
| Stop planning | column where `EOT`/`.` first enters top-5 | endpoint decision point |

---

## Interpretation contract

- Scores are first-order-approximated logits; `probability` is their softmax,
  normalized for display, **not calibrated confidence**.
- Top-k per cell is an ordered **ranking of verbalizable content**, not a
  phrase — no language-model coherence between neighboring cells.
- The lens reads the **decoder's** residual stream. It does not tell you which
  audio frame supports a token — that alignment question belongs to
  cross-attention analysis (the Attention tab).
- Readouts at the lowest layers are frequently noise; distrust them.
- Scores live on the model's own logit scale via the frozen output projection
  `E`, but the map `J_ℓ` is an *average over contexts*: a cell shows what the
  decoder is **disposed to say on average**, not a guaranteed continuation.

## Quick limitations checklist

1. First-order linearization — trust rankings more than absolute scores.
2. Teacher-forced fit targets — J reflects "given this prefix", not free-running generation dynamics.
3. Probe noise — 4–32 Rademacher probes leave residual estimation error in J.
4. Frequent-token geometry — common words dominate rankings more often than rare ones.
5. Model-bound artifact — refit whenever the model or revision changes.

## See also

- [`JACOBIAN_LENS.md`](./JACOBIAN_LENS.md) — fit/apply math, artifact format, redesign log.
- Gurnee, W. et al. (2026). *Verbalizable Representations Form a Global
  Workspace in Language Models.* transformer-circuits.pub — the construction
  this implementation follows; its Figures 3–10 are the LLM analogs of the
  readings above.
