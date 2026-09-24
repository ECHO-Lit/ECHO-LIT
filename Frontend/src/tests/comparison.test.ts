/**
 * Model & dataset comparison — the pure alignment and scoring logic.
 *
 * The async runner (`runComparison`) is a thin shell over the network; the part
 * worth testing is `buildComparison` and its helpers, where a wrong join or a
 * wrong metric direction would silently mislead a differential diagnosis.
 */
import { describe, it, expect } from "vitest";

import {
  buildComparison,
  groundTruthFor,
  incompatibilityReason,
  metricForItem,
  modelKind,
  predictionText,
  type SideRun,
} from "@/lib/comparison";
import type { CustomModel } from "@/lib/models";

const run = (label: string, model: string, dataset: string, items: SideRun["items"]): SideRun => ({
  spec: { model, dataset }, label, items,
});

describe("ground truth and prediction extraction", () => {
  it("reads the right truth column per task kind", () => {
    expect(groundTruthFor("asr", { sentence: "hello world" })).toBe("hello world");
    expect(groundTruthFor("classification", { emotion: "angry" })).toBe("angry");
    // "unknown" is Common Voice's not-annotated marker, not a label.
    expect(groundTruthFor("classification", { emotion: "unknown" })).toBeNull();
    expect(groundTruthFor("asr", { note: "x" })).toBeNull();
  });

  it("reduces varied prediction shapes to comparable text", () => {
    expect(predictionText("asr", { text: "a b c" })).toBe("a b c");
    expect(predictionText("asr", "plain string")).toBe("plain string");
    expect(predictionText("classification", { predicted_emotion: "happy" })).toBe("happy");
    expect(predictionText("classification", { predicted_label: "sad", confidence: 0.9 })).toBe("sad");
    expect(predictionText("asr", null)).toBe("");
  });
});

describe("per-item metric", () => {
  it("scores ASR as WER (lower is better) and returns null without truth", () => {
    expect(metricForItem("asr", "the cat sat", "the cat sat")).toBe(0);
    expect(metricForItem("asr", "the cat sat", null)).toBeNull();
    expect(metricForItem("asr", "the dog sat", "the cat sat")).toBeGreaterThan(0);
  });

  it("scores classification as 1/0 correctness, case-insensitively", () => {
    expect(metricForItem("classification", "Angry", "angry")).toBe(1);
    expect(metricForItem("classification", "happy", "angry")).toBe(0);
  });
});

describe("buildComparison — two models, one dataset (the acceptance path)", () => {
  const a = run("whisper-base", "whisper-base", "ravdess", [
    { filename: "1.wav", groundTruth: "the cat sat", prediction: "the cat sat" },   // WER 0
    { filename: "2.wav", groundTruth: "a red car", prediction: "a red car" },        // WER 0
  ]);
  const b = run("whisper-large", "whisper-large", "ravdess", [
    { filename: "2.wav", groundTruth: "a red car", prediction: "a red car" },        // WER 0
    { filename: "1.wav", groundTruth: "the cat sat", prediction: "the bat sat" },    // WER 1/3
  ]);

  it("aligns items by filename regardless of order", () => {
    const result = buildComparison("asr", "models", a, b);
    const one = result.items.find((i) => i.filename === "1.wav")!;
    expect(one.a?.prediction).toBe("the cat sat");
    expect(one.b?.prediction).toBe("the bat sat");
    expect(one.delta).toBeGreaterThan(0); // B is worse on this file
  });

  it("computes an aggregate difference and names the better model", () => {
    const result = buildComparison("asr", "models", a, b);
    expect(result.a.primary).toBeCloseTo(0, 5);       // base: perfect on both
    expect(result.b.primary).toBeGreaterThan(0);       // large: one error
    expect(result.primaryDelta).toBeGreaterThan(0);
    expect(result.betterSide).toBe("a");               // lower WER wins
  });

  it("reports an agreement rate over aligned items", () => {
    const result = buildComparison("asr", "models", a, b);
    expect(result.agreementRate).toBeCloseTo(0.5, 5); // agree on 2.wav, differ on 1.wav
  });

  it("flags the better classification side by higher accuracy", () => {
    const ca = run("m1", "m1", "ravdess", [
      { filename: "1.wav", groundTruth: "angry", prediction: "angry" },
      { filename: "2.wav", groundTruth: "happy", prediction: "sad" },
    ]);
    const cb = run("m2", "m2", "ravdess", [
      { filename: "1.wav", groundTruth: "angry", prediction: "angry" },
      { filename: "2.wav", groundTruth: "happy", prediction: "happy" },
    ]);
    const result = buildComparison("classification", "models", ca, cb);
    expect(result.a.primary).toBeCloseTo(0.5, 5);
    expect(result.b.primary).toBeCloseTo(1.0, 5);
    expect(result.betterSide).toBe("b"); // higher accuracy wins
  });
});

describe("buildComparison — two datasets, one model", () => {
  it("does not join across datasets and lists both sides' items", () => {
    const a = run("common-voice", "whisper-base", "common-voice", [
      { filename: "cv1.wav", groundTruth: "hello", prediction: "hello" },
    ]);
    const b = run("ravdess", "whisper-base", "ravdess", [
      { filename: "rv1.wav", groundTruth: "kids talking", prediction: "kids talking" },
    ]);
    const result = buildComparison("asr", "datasets", a, b);
    expect(result.items).toHaveLength(2);
    expect(result.items.every((i) => (i.a && !i.b) || (i.b && !i.a))).toBe(true);
    expect(result.agreementRate).toBeNull(); // no per-item join
  });
});

describe("incompatibilityReason — the rejection-with-guidance flow", () => {
  const customs: CustomModel[] = [];

  it("refuses two models of different task types", () => {
    const reason = incompatibilityReason("models", { model: "whisper-base", dataset: "ravdess" }, { model: "wav2vec2", dataset: "ravdess" }, customs);
    expect(reason).toMatch(/different tasks/i);
  });

  it("refuses two models on different datasets", () => {
    const reason = incompatibilityReason("models", { model: "whisper-base", dataset: "ravdess" }, { model: "whisper-large", dataset: "common-voice" }, customs);
    expect(reason).toMatch(/same dataset/i);
  });

  it("refuses comparing a model with itself", () => {
    expect(incompatibilityReason("models", { model: "whisper-base", dataset: "ravdess" }, { model: "whisper-base", dataset: "ravdess" }, customs)).toMatch(/two different models/i);
  });

  it("requires the same model across two datasets", () => {
    expect(incompatibilityReason("datasets", { model: "whisper-base", dataset: "ravdess" }, { model: "whisper-large", dataset: "common-voice" }, customs)).toMatch(/same model/i);
  });

  it("accepts a valid two-model selection", () => {
    expect(incompatibilityReason("models", { model: "whisper-base", dataset: "ravdess" }, { model: "whisper-large", dataset: "ravdess" }, customs)).toBeNull();
  });
});

describe("modelKind", () => {
  it("classifies built-ins and custom models", () => {
    expect(modelKind("whisper-base")).toBe("asr");
    expect(modelKind("wav2vec2")).toBe("classification");
    const custom: CustomModel[] = [
      { model_id: "c1", hf_repo: "x/y", status: "ready", kind: "audio_classification", capabilities: [], created_at: "" },
    ];
    expect(modelKind("c1", custom)).toBe("classification");
    expect(modelKind("unknown-model")).toBeNull();
  });
});
