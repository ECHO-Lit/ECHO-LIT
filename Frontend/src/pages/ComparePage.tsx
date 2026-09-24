/**
 * Model & dataset comparison lab (standalone route, like the J-Lens lab).
 *
 * Two models on one dataset, or one model on two datasets. It orchestrates the
 * existing `prediction` job on each side and hands the aligned results to
 * `ComparisonView`. Incompatible selections are refused *before* any compute,
 * with the reason shown inline — the use case's "mismatched or incomparable
 * selections are rejected with guidance" flow.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { AlertTriangle, ArrowLeft, GitCompareArrows, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

import { API_BASE } from "@/lib/api";
import { listCustomModels, type CustomModel } from "@/lib/models";
import {
  incompatibilityReason,
  runComparison,
  type CompareMode,
  type ComparisonResult,
  type RunProgress,
} from "@/lib/comparison";
import { ComparisonView } from "@/components/comparison/ComparisonView";

const BUILTIN_MODELS = [
  { value: "whisper-base", label: "Whisper Base (speech-to-text)" },
  { value: "whisper-large", label: "Whisper Large (speech-to-text)" },
  { value: "wav2vec2", label: "Wav2Vec2 (emotion)" },
];

const BUILTIN_DATASETS = [
  { value: "common-voice", label: "Common Voice" },
  { value: "ravdess", label: "RAVDESS" },
  { value: "l2-arctic", label: "L2-ARCTIC" },
  { value: "saa", label: "SAA" },
];

interface CustomDataset {
  dataset_name: string;
  formatted_name: string;
}

const phaseLabel: Record<RunProgress["phase"], string> = {
  metadata: "Loading dataset",
  materialising: "Preparing audio",
  predicting: "Running predictions",
  done: "Done",
};

export default function ComparePage() {
  const [customModels, setCustomModels] = useState<CustomModel[]>([]);
  const [customDatasets, setCustomDatasets] = useState<CustomDataset[]>([]);

  const [mode, setMode] = useState<CompareMode>("models");
  const [modelA, setModelA] = useState("whisper-base");
  const [modelB, setModelB] = useState("whisper-large");
  const [datasetA, setDatasetA] = useState("common-voice");
  const [datasetB, setDatasetB] = useState("ravdess");
  const [maxItems, setMaxItems] = useState("15");

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ComparisonResult | null>(null);

  useEffect(() => {
    listCustomModels().then((models) => setCustomModels(models.filter((m) => m.status === "ready"))).catch(() => setCustomModels([]));
    fetch(`${API_BASE}/upload/dataset/list`, { credentials: "include" })
      .then((r) => (r.ok ? r.json() : { datasets: [] }))
      .then((payload) => setCustomDatasets(payload.datasets || []))
      .catch(() => setCustomDatasets([]));
  }, []);

  const modelOptions = useMemo(
    () => [...BUILTIN_MODELS, ...customModels.map((m) => ({ value: m.model_id, label: `${m.hf_repo} (custom)` }))],
    [customModels],
  );
  const datasetOptions = useMemo(
    () => [...BUILTIN_DATASETS, ...customDatasets.map((d) => ({ value: d.formatted_name, label: `${d.dataset_name} (custom)` }))],
    [customDatasets],
  );

  // The request the current selection describes, per mode.
  const request = useMemo(() => {
    const items = Math.max(1, Math.min(Number.parseInt(maxItems, 10) || 15, 100));
    return mode === "models"
      ? { mode, a: { model: modelA, dataset: datasetA }, b: { model: modelB, dataset: datasetA }, maxItems: items }
      : { mode, a: { model: modelA, dataset: datasetA }, b: { model: modelA, dataset: datasetB }, maxItems: items };
  }, [mode, modelA, modelB, datasetA, datasetB, maxItems]);

  const reason = useMemo(
    () => incompatibilityReason(request.mode, request.a, request.b, customModels),
    [request, customModels],
  );

  const run = useCallback(async () => {
    setRunning(true);
    setError(null);
    setResult(null);
    setProgress("Starting…");
    const controller = new AbortController();
    try {
      const value = await runComparison(request, customModels, {
        signal: controller.signal,
        onProgress: (p) => {
          const job = p.job?.progress;
          setProgress(`Side ${p.side.toUpperCase()}: ${phaseLabel[p.phase]}${job ? ` (${job.current}/${job.total})` : "…"}`);
        },
      });
      setResult(value);
    } catch (caught) {
      if ((caught as Error).name !== "AbortError") setError(caught instanceof Error ? caught.message : "Comparison failed");
    } finally {
      setRunning(false);
      setProgress(null);
    }
  }, [request, customModels]);

  return (
    <main className="min-h-screen bg-background">
      <header className="h-14 border-b bg-panel-header flex items-center justify-between px-6">
        <div className="flex items-center gap-3">
          <GitCompareArrows className="h-5 w-5 text-primary" />
          <div>
            <h1 className="text-sm font-semibold">Comparison Lab</h1>
            <p className="text-[11px] text-muted-foreground">Run two models on one dataset, or one model on two datasets, and see the differences side by side.</p>
          </div>
        </div>
        <Button asChild size="sm" variant="outline" className="text-xs"><Link to="/"><ArrowLeft className="mr-1 h-3.5 w-3.5" />Analysis workspace</Link></Button>
      </header>

      <div className="mx-auto grid max-w-6xl gap-5 p-6 lg:grid-cols-[360px_minmax(0,1fr)]">
        <section className="space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">1. What to compare</CardTitle>
              <CardDescription>Choose the comparison, then the two sides.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-1 rounded-md border border-border p-1">
                {(["models", "datasets"] as const).map((value) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setMode(value)}
                    className={`rounded px-2 py-1.5 text-xs font-medium transition-colors ${
                      mode === value ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"
                    }`}
                  >
                    {value === "models" ? "Two models" : "Two datasets"}
                  </button>
                ))}
              </div>

              {mode === "models" ? (
                <>
                  <Field label="Model A"><ModelSelect value={modelA} onChange={setModelA} options={modelOptions} /></Field>
                  <Field label="Model B"><ModelSelect value={modelB} onChange={setModelB} options={modelOptions} /></Field>
                  <Field label="Dataset (shared)"><DatasetSelect value={datasetA} onChange={setDatasetA} options={datasetOptions} /></Field>
                </>
              ) : (
                <>
                  <Field label="Model (shared)"><ModelSelect value={modelA} onChange={setModelA} options={modelOptions} /></Field>
                  <Field label="Dataset A"><DatasetSelect value={datasetA} onChange={setDatasetA} options={datasetOptions} /></Field>
                  <Field label="Dataset B"><DatasetSelect value={datasetB} onChange={setDatasetB} options={datasetOptions} /></Field>
                </>
              )}

              <Field label="Files per side (1–100)">
                <Input
                  type="number"
                  min={1}
                  max={100}
                  value={maxItems}
                  onChange={(e) => setMaxItems(e.target.value)}
                  className="h-8"
                />
              </Field>

              {reason && (
                <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600 dark:text-amber-500" />
                  <p className="text-[11px] leading-relaxed">{reason}</p>
                </div>
              )}

              <Button onClick={run} disabled={running || !!reason} className="w-full" size="sm">
                {running ? (<><Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />{progress ?? "Running…"}</>) : "Run comparison"}
              </Button>
              <p className="text-[11px] text-muted-foreground">
                Each side runs a prediction over the first N files. On CPU this can take a minute or two per side.
              </p>
            </CardContent>
          </Card>
        </section>

        <section className="space-y-4">
          <h2 className="sr-only">Comparison result</h2>
          {error && (
            <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-destructive" />
              <p className="text-[11px] leading-relaxed">{error}</p>
            </div>
          )}
          {!error && !result && !running && (
            <Card><CardContent className="flex h-48 items-center justify-center text-center text-xs text-muted-foreground">
              Choose two sides and run a comparison to see aligned predictions and the metric difference here.
            </CardContent></Card>
          )}
          {running && !result && (
            <Card><CardContent className="flex h-48 flex-col items-center justify-center gap-2 text-xs text-muted-foreground">
              <Loader2 className="h-5 w-5 animate-spin" />
              {progress}
            </CardContent></Card>
          )}
          {result && <ComparisonView result={result} />}
        </section>
      </div>
    </main>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs">{label}</Label>
      {children}
    </div>
  );
}

function ModelSelect({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
      <SelectContent>{options.map((o) => <SelectItem key={o.value} value={o.value} className="text-xs">{o.label}</SelectItem>)}</SelectContent>
    </Select>
  );
}

function DatasetSelect({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
      <SelectContent>{options.map((o) => <SelectItem key={o.value} value={o.value} className="text-xs">{o.label}</SelectItem>)}</SelectContent>
    </Select>
  );
}
