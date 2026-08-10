import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, CheckSquare, FlaskConical, RefreshCw, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { API_BASE } from "@/lib/api";
import { materializeAudio } from "@/lib/jobs";
import { listCustomModels, listJacobianLenses, type CustomModel, type JacobianLens } from "@/lib/models";
import { useJob } from "@/hooks/use-job";

interface DatasetRow {
  filename?: string;
  file?: string;
  filepath?: string;
  path?: string;
  sentence?: string;
  transcript?: string;
  text?: string;
  statement?: string;
  [key: string]: unknown;
}

interface TrainingSample {
  filename: string;
  transcript: string;
}

interface CustomDataset {
  dataset_name: string;
  formatted_name: string;
}

interface LensFitResult {
  lens: {
    lens_id: string;
    architecture: "seq2seq" | "ctc";
    layer_count: number;
    sample_count: number;
  };
}

const builtInModels = [
  { value: "whisper-base", label: "Whisper Base" },
  { value: "whisper-large", label: "Whisper Large" },
];

const baseFilename = (row: DatasetRow) => {
  const raw = row.path || row.filepath || row.file || row.filename;
  if (typeof raw !== "string") return "";
  return raw.split("/").pop()?.split("\\").pop() || raw;
};

const transcriptFor = (row: DatasetRow) => {
  const raw = row.sentence ?? row.transcript ?? row.text ?? row.statement;
  return typeof raw === "string" ? raw.trim() : "";
};

export default function JacobianLensLab() {
  const [customModels, setCustomModels] = useState<CustomModel[]>([]);
  const [customDatasets, setCustomDatasets] = useState<CustomDataset[]>([]);
  const [model, setModel] = useState("whisper-base");
  const [dataset, setDataset] = useState("common-voice");
  const [rows, setRows] = useState<DatasetRow[]>([]);
  const [selectedFilenames, setSelectedFilenames] = useState<string[]>([]);
  const [sampleLimit, setSampleLimit] = useState("50");
  const [isLoadingDataset, setIsLoadingDataset] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lenses, setLenses] = useState<JacobianLens[]>([]);
  const [completedLensId, setCompletedLensId] = useState<string | null>(null);
  const fitJob = useJob<LensFitResult>();

  const modelOptions = useMemo(() => [
    ...builtInModels,
    ...customModels
      .filter((item) => item.status === "ready" && item.capabilities.includes("jacobian_lens_fit"))
      .map((item) => ({ value: item.model_id, label: item.hf_repo })),
  ], [customModels]);

  const trainableRows = useMemo<TrainingSample[]>(() => rows
    .map((row) => ({ filename: baseFilename(row), transcript: transcriptFor(row) }))
    .filter((row) => Boolean(row.filename && row.transcript)), [rows]);

  const selectedSamples = useMemo(
    () => trainableRows.filter((sample) => selectedFilenames.includes(sample.filename)),
    [selectedFilenames, trainableRows],
  );

  const refreshLenses = useCallback(async () => {
    try {
      setLenses(await listJacobianLenses(model));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load fitted lenses");
    }
  }, [model]);

  useEffect(() => {
    listCustomModels().then(setCustomModels).catch(() => setCustomModels([]));
    fetch(`${API_BASE}/upload/dataset/list`, { credentials: "include" })
      .then((response) => response.ok ? response.json() : { datasets: [] })
      .then((payload) => setCustomDatasets(payload.datasets || []))
      .catch(() => setCustomDatasets([]));
  }, []);

  useEffect(() => {
    if (!modelOptions.some((item) => item.value === model)) setModel("whisper-base");
  }, [model, modelOptions]);

  useEffect(() => {
    const controller = new AbortController();
    setIsLoadingDataset(true);
    setError(null);
    setRows([]);
    setSelectedFilenames([]);
    fetch(`${API_BASE}/${encodeURIComponent(dataset)}/metadata`, {
      credentials: "include",
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Could not load dataset metadata (${response.status})`);
        return response.json() as Promise<DatasetRow[]>;
      })
      .then(setRows)
      .catch((caught) => {
        if ((caught as Error).name !== "AbortError") setError(caught instanceof Error ? caught.message : "Could not load dataset metadata");
      })
      .finally(() => setIsLoadingDataset(false));
    return () => controller.abort();
  }, [dataset]);

  useEffect(() => { void refreshLenses(); }, [refreshLenses]);

  const selectFirstSamples = () => {
    const parsed = Number.parseInt(sampleLimit, 10);
    const limit = Number.isFinite(parsed) ? Math.max(2, Math.min(parsed, 200)) : 50;
    setSelectedFilenames(trainableRows.slice(0, limit).map((sample) => sample.filename));
  };

  const toggleSample = (filename: string) => {
    setSelectedFilenames((current) => current.includes(filename)
      ? current.filter((item) => item !== filename)
      : current.length < 200 ? [...current, filename] : current,
    );
  };

  const fitLens = async () => {
    if (selectedSamples.length < 2) return;
    setError(null);
    setCompletedLensId(null);
    try {
      // Dataset files become session-owned audio assets before fitting.  This
      // keeps the long worker job inside the same ownership boundary as jobs.
      const materialized = [] as { audio_id: string; transcript: string }[];
      for (let index = 0; index < selectedSamples.length; index += 8) {
        const batch = selectedSamples.slice(index, index + 8);
        const assets = await Promise.all(batch.map((sample) => materializeAudio(dataset, sample.filename)));
        materialized.push(...assets.map((asset, offset) => ({
          audio_id: asset.audio_id,
          transcript: batch[offset].transcript,
        })));
      }
      const result = await fitJob.start({
        operation: "jacobian_lens_fit",
        model,
        audio_ids: materialized.map((sample) => sample.audio_id),
        parameters: {
          samples: materialized,
          probe_count: 4,
          max_audio_seconds: 30,
        },
      });
      setCompletedLensId(result.lens.lens_id);
      await refreshLenses();
    } catch (caught) {
      if ((caught as Error).name !== "AbortError") setError(caught instanceof Error ? caught.message : "Lens fitting failed");
    }
  };

  return (
    <main className="min-h-screen bg-background">
      <header className="h-14 border-b bg-panel-header flex items-center justify-between px-6">
        <div className="flex items-center gap-3">
          <FlaskConical className="h-5 w-5 text-primary" />
          <div>
            <h1 className="text-sm font-semibold">J-Lens Lab</h1>
            <p className="text-[11px] text-muted-foreground">Fit and manage encoder Jacobian lenses for supported speech-to-text models.</p>
          </div>
        </div>
        <Button asChild size="sm" variant="outline" className="text-xs"><Link to="/"><ArrowLeft className="mr-1 h-3.5 w-3.5" />Analysis workspace</Link></Button>
      </header>

      <div className="mx-auto grid max-w-7xl gap-5 p-6 lg:grid-cols-[minmax(0,1fr)_340px]">
        <section className="space-y-5">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">1. Choose a model and dataset</CardTitle>
              <CardDescription>The dataset must contain an audio filename plus a transcript field (`sentence`, `transcript`, `text`, or RAVDESS `statement`).</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2"><Label>Speech-to-text model</Label><Select value={model} onValueChange={setModel}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent>{modelOptions.map((item) => <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>)}</SelectContent></Select></div>
              <div className="space-y-2"><Label>Transcript dataset</Label><Select value={dataset} onValueChange={setDataset}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="common-voice">Common Voice</SelectItem><SelectItem value="ravdess">RAVDESS</SelectItem>{customDatasets.map((item) => <SelectItem key={item.formatted_name} value={item.formatted_name}>{item.dataset_name}</SelectItem>)}</SelectContent></Select></div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between gap-3"><div><CardTitle className="text-base">2. Select fitting samples</CardTitle><CardDescription>{isLoadingDataset ? "Loading metadata…" : `${trainableRows.length} transcript-bearing samples available; select 2–200.`}</CardDescription></div><Badge variant="outline">{selectedSamples.length} selected</Badge></div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap items-end gap-2"><div className="space-y-1"><Label htmlFor="sample-limit" className="text-xs">Select first</Label><Input id="sample-limit" value={sampleLimit} onChange={(event) => setSampleLimit(event.target.value)} className="h-8 w-24" inputMode="numeric" /></div><Button size="sm" variant="outline" onClick={selectFirstSamples} disabled={!trainableRows.length || fitJob.isRunning}><CheckSquare className="mr-1 h-3.5 w-3.5" />Select samples</Button><Button size="sm" variant="ghost" onClick={() => setSelectedFilenames([])} disabled={!selectedFilenames.length || fitJob.isRunning}><Square className="mr-1 h-3.5 w-3.5" />Clear</Button></div>
              {!isLoadingDataset && !trainableRows.length && <p className="text-sm text-muted-foreground">This dataset has no usable transcript field, so it cannot fit a J-lens.</p>}
              <div className="max-h-[420px] overflow-auto rounded border">
                {trainableRows.map((sample) => <label key={sample.filename} className="flex cursor-pointer items-start gap-3 border-b p-3 last:border-b-0 hover:bg-muted/40"><Checkbox checked={selectedFilenames.includes(sample.filename)} onCheckedChange={() => toggleSample(sample.filename)} disabled={fitJob.isRunning || (!selectedFilenames.includes(sample.filename) && selectedFilenames.length >= 200)} /><span className="min-w-0"><span className="block font-mono text-xs">{sample.filename}</span><span className="block truncate text-xs text-muted-foreground">{sample.transcript}</span></span></label>)}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle className="text-base">3. Fit the lens</CardTitle><CardDescription>The model remains frozen. The worker estimates an average transport matrix for each encoder layer.</CardDescription></CardHeader>
            <CardContent className="flex flex-wrap items-center gap-3"><Button onClick={() => void fitLens()} disabled={selectedSamples.length < 2 || fitJob.isRunning}>{fitJob.isRunning ? "Fitting encoder lenses…" : `Fit with ${selectedSamples.length} samples`}</Button>{fitJob.isRunning && <Button variant="outline" onClick={() => void fitJob.cancel()}>Cancel</Button>}{fitJob.isRunning && <span className="text-sm text-muted-foreground">{fitJob.status?.progress.message || "Preparing worker…"}</span>}{(error || fitJob.error) && <span className="text-sm text-destructive">{error || fitJob.error}</span>}{completedLensId && <span className="text-sm text-emerald-700">Lens {completedLensId} is ready. Open the analysis workspace and choose J-Lens.</span>}</CardContent>
          </Card>
        </section>

        <aside><Card><CardHeader><div className="flex items-center justify-between"><CardTitle className="text-base">Saved lenses</CardTitle><Button size="icon" variant="ghost" className="h-8 w-8" onClick={() => void refreshLenses()}><RefreshCw className="h-3.5 w-3.5" /></Button></div><CardDescription>Available for {model} in this session.</CardDescription></CardHeader><CardContent className="space-y-2">{!lenses.length ? <p className="text-sm text-muted-foreground">No lenses fitted yet.</p> : lenses.map((lens) => <div key={lens.lens_id} className="rounded border p-2 text-xs"><div className="flex items-center justify-between gap-2"><span className="font-mono">{lens.lens_id.slice(-8)}</span><Badge variant={lens.status === "ready" ? "secondary" : "outline"}>{lens.status}</Badge></div><p className="mt-1 text-muted-foreground">{lens.architecture || "ASR"} · {lens.layer_count ?? "?"} layers · {lens.sample_count} samples</p>{lens.error && <p className="mt-1 text-destructive">{lens.error}</p>}</div>)}</CardContent></Card></aside>
      </div>
    </main>
  );
}
