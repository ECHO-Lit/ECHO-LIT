import { useCallback, useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Trash2 } from "lucide-react";
import { firstJobResult, resolveAudioId } from "@/lib/jobs";
import { listJacobianLenses, deleteJacobianLens, type JacobianLens } from "@/lib/models";
import { useJob } from "@/hooks/use-job";

interface AudioReference {
  audio_id?: string;
  file_id?: string;
  filename?: string;
}

interface LensToken {
  token_id: number;
  token: string;
  score: number;
  probability?: number;
}

interface LensFrame {
  start_time: number;
  end_time: number;
  tokens: LensToken[];
  raw_tokens?: LensToken[];
}

interface LensLayer {
  layer: number;
  quality?: LensLayerQuality;
  frames: LensFrame[];
}

interface LensLayerQuality {
  layer: number;
  validation_frames: number;
  cosine_similarity: number | null;
  top1_agreement: number | null;
}

interface JacobianLensResult {
  lens_id: string;
  architecture: "seq2seq" | "ctc";
  duration_seconds: number;
  method: string;
  prior_correction?: boolean;
  quality?: { layers?: LensLayerQuality[] };
  layers: LensLayer[];
}

interface JacobianLensVisualizationProps {
  selectedFile?: AudioReference | string | null;
  model?: string;
  dataset?: string;
  originalDataset?: string;
}

export const JacobianLensVisualization = ({
  selectedFile,
  model,
  dataset,
  originalDataset,
}: JacobianLensVisualizationProps) => {
  const [lenses, setLenses] = useState<JacobianLens[]>([]);
  const [lensId, setLensId] = useState("");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [result, setResult] = useState<JacobianLensResult | null>(null);
  const [selectedCell, setSelectedCell] = useState<{ layer: number; frame: LensFrame } | null>(null);
  const [topK, setTopK] = useState(0);
  const [hideStuckLayers, setHideStuckLayers] = useState(true);
  const lensJob = useJob<unknown>();

  const visibleLayers = useMemo(() => {
    if (!result || !hideStuckLayers) return result?.layers || [];
    return (result.layers || []).filter(
      (layer) => (layer.quality?.cosine_similarity ?? 1) >= 0.3,
    );
  }, [result, hideStuckLayers]);

  const readyLenses = useMemo(
    () => lenses.filter((lens) => lens.status === "ready" && lens.format_version === 2),
    [lenses],
  );
  const hasLegacyLenses = useMemo(
    () => lenses.some((lens) => lens.status === "ready" && lens.format_version !== 2),
    [lenses],
  );

  const refreshLenses = useCallback(async () => {
    if (!model) return;
    setLoadError(null);
    try {
      const nextLenses = await listJacobianLenses(model);
      setLenses(nextLenses);
      setLensId((current) =>
        nextLenses.some((lens) => lens.status === "ready" && lens.lens_id === current)
          ? current
          : nextLenses.find((lens) => lens.status === "ready")?.lens_id || "",
      );
    } catch (error) {
      setLenses([]);
      setLensId("");
      setLoadError(error instanceof Error ? error.message : "Could not load fitted lenses");
    }
  }, [model]);

  useEffect(() => {
    setResult(null);
    setSelectedCell(null);
    void refreshLenses();
  }, [refreshLenses]);

  const analyze = async () => {
    if (!model || !selectedFile || !lensId) return;
    try {
      setLoadError(null);
      setSelectedCell(null);
      const audioId = await resolveAudioId(selectedFile, originalDataset || dataset);
      const jobResult = await lensJob.start({
        operation: "jacobian_lens_apply",
        model,
        audio_ids: [audioId],
        parameters: { lens_id: lensId },
      });
      setResult(firstJobResult<JacobianLensResult>(jobResult));
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setLoadError(error instanceof Error ? error.message : "J-lens analysis failed");
    }
  };

  const deleteLens = async () => {
    if (!lensId) return;
    try {
      await deleteJacobianLens(lensId);
      await refreshLenses();
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Failed to delete lens");
    }
  };

  if (!selectedFile) {
    return <p className="text-xs text-muted-foreground p-3">Select an audio file to inspect encoder-layer token readouts.</p>;
  }

  return (
    <div className="space-y-3">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs">Encoder Jacobian Lens</CardTitle>
          <p className="text-[11px] text-muted-foreground">
            Calibrated, teacher-aligned vocabulary evidence from each encoder layer. It is not a transcript.
          </p>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex items-center gap-2">
                <Select value={lensId} onValueChange={setLensId} disabled={!readyLenses.length || lensJob.isRunning}>
                  <SelectTrigger className="h-8 text-xs flex-1">
                    <SelectValue placeholder="Choose a fitted lens" />
                  </SelectTrigger>
                  <SelectContent>
                    {readyLenses.map((lens) => (
                      <SelectItem key={lens.lens_id} value={lens.lens_id} className="text-xs">
                        {lens.architecture || "ASR"} · {lens.layer_count ?? "?"} layers · {lens.sample_count} samples
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {lensId && <Button size="icon" variant="ghost" className="h-8 w-8 shrink-0 text-destructive hover:text-red-700" title="Delete lens" onClick={() => void deleteLens()}>
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>}
                <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => void refreshLenses()} disabled={lensJob.isRunning}>
                  Refresh
                </Button>
                <Button size="sm" className="h-8 text-xs" onClick={() => void analyze()} disabled={!lensId || lensJob.isRunning}>
                  {lensJob.isRunning ? "Reading…" : "Analyze"}
                </Button>
              </div>
          {!readyLenses.length && !loadError && (
            <p className="text-xs text-muted-foreground">No calibrated lens is available for this model in this session.</p>
          )}
          {hasLegacyLenses && <p className="text-xs text-amber-700">An earlier uncalibrated lens is saved in this session. Refit it in J-Lens Lab before using its readout.</p>}
          {lensJob.isRunning && <p className="text-xs text-muted-foreground">{lensJob.status?.progress.message || "Reading encoder states…"}</p>}
          {(loadError || lensJob.error) && <p className="text-xs text-destructive">{loadError || lensJob.error}</p>}
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between gap-2">
              <CardTitle className="text-xs">Layer × audio-time readout</CardTitle>
              <Badge variant="outline" className="text-[10px]">{result.architecture}</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-[11px] text-muted-foreground">Frequemcy-corrected (PMI) readout: subtracts each token's training-set prior so that low-frequency content words with encoder evidence outrank common special tokens. Use the rank slider to change which rank is shown in the grid. Click a cell to inspect all top-{result.layers[0]?.frames[0]?.tokens.length || 5} tokens.</p>
            {result.quality?.layers?.some((item) => item.validation_frames > 0) ? (
              <p className="text-[11px] text-muted-foreground">Held-out calibration is shown per layer: cosine similarity compares the readout with the frozen teacher representation; token agreement compares their top vocabulary token.</p>
            ) : (
              <p className="text-[11px] text-amber-700">This lens has no held-out calibration because it was fitted with fewer than 10 samples. Treat the readout as exploratory.</p>
            )}
            <div className="flex items-center gap-3 px-1 pb-1">
              <Label className="text-[11px] shrink-0">Token rank</Label>
              <Slider
                value={[topK]}
                onValueChange={([v]) => setTopK(v)}
                min={0}
                max={Math.min(4, (result.layers[0]?.frames[0]?.tokens.length || 5) - 1)}
                step={1}
                className="w-28"
              />
              <span className="text-[11px] text-muted-foreground w-16">#{topK + 1}</span>
              {result.quality?.layers?.some((l) => l.validation_frames > 0) && (
                <label className="flex items-center gap-1.5 ml-auto cursor-pointer">
                  <Checkbox checked={hideStuckLayers} onCheckedChange={(v) => setHideStuckLayers(!!v)} />
                  <span className="text-[11px] text-muted-foreground">Hide stuck (cosine &lt; 0.3)</span>
                </label>
              )}
            </div>
            <div className="space-y-1 overflow-x-auto pb-1">
              {visibleLayers.map((layer) => (
                <div key={layer.layer} className="flex min-w-max gap-1 items-stretch">
                  <span className="w-12 shrink-0 text-[10px] text-muted-foreground pt-1">Layer {layer.layer + 1}</span>
                  {layer.frames.map((frame, index) => {
                    const token = frame.tokens[topK];
                    const quality = layer.quality;
                    const active = selectedCell?.layer === layer.layer && selectedCell.frame === frame;
                    return (
                      <button
                        key={`${layer.layer}-${index}`}
                        type="button"
                        className={`w-12 min-h-10 rounded border px-1 text-[9px] leading-tight break-all transition-colors ${
                          active ? "border-primary bg-primary/15" : "border-border bg-muted/40 hover:bg-muted"
                        }`}
                        title={`${frame.start_time.toFixed(2)}–${frame.end_time.toFixed(2)} s · rank #{topK + 1}: ${token?.token || "—"}${quality?.cosine_similarity != null ? ` · held-out cosine ${quality.cosine_similarity.toFixed(2)}` : " · unvalidated"}`}
                        onClick={() => setSelectedCell({ layer: layer.layer, frame })}
                      >
                        {token?.token || "—"}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>

            {selectedCell && (
              <div className="rounded border border-border bg-muted/30 p-2 text-xs space-y-2">
                <p className="font-medium">Layer {selectedCell.layer + 1}, {selectedCell.frame.start_time.toFixed(2)}–{selectedCell.frame.end_time.toFixed(2)} s</p>
                {result.layers.find((layer) => layer.layer === selectedCell.layer)?.quality?.cosine_similarity != null && (
                  <p className="text-muted-foreground">Held-out calibration: cosine {result.layers.find((layer) => layer.layer === selectedCell.layer)?.quality?.cosine_similarity?.toFixed(2)} · top-token agreement {((result.layers.find((layer) => layer.layer === selectedCell.layer)?.quality?.top1_agreement || 0) * 100).toFixed(0)}%</p>
                )}
                {result.prior_correction && <p className="text-[10px] text-muted-foreground">Frequency-corrected (PMI) — subtracts token prior so low-frequency content words with encoder evidence outrank common special tokens.</p>}
                <div>
                  <p className="text-[10px] font-medium mb-1">Corrected top tokens:</p>
                  <div className="flex flex-wrap gap-1">
                    {selectedCell.frame.tokens.map((token) => (
                      <Badge key={token.token_id} variant="secondary" className="text-[10px] font-mono">
                        {token.token} <span className="ml-1 text-muted-foreground">{token.probability != null ? `${(token.probability * 100).toFixed(1)}%` : token.score.toFixed(2)}</span>
                      </Badge>
                    ))}
                  </div>
                </div>
                {selectedCell.frame.raw_tokens && selectedCell.frame.raw_tokens.some((t, i) => t.token_id !== selectedCell.frame.tokens[i]?.token_id) && (
                  <div>
                    <p className="text-[10px] font-medium mb-1">Raw (uncorrected) top tokens:</p>
                    <div className="flex flex-wrap gap-1">
                      {selectedCell.frame.raw_tokens.map((token) => (
                        <Badge key={token.token_id} variant="outline" className="text-[10px] font-mono">
                          {token.token} <span className="ml-1 text-muted-foreground">{token.probability != null ? `${(token.probability * 100).toFixed(1)}%` : token.score.toFixed(2)}</span>
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};
