import { useCallback, useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { firstJobResult, resolveAudioId } from "@/lib/jobs";
import { listJacobianLenses, type JacobianLens } from "@/lib/models";
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
}

interface LensFrame {
  start_time: number;
  end_time: number;
  tokens: LensToken[];
}

interface LensLayer {
  layer: number;
  frames: LensFrame[];
}

interface JacobianLensResult {
  lens_id: string;
  architecture: "seq2seq" | "ctc";
  duration_seconds: number;
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
  const lensJob = useJob<unknown>();

  const readyLenses = useMemo(
    () => lenses.filter((lens) => lens.status === "ready"),
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

  if (!selectedFile) {
    return <p className="text-xs text-muted-foreground p-3">Select an audio file to inspect encoder-layer token readouts.</p>;
  }

  return (
    <div className="space-y-3">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs">Encoder Jacobian Lens</CardTitle>
          <p className="text-[11px] text-muted-foreground">
            Experimental readout of each encoder layer through the model’s vocabulary head.
          </p>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex gap-2">
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
            <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => void refreshLenses()} disabled={lensJob.isRunning}>
              Refresh
            </Button>
            <Button size="sm" className="h-8 text-xs" onClick={() => void analyze()} disabled={!lensId || lensJob.isRunning}>
              {lensJob.isRunning ? "Reading…" : "Analyze"}
            </Button>
          </div>
          {!readyLenses.length && !loadError && (
            <p className="text-xs text-muted-foreground">No fitted lens is available for this model in this session.</p>
          )}
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
            <p className="text-[11px] text-muted-foreground">Each cell is the top token for one pooled audio interval. Select a cell to inspect alternatives.</p>
            <div className="space-y-1 overflow-x-auto pb-1">
              {result.layers.map((layer) => (
                <div key={layer.layer} className="flex min-w-max gap-1 items-stretch">
                  <span className="w-12 shrink-0 text-[10px] text-muted-foreground pt-1">Layer {layer.layer + 1}</span>
                  {layer.frames.map((frame, index) => {
                    const top = frame.tokens[0];
                    const active = selectedCell?.layer === layer.layer && selectedCell.frame === frame;
                    return (
                      <button
                        key={`${layer.layer}-${index}`}
                        type="button"
                        className={`w-12 min-h-10 rounded border px-1 text-[9px] leading-tight break-all transition-colors ${
                          active ? "border-primary bg-primary/15" : "border-border bg-muted/40 hover:bg-muted"
                        }`}
                        title={`${frame.start_time.toFixed(2)}–${frame.end_time.toFixed(2)} s: ${top?.token || "—"}`}
                        onClick={() => setSelectedCell({ layer: layer.layer, frame })}
                      >
                        {top?.token || "—"}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>

            {selectedCell && (
              <div className="rounded border border-border bg-muted/30 p-2 text-xs">
                <p className="mb-1 font-medium">Layer {selectedCell.layer + 1}, {selectedCell.frame.start_time.toFixed(2)}–{selectedCell.frame.end_time.toFixed(2)} s</p>
                <div className="flex flex-wrap gap-1">
                  {selectedCell.frame.tokens.map((token) => (
                    <Badge key={token.token_id} variant="secondary" className="text-[10px] font-mono">
                      {token.token} <span className="ml-1 text-muted-foreground">{token.score.toFixed(2)}</span>
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};
