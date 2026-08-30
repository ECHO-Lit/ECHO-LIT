import { useCallback, useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
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

interface LensPosition {
  position: number;
  tokens: LensToken[];
}

interface LensLayer {
  layer: number;
  positions: LensPosition[];
}

interface LensPositionMeta {
  position: number;
  token_id: number;
  token: string;
}

interface JacobianLensResult {
  lens_id: string;
  architecture: string;
  duration_seconds: number;
  transcript: string;
  transcript_source: "generated" | "provided";
  positions: LensPositionMeta[];
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
  const [selectedCell, setSelectedCell] = useState<{ layer: number; position: number } | null>(null);
  const [topK, setTopK] = useState(0);
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

  const deleteLens = async () => {
    if (!lensId) return;
    try {
      await deleteJacobianLens(lensId);
      await refreshLenses();
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Failed to delete lens");
    }
  };

  const maxRank = useMemo(
    () => (result?.layers[0]?.positions[0]?.tokens.length || 5) - 1,
    [result],
  );

  if (!selectedFile) {
    return <p className="text-xs text-muted-foreground p-3">Select an audio file to inspect decoder-layer token readouts.</p>;
  }

  return (
    <div className="space-y-3">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs">Decoder Jacobian Lens</CardTitle>
          <p className="text-[11px] text-muted-foreground">
            What each decoder layer is poised to say at every transcript position, read through the model's own output head. It is not a transcript.
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
            <p className="text-xs text-muted-foreground">No fitted lens is available for this model in this session.</p>
          )}
          {lensJob.isRunning && <p className="text-xs text-muted-foreground">{lensJob.status?.progress.message || "Reading decoder states…"}</p>}
          {(loadError || lensJob.error) && <p className="text-xs text-destructive">{loadError || lensJob.error}</p>}
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between gap-2">
              <CardTitle className="text-xs">Layer × transcript-position readout</CardTitle>
              <Badge variant="outline" className="text-[10px]">{result.architecture}</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-[11px] text-muted-foreground">
              Transcript ({result.transcript_source}): <span className="font-mono">{result.transcript || "—"}</span>
            </p>
            <p className="text-[11px] text-muted-foreground">Each cell shows the #{topK + 1} ranked token for one decoder position at one layer. Select a cell to inspect alternatives.</p>
            <div className="flex items-center gap-3 px-1 pb-1">
              <Label className="text-[11px] shrink-0">Token rank</Label>
              <Slider
                value={[topK]}
                onValueChange={([v]) => setTopK(v)}
                min={0}
                max={Math.min(4, maxRank)}
                step={1}
                className="w-28"
              />
              <span className="text-[11px] text-muted-foreground w-16">#{topK + 1}</span>
            </div>
            <div className="space-y-1 overflow-x-auto pb-1">
              <div className="flex min-w-max gap-1 items-stretch">
                <span className="w-12 shrink-0 text-[10px] text-muted-foreground pt-1">Token</span>
                {result.positions.map((position) => (
                  <span
                    key={position.position}
                    className={`w-12 min-h-10 rounded border px-1 pt-1 text-[9px] leading-tight break-all ${
                      selectedCell?.position === position.position ? "border-primary bg-primary/15" : "border-border bg-muted/60"
                    }`}
                    title={`position ${position.position}`}
                  >
                    {position.token}
                  </span>
                ))}
              </div>
              {result.layers.map((layer) => (
                <div key={layer.layer} className="flex min-w-max gap-1 items-stretch">
                  <span className="w-12 shrink-0 text-[10px] text-muted-foreground pt-1">Layer {layer.layer + 1}</span>
                  {result.positions.map((positionMeta) => {
                    const token = layer.positions.find((cell) => cell.position === positionMeta.position)?.tokens[topK];
                    const active = selectedCell?.layer === layer.layer && selectedCell?.position === positionMeta.position;
                    return (
                      <button
                        key={`${layer.layer}-${positionMeta.position}`}
                        type="button"
                        className={`w-12 min-h-10 rounded border px-1 text-[9px] leading-tight break-all transition-colors ${
                          active ? "border-primary bg-primary/15" : "border-border bg-muted/40 hover:bg-muted"
                        }`}
                        title={`Layer ${layer.layer + 1} · position ${positionMeta.position} · rank #${topK + 1}: ${token?.token || "—"}`}
                        onClick={() => setSelectedCell({ layer: layer.layer, position: positionMeta.position })}
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
                <p className="font-medium">
                  Layer {selectedCell.layer + 1}, position {selectedCell.position}
                  {result.positions[selectedCell.position] && (
                    <span className="text-muted-foreground"> · after “{result.positions[selectedCell.position].token}”</span>
                  )}
                </p>
                <div>
                  <p className="text-[10px] font-medium mb-1">Top tokens:</p>
                  <div className="flex flex-wrap gap-1">
                    {(result.layers.find((layer) => layer.layer === selectedCell.layer)?.positions.find((cell) => cell.position === selectedCell.position)?.tokens || []).map((token) => (
                      <Badge key={token.token_id} variant="secondary" className="text-[10px] font-mono">
                        {token.token} <span className="ml-1 text-muted-foreground">{token.probability != null ? `${(token.probability * 100).toFixed(1)}%` : token.score.toFixed(2)}</span>
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};
