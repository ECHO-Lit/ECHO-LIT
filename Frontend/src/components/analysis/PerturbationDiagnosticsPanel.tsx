import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import { Loader2, Play, X } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { resolveAudioId } from "@/lib/jobs";
import {
  PropertySweepControls,
  defaultSweepState,
  toSweepConfigs,
  type SweepStateMap,
} from "./PropertySweepControls";
import { SensitivityProfileChart } from "@/components/visualization/SensitivityProfileChart";
import { FailureBoundaryCallout } from "./FailureBoundaryCallout";
import { VariantInspector } from "./VariantInspector";
import { useAnalysisJob } from "@/hooks/use-job-query";
import {
  estimateVariants,
  isGridOverLimit,
  MAX_GRID_VARIANTS,
  submitLinguisticAcoustic,
  type LinguisticAcousticResult,
} from "@/lib/linguisticAcoustic";

interface UploadedFile {
  audio_id?: string;
  file_id: string;
  filename: string;
  playback_url?: string;
}

interface PerturbationDiagnosticsPanelProps {
  selectedFile: UploadedFile | null;
  model?: string;
  dataset?: string;
  originalDataset?: string;
}

const getAudioUrl = (file: UploadedFile): string => {
  if (file.playback_url) return `${API_BASE}${file.playback_url}`;
  if (file.audio_id) return `${API_BASE}/audio/${file.audio_id}`;
  return "";
};

export function PerturbationDiagnosticsPanel({
  selectedFile, model, dataset, originalDataset,
}: PerturbationDiagnosticsPanelProps) {
  const [sweepState, setSweepState] = useState<SweepStateMap>(defaultSweepState);
  const [includeLexicalControl, setIncludeLexicalControl] = useState(true);
  const [selected, setSelected] = useState<{ property: string; theta: number } | null>(null);

  const job = useAnalysisJob<LinguisticAcousticResult, void>(async () => {
    if (!selectedFile) throw new Error("No file selected");
    if (!model) throw new Error("Select a model first");
    const audioId = await resolveAudioId(
      selectedFile,
      originalDataset && originalDataset !== "custom" ? originalDataset : dataset,
    );
    const sweeps = toSweepConfigs(sweepState);
    if (sweeps.length === 0) throw new Error("Enable at least one property to sweep");
    return submitLinguisticAcoustic({
      audio_ids: [audioId], model, task: "auto", sweeps,
      include_lexical_control: includeLexicalControl,
    });
  }, "fr7");

  const sweeps = useMemo(() => toSweepConfigs(sweepState), [sweepState]);
  const estimatedVariants = estimateVariants(sweeps, 1, includeLexicalControl);
  const overLimit = isGridOverLimit(sweeps, 1, includeLexicalControl);
  const noSweepsEnabled = sweeps.length === 0;

  const progressPct = job.progress && job.progress.total > 0
    ? Math.round((job.progress.current / job.progress.total) * 100)
    : 0;

  if (!selectedFile) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        Select an audio file to run a linguistic-vs-acoustic sensitivity analysis.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 p-3">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Sweep configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <PropertySweepControls
            state={sweepState}
            onChange={setSweepState}
            disabled={job.isRunning}
          />

          <div className="flex items-center justify-between rounded-md border border-border p-3">
            <label className="flex items-center gap-2 text-sm font-medium cursor-pointer">
              <Switch
                checked={includeLexicalControl}
                onCheckedChange={setIncludeLexicalControl}
                disabled={job.isRunning}
              />
              Lexical-destruction control
            </label>
            <span className="text-xs text-muted-foreground">masks ~30% of words as a reference ceiling</span>
          </div>

          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              ~{estimatedVariants} variant{estimatedVariants === 1 ? "" : "s"}
              {overLimit && (
                <span className="text-destructive font-medium">
                  {" "}— exceeds the {MAX_GRID_VARIANTS}-variant limit, reduce steps/repeats
                </span>
              )}
            </span>
          </div>

          {job.error && <p className="text-xs text-destructive">{job.error}</p>}

          <div className="flex gap-2">
            <Button
              size="sm"
              disabled={job.isRunning || job.isSubmitting || overLimit || noSweepsEnabled}
              onClick={() => job.start()}
            >
              {job.isRunning || job.isSubmitting ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <Play className="h-4 w-4 mr-1" />
              )}
              Run analysis
            </Button>
            {job.isRunning && (
              <Button size="sm" variant="outline" onClick={() => job.cancel()}>
                <X className="h-4 w-4 mr-1" />
                Cancel
              </Button>
            )}
          </div>

          {job.isRunning && job.progress && (
            <div className="space-y-1">
              <Progress value={progressPct} />
              <p className="text-xs text-muted-foreground">{job.progress.message}</p>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="space-y-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Sensitivity profile</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {job.result ? (
              <>
                <SensitivityProfileChart
                  profiles={job.result.properties}
                  onPointClick={(property, theta) => setSelected({ property, theta })}
                />
                <FailureBoundaryCallout profile={job.result.profile} />
              </>
            ) : (
              <div className="flex h-[280px] items-center justify-center text-sm text-muted-foreground border border-dashed border-border rounded-md">
                {job.isRunning ? "Rendering and re-inferring variants…" : "Run an analysis to see the sensitivity profile."}
              </div>
            )}
          </CardContent>
        </Card>

        {job.result && (
          <VariantInspector
            result={job.result}
            baselineAudioUrl={getAudioUrl(selectedFile)}
            selectedProperty={selected?.property ?? null}
            selectedTheta={selected?.theta ?? null}
          />
        )}
      </div>
    </div>
  );
}
