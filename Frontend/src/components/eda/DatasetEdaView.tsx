import { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { HelpCircle, RefreshCw, BarChart3 } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { materializeAudio, runJob } from "@/lib/jobs";
import { EDA_CHART_EXPLANATIONS, getFeatureExplanation } from "@/lib/audioFeatures";
import { SummaryTiles } from "./SummaryTiles";
import { HistogramChart } from "./HistogramChart";
import { ClassBalanceBar } from "./ClassBalanceBar";
import { BoxPlotChart } from "./BoxPlotChart";
import { CorrelationHeatmap } from "./CorrelationHeatmap";

interface MetadataEda {
  dataset: string;
  summary: {
    total_files: number;
    total_hours: number;
    mean_duration: number;
    median_duration: number;
    num_classes: number;
  };
  duration_histogram: { histogram: number[]; bins: number[] };
  class_balance: Record<string, number>;
  transcript_length_histogram: { histogram: number[]; bins: number[] };
  sample_rate_breakdown: Record<string, number>;
}

interface AcousticEda {
  aggregate_statistics: Record<string, { mean: number; std: number; min: number; max: number; median: number; q1: number; q3: number }>;
  feature_distributions: Record<string, { histogram: number[]; bins: number[] }>;
  individual_analyses: Array<{ filename: string; features: Record<string, number> }>;
  summary: { total_files: number; total_features_extracted: number; avg_duration: number; avg_tempo: number };
}

interface ChartCardProps {
  title: string;
  explanation?: string;
  children: React.ReactNode;
  // "y" = drag-resize taller only (default); "both" = also widen, for the
  // correlation matrix which needs more room to fit legible labels.
  resize?: "y" | "both";
}

const ChartCard = ({ title, explanation, children, resize = "y" }: ChartCardProps) => (
  <div className="border border-border rounded-lg bg-card p-2">
    <div className="flex items-center gap-1.5 mb-1 px-1">
      <span className="text-xs font-medium text-foreground">{title}</span>
      {explanation && (
        <Tooltip>
          <TooltipTrigger>
            <HelpCircle className="h-3 w-3 text-muted-foreground" />
          </TooltipTrigger>
          <TooltipContent className="max-w-xs text-xs">{explanation}</TooltipContent>
        </Tooltip>
      )}
    </div>
    <div
      className={`h-56 min-h-[10rem] max-h-[36rem] overflow-hidden ${resize === "both" ? "resize max-w-full" : "resize-y"}`}
    >
      {children}
    </div>
  </div>
);

interface DatasetEdaViewProps {
  dataset: string;
  availableFiles: string[];
}

const CORRELATION_FEATURE_LIMIT = 8;

export const DatasetEdaView = ({ dataset, availableFiles }: DatasetEdaViewProps) => {
  const [metadataEda, setMetadataEda] = useState<MetadataEda | null>(null);
  const [isLoadingMetadata, setIsLoadingMetadata] = useState(false);
  const [metadataError, setMetadataError] = useState<string | null>(null);

  const [acousticEda, setAcousticEda] = useState<AcousticEda | null>(null);
  const [isLoadingAcoustics, setIsLoadingAcoustics] = useState(false);
  const [acousticsError, setAcousticsError] = useState<string | null>(null);
  const [acousticsProgress, setAcousticsProgress] = useState<{ current: number; total: number } | null>(null);

  const fetchMetadataEda = useCallback(async (signal?: AbortSignal) => {
    if (!dataset || dataset === "custom") {
      setMetadataEda(null);
      return;
    }
    setIsLoadingMetadata(true);
    setMetadataError(null);
    try {
      const res = await fetch(`${API_BASE}/${dataset}/eda`, { credentials: "include", signal });
      if (!res.ok) throw new Error(`Failed to fetch EDA: ${res.status}`);
      const data: MetadataEda = await res.json();
      setMetadataEda(data);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setMetadataError(err instanceof Error ? err.message : "Unknown error");
      setMetadataEda(null);
    } finally {
      setIsLoadingMetadata(false);
    }
  }, [dataset]);

  useEffect(() => {
    const ac = new AbortController();
    fetchMetadataEda(ac.signal);
    // Acoustic EDA is dataset-scoped and expensive — clear it on dataset change
    // rather than silently showing stale-dataset charts.
    setAcousticEda(null);
    setAcousticsError(null);
    return () => ac.abort();
  }, [dataset, fetchMetadataEda]);

  const handleComputeAcoustics = async () => {
    if (availableFiles.length === 0) return;
    setIsLoadingAcoustics(true);
    setAcousticsError(null);
    setAcousticsProgress({ current: 0, total: availableFiles.length });
    try {
      const assets = await Promise.all(availableFiles.map((f) => materializeAudio(dataset, f)));
      const analysis = await runJob<AcousticEda>(
        { operation: "audio_features", audio_ids: assets.map((a) => a.audio_id) },
        { onProgress: (status) => setAcousticsProgress({ current: status.progress?.current ?? 0, total: availableFiles.length }) },
      );
      setAcousticEda(analysis);
    } catch (err) {
      setAcousticsError(err instanceof Error ? err.message : "Unknown error");
      setAcousticEda(null);
    } finally {
      setIsLoadingAcoustics(false);
      setAcousticsProgress(null);
    }
  };

  const featureKeys = acousticEda ? Object.keys(acousticEda.aggregate_statistics) : [];
  const featuresByFile = acousticEda
    ? Object.fromEntries(acousticEda.individual_analyses.map((a) => [a.filename, a.features]))
    : {};
  // Rank by std to surface the most variable (most informative) features first.
  const rankedFeatureKeys = [...featureKeys].sort(
    (a, b) => acousticEda!.aggregate_statistics[b].std - acousticEda!.aggregate_statistics[a].std,
  );

  return (
    <div className="space-y-3">
      {/* Metadata EDA */}
      {isLoadingMetadata ? (
        <div className="text-xs text-primary flex items-center gap-2 p-3 bg-primary/5 rounded-sm border border-primary/20">
          <div className="w-2 h-2 bg-primary rounded-full animate-ping"></div>
          Loading dataset metadata...
        </div>
      ) : metadataError ? (
        <div className="text-xs text-destructive flex items-center gap-2 p-3 bg-destructive/5 rounded-sm border border-destructive/20">
          {metadataError}
        </div>
      ) : !metadataEda ? (
        <div className="text-xs text-muted-foreground flex items-center gap-2 p-3 bg-muted/50 rounded-md border border-border">
          No metadata EDA available for this dataset selection.
        </div>
      ) : (
        <div className="space-y-3">
          <SummaryTiles
            totalFiles={metadataEda.summary.total_files}
            totalHours={metadataEda.summary.total_hours}
            meanDuration={metadataEda.summary.mean_duration}
            medianDuration={metadataEda.summary.median_duration}
            numClasses={metadataEda.summary.num_classes}
          />

          <ChartCard title="Duration distribution" explanation={EDA_CHART_EXPLANATIONS.duration_histogram}>
            <HistogramChart
              bins={metadataEda.duration_histogram.bins}
              histogram={metadataEda.duration_histogram.histogram}
              label="Duration (s)"
            />
          </ChartCard>

          {Object.keys(metadataEda.class_balance).length > 0 && (
            <ChartCard title="Class balance" explanation={EDA_CHART_EXPLANATIONS.class_balance}>
              <ClassBalanceBar counts={metadataEda.class_balance} />
            </ChartCard>
          )}

          {metadataEda.transcript_length_histogram.histogram.length > 0 && (
            <ChartCard title="Transcript length (words)" explanation={EDA_CHART_EXPLANATIONS.transcript_length_histogram}>
              <HistogramChart
                bins={metadataEda.transcript_length_histogram.bins}
                histogram={metadataEda.transcript_length_histogram.histogram}
                label="Words"
              />
            </ChartCard>
          )}
        </div>
      )}

      {/* Acoustic EDA */}
      <div className="border-t border-border pt-3 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <BarChart3 className="h-3.5 w-3.5 text-primary" />
            <span className="text-xs font-medium">Acoustic features</span>
          </div>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                size="sm"
                variant="secondary"
                className="h-7 text-xs"
                onClick={handleComputeAcoustics}
                disabled={isLoadingAcoustics || availableFiles.length === 0}
              >
                <RefreshCw className={`h-3 w-3 mr-1 ${isLoadingAcoustics ? "animate-spin" : ""}`} />
                Compute acoustics
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              Runs librosa feature extraction over all {availableFiles.length} files in the dataset. Slower than metadata EDA — cached per file after the first run.
            </TooltipContent>
          </Tooltip>
        </div>

        {isLoadingAcoustics && (
          <div className="text-xs text-primary flex items-center gap-2 p-3 bg-primary/5 rounded-sm border border-primary/20">
            <div className="w-2 h-2 bg-primary rounded-full animate-ping"></div>
            Extracting features{acousticsProgress ? ` (${acousticsProgress.current}/${acousticsProgress.total})` : "..."}
          </div>
        )}
        {acousticsError && (
          <div className="text-xs text-destructive flex items-center gap-2 p-3 bg-destructive/5 rounded-sm border border-destructive/20">
            {acousticsError}
          </div>
        )}
        {!acousticEda && !isLoadingAcoustics && !acousticsError && (
          <div className="text-xs text-muted-foreground flex items-center gap-2 p-3 bg-muted/50 rounded-md border border-border">
            Click "Compute acoustics" to extract spectral/MFCC/chroma features across the dataset.
          </div>
        )}

        {acousticEda && (
          <div className="space-y-3">
            <Badge variant="outline" className="text-[10px]">
              {acousticEda.summary.total_files} files &middot; {acousticEda.summary.total_features_extracted} features
            </Badge>

            {rankedFeatureKeys.slice(0, 6).map((key) => {
              const stats = acousticEda.aggregate_statistics[key];
              const dist = acousticEda.feature_distributions[key];
              return (
                <div key={key} className="grid grid-cols-2 gap-2">
                  <ChartCard title={`${key.replace(/_/g, " ")} — distribution`} explanation={getFeatureExplanation(key)}>
                    <HistogramChart bins={dist.bins} histogram={dist.histogram} label={key} color="hsl(var(--primary))" />
                  </ChartCard>
                  <ChartCard title={`${key.replace(/_/g, " ")} — spread`} explanation={EDA_CHART_EXPLANATIONS.feature_box_plot}>
                    <BoxPlotChart min={stats.min} q1={stats.q1} median={stats.median} q3={stats.q3} max={stats.max} label={key} />
                  </ChartCard>
                </div>
              );
            })}

            <ChartCard title="Feature correlation" explanation={EDA_CHART_EXPLANATIONS.correlation_heatmap} resize="both">
              <CorrelationHeatmap
                featuresByFile={featuresByFile}
                featureKeys={rankedFeatureKeys.slice(0, CORRELATION_FEATURE_LIMIT)}
              />
            </ChartCard>
          </div>
        )}
      </div>
    </div>
  );
};
