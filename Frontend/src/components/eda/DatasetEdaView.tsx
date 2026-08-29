import { useEffect, useState, useCallback, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { HelpCircle, RefreshCw, BarChart3, Download, AlertTriangle } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { materializeAudio, runJob } from "@/lib/jobs";
import { EDA_CHART_EXPLANATIONS, getFeatureExplanation } from "@/lib/audioFeatures";
import { correlationMatrix, topCorrelatedPairs, quartiles, zScores, bucketize, recomputeHistogram, type Quartiles } from "@/lib/edaStats";
import { exportAcousticFeaturesCsv, exportEdaJson } from "@/lib/edaExport";
import { readEdaCache, writeEdaCache } from "@/lib/edaCache";
import { SummaryTiles } from "./SummaryTiles";
import { HistogramChart } from "./HistogramChart";
import { ClassBalanceBar } from "./ClassBalanceBar";
import { BoxPlotChart } from "./BoxPlotChart";
import { CorrelationHeatmap } from "./CorrelationHeatmap";
import { ChartCard } from "./ChartCard";
import { ClusteringSection } from "./ClusteringSection";
import { NearestNeighborsPanel } from "./NearestNeighborsPanel";

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
  labels_by_file: Record<string, string>;
  durations_by_file: Record<string, number>;
}

interface AcousticEda {
  aggregate_statistics: Record<string, { mean: number; std: number; min: number; max: number; median: number; q1: number; q3: number }>;
  feature_distributions: Record<string, { histogram: number[]; bins: number[] }>;
  individual_analyses: Array<{ filename: string; features: Record<string, number> }>;
  summary: { total_files: number; total_features_extracted: number; avg_duration: number; avg_tempo: number };
  cache_info?: { cached_count: number; missing_count: number; cache_hit_rate: number };
}

interface DatasetEdaViewProps {
  dataset: string;
  availableFiles: string[];
  // Drives the nearest-neighbour panel.
  selectedFile?: string | null;
  // Jump the player to a file (reuses the same chain the embedding scatter uses).
  onFileSelect?: (filename: string) => void;
  // Toggle the file-table filter. Owned by EmbeddingPanel so this tab and the
  // Embeddings tab's cluster table share one active filter.
  onBucketClick: (key: string, filenames: string[]) => void;
  activeFilterKey: string | null;
}

const CORRELATION_FEATURE_LIMIT = 8;
const OUTLIER_Z_THRESHOLD = 3;
const OUTLIER_LIST_LIMIT = 10;
// Datasets with a server-side metadata EDA endpoint, i.e. those that can be compared.
const COMPARABLE_DATASETS = ["common-voice", "ravdess", "l2-arctic", "saa"];

async function fetchEdaFor(dataset: string, signal?: AbortSignal): Promise<MetadataEda> {
  const res = await fetch(`${API_BASE}/${dataset}/eda`, { credentials: "include", signal });
  if (!res.ok) throw new Error(`Failed to fetch EDA: ${res.status}`);
  return (await res.json()) as MetadataEda;
}

export const DatasetEdaView = ({
  dataset,
  availableFiles,
  selectedFile,
  onFileSelect,
  onBucketClick: handleBucketClick,
  activeFilterKey,
}: DatasetEdaViewProps) => {
  const [metadataEda, setMetadataEda] = useState<MetadataEda | null>(null);
  const [isLoadingMetadata, setIsLoadingMetadata] = useState(false);
  const [metadataError, setMetadataError] = useState<string | null>(null);

  const [acousticEda, setAcousticEda] = useState<AcousticEda | null>(null);
  const [isLoadingAcoustics, setIsLoadingAcoustics] = useState(false);
  const [acousticsError, setAcousticsError] = useState<string | null>(null);
  const [acousticsProgress, setAcousticsProgress] = useState<{ current: number; total: number } | null>(null);
  // True when the acoustics on screen came from sessionStorage rather than a fresh
  // run — surfaced as a badge so the numbers are never mistaken for a new result.
  const [acousticsFromCache, setAcousticsFromCache] = useState(false);

  // Optional second dataset overlaid on the metadata charts.
  const [compareDataset, setCompareDataset] = useState<string>("");
  const [compareEda, setCompareEda] = useState<MetadataEda | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);

  const fetchMetadataEda = useCallback(async (signal?: AbortSignal) => {
    if (!dataset || dataset === "custom") {
      setMetadataEda(null);
      return;
    }
    // Radix unmounts the inactive tab, so this view remounts every time the user
    // comes back. Paint the cached result immediately and only show the loading
    // banner on a genuinely cold load; the refetch below still runs and replaces
    // it, so nothing goes stale.
    const cached = readEdaCache<MetadataEda>("metadata", dataset, []);
    if (cached) setMetadataEda(cached);
    setIsLoadingMetadata(!cached);
    setMetadataError(null);
    try {
      const fresh = await fetchEdaFor(dataset, signal);
      setMetadataEda(fresh);
      writeEdaCache("metadata", dataset, [], fresh);
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
    // Acoustic EDA is dataset-scoped — drop it rather than silently showing stale
    // state. A cached result for this exact dataset + file list is restored
    // instead of recomputed. (The chart filter is reset by EmbeddingPanel, which
    // owns it.)
    const cached = readEdaCache<AcousticEda>("acoustic", dataset, availableFiles);
    setAcousticEda(cached);
    setAcousticsFromCache(Boolean(cached));
    setAcousticsError(null);
    return () => ac.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, fetchMetadataEda, availableFiles]);

  // Fetch (or clear) the comparison dataset's metadata EDA.
  useEffect(() => {
    if (!compareDataset) {
      setCompareEda(null);
      setCompareError(null);
      return;
    }
    const ac = new AbortController();
    fetchEdaFor(compareDataset, ac.signal)
      .then((data) => {
        setCompareEda(data);
        setCompareError(null);
      })
      .catch((err) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setCompareError(err instanceof Error ? err.message : "Unknown error");
        setCompareEda(null);
      });
    return () => ac.abort();
  }, [compareDataset]);

  // Clear the comparison when the primary dataset becomes the compared one.
  useEffect(() => {
    if (compareDataset === dataset) setCompareDataset("");
  }, [dataset, compareDataset]);

  // Re-bin the comparison durations onto the primary dataset's edges so the two
  // series sit on one shared axis.
  const compareDurationHistogram = useMemo(() => {
    if (!compareEda || !metadataEda?.duration_histogram.bins.length) return null;
    const values = Object.values(compareEda.durations_by_file ?? {});
    if (values.length === 0) return null;
    return recomputeHistogram(values, metadataEda.duration_histogram.bins);
  }, [compareEda, metadataEda]);

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
      setAcousticsFromCache(false);
      writeEdaCache("acoustic", dataset, availableFiles, analysis);
    } catch (err) {
      setAcousticsError(err instanceof Error ? err.message : "Unknown error");
      setAcousticEda(null);
    } finally {
      setIsLoadingAcoustics(false);
      setAcousticsProgress(null);
    }
  };

  const featureKeys = useMemo(
    () => (acousticEda ? Object.keys(acousticEda.aggregate_statistics) : []),
    [acousticEda],
  );
  const featuresByFile = useMemo(
    () => (acousticEda ? Object.fromEntries(acousticEda.individual_analyses.map((a) => [a.filename, a.features])) : {}),
    [acousticEda],
  );
  // Rank by std to surface the most variable (most informative) features first.
  const rankedFeatureKeys = useMemo(
    () => (acousticEda ? [...featureKeys].sort((a, b) => acousticEda.aggregate_statistics[b].std - acousticEda.aggregate_statistics[a].std) : []),
    [acousticEda, featureKeys],
  );

  // Bucket membership for click-to-filter, computed once per acoustics/metadata result.
  const durationBuckets = useMemo(() => {
    if (!metadataEda?.durations_by_file || !metadataEda.duration_histogram.bins.length) return null;
    const edges = metadataEda.duration_histogram.bins;
    const buckets: string[][] = edges.slice(0, -1).map(() => []);
    for (const [filename, duration] of Object.entries(metadataEda.durations_by_file)) {
      const idx = bucketize(duration, edges);
      if (idx >= 0) buckets[idx].push(filename);
    }
    return buckets;
  }, [metadataEda]);

  const classFilenames = useMemo(() => {
    if (!metadataEda?.labels_by_file) return null;
    const byClass: Record<string, string[]> = {};
    for (const [filename, label] of Object.entries(metadataEda.labels_by_file)) {
      (byClass[label] ||= []).push(filename);
    }
    return byClass;
  }, [metadataEda]);

  const featureHistogramBuckets = useMemo(() => {
    if (!acousticEda) return {};
    const result: Record<string, string[][]> = {};
    for (const key of rankedFeatureKeys.slice(0, 6)) {
      const dist = acousticEda.feature_distributions[key];
      if (!dist?.bins?.length) continue;
      const buckets: string[][] = dist.bins.slice(0, -1).map(() => []);
      for (const analysis of acousticEda.individual_analyses) {
        const value = analysis.features[key];
        if (typeof value !== "number") continue;
        const idx = bucketize(value, dist.bins);
        if (idx >= 0) buckets[idx].push(analysis.filename);
      }
      result[key] = buckets;
    }
    return result;
  }, [acousticEda, rankedFeatureKeys]);

  // Per-class quartiles for the top few features — shows which acoustics
  // actually separate classes.
  const perClassQuartiles = useMemo(() => {
    if (!acousticEda || !metadataEda?.labels_by_file) return {};
    const labels = metadataEda.labels_by_file;
    if (Object.keys(labels).length === 0) return {};
    const result: Record<string, Record<string, Quartiles>> = {};
    for (const key of rankedFeatureKeys.slice(0, 3)) {
      const byClass: Record<string, number[]> = {};
      for (const analysis of acousticEda.individual_analyses) {
        const className = labels[analysis.filename];
        const value = analysis.features[key];
        if (!className || typeof value !== "number") continue;
        (byClass[className] ||= []).push(value);
      }
      const quartilesByClass: Record<string, Quartiles> = {};
      for (const [className, values] of Object.entries(byClass)) {
        quartilesByClass[className] = quartiles(values);
      }
      if (Object.keys(quartilesByClass).length > 0) result[key] = quartilesByClass;
    }
    return result;
  }, [acousticEda, metadataEda, rankedFeatureKeys]);

  const correlationPairs = useMemo(() => {
    if (!acousticEda) return [];
    const keys = rankedFeatureKeys.slice(0, CORRELATION_FEATURE_LIMIT);
    const matrix = correlationMatrix(featuresByFile, keys);
    return topCorrelatedPairs(matrix, keys, 0.8).slice(0, 5);
  }, [acousticEda, featuresByFile, rankedFeatureKeys]);

  const outliers = useMemo(() => {
    if (!acousticEda) return [];
    return zScores(featuresByFile, acousticEda.aggregate_statistics, OUTLIER_Z_THRESHOLD).slice(0, OUTLIER_LIST_LIMIT);
  }, [acousticEda, featuresByFile]);

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
          {/* Compare against a second dataset. Metadata only — acoustics and
              clusters would each need their own job per dataset. */}
          <div className="flex items-center gap-1.5 px-1">
            <span className="text-[10px] text-muted-foreground">Compare with</span>
            <select
              value={compareDataset}
              onChange={(event) => setCompareDataset(event.target.value)}
              className="h-6 text-[10px] bg-background border border-border rounded px-1 flex-1 min-w-0"
              aria-label="Comparison dataset"
            >
              <option value="">None</option>
              {COMPARABLE_DATASETS.filter((name) => name !== dataset).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            <Tooltip>
              <TooltipTrigger>
                <HelpCircle className="h-3 w-3 text-muted-foreground" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs text-xs">
                {EDA_CHART_EXPLANATIONS.dataset_comparison}
              </TooltipContent>
            </Tooltip>
          </div>
          {compareError && (
            <div className="text-xs text-destructive p-2 bg-destructive/5 rounded-sm border border-destructive/20">
              Comparison dataset failed to load: {compareError}
            </div>
          )}
          {compareEda && (
            <div className="text-[10px] text-muted-foreground px-1">
              Chart filtering is disabled while comparing — the file table only holds {dataset}.
            </div>
          )}

          <SummaryTiles
            totalFiles={metadataEda.summary.total_files}
            totalHours={metadataEda.summary.total_hours}
            meanDuration={metadataEda.summary.mean_duration}
            medianDuration={metadataEda.summary.median_duration}
            numClasses={metadataEda.summary.num_classes}
          />

          {compareEda && (
            <div className="text-[10px] text-muted-foreground px-1 grid grid-cols-2 gap-x-2 gap-y-0.5">
              <span className="col-span-2 font-medium text-foreground">{compareDataset}</span>
              <span>{compareEda.summary.total_files} files</span>
              <span>{compareEda.summary.total_hours} h</span>
              <span>mean {compareEda.summary.mean_duration}s</span>
              <span>{compareEda.summary.num_classes} classes</span>
            </div>
          )}

          <ChartCard
            title={compareEda ? `Duration distribution — ${dataset} vs ${compareDataset}` : "Duration distribution"}
            explanation={compareEda ? EDA_CHART_EXPLANATIONS.dataset_comparison : EDA_CHART_EXPLANATIONS.duration_histogram}
          >
            <HistogramChart
              bins={metadataEda.duration_histogram.bins}
              histogram={metadataEda.duration_histogram.histogram}
              label={compareEda ? dataset : "Duration (s)"}
              customdata={compareEda ? undefined : durationBuckets ?? undefined}
              onBarClick={
                !compareEda && durationBuckets
                  ? (filenames) => handleBucketClick("duration", filenames)
                  : undefined
              }
              series2={
                compareDurationHistogram
                  ? { histogram: compareDurationHistogram, label: compareDataset }
                  : undefined
              }
            />
          </ChartCard>

          {Object.keys(metadataEda.class_balance).length > 0 && (
            <ChartCard title="Class balance" explanation={EDA_CHART_EXPLANATIONS.class_balance}>
              <ClassBalanceBar
                counts={metadataEda.class_balance}
                filenamesByClass={classFilenames ?? undefined}
                onBarClick={
                  !compareEda && classFilenames
                    ? (className, filenames) => handleBucketClick(`class:${className}`, filenames)
                    : undefined
                }
                counts2={
                  compareEda && Object.keys(compareEda.class_balance).length > 0
                    ? compareEda.class_balance
                    : undefined
                }
                label={dataset}
                label2={compareDataset}
              />
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

      {/* Clusters vs ground-truth labels. The silhouette score, cluster table and
          noise list live beside the scatter in the Embeddings tab. */}
      <ClusteringSection labelsByFile={metadataEda?.labels_by_file} />

      <div className="border-t border-border pt-3">
        <NearestNeighborsPanel selectedFile={selectedFile ?? null} onFileSelect={onFileSelect} />
      </div>

      {/* Acoustic EDA */}
      <div className="border-t border-border pt-3 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <BarChart3 className="h-3.5 w-3.5 text-primary" />
            <span className="text-xs font-medium">Acoustic features</span>
          </div>
          <div className="flex items-center gap-1.5">
            {acousticEda && (
              <>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 text-xs"
                      onClick={() => exportAcousticFeaturesCsv(acousticEda, dataset)}
                    >
                      <Download className="h-3 w-3 mr-1" />
                      CSV
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Download per-file feature table as CSV</TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 text-xs"
                      onClick={() => exportEdaJson({ metadata: metadataEda, acoustic: acousticEda }, dataset)}
                    >
                      <Download className="h-3 w-3 mr-1" />
                      JSON
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Download full EDA report as JSON</TooltipContent>
                </Tooltip>
              </>
            )}
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
            <div className="flex items-center gap-1.5 flex-wrap">
              <Badge variant="outline" className="text-[10px]">
                {acousticEda.summary.total_files} files &middot; {acousticEda.summary.total_features_extracted} features
              </Badge>
              {acousticEda.cache_info && (
                <Badge variant="outline" className="text-[10px] bg-primary/5">
                  {acousticEda.cache_info.cached_count}/{acousticEda.cache_info.cached_count + acousticEda.cache_info.missing_count} cached
                </Badge>
              )}
              {acousticsFromCache && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge variant="outline" className="text-[10px] bg-muted">restored</Badge>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs text-xs">
                    Loaded from this session's earlier run for the same files. Click "Compute acoustics" to recompute.
                  </TooltipContent>
                </Tooltip>
              )}
            </div>

            {rankedFeatureKeys.slice(0, 6).map((key) => {
              const stats = acousticEda.aggregate_statistics[key];
              const dist = acousticEda.feature_distributions[key];
              const buckets = featureHistogramBuckets[key];
              return (
                <div key={key} className="grid grid-cols-2 gap-2">
                  <ChartCard title={`${key.replace(/_/g, " ")} — distribution`} explanation={getFeatureExplanation(key)}>
                    <HistogramChart
                      bins={dist.bins}
                      histogram={dist.histogram}
                      label={key}
                      color="hsl(var(--primary))"
                      customdata={buckets}
                      onBarClick={buckets ? (filenames) => handleBucketClick(`feature:${key}`, filenames) : undefined}
                    />
                  </ChartCard>
                  <ChartCard title={`${key.replace(/_/g, " ")} — spread`} explanation={EDA_CHART_EXPLANATIONS.feature_box_plot}>
                    <BoxPlotChart min={stats.min} q1={stats.q1} median={stats.median} q3={stats.q3} max={stats.max} label={key} />
                  </ChartCard>
                </div>
              );
            })}

            {Object.keys(perClassQuartiles).length > 0 && (
              <div className="space-y-2">
                <div className="text-xs font-medium px-1">Acoustic features by class</div>
                {Object.entries(perClassQuartiles).map(([key, byClass]) => (
                  <ChartCard
                    key={key}
                    title={`${key.replace(/_/g, " ")} by class`}
                    explanation="Boxes that barely overlap across classes show the model has real acoustic signal to separate on."
                  >
                    <div className="grid h-full gap-1" style={{ gridTemplateColumns: `repeat(${Object.keys(byClass).length}, minmax(0, 1fr))` }}>
                      {Object.entries(byClass).map(([className, q]) => (
                        <div key={className} className="h-full flex flex-col min-w-0">
                          <div className="text-[9px] text-center text-muted-foreground truncate">{className}</div>
                          <div className="flex-1 min-h-0">
                            <BoxPlotChart min={q.min} q1={q.q1} median={q.median} q3={q.q3} max={q.max} label={className} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </ChartCard>
                ))}
              </div>
            )}

            <ChartCard title="Feature correlation" explanation={EDA_CHART_EXPLANATIONS.correlation_heatmap} resize="both">
              <CorrelationHeatmap
                featuresByFile={featuresByFile}
                featureKeys={rankedFeatureKeys.slice(0, CORRELATION_FEATURE_LIMIT)}
              />
            </ChartCard>

            {correlationPairs.length > 0 && (
              <div className="text-xs-tight space-y-1 px-1">
                <div className="font-medium text-foreground">Highly correlated pairs</div>
                {correlationPairs.map((pair) => (
                  <div key={`${pair.a}-${pair.b}`} className="text-muted-foreground">
                    <span className="font-mono text-[11px]">{pair.a}</span> &harr; <span className="font-mono text-[11px]">{pair.b}</span>
                    {" — r="}{pair.r.toFixed(2)}{Math.abs(pair.r) > 0.9 ? " — likely redundant" : ""}
                  </div>
                ))}
              </div>
            )}

            {outliers.length > 0 && (
              <div className="space-y-1.5">
                <div className="flex items-center gap-1.5 px-1">
                  <AlertTriangle className="h-3 w-3 text-amber-500" />
                  <span className="text-xs font-medium">Statistical outliers</span>
                  <Tooltip>
                    <TooltipTrigger>
                      <HelpCircle className="h-3 w-3 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs text-xs">
                      Files where a feature value sits more than {OUTLIER_Z_THRESHOLD} standard deviations from the dataset mean. This flags statistical anomalies, not confirmed defects — click a row to inspect the file.
                    </TooltipContent>
                  </Tooltip>
                </div>
                <div className="max-h-40 overflow-y-auto space-y-1">
                  {outliers.map((o, idx) => (
                    <button
                      key={`${o.filename}-${o.feature}-${idx}`}
                      type="button"
                      onClick={() => onFileSelect?.(o.filename)}
                      disabled={!onFileSelect}
                      className="w-full text-left text-xs-tight p-1.5 bg-muted/50 rounded border border-border hover:bg-muted disabled:cursor-default disabled:hover:bg-muted/50 flex items-center justify-between gap-2"
                    >
                      <span className="font-mono truncate">{o.filename}</span>
                      <span className="text-muted-foreground shrink-0">{o.feature.replace(/_/g, " ")} {o.z > 0 ? "+" : ""}{o.z.toFixed(1)}&sigma;</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
