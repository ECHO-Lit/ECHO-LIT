import { useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { AlertTriangle, Download, HelpCircle, Network, Play } from "lucide-react";
import { useEmbedding } from "@/contexts/EmbeddingContext";
import { useIsDarkMode } from "@/hooks/useIsDarkMode";
import { EDA_CHART_EXPLANATIONS } from "@/lib/audioFeatures";
import { clusterColor, clusterName, silhouetteBand, NOISE_LABEL } from "@/lib/clusterPalette";
import { exportClustersCsv } from "@/lib/edaExport";

interface ClusterInsightsProps {
  dataset: string;
  onFileSelect?: (filename: string) => void;
  /** Toggle the file-table filter. Shared with the EDA tab's charts, so one
      filter is active across the whole panel at a time. */
  onBucketClick: (key: string, filenames: string[]) => void;
  activeFilterKey: string | null;
}

const NOISE_LIST_LIMIT = 25;

/**
 * Silhouette score, cluster breakdown, and the unclustered-noise list.
 *
 * Lives in the Embeddings tab beside the scatter it describes — the numbers and
 * the picture they explain stay together. The EDA tab keeps only the
 * clusters-vs-labels cross-tab, which is dataset analysis rather than a readout
 * of the plot.
 */
export const ClusterInsights = ({
  dataset,
  onFileSelect,
  onBucketClick,
  activeFilterKey,
}: ClusterInsightsProps) => {
  const { embeddingData } = useEmbedding();
  const isDark = useIsDarkMode();
  const clustering = embeddingData?.clustering;

  const filenames = useMemo(
    () => embeddingData?.embeddings.map((entry) => entry.filename) ?? [],
    [embeddingData],
  );

  const filenamesByCluster = useMemo(() => {
    const byCluster: Record<number, string[]> = {};
    if (!clustering?.labels) return byCluster;
    clustering.labels.forEach((label, index) => {
      const filename = filenames[index];
      if (!filename) return;
      (byCluster[label] ||= []).push(filename);
    });
    return byCluster;
  }, [clustering, filenames]);

  // Noise points, least-confident first -- the most anomalous files lead.
  const noiseFiles = useMemo(() => {
    if (!clustering?.labels) return [];
    const rows = clustering.labels
      .map((label, index) => ({
        label,
        filename: filenames[index],
        probability: clustering.probabilities?.[index],
      }))
      .filter((row) => row.label === NOISE_LABEL && Boolean(row.filename));
    rows.sort((a, b) => (a.probability ?? 0) - (b.probability ?? 0));
    return rows.slice(0, NOISE_LIST_LIMIT);
  }, [clustering, filenames]);

  if (clustering?.error) {
    return (
      <div className="text-xs text-destructive p-3 bg-destructive/5 rounded-sm border border-destructive/20">
        Clustering failed: {clustering.error}
      </div>
    );
  }

  if (!clustering || clustering.n_clusters === 0) {
    return (
      <div className="text-xs text-muted-foreground p-3 bg-muted/50 rounded-md border border-border">
        {clustering?.skipped_reason
          ? `No clusters — ${clustering.skipped_reason}.`
          : "HDBSCAN found no dense groups at this setting. Every file was labelled noise — try lowering the minimum cluster size above."}
      </div>
    );
  }

  const band = silhouetteBand(clustering.silhouette_score);
  const total = clustering.labels.length || 1;

  return (
    <div className="space-y-2">
      {/* Silhouette headline — the at-a-glance quality verdict. */}
      <div className="border border-border rounded-lg bg-card p-2.5 space-y-1.5">
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-semibold tabular-nums">
            {clustering.silhouette_score === null ? "—" : clustering.silhouette_score.toFixed(2)}
          </span>
          <span className="text-xs font-medium">{band.label} separation</span>
          <Tooltip>
            <TooltipTrigger>
              <HelpCircle className="h-3 w-3 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs text-xs">
              {EDA_CHART_EXPLANATIONS.silhouette_score}
            </TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                size="sm"
                variant="ghost"
                className="h-6 text-[10px] ml-auto"
                onClick={() => exportClustersCsv(clustering, filenames, dataset)}
              >
                <Download className="h-3 w-3 mr-1" />
                CSV
              </Button>
            </TooltipTrigger>
            <TooltipContent>Download per-file cluster assignments as CSV</TooltipContent>
          </Tooltip>
        </div>
        {/* Silhouette runs -1..1; map onto the bar so negative reads as left-of-centre. */}
        <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden relative">
          <div
            className="h-full bg-primary rounded-full"
            style={{ width: `${(((clustering.silhouette_score ?? 0) + 1) / 2) * 100}%` }}
          />
        </div>
        <div className="text-[10px] text-muted-foreground">{band.hint}</div>
        <div className="flex items-center gap-1.5 flex-wrap pt-0.5">
          <Badge variant="outline" className="text-[10px]">
            {clustering.n_clusters} clusters
          </Badge>
          <Badge variant="outline" className="text-[10px]">
            {clustering.n_noise} noise
          </Badge>
          {clustering.params.pca_dims && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Badge variant="outline" className="text-[10px] cursor-help">
                  PCA {clustering.params.pca_dims}d
                </Badge>
              </TooltipTrigger>
              <TooltipContent className="max-w-xs text-xs">
                Embeddings were reduced to {clustering.params.pca_dims} dimensions before
                clustering — density estimates degrade in very high dimensions.
              </TooltipContent>
            </Tooltip>
          )}
        </div>
      </div>

      {/* Cluster table. Doubles as the "relief" the light-mode palette requires:
          every cluster is named in text next to its swatch, so the chart's colours
          are never the only way to tell clusters apart. */}
      <div className="space-y-1">
        <div className="flex items-center gap-1.5 px-1">
          <Network className="h-3 w-3 text-primary" />
          <span className="text-xs font-medium">Clusters</span>
          <Tooltip>
            <TooltipTrigger>
              <HelpCircle className="h-3 w-3 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs text-xs">
              {EDA_CHART_EXPLANATIONS.clustering_scatter}
            </TooltipContent>
          </Tooltip>
          <span className="ml-auto text-[10px] text-muted-foreground">size · silhouette</span>
        </div>
        {clustering.cluster_stats.map((stat) => {
          const key = `cluster:${stat.label}`;
          const medoidFile = filenames[stat.medoid_index];
          return (
            <div
              key={stat.label}
              className={`flex items-center gap-2 p-1.5 rounded border text-xs-tight ${
                activeFilterKey === key ? "border-primary bg-primary/5" : "border-border bg-muted/50"
              }`}
            >
              <button
                type="button"
                onClick={() => onBucketClick(key, filenamesByCluster[stat.label] ?? [])}
                className="flex items-center gap-2 flex-1 min-w-0 text-left hover:opacity-80"
                title={`Filter the file table to ${clusterName(stat.label)}`}
              >
                <span
                  className="h-2.5 w-2.5 rounded-full shrink-0"
                  style={{ backgroundColor: clusterColor(stat.label, isDark) }}
                  aria-hidden
                />
                <span className="font-medium shrink-0">{clusterName(stat.label)}</span>
                <span className="h-1 flex-1 min-w-[1.5rem] bg-border rounded-full overflow-hidden">
                  <span
                    className="block h-full"
                    style={{
                      width: `${(stat.size / total) * 100}%`,
                      backgroundColor: clusterColor(stat.label, isDark),
                    }}
                  />
                </span>
                <span className="text-muted-foreground tabular-nums shrink-0">{stat.size}</span>
                <span className="text-muted-foreground tabular-nums shrink-0 w-8 text-right">
                  {stat.mean_silhouette === null ? "—" : stat.mean_silhouette.toFixed(2)}
                </span>
              </button>
              {medoidFile && onFileSelect && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-5 w-5 p-0 shrink-0"
                      onClick={() => onFileSelect(medoidFile)}
                    >
                      <Play className="h-3 w-3" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs text-xs">
                    Play {medoidFile} — the most typical clip in this cluster
                  </TooltipContent>
                </Tooltip>
              )}
            </div>
          );
        })}
      </div>

      {noiseFiles.length > 0 && (
        <div className="space-y-1.5">
          <div className="flex items-center gap-1.5 px-1">
            <AlertTriangle className="h-3 w-3 text-amber-500" />
            <span className="text-xs font-medium">Unclustered (noise)</span>
            <Tooltip>
              <TooltipTrigger>
                <HelpCircle className="h-3 w-3 text-muted-foreground" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs text-xs">
                {EDA_CHART_EXPLANATIONS.cluster_noise}
              </TooltipContent>
            </Tooltip>
            <button
              type="button"
              className="ml-auto text-[10px] text-primary hover:underline"
              onClick={() =>
                onBucketClick(`cluster:${NOISE_LABEL}`, filenamesByCluster[NOISE_LABEL] ?? [])
              }
            >
              {activeFilterKey === `cluster:${NOISE_LABEL}` ? "Clear filter" : "Filter table"}
            </button>
          </div>
          <div className="max-h-40 overflow-y-auto space-y-1">
            {noiseFiles.map((row) => (
              <button
                key={row.filename}
                type="button"
                onClick={() => onFileSelect?.(row.filename)}
                disabled={!onFileSelect}
                className="w-full text-left text-xs-tight p-1.5 bg-muted/50 rounded border border-border hover:bg-muted disabled:cursor-default disabled:hover:bg-muted/50 flex items-center justify-between gap-2"
              >
                <span className="font-mono truncate">{row.filename}</span>
                {row.probability !== undefined && (
                  <span className="text-muted-foreground shrink-0 tabular-nums">
                    {(row.probability * 100).toFixed(0)}% fit
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
