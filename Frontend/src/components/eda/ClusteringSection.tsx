import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { HelpCircle, Network } from "lucide-react";
import { useEmbedding } from "@/contexts/EmbeddingContext";
import { EDA_CHART_EXPLANATIONS } from "@/lib/audioFeatures";
import { ChartCard } from "./ChartCard";
import { ClusterLabelHeatmap } from "./ClusterLabelHeatmap";

interface ClusteringSectionProps {
  /** filename -> ground-truth class, from the metadata EDA payload. */
  labelsByFile?: Record<string, string>;
}

/**
 * Clusters vs ground-truth labels.
 *
 * This is the only clustering view that belongs in the EDA tab: it compares the
 * clusters against dataset metadata rather than describing the scatter. The
 * silhouette score, cluster breakdown, and noise list live beside the plot in the
 * Embeddings tab (see `ClusterInsights`).
 *
 * Renders nothing for datasets with no labels — `common-voice` has none, and an
 * empty cross-tab would be noise rather than information.
 */
export const ClusteringSection = ({ labelsByFile }: ClusteringSectionProps) => {
  const { embeddingData } = useEmbedding();
  const clustering = embeddingData?.clustering;
  const hasLabels = labelsByFile && Object.keys(labelsByFile).length > 0;

  if (!hasLabels) return null;

  const header = (
    <div className="flex items-center gap-1.5">
      <Network className="h-3.5 w-3.5 text-primary" />
      <span className="text-xs font-medium">Clusters vs labels</span>
      <Tooltip>
        <TooltipTrigger>
          <HelpCircle className="h-3 w-3 text-muted-foreground" />
        </TooltipTrigger>
        <TooltipContent className="max-w-xs text-xs">
          {EDA_CHART_EXPLANATIONS.cluster_label_crosstab}
        </TooltipContent>
      </Tooltip>
    </div>
  );

  if (!clustering || clustering.error || clustering.n_clusters === 0 || !embeddingData?.clusterByFilename) {
    return (
      <div className="border-t border-border pt-3 space-y-2">
        {header}
        <div className="text-xs text-muted-foreground p-3 bg-muted/50 rounded-md border border-border">
          Turn on the cluster view in the Embeddings tab to compare clusters against this
          dataset's labels.
        </div>
      </div>
    );
  }

  return (
    <div className="border-t border-border pt-3 space-y-2">
      {header}
      <ChartCard
        title="Cluster × label counts"
        explanation={EDA_CHART_EXPLANATIONS.cluster_label_crosstab}
        resize="both"
      >
        <ClusterLabelHeatmap
          clusterByFilename={embeddingData.clusterByFilename}
          labelsByFile={labelsByFile}
        />
      </ChartCard>
    </div>
  );
};
