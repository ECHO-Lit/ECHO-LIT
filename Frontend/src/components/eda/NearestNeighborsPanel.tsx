import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { HelpCircle, Search } from "lucide-react";
import { useEmbedding } from "@/contexts/EmbeddingContext";
import { useIsDarkMode } from "@/hooks/useIsDarkMode";
import { EDA_CHART_EXPLANATIONS } from "@/lib/audioFeatures";
import { clusterColor, clusterName } from "@/lib/clusterPalette";
import {
  buildNeighborIndex,
  nearestNeighbors,
  type SimilarityMetric,
} from "@/lib/similarity";

interface NearestNeighborsPanelProps {
  /** Basename of the currently selected file, or null when nothing is selected. */
  selectedFile: string | null;
  onFileSelect?: (filename: string) => void;
}

const K_OPTIONS = [5, 10, 20];

export const NearestNeighborsPanel = ({ selectedFile, onFileSelect }: NearestNeighborsPanelProps) => {
  const { embeddingData } = useEmbedding();
  const isDark = useIsDarkMode();
  const [k, setK] = useState(5);
  const [metric, setMetric] = useState<SimilarityMetric>("cosine");

  // Pack vectors once; each selection change is then a single pass over the matrix.
  const index = useMemo(
    () => (embeddingData ? buildNeighborIndex(embeddingData.embeddings) : null),
    [embeddingData],
  );

  const neighbors = useMemo(() => {
    if (!index || !selectedFile) return [];
    return nearestNeighbors(selectedFile, index, k, metric, embeddingData?.clusterByFilename);
  }, [index, selectedFile, k, metric, embeddingData]);

  // Cosine is already 0..1-ish; euclidean needs normalising against the worst
  // result on screen so the bars stay comparable.
  const maxScore = useMemo(
    () => (neighbors.length ? Math.max(...neighbors.map((n) => Math.abs(n.score))) : 1),
    [neighbors],
  );

  const header = (
    <div className="flex items-center gap-1.5 px-1">
      <Search className="h-3 w-3 text-primary" />
      <span className="text-xs font-medium">Most similar clips</span>
      <Tooltip>
        <TooltipTrigger>
          <HelpCircle className="h-3 w-3 text-muted-foreground" />
        </TooltipTrigger>
        <TooltipContent className="max-w-xs text-xs">
          {EDA_CHART_EXPLANATIONS.nearest_neighbors}
        </TooltipContent>
      </Tooltip>
    </div>
  );

  if (!embeddingData) {
    return (
      <div className="space-y-1.5">
        {header}
        <div className="text-xs text-muted-foreground p-3 bg-muted/50 rounded-md border border-border">
          Generate embeddings in the Embeddings tab to find similar clips.
        </div>
      </div>
    );
  }

  if (!selectedFile) {
    return (
      <div className="space-y-1.5">
        {header}
        <div className="text-xs text-muted-foreground p-3 bg-muted/50 rounded-md border border-border">
          Select a file — in the table, the scatter, or any list here — to see its nearest neighbours.
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        {header}
        <div className="flex items-center gap-1">
          <select
            value={metric}
            onChange={(event) => setMetric(event.target.value as SimilarityMetric)}
            className="h-6 text-[10px] bg-background border border-border rounded px-1"
            aria-label="Similarity metric"
          >
            <option value="cosine">cosine</option>
            <option value="euclidean">euclidean</option>
          </select>
          <select
            value={k}
            onChange={(event) => setK(Number(event.target.value))}
            className="h-6 text-[10px] bg-background border border-border rounded px-1"
            aria-label="Number of neighbours"
          >
            {K_OPTIONS.map((option) => (
              <option key={option} value={option}>
                top {option}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="text-[10px] text-muted-foreground px-1 truncate">
        Similar to <span className="font-mono text-foreground">{selectedFile}</span>
      </div>

      {neighbors.length === 0 ? (
        <div className="text-xs text-muted-foreground p-3 bg-muted/50 rounded-md border border-border">
          This file has no embedding in the current result — regenerate embeddings to include it.
        </div>
      ) : (
        <div className="max-h-56 overflow-y-auto space-y-1">
          {neighbors.map((neighbor, rank) => (
            <button
              key={neighbor.filename}
              type="button"
              onClick={() => onFileSelect?.(neighbor.filename)}
              disabled={!onFileSelect}
              title={`${neighbor.filename} — ${metric} ${neighbor.score.toFixed(4)}`}
              className="w-full text-left text-xs-tight p-1.5 bg-muted/50 rounded border border-border hover:bg-muted disabled:cursor-default disabled:hover:bg-muted/50 flex items-center gap-2"
            >
              <span className="text-muted-foreground shrink-0 w-3 tabular-nums">{rank + 1}</span>
              {neighbor.cluster !== undefined && (
                <span
                  className="h-2 w-2 rounded-full shrink-0"
                  style={{ backgroundColor: clusterColor(neighbor.cluster, isDark) }}
                  aria-hidden
                />
              )}
              <span className="font-mono truncate flex-1 min-w-0">{neighbor.filename}</span>
              {neighbor.cluster !== undefined && (
                <Badge variant="outline" className="text-[9px] shrink-0 px-1">
                  {clusterName(neighbor.cluster)}
                </Badge>
              )}
              <span className="shrink-0 flex items-center gap-1">
                <span className="h-1 w-10 bg-border rounded-full overflow-hidden">
                  <span
                    className="block h-full bg-primary"
                    style={{ width: `${Math.min(100, (Math.abs(neighbor.score) / maxScore) * 100)}%` }}
                  />
                </span>
                <span className="text-muted-foreground tabular-nums w-8 text-right">
                  {neighbor.score.toFixed(2)}
                </span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
