import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { RefreshCw } from "lucide-react";

interface ClusterSizeControlProps {
  value: number;
  onChange: (value: number) => void;
  /** The value the current result was actually computed with. */
  appliedValue?: number;
  onApply: (value: number) => void;
  isBusy?: boolean;
}

/**
 * HDBSCAN's `min_cluster_size` knob.
 *
 * Lives beside the scatter's cluster toggle rather than in the EDA tab, because it
 * changes what the scatter shows -- the control and its effect stay in one place.
 */
export const ClusterSizeControl = ({
  value,
  onChange,
  appliedValue,
  onApply,
  isBusy,
}: ClusterSizeControlProps) => (
  <div className="flex items-center gap-2 px-1">
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="text-[10px] text-muted-foreground shrink-0 cursor-help">
          Min cluster size
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs text-xs">
        Smallest group HDBSCAN will call a cluster — not a target cluster count. Lower values
        find more, smaller clusters; higher values keep only large dense groups and push the
        rest to noise.
      </TooltipContent>
    </Tooltip>
    <Slider
      value={[value]}
      min={2}
      max={50}
      step={1}
      onValueChange={([next]) => onChange(next)}
      className="flex-1"
    />
    <span className="text-[10px] tabular-nums w-5 text-right">{value}</span>
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          size="sm"
          variant="secondary"
          className="h-6 text-[10px]"
          disabled={isBusy || value === appliedValue}
          onClick={() => onApply(value)}
        >
          <RefreshCw className={`h-3 w-3 mr-1 ${isBusy ? "animate-spin" : ""}`} />
          Recluster
        </Button>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs text-xs">
        Re-runs clustering with the new setting. Embeddings are cached per file, so the model
        does not re-run — only the grouping is recomputed.
      </TooltipContent>
    </Tooltip>
  </div>
);
