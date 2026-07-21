import { useMemo } from "react";
import Plot from "react-plotly.js";
import { correlationMatrix } from "@/lib/edaStats";

interface CorrelationHeatmapProps {
  // filename -> { featureKey -> value }
  featuresByFile: Record<string, Record<string, number>>;
  // limit to keep the matrix legible; caller can pass a pre-filtered feature list
  featureKeys: string[];
}

export const CorrelationHeatmap = ({ featuresByFile, featureKeys }: CorrelationHeatmapProps) => {
  const matrix = useMemo(
    () => correlationMatrix(featuresByFile, featureKeys),
    [featuresByFile, featureKeys],
  );

  if (featureKeys.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
        No features available
      </div>
    );
  }

  return (
    <div className="h-full">
      <Plot
        data={[
          {
            z: matrix,
            x: featureKeys,
            y: featureKeys,
            type: "heatmap",
            colorscale: "RdBu",
            zmin: -1,
            zmax: 1,
            colorbar: { thickness: 10, tickfont: { size: 8 } },
          },
        ]}
        layout={{
          autosize: true,
          margin: { l: 90, r: 10, t: 10, b: 90 },
          xaxis: { tickfont: { size: 8 }, tickangle: -45 },
          yaxis: { tickfont: { size: 8 } },
          plot_bgcolor: "transparent",
          paper_bgcolor: "transparent",
          font: { size: 10, color: "hsl(var(--foreground))" },
        }}
        config={{
          displayModeBar: "hover",
          displaylogo: false,
          modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
          scrollZoom: true,
          responsive: true,
        }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
};
