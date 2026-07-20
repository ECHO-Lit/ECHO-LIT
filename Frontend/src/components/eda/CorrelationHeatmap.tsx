import { useMemo } from "react";
import Plot from "react-plotly.js";

interface CorrelationHeatmapProps {
  // filename -> { featureKey -> value }
  featuresByFile: Record<string, Record<string, number>>;
  // limit to keep the matrix legible; caller can pass a pre-filtered feature list
  featureKeys: string[];
}

function pearson(a: number[], b: number[]): number {
  const n = a.length;
  if (n === 0) return 0;
  const meanA = a.reduce((s, v) => s + v, 0) / n;
  const meanB = b.reduce((s, v) => s + v, 0) / n;
  let num = 0, denomA = 0, denomB = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    num += da * db;
    denomA += da * da;
    denomB += db * db;
  }
  const denom = Math.sqrt(denomA * denomB);
  return denom === 0 ? 0 : num / denom;
}

export const CorrelationHeatmap = ({ featuresByFile, featureKeys }: CorrelationHeatmapProps) => {
  const matrix = useMemo(() => {
    const files = Object.keys(featuresByFile);
    return featureKeys.map((keyA) =>
      featureKeys.map((keyB) => {
        const a = files.map((f) => featuresByFile[f][keyA]).filter((v) => typeof v === "number");
        const b = files.map((f) => featuresByFile[f][keyB]).filter((v) => typeof v === "number");
        const n = Math.min(a.length, b.length);
        return n > 1 ? pearson(a.slice(0, n), b.slice(0, n)) : 0;
      }),
    );
  }, [featuresByFile, featureKeys]);

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
