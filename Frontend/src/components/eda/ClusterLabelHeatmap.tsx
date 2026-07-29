import { useMemo } from "react";
import Plot from "react-plotly.js";
import { clusterName } from "@/lib/clusterPalette";

interface ClusterLabelHeatmapProps {
  /** filename -> cluster label (-1 = noise) */
  clusterByFilename: Record<string, number>;
  /** filename -> ground-truth class label */
  labelsByFile: Record<string, string>;
}

// Sequential single-hue ramp (blue, light -> dark). Counts are magnitude, so this
// must stay one hue -- a categorical or rainbow scale would imply identity.
const BLUE_RAMP: Array<[number, string]> = [
  [0, "#f2f7fe"],
  [0.25, "#cde2fb"],
  [0.5, "#86b6ef"],
  [0.75, "#2a78d6"],
  [1, "#0d366b"],
];

/**
 * Cluster x ground-truth-label counts.
 *
 * Reads as: a cluster confined to one column means the acoustics separate that
 * class cleanly; a cluster smeared across columns means they do not.
 */
export const ClusterLabelHeatmap = ({ clusterByFilename, labelsByFile }: ClusterLabelHeatmapProps) => {
  const { z, xLabels, yLabels } = useMemo(() => {
    const classes = Array.from(new Set(Object.values(labelsByFile))).sort();
    const clusters = Array.from(new Set(Object.values(clusterByFilename))).sort((a, b) => a - b);
    const classIndex = Object.fromEntries(classes.map((name, i) => [name, i]));
    const clusterIndex = Object.fromEntries(clusters.map((label, i) => [label, i]));

    const counts = clusters.map(() => classes.map(() => 0));
    for (const [filename, cluster] of Object.entries(clusterByFilename)) {
      const className = labelsByFile[filename];
      if (className === undefined) continue;
      counts[clusterIndex[cluster]][classIndex[className]] += 1;
    }

    return {
      z: counts,
      xLabels: classes,
      yLabels: clusters.map((label) => clusterName(label)),
    };
  }, [clusterByFilename, labelsByFile]);

  if (z.length === 0 || xLabels.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
        No ground-truth labels to compare against
      </div>
    );
  }

  return (
    <div className="h-full">
      <Plot
        data={[
          {
            z,
            x: xLabels,
            y: yLabels,
            type: "heatmap",
            colorscale: BLUE_RAMP,
            zmin: 0,
            // Counts are integers; show them on the cells so identity never rests
            // on colour alone.
            text: z.map((row) => row.map((value) => (value ? String(value) : ""))),
            texttemplate: "%{text}",
            textfont: { size: 9 },
            hovertemplate: "%{y} · %{x}: %{z} files<extra></extra>",
            colorbar: { thickness: 10, tickfont: { size: 8 } },
          } as never,
        ]}
        layout={{
          autosize: true,
          margin: { l: 70, r: 10, t: 10, b: 70 },
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
