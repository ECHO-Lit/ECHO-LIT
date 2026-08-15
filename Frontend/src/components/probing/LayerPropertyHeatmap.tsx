/**
 * Property x layer grid of *selectivity* — the "where does information live" view.
 *
 * Selectivity (accuracy minus the shuffled-label control) has a meaningful zero:
 * at or below it, the probe found nothing the model actually encodes. So the
 * scale is diverging around 0 with a neutral midpoint, not sequential — a
 * sequential ramp would paint "no information" as merely "a bit less" of
 * something.
 *
 * Rows are ordered shallow -> deep by peak, so the emergence story reads down
 * the chart.
 */

import { useMemo } from "react";
import Plot from "react-plotly.js";

import { useIsDarkMode } from "@/hooks/useIsDarkMode";
import { byPeakDepth, propertyLabel, type LayerProbeResult } from "@/lib/probes";

interface LayerPropertyHeatmapProps {
  result: LayerProbeResult;
  properties: string[];
}

// Two hues plus a neutral midpoint. Negative = the probe beat nothing.
const DIVERGING_LIGHT: Array<[number, string]> = [
  [0, "#b8563a"],
  [0.5, "#eeece7"],
  [1, "#2a78d6"],
];
const DIVERGING_DARK: Array<[number, string]> = [
  [0, "#c9674a"],
  [0.5, "#33322f"],
  [1, "#3987e5"],
];

export const LayerPropertyHeatmap = ({ result, properties }: LayerPropertyHeatmapProps) => {
  const isDark = useIsDarkMode();

  const chart = useMemo(() => {
    const ordered = byPeakDepth(result)
      .filter(([key]) => properties.includes(key))
      .reverse(); // Plotly draws the first row at the bottom.
    if (ordered.length === 0) return null;

    const z = ordered.map(([, probe]) => probe.layers.map((layer) => layer.selectivity));
    const rows = ordered.map(([key]) => propertyLabel(key));
    const columns = result.layer_names.map((name) =>
      name === "input" ? "input" : name.replace("layer_", ""),
    );

    const finite = z.flat().filter((value): value is number => typeof value === "number");
    const extent = Math.max(0.1, ...finite.map((value) => Math.abs(value)));

    // Ring the peak layer of each property: the single cell each row's headline
    // number comes from.
    const shapes = ordered.map(([, probe], row) => ({
      type: "rect" as const,
      x0: (probe.best_layer ?? 0) - 0.5,
      x1: (probe.best_layer ?? 0) + 0.5,
      y0: row - 0.5,
      y1: row + 0.5,
      line: { color: isDark ? "#f5f3ee" : "#1f1f1e", width: 2 },
      fillcolor: "rgba(0,0,0,0)",
    }));

    return { z, rows, columns, extent, shapes, ordered };
  }, [result, properties, isDark]);

  if (!chart) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
        No property produced a usable probe
      </div>
    );
  }

  const ink = isDark ? "#e6e4df" : "#1f1f1e";

  return (
    <div className="h-full">
      <Plot
        data={[
          {
            z: chart.z,
            x: chart.columns,
            y: chart.rows,
            type: "heatmap",
            colorscale: isDark ? DIVERGING_DARK : DIVERGING_LIGHT,
            zmid: 0,
            zmin: -chart.extent,
            zmax: chart.extent,
            xgap: 2,
            ygap: 2,
            colorbar: {
              thickness: 8,
              tickfont: { size: 8, color: ink },
              title: { text: "selectivity", font: { size: 9, color: ink }, side: "right" },
            },
            hovertemplate:
              "<b>%{y}</b> — layer %{x}<br>selectivity %{z:.3f}<extra></extra>",
          } as any,
        ]}
        layout={{
          autosize: true,
          margin: { l: 96, r: 10, t: 8, b: 34 },
          shapes: chart.shapes,
          xaxis: {
            title: { text: "encoder layer", font: { size: 10, color: ink } },
            tickfont: { size: 9, color: ink },
            side: "bottom",
          },
          yaxis: { tickfont: { size: 9, color: ink }, automargin: true },
          plot_bgcolor: "transparent",
          paper_bgcolor: "transparent",
          font: { size: 10, color: ink },
        }}
        config={{
          displayModeBar: "hover",
          displaylogo: false,
          modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
          responsive: true,
        }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
};
