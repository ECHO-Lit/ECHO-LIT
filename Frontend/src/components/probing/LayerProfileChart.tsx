/**
 * The primary layer-probing view: accuracy per layer, one line per property.
 *
 * Two rules make this chart readable rather than decorative:
 *
 *  - Every property's majority baseline is drawn on the same axes, in the same
 *    colour, dashed. An unadorned 0.61 is unreadable; 0.61 against a dashed
 *    0.125 is a finding.
 *  - The y axis is pinned to [0, 1]. Auto-scaling would stretch fold-level noise
 *    into what looks like structure.
 */

import { useMemo } from "react";
import Plot from "react-plotly.js";

import { useIsDarkMode } from "@/hooks/useIsDarkMode";
import { clusterColor, clusterDash, clusterSymbol } from "@/lib/clusterPalette";
import { propertyLabel, type LayerProbeResult } from "@/lib/probes";

interface LayerProfileChartProps {
  result: LayerProbeResult;
  /** Property keys to draw, in a stable order so colour never migrates. */
  properties: string[];
  /** Draw the +/-1 std band around each line. */
  showBands?: boolean;
}

export const LayerProfileChart = ({
  result,
  properties,
  showBands = true,
}: LayerProfileChartProps) => {
  const isDark = useIsDarkMode();

  const { traces, shapes } = useMemo(() => {
    const x = result.layer_names.map((_, index) => index);
    const drawn: any[] = [];
    const baselines: any[] = [];

    properties.forEach((key, index) => {
      const probe = result.properties[key];
      if (!probe || probe.best_layer === null) return;
      const colour = clusterColor(index, isDark);
      const accuracy = probe.layers.map((layer) => layer.accuracy);

      if (showBands) {
        const upper = probe.layers.map((layer) =>
          layer.accuracy === null ? null : Math.min(1, layer.accuracy + (layer.accuracy_std ?? 0)),
        );
        const lower = probe.layers.map((layer) =>
          layer.accuracy === null ? null : Math.max(0, layer.accuracy - (layer.accuracy_std ?? 0)),
        );
        drawn.push(
          {
            x, y: upper, type: "scatter", mode: "lines",
            line: { width: 0 }, hoverinfo: "skip", showlegend: false,
          },
          {
            x, y: lower, type: "scatter", mode: "lines", fill: "tonexty",
            fillcolor: `${colour}22`, line: { width: 0 },
            hoverinfo: "skip", showlegend: false,
          },
        );
      }

      drawn.push({
        x,
        y: accuracy,
        type: "scatter",
        mode: "lines+markers",
        name: propertyLabel(key),
        line: { color: colour, width: 2, dash: clusterDash(index) },
        marker: { color: colour, size: 8, symbol: clusterSymbol(index) },
        customdata: probe.layers.map((layer) => [
          layer.control_accuracy ?? NaN,
          layer.selectivity ?? NaN,
          layer.macro_f1 ?? NaN,
          layer.accuracy_std ?? 0,
          probe.majority_baseline ?? NaN,
          probe.n_samples,
          result.layer_names[layer.layer],
        ]),
        hovertemplate:
          `<b>${propertyLabel(key)}</b> — %{customdata[6]}<br>` +
          "accuracy %{y:.3f} ± %{customdata[3]:.3f}<br>" +
          "majority baseline %{customdata[4]:.3f}<br>" +
          "control %{customdata[0]:.3f} · selectivity %{customdata[1]:.3f}<br>" +
          "macro F1 %{customdata[2]:.3f} · n=%{customdata[5]}<extra></extra>",
      });

      if (probe.majority_baseline !== null) {
        baselines.push({
          type: "line",
          x0: -0.25,
          x1: result.num_layers - 0.75,
          y0: probe.majority_baseline,
          y1: probe.majority_baseline,
          line: { color: colour, width: 1, dash: "dot" },
          opacity: 0.6,
          layer: "below",
        });
      }
    });

    return { traces: drawn, shapes: baselines };
  }, [result, properties, showBands, isDark]);

  if (traces.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground px-4 text-center">
        No property produced a usable probe. Check the sample counts below.
      </div>
    );
  }

  const ink = isDark ? "#e6e4df" : "#1f1f1e";
  const grid = isDark ? "#3a3a38" : "#e4e2dd";

  return (
    <div className="h-full">
      <Plot
        data={traces}
        layout={{
          autosize: true,
          margin: { l: 44, r: 12, t: 8, b: 40 },
          shapes,
          xaxis: {
            title: { text: "encoder layer", font: { size: 10, color: ink } },
            tickmode: "array",
            tickvals: result.layer_names.map((_, index) => index),
            ticktext: result.layer_names.map((name) => (name === "input" ? "input" : name.replace("layer_", ""))),
            tickfont: { size: 9, color: ink },
            gridcolor: grid,
            zeroline: false,
            range: [-0.3, result.num_layers - 0.7],
          },
          yaxis: {
            // Pinned, never auto-scaled: see the header comment.
            range: [0, 1],
            title: { text: "probe accuracy", font: { size: 10, color: ink } },
            tickfont: { size: 9, color: ink },
            gridcolor: grid,
            zeroline: false,
          },
          legend: {
            orientation: "h",
            y: -0.22,
            font: { size: 9, color: ink },
            bgcolor: "transparent",
          },
          hovermode: "closest",
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
