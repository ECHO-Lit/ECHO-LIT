/**
 * Claimed importance versus measured cost, one point per segment.
 *
 * The most directly interpretable view in the feature: the x axis is what the
 * saliency map said a segment was worth, the y axis is what removing that
 * segment actually cost the model. A faithful map puts the points on a rising
 * line. A cloud means the map's peaks are decorative.
 *
 * The y=0 line is drawn because points below it matter: a segment the map called
 * important whose removal *helped* the model is evidence against the map, and
 * without the reference line it reads as just another low point.
 */

import { useMemo } from "react";
import Plot from "react-plotly.js";

import { useIsDarkMode } from "@/hooks/useIsDarkMode";
import { type FaithfulnessResult } from "@/lib/faithfulness";

interface OcclusionScatterProps {
  result: FaithfulnessResult;
}

export const OcclusionScatter = ({ result }: OcclusionScatterProps) => {
  const isDark = useIsDarkMode();
  const points = result.occlusion;

  const traces = useMemo(() => {
    if (points.length === 0) return [];
    const colour = isDark ? "#fb923c" : "#c2410c";

    const drawn: any[] = [
      {
        x: points.map((point) => point.saliency),
        y: points.map((point) => point.drop),
        type: "scatter",
        mode: "markers",
        name: "segments",
        marker: { color: colour, size: 9, opacity: 0.8, line: { width: 0 } },
        customdata: points.map((point) => [
          point.word ?? "—",
          point.start_time,
          point.end_time,
        ]),
        hovertemplate:
          "<b>%{customdata[0]}</b><br>" +
          "%{customdata[1]:.2f}s – %{customdata[2]:.2f}s<br>" +
          "map says %{x:.3f} · removing it costs %{y:.3f}<extra></extra>",
        showlegend: false,
      },
    ];

    // Least-squares trend, drawn only when it can be fitted, as a reading aid
    // for the direction the rank correlation reports.
    const n = points.length;
    const xs = points.map((point) => point.saliency);
    const ys = points.map((point) => point.drop);
    const meanX = xs.reduce((a, b) => a + b, 0) / n;
    const meanY = ys.reduce((a, b) => a + b, 0) / n;
    const varianceX = xs.reduce((sum, x) => sum + (x - meanX) ** 2, 0);
    if (varianceX > 1e-12) {
      const slope =
        xs.reduce((sum, x, index) => sum + (x - meanX) * (ys[index] - meanY), 0) / varianceX;
      const intercept = meanY - slope * meanX;
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      drawn.push({
        x: [minX, maxX],
        y: [slope * minX + intercept, slope * maxX + intercept],
        type: "scatter",
        mode: "lines",
        line: { color: isDark ? "#94a3b8" : "#64748b", width: 1.5, dash: "dash" },
        hoverinfo: "skip",
        showlegend: false,
      });
    }
    return drawn;
  }, [points, isDark]);

  if (traces.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground px-4 text-center">
        Per-segment occlusion was not run for this clip.
      </div>
    );
  }

  const ink = isDark ? "#e6e4df" : "#1f1f1e";
  const grid = isDark ? "#3a3a38" : "#e4e2dd";
  const rho = result.metrics.occlusion_spearman;

  return (
    <div className="h-full">
      <Plot
        data={traces}
        layout={{
          autosize: true,
          margin: { l: 54, r: 12, t: 24, b: 42 },
          title: {
            text:
              rho === null
                ? "not enough variation to correlate"
                : `rank agreement ρ = ${rho.toFixed(2)}`,
            font: { size: 10, color: ink },
            x: 0,
          },
          shapes: [
            {
              type: "line",
              xref: "paper",
              x0: 0,
              x1: 1,
              y0: 0,
              y1: 0,
              line: { color: grid, width: 1 },
              layer: "below",
            },
          ],
          xaxis: {
            title: { text: "saliency the map assigned", font: { size: 10, color: ink } },
            tickfont: { size: 9, color: ink },
            gridcolor: grid,
            zeroline: false,
          },
          yaxis: {
            title: { text: "cost of removing it", font: { size: 10, color: ink } },
            tickfont: { size: 9, color: ink },
            gridcolor: grid,
            zeroline: false,
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
