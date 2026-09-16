/**
 * Deletion curves: what the model's score does as audio is progressively removed.
 *
 * Three rules make this chart evidence rather than decoration:
 *
 *  - The random baseline is drawn on the same axes, dashed, with its spread as a
 *    band. The gap between the two lines *is* the finding; a solid line on its
 *    own would let ordinary masking damage read as faithfulness.
 *  - The "least salient first" line is drawn too. A trustworthy map separates
 *    the two orderings — most-salient-first should fall fastest, least-salient-
 *    first slowest, and the random baseline should sit between them.
 *  - The y axis is pinned to [0, 1]. Auto-scaling would inflate a two-percent
 *    wobble into what looks like a collapse.
 */

import { useMemo } from "react";
import Plot from "react-plotly.js";

import { useIsDarkMode } from "@/hooks/useIsDarkMode";
import { targetDescription, type FaithfulnessResult } from "@/lib/faithfulness";

interface FaithfulnessCurveProps {
  result: FaithfulnessResult;
  /** Show the insertion curve (keep only the top-k) alongside the deletions. */
  showInsertion?: boolean;
}

export const FaithfulnessCurve = ({ result, showInsertion = false }: FaithfulnessCurveProps) => {
  const isDark = useIsDarkMode();

  const traces = useMemo(() => {
    const { deletion_saliency, deletion_random, deletion_inverse, insertion_saliency } =
      result.curves;
    if (deletion_saliency.length === 0) return [];

    // Anchor every curve at 0% masked = the clean score, so the lines start
    // together and the reader sees divergence rather than an unexplained offset.
    const withOrigin = (points: typeof deletion_saliency) => ({
      x: [0, ...points.map((point) => point.fraction * 100)],
      y: [result.baseline_score, ...points.map((point) => point.score)],
    });

    const salient = withOrigin(deletion_saliency);
    const random = withOrigin(deletion_random);
    const inverse = withOrigin(deletion_inverse);
    const spread = [0, ...deletion_random.map((point) => point.std ?? 0)];

    const drawn: any[] = [
      // Per-fraction spread of the random draws, drawn first so the lines sit on
      // top of it. This is the spread at each point, not the error bar on the
      // headline gain (`aopc_random_stderr`) — a wider band here is expected.
      {
        x: random.x,
        y: random.y.map((value, index) => value + spread[index]),
        type: "scatter", mode: "lines", line: { width: 0 },
        hoverinfo: "skip", showlegend: false,
      },
      {
        x: random.x,
        y: random.y.map((value, index) => value - spread[index]),
        type: "scatter", mode: "lines", fill: "tonexty",
        fillcolor: isDark ? "rgba(148,163,184,0.18)" : "rgba(100,116,139,0.15)",
        line: { width: 0 }, hoverinfo: "skip", showlegend: false,
      },
      {
        ...random,
        type: "scatter", mode: "lines+markers",
        name: "Random audio (baseline)",
        line: { color: isDark ? "#94a3b8" : "#64748b", width: 2, dash: "dash" },
        marker: { size: 6, symbol: "square" },
        hovertemplate: "random · %{x:.0f}% removed<br>score %{y:.3f}<extra></extra>",
      },
      {
        ...inverse,
        type: "scatter", mode: "lines+markers",
        name: "Least salient first",
        line: { color: isDark ? "#5eead4" : "#0f766e", width: 2, dash: "dot" },
        marker: { size: 6, symbol: "diamond" },
        hovertemplate: "least salient · %{x:.0f}% removed<br>score %{y:.3f}<extra></extra>",
      },
      {
        ...salient,
        type: "scatter", mode: "lines+markers",
        name: "Most salient first",
        line: { color: isDark ? "#fb923c" : "#c2410c", width: 3 },
        marker: { size: 8 },
        hovertemplate: "most salient · %{x:.0f}% removed<br>score %{y:.3f}<extra></extra>",
      },
    ];

    if (showInsertion && insertion_saliency.length > 0) {
      drawn.push({
        x: insertion_saliency.map((point) => point.fraction * 100),
        y: insertion_saliency.map((point) => point.score),
        type: "scatter", mode: "lines+markers",
        name: "Keep only the salient audio",
        line: { color: isDark ? "#c4b5fd" : "#6d28d9", width: 2, dash: "dashdot" },
        marker: { size: 6, symbol: "triangle-up" },
        hovertemplate: "kept top %{x:.0f}%<br>score %{y:.3f}<extra></extra>",
      });
    }
    return drawn;
  }, [result, showInsertion, isDark]);

  if (traces.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground px-4 text-center">
        {result.skipped_reason ?? "No curve was produced for this clip."}
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
          margin: { l: 52, r: 12, t: 8, b: 42 },
          xaxis: {
            title: { text: "% of audio removed", font: { size: 10, color: ink } },
            tickfont: { size: 9, color: ink },
            gridcolor: grid,
            zeroline: false,
            range: [0, 100],
          },
          yaxis: {
            // Pinned, never auto-scaled: see the header comment.
            range: [0, 1],
            title: { text: targetDescription(result.target), font: { size: 10, color: ink } },
            tickfont: { size: 9, color: ink },
            gridcolor: grid,
            zeroline: false,
          },
          legend: {
            orientation: "h",
            y: -0.24,
            font: { size: 9, color: ink },
            bgcolor: "transparent",
          },
          hovermode: "x unified",
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
