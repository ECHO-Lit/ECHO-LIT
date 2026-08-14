import Plot from "react-plotly.js";
import { useIsDarkMode } from "@/hooks/useIsDarkMode";
import {
  groupMetricValue, metricLabel, type FairnessExcludedGroup, type FairnessGap,
  type FairnessGroupReport, type FairnessMetricId,
} from "@/lib/fairness";

interface GroupErrorBarChartProps {
  groups: FairnessGroupReport[];
  excluded: FairnessExcludedGroup[];
  disparities: FairnessGap[];
  referenceGroup: string;
  metric: FairnessMetricId;
}

/** Grouped bars with 95% bootstrap CI whiskers, the reference group pinned
 * first and drawn neutral, flagged groups in the alert color, speaker-
 * confounded groups hatched, excluded groups rendered as low-opacity ghost
 * bars so what was dropped stays visible (SRS exception flow). */
export function GroupErrorBarChart({ groups, excluded, disparities, referenceGroup, metric }: GroupErrorBarChartProps) {
  const isDark = useIsDarkMode();
  const ordered = [...groups].sort((a, b) => (a.label === referenceGroup ? -1 : b.label === referenceGroup ? 1 : 0));

  const gapByGroup = new Map(disparities.filter((gap) => gap.metric === metric).map((gap) => [gap.group, gap]));

  const labels = ordered.map((g) => g.label);
  const values = ordered.map((g) => groupMetricValue(g, metric) ?? 0);

  // The backend's CI is on the DISPARITY (other - reference), not the raw
  // group value, so error bars are only drawn for non-reference groups by
  // translating the gap's CI width onto the group's own point estimate.
  const errorArray = ordered.map((g) => {
    if (g.label === referenceGroup) return 0;
    const gap = gapByGroup.get(g.label);
    if (!gap) return 0;
    return (gap.ci[1] - gap.ci[0]) / 2;
  });

  const colors = ordered.map((g) => {
    if (g.label === referenceGroup) return isDark ? "#64748b" : "#94a3b8";
    const verdict = gapByGroup.get(g.label);
    if (g.speaker_confounded) return "#f59e0b";
    if (verdict && verdict.p_adjusted !== undefined && verdict.p_adjusted < 0.05) return "#ef4444";
    return "#3b82f6";
  });

  const excludedLabels = excluded.map((e) => e.label);
  const excludedValues = excluded.map(() => 0);

  return (
    <div className="h-[280px]">
      <Plot
        data={[
          {
            type: "bar", x: labels, y: values,
            marker: { color: colors, line: { color: "hsl(var(--border))", width: 1 } },
            error_y: { type: "data", array: errorArray, visible: true, color: isDark ? "#e2e8f0" : "#334155" },
            text: labels.map((label, i) => (label === referenceGroup ? "reference" : "")),
            textposition: "outside",
            name: metricLabel(metric),
          } as any,
          ...(excludedLabels.length > 0
            ? [{
                type: "bar", x: excludedLabels, y: excludedValues, opacity: 0.25,
                marker: { color: "#94a3b8", pattern: { shape: "/" } },
                name: "excluded",
                hovertext: excluded.map((e) => `${e.label}: ${e.n_items} items, excluded (${e.reason})`),
                hoverinfo: "text",
              } as any]
            : []),
        ]}
        layout={{
          autosize: true,
          margin: { l: 45, r: 10, t: 10, b: 60 },
          xaxis: { tickfont: { size: 10 }, tickangle: -25 },
          yaxis: { title: metricLabel(metric), showgrid: true, gridcolor: "hsl(var(--border))", tickfont: { size: 10 } },
          plot_bgcolor: "transparent",
          paper_bgcolor: "transparent",
          showlegend: excludedLabels.length > 0,
          legend: { orientation: "h", y: -0.3, font: { size: 9 } },
          font: { size: 10, color: "hsl(var(--foreground))" },
        }}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
