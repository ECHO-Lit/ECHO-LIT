import Plot from "react-plotly.js";
import { useIsDarkMode } from "@/hooks/useIsDarkMode";
import { LOWER_IS_BETTER, metricLabel, type FairnessGap, type FairnessMetricId } from "@/lib/fairness";

interface DisparityRadarProps {
  disparities: FairnessGap[];
  referenceGroup: string;
}

const COLORS = ["#3b82f6", "#f97316", "#10b981", "#a855f7", "#ef4444", "#0ea5e9"];

/** One axis per metric, one polygon per non-reference group, every axis
 * normalised to "disparity ratio relative to the reference group, oriented
 * so further-from-center always means worse for that group" -- WER (lower is
 * better) and grounding_lift (higher is better) would otherwise point in
 * opposite directions on the same chart, which reads as backwards. */
export function DisparityRadar({ disparities, referenceGroup }: DisparityRadarProps) {
  const isDark = useIsDarkMode();
  const metrics = Array.from(new Set(disparities.map((d) => d.metric))) as FairnessMetricId[];
  const groups = Array.from(new Set(disparities.map((d) => d.group))).filter((g) => g !== referenceGroup);

  if (metrics.length < 3 || groups.length === 0) {
    return (
      <div className="flex h-[240px] items-center justify-center text-xs text-muted-foreground border border-dashed border-border rounded-md">
        Radar needs at least 3 comparable metrics.
      </div>
    );
  }

  const byGroupMetric = new Map(disparities.map((d) => [`${d.group}|${d.metric}`, d]));

  const traces = groups.map((group, index) => {
    const values = metrics.map((metric) => {
      const gap = byGroupMetric.get(`${group}|${metric}`);
      if (!gap) return 0;
      // "Worse for this group" = positive ratio-1 for lower-is-better
      // metrics, negative for higher-is-better ones -- flip the sign so both
      // read as "further out = worse" on a shared axis.
      const disparity = gap.ratio - 1;
      return LOWER_IS_BETTER.has(metric) ? disparity : -disparity;
    });
    return {
      type: "scatterpolar", r: [...values, values[0]],
      theta: [...metrics.map(metricLabel), metricLabel(metrics[0])],
      fill: "toself", name: group, opacity: 0.55,
      line: { color: COLORS[index % COLORS.length] },
    } as any;
  });

  return (
    <div className="h-[280px]">
      <Plot
        data={traces}
        layout={{
          autosize: true,
          margin: { l: 30, r: 30, t: 20, b: 20 },
          polar: {
            radialaxis: { visible: true, tickfont: { size: 8 }, gridcolor: "hsl(var(--border))" },
            angularaxis: { tickfont: { size: 9 } },
            bgcolor: "transparent",
          },
          paper_bgcolor: "transparent",
          showlegend: true,
          legend: { orientation: "h", y: -0.15, font: { size: 9 } },
          font: { size: 10, color: "hsl(var(--foreground))" },
        }}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
