import Plot from "react-plotly.js";
import { useIsDarkMode } from "@/hooks/useIsDarkMode";
import type { PropertyProfile } from "@/lib/linguisticAcoustic";

const COLORS = ["#3b82f6", "#f97316", "#10b981", "#a855f7", "#ef4444"];

interface SensitivityProfileChartProps {
  profiles: PropertyProfile[];
  onPointClick?: (property: string, theta: number) => void;
}

export function SensitivityProfileChart({ profiles, onPointClick }: SensitivityProfileChartProps) {
  const isDark = useIsDarkMode();
  const active = profiles.filter((profile) => profile.applicable && profile.curve && profile.curve.length > 0);

  if (active.length === 0) {
    return (
      <div className="flex h-[280px] items-center justify-center text-sm text-muted-foreground border border-dashed border-border rounded-md">
        No property could be isolated for this input yet.
      </div>
    );
  }

  const traces: any[] = [];
  active.forEach((profile, index) => {
    const color = COLORS[index % COLORS.length];
    const curve = profile.curve!;
    const x = curve.map((point) => point.theta);
    const y = curve.map((point) => point.degradation);
    const hasCI = curve.some((point) => point.ci95);

    if (hasCI) {
      traces.push({
        type: "scatter", mode: "lines", showlegend: false, hoverinfo: "skip",
        x: [...x, ...x.slice().reverse()],
        y: [
          ...curve.map((point) => point.ci95?.[1] ?? point.degradation),
          ...curve.map((point) => point.ci95?.[0] ?? point.degradation).reverse(),
        ],
        fill: "toself", fillcolor: color, opacity: 0.15, line: { width: 0 },
      });
    }

    traces.push({
      type: "scatter", mode: "lines+markers",
      name: `${profile.property}${profile.sensitivity_index !== undefined ? ` (S=${profile.sensitivity_index.toFixed(2)})` : ""}`,
      x, y, marker: { color, size: 7 }, line: { color, width: 2 },
      customdata: x.map(() => profile.property),
      hovertemplate:
        `<b>${profile.property}</b><br>%{x} ${profile.unit}<br>degradation %{y:.3f}<extra></extra>`,
    });
  });

  const breakdownShapes = active
    .filter((profile) => profile.breakdown_theta !== null && profile.breakdown_theta !== undefined)
    .map((profile, index) => ({
      type: "line" as const, xref: "x" as const, yref: "paper" as const,
      x0: profile.breakdown_theta, x1: profile.breakdown_theta, y0: 0, y1: 1,
      line: { dash: "dot" as const, width: 1, color: COLORS[index % COLORS.length] },
    }));

  return (
    <Plot
      data={traces}
      layout={{
        autosize: true, height: 320, margin: { l: 48, r: 12, t: 8, b: 40 },
        template: isDark ? "plotly_dark" : "plotly_white",
        paper_bgcolor: "transparent", plot_bgcolor: "transparent",
        yaxis: { title: "Output degradation", range: [0, 1], zeroline: false },
        xaxis: { title: active.length === 1 ? `${active[0].property} (${active[0].unit})` : "Swept parameter" },
        legend: { orientation: "h", y: -0.25 },
        shapes: [
          { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0.5, y1: 0.5,
            line: { dash: "dash", width: 1, color: isDark ? "#666" : "#aaa" } },
          ...breakdownShapes,
        ],
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      onClick={(event) => {
        const point = event.points?.[0] as any;
        if (!point || !onPointClick) return;
        const property = point.data?.customdata?.[0] ?? active[0]?.property;
        onPointClick(property, Number(point.x));
      }}
      useResizeHandler
    />
  );
}
