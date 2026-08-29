import Plot from "react-plotly.js";
import { useIsDarkMode } from "@/hooks/useIsDarkMode";
import { InfoTooltip } from "@/components/analysis/InfoTooltip";
import { FR7_GLOSSARY } from "@/lib/fr7Glossary";
import type { CurvePoint, PropertyProfile } from "@/lib/linguisticAcoustic";

const COLORS = ["#3b82f6", "#f97316", "#10b981", "#a855f7", "#ef4444"];

interface SensitivityProfileChartProps {
  profiles: PropertyProfile[];
  onPointClick?: (property: string, theta: number) => void;
}

/** Linear interpolation between tested grid points, same method used
 * server-side for breakdown_theta, so hovering anywhere along the line
 * (not just at a tested marker) reads an honest straight-line estimate
 * rather than nothing. */
function densify(curve: CurvePoint[], stepsPerSegment = 12): { x: number[]; y: number[] } {
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < curve.length - 1; i++) {
    const a = curve[i];
    const b = curve[i + 1];
    for (let step = 0; step < stepsPerSegment; step++) {
      const t = step / stepsPerSegment;
      x.push(a.theta + (b.theta - a.theta) * t);
      y.push(a.degradation + (b.degradation - a.degradation) * t);
    }
  }
  const last = curve[curve.length - 1];
  x.push(last.theta);
  y.push(last.degradation);
  return { x, y };
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

    // Dense line: continuous hover reading between tested points, honestly
    // labeled as interpolated. Not clickable into the inspector -- only
    // the markers trace below corresponds to a real, re-inferred variant.
    if (curve.length > 1) {
      const dense = densify(curve);
      traces.push({
        type: "scatter", mode: "lines", showlegend: false,
        x: dense.x, y: dense.y, line: { color, width: 2 },
        hovertemplate:
          `<b>${profile.property}</b> (interpolated)<br>%{x:.2f} ${profile.unit}<br>estimated degradation %{y:.3f}<extra></extra>`,
      });
    }

    traces.push({
      type: "scatter", mode: "markers",
      name: `${profile.property}${profile.sensitivity_index !== undefined ? ` (S=${profile.sensitivity_index.toFixed(2)})` : ""}`,
      x, y, marker: { color, size: 8, line: { color: isDark ? "#000" : "#fff", width: 1 } },
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
    <div>
      <div className="flex items-center gap-1 mb-1">
        <span className="text-xs font-medium text-muted-foreground">Sensitivity curve</span>
        <InfoTooltip text={FR7_GLOSSARY.interpolatedHover} />
      </div>
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
          if (point.data?.mode !== "markers") return; // dense line has no real variant behind it
          const property = point.data?.customdata?.[0] ?? active[0]?.property;
          onPointClick(property, Number(point.x));
        }}
        useResizeHandler
      />
    </div>
  );
}
