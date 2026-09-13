import Plot from "react-plotly.js";
import { useIsDarkMode } from "@/hooks/useIsDarkMode";
import { InfoTooltip } from "./InfoTooltip";
import { FR7_GLOSSARY } from "@/lib/fr7Glossary";
import type { SensitivityProfile } from "@/lib/linguisticAcoustic";

interface SensitivityRankingChartProps {
  profile: SensitivityProfile;
  lexicalDestructionDegradation: number | null;
}

/** Horizontal bar chart of every swept property's sensitivity index, ranked,
 * with a dashed reference line at the lexical-destruction control's
 * degradation, so the whole verdict is readable at a glance instead of
 * requiring three separate numbers to be read and mentally combined. */
export function SensitivityRankingChart({
  profile, lexicalDestructionDegradation,
}: SensitivityRankingChartProps) {
  const isDark = useIsDarkMode();
  const ranking = [...profile.ranking].sort((a, b) => b.sensitivity_index - a.sensitivity_index);

  if (ranking.length === 0) return null;

  const colors = ranking.map((entry) =>
    entry.property === profile.dominant_property ? "#ef4444" : "#3b82f6",
  );

  return (
    <div>
      <div className="flex items-center gap-1 mb-1">
        <span className="text-xs font-medium text-muted-foreground">Property ranking</span>
        <InfoTooltip text={FR7_GLOSSARY.relativeToWordRemoval} />
      </div>
      <Plot
        data={[
          {
            type: "bar",
            orientation: "h",
            y: ranking.map((entry) => entry.property),
            x: ranking.map((entry) => entry.sensitivity_index),
            marker: { color: colors },
            hovertemplate: "%{y}: sensitivity %{x:.3f}<extra></extra>",
          },
        ]}
        layout={{
          autosize: true,
          height: Math.max(120, ranking.length * 42),
          margin: { l: 90, r: 16, t: 24, b: 30 },
          template: isDark ? "plotly_dark" : "plotly_white",
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          xaxis: { title: "Sensitivity index", range: [0, 1] },
          yaxis: { automargin: true },
          shapes: lexicalDestructionDegradation != null ? [
            {
              type: "line", xref: "x", yref: "paper",
              x0: lexicalDestructionDegradation, x1: lexicalDestructionDegradation, y0: 0, y1: 1,
              line: { dash: "dash", width: 1, color: isDark ? "#999" : "#555" },
            },
          ] : [],
          annotations: lexicalDestructionDegradation != null ? [
            {
              x: lexicalDestructionDegradation, y: 1, yref: "paper", yanchor: "top",
              xanchor: lexicalDestructionDegradation > 0.7 ? "right" : "left",
              text: "word removal reference", showarrow: false, font: { size: 10 },
              bgcolor: isDark ? "rgba(0,0,0,0.6)" : "rgba(255,255,255,0.8)",
            },
          ] : [],
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}
