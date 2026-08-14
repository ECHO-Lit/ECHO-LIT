/**
 * Confusion matrix for the best layer of one property.
 *
 * Cells are row-normalised: with 24 speakers of 5-8 clips each, raw counts are
 * dominated by how many clips a class happens to have, which reads as structure
 * that isn't there. The raw count stays in the hover.
 */

import { useMemo } from "react";
import Plot from "react-plotly.js";

import { useIsDarkMode } from "@/hooks/useIsDarkMode";
import { propertyLabel, type PropertyProbe } from "@/lib/probes";

interface ProbeConfusionMatrixProps {
  probe: PropertyProbe;
  propertyKey: string;
  layerName: string;
}

export const ProbeConfusionMatrix = ({
  probe,
  propertyKey,
  layerName,
}: ProbeConfusionMatrixProps) => {
  const isDark = useIsDarkMode();

  const chart = useMemo(() => {
    if (!probe.confusion_matrix.length) return null;
    const normalised = probe.confusion_matrix.map((row) => {
      const total = row.reduce((sum, value) => sum + value, 0) || 1;
      return row.map((value) => value / total);
    });
    return { normalised, counts: probe.confusion_matrix };
  }, [probe]);

  if (!chart) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
        No confusion matrix for {propertyLabel(propertyKey)}
      </div>
    );
  }

  const ink = isDark ? "#e6e4df" : "#1f1f1e";
  // Sequential, single hue, light -> dark: this is a magnitude, not a polarity.
  const scale: Array<[number, string]> = isDark
    ? [[0, "#22221f"], [1, "#3987e5"]]
    : [[0, "#f4f2ed"], [1, "#2a78d6"]];

  return (
    <div className="h-full">
      <Plot
        data={[
          {
            z: chart.normalised,
            x: probe.class_labels,
            y: probe.class_labels,
            customdata: chart.counts,
            type: "heatmap",
            colorscale: scale,
            zmin: 0,
            zmax: 1,
            xgap: 2,
            ygap: 2,
            colorbar: { thickness: 8, tickfont: { size: 8, color: ink } },
            hovertemplate:
              "true <b>%{y}</b> → predicted <b>%{x}</b><br>%{z:.1%} of true class (%{customdata} clips)<extra></extra>",
          } as any,
        ]}
        layout={{
          autosize: true,
          margin: { l: 76, r: 10, t: 8, b: 66 },
          xaxis: {
            title: { text: `predicted · ${layerName}`, font: { size: 10, color: ink } },
            tickfont: { size: 8, color: ink },
            tickangle: -45,
            automargin: true,
          },
          yaxis: {
            title: { text: "true", font: { size: 10, color: ink } },
            tickfont: { size: 8, color: ink },
            automargin: true,
          },
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
