import Plot from "react-plotly.js";

interface BoxPlotChartProps {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  label: string;
  color?: string;
  fillColor?: string;
}

// Generalized from ScalarPlot.tsx — feeds real precomputed quartiles
// instead of a raw sample array, so no per-file values need to ship to the client.
export const BoxPlotChart = ({
  min, q1, median, q3, max, label,
  color = "hsl(var(--primary))",
  fillColor = "hsl(var(--primary) / 0.3)",
}: BoxPlotChartProps) => {
  return (
    <div className="h-full">
      <Plot
        data={[
          {
            type: "box",
            name: label,
            q1: [q1],
            median: [median],
            q3: [q3],
            lowerfence: [min],
            upperfence: [max],
            width: 0.4,
            marker: { color },
            line: { color },
            fillcolor: fillColor,
            hoverlabel: {
              bgcolor: "hsl(var(--popover))",
              bordercolor: color,
              font: { size: 9, color: "hsl(var(--popover-foreground))" },
            },
          } as any,
        ]}
        layout={{
          autosize: true,
          margin: { l: 40, r: 10, t: 10, b: 20 },
          // Extra horizontal room either side of the (fixed-width) box gives
          // Plotly's hover label space to render beside it instead of on top.
          yaxis: { showgrid: true, gridcolor: "hsl(var(--border))", tickfont: { size: 9 } },
          xaxis: { showticklabels: false, range: [-2, 2] },
          plot_bgcolor: "transparent",
          paper_bgcolor: "transparent",
          showlegend: false,
          font: { size: 10, color: "hsl(var(--foreground))" },
        }}
        config={{
          displayModeBar: "hover",
          displaylogo: false,
          modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
          scrollZoom: true,
          responsive: true,
        }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
};
