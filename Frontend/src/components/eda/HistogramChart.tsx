import Plot from "react-plotly.js";

interface HistogramChartProps {
  bins: number[];
  histogram: number[];
  label: string;
  color?: string;
  // Filenames falling into each bin, same length/order as `histogram`.
  // When provided, bars become clickable.
  customdata?: string[][];
  onBarClick?: (filenames: string[]) => void;
}

// bins is edges (n+1 values), histogram is bar heights (n values) —
// the shape numpy.histogram() returns and the backend passes through as-is.
export const HistogramChart = ({ bins, histogram, label, color = "hsl(var(--primary))", customdata, onBarClick }: HistogramChartProps) => {
  if (!bins.length || !histogram.length) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
        No data
      </div>
    );
  }

  const centers = histogram.map((_, i) => (bins[i] + bins[i + 1]) / 2);
  const widths = histogram.map((_, i) => bins[i + 1] - bins[i]);

  return (
    <div className="h-full">
      <Plot
        data={[
          {
            x: centers,
            y: histogram,
            width: widths,
            type: "bar",
            marker: { color, line: { color: "hsl(var(--border))", width: 1 } },
            name: label,
            ...(customdata ? { customdata } : {}),
          } as any,
        ]}
        layout={{
          autosize: true,
          margin: { l: 40, r: 10, t: 10, b: 30 },
          bargap: 0.02,
          xaxis: { showgrid: false, tickfont: { size: 9 } },
          yaxis: { showgrid: true, gridcolor: "hsl(var(--border))", tickfont: { size: 9 } },
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
        onClick={
          onBarClick
            ? (event: any) => {
                const point = event?.points?.[0];
                if (point?.customdata) onBarClick(point.customdata as string[]);
              }
            : undefined
        }
        style={{ width: "100%", height: "100%", cursor: onBarClick ? "pointer" : undefined }}
      />
    </div>
  );
};
