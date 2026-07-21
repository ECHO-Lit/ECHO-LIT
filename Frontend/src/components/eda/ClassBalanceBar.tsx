import Plot from "react-plotly.js";

interface ClassBalanceBarProps {
  counts: Record<string, number>;
  // filename lists per class label — when provided, bars become clickable.
  filenamesByClass?: Record<string, string[]>;
  onBarClick?: (className: string, filenames: string[]) => void;
}

export const ClassBalanceBar = ({ counts, filenamesByClass, onBarClick }: ClassBalanceBarProps) => {
  const entries = Object.entries(counts).sort(([, a], [, b]) => b - a);
  if (entries.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
        No class labels in this dataset
      </div>
    );
  }

  const customdata = filenamesByClass
    ? entries.map(([label]) => filenamesByClass[label] || [])
    : undefined;

  return (
    <div className="h-full">
      <Plot
        data={[
          {
            x: entries.map(([label]) => label),
            y: entries.map(([, count]) => count),
            type: "bar",
            marker: { color: "hsl(var(--primary))" },
            ...(customdata ? { customdata } : {}),
          } as any,
        ]}
        layout={{
          autosize: true,
          margin: { l: 40, r: 10, t: 10, b: 50 },
          xaxis: { tickfont: { size: 9 }, tickangle: -35 },
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
                if (point?.customdata) onBarClick(point.x as string, point.customdata as string[]);
              }
            : undefined
        }
        style={{ width: "100%", height: "100%", cursor: onBarClick ? "pointer" : undefined }}
      />
    </div>
  );
};
