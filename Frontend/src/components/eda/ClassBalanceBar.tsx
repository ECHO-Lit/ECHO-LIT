import Plot from "react-plotly.js";

interface ClassBalanceBarProps {
  counts: Record<string, number>;
  // filename lists per class label — when provided, bars become clickable.
  filenamesByClass?: Record<string, string[]>;
  onBarClick?: (className: string, filenames: string[]) => void;
  // Optional second dataset, rendered as grouped bars. Values are shown as a
  // share of each dataset so differently-sized datasets stay comparable.
  counts2?: Record<string, number>;
  label?: string;
  label2?: string;
}

export const ClassBalanceBar = ({
  counts,
  filenamesByClass,
  onBarClick,
  counts2,
  label = "Dataset",
  label2 = "Comparison",
}: ClassBalanceBarProps) => {
  const entries = Object.entries(counts).sort(([, a], [, b]) => b - a);
  if (entries.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
        No class labels in this dataset
      </div>
    );
  }

  const customdata = filenamesByClass
    ? entries.map(([className]) => filenamesByClass[className] || [])
    : undefined;

  // In comparison mode both datasets are shown as a percentage of their own total —
  // raw counts would make the larger dataset look uniformly "more of everything".
  const isComparing = Boolean(counts2);
  const total1 = entries.reduce((sum, [, count]) => sum + count, 0) || 1;
  const total2 = counts2 ? Object.values(counts2).reduce((sum, count) => sum + count, 0) || 1 : 1;
  const classNames = isComparing
    ? Array.from(new Set([...entries.map(([name]) => name), ...Object.keys(counts2 ?? {})]))
    : entries.map(([name]) => name);
  const toValue = (count: number, total: number) => (isComparing ? (count / total) * 100 : count);

  return (
    <div className="h-full">
      <Plot
        data={[
          {
            x: classNames,
            y: classNames.map((name) => toValue(counts[name] ?? 0, total1)),
            type: "bar",
            marker: { color: "#2a78d6" },
            name: label,
            ...(customdata && !isComparing ? { customdata } : {}),
          } as any,
          ...(counts2
            ? [
                {
                  x: classNames,
                  y: classNames.map((name) => toValue(counts2[name] ?? 0, total2)),
                  type: "bar",
                  marker: { color: "#eb6834" },
                  name: label2,
                } as any,
              ]
            : []),
        ]}
        layout={{
          autosize: true,
          margin: { l: 40, r: 10, t: 10, b: isComparing ? 60 : 50 },
          barmode: "group",
          xaxis: { tickfont: { size: 9 }, tickangle: -35 },
          yaxis: {
            showgrid: true,
            gridcolor: "hsl(var(--border))",
            tickfont: { size: 9 },
            title: isComparing ? { text: "% of dataset", font: { size: 9 } } : undefined,
          },
          plot_bgcolor: "transparent",
          paper_bgcolor: "transparent",
          showlegend: isComparing,
          legend: { orientation: "h", y: -0.32, x: 0, font: { size: 9 } },
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
