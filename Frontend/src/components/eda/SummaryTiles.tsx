interface SummaryTilesProps {
  totalFiles: number;
  totalHours: number;
  meanDuration: number;
  medianDuration: number;
  numClasses: number;
}

export const SummaryTiles = ({ totalFiles, totalHours, meanDuration, medianDuration, numClasses }: SummaryTilesProps) => {
  const tiles = [
    { label: "Files", value: totalFiles.toLocaleString() },
    { label: "Total hours", value: totalHours.toFixed(2) },
    { label: "Mean duration", value: `${meanDuration.toFixed(2)}s` },
    { label: "Median duration", value: `${medianDuration.toFixed(2)}s` },
    ...(numClasses > 0 ? [{ label: "Classes", value: String(numClasses) }] : []),
  ];

  return (
    <div className="grid grid-cols-2 gap-2">
      {tiles.map((tile) => (
        <div key={tile.label} className="p-2 bg-muted/50 rounded border border-border">
          <div className="text-[10px] text-muted-foreground">{tile.label}</div>
          <div className="text-sm font-semibold text-foreground">{tile.value}</div>
        </div>
      ))}
    </div>
  );
};
