// Client-side EDA report export — same Blob + createObjectURL idiom already
// used for audio playback in WaveformViewer.tsx, repurposed for file download.

function downloadBlob(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function csvEscape(value: unknown): string {
  const s = String(value ?? "");
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

interface AcousticEdaForExport {
  individual_analyses: Array<{ filename: string; features: Record<string, number> }>;
}

export function exportAcousticFeaturesCsv(acousticEda: AcousticEdaForExport, dataset: string): void {
  const analyses = acousticEda.individual_analyses;
  if (analyses.length === 0) return;

  const featureKeys = Array.from(
    new Set(analyses.flatMap((a) => Object.keys(a.features))),
  ).sort();

  const header = ["filename", ...featureKeys].map(csvEscape).join(",");
  const rows = analyses.map((a) =>
    [a.filename, ...featureKeys.map((k) => a.features[k] ?? "")].map(csvEscape).join(","),
  );

  downloadBlob([header, ...rows].join("\n"), `${dataset}-acoustic-features.csv`, "text/csv;charset=utf-8");
}

export function exportEdaJson(payload: unknown, dataset: string): void {
  downloadBlob(JSON.stringify(payload, null, 2), `${dataset}-eda-report.json`, "application/json");
}
