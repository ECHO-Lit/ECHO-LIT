/**
 * Attach an answer key to a custom dataset, and preview what it buys.
 *
 * Uploaded audio has no labels, so a layer probe has nothing to be graded
 * against. This tab is the two ways to fix that -- parse the filenames, or
 * upload a CSV -- plus the preview that tells the user, before any model runs,
 * exactly which properties will survive and which classes are about to be
 * dropped.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, CheckCircle2, FileSpreadsheet, Info, Tags, Trash2, Wand2 } from 'lucide-react';
import {
  clearDatasetLabels,
  deriveLabelsFromFilenames,
  fetchDatasetLabels,
  fetchLabelPatterns,
  uploadLabelCsv,
  type DatasetLabels,
  type LabelPattern,
  type PropertyPreview,
} from '@/lib/datasetLabels';

interface DatasetLabelsTabProps {
  datasets: Array<{ dataset_name: string; total_files: number }>;
}

const CSV_HELP =
  'One row per clip. A `filename` column joins it to the audio; every other column becomes a probeable property.';

/** A property that survives to the probe, or the reason it will not. */
const PropertyRow: React.FC<{ preview: PropertyPreview }> = ({ preview }) => {
  const classes = Object.entries(preview.class_counts);
  return (
    <div className="border border-border rounded-md p-2.5 space-y-1.5">
      <div className="flex items-center gap-2">
        {preview.probeable ? (
          <CheckCircle2 className="h-3.5 w-3.5 text-green-600 shrink-0" />
        ) : (
          <AlertCircle className="h-3.5 w-3.5 text-amber-600 shrink-0" />
        )}
        <span className="text-xs font-medium">{preview.property}</span>
        {preview.probeable && (
          <Badge variant="secondary" className="text-[10px] h-4 px-1.5">
            {preview.n_classes} classes · {preview.n_samples} files
          </Badge>
        )}
        {preview.majority_baseline !== null && (
          <span className="text-[10px] text-muted-foreground ml-auto shrink-0">
            guessing = {preview.majority_baseline.toFixed(3)}
          </span>
        )}
      </div>

      {preview.skipped_reason && (
        <div className="text-[11px] text-amber-700 dark:text-amber-500">
          Will be skipped - {preview.skipped_reason}
        </div>
      )}

      {preview.probeable && classes.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {classes.map(([label, count]) => (
            <span
              key={label}
              className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground"
            >
              {label} ({count})
            </span>
          ))}
        </div>
      )}

      {preview.dropped_classes.length > 0 && (
        <div className="text-[10px] text-muted-foreground">
          Dropped (too few files):{' '}
          {preview.dropped_classes.map((entry) => `${entry.label} (${entry.count})`).join(', ')}
        </div>
      )}

      {preview.n_missing > 0 && (
        <div className="text-[10px] text-muted-foreground">
          {preview.n_missing} file(s) unannotated for this property
        </div>
      )}
    </div>
  );
};

export const DatasetLabelsTab: React.FC<DatasetLabelsTabProps> = ({ datasets }) => {
  const [selected, setSelected] = useState('');
  const [patterns, setPatterns] = useState<LabelPattern[]>([]);
  const [patternId, setPatternId] = useState('');
  const [labels, setLabels] = useState<DatasetLabels | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [csvFile, setCsvFile] = useState<File | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchLabelPatterns(controller.signal)
      .then(setPatterns)
      .catch(() => setPatterns([]));
    return () => controller.abort();
  }, []);

  const refresh = useCallback(async (datasetName: string) => {
    if (!datasetName) {
      setLabels(null);
      return;
    }
    setError(null);
    try {
      setLabels(await fetchDatasetLabels(datasetName));
    } catch (cause) {
      setLabels(null);
      setError(cause instanceof Error ? cause.message : 'Failed to load labels');
    }
  }, []);

  useEffect(() => {
    void refresh(selected);
  }, [selected, refresh]);

  // Every mutation funnels through here so the preview is always refreshed from
  // the response rather than from a second round-trip that could disagree.
  const act = async (operation: () => Promise<DatasetLabels>) => {
    setBusy(true);
    setError(null);
    try {
      setLabels(await operation());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Operation failed');
    } finally {
      setBusy(false);
    }
  };

  const chosenPattern = patterns.find((entry) => entry.pattern_id === patternId);
  const preview = labels?.preview;

  return (
    <div className="space-y-4">
      <div className="text-xs text-muted-foreground flex gap-2 p-2.5 bg-muted/50 rounded-md border border-border">
        <Info className="h-3.5 w-3.5 shrink-0 mt-0.5" />
        <span>
          Layer probing grades a classifier against known labels. Uploaded audio has none, so
          attach an answer key here - from the filenames, or from your own CSV. Clip length is
          always available without any labels at all.
        </span>
      </div>

      <div>
        <Label htmlFor="labels-dataset">Dataset</Label>
        <select
          id="labels-dataset"
          value={selected}
          onChange={(event) => setSelected(event.target.value)}
          className="w-full mt-1 px-3 py-2 text-sm border border-border rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-ring"
        >
          <option value="">Choose a dataset...</option>
          {datasets.map((dataset) => (
            <option key={dataset.dataset_name} value={dataset.dataset_name}>
              {dataset.dataset_name} ({dataset.total_files} files)
            </option>
          ))}
        </select>
      </div>

      {selected && (
        <>
          {/* Option 1 — filenames. First because for a corpus that encodes its
              labels this way it is one click and needs no file at all. */}
          <div className="space-y-2 border border-border rounded-md p-3">
            <div className="flex items-center gap-2">
              <Wand2 className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Derive from filenames</span>
            </div>
            <select
              value={patternId}
              onChange={(event) => setPatternId(event.target.value)}
              className="w-full px-3 py-2 text-sm border border-border rounded-md bg-background"
            >
              <option value="">Choose a corpus pattern...</option>
              {patterns.map((pattern) => (
                <option key={pattern.pattern_id} value={pattern.pattern_id}>
                  {pattern.label} - {pattern.example}
                </option>
              ))}
            </select>
            {chosenPattern && (
              <div className="text-[11px] text-muted-foreground space-y-1">
                <div>{chosenPattern.description}</div>
                <div>
                  Extracts:{' '}
                  {chosenPattern.properties.map((property) => (
                    <Badge key={property} variant="outline" className="text-[10px] h-4 px-1 mr-1">
                      {property}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            <Button
              size="sm"
              className="w-full"
              disabled={busy || !patternId}
              onClick={() => act(() => deriveLabelsFromFilenames(selected, patternId))}
            >
              <Tags className="h-3.5 w-3.5 mr-2" />
              Apply pattern
            </Button>
          </div>

          {/* Option 2 — a user CSV. The general path: the only one that can carry
              a property the audio and filenames do not already contain. */}
          <div className="space-y-2 border border-border rounded-md p-3">
            <div className="flex items-center gap-2">
              <FileSpreadsheet className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Upload a label CSV</span>
            </div>
            <div className="text-[11px] text-muted-foreground">{CSV_HELP}</div>
            <pre className="text-[10px] bg-muted p-2 rounded overflow-x-auto">
{`filename,speaker,emotion
DC_a01.wav,DC,anger
JE_sa04.wav,JE,sadness`}
            </pre>
            <Input
              type="file"
              accept=".csv,text/csv"
              onChange={(event) => setCsvFile(event.target.files?.[0] ?? null)}
              className="text-xs"
            />
            <Button
              size="sm"
              className="w-full"
              disabled={busy || !csvFile}
              onClick={() => csvFile && act(() => uploadLabelCsv(selected, csvFile))}
            >
              <FileSpreadsheet className="h-3.5 w-3.5 mr-2" />
              Upload labels
            </Button>
          </div>

          {error && (
            <div className="text-xs text-destructive p-2.5 bg-destructive/5 rounded-md border border-destructive/20">
              {error}
            </div>
          )}

          {labels?.warnings.map((warning) => (
            <div
              key={warning}
              className="text-[11px] text-amber-700 dark:text-amber-500 p-2 bg-amber-500/5 rounded-md border border-amber-500/20"
            >
              {warning}
            </div>
          ))}

          {preview && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">
                  Probe preview — {preview.probeable_count} of {preview.properties.length} usable
                </span>
                {labels?.source && (
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={busy}
                    onClick={() => act(() => clearDatasetLabels(selected))}
                    className="h-6 text-[11px] text-muted-foreground"
                  >
                    <Trash2 className="h-3 w-3 mr-1" />
                    Clear labels
                  </Button>
                )}
              </div>
              <div className="text-[11px] text-muted-foreground">
                What the probe will do with these {preview.n_files} files, before any model runs.
                A property is only worth reading if its accuracy beats the guessing baseline.
              </div>
              <div className="space-y-1.5">
                {preview.properties.map((property) => (
                  <PropertyRow key={property.property} preview={property} />
                ))}
              </div>
              {preview.properties.length === 0 && (
                <div className="text-xs text-muted-foreground text-center py-4">
                  No probeable properties yet — attach labels above.
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
};
