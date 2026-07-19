import {
  useReactTable,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  ColumnDef,
} from "@tanstack/react-table";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Play, ChevronLeft, ChevronRight } from "lucide-react";
import { useMemo, useCallback, useEffect } from "react";

interface UploadedFile {
  audio_id?: string;
  file_id: string;
  filename: string;
  playback_url?: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
  prediction?:string;
}

interface AudioData {
  id: string;
  filename: string;
  prediction?: string;
  groundTruthLabel: string;
  confidence: number;
  duration: number;
  size?: number;
}

interface AudioDataTableProps {
  selectedRow: string | null;
  onRowSelect: (id: string) => void;
  searchQuery: string;
  apiData?: unknown;
  model: string;
  dataset: string; // "custom" | "sample-speech" | "common-voice" | "ravdess"
  datasetMetadata?: Record<string, string | number>[];
  uploadedFiles?: UploadedFile[];
  onFilePlay?: (file: UploadedFile) => void;
  predictionMap?: Record<string, string>;
  inferenceStatus?: Record<string, 'idle' | 'loading' | 'done' | 'error'>;
  onVisibleRowIdsChange?: (rowIds: string[]) => void;
}

export const AudioDataTable = ({ selectedRow, onRowSelect, searchQuery, apiData, model, dataset, datasetMetadata, uploadedFiles, onFilePlay, predictionMap, inferenceStatus, onVisibleRowIdsChange }: AudioDataTableProps) => {
  // Branch: dataset mode vs custom uploads
  const hasDatasetMetadata = (datasetMetadata?.length || 0) > 0;
  const hasUploadedFiles = uploadedFiles && uploadedFiles.length > 0;

  // Dataset metadata data and columns
  type DatasetRow = Record<string, string | number | null | undefined>;
  const datasetRows: DatasetRow[] = useMemo(() => datasetMetadata ?? [], [datasetMetadata]);

  const getFrom = useCallback((row: DatasetRow, keys: string[], fallback = ""): string => {
    for (const k of keys) {
      const v = row[k];
      if (v !== undefined && v !== null && String(v).length > 0) return String(v);
    }
    return fallback;
  }, []);

  // Custom uploads data and columns
  const customTableData: AudioData[] = useMemo(() => (
    uploadedFiles?.map(file => ({
      id: file.file_id,
      filename: file.filename,
      prediction: file.prediction || "",
      groundTruthLabel: "",
      confidence: 0,
      duration: typeof file.duration === 'number' ? file.duration : 0,
      size: file.size
    })) || []
  ), [uploadedFiles]);

  const customColumns: ColumnDef<unknown, unknown>[] = useMemo(() => [
    {
      id: "filename",
      header: "Filename",
      cell: ({ row }) => {
        // Handle both AudioData (uploaded files) and DatasetRow (dataset files)
        if ('groundTruthLabel' in (row.original as any)) {
          // This is an uploaded file (AudioData)
        const data = row.original as AudioData;
        const file = uploadedFiles?.find(f => f.file_id === data.id);
        return (
          <div className="flex items-center gap-2">
            <Button
              type="button"
              size="sm"
              variant="ghost"
              className="h-6 w-6 p-0"
              onClick={(e) => {
                e.stopPropagation();
                e.preventDefault();
                if (file && onFilePlay) onFilePlay(file);
              }}
            >
              <Play className="h-3 w-3" />
            </Button>
            <span className="font-mono text-xs">{data.filename}</span>
          </div>
        );
        } else {
          // This is a dataset file (DatasetRow)
          const data = row.original as DatasetRow;
          const path = getFrom(data, ["path", "filepath", "file", "filename"], "");
          const filename = path.split("/").pop() || path;
          return <span className="font-mono text-xs">{filename}</span>;
        }
      },
    },
    {
      id: "prediction",
      header: model.startsWith("whisper") ? "Predicted Transcript" : "Predicted Label",
      cell: ({ row }) => {
        const rowId = row.id as string;
        const status = inferenceStatus?.[rowId];
        if (status === 'loading') {
          return "";
        }
        
        // Handle both AudioData and DatasetRow
        if ('groundTruthLabel' in (row.original as any)) {
          // This is an uploaded file (AudioData) - use predictionMap like dataset files
          const data = row.original as AudioData;
          const pred = predictionMap?.[rowId] || data.prediction || "";
          if (!pred) return "";
          
          // Handle object predictions (different models return different object structures)
          const predictionText = typeof pred === 'string' ? pred : 
            (typeof pred === 'object' && pred !== null) ? 
              (pred as any).predicted_transcript || (pred as any).predicted_emotion || (pred as any).prediction || (pred as any).text || JSON.stringify(pred) : 
              String(pred);
          
          return <Badge variant="outline" className="text-xs">{predictionText}</Badge>;
        } else {
          // This is a dataset file (DatasetRow)
          const pred = predictionMap?.[rowId] ?? "";
          
          // Handle object predictions (different models return different object structures)
          const predictionText = typeof pred === 'string' ? pred : 
            (typeof pred === 'object' && pred !== null) ? 
              (pred as any).predicted_transcript || (pred as any).predicted_emotion || (pred as any).prediction || (pred as any).text || JSON.stringify(pred) : 
              String(pred);
              
          return <span className="text-xs">{predictionText}</span>;
        }
      },
    },
    {
      id: "groundTruthLabel",
      header: "Ground Truth",
      cell: ({ row }) => {
        // Handle both AudioData and DatasetRow
        if ('groundTruthLabel' in (row.original as any)) {
          // This is an uploaded file (AudioData)
        const data = row.original as AudioData;
        return <span className="text-xs">{data.groundTruthLabel}</span>;
        } else {
          // This is a dataset file (DatasetRow)
          const data = row.original as DatasetRow;
          return <span className="text-xs">{getFrom(data, ["sentence", "transcript", "text", "emotion", "label"], "")}</span>;
        }
      },
    },
    {
      id: "confidence",
      header: "Confidence",
      cell: ({ row }) => {
        // Only show confidence for uploaded files (AudioData)
        if ('groundTruthLabel' in (row.original as any)) {
        const data = row.original as AudioData;
        // Don't display confidence if it's 0
        if (data.confidence === 0) return null;
        return <span className="text-xs">{data.confidence}</span>;
        } else {
          // For dataset files, show N/A or empty
          return <span className="text-xs text-muted-foreground">N/A</span>;
        }
      },
    },
    {
      id: "duration",
      header: "Duration",
      cell: ({ row }) => {
        // Handle both AudioData and DatasetRow
        if ('groundTruthLabel' in (row.original as any)) {
          // This is an uploaded file (AudioData)
        const data = row.original as AudioData;
          const duration = typeof data.duration === 'number' ? data.duration : 0;
          return <span className="text-xs">{duration.toFixed(2)}s</span>;
        } else {
          // This is a dataset file (DatasetRow)
          const data = row.original as DatasetRow;
          const d = Number(getFrom(data, ["duration", "length"], "0"));
          if (d > 0) {
            return <span className="text-xs">{d.toFixed(2)}s</span>;
          }
          return <span className="text-xs text-muted-foreground">N/A</span>;
        }
      },
    },
  ], [model, uploadedFiles, onFilePlay, predictionMap, inferenceStatus, getFrom]);

  const getDatasetRowId = useCallback((row: DatasetRow, fallback: string): string => {
    const v = row["id"] ?? row["path"] ?? row["filepath"] ?? row["file"] ?? row["filename"];
    return v !== undefined && v !== null && String(v).length > 0 ? String(v) : fallback;
  }, []);

  // Helper function to determine if ground truth should be shown
  const shouldShowGroundTruth = useMemo(() => {
    if (model.startsWith("whisper")) {
      // Whisper models can show ground truth only for common-voice (has transcript)
      return dataset === "common-voice";
    } else if (model === "wav2vec2") {
      // Wav2Vec2 models can show ground truth only for RAVDESS (has emotion labels)
      return dataset === "ravdess";
    }
    return false;
  }, [model, dataset]);

  const datasetColumnsCommonVoice: ColumnDef<unknown, unknown>[] = useMemo(() => {
    const baseColumns = [
      {
        id: "filename",
        header: "Filename",
        cell: ({ row }) => {
          const data = row.original as DatasetRow;
          const path = getFrom(data, ["path", "filepath", "file", "filename"], "");
          const filename = path.split("/").pop() || path;
          return <span className="font-mono text-xs">{filename}</span>;
        },
      },
      {
        id: "prediction",
        header: model.startsWith("whisper") ? "Predicted Transcript" : "Predicted Emotion",
        cell: ({ row }) => {
          const rowId = row.id as string;
          const status = inferenceStatus?.[rowId];
          
          if (status === 'loading') {
            return <span className="text-xs text-blue-600">Loading...</span>;
          }
          
          if (status !== 'done') {
            return <span className="text-xs text-gray-400">-</span>;
          }
          
          const pred = predictionMap?.[rowId] ?? "";
          
          // Handle object predictions (different models return different object structures)
          const predictionText = typeof pred === 'string' ? pred : 
            (typeof pred === 'object' && pred !== null) ? 
              (pred as any).predicted_transcript || (pred as any).predicted_emotion || (pred as any).prediction || (pred as any).text || JSON.stringify(pred) : 
              String(pred);
              
          return <span className="text-xs">{predictionText || <span className="text-gray-400">No prediction</span>}</span>;
        },
      },
    ];

    // Add ground truth column only if applicable for this model-dataset combination
    if (shouldShowGroundTruth) {
      baseColumns.push({
        id: "ground_truth",
        header: model.startsWith("whisper") ? "Ground Truth Transcript" : "Ground Truth Emotion",
        cell: ({ row }) => {
          const data = row.original as DatasetRow;
          const groundTruthValue = model.startsWith("whisper") 
            ? getFrom(data, ["sentence", "transcript", "text"], "")
            : getFrom(data, ["emotion", "label"], "");
          return <span className="text-xs">{groundTruthValue}</span>;
        },
      });
    }

    baseColumns.push({
      id: "duration",
      header: "Duration",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        const d = Number(getFrom(data, ["duration"], "0"));
        return <span className="text-xs">{isNaN(d) ? "" : `${d.toFixed(2)}s`}</span>;
      },
    });

    return baseColumns;
  }, [getFrom, model, predictionMap, inferenceStatus, shouldShowGroundTruth]);

  const datasetColumnsRavdess: ColumnDef<unknown, unknown>[] = useMemo(() => {
    const baseColumns = [
      {
        id: "filename",
        header: "Filename",
        cell: ({ row }) => {
          const data = row.original as DatasetRow;
          const path = getFrom(data, ["path", "filepath", "file", "filename"], "");
          const filename = path.split("/").pop() || path;
          return <span className="font-mono text-xs">{filename}</span>;
        },
      },
      {
        id: "prediction",
        header: model.startsWith("whisper") ? "Predicted Transcript" : "Predicted Emotion",
        cell: ({ row }) => {
          const rowId = row.id as string;
          const status = inferenceStatus?.[rowId];
          
          if (status === 'loading') {
            return <span className="text-xs text-blue-600">Loading...</span>;
          }
          
          if (status !== 'done') {
            return <span className="text-xs text-gray-400">-</span>;
          }
          
          const pred = predictionMap?.[rowId] ?? "";
          
          // Handle object predictions (different models return different object structures)
          const predictionText = typeof pred === 'string' ? pred : 
            (typeof pred === 'object' && pred !== null) ? 
              (pred as any).predicted_transcript || (pred as any).predicted_emotion || (pred as any).prediction || (pred as any).text || JSON.stringify(pred) : 
              String(pred);
              
          return <span className="text-xs">{predictionText || <span className="text-gray-400">No prediction</span>}</span>;
        },
      },
    ];

    // Add ground truth column only if applicable for this model-dataset combination
    if (shouldShowGroundTruth) {
      baseColumns.push({
        id: "ground_truth",
        header: model.startsWith("whisper") ? "Ground Truth Transcript" : "Ground Truth Emotion",
        cell: ({ row }) => {
          const data = row.original as DatasetRow;
          const groundTruthValue = model.startsWith("whisper") 
            ? getFrom(data, ["statement", "text", "transcript", "sentence"], "")
            : getFrom(data, ["emotion", "label"], "");
          return <span className="text-xs">{groundTruthValue}</span>;
        },
      });
    }

    baseColumns.push({
      id: "duration",
      header: "Duration",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        // Ravdess dataset doesn't have duration in metadata, so we'll show "N/A"
        const d = Number(getFrom(data, ["duration", "length"], "0"));
        if (d > 0) {
          return <span className="text-xs">{d.toFixed(2)}s</span>;
        }
        return <span className="text-xs text-muted-foreground">N/A</span>;
      },
    });

    return baseColumns;
  }, [getFrom, model, predictionMap, inferenceStatus, shouldShowGroundTruth]);

  // Build table config based on mode
  const data: unknown[] = useMemo(() => {
    if (hasUploadedFiles) {
      // When there are uploaded files, show uploaded files at top, then dataset files
      const combinedData: unknown[] = [...customTableData]; // Uploaded files first
      if (hasDatasetMetadata) {
        // Add dataset files after uploaded files
        combinedData.push(...datasetRows);
      }
      return combinedData;
    }
    // When no uploaded files, use original logic
    return hasDatasetMetadata ? datasetRows : customTableData;
  }, [hasDatasetMetadata, hasUploadedFiles, datasetRows, customTableData]);
  const columns: ColumnDef<unknown, unknown>[] = useMemo(
    () => {
      if (hasUploadedFiles) {
        // When showing combined data, use custom columns that can handle both types
        return customColumns;
      }
      // When no uploaded files, use original logic
      return hasDatasetMetadata ? (dataset === "ravdess" ? datasetColumnsRavdess : datasetColumnsCommonVoice) : customColumns;
    },
    [hasUploadedFiles, hasDatasetMetadata, dataset, datasetColumnsRavdess, datasetColumnsCommonVoice, customColumns]
  );

  const getRowId = useCallback((row: unknown, index?: number) => {
    if (hasUploadedFiles) {
      // When showing combined data, check if it's a dataset row or uploaded file
      if ('file_id' in (row as any)) {
        return (row as AudioData).id;
      } else {
        return getDatasetRowId(row as DatasetRow, String(index ?? ""));
      }
    }
    if (hasDatasetMetadata) {
      return getDatasetRowId(row as DatasetRow, String(index ?? ""));
    }
    return (row as AudioData).id;
  }, [hasUploadedFiles, hasDatasetMetadata, getDatasetRowId]);

  const table = useReactTable<unknown>({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    state: {
      globalFilter: searchQuery,
    },
    onGlobalFilterChange: () => {},
    initialState: {
      pagination: {
        pageSize: 20,
      },
    },
    getRowId,
  });

  // Notify parent of currently visible row ids (for sequential per-page inference)
  useEffect(() => {
    if (!onVisibleRowIdsChange) return;
    const rows = table.getRowModel().rows;
    const ids = rows.map(r => String(r.id));
    onVisibleRowIdsChange(ids);
  }, [onVisibleRowIdsChange, searchQuery, dataset, model, hasDatasetMetadata]);

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 overflow-auto">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead key={header.id} className="h-8 text-xs">
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  data-state={(() => {
                    let rowId: string;
                    if (hasUploadedFiles) {
                      // When showing combined data, check if it's a dataset row or uploaded file
                      if ('file_id' in (row.original as any)) {
                        rowId = (row.original as AudioData).id;
                      } else {
                        rowId = getDatasetRowId(row.original as DatasetRow, String(row.id));
                      }
                    } else if (hasDatasetMetadata) {
                      rowId = getDatasetRowId(row.original as DatasetRow, String(row.id));
                    } else {
                      rowId = (row.original as AudioData).id;
                    }
                    return selectedRow === rowId ? "selected" : undefined;
                  })()}
                  className="cursor-pointer hover:bg-muted/50 data-[state=selected]:bg-blue-50 data-[state=selected]:border-blue-200 data-[state=selected]:shadow-sm"
                  onClick={() => {
                    let rowId: string;
                    if (hasUploadedFiles) {
                      // When showing combined data, check if it's a dataset row or uploaded file
                      if ('file_id' in (row.original as any)) {
                        rowId = (row.original as AudioData).id;
                      } else {
                        rowId = getDatasetRowId(row.original as DatasetRow, String(row.id));
                      }
                    } else if (hasDatasetMetadata) {
                      rowId = getDatasetRowId(row.original as DatasetRow, String(row.id));
                    } else {
                      rowId = (row.original as AudioData).id;
                    }
                    onRowSelect(rowId);
                  }}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id} className="py-2">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={Array.isArray(columns) ? columns.length : 0} className="h-24 text-center text-xs">
                  No results.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
      
      {/* Pagination */}
      <div className="border-t panel-border p-2 flex items-center justify-between">
        <div className="text-xs text-muted-foreground">
          Page {table.getState().pagination.pageIndex + 1} of {table.getPageCount()}
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            className="h-6 w-6 p-0"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            <ChevronLeft className="h-3 w-3" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-6 w-6 p-0"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            <ChevronRight className="h-3 w-3" />
          </Button>
        </div>
      </div>
    </div>
  );
};
