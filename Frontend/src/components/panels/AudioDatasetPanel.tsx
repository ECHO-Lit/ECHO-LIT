import { useState, useRef, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { Upload, Search, Play, Pause, RefreshCw, HelpCircle } from "lucide-react";
import { AudioUploader } from "../audio/AudioUploader";
import { AudioDataTable } from "../audio/AudioDataTable";
import { toast } from "sonner";
import { API_BASE } from '@/lib/api';
import { materializeAudio, runJob } from '@/lib/jobs';

interface UploadedFile {
  audio_id?: string;
  file_id: string;
  filename: string;
  playback_url?: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
  prediction?: string;
  ground_truth?: string;
}

interface AudioDatasetPanelProps {
  apiData?: unknown;
  model: string | null;
  dataset: string;
  originalDataset?: string;
  uploadedFiles?: UploadedFile[];
  selectedFile?: UploadedFile | null;
  onFileSelect?: (file: UploadedFile) => void;
  onUploadSuccess?: (uploadResponse: UploadedFile) => void;
  batchInferenceStatus?: 'idle' | 'running' | 'done';
  onBatchInferenceStart?: () => void;
  onBatchInferenceComplete?: () => void;
  onAvailableFilesChange?: (files: string[]) => void;
  // Full dataset metadata rows, so the caller can look up ground truth for
  // files selected outside this table (e.g. the embedding scatter, EDA outliers).
  onDatasetMetadataChange?: (rows: Record<string, string | number>[]) => void;
  onPredictionUpdate?: (fileId: string, prediction: string) => void;
  predictionMap?: Record<string, string>;
  // Driven by clicking a Dataset EDA chart bucket — restricts the table to
  // these filenames. onClearFilter resets it from the caller's state.
  filterFilenames?: string[] | null;
  onClearFilter?: () => void;
}

export const AudioDatasetPanel = ({
  apiData,
  model,
  dataset,
  originalDataset,
  selectedFile,
  onFileSelect,
  onUploadSuccess,
  batchInferenceStatus,
  onBatchInferenceStart,
  onBatchInferenceComplete,
  onAvailableFilesChange,
  onDatasetMetadataChange,
  onPredictionUpdate,
  predictionMap: externalPredictionMap,
  filterFilenames,
  onClearFilter,
}: AudioDatasetPanelProps) => {
  const [selectedRow, setSelectedRow] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [datasetMetadata, setDatasetMetadata] = useState<Record<string, string | number>[]>([]);
  // Use external predictionMap from parent
  const predictionMap = externalPredictionMap || {};
  const [inferenceStatus, setInferenceStatus] = useState<Record<string, 'idle' | 'loading' | 'done' | 'error'>>({});
  
  // Batch inference state
  const [currentInferenceIndex, setCurrentInferenceIndex] = useState(0);
  const [batchInferenceQueue, setBatchInferenceQueue] = useState<string[]>([]);
  const [isInferenceComplete, setIsInferenceComplete] = useState(false);
  const [currentModelDataset, setCurrentModelDataset] = useState<string>("");
  const abortControllerRef = useRef<AbortController | null>(null);
  const runningBatchKeyRef = useRef<string | null>(null);

  // Sync selectedRow when selectedFile changes from external selection (e.g., embeddings)
  useEffect(() => {
    if (selectedFile) {
      // For uploaded files, use file_id
      if (uploadedFiles.some(f => f.file_id === selectedFile.file_id)) {
        setSelectedRow(selectedFile.file_id);
        return;
      }
      
      // For dataset files, find matching row by filename
      if (datasetMetadata.length > 0) {
        const matchingRow = datasetMetadata.find(row => {
          const pathVal = (row["path"] || row["filepath"] || row["file"] || row["filename"]) as string;
          const filename = pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || pathVal) : String(row["id"]);
          return filename === selectedFile.filename;
        });
        
        if (matchingRow) {
          const rowId = String(matchingRow["id"] || matchingRow["path"] || matchingRow["filepath"] || matchingRow["file"] || matchingRow["filename"]);
          setSelectedRow(rowId);
        }
      }
    }
  }, [selectedFile, uploadedFiles, datasetMetadata]);

  // Stable handlers to prevent downstream re-renders
  const handleRowSelect = useCallback(async (id: string) => {
    setSelectedRow(id);
    
    // When a row is selected, just propagate the file selection for UI/audio playback
    // No inference should be triggered here
    if (!onFileSelect) {
      return;
    }
    
    // When showing combined data (uploaded + dataset files), check if it's an uploaded file first
    if (dataset === "custom") {
      const uploadedFile = uploadedFiles?.find(f => f.file_id === id);
      if (uploadedFile) {
        onFileSelect(uploadedFile);
        return;
      }
      // If not an uploaded file, treat it as a dataset file (fall through to dataset logic)
    }

    const findMatch = () => {
      for (const row of datasetMetadata) {
        const rowId = row["id"]; 
        const path = row["path"] || row["filepath"] || row["file"] || row["filename"];
        if (typeof rowId === "string" && rowId === id) return row;
        if (typeof path === "string" && (path === id || path.endsWith(`/${id}`) || path.endsWith(`\\${id}`))) return row;
      }
      return null;
    };

    const match = findMatch();
    if (!match) return;

    const pathVal = (match["path"] || match["filepath"] || match["file"] || match["filename"]) as string | undefined;
    const filename = pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || String(id)) : String(id);

    // Ground truth lives in the dataset metadata row; attach it here so the
    // prediction panels can display it and compute accuracy metrics.
    const groundTruth = String(
      match["sentence"] ?? match["transcript"] ?? match["text"] ?? match["emotion"] ?? match["label"] ?? "",
    );

    try {
      const audio = await materializeAudio(originalDataset || dataset, filename);
      onFileSelect({
        audio_id: audio.audio_id,
        file_id: audio.file_id,
        filename: audio.filename,
        playback_url: audio.playback_url,
        message: audio.message || "Selected from dataset",
        size: audio.size_bytes,
        duration: audio.duration_seconds,
        sample_rate: audio.sample_rate,
        ground_truth: groundTruth || undefined,
      });
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Unable to prepare audio');
    }
  }, [dataset, originalDataset, datasetMetadata, onFileSelect, uploadedFiles]);

  const handleFilePlay = useCallback((file: UploadedFile) => {
    if (onFileSelect) {
      onFileSelect(file);
    }
  }, [onFileSelect]);

  // No need for local prediction update handling since we use external predictionMap

  const handleVisibleRowIdsChange = useCallback((ids: string[]) => {
    // This is now just for pagination, no inference triggering
  }, []);

  // Batch inference for entire dataset when model/dataset changes
  useEffect(() => {
    console.log('DEBUG: Batch inference useEffect triggered', {
      dataset,
      model,
      datasetMetadataLength: datasetMetadata.length,
      isCustom: dataset === "custom",
      hasModel: !!model
    });
    
    // Skip batch inference for legacy "custom" (uploaded files) but allow for custom datasets
    if (dataset === "custom" || !model) return;
    if (datasetMetadata.length === 0) return;
    
    const datasetToUse = originalDataset || dataset;
    const modelDatasetKey = `${model}-${datasetToUse}`;
    
    // If we've already completed inference for this model+dataset combination, don't restart
    if (isInferenceComplete && currentModelDataset === modelDatasetKey) {
      console.log(`Inference already completed for ${modelDatasetKey}, skipping`);
      return;
    }

    // Guard against duplicate concurrent runs for the same model+dataset —
    // StrictMode double-fires effects and unstable parent callbacks re-trigger
    // this effect, which previously spawned duplicate batch jobs.
    if (runningBatchKeyRef.current === modelDatasetKey) {
      return;
    }
    runningBatchKeyRef.current = modelDatasetKey;

    console.log(`Starting batch inference check for ${model} on ${datasetMetadata.length} files in ${dataset} dataset`);

    // Abort any ongoing inference
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();
    
    // Reset state for new model/dataset combination
    setCurrentModelDataset(modelDatasetKey);
    setIsInferenceComplete(false);
    setCurrentInferenceIndex(0);
    setBatchInferenceQueue([]);
    setInferenceStatus({}); // Clear inference status for new dataset
    
    // Run the whole dataset as ONE batch job. The worker fans it out into
    // per-file tasks and content-addressed caching makes repeat runs (page
    // refresh, model switch back) complete without re-running the models.
    const runBatchInference = async () => {
      const signal = abortControllerRef.current?.signal;
      const entries = datasetMetadata.map((row, index) => {
        const fileId = String(row["id"] || row["path"] || row["filepath"] || row["file"] || row["filename"] || index);
        const pathVal = (row["path"] || row["filepath"] || row["file"] || row["filename"]) as string;
        const filename = pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || pathVal) : fileId;
        return { fileId, filename };
      });

      try {
        setBatchInferenceQueue(entries.map((entry) => entry.fileId));
        setInferenceStatus(Object.fromEntries(entries.map((entry) => [entry.fileId, 'loading' as const])));
        onBatchInferenceStart?.();

        const assets = await Promise.all(
          entries.map((entry) => materializeAudio(datasetToUse, entry.filename, signal)),
        );
        const result: any = await runJob({
          operation: 'prediction',
          model: model || 'whisper-base',
          audio_ids: assets.map((asset) => asset.audio_id),
        }, {
          signal,
          onProgress: (status) => setCurrentInferenceIndex(status.progress?.current ?? 0),
        });

        // Aggregated batch results carry per-file values keyed by filename.
        const byFilename: Record<string, string> = {};
        for (const entry of result.individual_transcripts || []) byFilename[entry.filename] = entry.transcript;
        for (const entry of result.individual_predictions || []) byFilename[entry.filename] = entry.predicted_emotion;
        if (!result.individual_transcripts && !result.individual_predictions && Array.isArray(result.items)) {
          // Single-file jobs skip aggregation; items are request-ordered.
          result.items.forEach((item: any, index: number) => {
            const value = item.result;
            byFilename[entries[index]?.filename] = typeof value === 'string'
              ? value : value?.text || value?.predicted_emotion || JSON.stringify(value);
          });
        }

        const statuses: Record<string, 'idle' | 'loading' | 'done' | 'error'> = {};
        entries.forEach((entry) => {
          const text = byFilename[entry.filename];
          if (text !== undefined) {
            onPredictionUpdate?.(entry.fileId, text);
            statuses[entry.fileId] = 'done';
          } else {
            statuses[entry.fileId] = 'error';
          }
        });
        setInferenceStatus(statuses);
        setCurrentInferenceIndex(entries.length);
        setIsInferenceComplete(true);
        runningBatchKeyRef.current = null;
        onBatchInferenceComplete?.();
      } catch (error: any) {
        runningBatchKeyRef.current = null;
        if (error?.name === 'AbortError') return;
        console.error('Batch inference failed:', error);
        setInferenceStatus({});
        setBatchInferenceQueue([]);
        onBatchInferenceComplete?.();
      }
    };

    runBatchInference();
  }, [model, dataset, originalDataset, datasetMetadata, onBatchInferenceStart, onBatchInferenceComplete]);

  // Cleanup on unmount or when dataset changes
  // Reload function to refresh dataset metadata
  const handleReloadDataset = useCallback(async () => {
    const allowed = ["common-voice", "ravdess"];
    const datasetToUse = originalDataset || dataset;
    if (!allowed.includes(datasetToUse)) {
      setDatasetMetadata([]);
      return;
    }
    
    try {
      const res = await fetch(`${API_BASE}/${dataset}/metadata`, { credentials: 'include' });
      if (!res.ok) throw new Error(`Failed to fetch metadata: ${res.status}`);
      const data = await res.json();
      if (Array.isArray(data)) {
        const rows = data as Record<string, string | number>[];
        setDatasetMetadata(rows);
        onDatasetMetadataChange?.(rows);

        // Extract filenames for embeddings
        const filenames = data.map((row: Record<string, string | number>) => {
          const pathVal = row["path"] || row["filepath"] || row["file"] || row["filename"];
          const filename = typeof pathVal === 'string' ?
            (pathVal.split("/").pop() || pathVal.split("\\").pop() || pathVal) :
            String(pathVal);
          return filename;
        });

        onAvailableFilesChange?.(filenames);
        toast.success("Dataset reloaded successfully");
      } else {
        setDatasetMetadata([]);
        onDatasetMetadataChange?.([]);
        onAvailableFilesChange?.([]);
      }
    } catch (error) {
      console.error('Failed to reload dataset:', error);
      toast.error("Failed to reload dataset");
    }
  }, [dataset, originalDataset, onAvailableFilesChange, onDatasetMetadataChange]);

  useEffect(() => {
    abortControllerRef.current = new AbortController();
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [model, dataset]);

  // Fetch dataset metadata when originalDataset changes 
  useEffect(() => {
    const datasetToUse = originalDataset || dataset;
    
    // Skip legacy "custom" (individual uploaded files)
    if (datasetToUse === "custom") {
      setDatasetMetadata([]);
      return;
    }
    
    // Handle both global datasets and custom datasets
    const allowed = ["common-voice", "ravdess"];
    const isCustomDataset = datasetToUse.startsWith('custom:');
    
    if (!allowed.includes(datasetToUse) && !isCustomDataset) {
      setDatasetMetadata([]);
      return;
    }
    
    const ac = new AbortController();
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/${datasetToUse}/metadata`, { signal: ac.signal, credentials: 'include' });
        if (!res.ok) throw new Error(`Failed to fetch metadata: ${res.status}`);
        const data = await res.json();
        if (Array.isArray(data)) {
          const rows = data as Record<string, string | number>[];
          setDatasetMetadata(rows);
          onDatasetMetadataChange?.(rows);

          // Extract filenames for embeddings
          const filenames = data.map((row: Record<string, string | number>) => {
            const pathVal = row["path"] || row["filepath"] || row["file"] || row["filename"];
            const filename = typeof pathVal === 'string' ?
              (pathVal.split("/").pop() || pathVal.split("\\").pop() || pathVal) :
              String(pathVal);
            return filename;
          });

          onAvailableFilesChange?.(filenames);
        } else {
          setDatasetMetadata([]);
          onDatasetMetadataChange?.([]);
          onAvailableFilesChange?.([]);
        }
      } catch (e) {
        const name = (e as { name?: string } | null)?.name;
        if (name !== 'AbortError') console.error(e);
      }
    })();
    return () => ac.abort();
  }, [originalDataset, dataset]);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        // Check both MIME type and file extension for better .flac support
        const allowedExtensions = ['.wav', '.mp3', '.m4a', '.flac'];
        const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
        const isValidFile = file.type.startsWith('audio/') || allowedExtensions.includes(fileExtension);
        
        if (isValidFile) {
          try {
            await uploadFile(file, model ?? "");
          } catch (error) {
            console.error('Upload error:', error);
          }
        } else {
          toast.error(`Invalid file type: ${file.name}. Supported formats: WAV, MP3, M4A, FLAC`);
        }
      }
    }
    // Reset the input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const uploadFile = async (file: File, model: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        credentials: 'include',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await response.json();
      setUploadedFiles(prevFiles => [...prevFiles, data]);
      toast.success(`Uploaded: ${file.name}`);
      
      if (onUploadSuccess) {
        onUploadSuccess(data);
      }
      
      return data;
    } catch (error) {
      console.error('Upload error:', error);
      const msg = error instanceof Error ? error.message : 'Unknown error';
      toast.error(`Failed to upload ${file.name}: ${msg}`);
      throw error;
    }
  };

  return (
    <TooltipProvider>
      <div className="h-full bg-panel-background flex flex-col">
        <div className="bg-panel-header p-3 border-b border-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <h3 className="font-semibold text-foreground text-sm">Audio Dataset</h3>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
                </TooltipTrigger>
                <TooltipContent className="text-xs space-y-1">
                  <p>Browse and manage audio files in your selected dataset.</p>
                  <p>Upload new files or select from existing datasets.</p>
                </TooltipContent>
              </Tooltip>
            </div>
            <div className="flex items-center gap-1.5">
              <Badge variant="outline" className="text-[10px] bg-muted">
                {uploadedFiles ? `${uploadedFiles.length} uploaded` : "0 files"}
              </Badge>
              {batchInferenceStatus === 'running' && batchInferenceQueue.length > 0 && (
                <Badge variant="outline" className="text-[10px] bg-primary/10 text-primary border-primary/20">
                  Inferencing... {currentInferenceIndex}/{batchInferenceQueue.length}
                </Badge>
              )}
              {(batchInferenceStatus === 'done' || isInferenceComplete) && (
                <Badge variant="outline" className="text-[10px] bg-primary text-primary-foreground border-primary">
                  ✓ Inference Complete
                </Badge>
              )}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button size="sm" variant="secondary" className="h-7 text-xs" onClick={handleUploadClick}>
                    <Upload className="h-3 w-3 mr-1" />
                    Upload
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Upload audio files (.wav, .mp3, .m4a, .flac)</p>
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button size="sm" variant="secondary" className="h-7 w-7 p-0" onClick={handleReloadDataset} title="Reload dataset">
                    <RefreshCw className="h-3 w-3" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Reload dataset metadata and refresh the file list</p>
                </TooltipContent>
              </Tooltip>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*,.flac,.wav,.mp3,.m4a"
              multiple
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>
        </div>
        
        {/* Search bar */}
        <div className="px-3 pt-2.5 pb-1">
          <div className="relative border border-gray-200 rounded-lg px-2 py-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-3 w-3 text-muted-foreground" />
            <Tooltip>
              <TooltipTrigger asChild>
                <Input
                  placeholder="Search audio files..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9 h-6 text-xs bg-transparent border-0 focus:ring-0 rounded-md"
                />
              </TooltipTrigger>
              <TooltipContent>
                <p>Search by filename or any metadata field</p>
              </TooltipContent>
            </Tooltip>
          </div>
          {filterFilenames && filterFilenames.length > 0 && (
            <div className="flex items-center gap-1.5 pt-1.5">
              <Badge variant="outline" className="text-[10px] bg-primary/10 text-primary border-primary/20">
                Filtered to {filterFilenames.length} files
              </Badge>
              <Button
                size="sm"
                variant="ghost"
                className="h-5 px-1.5 text-[10px]"
                onClick={onClearFilter}
              >
                Clear
              </Button>
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-hidden px-3 pb-3">
        <Card className="h-full rounded-lg">
          <CardContent className="p-0 h-full">
            <AudioDataTable
              selectedRow={selectedRow}
              onRowSelect={handleRowSelect}
              searchQuery={searchQuery}
              apiData={apiData}
              model={model ?? ""}
              dataset={dataset}
              datasetMetadata={datasetMetadata}
              uploadedFiles={uploadedFiles}
              onFilePlay={handleFilePlay}
              predictionMap={predictionMap}
              inferenceStatus={inferenceStatus}
              onVisibleRowIdsChange={handleVisibleRowIdsChange}
              filterFilenames={filterFilenames}
            />
          </CardContent>
        </Card>
      </div>
      
      {/* Upload overlay */}
      <AudioUploader onUploadSuccess={onUploadSuccess} model={model} />
    </div>
    </TooltipProvider>
  );
};
