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
import { firstJobResult, materializeAudio, runJob } from '@/lib/jobs';

interface UploadedFile {
  audio_id?: string;
  file_id: string;
  filename: string;
  playback_url?: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
  prediction?:string
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
  onPredictionUpdate?: (fileId: string, prediction: string) => void;
  predictionMap?: Record<string, string>;
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
  onPredictionUpdate,
  predictionMap: externalPredictionMap
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
    
    // First, check what's already cached
    const checkCachedResults = async () => {
      try {
        const filenames = datasetMetadata.map(row => {
          const pathVal = (row["path"] || row["filepath"] || row["file"] || row["filename"]) as string;
          return pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || pathVal) : String(row["id"] || "unknown");
        });

        // The job worker performs content-addressed cache lookup. Queue all files;
        // cached jobs complete without model execution.
        const cached_results: Record<string, string> = {};
        const missing_files = filenames;
        const cache_hit_rate = 0;
        
        console.log(`Cache hit rate: ${(cache_hit_rate * 100).toFixed(1)}% (${Object.keys(cached_results).length}/${filenames.length})`);
        
        // Load cached results
        const newPredictionMap: Record<string, string> = {};
        const newInferenceStatus: Record<string, 'idle' | 'loading' | 'done' | 'error'> = {};
        
        // Map cached results to file IDs
        datasetMetadata.forEach((row, index) => {
          const fileId = String(row["id"] || row["path"] || row["filepath"] || row["file"] || row["filename"] || index);
          const pathVal = (row["path"] || row["filepath"] || row["file"] || row["filename"]) as string;
          const filename = pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || pathVal) : fileId;
          
          if (cached_results[filename]) {
            newPredictionMap[fileId] = cached_results[filename];
            newInferenceStatus[fileId] = 'done';
          } else {
            newInferenceStatus[fileId] = 'idle';
          }
        });
        
        // Update external predictionMap via callback
        Object.entries(newPredictionMap).forEach(([fileId, prediction]) => {
          if (onPredictionUpdate) {
            onPredictionUpdate(fileId, prediction);
          }
        });
        setInferenceStatus(newInferenceStatus);
        
        if (missing_files.length === 0) {
          // All files are cached, we're done!
          console.log('All files are cached, inference complete');
          setIsInferenceComplete(true);
          if (onBatchInferenceComplete) {
            onBatchInferenceComplete();
          }
          return;
        }
        
        // Queue only missing files for inference
        const fileIds = datasetMetadata
          .filter((row, index) => {
            const pathVal = (row["path"] || row["filepath"] || row["file"] || row["filename"]) as string;
            const filename = pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || pathVal) : String(row["id"] || index);
            return missing_files.includes(filename);
          })
          .map((row, index) => String(row["id"] || row["path"] || row["filepath"] || row["file"] || row["filename"] || index));
        
        setBatchInferenceQueue(fileIds);
        console.log(`Queuing ${fileIds.length} files for inference:`, fileIds);
        
        if (onBatchInferenceStart) {
          onBatchInferenceStart();
        }
        
      } catch (error: any) {
        if (error.name === 'AbortError') return;
        console.error('Failed to check cached results:', error);
        
        // Fallback: run inference on all files
        const fileIds = datasetMetadata.map((row, index) => {
          const id = row["id"] || row["path"] || row["filepath"] || row["file"] || row["filename"] || String(index);
          return String(id);
        });
        
        setBatchInferenceQueue(fileIds);
        setInferenceStatus({});
        
        if (onBatchInferenceStart) {
          onBatchInferenceStart();
        }
      }
    };
    
    checkCachedResults();
  }, [model, dataset, originalDataset, datasetMetadata, onBatchInferenceStart, onBatchInferenceComplete]);

  // Process batch inference queue
  useEffect(() => {
    if (batchInferenceQueue.length === 0) return;
    if (currentInferenceIndex >= batchInferenceQueue.length) {
      // Batch inference complete
      console.log('Batch inference completed');
      setIsInferenceComplete(true);
      if (onBatchInferenceComplete) {
        onBatchInferenceComplete();
      }
      return;
    }

    const currentFileId = batchInferenceQueue[currentInferenceIndex];
    const currentRow = datasetMetadata.find(row => {
      const id = row["id"] || row["path"] || row["filepath"] || row["file"] || row["filename"];
      return String(id) === currentFileId;
    });

    if (!currentRow) {
      // Skip this file and continue
      setCurrentInferenceIndex(prev => prev + 1);
      return;
    }

    const runInference = async () => {
      try {
        setInferenceStatus(prev => ({ ...prev, [currentFileId]: 'loading' }));
        
        const pathVal = (currentRow["path"] || currentRow["filepath"] || currentRow["file"] || currentRow["filename"]) as string;
        const filename = pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || currentFileId) : currentFileId;

        const asset = await materializeAudio(dataset, filename, abortControllerRef.current?.signal);
        const prediction = firstJobResult<any>(await runJob({
          operation: 'prediction', model: model || 'whisper-base', audio_ids: [asset.audio_id],
        }, { signal: abortControllerRef.current?.signal }));
        const predictionText = typeof prediction === 'string'
          ? prediction
          : prediction?.text || prediction?.predicted_emotion || JSON.stringify(prediction);

        // Update external predictionMap via callback
        if (onPredictionUpdate) {
          onPredictionUpdate(currentFileId, predictionText);
        }
        setInferenceStatus(prev => ({ ...prev, [currentFileId]: 'done' }));
        
        console.log(`Inference complete for ${filename}: ${predictionText}`);
        
      } catch (error: any) {
        if (error.name === 'AbortError') return;
        console.error(`Inference failed for ${currentFileId}:`, error);
        setInferenceStatus(prev => ({ ...prev, [currentFileId]: 'error' }));
      }
      
      // Move to next file
      setCurrentInferenceIndex(prev => prev + 1);
    };

    // Add small delay to prevent overwhelming the server
    const timeoutId = setTimeout(runInference, 100);
    
    return () => clearTimeout(timeoutId);
  }, [batchInferenceQueue, currentInferenceIndex, datasetMetadata, model, dataset, originalDataset, onBatchInferenceComplete]);

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
        setDatasetMetadata(data as Record<string, string | number>[]);
        
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
        onAvailableFilesChange?.([]);
      }
    } catch (error) {
      console.error('Failed to reload dataset:', error);
      toast.error("Failed to reload dataset");
    }
  }, [dataset, originalDataset, onAvailableFilesChange]);

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
          setDatasetMetadata(data as Record<string, string | number>[]);
          
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
