import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { AudioPlayer } from "../audio/AudioPlayer";
import { WaveformViewer } from "../audio/WaveformViewer";
import { PredictionDisplay } from "../predictions/PredictionDisplay";
import { Play, Pause, RotateCcw, Trash2, Plus, HelpCircle } from "lucide-react";
import WaveSurfer from "wavesurfer.js";
import { API_BASE } from '@/lib/api';

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

interface Wav2Vec2Prediction {
  predicted_emotion: string;
  probabilities: Record<string, number>;
  confidence: number;
  ground_truth_emotion?: string;
}

interface WhisperPrediction {
  predicted_transcript: string;
  ground_truth: string;
  accuracy_percentage: number | null;
  word_error_rate: number | null;
  character_error_rate: number | null;
  levenshtein_distance: number | null;
  exact_match: number | null;
  character_similarity: number | null;
  word_count_predicted: number;
  word_count_truth: number;
}

interface PerturbationResult {
  perturbed_file: string;
  filename: string;
  duration_ms: number;
  sample_rate: number;
  applied_perturbations: Array<{
    type: string;
    params: Record<string, any>;
    status: string;
    error?: string;
  }>;
  success: boolean;
  error?: string;
}

interface DatapointEditorPanelProps {
  selectedFile?: UploadedFile | null;
  selectedEmbeddingFile?: string | null;
  dataset?: string; // "custom" | dataset key (effective dataset)
  originalDataset?: string; // Original dataset selection from toolbar
  perturbationResult?: PerturbationResult | null;
  predictionMap?: Record<string, string>;
  model?: string;
  wav2vecPrediction?: Wav2Vec2Prediction | null;
  whisperPrediction?: WhisperPrediction | null;
  perturbedPredictions?: Wav2Vec2Prediction | WhisperPrediction | null;
  isLoadingPredictions?: boolean;
  isLoadingPerturbed?: boolean;
  predictionError?: string | null;
}

export const DatapointEditorPanel = ({ 
  selectedFile, 
  selectedEmbeddingFile,
  dataset = "custom", 
  originalDataset,
  perturbationResult, 
  predictionMap,
  model,
  wav2vecPrediction,
  whisperPrediction,
  perturbedPredictions,
  isLoadingPredictions,
  isLoadingPerturbed,
  predictionError
}: DatapointEditorPanelProps) => {
  const [selectedLabel, setSelectedLabel] = useState<string>("neutral");
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [showPerturbed, setShowPerturbed] = useState(false);
  const wavesurferRef = useRef<WaveSurfer | null>(null);

  const audioUrl = (() => {
    // If showing perturbed audio and it's available
    if (showPerturbed && perturbationResult?.success) {
      const filename = perturbationResult.filename;
      return `${API_BASE}/upload/file/${filename}`;
    }
    
    // Otherwise show original audio
    if (!selectedFile) return undefined;
    
    // Check if this is an uploaded file - more precise detection
    const isUploadedFile = selectedFile.file_path && (
      selectedFile.file_path.includes('uploads/') || 
      selectedFile.file_path.startsWith('uploads/') ||
      selectedFile.message === "Perturbed file" ||
      selectedFile.message === "File uploaded successfully" ||
      selectedFile.message === "File uploaded and processed successfully"
    ) && selectedFile.message !== "Selected from embeddings" && selectedFile.message !== "Selected from dataset";
    
    if (isUploadedFile) {
      // This is an uploaded file, use the upload endpoint
      return `${API_BASE}/upload/file/${selectedFile.file_id}`;
    }
    
    // For dataset files (including files selected from embeddings)
    // Use original dataset if available and it's a real dataset
    const datasetToUse = originalDataset && originalDataset !== "custom" ? originalDataset : dataset;
    
    if (datasetToUse && datasetToUse !== "custom") {
      // This is a dataset file from built-in or custom datasets
      const filename = encodeURIComponent(selectedFile.filename);
      
      // Handle custom datasets vs built-in datasets
      if (datasetToUse.startsWith('custom:')) {
        // Custom dataset: use the original route /{dataset}/file/{filename}
        // The backend handles the custom dataset format properly
        return `${API_BASE}/${encodeURIComponent(datasetToUse)}/file/${filename}`;
      } else {
        // Built-in dataset: use /{dataset}/file/{filename}
        return `${API_BASE}/${encodeURIComponent(datasetToUse)}/file/${filename}`;
      }
    } else {
      // Fallback to upload endpoint when dataset is "custom" (generic case)
      return `${API_BASE}/upload/file/${selectedFile.file_id}`;
    }
  })();

  // Get current file info (original or perturbed) with better data handling
  const currentFileInfo = (() => {
    if (showPerturbed && perturbationResult?.success) {
      return {
        filename: perturbationResult.filename,
        duration: perturbationResult.duration_ms / 1000,
        sample_rate: perturbationResult.sample_rate,
        size: undefined
      };
    }
    
    // For original file, try to get the most accurate data
    if (selectedFile) {
      return {
        filename: selectedFile.filename,
        duration: selectedFile.duration || undefined,
        sample_rate: selectedFile.sample_rate || undefined,
        size: selectedFile.size || undefined
      };
    }
    
    return null;
  })();

  // Add a state to track audio metadata from wavesurfer
  const [audioMetadata, setAudioMetadata] = useState<{
    duration?: number;
    sampleRate?: number;
  }>({});

  // Debug logging for selectedFile and audioUrl
  useEffect(() => {
  }, [selectedFile, audioUrl, dataset, originalDataset]);

  // Reset playback when file changes or when switching between original/perturbed
  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setAudioMetadata({}); // Reset metadata when file changes

    // Reset wavesurfer instance if it exists
    if (wavesurferRef.current) {
      wavesurferRef.current.stop();
    }
  }, [selectedFile?.file_id, dataset, showPerturbed, perturbationResult?.filename]);

  // Snap the Original/Perturbed toggle back to Original whenever the selected
  // file changes or when there is no valid perturbation for the current file.
  // Without this, `showPerturbed` (local state) sticks after switching files,
  // leaving the panel rendering the "Perturbed" tab with only a placeholder.
  useEffect(() => {
    setShowPerturbed(false);
  }, [selectedFile?.file_id]);
  
  return (
    <TooltipProvider>
      <div className="h-full bg-panel-background border-l border-border flex flex-col">
        <div className="bg-panel-header p-3 border-b border-border">
          <h3 className="font-semibold text-sm text-foreground flex items-center gap-1.5">
            Datapoint Editor
            <Tooltip>
              <TooltipTrigger>
                <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
              </TooltipTrigger>
              <TooltipContent>
                Edit and analyze individual audio samples with predictions and perturbations
              </TooltipContent>
            </Tooltip>
          </h3>
        </div>

      <div className="flex-1 p-3 overflow-auto space-y-3">
        {/* Sample Info - Top */}
        <Card>
          <CardHeader className="bg-panel-header">
            <div className="flex items-center justify-between">
              <CardTitle className="text-xs flex items-center gap-1.5">
                Sample Info
                <Tooltip>
                  <TooltipTrigger>
                    <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
                  </TooltipTrigger>
                  <TooltipContent>
                    Detailed information about the selected audio sample
                  </TooltipContent>
                </Tooltip>
              </CardTitle>
              {perturbationResult?.success && (
                <div className="flex items-center gap-0.5 p-0.5 bg-muted border border-border rounded-md">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant={!showPerturbed ? "default" : "ghost"}
                        size="sm"
                        onClick={() => setShowPerturbed(false)}
                        className={`text-[10px] h-6 px-2.5 transition-all ${
                          !showPerturbed
                            ? 'bg-primary hover:bg-primary-hover text-primary-foreground shadow-aws-sm'
                            : 'text-muted-foreground hover:bg-background'
                        }`}
                      >
                        Original
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      View the original unmodified audio file
                    </TooltipContent>
                  </Tooltip>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant={showPerturbed ? "default" : "ghost"}
                        size="sm"
                        onClick={() => setShowPerturbed(true)}
                        className={`text-[10px] h-6 px-2.5 transition-all ${
                          showPerturbed
                            ? 'bg-primary hover:bg-primary-hover text-primary-foreground shadow-aws-sm'
                            : 'text-muted-foreground hover:bg-background'
                        }`}
                      >
                        Perturbed
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent className="font-normal">
                      View the modified audio file with applied perturbations
                    </TooltipContent>
                  </Tooltip>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-1.5">
            <div className="text-xs-tight">
              <span className="text-gray-500">File:</span>
              <span className="ml-2 font-mono text-gray-700">{currentFileInfo?.filename || "No file selected"}</span>
              {showPerturbed && (
                <Badge variant="secondary" className="ml-2 text-[9px] bg-blue-100 text-blue-700 border-blue-200">P</Badge>
              )}
            </div>
            <div className="text-xs-tight">
              <span className="text-gray-500">Duration:</span>
              <span className="ml-2 text-gray-700">
                {currentFileInfo?.duration 
                  ? `${currentFileInfo.duration.toFixed(1)}s` 
                  : audioMetadata.duration 
                  ? `${audioMetadata.duration.toFixed(1)}s` 
                  : "Loading..."}
              </span>
            </div>
            <div className="text-xs-tight">
              <span className="text-gray-500">Sample Rate:</span>
              <span className="ml-2 text-gray-700">
                {currentFileInfo?.sample_rate 
                  ? `${(currentFileInfo.sample_rate / 1000).toFixed(1)}kHz` 
                  : audioMetadata.sampleRate 
                  ? `${(audioMetadata.sampleRate / 1000).toFixed(1)}kHz` 
                  : "Loading..."}
              </span>
            </div>
            {currentFileInfo?.size && (
              <div className="text-xs-tight">
                <span className="text-gray-500">Size:</span>
                <span className="ml-2 text-gray-700">{(currentFileInfo.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>
            )}
            {showPerturbed && perturbationResult?.applied_perturbations && (
              <div className="text-xs-tight">
                <span className="text-gray-500">Applied:</span>
                <div className="ml-2 mt-1 space-y-1">
                  {perturbationResult.applied_perturbations.map((pert, idx) => (
                    <Badge key={idx} variant="outline" className="text-[9px] mr-1 border-blue-300 text-blue-700">
                      {pert.type.replace('_', ' ')}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            {showPerturbed && perturbationResult?.filename && predictionMap && (
              <div className="text-xs-tight mt-2">
                <span className="text-gray-500">Perturbed Prediction:</span>
                <div className="ml-2 mt-1">
                  <Badge variant="secondary" className="text-[10px] bg-blue-100 text-blue-700 border-blue-200">
                    {predictionMap[perturbationResult.filename] || "Loading..."}
                  </Badge>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Predictions Section - Middle */}
        <PredictionDisplay
          selectedFile={selectedFile}
          selectedEmbeddingFile={selectedEmbeddingFile}
          model={model}
          wav2vecPrediction={wav2vecPrediction}
          whisperPrediction={whisperPrediction}
          perturbedPredictions={perturbedPredictions}
          isLoading={isLoadingPredictions}
          isLoadingPerturbed={isLoadingPerturbed}
          error={predictionError}
          showPerturbed={showPerturbed}
        />

        {/* Audio Player & Waveform - Bottom */}
        <Card>
          <CardHeader className="bg-panel-header">
            <CardTitle className="text-xs flex items-center gap-1.5">
              Audio Playback
              <Tooltip>
                <TooltipTrigger>
                  <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
                </TooltipTrigger>
                <TooltipContent>
                  Interactive audio player with waveform visualization
                </TooltipContent>
              </Tooltip>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2.5">
            <WaveformViewer 
              audioUrl={audioUrl}
              isPlaying={isPlaying}
              onReady={(wavesurfer) => {
      
                wavesurferRef.current = wavesurfer;
                const duration = wavesurfer.getDuration();
                setDuration(duration);
                
                // Update metadata state for file info display
                setAudioMetadata({
                  duration: duration,
                  sampleRate: wavesurfer.getDecodedData()?.sampleRate || undefined
                });
              }}
              onProgress={(time, dur) => {
                setCurrentTime(time);
                setDuration(dur);
                
                // Update duration in metadata if not already set
                if (!audioMetadata.duration && dur > 0) {
                  setAudioMetadata(prev => ({ ...prev, duration: dur }));
                }
              }}
            />
            <AudioPlayer 
              isPlaying={isPlaying}
              onPlayPause={() => {
                setIsPlaying(!isPlaying);
                if (wavesurferRef.current) {
                  if (isPlaying) {
                    wavesurferRef.current.pause();
                  } else {
                    wavesurferRef.current.play();
                  }
                }
              }}
              currentTime={currentTime}
              duration={duration}
              onSeek={(time) => {
                if (wavesurferRef.current) {
                  wavesurferRef.current.seekTo(time / duration);
                }
              }}
              onVolumeChange={(volume) => {
                if (wavesurferRef.current) {
                  wavesurferRef.current.setVolume(volume);
                }
              }}
            />
          </CardContent>
        </Card>
      </div>
    </div>
    </TooltipProvider>
  );
};