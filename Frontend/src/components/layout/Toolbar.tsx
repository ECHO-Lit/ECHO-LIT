import React, { useState, useEffect } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  SelectSeparator,
  SelectLabel,
  SelectGroup,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { Upload, HelpCircle, PanelLeft, PanelBottom, PanelRight } from "lucide-react";
import { Link } from "react-router-dom";
import { API_BASE } from '@/lib/api';
import { CustomDatasetManager } from '@/components/dataset/CustomDatasetManager';
import { CustomModelManager } from '@/components/model/CustomModelManager';
import { listCustomModels, type CustomModel } from '@/lib/models';
import { toast } from 'sonner';
import { describeHttpError } from '@/lib/httpError';

interface UploadedFile {
  audio_id?: string;
  file_id: string;
  filename: string;
  playback_url?: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

interface ToolbarProps {
  apiData: unknown;
  setApiData: (data: unknown) => void;
  selectedFile?: UploadedFile | null;
  uploadedFiles?: UploadedFile[];
  onFileSelect?: (file: UploadedFile) => void;
  model: string;
  setModel: (model: string) => void; // important for lifting state
  dataset: string;
  setDataset: (dataset: string) => void;
  onBatchInference?: (model: string, dataset: string) => void; // New callback for batch inference
  leftPanelOpen?: boolean;
  rightPanelOpen?: boolean;
  bottomPanelOpen?: boolean;
  onToggleLeftPanel?: () => void;
  onToggleRightPanel?: () => void;
  onToggleBottomPanel?: () => void;
  /** Opens the dataset panel's file picker. Without it the Upload button is inert. */
  onUploadClick?: () => void;
}

interface CustomDataset {
  dataset_name: string;
  formatted_name: string;
  total_files: number;
}

const modelDatasetMap: Record<string, string[]> = {
  "whisper-base": ["common-voice", "ravdess", "l2-arctic", "saa", "custom"],
  "whisper-large": ["common-voice", "ravdess", "l2-arctic", "saa", "custom"],
  "wav2vec2": ["common-voice", "ravdess", "l2-arctic", "saa", "custom"],
};

const defaultDatasetForModel: Record<string, string> = {
  "whisper-base": "common-voice",
  "whisper-large": "common-voice",
  "wav2vec2": "ravdess",
};

export const Toolbar = ({apiData, setApiData, selectedFile, uploadedFiles, onFileSelect, model, setModel, dataset, setDataset, onBatchInference, leftPanelOpen, rightPanelOpen, bottomPanelOpen, onToggleLeftPanel, onToggleRightPanel, onToggleBottomPanel, onUploadClick}: ToolbarProps) => {
  const [customDatasets, setCustomDatasets] = useState<CustomDataset[]>([]);
  const [customModels, setCustomModels] = useState<CustomModel[]>([]);

  const fetchCustomDatasets = async () => {
    try {
      const response = await fetch(`${API_BASE}/upload/dataset/list`, {
        credentials: 'include'
      });
      
      if (response.ok) {
        const data = await response.json();
        setCustomDatasets(data.datasets || []);
      } else {
        // An HTTP error used to be swallowed entirely here — the `if (response.ok)`
        // had no else, so a 500 did not even reach console.error. The dropdown
        // simply rendered with no custom datasets and the user concluded their
        // uploads were gone.
        throw await describeHttpError(response);
      }
    } catch (err) {
      console.error('Error fetching custom datasets:', err);
      toast.error('Could not load your custom datasets. They are still there — reload the page to try again.');
    }
  };

  useEffect(() => {
    fetchCustomDatasets();
  }, []);

  useEffect(() => {
    listCustomModels().then(setCustomModels).catch((error) => {
      console.error('Error fetching custom models:', error);
      toast.error('Could not load your registered models. Reload the page to try again.');
    });
  }, []);

  const handleDatasetCreated = (datasetName: string) => {
    fetchCustomDatasets(); // Refresh the list
    setDataset(datasetName); // Automatically select the new dataset
  };

  const handleDatasetSelected = (datasetName: string) => {
    setDataset(datasetName);
  };

const onModelChange = (value: string) => {
  setModel(value);
  
  // Update dataset based on model
  const allowedDatasets = modelDatasetMap[value] || ["custom"];
  const defaultDataset = defaultDatasetForModel[value] || "custom";
  
  // Check if current dataset is a custom dataset
  const isCurrentCustomDataset = dataset.startsWith('custom:');

  if (!allowedDatasets.includes(dataset) && !isCurrentCustomDataset) {
    // Use the canonical handler so all side effects fire (metadata loading)
    onDatasetChange(defaultDataset);
  } else if (!isCurrentCustomDataset && dataset !== 'custom' && onBatchInference) {
    // Dataset is already valid and not custom, fire batch inference directly
    onBatchInference(value, dataset);
  }
};

  const onDatasetChange = (value: string) => {
    setDataset(value);
    
    // Check if this is a custom dataset (formatted as custom:session_id:dataset_name)
    const isCustomDataset = value.startsWith('custom:');
    
    // Trigger batch inference when dataset changes (except for custom datasets)
    if (!isCustomDataset && value !== 'custom' && onBatchInference) {
      onBatchInference(model, value);
    }
  };

  // Get datasets allowed for current model
  const allowedDatasets = modelDatasetMap[model] || ["custom"];

  return (
    <TooltipProvider>
      <div className="h-12 bg-white border-b border-border px-5 flex items-center justify-between">
        {/* Left side: Model and Dataset selectors */}
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-2.5">
            <span className="text-base font-bold text-foreground">LIT for Voice</span>
            <Badge variant="outline" className="text-[10px] bg-primary/10 text-primary border-primary/20">
              v1.0
            </Badge>
          </div>
          <Button asChild variant="ghost" size="sm" className="h-7 text-xs text-muted-foreground">
            <Link to="/j-lens">J-Lens Lab</Link>
          </Button>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <div className="flex items-center gap-1">
                <span className="text-xs font-medium text-foreground">Model:</span>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      aria-label="About the model options"
                      className="text-muted-foreground hover:text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-sm"
                    >
                      <HelpCircle className="h-3 w-3" aria-hidden="true" />
                    </button>
                  </TooltipTrigger>
                    <TooltipContent className="space-y-1">
                    <p className="text-xs">Choose the AI model for audio analysis:</p>
                    <p className="text-xs">• Whisper: Speech-to-text transcription</p>
                    <p className="text-xs">• Wav2Vec2: Emotion recognition</p>
                    </TooltipContent>
                </Tooltip>
              </div>
              <Select value={model} onValueChange={onModelChange}>
                <SelectTrigger className="w-32 h-7 border-border text-xs" aria-label="Model">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="whisper-base">Whisper Base</SelectItem>
                  <SelectItem value="wav2vec2">Wav2Vec2</SelectItem>
                  {customModels.filter((customModel) => customModel.status === 'ready').map((customModel) => (
                    <SelectItem key={customModel.model_id} value={customModel.model_id}>
                      {customModel.hf_repo}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-1.5">
              <div className="flex items-center gap-1">
                <span className="text-xs font-medium text-foreground">Dataset:</span>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      aria-label="About the dataset options"
                      className="text-muted-foreground hover:text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-sm"
                    >
                      <HelpCircle className="h-3 w-3" aria-hidden="true" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent className="space-y-1">
                    <p className="text-xs">Select the audio dataset to analyze:</p>
                    <p className="text-xs">• Common Voice: Speech recognition dataset</p>
                    <p className="text-xs">• RAVDESS: Emotion recognition dataset</p>
                    <p className="text-xs">• L2-ARCTIC: L2 speech error dataset</p>
                    <p className="text-xs">• SAA: Speaker accent dataset</p>
                    <p className="text-xs">• Custom: Your uploaded datasets</p>
                  </TooltipContent>
                </Tooltip>
              </div>
              <Select value={dataset} onValueChange={onDatasetChange}>
              <SelectTrigger className="w-40 h-7 border-border text-xs" aria-label="Dataset">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {/* Built-in datasets */}
                {allowedDatasets.filter(ds => ds !== 'custom').map((ds) => {
                  let label = ds;
                  if (ds === "common-voice") label = "Common Voice";
                  else if (ds === "ravdess") label = "RAVDESS";
                  else if (ds === "l2-arctic") label = "L2-ARCTIC";
                  else if (ds === "saa") label = "SAA";
                  return (
                    <SelectItem key={ds} value={ds}>
                      {label}
                    </SelectItem>
                  );
                })}
                
                {/* Custom datasets */}
                {customDatasets.length > 0 && (
                  <>
                    <SelectSeparator />
                    <SelectGroup>
                      <SelectLabel>Custom Datasets</SelectLabel>
                      {customDatasets.map((customDataset) => (
                        <SelectItem key={customDataset.formatted_name} value={customDataset.formatted_name}>
                          {customDataset.dataset_name}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </>
                )}
              </SelectContent>
            </Select>
          </div>

          {uploadedFiles && uploadedFiles.length > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-medium text-foreground">File:</span>
              <Select
                value={selectedFile?.file_id || ""}
                onValueChange={(fileId) => {
                  const file = uploadedFiles.find(f => f.file_id === fileId);
                  if (file && onFileSelect) {
                    onFileSelect(file);
                  }
                }}
              >
                <SelectTrigger className="w-48 h-7 border-border text-xs">
                  <SelectValue placeholder="Select uploaded file" />
                </SelectTrigger>
                <SelectContent>
                  {uploadedFiles.map((file) => (
                    <SelectItem key={file.file_id} value={file.file_id}>
                      {file.filename}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>
      </div>

      {/* Right side: Action buttons */}
      <div className="flex items-center gap-2.5">
        <div className="flex items-center gap-1">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant={leftPanelOpen ? "secondary" : "ghost"}
                size="icon"
                className="h-7 w-7"
                aria-label="Toggle left panel"
                aria-pressed={Boolean(leftPanelOpen)}
                onClick={onToggleLeftPanel}
              >
                <PanelLeft className="h-3.5 w-3.5" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Toggle left panel</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant={bottomPanelOpen ? "secondary" : "ghost"}
                size="icon"
                className="h-7 w-7"
                aria-label="Toggle bottom panel"
                aria-pressed={Boolean(bottomPanelOpen)}
                onClick={onToggleBottomPanel}
              >
                <PanelBottom className="h-3.5 w-3.5" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Toggle bottom panel</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant={rightPanelOpen ? "secondary" : "ghost"}
                size="icon"
                className="h-7 w-7"
                aria-label="Toggle right panel"
                aria-pressed={Boolean(rightPanelOpen)}
                onClick={onToggleRightPanel}
              >
                <PanelRight className="h-3.5 w-3.5" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Toggle right panel</p>
            </TooltipContent>
          </Tooltip>
        </div>

        <CustomDatasetManager
          onDatasetCreated={handleDatasetCreated}
          onDatasetSelected={handleDatasetSelected}
        />
        <CustomModelManager onModelsChanged={setCustomModels} />

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="default"
              size="sm"
              className="h-7 text-xs shadow-aws-sm"
              onClick={onUploadClick}
            >
              <Upload className="h-3.5 w-3.5 mr-1.5" aria-hidden="true" />
              Upload
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Upload audio files for analysis</p>
            <p className="text-xs text-muted-foreground">WAV, MP3, M4A or FLAC · up to 100 MB and 10 minutes per file</p>
          </TooltipContent>
        </Tooltip>
      </div>
    </div>
    </TooltipProvider>
  );
};
