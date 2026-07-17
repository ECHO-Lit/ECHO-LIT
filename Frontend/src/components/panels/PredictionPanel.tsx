import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { SaliencyVisualization } from "../visualization/SaliencyVisualization";
import { AttentionVisualization } from "../visualization/AttentionVisualization";
import { PerturbationTools } from "../analysis/PerturbationTools";
import { useState, useEffect } from "react";
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
}

interface WhisperPrediction {
  predicted_transcript: string;
  ground_truth: string;
  accuracy_percentage: number;
  word_error_rate: number;
  character_error_rate: number;
  levenshtein_distance: number;
  exact_match: number;
  character_similarity: number;
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

interface PredictionPanelProps {
  selectedFile?: UploadedFile | null;
  selectedEmbeddingFile?: string | null;
  model?: string;
  dataset?: string;
  originalDataset?: string;
  onPerturbationComplete?: (result: PerturbationResult) => void;
  onPredictionRefresh?: (file: UploadedFile, prediction: string) => void;
  onPredictionUpdate?: (fileId: string, prediction: string) => void;
}

export const PredictionPanel = ({ selectedFile, selectedEmbeddingFile, model, dataset, originalDataset, onPerturbationComplete, onPredictionRefresh, onPredictionUpdate }: PredictionPanelProps) => {
  const [wav2vecPrediction, setWav2vecPrediction] = useState<Wav2Vec2Prediction | null>(null);
  const [whisperPrediction, setWhisperPrediction] = useState<WhisperPrediction | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [perturbedPredictions, setPerturbedPredictions] = useState<Wav2Vec2Prediction | WhisperPrediction | null>(null);
  const [originalFile, setOriginalFile] = useState<UploadedFile | null>(selectedFile || null);
  const [perturbedFile, setPerturbedFile] = useState<UploadedFile | null>(null);
  const [isLoadingPerturbed, setIsLoadingPerturbed] = useState(false);


  // Handle perturbation completion
  const handlePerturbationComplete = async (result: PerturbationResult) => {
    if (!result.success) {
      console.error("Perturbation failed:", result.error);
      return;
    }

    // Create a file-like object for the perturbed audio
    const perturbedFileObj: UploadedFile = {
      file_id: result.filename,
      filename: result.filename,
      file_path: result.perturbed_file,
      message: "Perturbed audio",
      duration: result.duration_ms / 1000,
      sample_rate: result.sample_rate
    };
    
    setPerturbedFile(perturbedFileObj);
    
    // Notify parent component about perturbation completion
    if (onPerturbationComplete) {
      onPerturbationComplete(result);
    }
    
    // Run inference on the perturbed audio
    await runInferenceOnPerturbed(perturbedFileObj);
  };

  // Run inference on perturbed audio
  const runInferenceOnPerturbed = async (perturbedFile: UploadedFile) => {
    if (!model) return;

    setIsLoadingPerturbed(true);
    setError(null);

    try {
      
      let response;
      let prediction;
      
      if (model === "wav2vec2") {
        const requestBody = {
          file_path: perturbedFile.file_path
        };

        response = await fetch(`${API_BASE}/inferences/wav2vec2-detailed`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch perturbed wav2vec2 prediction: ${response.status}`);
        }

        prediction = await response.json();
      } else if (model?.includes("whisper")) {
        const requestBody = {
          model: model,
          file_path: perturbedFile.file_path
        };

        response = await fetch(`${API_BASE}/inferences/whisper-accuracy`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch perturbed whisper prediction: ${response.status}`);
        }

        prediction = await response.json();
      }

      setPerturbedPredictions(prediction);
      
      // Extract prediction text and notify parent component
      let predictionText = "";
      if (model?.includes("whisper")) {
        // For whisper, extract the transcription text
        predictionText = prediction?.transcript || prediction?.prediction || "";
      } else if (model === "wav2vec2") {
        // For wav2vec2, extract the emotion prediction
        predictionText = prediction?.emotion || prediction?.prediction || "";
      }
      
      if (predictionText && onPredictionRefresh) {
        onPredictionRefresh(perturbedFile, predictionText);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setError(errorMessage);
      console.error("Error fetching perturbed prediction:", err);
    } finally {
      setIsLoadingPerturbed(false);
    }
  };

  // Fetch wav2vec prediction when model is wav2vec2 and file is selected
  useEffect(() => {
    let isMounted = true;
    const abortController = new AbortController();
    
    const fetchWav2vecPrediction = async () => {
      if (model !== "wav2vec2" || (!selectedFile && !selectedEmbeddingFile)) {
        if (isMounted) {
          setWav2vecPrediction(null);
          setError(null);
          setIsLoading(false);
        }
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        let requestBody: any = {};
        
        if (selectedFile) {
          // Check if this is an uploaded file - more precise detection
          const isUploadedFile = selectedFile.file_path && (
            selectedFile.file_path.includes('uploads/') || 
            selectedFile.file_path.startsWith('uploads/') ||
            selectedFile.message === "Perturbed file" ||
            selectedFile.message === "File uploaded successfully" ||
            selectedFile.message === "File uploaded and processed successfully"
          ) && selectedFile.message !== "Selected from dataset";
          
          if (isUploadedFile) {
            // This is an uploaded file, use file_path
            requestBody.file_path = selectedFile.file_path;
          } else {
            // This is a dataset file, use originalDataset and dataset_file
            requestBody.dataset = originalDataset || dataset;
            requestBody.dataset_file = selectedFile.filename;
          }
        } else if (selectedEmbeddingFile && dataset) {
          // Use embedding file selection
          requestBody.dataset = originalDataset || dataset;
          requestBody.dataset_file = selectedEmbeddingFile;
        }

        const response = await fetch(`${API_BASE}/inferences/wav2vec2-detailed`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          credentials: 'include',
          body: JSON.stringify(requestBody),
          signal: abortController.signal
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch prediction: ${response.status}`);
        }

        const prediction = await response.json();
        if (isMounted) {
          setWav2vecPrediction(prediction);
        }
        
        // Update predictionMap for uploaded files (same as dataset files)
        if (selectedFile && onPredictionUpdate) {
          const isUploadedFile = selectedFile.file_path && (
            selectedFile.file_path.includes('uploads/') || 
            selectedFile.file_path.startsWith('uploads/') ||
            selectedFile.message === "Perturbed file" ||
            selectedFile.message === "File uploaded successfully" ||
            selectedFile.message === "File uploaded and processed successfully"
          ) && selectedFile.message !== "Selected from embeddings" && selectedFile.message !== "Selected from dataset";
          
          if (isUploadedFile) {
            const predictionText = typeof prediction === 'string' ? prediction : 
              prediction?.predicted_emotion || prediction?.prediction || prediction?.emotion || JSON.stringify(prediction);
            onPredictionUpdate(selectedFile.file_id, predictionText);
          }
        }
      } catch (err) {
        // Ignore abort errors
        if (err.name === 'AbortError') return;
        
        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        if (isMounted) {
          setError(errorMessage);
          console.error("Error fetching wav2vec2 prediction:", err);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchWav2vecPrediction();
    
    return () => {
      isMounted = false;
      abortController.abort();
    };
  }, [selectedFile, selectedEmbeddingFile, model, dataset]);

  // Fetch whisper prediction when model includes whisper and file is selected
  useEffect(() => {
    let isMounted = true;
    
    const fetchWhisperPrediction = async () => {
      if (!model?.includes("whisper") || (!selectedFile && !selectedEmbeddingFile)) {
        if (isMounted) {
          setWhisperPrediction(null);
          setError(null);
          setIsLoading(false);
        }
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        let requestBody: any = {
          model: model
        };
        
        let isUploadedFile = false;
        
        if (selectedFile) {
          // Check if this is an uploaded file - more precise detection
          isUploadedFile = selectedFile.file_path && (
            selectedFile.file_path.includes('uploads/') || 
            selectedFile.file_path.startsWith('uploads/') ||
            selectedFile.message === "Perturbed file" ||
            selectedFile.message === "File uploaded successfully" ||
            selectedFile.message === "File uploaded and processed successfully"
          ) && selectedFile.message !== "Selected from dataset";
          
          if (isUploadedFile) {
            // This is an uploaded file, use file_path
            requestBody.file_path = selectedFile.file_path;
          } else {
            // This is a dataset file, use originalDataset and dataset_file
            requestBody.dataset = originalDataset || dataset;
            requestBody.dataset_file = selectedFile.filename;
          }
        } else if (selectedEmbeddingFile && dataset) {
          // Use embedding file selection - this is a dataset file
          requestBody.dataset = originalDataset || dataset;
          requestBody.dataset_file = selectedEmbeddingFile;
          isUploadedFile = false;
        }

        // Choose the correct endpoint based on file type
        let endpoint: string;
        if (isUploadedFile) {
          // For uploaded files, use basic inference endpoint (no ground truth available)
          endpoint = `${API_BASE}/inferences/run`;
        } else {
          // For dataset files, use accuracy endpoint to get ground truth and metrics
          endpoint = `${API_BASE}/inferences/whisper-accuracy`;
        }

        const response = await fetch(`${API_BASE}/inferences/whisper-accuracy`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          credentials: 'include',
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch whisper prediction: ${response.status}`);
        }

        const prediction = await response.json();
        
        let whisperPrediction: WhisperPrediction;
        
        if (isUploadedFile) {
          // For uploaded files, convert basic prediction to expected format
          whisperPrediction = {
            predicted_transcript: typeof prediction === 'string' ? prediction : prediction?.text || JSON.stringify(prediction),
            ground_truth: "",
            accuracy_percentage: 0,
            word_error_rate: 0,
            character_error_rate: 0,
            levenshtein_distance: 0,
            exact_match: 0,
            character_similarity: 0,
            word_count_predicted: 0,
            word_count_truth: 0
          };
        } else {
          // For dataset files, the accuracy endpoint returns all the metrics
          whisperPrediction = {
            predicted_transcript: prediction.predicted_transcript || "",
            ground_truth: prediction.ground_truth || "",
            accuracy_percentage: prediction.accuracy_percentage || 0,
            word_error_rate: prediction.word_error_rate || 0,
            character_error_rate: prediction.character_error_rate || 0,
            levenshtein_distance: prediction.levenshtein_distance || 0,
            exact_match: prediction.exact_match || 0,
            character_similarity: prediction.character_similarity || 0,
            word_count_predicted: prediction.word_count_predicted || 0,
            word_count_truth: prediction.word_count_truth || 0
          };
        }
        
        setWhisperPrediction(whisperPrediction);
        
        // Update predictionMap for uploaded files (same as dataset files)
        if (selectedFile && onPredictionUpdate && isUploadedFile) {
          onPredictionUpdate(selectedFile.file_id, whisperPrediction.predicted_transcript);
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        setError(errorMessage);
        console.error("Error fetching whisper prediction:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchWhisperPrediction();
  }, [selectedFile, selectedEmbeddingFile, model, dataset, originalDataset]);

  const hasAttention = !!model && model.includes('whisper');

  return (
    <div className="h-full bg-panel-background border-t border-border">
      <Tabs defaultValue="saliency" className="h-full">
        <div className="bg-panel-header border-b border-border px-3 py-2">
          <TabsList className={`h-7 grid w-full ${hasAttention ? 'grid-cols-3' : 'grid-cols-2'} bg-muted`}>
            <TabsTrigger value="saliency" className="text-xs">Saliency</TabsTrigger>
            {hasAttention && <TabsTrigger value="attention" className="text-xs">Attention</TabsTrigger>}
            <TabsTrigger value="perturbation" className="text-xs">Perturbation</TabsTrigger>
          </TabsList>
        </div>

        <div className="h-[calc(100%-2.5rem)] overflow-auto bg-background">
          <TabsContent value="saliency" className="m-0 h-full">
            <div className="p-3">
              <SaliencyVisualization
                selectedFile={selectedFile || selectedEmbeddingFile}
                model={model}
                dataset={dataset}
                originalDataset={originalDataset}
              />
            </div>
          </TabsContent>

          {hasAttention && (
            <TabsContent value="attention" className="m-0 h-full">
              <div className="p-3">
                <AttentionVisualization
                  selectedFile={selectedFile || selectedEmbeddingFile}
                  model={model}
                  dataset={originalDataset && originalDataset !== 'custom' ? originalDataset : dataset}
                />
              </div>
            </TabsContent>
          )}

          <TabsContent value="perturbation" className="m-0 h-full">
            <div className="p-3">
              <PerturbationTools
                selectedFile={selectedFile}
                onPerturbationComplete={handlePerturbationComplete}
                onPredictionRefresh={onPredictionRefresh}
                model={model}
                dataset={dataset}
                originalDataset={originalDataset}
              />
            </div>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
};