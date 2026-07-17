import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { Toolbar } from "./Toolbar";
import { EmbeddingPanel } from "../panels/EmbeddingPanel";
import { AudioDatasetPanel } from "../panels/AudioDatasetPanel";
import { DatapointEditorPanel } from "../panels/DatapointEditorPanel";
import { PredictionPanel } from "../panels/PredictionPanel";
import { EmbeddingProvider } from "../../contexts/EmbeddingContext";
import React, { useState, useEffect, useCallback, useRef } from "react";
import { AudioReference } from '@/lib/api';
import { firstJobResult, materializeAudio, resolveAudioId, runJob } from '@/lib/jobs';

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

export const MainLayout = () => {
  const [apiData, setApiData] = useState<unknown>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<UploadedFile | null>(null);
  const [model, setModel] = useState("whisper-base");
  const [dataset, setDataset] = useState("common-voice");
  const [batchInferenceStatus, setBatchInferenceStatus] = useState<'idle' | 'running' | 'done'>('idle');
  const [availableFiles, setAvailableFiles] = useState<string[]>([]);
  const [selectedEmbeddingFile, setSelectedEmbeddingFile] = useState<string | null>(null);
  const [perturbationResult, setPerturbationResult] = useState<any>(null);
  
  // Prediction state
  const [wav2vecPrediction, setWav2vecPrediction] = useState<Wav2Vec2Prediction | null>(null);
  const [whisperPrediction, setWhisperPrediction] = useState<WhisperPrediction | null>(null);
  const [isLoadingPredictions, setIsLoadingPredictions] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [perturbedPredictions, setPerturbedPredictions] = useState<Wav2Vec2Prediction | WhisperPrediction | null>(null);
  const [isLoadingPerturbed, setIsLoadingPerturbed] = useState(false);

  // Refs to track ongoing requests and prevent duplicates
  const wav2vecRequestRef = useRef<AbortController | null>(null);
  const whisperRequestRef = useRef<AbortController | null>(null);

  // Clear perturbation result and predictions when selected file changes
  useEffect(() => {
    setPerturbationResult(null);
    setWav2vecPrediction(null);
    setWhisperPrediction(null);
    setPerturbedPredictions(null);
    setPredictionError(null);
  }, [selectedFile, selectedEmbeddingFile]);

  // Fetch perturbed predictions when perturbation result is available
  useEffect(() => {
    const fetchPerturbedPredictions = async () => {
      if (!perturbationResult?.success || !model) {
        setPerturbedPredictions(null);
        return;
      }

      setIsLoadingPerturbed(true);
      setPredictionError(null);

      try {
        if (!perturbationResult.audio_id) throw new Error('Perturbation returned no audio ID');
        const jobResult = await runJob({
          operation: 'prediction',
          model,
          audio_ids: [perturbationResult.audio_id],
        });
        const prediction = firstJobResult(jobResult);
        setPerturbedPredictions(prediction);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        setPredictionError(errorMessage);
        console.error("Error fetching perturbed predictions:", err);
      } finally {
        setIsLoadingPerturbed(false);
      }
    };

    fetchPerturbedPredictions();
  }, [perturbationResult, model]);

  // Fetch wav2vec prediction when model is wav2vec2 and file is selected
  useEffect(() => {
    const fetchWav2vecPrediction = async () => {
      if (model !== "wav2vec2" || (!selectedFile && !selectedEmbeddingFile)) {
        setWav2vecPrediction(null);
        setPredictionError(null);
        setIsLoadingPredictions(false);
        return;
      }

      // Cancel any existing request
      if (wav2vecRequestRef.current) {
        wav2vecRequestRef.current.abort();
      }

      // Create new abort controller
      const abortController = new AbortController();
      wav2vecRequestRef.current = abortController;

      setIsLoadingPredictions(true);
      setPredictionError(null);

      try {
        const audioId = await resolveAudioId(
          selectedFile || selectedEmbeddingFile || '', dataset, abortController.signal,
        );
        const prediction = firstJobResult<Wav2Vec2Prediction>(await runJob({
          operation: 'prediction', model: 'wav2vec2', audio_ids: [audioId],
        }, { signal: abortController.signal }));
        setWav2vecPrediction(prediction);
        
        // Update predictionMap for uploaded files
        if (selectedFile && prediction) {
          const isUploadedFile = Boolean(selectedFile.audio_id);
          
          if (isUploadedFile) {
            const predictionText = typeof prediction === 'string' ? prediction : 
              prediction?.predicted_emotion || prediction?.prediction || prediction?.emotion || JSON.stringify(prediction);
            handlePredictionUpdate(selectedFile.file_id, predictionText);
          }
        }
      } catch (err) {
        // Ignore abort errors
        if (err instanceof DOMException && err.name === 'AbortError') return;
        
        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        setPredictionError(errorMessage);
        console.error("Error fetching wav2vec2 prediction:", err);
      } finally {
        setIsLoadingPredictions(false);
        // Clear the request ref if this is the current request
        if (wav2vecRequestRef.current === abortController) {
          wav2vecRequestRef.current = null;
        }
      }
    };

    fetchWav2vecPrediction();
    
    // Cleanup function
    return () => {
      if (wav2vecRequestRef.current) {
        wav2vecRequestRef.current.abort();
        wav2vecRequestRef.current = null;
      }
    };
  }, [selectedFile, selectedEmbeddingFile, model, dataset]);

  // Fetch whisper prediction when model includes whisper and file is selected
  useEffect(() => {
    const fetchWhisperPrediction = async () => {
      if (!model?.includes("whisper") || (!selectedFile && !selectedEmbeddingFile)) {
        setWhisperPrediction(null);
        setPredictionError(null);
        setIsLoadingPredictions(false);
        return;
      }

      // Cancel any existing request
      if (whisperRequestRef.current) {
        whisperRequestRef.current.abort();
      }

      // Create new abort controller
      const abortController = new AbortController();
      whisperRequestRef.current = abortController;

      setIsLoadingPredictions(true);
      setPredictionError(null);

      try {
        const isUploadedFile = Boolean(selectedFile?.audio_id);
        const isCustomDataset = dataset?.startsWith('custom:');

        const audioId = await resolveAudioId(
          selectedFile || selectedEmbeddingFile || '', dataset, abortController.signal,
        );
        const prediction = firstJobResult<any>(await runJob({
          operation: 'prediction', model, audio_ids: [audioId],
        }, { signal: abortController.signal }));
        
        let whisperPrediction: WhisperPrediction;
        
        if (isUploadedFile || isCustomDataset) {
          // For uploaded files or custom datasets, convert basic prediction to expected format
          whisperPrediction = {
            predicted_transcript: typeof prediction === 'string' ? prediction : prediction?.text || JSON.stringify(prediction),
            ground_truth: "",
            accuracy_percentage: null,
            word_error_rate: null,
            character_error_rate: null,
            levenshtein_distance: null,
            exact_match: null,
            character_similarity: null,
            word_count_predicted: 0,
            word_count_truth: 0
          };
        } else {
          // For regular dataset files, the accuracy endpoint returns all the metrics
          whisperPrediction = {
            predicted_transcript: prediction.predicted_transcript || "",
            ground_truth: prediction.ground_truth || "",
            accuracy_percentage: prediction.accuracy_percentage !== null ? prediction.accuracy_percentage : null,
            word_error_rate: prediction.word_error_rate !== null ? prediction.word_error_rate : null,
            character_error_rate: prediction.character_error_rate !== null ? prediction.character_error_rate : null,
            levenshtein_distance: prediction.levenshtein_distance !== null ? prediction.levenshtein_distance : null,
            exact_match: prediction.exact_match !== null ? prediction.exact_match : null,
            character_similarity: prediction.character_similarity !== null ? prediction.character_similarity : null,
            word_count_predicted: prediction.word_count_predicted || 0,
            word_count_truth: prediction.word_count_truth || 0
          };
        }
        
        setWhisperPrediction(whisperPrediction);
        
        // Update predictionMap for uploaded files and custom datasets
        if (selectedFile && (isUploadedFile || isCustomDataset)) {
          handlePredictionUpdate(selectedFile.file_id, whisperPrediction.predicted_transcript);
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        setPredictionError(errorMessage);
        console.error("Error fetching whisper prediction:", err);
      } finally {
        setIsLoadingPredictions(false);
        // Clear the request ref if this is the current request
        if (whisperRequestRef.current === abortController) {
          whisperRequestRef.current = null;
        }
      }
    };

    fetchWhisperPrediction();
    
    // Cleanup function
    return () => {
      if (whisperRequestRef.current) {
        whisperRequestRef.current.abort();
        whisperRequestRef.current = null;
      }
    };
  }, [selectedFile, selectedEmbeddingFile, model, dataset]);
  
  // Determine effective dataset based on uploaded files and custom datasets
  const effectiveDataset = (() => {
    // If a custom dataset is selected (starts with custom:), use it as-is
    if (dataset.startsWith('custom:')) {
      return dataset;
    }
    // Legacy behavior: if there are user-uploaded files and no custom dataset,
    // show as "custom". Ignore perturbation-produced files here, since they
    // originate from a dataset selection and should not flip the whole UI to
    // a custom-dataset state (which hides ground truth, etc.).
    const nonPerturbedUploads = (uploadedFiles || []).filter(
      (f) => f?.message !== "Perturbed file"
    );
    if (nonPerturbedUploads.length > 0) {
      return "custom";
    }
    // Otherwise use the selected dataset
    return dataset;
  })();

  const [predictionMap, setPredictionMap] = useState<Record<string, string>>({});

  const handlePredictionUpdate = (fileId: string, prediction: string) => {
    setPredictionMap(prev => {
      const updated = { ...prev, [fileId]: prediction };
      return updated;
    });
  };

  const toUploadedFile = (audio: AudioReference): UploadedFile => ({
    audio_id: audio.audio_id,
    file_id: audio.file_id,
    filename: audio.filename,
    playback_url: audio.playback_url,
    message: audio.message || "Audio ready",
    size: audio.size_bytes,
    duration: audio.duration_seconds,
    sample_rate: audio.sample_rate,
  });

  const handleUploadSuccess = (uploadResponse: AudioReference) => {
    const uploadedFile = toUploadedFile(uploadResponse);
    setUploadedFiles(prev => {
      const newFiles = [...prev, uploadedFile];
      return newFiles;
    });
    // Always select the newly uploaded file
    setSelectedFile(uploadedFile);
  };

  const handleFileSelection = (file: UploadedFile) => {
    setSelectedFile(file);
    // Sync embedding selection with audio dataset selection
    setSelectedEmbeddingFile(file.filename);
  };

  const handleEmbeddingSelection = async (filename: string) => {
    setSelectedEmbeddingFile(filename);
    
    // Try to find and select corresponding file in audio dataset
    // First check uploaded files
    const matchingUploadedFile = uploadedFiles.find(f => f.filename === filename);
    if (matchingUploadedFile) {
      setSelectedFile(matchingUploadedFile);
      return;
    }
    
    try {
      const audio = await materializeAudio(dataset, filename);
      setSelectedFile(toUploadedFile(audio));
    } catch (error) {
      console.error('Failed to prepare selected audio:', error);
      setSelectedFile(null);
    }
  };

  const handlePerturbationComplete = (result: any) => {
    setPerturbationResult(result);
    
    // Clear any existing perturbed predictions since we have a new perturbation
    setPerturbedPredictions(null);
  };

  const handlePredictionRefresh = (file: UploadedFile, prediction: string) => {
    if (file.message === "Perturbed file") {
      // Add the perturbed file to uploaded files
      setUploadedFiles(prevFiles => {
        const existingFile = prevFiles.find(f => f.file_id === file.file_id);
        if (existingFile) {
          return prevFiles.map(f => 
            f.file_id === file.file_id 
              ? { ...f, prediction: prediction }
              : f
          );
        } else {
          return [...prevFiles, { ...file, prediction: prediction }];
        }
      });
      
      // Update predictionMap for perturbed file
      setPredictionMap(prev => {
        const updated = { ...prev, [file.filename]: prediction };
        return updated;
      });
    }
    
    // Update selected file if it's the same file
    if (selectedFile && selectedFile.file_id === file.file_id) {
      setSelectedFile(prev => prev ? { ...prev, prediction: prediction } : null);
    }
  };

  const handleBatchInferenceStart = useCallback(() => {
    setBatchInferenceStatus('running');
  }, []);

  const handleBatchInferenceComplete = useCallback(() => {
    setBatchInferenceStatus('done');
  }, []);

  // Clear predictions when model or dataset changes
  useEffect(() => {
    setPredictionMap({});
    setBatchInferenceStatus('idle');
  }, [model, dataset]);

  const handleBatchInference = async (selectedModel: string, selectedDataset: string) => {
    // Don't run batch inference for legacy "custom" (uploaded files only)
    if (selectedDataset === 'custom') return;
    
    // Clear predictions when dataset/model changes to avoid showing old predictions
    setPredictionMap({});
    
    setBatchInferenceStatus('running');
    try {
      // This will be implemented by AudioDatasetPanel to run inference on all files
      // For now, just set the status to indicate batch inference is requested
      setBatchInferenceStatus('done');
    } catch (error) {
      console.error('Batch inference failed:', error);
      setBatchInferenceStatus('idle');
    }
  };
  return (
    <EmbeddingProvider>
      <div className="h-screen flex flex-col bg-background">
        {/* Top Navigation Bar */}
        <Toolbar
          apiData={apiData}
          setApiData={setApiData}
          selectedFile={selectedFile}
          uploadedFiles={uploadedFiles}
          onFileSelect={setSelectedFile}
          model={model}        // current model value
          setModel={setModel}
          dataset={dataset}
          setDataset={setDataset}
          onBatchInference={handleBatchInference}
        />

        {/* Main Content Area */}
        <div className="flex-1 overflow-hidden bg-background">
          <PanelGroup direction="horizontal" className="h-full">
            {/* Left Panel: Embeddings & Scalar Plots */}
            <Panel defaultSize={25} minSize={20}>
              <EmbeddingPanel
                model={model}
                dataset={dataset}
                availableFiles={availableFiles}
                selectedFile={selectedEmbeddingFile}
                onFileSelect={handleEmbeddingSelection}
              />
            </Panel>

            <PanelResizeHandle className="w-1 bg-border hover:bg-primary/20 transition-colors" />
            
            {/* Center Panel: Predictions */}
            <Panel defaultSize={50} minSize={30}>
              <PanelGroup direction="vertical">
                <Panel defaultSize={70} minSize={40}>
                  <PredictionPanel 
                    selectedFile={selectedFile}
                    selectedEmbeddingFile={selectedEmbeddingFile}
                    model={model}
                    dataset={effectiveDataset}
                    originalDataset={dataset}
                    onPerturbationComplete={handlePerturbationComplete}
                    onPredictionRefresh={handlePredictionRefresh}
                    onPredictionUpdate={handlePredictionUpdate}
                  />
                </Panel>
                
                <PanelResizeHandle className="h-1 bg-border hover:bg-primary/20 transition-colors" />

                {/* Bottom Panel: Audio Dataset Table */}
                <Panel defaultSize={30} minSize={20}>
                  <AudioDatasetPanel
                    apiData={apiData}
                    uploadedFiles={uploadedFiles}
                    selectedFile={selectedFile}
                    onFileSelect={handleFileSelection}
                    onUploadSuccess={handleUploadSuccess}
                    model={model}
                    dataset={effectiveDataset}
                    originalDataset={dataset}
                    batchInferenceStatus={batchInferenceStatus}
                    onBatchInferenceStart={handleBatchInferenceStart}
                    onBatchInferenceComplete={handleBatchInferenceComplete}
                    onAvailableFilesChange={setAvailableFiles}
                    onPredictionUpdate={handlePredictionUpdate}
                    predictionMap={predictionMap}
                  />
                </Panel>
              </PanelGroup>
            </Panel>

            <PanelResizeHandle className="w-1 bg-border hover:bg-primary/20 transition-colors" />
            
            {/* Right Panel: Audio Player & Label Editor */}
            <Panel defaultSize={25} minSize={20}>
              <DatapointEditorPanel 
                selectedFile={selectedFile}
                selectedEmbeddingFile={selectedEmbeddingFile}
                dataset={effectiveDataset}
                originalDataset={dataset}
                perturbationResult={perturbationResult}
                predictionMap={predictionMap}
                model={model}
                wav2vecPrediction={wav2vecPrediction}
                whisperPrediction={whisperPrediction}
                perturbedPredictions={perturbedPredictions}
                isLoadingPredictions={isLoadingPredictions}
                isLoadingPerturbed={isLoadingPerturbed}
                predictionError={predictionError}
              />
            </Panel>
          </PanelGroup>
        </div>
      </div>
    </EmbeddingProvider>
  );
};
