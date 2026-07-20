import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { EmbeddingPlot } from "../visualization/EmbeddingPlot";
import { ScalarPlot } from "../visualization/ScalarPlot";
import { DatasetEdaView } from "../eda/DatasetEdaView";
import { useEmbedding } from "../../contexts/EmbeddingContext";
import { RefreshCw, Eye, Box, Square, BarChart3, HelpCircle } from "lucide-react";
import { getFeatureExplanation } from "@/lib/audioFeatures";
import { API_BASE } from "@/lib/api";
import { materializeAudio, runJob } from '@/lib/jobs';

interface EmbeddingPanelProps {
  model?: string;
  dataset?: string;
  availableFiles?: string[];
  selectedFile?: string | null;
  onFileSelect?: (filename: string) => void;
}

// Audio Frequency Analysis interface (reusing from ScalersVisualization)
interface AudioFrequencyAnalysis {
  model_context: string;
  individual_analyses: Array<{
    filename: string;
    features: Record<string, number>;
  }>;
  aggregate_statistics: Record<string, {
    mean: number;
    std: number;
    min: number;
    max: number;
    median: number;
  }>;
  feature_distributions: Record<string, {
    histogram: number[];
    bins: number[];
  }>;
  most_common_features: Array<{
    feature: string;
    normalized_mean: number;
    stability_score: number;
    prevalence_score: number;
    mean: number;
    std: number;
  }>;
  feature_categories: Record<string, string[]>;
  summary: {
    total_files: number;
    total_features_extracted: number;
    avg_duration: number;
    avg_tempo: number;
  };
  cache_info: {
    cached_count: number;
    missing_count: number;
    cache_hit_rate: number;
  };
}

// Batch Prediction Analysis interface (Wav2Vec2)
interface BatchPredictionAnalysis {
  emotion_distribution: Record<string, number>;  // Percentage of files predicted as each emotion
  emotion_counts: Record<string, number>;        // Raw counts for each emotion
  individual_predictions: Array<{
    filename: string;
    predicted_emotion: string;
    probabilities: Record<string, number>;
    confidence: number;
  }>;
  summary: {
    total_files: number;
    dominant_emotion: string;
    dominant_count: number;
    dominant_percentage: number;
  };
  cache_info: {
    cached_count: number;
    missing_count: number;
    cache_hit_rate: number;
  };
}

// Whisper Analysis interface
interface WhisperAnalysis {
  common_terms: Array<{
    term: string;
    count: number;
    percentage: number;
  }>;
  individual_transcripts: Array<{
    filename: string;
    transcript: string;
    word_count: number;
  }>;
  summary: {
    total_files: number;
    total_words: number;
    unique_words: number;
    avg_words_per_file: number;
  };
  cache_info: {
    cached_count: number;
    missing_count: number;
    cache_hit_rate: number;
  };
}

export const EmbeddingPanel = ({ model = "whisper-base", dataset = "common-voice", availableFiles = [], selectedFile, onFileSelect }: EmbeddingPanelProps) => {
  const [reductionMethod, setReductionMethod] = useState("pca");
  const [is3D, setIs3D] = useState(false);
  const [selectionMode, setSelectionMode] = useState<'box' | 'lasso'>('box');
  const [analysisType, setAnalysisType] = useState<'predictions' | 'common-terms' | 'audio-features'>('audio-features');
  const [selectedByAngle, setSelectedByAngle] = useState<string[]>([]);
  const [selectedPoints2D, setSelectedPoints2D] = useState<string[]>([]);
  const [audioFrequencyAnalysis, setAudioFrequencyAnalysis] = useState<AudioFrequencyAnalysis | null>(null);
  const [batchPrediction, setBatchPrediction] = useState<BatchPredictionAnalysis | null>(null);
  const [whisperAnalysis, setWhisperAnalysis] = useState<WhisperAnalysis | null>(null);
  const [isLoadingAnalysis, setIsLoadingAnalysis] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const debounceRef = useRef<NodeJS.Timeout | null>(null);
  const { embeddingData, isLoading, error, fetchEmbeddings, clearEmbeddings } = useEmbedding();

  // Get available analysis types based on model
  const getAvailableAnalysisTypes = () => {
    if (model === 'wav2vec2') {
      return ['predictions', 'audio-features'] as const;
    } else if (model?.includes('whisper')) {
      return ['common-terms', 'audio-features'] as const;
    }
    return ['audio-features'] as const; // Default for other models
  };

  // Update analysis type when model changes to ensure it's valid
  useEffect(() => {
    const availableTypes = getAvailableAnalysisTypes();
    if (!availableTypes.includes(analysisType as any)) {
      // Set to first available type if current type is not valid for this model
      setAnalysisType(availableTypes[0]);
      clearAnalysisResults();
    }
  }, [model]);

  // Auto-fetch embeddings when model, dataset, or reduction method changes
  useEffect(() => {
    if (availableFiles.length > 0 && model && dataset) {
      const filesToProcess = availableFiles;
      const nComponents = is3D ? 3 : 2;
      fetchEmbeddings(model, dataset, filesToProcess, reductionMethod, nComponents);
    }
  }, [model, dataset, availableFiles, reductionMethod, fetchEmbeddings]);

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  const handleFetchEmbeddings = () => {
    if (availableFiles.length > 0) {
      // Use entire dataset for better visualization
      const filesToProcess = availableFiles;
      const nComponents = is3D ? 3 : 2;
      fetchEmbeddings(model, dataset, filesToProcess, reductionMethod, nComponents);
    }
  };

  const handleReductionMethodChange = (method: string) => {
    setReductionMethod(method);
    // The useEffect will handle re-fetching when method changes
  };

  const handle3DToggle = (checked: boolean) => {
    setIs3D(checked);
    // Re-fetch with new dimensionality using entire dataset
    if (embeddingData && availableFiles.length > 0) {
      const filesToProcess = availableFiles;
      const nComponents = checked ? 3 : 2;
      fetchEmbeddings(model, dataset, filesToProcess, reductionMethod, nComponents);
    }
  };

  const handlePointSelect = (filename: string, coordinates: number[]) => {
    if (onFileSelect) {
      onFileSelect(filename);
    }
  };

  const handleAngleRangeSelect = (selectedFiles: string[]) => {
    // Only update if the selection has actually changed
    const currentSelection = selectedByAngle.sort().join(',');
    const newSelection = selectedFiles.sort().join(',');
    
    if (currentSelection !== newSelection) {
      setSelectedByAngle(selectedFiles);
      
      // Clear any existing debounce timer
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      
      // Debounce the analysis fetch to prevent rapid successive calls
      debounceRef.current = setTimeout(() => {
        if (selectedFiles.length > 0) {
          fetchAnalysis(selectedFiles);
        } else {
          clearAnalysisResults();
        }
      }, 300); // 300ms debounce
    }
  };

  const handle2DSelectionChange = (selectedFiles: string[]) => {
    // Only update if the selection has actually changed
    const currentSelection = selectedPoints2D.sort().join(',');
    const newSelection = selectedFiles.sort().join(',');
    
    if (currentSelection !== newSelection) {
      setSelectedPoints2D(selectedFiles);
      
      // Clear any existing debounce timer
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      
      // Debounce the analysis fetch to prevent rapid successive calls
      debounceRef.current = setTimeout(() => {
        if (selectedFiles.length > 0) {
          fetchAnalysis(selectedFiles);
        } else {
          clearAnalysisResults();
        }
      }, 300); // 300ms debounce
    }
  };

  // Clear all analysis results
  const clearAnalysisResults = () => {
    setAudioFrequencyAnalysis(null);
    setBatchPrediction(null);
    setWhisperAnalysis(null);
  };

  // Fetch analysis based on type
  const fetchAnalysis = (filenames: string[]) => {
    switch (analysisType) {
      case 'predictions':
        fetchBatchPredictions(filenames);
        break;
      case 'common-terms':
        fetchWhisperAnalysis(filenames);
        break;
      case 'audio-features':
        fetchFrequencyAnalysis(filenames);
        break;
    }
  };

  // Fetch audio frequency analysis for selected files
  const fetchFrequencyAnalysis = async (filenames: string[]) => {
    if (filenames.length === 0) return;

    setIsLoadingAnalysis(true);
    setAnalysisError(null);
    
    try {
      const requestBody: any = {
        filenames: filenames,
        model: model,
      };

      if (dataset) {
        requestBody.dataset = dataset;
      }

      const assets = await Promise.all(filenames.map((filename) => materializeAudio(dataset, filename)));
      const analysis = await runJob<AudioFrequencyAnalysis>({
        operation: 'audio_features', audio_ids: assets.map((asset) => asset.audio_id),
      });
      setAudioFrequencyAnalysis(analysis);
      setBatchPrediction(null);
      setWhisperAnalysis(null);
    } catch (error) {
      console.error("Error fetching audio frequency analysis:", error);
      setAnalysisError(error instanceof Error ? error.message : 'Unknown error occurred');
      setAudioFrequencyAnalysis(null);
    } finally {
      setIsLoadingAnalysis(false);
    }
  };

  // Fetch batch predictions for selected files
  const fetchBatchPredictions = async (filenames: string[]) => {
    if (filenames.length === 0) return;

    setIsLoadingAnalysis(true);
    setAnalysisError(null);
    
    try {
      const requestBody: any = {
        filenames: filenames,
      };

      if (dataset) {
        requestBody.dataset = dataset;
      }

      const assets = await Promise.all(filenames.map((filename) => materializeAudio(dataset, filename)));
      const prediction = await runJob<BatchPredictionAnalysis>({
        operation: 'prediction', model: 'wav2vec2', audio_ids: assets.map((asset) => asset.audio_id),
      });
      setBatchPrediction(prediction);
      setAudioFrequencyAnalysis(null);
      setWhisperAnalysis(null);
    } catch (error) {
      console.error("Error fetching batch predictions:", error);
      setAnalysisError(error instanceof Error ? error.message : 'Unknown error occurred');
      setBatchPrediction(null);
    } finally {
      setIsLoadingAnalysis(false);
    }
  };

  // Fetch whisper analysis for selected files
  const fetchWhisperAnalysis = async (filenames: string[]) => {
    if (filenames.length === 0) return;

    setIsLoadingAnalysis(true);
    setAnalysisError(null);
    
    try {
      const requestBody: any = {
        filenames: filenames,
        model: model,
      };

      if (dataset) {
        requestBody.dataset = dataset;
      }

      const assets = await Promise.all(filenames.map((filename) => materializeAudio(dataset, filename)));
      const analysis = await runJob<WhisperAnalysis>({
        operation: 'prediction', model, audio_ids: assets.map((asset) => asset.audio_id),
      });
      setWhisperAnalysis(analysis);
      setAudioFrequencyAnalysis(null);
      setBatchPrediction(null);
    } catch (error) {
      console.error("Error fetching whisper analysis:", error);
      setAnalysisError(error instanceof Error ? error.message : 'Unknown error occurred');
      setWhisperAnalysis(null);
    } finally {
      setIsLoadingAnalysis(false);
    }
  };

  return (
    <TooltipProvider>
      <div className="h-full bg-white border-r border-gray-200 flex flex-col">
        <div className="panel-header p-3 border-b border-gray-200">
          <h3 className="font-bold text-sm text-gray-800 flex items-center gap-1.5">
        Audio Embeddings
        <Tooltip>
          <TooltipTrigger>
            <HelpCircle className="h-3.5 w-3.5 text-muted-foreground" />
          </TooltipTrigger>
          <TooltipContent className="font-normal">
            Visualize high-dimensional audio features in 2D/3D space
          </TooltipContent>
        </Tooltip>
          </h3>
        </div>

      <Tabs defaultValue="embeddings" className="flex-1 flex flex-col overflow-hidden">
        <div className="px-3 pt-2 border-b border-gray-200">
          <TabsList className="h-8">
            <TabsTrigger value="embeddings" className="text-xs">Embeddings</TabsTrigger>
            <TabsTrigger value="eda" className="text-xs">Dataset EDA</TabsTrigger>
          </TabsList>
        </div>

      <TabsContent value="embeddings" className="flex-1 p-3 bg-panel-background overflow-auto m-0">
        <div className="space-y-3">
          {/* Controls Section */}
          <div className="flex-shrink-0 space-y-2.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                {embeddingData && (
                  <Badge variant="outline" className="text-[10px] bg-primary/10 text-primary border-primary/20">
                    {embeddingData.model.toUpperCase()}
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-1.5">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div>
                      <Select value={reductionMethod} onValueChange={handleReductionMethodChange}>
                        <SelectTrigger className="w-20 h-7 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="pca">PCA</SelectItem>
                          <SelectItem value="umap">UMAP</SelectItem>
                          <SelectItem value="tsne">t-SNE</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    Choose dimensionality reduction method for visualization
                  </TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={handleFetchEmbeddings}
                      disabled={isLoading || availableFiles.length === 0}
                      className="h-7 w-7 p-0"
                    >
                      <RefreshCw className={`h-3 w-3 ${isLoading ? 'animate-spin text-primary' : ''}`} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    Refresh embeddings visualization
                  </TooltipContent>
                </Tooltip>
              </div>
            </div>
            
            {/* Combined Row: 3D Toggle, Selection Mode (2D only), and Analysis Type */}
            <div className="flex items-center gap-3">
              {/* 3D Toggle */}
              <div className="flex items-center gap-1.5 h-8 px-2 bg-gray-50 rounded-md border border-border">
              <Switch
                id="3d-mode"
                checked={is3D}
                onCheckedChange={handle3DToggle}
                disabled={isLoading}
                className={`relative inline-flex h-6 w-10 shrink-0 cursor-pointer rounded-full transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                is3D ? 'bg-primary/90' : 'bg-muted/40'
                } ${isLoading ? 'opacity-60 pointer-events-none' : ''}`}
              />
              <Label htmlFor="3d-mode" className="text-[11px] flex items-center gap-1 font-medium cursor-pointer">
                {is3D ? <Box className="h-3 w-3 text-primary" /> : <Square className="h-3 w-3 text-muted-foreground" />}
                <span className={is3D ? "text-primary" : "text-muted-foreground"}>
                {is3D ? '3D' : '2D'}
                </span>
              </Label>
              </div>

              {/* Selection Mode (2D only) */}
              {!is3D && (
              <Select value={selectionMode} onValueChange={(value: 'box' | 'lasso') => setSelectionMode(value)}>
                <SelectTrigger className="w-20 h-8 text-xs border border-gray-200 rounded-md">
                <SelectValue />
                </SelectTrigger>
                <SelectContent>
                <SelectItem value="box">Box</SelectItem>
                <SelectItem value="lasso">Lasso</SelectItem>
                </SelectContent>
              </Select>
              )}

              {/* Analysis Type */}
              <Select
              value={analysisType}
              onValueChange={(value: 'predictions' | 'common-terms' | 'audio-features') => {
                setAnalysisType(value);
                // Clear all analysis results and selections when changing analysis type
                clearAnalysisResults();
                setSelectedByAngle([]);
                setSelectedPoints2D([]);
              }}
              >
              <SelectTrigger className="flex-1 h-8 text-xs border border-gray-200 rounded-md">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {model === 'wav2vec2' && (
                <SelectItem value="predictions">Predictions</SelectItem>
                )}
                {model?.includes('whisper') && (
                <SelectItem value="common-terms">Common Terms</SelectItem>
                )}
                <SelectItem value="audio-features">Audio Features</SelectItem>
              </SelectContent>
              </Select>
            </div>

            {/* Status Messages */}
            {availableFiles.length === 0 && (
              <div className="text-xs text-muted-foreground flex items-center gap-2 p-3 bg-muted/50 rounded-md border border-border">
                <div className="w-2 h-2 bg-muted-foreground rounded-full"></div>
                No files available for embedding extraction
              </div>
            )}
            {availableFiles.length > 0 && !embeddingData && !isLoading && (
              <div className="text-xs flex items-center gap-2 p-3 bg-primary/5 rounded-sm border border-primary/20">
                <div className="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
                Click <RefreshCw className="inline h-3 w-3 mx-1" /> to extract embeddings from all {availableFiles.length} files
              </div>
            )}
            {isLoading && (
              <div className="text-xs text-primary flex items-center gap-2 p-3 bg-primary/5 rounded-sm border border-primary/20">
                <div className="w-2 h-2 bg-primary rounded-full animate-ping"></div>
                Processing {availableFiles.length} files... This may take a few moments.
              </div>
            )}
            {error && (
              <div className="text-xs text-destructive flex items-center gap-2 p-3 bg-destructive/5 rounded-sm border border-destructive/20">
                <div className="w-2 h-2 bg-destructive rounded-full"></div>
                {error}
              </div>
            )}
          </div>

          {/* Embedding Plot */}
          <div className="h-[450px] border border-border rounded-lg bg-card p-1.5 overflow-hidden">
            <EmbeddingPlot 
              selectedMethod={reductionMethod} 
              is3D={is3D}
              onPointSelect={handlePointSelect}
              onAngleRangeSelect={handleAngleRangeSelect}
              selectedFile={selectedFile}
              selectionMode={selectionMode}
              onSelectionChange={handle2DSelectionChange}
            />
          </div>

          {/* Analysis Panel - Show when files are selected (2D or 3D) */}
          {(selectedByAngle.length > 0 || selectedPoints2D.length > 0) && (
            <div className="border border-gray-200 rounded-lg bg-white">
              <Tabs defaultValue="analysis" className="w-full">
                <div className="border-b border-gray-200 px-4 py-2 bg-gray-50">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-blue-600" />
                      <span className="text-sm font-medium">
                        {analysisType === 'predictions' ? 'Predictions Analysis' : 
                         analysisType === 'common-terms' ? 'Transcript Analysis' : 
                         'Audio Features Analysis'}
                      </span>
                      <Badge variant="outline" className="text-xs">
                        {is3D ? selectedByAngle.length : selectedPoints2D.length} files
                      </Badge>
                    </div>
                  </div>
                </div>
                
                <TabsContent value="analysis" className="mt-0">
                  <div className="p-4 max-h-96 overflow-y-auto">
                    {isLoadingAnalysis ? (
                      <div className="text-xs-tight text-gray-600 flex items-center gap-2">
                        <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                        Loading {analysisType === 'predictions' ? 'predictions' : analysisType === 'common-terms' ? 'transcripts' : 'audio features'}...
                      </div>
                    ) : analysisError ? (
                      <div className="text-xs-tight text-red-600">
                        <div className="font-medium">Error loading analysis:</div>
                        <div className="mt-1">{analysisError}</div>
                      </div>
                    ) : analysisType === 'audio-features' && audioFrequencyAnalysis ? (
                      <div className="space-y-4">
                        {/* Cache Info
                        {audioFrequencyAnalysis.cache_info && (
                          <div className="text-xs text-blue-600 bg-blue-50 p-2 rounded">
                            <div className="flex justify-between">
                              <span>Cache hits:</span>
                              <span>{audioFrequencyAnalysis.cache_info.cached_count}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>New extractions:</span>
                              <span>{audioFrequencyAnalysis.cache_info.missing_count}</span>
                            </div>
                            <div className="flex justify-between font-medium">
                              <span>Hit rate:</span>
                              <span>{(audioFrequencyAnalysis.cache_info.cache_hit_rate * 100).toFixed(1)}%</span>
                            </div>
                          </div>
                        )} */}

                        {/* Summary */}
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-1">
                            <div className="text-sm-tight font-medium">Summary</div>
                            <div className="space-y-1">
                              <div className="text-xs-tight text-gray-500">
                                <span className="text-gray-700 font-medium">Files:</span> {audioFrequencyAnalysis.summary.total_files}
                              </div>
                              <div className="text-xs-tight text-gray-500">
                                <span className="text-gray-700 font-medium">Features:</span> {audioFrequencyAnalysis.summary.total_features_extracted}
                              </div>
                            </div>
                          </div>
                          <div className="space-y-1">
                            <div className="text-sm-tight font-medium">Audio Metrics</div>
                            <div className="space-y-1">
                              <div className="text-xs-tight text-gray-500">
                                <span className="text-gray-700 font-medium">Avg Duration:</span> {audioFrequencyAnalysis.summary.avg_duration.toFixed(1)}s
                              </div>
                              <div className="text-xs-tight text-gray-500">
                                <span className="text-gray-700 font-medium">Avg Tempo:</span> {audioFrequencyAnalysis.summary.avg_tempo.toFixed(0)} BPM
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Top Features */}
                        <div className="space-y-2">
                          <div className="text-sm-tight font-medium flex items-center gap-2">
                            Top 5 Most Common Features
                            <Tooltip>
                              <TooltipTrigger>
                                <HelpCircle className="h-3 w-3 text-muted-foreground" />
                              </TooltipTrigger>
                              <TooltipContent className="max-w-xs">
                                Features ranked by prevalence and stability across selected audio files
                              </TooltipContent>
                            </Tooltip>
                          </div>
                          <div className="space-y-2">
                            {audioFrequencyAnalysis.most_common_features.slice(0, 5).map((feature, index) => (
                              <div key={index} className="p-2 bg-gray-50 rounded border">
                                <div className="flex justify-between items-start">
                                  <div className="flex items-center gap-2 flex-1">
                                    <span className="font-mono text-blue-700 text-xs-tight font-medium">
                                      {feature.feature.replace(/_/g, ' ').toUpperCase()}
                                    </span>
                                    <Tooltip>
                                      <TooltipTrigger>
                                        <HelpCircle className="h-3 w-3 text-gray-400 hover:text-gray-600" />
                                      </TooltipTrigger>
                                      <TooltipContent className="max-w-sm">
                                        <div className="space-y-1">
                                          <div className="font-medium text-xs">{feature.feature.replace(/_/g, ' ')}</div>
                                          <div className="text-xs">{getFeatureExplanation(feature.feature)}</div>
                                        </div>
                                      </TooltipContent>
                                    </Tooltip>
                                  </div>
                                  <span className="text-xs-tight text-gray-600">Score: {feature.prevalence_score.toFixed(2)}</span>
                                </div>
                                <Progress 
                                  value={feature.prevalence_score * 100} 
                                  className="h-1 my-1"
                                />
                                <div className="text-xs-tight text-gray-500">
                                  Mean: {feature.mean.toFixed(3)} • Std: {feature.std.toFixed(3)} • Stability: {feature.stability_score.toFixed(2)}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Feature Categories */}
                        <div className="space-y-2">
                          <div className="text-sm-tight font-medium flex items-center gap-2">
                            Feature Categories
                            <Tooltip>
                              <TooltipTrigger>
                                <HelpCircle className="h-3 w-3 text-muted-foreground" />
                              </TooltipTrigger>
                              <TooltipContent className="max-w-xs">
                                Audio features grouped by type: spectral (frequency-based), temporal (time-based), and harmonic (pitch-based)
                              </TooltipContent>
                            </Tooltip>
                          </div>
                          <div className="grid grid-cols-2 gap-2">
                            {Object.entries(audioFrequencyAnalysis.feature_categories)
                              .filter(([_, features]) => features.length > 0)
                              .map(([category, features]) => (
                                <div key={category} className="p-2 bg-gray-50 rounded border">
                                  <div className="flex items-center justify-between">
                                    <Badge variant="outline" className="text-xs-tight capitalize">
                                      {category}
                                    </Badge>
                                    <span className="text-xs-tight text-gray-600">{features.length}</span>
                                  </div>
                                  {features.length <= 3 && (
                                    <div className="mt-1 space-y-1">
                                      {features.map((feature, idx) => (
                                        <div key={idx} className="text-xs-tight text-gray-500 truncate">
                                          {feature.replace(/_/g, ' ')}
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              ))}
                          </div>
                        </div>

                        {/* Selected Files Preview */}
                        <div className="space-y-2">
                          <div className="text-sm-tight font-medium">Selected Files ({is3D ? selectedByAngle.length : selectedPoints2D.length} total)</div>
                          <div className="max-h-24 overflow-y-auto space-y-1">
                            {(is3D ? selectedByAngle : selectedPoints2D).slice(0, 5).map((filename, index) => (
                              <div key={index} className="text-xs-tight font-mono text-blue-700 truncate bg-gray-50 px-2 py-1 rounded border">
                                {filename}
                              </div>
                            ))}
                            {(is3D ? selectedByAngle.length : selectedPoints2D.length) > 5 && (
                              <div className="text-xs-tight text-gray-500 text-center">
                                ... and {(is3D ? selectedByAngle.length : selectedPoints2D.length) - 5} more files
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ) : analysisType === 'predictions' && batchPrediction ? (
                      <div className="space-y-4">
                        {/* Wav2Vec2 Emotion Prediction Results */}
                        {/* Dominant Emotion */}
                        <div className="space-y-2">
                          <div className="text-sm-tight font-medium">Dominant Emotion</div>
                          <div className="flex items-center gap-2">
                            <Badge variant="default" className="text-xs-tight capitalize text-white font-normal">
                              {batchPrediction.summary.dominant_emotion}
                            </Badge>
                            <span className="text-xs-tight text-gray-600">
                              {(batchPrediction.summary.dominant_percentage * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>

                        {/* Emotion Distribution */}
                        <div className="space-y-2">
                          <div className="text-sm-tight font-medium">Emotion Distribution</div>
                          <div className="space-y-1">
                            {Object.entries(batchPrediction.emotion_distribution)
                              .sort(([,a], [,b]) => b - a)
                              .map(([emotion, percentage]) => (
                                <div key={emotion} className="space-y-1">
                                  <div className="flex justify-between text-xs-tight">
                                    <span className="capitalize">{emotion}</span>
                                    <span>{(percentage * 100).toFixed(1)}% ({batchPrediction.emotion_counts[emotion]} files)</span>
                                  </div>
                                  <Progress 
                                    value={percentage * 100} 
                                    className="h-1"
                                  />
                                </div>
                              ))}
                          </div>
                        </div>

                        {/* Individual Predictions */}
                        <div className="space-y-2">
                          <div className="text-sm-tight font-medium">Individual Predictions ({batchPrediction.individual_predictions.length} total)</div>
                          <div className="max-h-32 overflow-y-auto space-y-1">
                            {batchPrediction.individual_predictions.slice(0, 8).map((pred, index) => (
                              <div key={index} className="text-xs-tight p-2 bg-gray-50 rounded border">
                                <div className="font-mono text-blue-700 truncate">
                                  {pred.filename}
                                </div>
                                <div className="flex items-center gap-2 mt-1">
                                  <Badge variant="outline" className="text-xs-tight capitalize">
                                    {pred.predicted_emotion}
                                  </Badge>
                                  <span className="text-gray-600">
                                    {(pred.confidence * 100).toFixed(1)}%
                                  </span>
                                </div>
                              </div>
                            ))}
                            {batchPrediction.individual_predictions.length > 8 && (
                              <div className="text-xs-tight text-gray-500 text-center pt-1">
                                ... and {batchPrediction.individual_predictions.length - 8} more predictions
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Selected Files */}
                        <div className="space-y-2">
                          <div className="text-sm-tight font-medium">Selected Files ({is3D ? selectedByAngle.length : selectedPoints2D.length} total)</div>
                          <div className="max-h-24 overflow-y-auto space-y-1">
                            {(is3D ? selectedByAngle : selectedPoints2D).slice(0, 5).map((filename, index) => (
                              <div key={index} className="text-xs-tight font-mono text-blue-700 truncate bg-gray-50 px-2 py-1 rounded border">
                                {filename}
                              </div>
                            ))}
                            {(is3D ? selectedByAngle.length : selectedPoints2D.length) > 5 && (
                              <div className="text-xs-tight text-gray-500 text-center">
                                ... and {(is3D ? selectedByAngle.length : selectedPoints2D.length) - 5} more files
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ) : analysisType === 'common-terms' && whisperAnalysis ? (
                      <div className="space-y-4">
                        {/* Whisper Transcript Analysis */}
                        {/* Summary */}
                        <div className="space-y-2">
                          <div className="text-sm-tight font-medium">Summary</div>
                          <div className="grid grid-cols-2 gap-2">
                            <div className="text-xs-tight text-gray-500">
                              <span className="text-gray-700 font-medium">Total Words:</span> {whisperAnalysis.summary.total_words}
                            </div>
                            <div className="text-xs-tight text-gray-500">
                              <span className="text-gray-700 font-medium">Unique Words:</span> {whisperAnalysis.summary.unique_words}
                            </div>
                            <div className="text-xs-tight text-gray-500">
                              <span className="text-gray-700 font-medium">Avg/File:</span> {whisperAnalysis.summary.avg_words_per_file.toFixed(1)}
                            </div>
                            <div className="text-xs-tight text-gray-500">
                              <span className="text-gray-700 font-medium">Files:</span> {whisperAnalysis.summary.total_files}
                            </div>
                          </div>
                        </div>

                        {/* Top Common Terms */}
                        <div className="space-y-2">
                          <div className="text-sm-tight font-medium">Top 5 Common Terms</div>
                          <div className="space-y-2">
                            {whisperAnalysis.common_terms.slice(0, 5).map((term, index) => (
                              <div key={index} className="space-y-1">
                                <div className="flex justify-between text-xs-tight">
                                  <span className="font-mono text-blue-700">"{term.term}"</span>
                                  <span className="text-gray-600">{term.percentage.toFixed(1)}% ({term.count}x)</span>
                                </div>
                                <Progress 
                                  value={term.percentage} 
                                  className="h-1"
                                />
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Individual Transcripts */}
                        <div className="space-y-2">
                          <div className="text-sm-tight font-medium">Transcripts from Selected Files ({whisperAnalysis.individual_transcripts.length} total)</div>
                          <div className="max-h-48 overflow-y-auto space-y-2">
                            {whisperAnalysis.individual_transcripts.map((transcript, index) => (
                              <div key={index} className="text-xs-tight p-2 bg-gray-50 rounded border">
                                <div className="font-mono text-blue-700 truncate text-xs-tight">
                                  {transcript.filename}
                                </div>
                                <div className="text-gray-600 mt-1 text-xs-tight">
                                  {transcript.word_count} words
                                </div>
                                <div className="text-gray-800 mt-1.5 text-xs-tight leading-relaxed">
                                  "{transcript.transcript}"
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Selected Files Summary */}
                        <div className="space-y-2">
                          <div className="text-sm-tight font-medium">Selection Summary</div>
                          <div className="text-xs-tight text-gray-600 bg-blue-50 p-2 rounded border border-blue-200">
                            {is3D ? selectedByAngle.length : selectedPoints2D.length} files selected
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-xs-tight text-gray-600 text-center">
                        No analysis data available
                      </div>
                    )}
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          )}

        </div>
      </TabsContent>

      <TabsContent value="eda" className="flex-1 p-3 bg-panel-background overflow-auto m-0">
        <DatasetEdaView dataset={dataset} availableFiles={availableFiles} />
      </TabsContent>
      </Tabs>
    </div>
    </TooltipProvider>
  );
};
