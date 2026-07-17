import React, { useState, useEffect, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { useEmbedding } from '@/contexts/EmbeddingContext';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { HelpCircle } from "lucide-react";
import { API_BASE } from '@/lib/api';
import { materializeAudio, runJob } from '@/lib/jobs';
import { getFeatureExplanation } from "@/lib/audioFeatures";

interface ScalersVisualizationProps {
  model?: string;
  dataset?: string;
}

interface Wav2Vec2BatchPrediction {
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

interface WhisperBatchAnalysis {
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

export const ScalersVisualization = ({ model, dataset }: ScalersVisualizationProps) => {
  const { embeddingData, isLoading, error } = useEmbedding();
  const [selectedPoints, setSelectedPoints] = useState<string[]>([]);
  const [batchPrediction, setBatchPrediction] = useState<Wav2Vec2BatchPrediction | null>(null);
  const [whisperAnalysis, setWhisperAnalysis] = useState<WhisperBatchAnalysis | null>(null);
  const [audioFrequencyAnalysis, setAudioFrequencyAnalysis] = useState<AudioFrequencyAnalysis | null>(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [reductionMethod, setReductionMethod] = useState("pca");
  const [selectionMode, setSelectionMode] = useState<"box" | "lasso">("box");
  const [analysisType, setAnalysisType] = useState<"default" | "frequency">("default");

  // Get the 2D coordinates from the embedding data
  const get2DCoordinates = () => {
    if (!embeddingData || !embeddingData.reduced_embeddings) return { x: [], y: [], text: [] };
    
    const coordinates = embeddingData.reduced_embeddings;
    if (!coordinates || coordinates.length === 0) return { x: [], y: [], text: [] };
    
    return {
      x: coordinates.map(point => point.coordinates[0]),
      y: coordinates.map(point => point.coordinates[1]),
      text: coordinates.map(point => point.filename)
    };
  };

  // Handle point selection (box select or lasso)
  const handleSelection = useCallback((event: any) => {
    if (event.points && event.points.length > 0) {
      const selectedFiles = event.points.map((point: any) => point.text);
      setSelectedPoints(selectedFiles);
    }
  }, []);

  // Clear selection
  const clearSelection = () => {
    setSelectedPoints([]);
    setBatchPrediction(null);
    setWhisperAnalysis(null);
    setAudioFrequencyAnalysis(null);
  };

  // Fetch audio frequency analysis for selected points
  const fetchAudioFrequencyAnalysis = async () => {
    if (selectedPoints.length === 0) return;

    setPredictionLoading(true);
    setPredictionError(null);
    try {
      const requestBody: any = {
        filenames: selectedPoints,
        model: model,
      };

      if (dataset) {
        requestBody.dataset = dataset;
      }

      const assets = await Promise.all(selectedPoints.map((filename) => materializeAudio(dataset || '', filename)));
      const analysis = await runJob<AudioFrequencyAnalysis>({
        operation: 'audio_features', audio_ids: assets.map((asset) => asset.audio_id),
      });
      setAudioFrequencyAnalysis(analysis);
    } catch (error) {
      console.error("Error fetching audio frequency analysis:", error);
      setPredictionError(error instanceof Error ? error.message : 'Unknown error occurred');
      setAudioFrequencyAnalysis(null);
    } finally {
      setPredictionLoading(false);
    }
  };

  // Fetch whisper batch analysis for selected points
  const fetchWhisperAnalysis = async () => {
    if (selectedPoints.length === 0 || !model?.includes('whisper')) return;

    setPredictionLoading(true);
    setPredictionError(null);
    try {
      const requestBody: any = {
        filenames: selectedPoints,
        model: model,
      };

      if (dataset) {
        requestBody.dataset = dataset;
      }

      const assets = await Promise.all(selectedPoints.map((filename) => materializeAudio(dataset || '', filename)));
      const analysis = await runJob<WhisperBatchAnalysis>({
        operation: 'prediction', model, audio_ids: assets.map((asset) => asset.audio_id),
      });
      setWhisperAnalysis(analysis);
    } catch (error) {
      console.error("Error fetching whisper analysis:", error);
      setPredictionError(error instanceof Error ? error.message : 'Unknown error occurred');
      setWhisperAnalysis(null);
    } finally {
      setPredictionLoading(false);
    }
  };

  // Fetch batch predictions for selected points
  const fetchBatchPredictions = async () => {
    if (selectedPoints.length === 0 || model !== 'wav2vec2') return;

    setPredictionLoading(true);
    setPredictionError(null);
    try {
      const requestBody: any = {
        filenames: selectedPoints,
      };

      if (dataset) {
        requestBody.dataset = dataset;
      }

      const assets = await Promise.all(selectedPoints.map((filename) => materializeAudio(dataset || '', filename)));
      const prediction = await runJob<Wav2Vec2BatchPrediction>({
        operation: 'prediction', model: 'wav2vec2', audio_ids: assets.map((asset) => asset.audio_id),
      });
      setBatchPrediction(prediction);
    } catch (error) {
      console.error("Error fetching batch predictions:", error);
      setPredictionError(error instanceof Error ? error.message : 'Unknown error occurred');
      setBatchPrediction(null);
    } finally {
      setPredictionLoading(false);
    }
  };

  // Auto-fetch predictions when selection changes
  useEffect(() => {
    if (selectedPoints.length > 0) {
      if (analysisType === "frequency") {
        fetchAudioFrequencyAnalysis();
      } else if (model === 'wav2vec2') {
        fetchBatchPredictions();
      } else if (model?.includes('whisper')) {
        fetchWhisperAnalysis();
      }
    }
  }, [selectedPoints, model, dataset, analysisType]);

  const { x, y, text } = get2DCoordinates();

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="text-sm text-gray-600">Loading embedding data...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center text-red-600">
          <div className="text-sm">Error loading embeddings</div>
          <div className="text-xs mt-1">{error}</div>
        </div>
      </div>
    );
  }

  if (!embeddingData) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center text-gray-600">
          <div className="text-sm">No embedding data available</div>
          <div className="text-xs mt-1">Generate embeddings first in the main panel</div>
        </div>
      </div>
    );
  }

  // Create colors for selected vs unselected points - using gold for selected to match EmbeddingPlot
  const colors = text.map(filename => 
    selectedPoints.includes(filename) ? '#ff0000ff' : '#3b82f6'
  );

  const trace = {
    x,
    y,
    mode: 'markers' as const,
    type: 'scatter' as const,
    text,
    hovertemplate: '%{text}<extra></extra>',
    marker: {
      size: text.map(filename => selectedPoints.includes(filename) ? 10 : 6),
      color: colors,
      opacity: 0.7,
      line: { width: 1, color: 'white' }
    },
    name: 'Audio Files'
  };

  const layout = {
    autosize: true,
    margin: { l: 40, r: 40, t: 40, b: 40 },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    showlegend: false,
    font: { size: 10, color: '#374151' },
    xaxis: { 
      title: 'Component 1',
      gridcolor: '#e5e7eb',
      showgrid: true
    },
    yaxis: { 
      title: 'Component 2',
      gridcolor: '#e5e7eb',
      showgrid: true
    },
    dragmode: selectionMode === 'box' ? 'select' : 'lasso',
    selectdirection: 'any'
  };

  const config = {
    displayModeBar: false, // Hide the mode bar completely
    displaylogo: false,
    responsive: true
  };

  return (
    <TooltipProvider>
      <div className="h-full flex flex-col space-y-3">
      {/* Controls */}
      <div className="flex flex-wrap gap-2 items-center text-xs">
        
        <Select value={selectionMode} onValueChange={(value: "box" | "lasso") => setSelectionMode(value)}>
          <SelectTrigger className="w-24 h-7 text-[11px] rounded-md border border-gray-200">
        <SelectValue />
          </SelectTrigger>
          <SelectContent className="text-[11px]">
        <SelectItem value="box" className="text-[11px]">Box</SelectItem>
        <SelectItem value="lasso" className="text-[11px]">Lasso</SelectItem>
          </SelectContent>
        </Select>

        <Select value={analysisType} onValueChange={(value: "default" | "frequency") => setAnalysisType(value)}>
          <SelectTrigger className="w-36 h-7 text-[11px] rounded-md border border-gray-200">
        <SelectValue />
          </SelectTrigger>
          <SelectContent className="text-[11px]">
        <SelectItem value="default" className="text-[11px]">
          {model === 'wav2vec2' ? 'Predictions' : 'Common Terms'}
        </SelectItem>
        <SelectItem value="frequency" className="text-[11px]">Audio Features</SelectItem>
          </SelectContent>
        </Select>
        <div className="ml-auto">
          <Button
            variant="outline"
            size="sm"
            onClick={clearSelection}
            disabled={selectedPoints.length === 0}
            className="h-6 text-[10px] shadow-none border border-gray-200"
          >
            Clear ({selectedPoints.length})
          </Button>
        </div>
      </div>

      <div className="flex-1 flex gap-3">
        {/* 2D Plot - Fixed Height */}
        <div className="flex-1 border border-gray-200 rounded-lg bg-white" style={{ height: '400px' }}>
          <Plot
            data={[trace]}
            layout={layout}
            config={config}
            style={{ width: '100%', height: '100%' }}
            onSelected={handleSelection}
            onDeselect={() => setSelectedPoints([])}
          />
        </div>

        {/* Analysis Results - Single Consolidated Card */}
        <div className="w-80">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">
                {selectedPoints.length === 0 ? "Analysis Results" : (
                  analysisType === "frequency" ? "Audio Frequency Analysis" : 
                  model === 'wav2vec2' ? "Emotion Predictions" : "Transcript Analysis"
                )} {selectedPoints.length > 0 && `(${selectedPoints.length})`}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {selectedPoints.length === 0 ? (
                <div className="text-xs text-gray-600 text-center">
                  Select points on the plot to see analysis results
                </div>
              ) : predictionLoading ? (
                <div className="text-xs text-gray-600">
                  Loading {analysisType === "frequency" ? "audio features" : 
                          model === 'wav2vec2' ? "predictions" : "analysis"}...
                </div>
              ) : predictionError ? (
                <div className="text-xs text-red-600">
                  <div className="font-medium">Error loading analysis:</div>
                  <div className="mt-1">{predictionError}</div>
                  <div className="mt-2 text-gray-600">
                    Make sure the backend server is running.
                  </div>
                </div>
              ) : analysisType === "frequency" && audioFrequencyAnalysis ? (
                <>
                  {/* Audio Frequency Analysis Results */}
                  {/* Cache Info
                  {audioFrequencyAnalysis.cache_info && (
                    <div className="space-y-2">
                      <div className="text-xs font-medium">Cache Performance</div>
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
                    </div>
                  )} */}

                  {/* Summary */}
                  <div className="space-y-2">
                    <div className="text-sm-tight font-medium">Summary Statistics</div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="text-xs-tight text-gray-500">
                        <span className="text-gray-700 font-medium">Files:</span> {audioFrequencyAnalysis.summary.total_files}
                      </div>
                      <div className="text-xs-tight text-gray-500">
                        <span className="text-gray-700 font-medium">Features:</span> {audioFrequencyAnalysis.summary.total_features_extracted}
                      </div>
                      <div className="text-xs-tight text-gray-500">
                        <span className="text-gray-700 font-medium">Avg Duration:</span> {audioFrequencyAnalysis.summary.avg_duration.toFixed(1)}s
                      </div>
                      <div className="text-xs-tight text-gray-500">
                        <span className="text-gray-700 font-medium">Avg Tempo:</span> {audioFrequencyAnalysis.summary.avg_tempo.toFixed(0)} BPM
                      </div>
                    </div>
                  </div>

                  {/* Most Common Features */}
                  <div className="space-y-2">
                    <div className="text-sm-tight font-medium flex items-center gap-2">
                      Top 5 Most Common Features
                      <Tooltip>
                        <TooltipTrigger>
                          <HelpCircle className="h-3 w-3 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs">
                          Features ranked by prevalence and stability across all audio files
                        </TooltipContent>
                      </Tooltip>
                    </div>
                    <div className="space-y-2">
                      {audioFrequencyAnalysis.most_common_features.slice(0, 5).map((feature, index) => (
                        <div key={index} className="p-2 bg-gray-50 rounded border">
                          <div className="flex justify-between items-start text-xs-tight">
                            <div className="flex items-center gap-2 flex-1">
                              <span className="font-mono text-blue-700 font-medium">
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
                            <span className="text-gray-600">Score: {feature.prevalence_score.toFixed(2)}</span>
                          </div>
                          <div className="text-xs-tight text-gray-500 mt-1">
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

                  {/* Individual Files */}
                  <div className="space-y-2">
                    <div className="text-sm-tight font-medium">Individual Files ({audioFrequencyAnalysis.individual_analyses.length} total)</div>
                    <div className="max-h-32 overflow-y-auto space-y-1">
                      {audioFrequencyAnalysis.individual_analyses.slice(0, 8).map((analysis, index) => (
                        <div key={index} className="text-xs-tight p-2 bg-gray-50 rounded border">
                          <div className="font-mono text-blue-700 truncate text-xs-tight">
                            {analysis.filename}
                          </div>
                          <div className="text-gray-500 mt-1 flex justify-between">
                            <span>Duration: {analysis.features.duration?.toFixed(1)}s</span>
                            <span>Tempo: {analysis.features.tempo?.toFixed(0)} BPM</span>
                          </div>
                        </div>
                      ))}
                      {audioFrequencyAnalysis.individual_analyses.length > 8 && (
                        <div className="text-xs-tight text-gray-500 text-center pt-1">
                          ... and {audioFrequencyAnalysis.individual_analyses.length - 8} more files
                        </div>
                      )}
                    </div>
                  </div>
                </>
              ) : analysisType === "default" && model === 'wav2vec2' && batchPrediction ? (
                <>
                  {/* Wav2Vec2 Emotion Analysis */}
                  {/* Cache Info
                  {batchPrediction.cache_info && (
                    <div className="space-y-2">
                      <div className="text-xs font-medium">Cache Performance</div>
                      <div className="text-xs text-blue-600 bg-blue-50 p-2 rounded">
                        <div className="flex justify-between">
                          <span>Cache hits:</span>
                          <span>{batchPrediction.cache_info.cached_count}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>New predictions:</span>
                          <span>{batchPrediction.cache_info.missing_count}</span>
                        </div>
                        <div className="flex justify-between font-medium">
                          <span>Hit rate:</span>
                          <span>{(batchPrediction.cache_info.cache_hit_rate * 100).toFixed(1)}%</span>
                        </div>
                        {batchPrediction.cache_info.missing_count === 0 && (
                          <div className="text-green-700 mt-1 text-center">
                            ✓ All predictions loaded from cache
                          </div>
                        )}
                      </div>
                    </div>
                  )} */}

                  {/* Summary */}
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Dominant Emotion</div>
                    <div className="flex items-center gap-2">
                      <Badge variant="default" className="text-xs">
                        {batchPrediction.summary.dominant_emotion}
                      </Badge>
                      <span className="text-xs text-gray-600">
                        {(batchPrediction.summary.dominant_percentage * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Emotion Distribution */}
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Emotion Distribution</div>
                    <div className="space-y-1">
                      {Object.entries(batchPrediction.emotion_distribution)
                        .sort(([,a], [,b]) => b - a)
                        .map(([emotion, percentage]) => (
                          <div key={emotion} className="space-y-1">
                            <div className="flex justify-between text-xs">
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

                  {/* Individual Files */}
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Individual Files</div>
                    <div className="max-h-32 overflow-y-auto space-y-1">
                      {batchPrediction.individual_predictions.map((pred, index) => (
                        <div key={index} className="text-xs p-2 bg-gray-50 rounded border">
                          <div className="font-mono text-blue-700 truncate">
                            {pred.filename}
                          </div>
                          <div className="flex items-center gap-2 mt-1">
                            <Badge variant="outline" className="text-xs">
                              {pred.predicted_emotion}
                            </Badge>
                            <span className="text-gray-600">
                              {(pred.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : analysisType === "default" && model?.includes('whisper') && whisperAnalysis ? (
                <>
                  {/* Whisper Transcript Analysis */}
                  {/* Summary */}
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Summary</div>
                    <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                      <div>Total Words: {whisperAnalysis.summary.total_words}</div>
                      <div>Unique Words: {whisperAnalysis.summary.unique_words}</div>
                      <div>Avg/File: {whisperAnalysis.summary.avg_words_per_file.toFixed(1)}</div>
                      <div>Files: {whisperAnalysis.summary.total_files}</div>
                    </div>
                  </div>

                  {/* Top Common Terms */}
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Top 5 Common Terms</div>
                    <div className="space-y-1">
                      {whisperAnalysis.common_terms.slice(0, 5).map((term, index) => (
                        <div key={index} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="font-mono text-blue-700">"{term.term}"</span>
                            <span>{term.percentage.toFixed(1)}% ({term.count}x)</span>
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
                    <div className="text-xs font-medium">Individual Transcripts</div>
                    <div className="max-h-32 overflow-y-auto space-y-1">
                      {whisperAnalysis.individual_transcripts.map((transcript, index) => (
                        <div key={index} className="text-xs p-2 bg-gray-50 rounded border">
                          <div className="font-mono text-blue-700 truncate">
                            {transcript.filename}
                          </div>
                          <div className="text-gray-600 mt-1 text-xs">
                            {transcript.word_count} words
                          </div>
                          <div className="text-gray-800 mt-1 text-xs italic line-clamp-2">
                            "{transcript.transcript}"
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : selectedPoints.length > 0 && !model?.includes('whisper') && model !== 'wav2vec2' ? (
                <div className="text-xs text-gray-600 text-center">
                  {analysisType === "frequency" ? (
                    <div>
                      Audio frequency analysis is available for all models.
                      <div className="mt-2">
                        Selected files: {selectedPoints.length}
                      </div>
                    </div>
                  ) : (
                    <div>
                      Batch analysis is available for wav2vec2 (emotions) and whisper (transcripts) models.
                      <div className="mt-2">
                        Selected files: {selectedPoints.length}
                      </div>
                    </div>
                  )}
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
    </TooltipProvider>
  );
};
