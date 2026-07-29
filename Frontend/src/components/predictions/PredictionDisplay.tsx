import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

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

interface PredictionDisplayProps {
  selectedFile?: UploadedFile | null;
  selectedEmbeddingFile?: string | null;
  model?: string;
  wav2vecPrediction?: Wav2Vec2Prediction | null;
  whisperPrediction?: WhisperPrediction | null;
  perturbedPredictions?: Wav2Vec2Prediction | WhisperPrediction | null;
  isLoading?: boolean;
  isLoadingPerturbed?: boolean;
  error?: string | null;
  showPerturbed?: boolean; // New prop to control which predictions to show
}

export const PredictionDisplay = ({
  selectedFile,
  selectedEmbeddingFile,
  model,
  wav2vecPrediction,
  whisperPrediction,
  perturbedPredictions,
  isLoading,
  isLoadingPerturbed,
  error,
  showPerturbed = false
}: PredictionDisplayProps) => {
  if (!selectedFile && !selectedEmbeddingFile) {
    return (
      <Card>
        <CardContent className="p-3 text-center text-muted-foreground">
          <div className="text-xs">No file selected</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="bg-panel-header">
        <CardTitle className="text-xs">
          {model === "wav2vec2" ? "Classification Results" : model?.includes("whisper") ? "Transcription Results" : "Prediction Results"}
          {model === "wav2vec2" && (
            <Badge variant="outline" className="ml-1.5 text-[10px] bg-primary/10 text-primary border-primary/20">
              Wav2Vec2 Emotion
            </Badge>
          )}
          {model?.includes("whisper") && (
            <Badge variant="outline" className="ml-1.5 text-[10px] bg-primary/10 text-primary border-primary/20">
              {model.includes("large") ? "Whisper Large" : "Whisper Base"}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1.5">
        {isLoading && (
          <div className="text-xs text-muted-foreground flex items-center gap-2">
            <div className="w-3 h-3 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
            Loading prediction...
          </div>
        )}

        {error && (
          <div className="text-xs text-destructive p-2 bg-destructive/5 rounded-sm border border-destructive/20">
            Error: {error}
          </div>
        )}
        
        {model === "wav2vec2" && wav2vecPrediction && !isLoading ? (
          // Display wav2vec2 emotion predictions based on tab selection
          <div className="space-y-3">
            {!showPerturbed ? (
              // Original Tab - Show only original predictions
              <div className="space-y-2">
                <div className="text-xs-tight font-medium flex items-center gap-2">
                  Original Audio Prediction
                  <span className="text-xs-tight text-gray-500 border border-gray-300 px-1 rounded">Original</span>
                </div>
                {Object.entries(wav2vecPrediction.probabilities)
                  .sort(([,a], [,b]) => b - a)
                  .map(([emotion, probability]) => {
                    const isPredicted = emotion === wav2vecPrediction.predicted_emotion;
                    return (
                      <div key={emotion} className="flex items-center justify-between text-xs-tight">
                        <div className="flex items-center gap-2">
                          <span className="capitalize">{emotion}</span>
                          {isPredicted && <span className="text-xs-tight text-gray-600 font-medium">Predicted</span>}
                        </div>
                        <div className="flex items-center gap-2 flex-1 max-w-[120px]">
                          <Progress value={probability * 100} className="h-2" />
                          <span className="text-muted-foreground min-w-[2rem]">
                            {(probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
              </div>
            ) : (
              // Perturbed Tab - Show perturbed predictions with comparison
              <div className="space-y-2">
                <div className="text-xs-tight font-medium flex items-center gap-2">
                  Perturbed Audio Prediction
                  <span className="text-xs-tight text-gray-500 border border-gray-300 px-1 rounded">Perturbed</span>
                  {isLoadingPerturbed && (
                    <div className="w-3 h-3 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
                  )}
                </div>
                {!isLoadingPerturbed && perturbedPredictions ? (
                  <div className="space-y-2">
                    {Object.entries((perturbedPredictions as Wav2Vec2Prediction).probabilities)
                      .sort(([,a], [,b]) => b - a)
                      .map(([emotion, probability]) => {
                        const isPredicted = emotion === (perturbedPredictions as Wav2Vec2Prediction).predicted_emotion;
                        const originalProb = wav2vecPrediction.probabilities[emotion] || 0;
                        const change = (probability - originalProb) * 100;
                        const isSignificantChange = Math.abs(change) > 1; // Only highlight changes > 1%
                        return (
                          <div key={emotion} className="flex items-center justify-between text-xs-tight">
                            <div className="flex items-center gap-2">
                              <span className="capitalize">{emotion}</span>
                              {isPredicted && <span className="text-xs-tight text-gray-700 font-medium">Predicted</span>}
                            </div>
                            <div className="flex items-center gap-2 flex-1 max-w-[140px]">
                              <Progress value={probability * 100} className="h-2" />
                              <span className="text-muted-foreground min-w-[2rem]">
                                {(probability * 100).toFixed(1)}%
                              </span>
                              <span className={`text-[10px] min-w-[3rem] font-medium ${
                                !isSignificantChange ? "text-muted-foreground" :
                                change > 0 ? "text-green-600" : change < 0 ? "text-red-600" : "text-muted-foreground"
                              }`}>
                                {change > 0 ? "+" : ""}{change.toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    
                    {/* Show predicted emotion change summary */}
                    <div className="mt-3 p-2 bg-blue-50 rounded border border-blue-200">
                      <div className="text-xs font-medium text-blue-800">Prediction Change</div>
                      <div className="text-xs text-blue-700 mt-1">
                        Original: <span className="font-medium">{wav2vecPrediction.predicted_emotion}</span>
                        {" → "}
                        Perturbed: <span className="font-medium">{(perturbedPredictions as Wav2Vec2Prediction).predicted_emotion}</span>
                        {wav2vecPrediction.predicted_emotion !== (perturbedPredictions as Wav2Vec2Prediction).predicted_emotion && (
                          <span className="text-red-600 font-medium ml-2">Changed!</span>
                        )}
                      </div>
                    </div>
                  </div>
                ) : isLoadingPerturbed ? (
                  <div className="text-xs-tight text-blue-500 flex items-center gap-2">
                    <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                    Loading perturbed prediction...
                  </div>
                ) : (
                  <div className="text-xs-tight text-gray-500 p-2 bg-gray-50 rounded border">
                    No perturbed audio data available. Apply perturbations to compare predictions.
                  </div>
                )}
              </div>
            )}
          </div>
        ) : model?.includes("whisper") && whisperPrediction && !isLoading ? (
          // Display Whisper transcription results based on tab selection
          <div className="space-y-3">
            {!showPerturbed ? (
              // Original Tab - Show only original transcription and metrics
              <div className="space-y-3">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="text-xs font-semibold flex items-center gap-2">
                      Original Transcription Metrics
                      <span className="text-xs-tight text-blue-600 border border-blue-300 px-1 rounded">Original</span>
                    </div>
                    {whisperPrediction.ground_truth && whisperPrediction.ground_truth.trim() !== "" ? (
                      // Show metrics when ground truth is available
                      whisperPrediction.accuracy_percentage !== null && whisperPrediction.word_error_rate !== null ? (
                        <div className="mt-2 grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                          <div className="p-2 bg-gray-50 rounded border text-gray-700">
                            <div className="text-[10px] text-gray-500">WER</div>
                            <div className="font-medium">{whisperPrediction.word_error_rate.toFixed(3)}</div>
                          </div>
                          <div className="p-2 bg-gray-50 rounded border text-gray-700">
                            <div className="text-[10px] text-gray-500">CER</div>
                            <div className="font-medium">{whisperPrediction.character_error_rate.toFixed(3)}</div>
                          </div>
                          <div className="p-2 bg-gray-50 rounded border text-gray-700">
                            <div className="text-[10px] text-gray-500">Accuracy</div>
                            <div className="font-medium">{whisperPrediction.accuracy_percentage.toFixed(1)}%</div>
                          </div>
                          <div className="p-2 bg-gray-50 rounded border text-gray-700">
                            <div className="text-[10px] text-gray-500">Words (Pred)</div>
                            <div className="font-medium">{whisperPrediction.word_count_predicted}</div>
                          </div>
                          <div className="p-2 bg-gray-50 rounded border text-gray-700">
                            <div className="text-[10px] text-gray-500">Words (Truth)</div>
                            <div className="font-medium">{whisperPrediction.word_count_truth}</div>
                          </div>
                          <div className="p-2 bg-gray-50 rounded border text-gray-700">
                            <div className="text-[10px] text-gray-500">Levenshtein</div>
                            <div className="font-medium">{whisperPrediction.levenshtein_distance}</div>
                          </div>
                        </div>
                      ) : (
                        // Ground truth exists but metrics aren't calculated yet
                        <div className="mt-2 p-3 bg-blue-50 rounded border border-blue-200 text-xs text-blue-700">
                          <div className="font-medium">Ground Truth Available</div>
                          <div className="mt-1">Accuracy metrics are being calculated...</div>
                        </div>
                      )
                    ) : (
                      // Show message when ground truth is not available
                      <div className="mt-2 p-3 bg-yellow-50 rounded border border-yellow-200 text-xs text-yellow-700">
                        <div className="font-medium">No Ground Truth Available</div>
                        <div className="mt-1">Accuracy metrics are not available for this dataset-model combination.</div>
                      </div>
                    )}
                  </div>
                </div>

                {whisperPrediction.ground_truth ? (
                  // When ground truth is available, show both in grid layout
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <div className="text-xs font-medium">Predicted Transcript</div>
                      <div className="text-xs p-2 bg-green-50 rounded border font-mono whitespace-pre-wrap">
                        {whisperPrediction.predicted_transcript ? `"${whisperPrediction.predicted_transcript}"` : <span className="italic text-gray-400">No prediction</span>}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-medium">Ground Truth</div>
                      <div className="text-xs p-2 bg-gray-50 rounded border font-mono whitespace-pre-wrap">
                        {`"${whisperPrediction.ground_truth}"`}
                      </div>
                    </div>
                  </div>
                ) : (
                  // When no ground truth is available, show predicted transcript in full width with larger format
                  <div className="w-full">
                    <div className="text-xs font-medium mb-2">Predicted Transcript</div>
                    <div className="text-xs p-4 bg-green-50 rounded-lg border border-green-200 font-mono whitespace-pre-wrap leading-relaxed">
                      {whisperPrediction.predicted_transcript ? `"${whisperPrediction.predicted_transcript}"` : <span className="italic text-gray-400">No prediction available</span>}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              // Perturbed Tab - Show perturbed transcription with comparison
              <div className="space-y-3">
                <div className="text-xs font-semibold flex items-center gap-2">
                  Perturbed Transcription Results
                  <span className="text-xs-tight text-blue-600 border border-blue-300 px-1 rounded">Perturbed</span>
                  {isLoadingPerturbed && (
                    <div className="w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                  )}
                </div>
                
                {!isLoadingPerturbed && perturbedPredictions ? (
                  <div className="space-y-4">
                    {/* Perturbed metrics if available */}
                    {typeof perturbedPredictions === 'object' && (perturbedPredictions as WhisperPrediction).word_error_rate !== null && (
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                        <div className="p-2 bg-blue-50 rounded border text-gray-700">
                          <div className="text-[10px] text-gray-500">WER</div>
                          <div className="font-medium">{(perturbedPredictions as WhisperPrediction).word_error_rate?.toFixed(3) || 'N/A'}</div>
                        </div>
                        <div className="p-2 bg-blue-50 rounded border text-gray-700">
                          <div className="text-[10px] text-gray-500">CER</div>
                          <div className="font-medium">{(perturbedPredictions as WhisperPrediction).character_error_rate?.toFixed(3) || 'N/A'}</div>
                        </div>
                        <div className="p-2 bg-blue-50 rounded border text-gray-700">
                          <div className="text-[10px] text-gray-500">Accuracy</div>
                          <div className="font-medium">{(perturbedPredictions as WhisperPrediction).accuracy_percentage?.toFixed(1) || 'N/A'}%</div>
                        </div>
                        <div className="p-2 bg-blue-50 rounded border text-gray-700">
                          <div className="text-[10px] text-gray-500">Words (P)</div>
                          <div className="font-medium">{(perturbedPredictions as WhisperPrediction).word_count_predicted || 'N/A'}</div>
                        </div>
                        <div className="p-2 bg-blue-50 rounded border text-gray-700">
                          <div className="text-[10px] text-gray-500">Words (T)</div>
                          <div className="font-medium">{(perturbedPredictions as WhisperPrediction).word_count_truth || 'N/A'}</div>
                        </div>
                        <div className="p-2 bg-blue-50 rounded border text-gray-700">
                          <div className="text-[10px] text-gray-500">Levenshtein</div>
                          <div className="font-medium">{(perturbedPredictions as WhisperPrediction).levenshtein_distance || 'N/A'}</div>
                        </div>
                      </div>
                    )}

                    {/* Show perturbed transcript */}
                    <div className="w-full">
                      <div className="text-xs font-medium mb-2">Perturbed Transcript</div>
                      <div className="text-xs p-4 bg-blue-50 rounded-lg border border-blue-200 font-mono whitespace-pre-wrap leading-relaxed">
                        {(perturbedPredictions as WhisperPrediction).predicted_transcript ? `"${(perturbedPredictions as WhisperPrediction).predicted_transcript}"` : <span className="italic text-gray-400">No prediction available</span>}
                      </div>
                    </div>

                    {/* Comparison summary if both predictions are available */}
                    {whisperPrediction && (perturbedPredictions as WhisperPrediction).predicted_transcript && (
                      <div className="mt-3 p-3 bg-blue-50 rounded border border-blue-200">
                        <div className="text-xs font-medium text-blue-800 mb-2">Transcription Comparison</div>
                        <div className="space-y-2 text-xs text-blue-700">
                          <div>
                            <span className="font-medium">Original:</span> 
                            <span className="ml-2 font-mono">"{whisperPrediction.predicted_transcript || 'N/A'}"</span>
                          </div>
                          <div>
                            <span className="font-medium">Perturbed:</span> 
                            <span className="ml-2 font-mono">"{(perturbedPredictions as WhisperPrediction).predicted_transcript || 'N/A'}"</span>
                          </div>
                          {whisperPrediction.predicted_transcript !== (perturbedPredictions as WhisperPrediction).predicted_transcript && (
                            <div className="text-red-600 font-medium">
                              ⚠ Transcription changed due to perturbation!
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ) : isLoadingPerturbed ? (
                  <div className="text-xs-tight text-blue-500 flex items-center gap-2">
                    <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                    Loading perturbed transcription...
                  </div>
                ) : (
                  <div className="text-xs-tight text-gray-500 p-2 bg-gray-50 rounded border">
                    No perturbed audio data available. Apply perturbations to compare transcriptions.
                  </div>
                )}
              </div>
            )}
          </div>
        ) : !model?.includes("whisper") && model !== "wav2vec2" ? (
          // Display placeholder/mock data for other models
          [
            { label: "Neutral", probability: 0.87, isTrue: true, isPredicted: true },
            { label: "Happy", probability: 0.08, isTrue: false, isPredicted: false },
            { label: "Sad", probability: 0.03, isTrue: false, isPredicted: false },
            { label: "Angry", probability: 0.02, isTrue: false, isPredicted: false },
          ].map((item, idx) => (
            <div key={idx} className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                <span>{item.label}</span>
                {item.isPredicted && <Badge variant="default" className="text-[10px] px-1">P</Badge>}
                {item.isTrue && <Badge variant="outline" className="text-[10px] px-1">T</Badge>}
              </div>
              <div className="flex items-center gap-2 flex-1 max-w-[120px]">
                <Progress value={item.probability * 100} className="h-2" />
                <span className="text-muted-foreground min-w-[2rem]">
                  {(item.probability * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          ))
        ) : null}
      </CardContent>
    </Card>
  );
};
