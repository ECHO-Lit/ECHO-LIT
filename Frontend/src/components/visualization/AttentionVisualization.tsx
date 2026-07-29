import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useState, useEffect } from "react";
import { firstJobResult, resolveAudioId } from '@/lib/jobs';
import { useJob } from '@/hooks/use-job';

interface AttentionPair {
  from_word: string;
  to_word: string;
  from_time: [number, number];
  to_time: [number, number];
  attention_weight: number;
  from_index: number;
  to_index: number;
}

interface TimestampAttention {
  time: number;
  attention: number;
  max_outgoing?: number;
  avg_incoming?: number;
  self_attention?: number;
  attention_entropy?: number;
  frame_index?: number;
}

interface AttentionVisualizationProps {
  selectedFile?: any;
  model?: string;
  dataset?: string;
}

export const AttentionVisualization = ({ selectedFile, model, dataset }: AttentionVisualizationProps) => {
  const [selectedLayer, setSelectedLayer] = useState(6);  // Middle layer for better semantic attention patterns
  const [selectedHead, setSelectedHead] = useState(0);
  const [attentionData, setAttentionData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const attentionJob = useJob<any>();

  const fetchAttentionData = async () => {
    console.log("AttentionVisualization - fetchAttentionData called:", {
      selectedFile,
      model,
      dataset,
      hasWhisper: model?.includes('whisper')
    });

    if (!selectedFile || !model || !model.includes('whisper')) {
      console.log("AttentionVisualization - Skipping fetch due to conditions");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const audioId = await resolveAudioId(selectedFile, dataset);
      const data = firstJobResult(await attentionJob.start({
        operation: 'attention',
        model,
        audio_ids: [audioId],
        parameters: { layer_idx: selectedLayer, head_idx: selectedHead },
      }));
      setAttentionData(data);

    } catch (err: any) {
      console.error("AttentionVisualization - Error:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAttentionData();
  }, [selectedFile, model, dataset, selectedLayer, selectedHead]);

  const renderWordPairsMatrix = () => {
    if (!attentionData?.attention_pairs) return null;

    const pairs = attentionData.attention_pairs as AttentionPair[];
    const words = [...new Set(pairs.map(p => p.from_word))];
    
    // Create attention matrix based on actual sequence length, not just unique words
    const maxIndex = Math.max(
      ...pairs.map(p => Math.max(p.from_index, p.to_index)),
      words.length - 1
    );
    const matrixSize = maxIndex + 1;
    
    // Create attention matrix with proper size
    const matrix: number[][] = Array(matrixSize).fill(null).map(() => Array(matrixSize).fill(0));
    
    // Safely populate matrix with bounds checking
    pairs.forEach(pair => {
      const fromIdx = pair.from_index;
      const toIdx = pair.to_index;
      
      // Bounds check to prevent array access errors
      if (fromIdx >= 0 && fromIdx < matrixSize && toIdx >= 0 && toIdx < matrixSize) {
        matrix[fromIdx][toIdx] = pair.attention_weight;
      } else {
        console.warn(`Index out of bounds: from=${fromIdx}, to=${toIdx}, matrixSize=${matrixSize}`);
      }
    });

    // Calculate color scale excluding self-attention for better contrast
    const nonSelfAttentionValues = pairs
      .filter(p => p.from_index !== p.to_index) // Exclude self-attention
      .map(p => p.attention_weight)
      .filter(val => val > 0);
    
    const minNonSelfAttention = Math.min(...nonSelfAttentionValues, 0);
    const maxNonSelfAttention = Math.max(...nonSelfAttentionValues, 1);
    
    // Function to normalize attention values for coloring
    const normalizeAttention = (value: number, isSelfAttention: boolean) => {
      if (isSelfAttention) {
        // Self-attention gets a distinct color (gold/yellow)
        return { intensity: 0.8, isSelf: true };
      } else {
        // Non-self attention uses relative scale
        const normalized = (value - minNonSelfAttention) / (maxNonSelfAttention - minNonSelfAttention);
        return { intensity: Math.max(normalized, 0.1), isSelf: false };
      }
    };

    // Get top pairs for display (exclude self-attention pairs)
    const topPairs = pairs
      .filter(pair => pair.from_index !== pair.to_index) // Exclude self-attention
      .sort((a, b) => b.attention_weight - a.attention_weight)
      .slice(0, 10);

    return (
      <div className="space-y-4">
        {/* Attention Matrix */}
        <Card>
          <CardHeader>
            <CardTitle className="text-xs">Word-to-Word Attention Matrix</CardTitle>
          </CardHeader>
          <CardContent>
            {/* Limit matrix display size for performance and readability */}
            {words.length > 50 ? (
              <div className="text-xs text-muted-foreground p-4">
                Attention matrix too large to display ({words.length}x{words.length}). 
                Showing top attention pairs below instead.
              </div>
            ) : (
              <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${words.length + 1}, minmax(0, 1fr))` }}>
                <div></div>
                {words.map((word, i) => (
                  <div key={i} className="text-[9px] p-1 text-center font-medium truncate" title={word}>
                    {word.length > 6 ? word.substring(0, 6) + '...' : word}
                  </div>
                ))}
                
                {words.map((fromWord, i) => (
                <>
                  <div key={`row-${i}`} className="text-[9px] p-1 font-medium truncate" title={fromWord}>
                    {fromWord.length > 8 ? fromWord.substring(0, 8) + '...' : fromWord}
                  </div>
                  {words.map((toWord, j) => {
                    // Safe matrix access with bounds checking
                    const attentionValue = (i < matrix.length && j < matrix[0]?.length) ? matrix[i][j] : 0;
                    const isSelfAttention = i === j;
                    
                    if (isSelfAttention) {
                      // Block out diagonal cells (self-attention)
                      return (
                        <div
                          key={`cell-${i}-${j}`}
                          className="aspect-square border border-gray-300 flex items-center justify-center"
                          style={{
                            backgroundColor: 'rgba(0, 0, 0, 0.1)', // Light black/gray
                            minWidth: '20px',
                            minHeight: '20px'
                          }}
                          title="Self-attention (blocked)"
                        >
                          {/* No value displayed */}
                        </div>
                      );
                    }
                    
                    // For non-diagonal cells, use relative coloring
                    const colorInfo = normalizeAttention(attentionValue, false);
                    
                    return (
                      <div
                        key={`cell-${i}-${j}`}
                        className="aspect-square border border-gray-200 text-[9px] flex items-center justify-center cursor-pointer hover:border-gray-400 transition-all"
                        style={{
                          backgroundColor: `rgba(59, 130, 246, ${colorInfo.intensity})`,
                          color: colorInfo.intensity > 0.6 ? 'white' : 'black',
                          minWidth: '20px',
                          minHeight: '20px'
                        }}
                        title={`${fromWord} → ${toWord}: ${(attentionValue * 100).toFixed(2)}%`}
                      >
                        {attentionValue > 0.01 ? (attentionValue * 100).toFixed(0) : '0'}
                      </div>
                    );
                  })}
                </>
              ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Top Attention Pairs */}
        <Card>
          <CardHeader>
            <CardTitle className="text-xs">Strongest Attention Relationships</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {topPairs.map((pair, i) => (
                <div key={i} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline">{pair.from_word}</Badge>
                    <span className="text-muted-foreground">→</span>
                    <Badge variant="outline">{pair.to_word}</Badge>
                  </div>
                  <div className="text-xs font-medium">
                    {(pair.attention_weight * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderTimelineView = () => {
    if (!attentionData?.timestamp_attention) return null;

    const timestamps = attentionData.timestamp_attention as TimestampAttention[];
    if (timestamps.length === 0) return null;

    try {
      const attentionValues = timestamps.map(t => t.attention || 0);
      
      // Check for valid data
      if (attentionValues.length === 0 || attentionValues.some(val => isNaN(val))) {
        return <div className="text-red-500 p-4">Invalid attention data</div>;
      }
      
      const minVal = Math.min(...attentionValues);
      const maxVal = Math.max(...attentionValues);
      const range = maxVal - minVal;
      
      // Always use full 0-1 scale for maximum visual clarity
      const normalizedValues = attentionValues.map(val => (val - minVal) / (range || 1));
    
    // Responsive width based on audio duration and number of points
    const duration = attentionData.total_duration || timestamps[timestamps.length - 1]?.time || 10;
    const minWidth = 800;
    const maxWidth = 1200;
    const width = Math.min(maxWidth, Math.max(minWidth, duration * 60)); // 60px per second, with limits
    const height = 250;
    const padding = 80; // Increased padding for better label spacing

    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-xs">Attention Timeline</CardTitle>
          <p className="text-xs text-muted-foreground">
            Showing attention values over time - scaled to highlight differences
          </p>
        </CardHeader>
        <CardContent>
          <div className="flex justify-center overflow-x-auto">
            <svg width={width} height={height} className="border rounded bg-white" style={{minWidth: `${width}px`}}>
              {/* Grid lines for reference */}
              {[0, 0.2, 0.4, 0.6, 0.8, 1].map(ratio => {
                const y = padding + (1 - ratio) * (height - 2 * padding);
                const actualValue = minVal + ratio * range;
                return (
                  <g key={`grid-${ratio}`}>
                    <line
                      x1={padding}
                      y1={y}
                      x2={width - padding}
                      y2={y}
                      stroke={ratio === 0 || ratio === 1 ? "#666" : "#e5e5e5"}
                      strokeWidth={ratio === 0 || ratio === 1 ? "2" : "1"}
                      strokeDasharray={ratio === 0 || ratio === 1 ? "none" : "2,2"}
                    />
                    <text x={padding - 15} y={y + 4} textAnchor="end" className="text-xs fill-gray-600">
                      {(actualValue * 100).toFixed(2)}%
                    </text>
                  </g>
                );
              })}
              
              {/* Smooth attention curve */}
              <polyline
                points={normalizedValues.map((val, i) => {
                  const x = padding + (i / (normalizedValues.length - 1)) * (width - 2 * padding);
                  const y = padding + (1 - val) * (height - 2 * padding);
                  return `${x},${y}`;
                }).join(' ')}
                fill="none"
                stroke="#2563eb"
                strokeWidth="2"
              />
              
              {/* Data points */}
              {normalizedValues.map((val, i) => {
                const x = padding + (i / (normalizedValues.length - 1)) * (width - 2 * padding);
                const y = padding + (1 - val) * (height - 2 * padding);
                const originalVal = attentionValues[i];
                
                return (
                  <circle
                    key={`point-${i}`}
                    cx={x}
                    cy={y}
                    r="3"
                    fill="#2563eb"
                    stroke="white"
                    strokeWidth="1"
                    className="hover:r-4 transition-all"
                  >
                    <title>
                      Time: {timestamps[i].time.toFixed(2)}s
                      Attention: {(originalVal * 100).toFixed(4)}%
                    </title>
                  </circle>
                );
              })}
              
              {/* Time axis - adaptive markers based on duration */}
              {(() => {
                const numMarkers = duration > 10 ? 6 : 5;
                const markers = Array.from({ length: numMarkers }, (_, i) => i / (numMarkers - 1));
                
                return markers.map(ratio => {
                  const x = padding + ratio * (width - 2 * padding);
                  const timeIndex = Math.floor(ratio * (timestamps.length - 1));
                  const time = timestamps[timeIndex]?.time || 0;
                  return (
                    <g key={`time-${ratio}`}>
                      <line x1={x} y1={height - padding} x2={x} y2={height - padding + 5} stroke="#666" />
                      <text x={x} y={height - padding + 18} textAnchor="middle" className="text-xs fill-gray-600">
                        {time.toFixed(1)}s
                      </text>
                    </g>
                  );
                });
              })()}
              
              {/* Axis labels */}
              <text x={width / 2} y={height - 15} textAnchor="middle" className="text-sm font-medium fill-gray-700">
                Time (seconds)
              </text>
              <text x={20} y={height / 2} textAnchor="middle" className="text-sm font-medium fill-gray-700" transform={`rotate(-90, 20, ${height / 2})`}>
                Attention Values
              </text>
            </svg>
          </div>
          
          {/* Simple stats */}
          <div className="mt-4 flex justify-center gap-6 text-xs flex-wrap">
            <div><span className="font-medium">Duration:</span> {duration.toFixed(1)}s</div>
            <div><span className="font-medium">Min:</span> {(minVal * 100).toFixed(4)}%</div>
            <div><span className="font-medium">Max:</span> {(maxVal * 100).toFixed(4)}%</div>
            <div><span className="font-medium">Range:</span> {(range * 100).toFixed(4)}%</div>
            <div><span className="font-medium">Points:</span> {attentionValues.length}</div>
            <div><span className="font-medium">Resolution:</span> {(duration / attentionValues.length * 1000).toFixed(0)}ms</div>
          </div>
          
          <div className="mt-2 text-xs text-center text-gray-500">
            Hover over points for exact values • Scale adjusted to show all differences
          </div>

          <div className="mt-3 text-xs text-muted-foreground space-y-1">
            <div>
              Duration: {attentionData.total_duration?.toFixed(1)}s | 
              Points: {timestamps.length} | 
              Resolution: {((attentionData.total_duration || 0) / timestamps.length * 1000).toFixed(0)}ms
            </div>
          </div>
        </CardContent>
      </Card>
    );
    } catch (error) {
      console.error('Error rendering timeline:', error);
      return (
        <Card>
          <CardHeader>
            <CardTitle>Attention Timeline</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-red-500 p-4">
              Error displaying attention timeline. Please try again.
            </div>
          </CardContent>
        </Card>
      );
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-xs">Attention Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center gap-3 py-8">
            <div className="flex items-center space-x-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
              <span className="text-xs text-muted-foreground">
                {attentionJob.status?.progress.message || 'Extracting attention patterns...'}
              </span>
            </div>
            <Button size="sm" variant="outline" onClick={() => void attentionJob.cancel()}>Cancel</Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-xs">Attention Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <div className="text-red-500 text-xs">{error}</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!selectedFile || !model?.includes('whisper')) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-xs">Attention Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-xs text-muted-foreground py-8">
            Select a Whisper model and audio file to analyze attention patterns
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="text-xs">Attention Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-medium">Layer</label>
              <Select value={selectedLayer.toString()} onValueChange={(v) => setSelectedLayer(parseInt(v))}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Array.from({ length: 12 }, (_, i) => (
                <SelectItem key={i} value={i.toString()}>Layer {i}</SelectItem>
                ))}
              </SelectContent>
              </Select>
            </div>
            
            <div>
              <label className="text-xs font-medium">Head</label>
              <Select value={selectedHead.toString()} onValueChange={(v) => setSelectedHead(parseInt(v))}>
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({ length: 12 }, (_, i) => (
                    <SelectItem key={i} value={i.toString()}>Head {i}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {attentionData && (
            <div className="text-xs text-muted-foreground">
              Model: {attentionData.model} | Words: {attentionData.word_chunks?.length || 0} | 
              Sequence Length: {attentionData.sequence_length}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Attention Views */}
      <Tabs defaultValue="pairs" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="pairs" className="text-xs">Word Pairs</TabsTrigger>
          <TabsTrigger value="timeline" className="text-xs">Timeline</TabsTrigger>
        </TabsList>
        
        <TabsContent value="pairs" className="mt-4">
          {renderWordPairsMatrix()}
        </TabsContent>
        
        <TabsContent value="timeline" className="mt-4">
          {renderTimelineView()}
        </TabsContent>
      </Tabs>
    </div>
  );
};
