import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Eye, Download, Loader2 } from "lucide-react";
import { useState, useEffect, useRef, useMemo } from "react";
import { API_BASE } from '@/lib/api';
import { firstJobResult, resolveAudioId } from '@/lib/jobs';
import { useJob } from '@/hooks/use-job';

interface SaliencySegment {
  start_time: number;
  end_time: number;
  saliency: number;
  intensity: number;
  word?: string;
}

interface SaliencyData {
  model: string;
  method: string;
  segments: SaliencySegment[];
  total_duration: number;
  emotion?: string;
  series?: number[];
}

interface SaliencyVisualizationProps {
  selectedFile?: any;
  model?: string;
  dataset?: string;
  originalDataset?: string;
}

interface StaticWaveformProps {
  audioUrl: string;
  duration: number;
}

const StaticWaveform = ({ audioUrl, duration }: StaticWaveformProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [waveformData, setWaveformData] = useState<number[] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!audioUrl) return;
    
    const generateWaveform = async () => {
      setLoading(true);
      try {
        // Create audio context for waveform analysis
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        
        // Fetch audio data
        const response = await fetch(audioUrl, {
          credentials: 'include',
          mode: 'cors'
        });
        
        if (!response.ok) throw new Error('Failed to fetch audio');
        
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        // Get audio data from the first channel
        const channelData = audioBuffer.getChannelData(0);
        
        // Downsample for visualization (target ~300 points for smooth display)
        const targetSamples = 300;
        const blockSize = Math.floor(channelData.length / targetSamples);
        const waveform: number[] = [];
        
        for (let i = 0; i < targetSamples; i++) {
          const start = i * blockSize;
          const end = Math.min(start + blockSize, channelData.length);
          
          // Calculate RMS for this block
          let sum = 0;
          for (let j = start; j < end; j++) {
            sum += channelData[j] * channelData[j];
          }
          const rms = Math.sqrt(sum / (end - start));
          waveform.push(rms);
        }
        
        setWaveformData(waveform);
        audioContext.close();
      } catch (error) {
        console.warn('Failed to generate waveform:', error);
        // Fallback: create a simple placeholder waveform
        const fallbackWaveform = Array.from({ length: 100 }, (_, i) => 
          Math.sin(i * 0.1) * 0.3 + 0.1 + Math.random() * 0.1
        );
        setWaveformData(fallbackWaveform);
      } finally {
        setLoading(false);
      }
    };
    
    generateWaveform();
  }, [audioUrl]);

  useEffect(() => {
    if (!waveformData || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    const width = rect.width;
    const height = rect.height;
    
    // Clear canvas
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, width, height);
    
    // Draw waveform
    const barWidth = width / waveformData.length;
    const maxAmplitude = Math.max(...waveformData);
    
    ctx.fillStyle = '#6366f1';
    ctx.globalAlpha = 0.7;
    
    waveformData.forEach((amplitude, i) => {
      const normalizedHeight = (amplitude / maxAmplitude) * height * 0.8;
      const x = i * barWidth;
      const y = (height - normalizedHeight) / 2;
      
      ctx.fillRect(x, y, Math.max(1, barWidth - 0.5), normalizedHeight);
    });
    
    ctx.globalAlpha = 1;
  }, [waveformData]);

  return (
    <div className="relative">
      <canvas 
        ref={canvasRef} 
        className="w-full h-12 rounded border border-border/20"
        style={{ width: '100%', height: '48px' }}
      />
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 rounded">
          <div className="text-xs text-muted-foreground">Analyzing audio...</div>
        </div>
      )}
      {!loading && !waveformData && (
        <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
          No waveform data
        </div>
      )}
      <div className="absolute bottom-0 left-1 text-[10px] text-muted-foreground bg-background/60 px-1 rounded">
        {duration.toFixed(1)}s
      </div>
    </div>
  );
};

export const SaliencyVisualization = ({ selectedFile, model, dataset, originalDataset }: SaliencyVisualizationProps) => {
  const [saliencyData, setSaliencyData] = useState<SaliencyData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMethod, setSelectedMethod] = useState("gradcam");
  const [viewMode, setViewMode] = useState<"segments" | "heatmap">("segments");
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const canvasWrapRef = useRef<HTMLDivElement | null>(null);
  const [smoothStrength, setSmoothStrength] = useState(0.5); // 0..1
  const [clipPercent, setClipPercent] = useState(95); // 80..99
  const [hover, setHover] = useState<{ x: number; t: number; v: number } | null>(null);
  const saliencyJob = useJob<any>();
  
  // Generate audio URL for waveform display
  const audioUrl = useMemo(() => {
    if (!selectedFile) return '';
    if (typeof selectedFile === 'object' && selectedFile.playback_url) {
      return `${API_BASE}${selectedFile.playback_url}`;
    }
    if (typeof selectedFile === 'object' && selectedFile.audio_id) {
      return `${API_BASE}/audio/${selectedFile.audio_id}`;
    }
    
    return '';
  }, [selectedFile]);

  const fetchSaliencyData = async () => {
    if (!selectedFile || !model) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Send through the selected model so backend can infer base vs large
      // Determine which dataset to use for request
      // Use original dataset if available and it's a real dataset
      const datasetToUse = originalDataset && originalDataset !== "custom" ? originalDataset : dataset;

      const audioId = await resolveAudioId(selectedFile, datasetToUse);
      const data = firstJobResult<SaliencyData>(await saliencyJob.start({
        operation: 'saliency',
        model,
        audio_ids: [audioId],
        parameters: { method: selectedMethod },
      }));
      setSaliencyData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch saliency data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSaliencyData();
  }, [selectedFile, model, selectedMethod]);

  // Draw heatmap when series updates
  useEffect(() => {
    if (!saliencyData?.series || viewMode !== "heatmap") return;
    const series = saliencyData.series;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = (canvas.parentElement as HTMLElement)?.clientWidth || 600;
    const cssHeight = 64; // px
    canvas.width = Math.max(1, Math.floor(cssWidth * dpr));
    canvas.height = Math.max(1, Math.floor(cssHeight * dpr));
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    // simple moving average smoothing (frontend-only)
    const N = series.length;
    const win = Math.max(1, Math.floor((N / 100) * (0.1 + 0.9 * smoothStrength)));
    const smooth: number[] = new Array(N).fill(0);
    if (win > 1) {
      let acc = 0;
      for (let i = 0; i < N; i++) {
        acc += series[i];
        if (i >= win) acc -= series[i - win];
        smooth[i] = acc / Math.min(win, i + 1);
      }
    } else {
      for (let i = 0; i < N; i++) smooth[i] = series[i];
    }
    // percentile clipping (frontend)
    const sorted = [...smooth].sort((a, b) => a - b);
    const p = Math.max(80, Math.min(99, clipPercent));
    const idx = Math.min(N - 1, Math.max(0, Math.floor((p / 100) * (N - 1))));
    const pVal = sorted[idx] || 1;
    for (let i = 0; i < N; i++) smooth[i] = Math.min(smooth[i], pVal);
    // renormalize to [0,1]
    const mn = Math.min(...smooth);
    const mx = Math.max(...smooth);
    const rng = Math.max(1e-9, mx - mn);
    for (let i = 0; i < N; i++) smooth[i] = (smooth[i] - mn) / rng;
    // draw columns
    const w = cssWidth / N;
    const h = cssHeight;
    for (let i = 0; i < N; i++) {
      const v = Math.max(0, Math.min(1, smooth[i]));
      ctx.fillStyle = intensityToColor(v);
      ctx.globalAlpha = 0.2 + 0.8 * v;
      ctx.fillRect(i * w, 0, Math.max(1, w), h);
    }
    ctx.globalAlpha = 1;
  }, [saliencyData?.series, viewMode, smoothStrength, clipPercent]);

  const intensityToColor = (v: number) => {
    // Interpolate between low (teal), medium (yellow), high (orange-red)
    const clamp = (x: number) => Math.max(0, Math.min(1, x));
    const mix = (a: number, b: number, t: number) => a + (b - a) * t;
    // HSL anchors (from index.css design system):
    // low: 178 68% 78%  | medium: 45 93% 58% | high: 15 86% 58%
    let h: number, s: number, l: number;
    if (v < 0.5) {
      const t = clamp(v / 0.5);
      h = mix(178, 45, t);
      s = mix(68, 93, t);
      l = mix(78, 58, t);
    } else {
      const t = clamp((v - 0.5) / 0.5);
      h = mix(45, 15, t);
      s = mix(93, 86, t);
      l = mix(58, 58, t);
    }
    return `hsl(${h} ${s}% ${l}%)`;
  };

  const generateSaliencyBars = () => {
    if (!saliencyData || !saliencyData.segments || saliencyData.segments.length === 0) return [];
    const total = Math.max(0.001, saliencyData.total_duration || 0);

    return saliencyData.segments.map((segment, i) => {
      const duration = Math.max(0, (segment.end_time - segment.start_time));
      const widthPct = Math.max(0.1, (duration / total) * 100); // ensure minimally visible
      const intensity = typeof segment.intensity === 'number' ? segment.intensity : 0;
      let colorClass = "saliency-low";
      if (intensity > 0.7) colorClass = "saliency-high";
      else if (intensity > 0.4) colorClass = "saliency-medium";

      return (
        <div
          key={i}
          className={`${colorClass} h-full border-r border-background/20`}
          style={{
            opacity: Math.max(0.15, Math.min(1, 0.3 + intensity * 0.7)),
            width: `${widthPct}%`,
            minWidth: 1
          }}
          title={`Time: ${segment.start_time.toFixed(1)}-${segment.end_time.toFixed(1)}s, Intensity: ${(intensity * 100).toFixed(0)}%${segment.word ? `, Word: ${segment.word}` : ''}`}
        />
      );
    });
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">Saliency Overlay</CardTitle>
            <div className="flex items-center gap-2">
              {/* View toggle */}
              <div className="flex rounded border border-border overflow-hidden">
                <button
                  className={`px-2 h-6 text-xs ${viewMode === 'segments' ? 'bg-secondary' : 'bg-background'}`}
                  onClick={() => setViewMode('segments')}
                >Segments</button>
                <button
                  className={`px-2 h-6 text-xs ${viewMode === 'heatmap' ? 'bg-secondary' : 'bg-background'}`}
                  onClick={() => setViewMode('heatmap')}
                >Heatmap</button>
              </div>
              {viewMode === 'heatmap' && (
                <div className="flex items-center gap-2">
                  <label className="flex items-center gap-1 text-[10px] text-muted-foreground">
                    Smooth
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={smoothStrength}
                      onChange={(e) => setSmoothStrength(parseFloat(e.target.value))}
                    />
                  </label>
                  <label className="flex items-center gap-1 text-[10px] text-muted-foreground">
                    Clip {clipPercent}%
                    <input
                      type="range"
                      min={80}
                      max={99}
                      step={1}
                      value={clipPercent}
                      onChange={(e) => setClipPercent(parseInt(e.target.value))}
                    />
                  </label>
                </div>
              )}
              <Select value={selectedMethod} onValueChange={setSelectedMethod}>
                <SelectTrigger className="w-24 h-6 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gradcam">GradCAM</SelectItem>
                  <SelectItem value="lime">LIME</SelectItem>
                  <SelectItem value="shap">SHAP</SelectItem>
                </SelectContent>
              </Select>
              <Button
                size="sm"
                variant="outline"
                className="h-6"
                onClick={loading ? () => void saliencyJob.cancel() : fetchSaliencyData}
              >
                {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Download className="h-3 w-3" />}
                <span className="ml-1">{loading ? 'Cancel' : 'Refresh'}</span>
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {error && (
            <div className="text-xs text-destructive bg-destructive/10 p-2 rounded">
              Error: {error}
            </div>
          )}
          
          {loading && (
            <div className="h-16 bg-muted/30 rounded mb-2 flex items-center justify-center">
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
              <span className="text-xs text-muted-foreground">
                {saliencyJob.status?.progress.message || 'Generating saliency...'}
              </span>
            </div>
          )}
          
          {!loading && !error && saliencyData && (
            <>
              {/* Static Waveform Display */}
              {audioUrl && (
                <div className="mb-3">
                  <div className="text-xs text-muted-foreground mb-2 font-medium">Audio Waveform</div>
                  <StaticWaveform 
                    audioUrl={audioUrl} 
                    duration={saliencyData.total_duration || 0} 
                  />
                </div>
              )}
              
              {/* Saliency overlay */}
              <div className="relative">
                <div className="h-16 bg-muted/30 rounded mb-2"></div>
                
                {/* Saliency overlay */}
                {viewMode === 'heatmap' && saliencyData.series && saliencyData.series.length > 0 ? (
                  <div ref={canvasWrapRef} className="absolute inset-0 z-10 rounded overflow-hidden"
                       onMouseMove={(e) => {
                         if (!canvasWrapRef.current || !saliencyData?.series) return;
                         const rect = canvasWrapRef.current.getBoundingClientRect();
                         const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
                         const N = saliencyData.series.length;
                         const idx = Math.max(0, Math.min(N - 1, Math.floor((x / rect.width) * N)));
                         const t = (saliencyData.total_duration || 0) * (idx / Math.max(1, N - 1));
                         const v = saliencyData.series[idx] || 0;
                         setHover({ x, t, v });
                       }}
                       onMouseLeave={() => setHover(null)}>
                    <canvas ref={canvasRef} className="w-full h-16" />
                    {hover && (
                      <div className="absolute -top-6" style={{ left: Math.max(0, Math.min(hover.x, (canvasWrapRef.current?.clientWidth||0)-40)) }}>
                        <div className="px-1 py-0.5 rounded bg-background/80 text-[10px] shadow border border-border">
                          {hover.t.toFixed(2)}s · {(hover.v * 100).toFixed(0)}%
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="absolute inset-0 z-10 rounded overflow-hidden flex">
                    {saliencyData.segments && saliencyData.segments.length > 0 ? (
                      generateSaliencyBars()
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-xs text-muted-foreground">
                        No saliency segments returned
                      </div>
                    )}
                  </div>
                )}
                {/* Debug info to help verify */}
                <div className="absolute bottom-0 right-1 z-20 text-[10px] text-muted-foreground bg-background/60 rounded px-1">
                  {saliencyData.segments?.length || 0} segs · {saliencyData.total_duration?.toFixed?.(1) || 0}s{saliencyData.series?.length ? ` · ${saliencyData.series.length} pts` : ''}
                </div>
              </div>
            </>
          )}

          {/* Legend + method helper */}
          <div className="flex flex-wrap items-center gap-4 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 saliency-low rounded"></div>
              <span>Low</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 saliency-medium rounded"></div>
              <span>Medium</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 saliency-high rounded"></div>
              <span>High</span>
            </div>
            {viewMode === 'heatmap' && (
              <div className="flex items-center gap-2">
                <div className="w-40 h-2 rounded bg-gradient-to-r from-[hsl(var(--saliency-low))] via-[hsl(var(--saliency-medium))] to-[hsl(var(--saliency-high))]"></div>
                <span className="text-muted-foreground">Heatmap intensity</span>
              </div>
            )}
            <div className="text-muted-foreground">
              {selectedMethod === 'shap' && 'SHAP: additive feature attributions (dense, more stable).'}
              {selectedMethod === 'lime' && 'LIME: local perturbation explanations (can be noisy).'}
              {selectedMethod === 'gradcam' && 'Grad-CAM: gradient-weighted attention (coarser focus).'}
            </div>
          </div>

          {/* Time segments */}
          {!loading && !error && saliencyData && (
            <div className="text-xs space-y-2">
              <div className="font-medium">Top Salient Segments:</div>
              {saliencyData.segments
                .sort((a, b) => b.intensity - a.intensity)
                .slice(0, 5)
                .map((segment, idx) => (
                  <div key={idx} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-[10px]">
                        {segment.start_time.toFixed(1)}-{segment.end_time.toFixed(1)}s
                      </Badge>
                      <span className="font-mono">{segment.word || 'segment'}</span>
                    </div>
                    <span className="text-muted-foreground">{(segment.intensity * 100).toFixed(0)}%</span>
                  </div>
                ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
