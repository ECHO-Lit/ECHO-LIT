/**
 * Encoder structural analysis view: attention distance profiles ("routing")
 * and layer CKA / participation ratio ("geometry").
 *
 * Runs the `encoder_analysis` operation on one file. Label-free, single forward
 * pass on the backend -- no decoder, no transcript. Encoder attention runs over
 * mel frames only, so unlike the decoder attention view there are no special
 * tokens or sinks here.
 */

import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useJob } from "@/hooks/use-job";
import { useIsDarkMode } from "@/hooks/useIsDarkMode";
import { clusterColor } from "@/lib/clusterPalette";
import { firstJobResult, resolveAudioId } from "@/lib/jobs";

interface EncoderAnalysisResult {
  model: string;
  encoder_positions: number;
  position_step_ms: number;
  audio_seconds_analyzed: number;
  attention_profiles: {
    n_layers: number;
    n_heads: number;
    bin_edges_ms: number[];
    layers: Array<{
      head_profiles: number[][];
      mean_profile: number[];
      diagonal_mass: number[];
      profile_entropy: number[];
    }>;
  };
  cka: {
    layer_names: string[];
    matrix: number[][];
    adjacent_cka: number[];
    participation_ratio: number[];
  };
}

interface EncoderAnalysisVisualizationProps {
  selectedFile?: any;
  model?: string;
  dataset?: string;
}

export const EncoderAnalysisVisualization = ({ selectedFile, model, dataset }: EncoderAnalysisVisualizationProps) => {
  const [data, setData] = useState<EncoderAnalysisResult | null>(null);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const job = useJob<EncoderAnalysisResult>();
  const isDark = useIsDarkMode();
  const isWhisper = Boolean(model?.includes("whisper"));

  useEffect(() => {
    if (!selectedFile || !isWhisper) {
      setData(null);
      setError(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const audioId = await resolveAudioId(selectedFile, dataset);
        const result = firstJobResult<EncoderAnalysisResult>(
          await job.start({ operation: "encoder_analysis", model, audio_ids: [audioId] }),
        );
        if (!cancelled) setData(result);
      } catch (err: any) {
        if (!cancelled && err?.name !== "AbortError") setError(err?.message || "Analysis failed");
      }
    })();
    return () => {
      cancelled = true;
      job.stopPolling();
    };
  }, [selectedFile, model, dataset]);

  const ink = isDark ? "#e6e4df" : "#1f1f1e";
  const grid = isDark ? "#3a3a38" : "#e4e2dd";
  const plotConfig = {
    displayModeBar: "hover" as const,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
    responsive: true,
  };
  const baseLayout = {
    autosize: true,
    plot_bgcolor: "transparent",
    paper_bgcolor: "transparent",
    font: { size: 10, color: ink },
    hovermode: "closest" as const,
  };

  const layerIndex = Math.min(selectedLayer, (data?.attention_profiles.n_layers ?? 1) - 1);
  const layer = data?.attention_profiles.layers[layerIndex];

  const routingTraces = useMemo(() => {
    if (!data || !layer) return [];
    const edges = data.attention_profiles.bin_edges_ms;
    const x = edges.slice(0, -1);
    const traces: any[] = [];
    layer.head_profiles.forEach((profile, head) => {
      traces.push({
        x,
        y: profile,
        type: "scatter",
        mode: "lines",
        name: `head ${head}`,
        line: { color: clusterColor(head, isDark), width: 1 },
        opacity: 0.55,
        hoverinfo: "skip",
      });
    });
    traces.push({
      x,
      y: layer.mean_profile,
      type: "scatter",
      mode: "lines+markers",
      name: "mean",
      line: { color: ink, width: 2.5 },
      marker: { size: 4 },
      hovertemplate: "offset ≤ %{x:.0f} ms<br>mass %{y:.3f}<extra>mean</extra>",
    });
    return traces;
  }, [data, layer, isDark, ink]);

  const boundaryX = useMemo(() => {
    if (!data) return [];
    return data.cka.layer_names.slice(0, -1).map((name, index) => `${name}→${data.cka.layer_names[index + 1]}`);
  }, [data]);

  if (!isWhisper) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground px-4 text-center">
        Encoder analysis is available for Whisper models.
      </div>
    );
  }

  if (!selectedFile) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground px-4 text-center">
        Select an audio file to analyze the encoder's routing and layer geometry.
      </div>
    );
  }

  const renderStatus = () => {
    if (error) {
      return (
        <div className="text-red-500 p-4 text-xs">
          <p>Encoder analysis failed: {error}</p>
          <Button size="sm" variant="outline" className="mt-2" onClick={() => { setError(null); setData(null); }}>
            Retry
          </Button>
        </div>
      );
    }
    if (!data) {
      return (
        <div className="p-4 text-xs text-muted-foreground">
          {job.isRunning
            ? `Analyzing encoder... ${job.status?.progress.message ?? ""}`
            : "No result yet."}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="h-full flex flex-col gap-3">
      <div className="flex items-center gap-2 flex-wrap">
        <Badge variant="secondary" className="text-xs">{data?.model ?? model}</Badge>
        {data && (
          <>
            <Badge variant="outline" className="text-xs">
              {data.encoder_positions} positions × {data.position_step_ms} ms
            </Badge>
            <Badge variant="outline" className="text-xs">
              {data.audio_seconds_analyzed.toFixed(1)} s analyzed
            </Badge>
            {job.status?.cache_hit && <Badge variant="outline" className="text-xs">cached</Badge>}
          </>
        )}
      </div>

      {renderStatus()}

      {data && (
        <Tabs defaultValue="routing" className="w-full">
          <TabsList className="h-7">
            <TabsTrigger value="routing" className="text-xs">Routing</TabsTrigger>
            <TabsTrigger value="geometry" className="text-xs">Geometry</TabsTrigger>
          </TabsList>

          <TabsContent value="routing" className="mt-3 space-y-3">
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Layer</span>
              <Select value={String(layerIndex)} onValueChange={(value) => setSelectedLayer(Number(value))}>
                <SelectTrigger className="w-40 h-7 text-xs">
                  <SelectValue placeholder="layer" />
                </SelectTrigger>
                <SelectContent>
                  {data.attention_profiles.layers.map((_, index) => (
                    <SelectItem key={index} value={String(index)} className="text-xs">
                      layer {index}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Card>
              <CardHeader className="pb-1">
                <CardTitle className="text-xs">
                  Distance profiles — how far layer {layerIndex}'s heads look
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <Plot
                    data={routingTraces}
                    layout={{
                      ...baseLayout,
                      margin: { l: 44, r: 12, t: 8, b: 44 },
                      xaxis: {
                        type: "log",
                        title: { text: "attention offset (ms, log)", font: { size: 10, color: ink } },
                        tickfont: { size: 9, color: ink },
                        gridcolor: grid,
                        zeroline: false,
                      },
                      yaxis: {
                        title: { text: "attention mass", font: { size: 10, color: ink } },
                        tickfont: { size: 9, color: ink },
                        gridcolor: grid,
                        zeroline: false,
                      },
                      legend: { orientation: "h", y: -0.3, font: { size: 9, color: ink }, bgcolor: "transparent" },
                      showlegend: data.attention_profiles.n_heads <= 8,
                    }}
                    config={plotConfig}
                    useResizeHandler
                    style={{ width: "100%", height: "100%" }}
                  />
                </div>
                <p className="text-[10px] text-muted-foreground mt-1">
                  A diagonal band (mass concentrated near 0 ms) means local acoustic processing; a long tail means
                  utterance-level integration. Faded lines are individual heads; the bold line is the head average.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-1">
                <CardTitle className="text-xs">Diagonal mass per layer (≤ ±{2 * data.position_step_ms} ms)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-40">
                  <Plot
                    data={[
                      {
                        x: data.attention_profiles.layers.map((_, index) => index),
                        y: data.attention_profiles.layers.map(
                          (l) => l.diagonal_mass.reduce((a, b) => a + b, 0) / l.diagonal_mass.length,
                        ),
                        type: "bar",
                        marker: { color: clusterColor(0, isDark) },
                        hovertemplate: "layer %{x}<br>diagonal mass %{y:.3f}<extra></extra>",
                      },
                    ]}
                    layout={{
                      ...baseLayout,
                      margin: { l: 44, r: 12, t: 8, b: 36 },
                      xaxis: {
                        title: { text: "encoder layer", font: { size: 10, color: ink } },
                        tickmode: "array",
                        tickvals: data.attention_profiles.layers.map((_, index) => index),
                        tickfont: { size: 9, color: ink },
                        gridcolor: grid,
                        zeroline: false,
                      },
                      yaxis: {
                        range: [0, 1],
                        title: { text: "diagonal mass", font: { size: 10, color: ink } },
                        tickfont: { size: 9, color: ink },
                        gridcolor: grid,
                        zeroline: false,
                      },
                      showlegend: false,
                    }}
                    config={plotConfig}
                    useResizeHandler
                    style={{ width: "100%", height: "100%" }}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="geometry" className="mt-3 space-y-3">
            <Card>
              <CardHeader className="pb-1">
                <CardTitle className="text-xs">CKA between layers (1 = same representation)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <Plot
                    data={[
                      {
                        z: data.cka.matrix,
                        x: data.cka.layer_names,
                        y: data.cka.layer_names,
                        type: "heatmap",
                        colorscale: "Viridis",
                        zmin: 0,
                        zmax: 1,
                        hovertemplate: "%{y} vs %{x}<br>CKA %{z:.3f}<extra></extra>",
                      },
                    ]}
                    layout={{
                      ...baseLayout,
                      margin: { l: 64, r: 12, t: 8, b: 56 },
                      xaxis: { tickfont: { size: 9, color: ink }, tickangle: 30 },
                      yaxis: { tickfont: { size: 9, color: ink }, autorange: "reversed" },
                    }}
                    config={plotConfig}
                    useResizeHandler
                    style={{ width: "100%", height: "100%" }}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-1">
                <CardTitle className="text-xs">Adjacent-layer CKA and effective dimensionality</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-48">
                  <Plot
                    data={[
                      {
                        x: boundaryX,
                        y: data.cka.adjacent_cka,
                        type: "scatter",
                        mode: "lines+markers",
                        name: "CKA(L, L+1)",
                        line: { color: clusterColor(0, isDark), width: 2 },
                        hovertemplate: "%{x}<br>CKA %{y:.3f}<extra></extra>",
                      },
                      {
                        x: data.cka.layer_names,
                        y: data.cka.participation_ratio,
                        type: "scatter",
                        mode: "lines+markers",
                        name: "participation ratio",
                        yaxis: "y2",
                        line: { color: clusterColor(1, isDark), width: 2, dash: "dot" },
                        hovertemplate: "%{x}<br>PR %{y:.3f}<extra></extra>",
                      },
                    ]}
                    layout={{
                      ...baseLayout,
                      margin: { l: 44, r: 48, t: 8, b: 70 },
                      xaxis: { tickfont: { size: 8, color: ink }, tickangle: 30, gridcolor: grid, zeroline: false },
                      yaxis: {
                        range: [0, 1],
                        title: { text: "CKA", font: { size: 10, color: ink } },
                        tickfont: { size: 9, color: ink },
                        gridcolor: grid,
                        zeroline: false,
                      },
                      yaxis2: {
                        range: [0, 1],
                        overlaying: "y",
                        side: "right",
                        title: { text: "PR", font: { size: 10, color: ink } },
                        tickfont: { size: 9, color: ink },
                        zeroline: false,
                      },
                      legend: { orientation: "h", y: -0.4, font: { size: 9, color: ink }, bgcolor: "transparent" },
                    }}
                    config={plotConfig}
                    useResizeHandler
                    style={{ width: "100%", height: "100%" }}
                  />
                </div>
                <p className="text-[10px] text-muted-foreground mt-1">
                  CKA near 1 between adjacent layers means the later layer re-uses the earlier one's representation;
                  participation ratio is the effective share of dimensions carrying variance.
                </p>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
};
