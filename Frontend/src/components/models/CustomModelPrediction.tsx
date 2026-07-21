import React, { useEffect, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { AlertCircle, Clock } from 'lucide-react';
import { API_BASE } from '@/lib/api';
import { CustomModelPredictionResult, isCustomModel } from '@/lib/customModels';

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
}

interface CustomModelPredictionProps {
  model?: string;
  selectedFile?: UploadedFile | string | null;
  dataset?: string;
}

/** Renders a registered model's output, shaped by its task.
 *
 * Which view appears is driven entirely by `task` in the response, so a newly
 * supported architecture within an existing task family needs no change here.
 */
export const CustomModelPrediction = ({ model, selectedFile, dataset }: CustomModelPredictionProps) => {
  const [result, setResult] = useState<CustomModelPredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isCustomModel(model) || !selectedFile) {
      setResult(null);
      return;
    }

    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const body: Record<string, unknown> = { model };
        if (typeof selectedFile === 'string') {
          body.dataset = dataset;
          body.dataset_file = selectedFile;
        } else if (selectedFile.file_path) {
          body.file_path = selectedFile.file_path;
        } else {
          body.dataset = dataset;
          body.dataset_file = selectedFile.filename;
        }

        const res = await fetch(`${API_BASE}/inferences/run`, {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data?.detail || `Inference failed: ${res.status}`);
        if (!cancelled) setResult(data);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Inference failed');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    run();
    return () => {
      cancelled = true;
    };
  }, [model, selectedFile, dataset]);

  if (!isCustomModel(model)) return null;

  if (!selectedFile) {
    return (
      <div className="text-center py-8 text-muted-foreground text-sm">
        Select an audio file to run this custom model.
      </div>
    );
  }

  if (loading) {
    return (
      <div className="text-center py-8">
        <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
        <p className="text-sm text-muted-foreground">Running custom model...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-3 bg-red-50 border border-red-200 rounded-md flex items-start gap-2">
        <AlertCircle className="h-4 w-4 text-red-500 shrink-0 mt-0.5" />
        <span className="text-sm text-red-700">{error}</span>
      </div>
    );
  }

  if (!result) return null;

  const sortedProbs = Object.entries(result.probabilities || {}).sort((a, b) => b[1] - a[1]);

  return (
    <div className="space-y-3">
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between gap-2">
            <CardTitle className="text-sm truncate">{result.model_name}</CardTitle>
            <div className="flex items-center gap-1.5 shrink-0">
              <Badge variant="secondary" className="text-[10px]">
                {result.task_label}
              </Badge>
              <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                <Clock className="h-3 w-3" />
                {result.inference_seconds.toFixed(2)}s
              </span>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {result.limit_exceeded && (
            <div className="flex items-start gap-1.5 text-xs text-amber-600">
              <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-px" />
              <span>{result.limit_exceeded}</span>
            </div>
          )}

          {/* ASR tasks: seq2seq and CTC both return `text`. */}
          {result.text !== undefined && (
            <div>
              <p className="text-xs font-medium mb-1">Transcript</p>
              <p className="text-sm bg-muted rounded p-2 break-words">
                {result.text || <span className="text-muted-foreground italic">(empty)</span>}
              </p>
            </div>
          )}

          {/* Classification. */}
          {result.predicted_label !== undefined && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <p className="text-xs font-medium">Predicted class</p>
                <Badge>{result.predicted_label}</Badge>
                <span className="text-xs text-muted-foreground">
                  {((result.confidence || 0) * 100).toFixed(1)}% confidence
                </span>
              </div>
              <div className="space-y-1.5">
                {sortedProbs.map(([label, prob]) => (
                  <div key={label} className="space-y-0.5">
                    <div className="flex justify-between text-[11px]">
                      <span className="truncate">{label}</span>
                      <span className="text-muted-foreground">{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={prob * 100} className="h-1.5" />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* CTC only: per-frame argmax token and its probability. */}
          {result.frame_probabilities && (
            <div>
              <p className="text-xs font-medium mb-1">
                Frame-level token probabilities
                <span className="text-muted-foreground font-normal">
                  {' '}
                  ({result.frame_probabilities.total_frames} frames over{' '}
                  {result.frame_probabilities.duration.toFixed(1)}s)
                </span>
              </p>
              <div className="flex flex-wrap gap-px max-h-40 overflow-y-auto bg-muted rounded p-1.5">
                {result.frame_probabilities.frames.map((f, i) => (
                  <span
                    key={i}
                    title={`${f.time.toFixed(2)}s · ${f.token} · ${(f.probability * 100).toFixed(1)}%`}
                    className="text-[10px] font-mono px-1 rounded"
                    style={{
                      // Opacity encodes confidence, so low-certainty frames
                      // visually recede without needing a second channel.
                      backgroundColor: `rgba(37, 99, 235, ${f.probability * 0.8})`,
                      color: f.probability > 0.6 ? 'white' : 'inherit',
                    }}
                  >
                    {f.token === '<pad>' || f.token === '|' ? '␣' : f.token}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="flex flex-wrap gap-1 pt-1 border-t">
            {result.capabilities.map((c) => (
              <Badge key={c} variant="outline" className="text-[9px]">
                {c.replace(/_/g, ' ')}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
