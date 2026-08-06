import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { WaveformViewer } from "@/components/audio/WaveformViewer";
import { API_BASE } from "@/lib/api";
import { wordDiff } from "@/lib/wordDiff";
import { FR7_GLOSSARY } from "@/lib/fr7Glossary";
import { InfoTooltip } from "./InfoTooltip";
import type { CurvePoint, LinguisticAcousticResult } from "@/lib/linguisticAcoustic";

interface VariantInspectorProps {
  result: LinguisticAcousticResult;
  baselineAudioUrl: string;
  selectedProperty: string | null;
  selectedTheta: number | null;
}

function closestPoint(theta: number, points: CurvePoint[]): CurvePoint {
  return points.reduce((best, point) =>
    Math.abs(point.theta - theta) < Math.abs(best.theta - theta) ? point : best
  );
}

function WordDiffView({ reference, hypothesis }: { reference: string; hypothesis: string }) {
  const ops = useMemo(() => wordDiff(reference, hypothesis), [reference, hypothesis]);
  return (
    <p className="text-xs font-mono break-words leading-relaxed">
      {ops.map((op, index) => {
        if (op.type === "match") {
          return <span key={index}>{op.word} </span>;
        }
        if (op.type === "substitute") {
          return (
            <span key={index} className="bg-amber-500/20 text-amber-700 dark:text-amber-400 rounded px-0.5">
              {op.hypWord}{" "}
            </span>
          );
        }
        if (op.type === "insert") {
          return (
            <span key={index} className="bg-red-500/20 text-red-700 dark:text-red-400 rounded px-0.5">
              {op.word}{" "}
            </span>
          );
        }
        return (
          <span key={index} className="bg-muted text-muted-foreground line-through rounded px-0.5">
            {op.word}{" "}
          </span>
        );
      })}
    </p>
  );
}

export function VariantInspector({
  result, baselineAudioUrl, selectedProperty, selectedTheta,
}: VariantInspectorProps) {
  const profile = useMemo(
    () => result.properties.find((p) => p.property === selectedProperty && p.curve?.length),
    [result, selectedProperty],
  );
  const point = useMemo(() => {
    if (!profile?.curve?.length || selectedTheta === null) return null;
    return closestPoint(selectedTheta, profile.curve);
  }, [profile, selectedTheta]);

  if (!selectedProperty || !profile || !point) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground border border-dashed border-border rounded-md p-6 text-center">
        Click a point on the sensitivity curve to compare the baseline against that variant.
      </div>
    );
  }

  const variantUrl = point.playback_url ? `${API_BASE}${point.playback_url}` : undefined;
  const isClassification = result.task === "classification";
  const variantTranscript = typeof point.raw.transcript === "string" ? point.raw.transcript : null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          Variant inspector
          <Badge variant="outline" className="text-xs font-mono">
            {profile.property} = {point.theta.toFixed(2)} {profile.unit}
          </Badge>
          {point.measured_snr_db != null && (
            <Badge variant="outline" className="text-xs font-mono">
              measured SNR {point.measured_snr_db.toFixed(1)} dB
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <div className="text-xs font-medium text-muted-foreground mb-1">Baseline</div>
          <WaveformViewer audioUrl={baselineAudioUrl} />
          {result.baseline.transcript && (
            <p className="text-xs mt-1 font-mono break-words">{result.baseline.transcript}</p>
          )}
        </div>

        <div>
          <div className="text-xs font-medium text-muted-foreground mb-1 flex items-center gap-2">
            Variant
            {!variantUrl && (
              <span className="italic">
                (playback unavailable, served from cache)
              </span>
            )}
          </div>
          {variantUrl ? (
            <WaveformViewer audioUrl={variantUrl} />
          ) : (
            <div className="h-20 rounded border border-dashed border-border flex items-center justify-center text-xs text-muted-foreground">
              No variant audio for this cached result
            </div>
          )}

          {!isClassification && variantTranscript && (
            <div className="text-xs mt-1 space-y-0.5">
              <div className="flex items-center gap-1">
                <span className="text-muted-foreground">word diff against baseline</span>
                <InfoTooltip text={FR7_GLOSSARY.wordDiffColors} />
              </div>
              {result.baseline.transcript ? (
                <WordDiffView reference={result.baseline.transcript} hypothesis={variantTranscript} />
              ) : (
                <p className="font-mono break-words">{variantTranscript}</p>
              )}
              {typeof point.raw.wer === "number" && (
                <p className="text-muted-foreground flex items-center gap-1">
                  <span>
                    self-WER {(point.raw.wer as number).toFixed(2)} · CER{" "}
                    {typeof point.raw.cer === "number" ? (point.raw.cer as number).toFixed(2) : "n/a"}
                  </span>
                  <InfoTooltip text={FR7_GLOSSARY.selfWer} />
                </p>
              )}
            </div>
          )}

          {isClassification && (
            <div className="text-xs mt-1 space-y-0.5">
              <p>
                predicted <span className="font-mono">{String(point.raw.predicted_label ?? "n/a")}</span>{" "}
                (baseline <span className="font-mono">{String(point.raw.baseline_label ?? "n/a")}</span>)
              </p>
              {typeof point.raw.confidence_delta === "number" && (
                <p className="text-muted-foreground">
                  confidence change {(point.raw.confidence_delta as number).toFixed(2)}, label
                  flipped: {point.raw.label_flipped ? "yes" : "no"}
                </p>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
