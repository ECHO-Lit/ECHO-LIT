import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { WaveformViewer } from "@/components/audio/WaveformViewer";
import { API_BASE } from "@/lib/api";
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
                (playback unavailable — served from cache)
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

          {!isClassification && typeof point.raw.transcript === "string" && (
            <div className="text-xs mt-1 space-y-0.5">
              <p className="font-mono break-words">{point.raw.transcript}</p>
              {typeof point.raw.wer === "number" && (
                <p className="text-muted-foreground">
                  WER {(point.raw.wer as number).toFixed(2)} · CER{" "}
                  {typeof point.raw.cer === "number" ? (point.raw.cer as number).toFixed(2) : "—"}
                </p>
              )}
            </div>
          )}

          {isClassification && (
            <div className="text-xs mt-1 space-y-0.5">
              <p>
                predicted <span className="font-mono">{String(point.raw.predicted_label ?? "—")}</span>{" "}
                (baseline <span className="font-mono">{String(point.raw.baseline_label ?? "—")}</span>)
              </p>
              {typeof point.raw.confidence_delta === "number" && (
                <p className="text-muted-foreground">
                  confidence Δ {(point.raw.confidence_delta as number).toFixed(2)} · label
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
