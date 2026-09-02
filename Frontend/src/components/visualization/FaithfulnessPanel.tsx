/**
 * The faithfulness test, as offered inside the Saliency tab.
 *
 * Opt-in rather than automatic: the job re-runs the model dozens of times, so it
 * is a button the user presses for a clip they care about, not something that
 * fires on every selection.
 *
 * The test always evaluates the map for the method currently on screen. A
 * faithfulness score is a property of one attribution method on one clip, not of
 * the model, so the two must never drift apart — changing the method invalidates
 * the result and clears it.
 */

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, ShieldCheck, X } from "lucide-react";

import { resolveAudioId } from "@/lib/jobs";
import { runFaithfulness, type FaithfulnessResult } from "@/lib/faithfulness";
import { FaithfulnessComparison } from "./FaithfulnessComparison";
import { FaithfulnessCurve } from "./FaithfulnessCurve";
import { FaithfulnessScore } from "./FaithfulnessScore";
import { OcclusionScatter } from "./OcclusionScatter";

interface FaithfulnessPanelProps {
  selectedFile?: any;
  model?: string;
  dataset?: string;
  /** The attribution method currently displayed, and therefore the one tested. */
  method: string;
}

/**
 * Presets rather than raw knobs.
 *
 * Every setting here trades runtime against how tightly the random baseline is
 * pinned down, which is not a choice most users can make from the parameter
 * names. `random_repeats` matters most: it is what turns the headline gain from
 * a single noisy draw into something with an error bar.
 */
const DEPTH_PRESETS = {
  quick: { label: "Quick", nSteps: 5, randomRepeats: 2, includeOcclusion: false, blurb: "~15 passes" },
  standard: { label: "Standard", nSteps: 9, randomRepeats: 3, includeOcclusion: true, blurb: "~70 passes" },
  thorough: { label: "Thorough", nSteps: 12, randomRepeats: 5, includeOcclusion: true, blurb: "~130 passes" },
} as const;

type DepthKey = keyof typeof DEPTH_PRESETS;

export const FaithfulnessPanel = ({
  selectedFile,
  model,
  dataset,
  method,
}: FaithfulnessPanelProps) => {
  const [result, setResult] = useState<FaithfulnessResult | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [depth, setDepth] = useState<DepthKey>("standard");

  // A result belongs to one (clip, model, method). Anything else on screen makes
  // it stale, and a stale verdict beside a different map is worse than none.
  useEffect(() => {
    setResult(null);
    setError(null);
  }, [selectedFile, model, method]);

  const run = async () => {
    if (!selectedFile || !model) return;
    setRunning(true);
    setError(null);
    setProgress("Queued");
    try {
      const audioId = await resolveAudioId(selectedFile, dataset);
      const preset = DEPTH_PRESETS[depth];
      const value = await runFaithfulness(
        {
          model,
          audioId,
          method: method as "gradcam" | "lime" | "shap",
          nSteps: preset.nSteps,
          randomRepeats: preset.randomRepeats,
          includeOcclusion: preset.includeOcclusion,
        },
        { onProgress: (status) => setProgress(status.progress?.message || status.status) },
      );
      setResult(value);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Faithfulness test failed");
    } finally {
      setRunning(false);
      setProgress(null);
    }
  };

  if (!selectedFile || !model) return null;

  return (
    <div className="mt-6 border-t border-border pt-4 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-muted-foreground" />
            <h4 className="text-sm font-medium">Is this map telling the truth?</h4>
            <Badge variant="secondary" className="text-[10px]">
              {method}
            </Badge>
          </div>
          <p className="text-[11px] text-muted-foreground mt-0.5 max-w-prose">
            Removes the audio this map calls important and checks whether the model actually
            breaks — and whether it breaks more than when the same amount is removed at random.
          </p>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <Select value={depth} onValueChange={(value) => setDepth(value as DepthKey)}>
            <SelectTrigger className="w-[132px] h-8 text-xs" disabled={running}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(DEPTH_PRESETS).map(([key, preset]) => (
                <SelectItem key={key} value={key} className="text-xs">
                  {preset.label} · {preset.blurb}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button size="sm" onClick={run} disabled={running} className="h-8">
            {running ? (
              <>
                <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                {progress ?? "Running"}
              </>
            ) : result ? (
              "Re-run test"
            ) : (
              "Run faithfulness test"
            )}
          </Button>
        </div>
      </div>

      {running && (
        <p className="text-[11px] text-muted-foreground">
          Re-running the model on {DEPTH_PRESETS[depth].blurb.replace("~", "")} of masked audio.
          This takes appreciably longer than generating the map itself.
        </p>
      )}

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2">
          <X className="h-3.5 w-3.5 text-destructive mt-0.5 shrink-0" />
          <p className="text-[11px]">{error}</p>
        </div>
      )}

      {result && (
        <div className="space-y-4">
          <FaithfulnessScore result={result} />

          <Tabs defaultValue="comparison">
            <TabsList className="h-8">
              <TabsTrigger value="comparison" className="text-xs">
                Before / after
              </TabsTrigger>
              <TabsTrigger value="curve" className="text-xs">
                Deletion curve
              </TabsTrigger>
              <TabsTrigger value="segments" className="text-xs">
                Per segment
              </TabsTrigger>
            </TabsList>

            <TabsContent value="comparison" className="mt-3">
              <FaithfulnessComparison result={result} />
            </TabsContent>

            <TabsContent value="curve" className="mt-3">
              <div className="h-[300px]">
                <FaithfulnessCurve result={result} showInsertion />
              </div>
              <p className="text-[11px] text-muted-foreground mt-2 leading-relaxed">
                A faithful map makes the orange line fall fastest and the teal line slowest, with
                the grey baseline between them. Lines lying on top of each other mean the map's
                ordering carries no information.
              </p>
            </TabsContent>

            <TabsContent value="segments" className="mt-3">
              <div className="h-[300px]">
                <OcclusionScatter result={result} />
              </div>
              <p className="text-[11px] text-muted-foreground mt-2 leading-relaxed">
                Each point is one segment: what the map claimed it was worth, against what removing
                it actually cost. Points on a rising line mean the map ranked the audio the way the
                model does.
              </p>
            </TabsContent>
          </Tabs>

          <p className="text-[10px] text-muted-foreground">
            {result.eval_frames} evaluation frames over {result.audio_seconds.toFixed(1)}s ·{" "}
            {result.n_steps} deletion steps · {result.random_repeats} random draws · seed{" "}
            {result.seed}
          </p>
        </div>
      )}
    </div>
  );
};
