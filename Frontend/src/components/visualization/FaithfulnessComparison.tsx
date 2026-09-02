/**
 * Before / after: what the model does once its own highlighted audio is taken away.
 *
 * Three rows, deliberately in this order:
 *
 *   1. the clip as the model heard it, with the saliency map over it
 *   2. the same clip with the most salient regions cut out
 *   3. the same clip with the *same amount* of audio cut out at random
 *
 * Row 3 is the whole point. Cutting a fifth of any clip damages a model, so row 2
 * alone always looks dramatic. Only the distance between rows 2 and 3 says the
 * map found something. Showing row 2 without row 3 would be a misleading chart,
 * so the component renders them together or not at all.
 */

import { useMemo } from "react";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { HelpCircle } from "lucide-react";
import { targetDescription, type FaithfulnessResult } from "@/lib/faithfulness";

interface FaithfulnessComparisonProps {
  result: FaithfulnessResult;
}

/** Saliency colour ramp, matching `SaliencyVisualization`'s heatmap anchors. */
const intensityToColor = (value: number) => {
  const clamp = (x: number) => Math.max(0, Math.min(1, x));
  const mix = (a: number, b: number, t: number) => a + (b - a) * t;
  let h: number, s: number, l: number;
  if (value < 0.5) {
    const t = clamp(value / 0.5);
    h = mix(178, 45, t); s = mix(68, 93, t); l = mix(78, 58, t);
  } else {
    const t = clamp((value - 0.5) / 0.5);
    h = mix(45, 15, t); s = mix(93, 86, t); l = mix(58, 58, t);
  }
  return `hsl(${h} ${s}% ${l}%)`;
};

interface StripProps {
  series: number[];
  duration: number;
  /** Regions cut out of the audio, in seconds. */
  removed: Array<[number, number]>;
  /** Paint the saliency map, or show the clip as a neutral band. */
  showSaliency: boolean;
}

const Strip = ({ series, duration, removed, showSaliency }: StripProps) => {
  const cells = useMemo(() => {
    if (series.length === 0) return [];
    const width = 100 / series.length;
    return series.map((value, index) => {
      const start = (index / series.length) * duration;
      const end = ((index + 1) / series.length) * duration;
      const isRemoved = removed.some(([from, to]) => start < to && end > from);
      return { left: index * width, width, value, isRemoved };
    });
  }, [series, duration, removed]);

  if (cells.length === 0) {
    return <div className="h-9 rounded bg-muted" />;
  }

  return (
    <div className="relative h-9 rounded overflow-hidden bg-muted/40 border border-border">
      {cells.map((cell, index) => (
        <div
          key={index}
          className="absolute top-0 bottom-0"
          style={{
            left: `${cell.left}%`,
            width: `${cell.width + 0.15}%`,
            background: cell.isRemoved
              ? undefined
              : showSaliency
                ? intensityToColor(cell.value)
                : "hsl(var(--muted-foreground) / 0.35)",
            opacity: cell.isRemoved ? 1 : showSaliency ? 0.25 + 0.75 * cell.value : 0.8,
          }}
        >
          {cell.isRemoved && (
            // Cut audio reads as absence, not as another intensity: a hatched
            // gap can never be confused with a low-saliency region.
            <div
              className="absolute inset-0"
              style={{
                backgroundImage:
                  "repeating-linear-gradient(45deg, hsl(var(--muted-foreground) / 0.28) 0 3px, transparent 3px 6px)",
                backgroundColor: "hsl(var(--background))",
              }}
            />
          )}
        </div>
      ))}
    </div>
  );
};

const ScoreBar = ({ value, tone }: { value: number; tone: string }) => (
  <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
    <div
      className={`h-full rounded-full ${tone}`}
      style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
    />
  </div>
);

interface RowProps {
  title: string;
  caption: string;
  score: number;
  delta?: number;
  tone: string;
  children: React.ReactNode;
}

const Row = ({ title, caption, score, delta, tone, children }: RowProps) => (
  <div className="space-y-1.5">
    <div className="flex items-baseline justify-between gap-3">
      <div className="min-w-0">
        <div className="text-xs font-medium">{title}</div>
        <div className="text-[11px] text-muted-foreground truncate">{caption}</div>
      </div>
      <div className="text-right shrink-0">
        <div className="text-sm font-semibold tabular-nums">{score.toFixed(3)}</div>
        {delta !== undefined && (
          <div
            className={`text-[11px] tabular-nums ${
              delta < -0.001 ? "text-orange-600 dark:text-orange-400" : "text-muted-foreground"
            }`}
          >
            {delta >= 0 ? "+" : ""}
            {delta.toFixed(3)}
          </div>
        )}
      </div>
    </div>
    {children}
    <ScoreBar value={score} tone={tone} />
  </div>
);

export const FaithfulnessComparison = ({ result }: FaithfulnessComparisonProps) => {
  const { comparison, saliency } = result;
  const series = saliency.series ?? [];
  const duration = saliency.total_duration || result.audio_seconds || 0;
  const removedSeconds = comparison.removed_spans.reduce((sum, [a, b]) => sum + (b - a), 0);
  const percentRemoved = Math.round(comparison.fraction * 100);
  const tracked = targetDescription(result.target);

  if (result.skipped_reason) {
    return (
      <div className="text-xs text-muted-foreground px-1 py-6 text-center">
        {result.skipped_reason}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-1.5">
        <p className="text-[11px] text-muted-foreground leading-relaxed">
          The model's {tracked}, before and after {percentRemoved}% of the audio (
          {removedSeconds.toFixed(2)}s) is cut out. The third row removes the same amount from
          random places — the honest control for the second.
        </p>
        <Tooltip>
          <TooltipTrigger>
            <HelpCircle className="h-3 w-3 text-muted-foreground mt-0.5 shrink-0" />
          </TooltipTrigger>
          <TooltipContent className="max-w-xs text-xs">
            Cut audio is replaced with a near-silent noise floor rather than true silence, which
            keeps the model's front end well behaved. Hatched blocks are the removed regions.
          </TooltipContent>
        </Tooltip>
      </div>

      <Row
        title="Before — original audio"
        caption={
          result.target.label
            ? `model output: "${result.target.label}"`
            : "the clip as the model heard it"
        }
        score={comparison.clean_score}
        tone="bg-emerald-500"
      >
        <Strip series={series} duration={duration} removed={[]} showSaliency />
      </Row>

      <Row
        title={`After — top ${percentRemoved}% most salient removed`}
        caption="the audio the saliency map called important"
        score={comparison.masked_score}
        delta={comparison.masked_score - comparison.clean_score}
        tone="bg-orange-500"
      >
        <Strip
          series={series}
          duration={duration}
          removed={comparison.removed_spans as Array<[number, number]>}
          showSaliency
        />
      </Row>

      <Row
        title={`Control — a random ${percentRemoved}% removed`}
        caption="same amount of audio, chosen at random"
        score={comparison.random_score}
        delta={comparison.random_score - comparison.clean_score}
        tone="bg-slate-400"
      >
        <Strip
          series={series}
          duration={duration}
          removed={comparison.random_spans as Array<[number, number]>}
          showSaliency={false}
        />
      </Row>

      <div className="rounded-md border border-border bg-muted/30 px-3 py-2">
        <p className="text-[11px] leading-relaxed">
          <span className="font-medium">What to read here: </span>
          {comparison.masked_score < comparison.random_score - 0.01 ? (
            <>
              cutting the highlighted audio cost{" "}
              <span className="font-semibold tabular-nums">
                {(comparison.random_score - comparison.masked_score).toFixed(3)}
              </span>{" "}
              more than cutting the same amount at random. The map is pointing at audio this model
              genuinely depends on.
            </>
          ) : (
            <>
              cutting the highlighted audio cost no more than cutting random audio
              {comparison.masked_score > comparison.random_score ? " — in fact it cost less" : ""}.
              On this clip the map is not identifying what the model relies on.
            </>
          )}
        </p>
      </div>
    </div>
  );
};
