/**
 * The headline: is this saliency map worth believing, and by how much.
 *
 * The gauge shows `faithfulness_gain` — how much *more* the model suffers when
 * its most salient audio is removed than when the same duration is removed at
 * random. Two presentation rules keep it honest:
 *
 *  - The error bar on the random baseline is drawn on the gauge as a dead zone.
 *    A gain inside that zone is noise, and the reader sees that directly rather
 *    than being told a number and left to over-read it.
 *  - The verdict comes from the backend (`result.verdict`), which applies the
 *    threshold rule once. Re-deriving a judgement here would let two views of
 *    the same result disagree.
 */

import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { AlertTriangle, HelpCircle } from "lucide-react";

import {
  METRIC_HELP,
  VERDICT_COPY,
  formatSigned,
  targetDescription,
  type FaithfulnessResult,
} from "@/lib/faithfulness";

interface FaithfulnessScoreProps {
  result: FaithfulnessResult;
}

const VERDICT_TONE: Record<string, { text: string; ring: string; arc: string }> = {
  faithful: {
    text: "text-emerald-700 dark:text-emerald-500",
    ring: "border-emerald-500/40 bg-emerald-500/10",
    arc: "stroke-emerald-500",
  },
  weak: {
    text: "text-amber-600 dark:text-amber-500",
    ring: "border-amber-500/40 bg-amber-500/10",
    arc: "stroke-amber-500",
  },
  uninformative: {
    text: "text-muted-foreground",
    ring: "border-border bg-muted/40",
    arc: "stroke-muted-foreground",
  },
};

/**
 * Semicircular gauge over a fixed 0..0.5 gain range.
 *
 * The range is fixed rather than fitted to the value: a gauge that rescales
 * itself makes every result look equally impressive, and gains above ~0.5 are
 * saturating anyway.
 */
const GAUGE_MAX = 0.5;

const Gauge = ({ gain, noise, tone }: { gain: number; noise: number; tone: string }) => {
  const radius = 52;
  const circumference = Math.PI * radius;
  const clamped = Math.max(0, Math.min(GAUGE_MAX, gain));
  const filled = (clamped / GAUGE_MAX) * circumference;
  const deadZone = (Math.min(GAUGE_MAX, Math.max(noise, 0.02)) / GAUGE_MAX) * circumference;

  return (
    <svg viewBox="0 0 128 72" className="w-32 h-[72px]" role="img" aria-label="faithfulness gain">
      <path
        d="M 12 64 A 52 52 0 0 1 116 64"
        fill="none"
        strokeWidth={9}
        strokeLinecap="round"
        className="stroke-muted"
      />
      {/* Dead zone: anything inside the baseline's own spread is not a finding. */}
      <path
        d="M 12 64 A 52 52 0 0 1 116 64"
        fill="none"
        strokeWidth={9}
        strokeLinecap="butt"
        strokeDasharray={`${deadZone} ${circumference}`}
        className="stroke-muted-foreground/30"
      />
      <path
        d="M 12 64 A 52 52 0 0 1 116 64"
        fill="none"
        strokeWidth={9}
        strokeLinecap="round"
        strokeDasharray={`${filled} ${circumference}`}
        className={tone}
      />
      <text
        x="64" y="58" textAnchor="middle"
        className="fill-foreground font-semibold"
        style={{ fontSize: 20 }}
      >
        {formatSigned(gain, 2)}
      </text>
    </svg>
  );
};

const Tile = ({
  label,
  value,
  help,
  hint,
}: {
  label: string;
  value: string;
  help: string;
  hint?: string;
}) => (
  <div className="rounded-md border border-border bg-card/60 px-3 py-2">
    <div className="flex items-center gap-1">
      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</span>
      <Tooltip>
        <TooltipTrigger>
          <HelpCircle className="h-3 w-3 text-muted-foreground" />
        </TooltipTrigger>
        <TooltipContent className="max-w-xs text-xs">{help}</TooltipContent>
      </Tooltip>
    </div>
    <div className="text-base font-semibold tabular-nums">{value}</div>
    {hint && <div className="text-[10px] text-muted-foreground">{hint}</div>}
  </div>
);

export const FaithfulnessScore = ({ result }: FaithfulnessScoreProps) => {
  const { metrics, verdict } = result;
  const tone = VERDICT_TONE[verdict] ?? VERDICT_TONE.uninformative;
  const copy = VERDICT_COPY[verdict];
  const rho = metrics.occlusion_spearman;

  return (
    <div className="space-y-3">
      {result.attribution_source === "energy_fallback" && (
        <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2">
          <AlertTriangle className="h-3.5 w-3.5 text-amber-600 dark:text-amber-500 mt-0.5 shrink-0" />
          <p className="text-[11px] leading-relaxed">
            <span className="font-medium">This is not a real attribution.</span> The requested
            method failed for this clip and the saliency service substituted an encoder energy map.
            The verdict below describes that fallback, not {result.method}.
          </p>
        </div>
      )}

      <div className={`rounded-lg border ${tone.ring} px-4 py-3`}>
        <div className="flex items-center gap-4">
          <Gauge
            gain={metrics.faithfulness_gain}
            noise={metrics.aopc_random_stderr}
            tone={tone.arc}
          />
          <div className="min-w-0 space-y-1">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className={`${tone.text} border-current`}>
                {copy.label}
              </Badge>
              <span className="text-[11px] text-muted-foreground">
                tracking {targetDescription(result.target)}
              </span>
            </div>
            <p className="text-[11px] leading-relaxed text-muted-foreground">{copy.blurb}</p>
          </div>
        </div>
        <div className="mt-2 pt-2 border-t border-border/60 text-[11px] text-muted-foreground">
          Salient audio costs{" "}
          <span className="font-semibold text-foreground tabular-nums">
            {metrics.aopc_deletion.toFixed(3)}
          </span>{" "}
          to remove; the same amount of random audio costs{" "}
          <span className="font-semibold text-foreground tabular-nums">
            {metrics.aopc_random.toFixed(3)}
          </span>
          {metrics.aopc_random_stderr > 0 && (
            <> ± {metrics.aopc_random_stderr.toFixed(3)} over {result.random_repeats} draws</>
          )}
          . The difference is the score above.
        </div>
      </div>

      {verdict !== "faithful" && rho !== null && rho >= 0.3 && (
        <p className="text-[11px] text-muted-foreground leading-relaxed px-1">
          <span className="font-medium text-foreground">Note the disagreement: </span>
          the deletion curve cannot separate this map from random, but segment by segment the map
          still ranks the audio roughly the way the model does (ρ = {rho.toFixed(2)}). That usually
          means the model is fragile — it collapses when any sizeable chunk is removed, so the
          curve saturates before the map's ordering can show. Trust the per-segment view here, and
          read the individual peaks cautiously.
        </p>
      )}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
        <Tile
          label="Gain vs random"
          value={formatSigned(metrics.faithfulness_gain)}
          help={METRIC_HELP.faithfulness_gain}
          hint={`noise ±${metrics.aopc_random_stderr.toFixed(3)}`}
        />
        <Tile
          label="Segment agreement"
          value={rho === null ? "—" : rho.toFixed(2)}
          help={METRIC_HELP.occlusion_spearman}
          hint={
            rho === null
              ? "not enough segments"
              : metrics.occlusion_p_value !== null
                ? `p = ${metrics.occlusion_p_value < 0.001 ? "<0.001" : metrics.occlusion_p_value.toFixed(3)}`
                : undefined
          }
        />
        <Tile
          label="Comprehensiveness"
          value={metrics.comprehensiveness.toFixed(3)}
          help={METRIC_HELP.comprehensiveness}
          hint="higher is better"
        />
        <Tile
          label="Sufficiency"
          value={metrics.sufficiency.toFixed(3)}
          help={METRIC_HELP.sufficiency}
          hint="lower is better"
        />
      </div>
    </div>
  );
};
