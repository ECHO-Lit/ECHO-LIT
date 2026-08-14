/**
 * Per-property headline cards plus the plain-English emergence narrative.
 *
 * This is also where the palette's light-mode contrast obligation is met: every
 * property is listed in text beside its colour swatch, so identity never rests
 * on a hue that sits below 3:1 against the surface.
 */

import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { HelpCircle } from "lucide-react";

import { useIsDarkMode } from "@/hooks/useIsDarkMode";
import { clusterColor } from "@/lib/clusterPalette";
import {
  byPeakDepth,
  propertyLabel,
  propertyMeaning,
  type LayerProbeResult,
  type PropertyProbe,
} from "@/lib/probes";

interface LayerProbeSummaryProps {
  result: LayerProbeResult;
  properties: string[];
  selectedProperty: string | null;
  onSelectProperty: (key: string) => void;
}

const percent = (value: number | null | undefined) =>
  value === null || value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;

/** Qualitative reading of selectivity, mirroring `silhouetteBand`'s job. */
function selectivityBand(probe: PropertyProbe): { label: string; tone: string; hint: string } {
  const selectivity = probe.best_selectivity;
  const beatsMajority =
    probe.best_accuracy !== null &&
    probe.majority_baseline !== null &&
    probe.best_accuracy > probe.majority_baseline + 0.02;

  if (selectivity === null) {
    return {
      label: "No control",
      tone: "text-muted-foreground",
      hint: "The control probe was disabled, so this accuracy cannot be separated from memorisation.",
    };
  }
  if (selectivity < 0.05 || !beatsMajority) {
    return {
      label: "No information",
      tone: "text-muted-foreground",
      hint: "The probe did no better than shuffled labels or than always guessing the largest class. The model does not linearly encode this property.",
    };
  }
  if (selectivity < 0.2) {
    return {
      label: "Weak",
      tone: "text-amber-600 dark:text-amber-500",
      hint: "Some information is present, but a probe on shuffled labels gets close. Treat the peak layer as approximate.",
    };
  }
  return {
    label: "Strong",
    tone: "text-emerald-700 dark:text-emerald-500",
    hint: "The probe clearly beats both its shuffled-label control and the majority baseline. This property is genuinely encoded at this depth.",
  };
}

function narrative(result: LayerProbeResult, properties: string[]): string | null {
  const ranked = byPeakDepth(result).filter(([key]) => properties.includes(key));
  if (ranked.length < 2) return null;
  const names = ranked.map(([key]) => propertyLabel(key).toLowerCase());
  const last = names.pop();
  return `Reading the stack from the input upward, information emerges in this order: ${names.join(", ")}, then ${last}. Properties peaking lower are acoustic; properties peaking higher are linguistic.`;
}

export const LayerProbeSummary = ({
  result,
  properties,
  selectedProperty,
  onSelectProperty,
}: LayerProbeSummaryProps) => {
  const isDark = useIsDarkMode();
  const story = narrative(result, properties);
  const shown = properties.filter((key) => result.properties[key]);

  return (
    <div className="space-y-2">
      {story && (
        <div className="text-[11px] leading-relaxed text-muted-foreground bg-muted/40 border border-border rounded-md p-2">
          {story}
        </div>
      )}

      {shown.map((key) => {
        const probe = result.properties[key];
        const colour = clusterColor(properties.indexOf(key), isDark);
        const isSelected = selectedProperty === key;

        if (probe.skipped_reason) {
          return (
            <div key={key} className="border border-border rounded-md p-2 bg-card">
              <div className="flex items-center gap-1.5">
                <span
                  className="inline-block w-2.5 h-2.5 rounded-sm shrink-0"
                  style={{ backgroundColor: colour }}
                />
                <span className="text-xs font-medium text-foreground">{propertyLabel(key)}</span>
                <Badge variant="outline" className="text-[10px] ml-auto">
                  Not probed
                </Badge>
              </div>
              <div className="text-[11px] text-muted-foreground mt-1">{probe.skipped_reason}</div>
              {probe.dropped_classes.length > 0 && (
                <div className="text-[11px] text-muted-foreground mt-1">
                  Dropped:{" "}
                  {probe.dropped_classes
                    .map((entry) => `${entry.label} (${entry.count})`)
                    .join(", ")}
                </div>
              )}
            </div>
          );
        }

        const band = selectivityBand(probe);
        return (
          <button
            key={key}
            type="button"
            onClick={() => onSelectProperty(key)}
            className={`w-full text-left border rounded-md p-2 bg-card transition-colors ${
              isSelected ? "border-primary ring-1 ring-primary/30" : "border-border hover:bg-muted/40"
            }`}
          >
            <div className="flex items-center gap-1.5">
              <span
                className="inline-block w-2.5 h-2.5 rounded-sm shrink-0"
                style={{ backgroundColor: colour }}
              />
              <span className="text-xs font-medium text-foreground">{propertyLabel(key)}</span>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span>
                    <HelpCircle className="h-3 w-3 text-muted-foreground" />
                  </span>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs text-xs">{propertyMeaning(key)}</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className={`text-[10px] font-medium ml-auto ${band.tone}`}>{band.label}</span>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs text-xs">{band.hint}</TooltipContent>
              </Tooltip>
            </div>

            <div className="text-[11px] text-muted-foreground mt-1 leading-relaxed">
              Peaks at <span className="text-foreground font-medium">
                {result.layer_names[probe.best_layer ?? 0]}
              </span>{" "}
              of {result.num_layers - 1} ({Math.round((probe.peak_depth ?? 0) * 100)}% depth) ·{" "}
              <span className="text-foreground font-medium">{percent(probe.best_accuracy)}</span>{" "}
              accuracy vs {percent(probe.majority_baseline)} majority · selectivity{" "}
              <span className="text-foreground font-medium">
                {probe.best_selectivity === null
                  ? "—"
                  : `${probe.best_selectivity >= 0 ? "+" : ""}${(probe.best_selectivity * 100).toFixed(1)}pp`}
              </span>
            </div>

            <div className="text-[11px] text-muted-foreground mt-0.5">
              {probe.n_samples} labelled · {probe.n_classes} classes · {probe.cv_folds_used}-fold CV
              {probe.n_missing > 0 && ` · ${probe.n_missing} unlabelled`}
              {probe.dropped_classes.length > 0 &&
                ` · dropped ${probe.dropped_classes
                  .map((entry) => `${entry.label} (${entry.count})`)
                  .join(", ")}`}
            </div>
          </button>
        );
      })}
    </div>
  );
};
