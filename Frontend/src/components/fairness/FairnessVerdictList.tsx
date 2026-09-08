import { AlertCircle, HelpCircle, MinusCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { InfoTooltip } from "@/components/analysis/InfoTooltip";
import { FR10_GLOSSARY } from "@/lib/fairnessGlossary";
import { metricLabel, type FairnessVerdict } from "@/lib/fairness";

const RANK: Record<string, number> = { disparity_detected: 0, inconclusive: 1, no_evidence_of_disparity: 2 };

const ICON: Record<string, JSX.Element> = {
  disparity_detected: <AlertCircle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />,
  inconclusive: <HelpCircle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />,
  no_evidence_of_disparity: <MinusCircle className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />,
};

interface FairnessVerdictListProps {
  verdicts: FairnessVerdict[];
}

/** Ranked verdict cards: disparities first, then inconclusive, then
 * no-evidence, matching docs/FR10plan.md Part 1 S4.6's severity order.
 * "no_evidence_of_disparity" renders in neutral gray, never a green check --
 * that evidence does not support a certification of fairness. */
export function FairnessVerdictList({ verdicts }: FairnessVerdictListProps) {
  if (verdicts.length === 0) {
    return <p className="text-xs text-muted-foreground">No comparisons were computed for the requested metrics.</p>;
  }
  const sorted = [...verdicts].sort((a, b) => (RANK[a.verdict] ?? 3) - (RANK[b.verdict] ?? 3));

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1.5">
        <span className="text-xs font-medium text-muted-foreground">Verdicts</span>
        <InfoTooltip text={FR10_GLOSSARY.verdict} />
      </div>
      {sorted.map((verdict, index) => (
        <div key={index} className="flex items-start gap-2 rounded-md border border-border p-2.5">
          {ICON[verdict.verdict] ?? ICON.inconclusive}
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-medium">{verdict.group}</span>
              {verdict.metric !== "*" && (
                <Badge variant="outline" className="text-[10px] px-1.5 py-0">{metricLabel(verdict.metric)}</Badge>
              )}
              <Badge
                variant={verdict.verdict === "disparity_detected" ? "destructive" : "secondary"}
                className="text-[10px] px-1.5 py-0"
              >
                {verdict.severity}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{verdict.message}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
