import { AlertTriangle, CheckCircle2, HelpCircle } from "lucide-react";
import { InfoTooltip } from "@/components/analysis/InfoTooltip";
import { FR10_GLOSSARY } from "@/lib/fairnessGlossary";
import { DESIGN_LABEL, type FairnessReport } from "@/lib/fairness";

interface DesignBannerProps {
  design: FairnessReport["design"];
  caveats: string[];
}

const ICON = {
  matched: <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0" />,
  partially_matched: <HelpCircle className="h-4 w-4 text-amber-500 shrink-0" />,
  unmatched: <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0" />,
};

/** Renders the single most important fact about a fairness report before any
 * number: whether groups read matched content (SRS-driven; see
 * docs/FR10plan.md Part 1 S2.4). A disparity number without this context is
 * a misleading artifact. */
export function DesignBanner({ design, caveats }: DesignBannerProps) {
  return (
    <div className="rounded-md border border-border bg-muted/40 p-3 space-y-2">
      <div className="flex items-center gap-2">
        {ICON[design.type]}
        <span className="text-sm font-medium">{DESIGN_LABEL[design.type]}</span>
        <InfoTooltip text={FR10_GLOSSARY.design} />
        {typeof design.jaccard === "number" && (
          <span className="text-xs text-muted-foreground ml-auto">
            {design.n_shared_content} / {design.n_total_content} shared content items
            {" "}({Math.round(design.jaccard * 100)}% overlap)
          </span>
        )}
      </div>
      {caveats.length > 0 && (
        <ul className="text-xs text-muted-foreground space-y-1 list-disc pl-5">
          {caveats.map((caveat, index) => (
            <li key={index}>{caveat}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
