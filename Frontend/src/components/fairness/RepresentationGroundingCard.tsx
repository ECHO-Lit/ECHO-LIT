import { InfoTooltip } from "@/components/analysis/InfoTooltip";
import { FR10_GLOSSARY } from "@/lib/fairnessGlossary";
import type { FairnessGroupReport, FairnessRepresentation } from "@/lib/fairness";

interface RepresentationGroundingCardProps {
  representation: FairnessRepresentation;
  groups: FairnessGroupReport[];
}

function Stat({ label, value, tooltip }: { label: string; value: string; tooltip?: string }) {
  return (
    <div className="flex items-center justify-between text-xs py-1">
      <span className="text-muted-foreground flex items-center gap-1">
        {label}
        {tooltip && <InfoTooltip text={tooltip} />}
      </span>
      <span className="font-medium tabular-nums">{value}</span>
    </div>
  );
}

/** Section 4.3's 2x2 reading (separable x disparate) collapsed into one
 * interpretation line, plus the raw silhouette/leakage numbers and per-group
 * grounding means so a researcher can check the reading themselves. */
export function RepresentationGroundingCard({ representation, groups }: RepresentationGroundingCardProps) {
  const explainedGroups = groups.filter((g) => g.grounding.status === "ok");

  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center gap-1.5 mb-1">
          <span className="text-xs font-medium text-muted-foreground">Representation</span>
          <InfoTooltip text={FR10_GLOSSARY.silhouette} />
        </div>
        {representation.status === "ok" ? (
          <div className="rounded-md border border-border p-2.5">
            <Stat
              label="Silhouette by group"
              value={representation.silhouette_by_group_label!.toFixed(3)}
              tooltip={FR10_GLOSSARY.silhouette}
            />
            <Stat
              label="  vs. permutation null"
              value={`${representation.permutation_null!.mean.toFixed(3)} ± ${representation.permutation_null!.sd.toFixed(3)} (z=${representation.silhouette_z!.toFixed(1)})`}
            />
            {representation.silhouette_by_speaker != null && (
              <Stat label="Silhouette by speaker" value={representation.silhouette_by_speaker.toFixed(3)} />
            )}
            {representation.leakage?.status === "ok" && (
              <>
                <Stat
                  label="Group leakage (k-NN)"
                  value={`${(representation.leakage.balanced_accuracy! * 100).toFixed(1)}% (chance ${(representation.leakage.chance! * 100).toFixed(0)}%)`}
                  tooltip={FR10_GLOSSARY.leakage}
                />
                <Stat label="  leakage lift" value={`${representation.leakage.leakage_lift!.toFixed(2)}x`} />
              </>
            )}
            {representation.leakage?.status === "skipped" && (
              <p className="text-[11px] text-muted-foreground mt-1">Leakage skipped: {representation.leakage.reason}</p>
            )}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            {representation.status === "insufficient_data"
              ? "Not enough embeddings to compare representations."
              : "Representational comparison unavailable for this model/run."}
          </p>
        )}
      </div>

      <div>
        <div className="flex items-center gap-1.5 mb-1">
          <span className="text-xs font-medium text-muted-foreground">Explanation grounding</span>
          <InfoTooltip text={FR10_GLOSSARY.groundingLift} />
        </div>
        {explainedGroups.length > 0 ? (
          <div className="rounded-md border border-border divide-y divide-border">
            {explainedGroups.map((group) => (
              <div key={group.label} className="flex items-center justify-between px-2.5 py-1.5 text-xs">
                <span className="font-medium">{group.label}</span>
                <span className="text-muted-foreground tabular-nums">
                  lift {group.grounding.grounding_lift?.toFixed(2)} · entropy {group.grounding.attribution_entropy?.toFixed(2)} · n={group.grounding.n_explained}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">No explanations were computed for this run.</p>
        )}
      </div>
    </div>
  );
}
