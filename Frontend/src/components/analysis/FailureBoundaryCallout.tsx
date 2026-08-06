import { Badge } from "@/components/ui/badge";
import { AlertTriangle, TrendingDown, HelpCircle, Waves } from "lucide-react";
import { InfoTooltip } from "./InfoTooltip";
import { FR7_GLOSSARY } from "@/lib/fr7Glossary";
import type { SensitivityProfile } from "@/lib/linguisticAcoustic";

const VERDICT_META: Record<
  SensitivityProfile["verdict"],
  { label: string; icon: typeof Waves; className: string }
> = {
  linguistically_driven: {
    label: "Linguistically driven",
    icon: Waves,
    className: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/30",
  },
  mixed: {
    label: "Mixed influence",
    icon: TrendingDown,
    className: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30",
  },
  acoustically_dominated: {
    label: "Acoustically dominated",
    icon: AlertTriangle,
    className: "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/30",
  },
  inconclusive: {
    label: "Inconclusive",
    icon: HelpCircle,
    className: "bg-muted text-muted-foreground border-border",
  },
};

interface FailureBoundaryCalloutProps {
  profile: SensitivityProfile;
}

export function FailureBoundaryCallout({ profile }: FailureBoundaryCalloutProps) {
  const meta = VERDICT_META[profile.verdict];
  const Icon = meta.icon;

  return (
    <div className={`rounded-md border p-3 space-y-2 ${meta.className}`}>
      <div className="flex items-center gap-2">
        <Icon className="h-4 w-4 shrink-0" />
        <span className="font-semibold text-sm">{meta.label}</span>
        <InfoTooltip text={FR7_GLOSSARY.verdict} />
        {profile.dominant_property && (
          <Badge variant="outline" className="text-xs">
            dominant: {profile.dominant_property}
          </Badge>
        )}
      </div>

      {profile.verdict !== "inconclusive" ? (
        <div className="text-xs grid grid-cols-[1fr_auto_auto] items-center gap-x-2 gap-y-1">
          <span>Acoustic influence</span>
          <InfoTooltip text={FR7_GLOSSARY.acousticInfluence} />
          <span className="font-mono text-right">{profile.acoustic_influence.toFixed(2)}</span>
          <span>Linguistic robustness</span>
          <InfoTooltip text={FR7_GLOSSARY.linguisticRobustness} />
          <span className="font-mono text-right">{profile.linguistic_robustness.toFixed(2)}</span>
          <span>Relative to word removal</span>
          <InfoTooltip text={FR7_GLOSSARY.relativeToWordRemoval} />
          <span className="font-mono text-right">{profile.relative_to_lexical_destruction.toFixed(2)}</span>
        </div>
      ) : (
        <p className="text-xs">{profile.reason ?? "No property could be isolated for this input."}</p>
      )}

      {profile.evidence.length > 0 && (
        <ul className="text-xs space-y-1 list-disc list-inside opacity-90">
          {profile.evidence.map((line, index) => (
            <li key={index}>{line}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
