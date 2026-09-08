import Plot from "react-plotly.js";
import { useIsDarkMode } from "@/hooks/useIsDarkMode";
import { InfoTooltip } from "@/components/analysis/InfoTooltip";
import { FR10_GLOSSARY } from "@/lib/fairnessGlossary";
import type { PhoneConfusionMatrix as PhoneConfusionMatrixType } from "@/lib/fairness";

interface PhoneConfusionMatrixProps {
  matrix: PhoneConfusionMatrixType;
}

// Same fixed categorical order as DisparityRadar.tsx, reused here so a given
// L1 group reads as the same color across every chart in the panel.
const COLORS = ["#3b82f6", "#f97316", "#10b981", "#a855f7", "#ef4444", "#0ea5e9"];

const TOP_N = 5;
// Below this many saliency-scored intervals, a phone-class bar is rendered
// muted rather than hidden -- thin evidence should read as thin, not absent.
const MIN_SALIENCY_N = 5;

/** FR-10 S6.2: canonical->perceived phone confusion, per L1 group, from the
 * L2-ARCTIC human annotations, weighted by how much the model's saliency
 * attends to each annotated interval. Both axes are human-annotated (no
 * model-side phone recognizer exists in this system); the saliency ratio is
 * the only model signal available, and only ~4-5 confusion PAIRS have
 * support in all three L1 groups, so this renders three tiers rather than a
 * dense heatmap: top pairs per group, the pairs that are directly
 * comparable across groups, and attribution aggregated to broad phone
 * classes (the statistically viable cross-group comparison). */
export function PhoneConfusionMatrix({ matrix }: PhoneConfusionMatrixProps) {
  const isDark = useIsDarkMode();

  if (matrix.status !== "ok" || !matrix.cells || !matrix.groups) {
    return <p className="text-xs text-muted-foreground">Phone confusion data unavailable for this run.</p>;
  }

  const groups = matrix.groups;
  const colorByGroup = new Map(groups.map((g, i) => [g, COLORS[i % COLORS.length]]));

  const topByGroup = new Map(
    groups.map((group) => {
      const cellsForGroup = matrix.cells!
        .filter((c) => c.group === group)
        .reduce((acc, c) => {
          // Collapse error_type back out -- the pair is what a reader scans for.
          const key = `${c.canonical}→${c.perceived}`;
          acc.set(key, (acc.get(key) ?? 0) + c.n);
          return acc;
        }, new Map<string, number>());
      const ranked = Array.from(cellsForGroup.entries()).sort((a, b) => b[1] - a[1]).slice(0, TOP_N);
      return [group, ranked] as const;
    }),
  );
  const maxTopN = Math.max(1, ...Array.from(topByGroup.values()).flatMap((rows) => rows.map(([, n]) => n)));

  const sharedPairs = (matrix.shared_pairs ?? []).slice(0, 8);

  const classNames = Array.from(
    new Set(Object.values(matrix.attribution_by_phone_class ?? {}).flatMap((byClass) => Object.keys(byClass))),
  ).sort();
  const attributionTraces = groups.map((group, i) => {
    const byClass = matrix.attribution_by_phone_class?.[group] ?? {};
    return {
      type: "bar",
      name: group,
      x: classNames,
      y: classNames.map((c) => byClass[c]?.mean_saliency_ratio ?? 0),
      marker: {
        color: COLORS[i % COLORS.length],
        opacity: classNames.map((c) => ((byClass[c]?.n_saliency ?? 0) >= MIN_SALIENCY_N ? 1 : 0.3)),
      },
      text: classNames.map((c) => (byClass[c] ? `n=${byClass[c].n}, n_saliency=${byClass[c].n_saliency}` : "")),
      hoverinfo: "x+y+text+name",
    } as any;
  });

  return (
    <div className="space-y-4">
      <p className="text-[11px] text-muted-foreground">
        n={matrix.n_intervals} deduped intervals ({matrix.n_intervals_raw} raw rows before dedup; the
        sub_/add_/del_ filename prefixes are a sampling stratum over shared recordings, not distinct audio).
      </p>

      <div>
        <div className="flex items-center gap-1.5 mb-1.5">
          <span className="text-xs font-medium text-muted-foreground">Top confusions per group</span>
          <InfoTooltip text={FR10_GLOSSARY.confusionPair} />
        </div>
        <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${groups.length}, minmax(0, 1fr))` }}>
          {groups.map((group) => (
            <div key={group} className="space-y-1">
              <div className="flex items-center gap-1.5 text-xs font-medium">
                <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: colorByGroup.get(group) }} />
                {group}
              </div>
              {(topByGroup.get(group) ?? []).map(([pair, n]) => (
                <div key={pair} className="text-[11px]">
                  <div className="flex items-center justify-between">
                    <span className="font-mono">{pair}</span>
                    <span className="text-muted-foreground tabular-nums">{n}</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-muted/50 overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${(n / maxTopN) * 100}%`, backgroundColor: colorByGroup.get(group) }}
                    />
                  </div>
                </div>
              ))}
              {(topByGroup.get(group) ?? []).length === 0 && (
                <p className="text-[11px] text-muted-foreground">No annotated intervals.</p>
              )}
            </div>
          ))}
        </div>
      </div>

      {sharedPairs.length > 0 && (
        <div>
          <div className="flex items-center gap-1.5 mb-1.5">
            <span className="text-xs font-medium text-muted-foreground">Directly comparable pairs</span>
            <InfoTooltip text="Confusion pairs that occur in 2 or more L1 groups -- the only cells with enough cross-group support to compare a rate directly, out of ~150-170 distinct pairs total." />
          </div>
          <div className="rounded-md border border-border divide-y divide-border">
            {sharedPairs.map((pair) => (
              <div key={`${pair.canonical}-${pair.perceived}`} className="flex items-center justify-between px-2.5 py-1.5 text-xs">
                <span className="font-mono">{pair.canonical}→{pair.perceived}</span>
                <span className="text-muted-foreground tabular-nums">
                  {groups
                    .filter((g) => pair.n_by_group[g] != null)
                    .map((g) => `${g} ${pair.n_by_group[g]}`)
                    .join(" · ")}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {classNames.length > 0 && (
        <div>
          <div className="flex items-center gap-1.5 mb-1">
            <span className="text-xs font-medium text-muted-foreground">Model attribution by phone class</span>
            <InfoTooltip text={FR10_GLOSSARY.saliencyRatio} />
          </div>
          <div className="h-[220px]">
            <Plot
              data={attributionTraces}
              layout={{
                autosize: true,
                margin: { l: 40, r: 10, t: 10, b: 45 },
                barmode: "group",
                xaxis: { tickfont: { size: 9 } },
                yaxis: { title: "saliency ratio", showgrid: true, gridcolor: "hsl(var(--border))", tickfont: { size: 9 } },
                plot_bgcolor: "transparent",
                paper_bgcolor: "transparent",
                showlegend: true,
                legend: { orientation: "h", y: -0.35, font: { size: 9 } },
                font: { size: 10, color: "hsl(var(--foreground))" },
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%", height: "100%" }}
            />
          </div>
          <p className="text-[11px] text-muted-foreground mt-1">
            Faded bars have fewer than {MIN_SALIENCY_N} saliency-scored intervals -- thin evidence, not a small
            effect.
          </p>
        </div>
      )}

      {matrix.caveat && <p className="text-[11px] text-amber-600 dark:text-amber-400">{matrix.caveat}</p>}
    </div>
  );
}
