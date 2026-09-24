/**
 * The side-by-side result of a model/dataset comparison.
 *
 * Reading order is top-down: the headline is the aggregate metric difference
 * (the graded acceptance criterion), then per-side aggregates, then the aligned
 * per-item table where the actual disagreements live. Everything a reader needs
 * to make a differential diagnosis is one scroll.
 */

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowRight, Minus } from "lucide-react";

import {
  formatPrimary,
  type ComparisonResult,
  type ItemComparison,
  type ModelKind,
} from "@/lib/comparison";

const pct = (value: number | null) => (value == null ? "—" : `${(value * 100).toFixed(1)}%`);

/** Colour a per-item delta by whether B improved on A, given the metric's direction. */
function deltaTone(kind: ModelKind, delta: number | null): string {
  if (delta == null || Math.abs(delta) < 1e-9) return "text-muted-foreground";
  const bBetter = kind === "asr" ? delta < 0 : delta > 0;
  return bBetter ? "text-emerald-600 dark:text-emerald-500" : "text-orange-600 dark:text-orange-500";
}

function Headline({ result }: { result: ComparisonResult }) {
  const { a, b, primaryLabel, primaryDelta, betterSide, kind } = result;
  const winner = betterSide === "a" ? a : betterSide === "b" ? b : null;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Aggregate {primaryLabel.toLowerCase()}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-4">
          <SideStat side={a} label="A" primaryLabel={primaryLabel} highlight={betterSide === "a"} />
          <div className="flex flex-col items-center gap-1 text-center">
            {primaryDelta == null ? (
              <Minus className="h-5 w-5 text-muted-foreground" />
            ) : (
              <>
                <span className={`text-lg font-semibold tabular-nums ${deltaTone(kind, primaryDelta)}`}>
                  {primaryDelta >= 0 ? "+" : ""}
                  {(primaryDelta * 100).toFixed(1)} pts
                </span>
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground">difference</span>
              </>
            )}
          </div>
          <SideStat side={b} label="B" primaryLabel={primaryLabel} highlight={betterSide === "b"} />
        </div>

        <p className="mt-4 text-xs text-muted-foreground leading-relaxed">
          {primaryDelta == null ? (
            <>No ground truth was available on both sides, so an accuracy difference cannot be computed. The predictions are still shown side by side below.</>
          ) : winner ? (
            <>
              <span className="font-medium text-foreground">{winner.label}</span> has the better{" "}
              {primaryLabel.toLowerCase()} on this sample
              {result.mode === "models" ? " of shared files" : ""}
              {kind === "asr" ? " (lower is better)" : " (higher is better)"}.
              {result.agreementRate != null && (
                <> The two models gave the same answer on {(result.agreementRate * 100).toFixed(0)}% of items.</>
              )}
            </>
          ) : (
            <>The two sides are tied on {primaryLabel.toLowerCase()} for this sample.</>
          )}
        </p>
      </CardContent>
    </Card>
  );
}

function SideStat({
  side,
  label,
  primaryLabel,
  highlight,
}: {
  side: ComparisonResult["a"];
  label: "A" | "B";
  primaryLabel: string;
  highlight: boolean;
}) {
  return (
    <div className={`rounded-lg border px-3 py-2.5 ${highlight ? "border-emerald-500/50 bg-emerald-500/5" : "border-border"}`}>
      <div className="flex items-center gap-1.5">
        <Badge variant="outline" className="h-5 px-1.5 text-[10px]">{label}</Badge>
        <span className="truncate text-xs font-medium" title={side.label}>{side.label}</span>
      </div>
      <div className="mt-1.5 text-2xl font-semibold tabular-nums">{pct(side.primary)}</div>
      <div className="text-[11px] text-muted-foreground">
        {primaryLabel.toLowerCase()} · {side.withTruth}/{side.itemCount} scored
      </div>
    </div>
  );
}

function MetricCell({ kind, value }: { kind: ModelKind; value: number | null }) {
  if (value == null) return <span className="text-muted-foreground">—</span>;
  return <span className="tabular-nums">{kind === "asr" ? `${(value * 100).toFixed(0)}% WER` : value ? "correct" : "wrong"}</span>;
}

function AlignedTable({ result }: { result: ComparisonResult }) {
  const { kind, items, a, b } = result;
  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-border bg-muted/40 text-left text-[10px] uppercase tracking-wide text-muted-foreground">
            <th className="px-3 py-2 font-medium">File</th>
            <th className="px-3 py-2 font-medium">Ground truth</th>
            <th className="px-3 py-2 font-medium">A · {a.label}</th>
            <th className="px-3 py-2 font-medium">B · {b.label}</th>
            <th className="px-3 py-2 font-medium text-right">Δ</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item, index) => (
            <ItemRow key={`${item.filename}-${index}`} kind={kind} item={item} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ItemRow({ kind, item }: { kind: ModelKind; item: ItemComparison }) {
  return (
    <tr className={`border-b border-border/60 last:border-0 ${item.agree === false ? "bg-orange-500/[0.04]" : ""}`}>
      <td className="max-w-[140px] truncate px-3 py-2 font-mono text-[11px]" title={item.filename}>{item.filename}</td>
      <td className="max-w-[200px] px-3 py-2 text-muted-foreground">{item.groundTruth ?? <span className="italic">none</span>}</td>
      <td className="max-w-[220px] px-3 py-2">
        {item.a ? (
          <div>
            <div className="truncate" title={item.a.prediction}>{item.a.prediction || <span className="italic text-muted-foreground">(empty)</span>}</div>
            <div className="text-[10px] text-muted-foreground"><MetricCell kind={kind} value={item.a.metric} /></div>
          </div>
        ) : <span className="text-muted-foreground">—</span>}
      </td>
      <td className="max-w-[220px] px-3 py-2">
        {item.b ? (
          <div>
            <div className="truncate" title={item.b.prediction}>{item.b.prediction || <span className="italic text-muted-foreground">(empty)</span>}</div>
            <div className="text-[10px] text-muted-foreground"><MetricCell kind={kind} value={item.b.metric} /></div>
          </div>
        ) : <span className="text-muted-foreground">—</span>}
      </td>
      <td className={`px-3 py-2 text-right tabular-nums ${deltaTone(kind, item.delta)}`}>
        {item.delta == null ? "—" : `${item.delta >= 0 ? "+" : ""}${(item.delta * 100).toFixed(0)}`}
      </td>
    </tr>
  );
}

export function ComparisonView({ result }: { result: ComparisonResult }) {
  return (
    <div className="space-y-4">
      <Headline result={result} />

      <div>
        <div className="mb-2 flex items-center gap-2 text-sm font-medium">
          Per-item predictions
          {result.mode === "models" ? (
            <span className="flex items-center gap-1 text-xs font-normal text-muted-foreground">
              aligned by file <ArrowRight className="h-3 w-3" /> disagreements highlighted
            </span>
          ) : (
            <span className="text-xs font-normal text-muted-foreground">two datasets, listed in parallel</span>
          )}
        </div>
        <AlignedTable result={result} />
      </div>

      <p className="text-[11px] text-muted-foreground">
        This slice compares predictions and error metrics. Embedding and saliency comparison are
        described in the same use case but are not part of this view yet.
      </p>
    </div>
  );
}
