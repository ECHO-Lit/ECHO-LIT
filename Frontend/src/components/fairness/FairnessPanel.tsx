import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Loader2, Play, X } from "lucide-react";
import { useAnalysisJob } from "@/hooks/use-job-query";
import { InfoTooltip } from "@/components/analysis/InfoTooltip";
import { FR10_GLOSSARY } from "@/lib/fairnessGlossary";
import {
  cancelAllFairnessAnalyses, fetchGroupableColumns, submitFairnessAnalysis,
  type FairnessMetricId, type FairnessReport, type FairnessRequest,
} from "@/lib/fairness";
import { DesignBanner } from "./DesignBanner";
import { GroupErrorBarChart } from "./GroupErrorBarChart";
import { DisparityRadar } from "./DisparityRadar";
import { RepresentationGroundingCard } from "./RepresentationGroundingCard";
import { FairnessVerdictList } from "./FairnessVerdictList";
import { PhoneConfusionMatrix } from "./PhoneConfusionMatrix";

interface FairnessPanelProps {
  model?: string;
  dataset?: string;
  originalDataset?: string;
}

const ASR_METRICS: FairnessMetricId[] = ["wer", "cer"];
const CLASSIFICATION_METRICS: FairnessMetricId[] = ["accuracy", "macro_f1"];

/** FR-10: accent/language fairness dashboard. Unlike every other tab in
 * PredictionPanel, this one operates on a whole dataset + grouping column,
 * not on the single selected audio file -- see docs/FR10plan.md Part 1 S7. */
export function FairnessPanel({ model, dataset, originalDataset }: FairnessPanelProps) {
  const effectiveDataset = originalDataset && originalDataset !== "custom" ? originalDataset : dataset;

  const [groupingColumn, setGroupingColumn] = useState<string>("");
  const [minGroupSize, setMinGroupSize] = useState(8);
  const [minSpeakers, setMinSpeakers] = useState(2);
  const [includeRepresentation, setIncludeRepresentation] = useState(true);
  const [includeExplanations, setIncludeExplanations] = useState(true);
  const [chartMetric, setChartMetric] = useState<FairnessMetricId>("wer");

  const columnsQuery = useQuery({
    queryKey: ["fairness-groupable", effectiveDataset],
    queryFn: ({ signal }) => fetchGroupableColumns(effectiveDataset as string, signal),
    enabled: !!effectiveDataset && !effectiveDataset.startsWith("custom:__"),
    staleTime: 60_000,
  });

  const groupableColumns = columnsQuery.data?.columns.filter((c) => !c.is_stratum_only) ?? [];

  useEffect(() => {
    setGroupingColumn("");
  }, [effectiveDataset]);

  useEffect(() => {
    if (!groupingColumn && groupableColumns.length > 0) {
      setGroupingColumn(groupableColumns[0].column);
    }
  }, [groupableColumns, groupingColumn]);

  const job = useAnalysisJob<FairnessReport, void>(async () => {
    if (!effectiveDataset) throw new Error("Select a dataset first");
    if (!model) throw new Error("Select a model first");
    if (!groupingColumn) throw new Error("Select a column to group by");
    const body: FairnessRequest = {
      dataset: effectiveDataset, grouping_key: [groupingColumn], model, task: "auto",
      min_group_size: minGroupSize, min_speakers_per_group: minSpeakers,
      include_representation: includeRepresentation, include_explanations: includeExplanations,
      metrics: [...ASR_METRICS, ...CLASSIFICATION_METRICS, "grounding_lift", "attribution_entropy"],
    };
    return submitFairnessAnalysis(body);
  }, "fr10");

  // Separate from `job.cancel()`: that only knows about the ONE job this
  // panel's own React Query cache is currently tracking. If the panel ever
  // lost track of a running job (the tab-unmount/gcTime bug, opening a
  // second tab, a page reload mid-run), the backend job keeps going with no
  // local reference to cancel it. This walks the session's own job list
  // server-side instead, so it works regardless of what the UI remembers.
  const cancelAll = useMutation({
    mutationFn: () => cancelAllFairnessAnalyses(),
    onSuccess: ({ count }) => {
      toast.success(count > 0 ? `Cancelled ${count} running fairness job(s).` : "No running fairness jobs to cancel.");
      job.reset();
    },
    onError: (err: Error) => toast.error(err.message || "Failed to cancel fairness jobs."),
  });

  const progressPct = job.progress && job.progress.total > 0
    ? Math.round((job.progress.current / job.progress.total) * 100)
    : 0;

  const availableChartMetrics = useMemo(() => {
    if (!job.result) return ASR_METRICS;
    const fromDisparities = Array.from(new Set(job.result.disparities.map((d) => d.metric)));
    return fromDisparities.length > 0 ? fromDisparities : ASR_METRICS;
  }, [job.result]);

  useEffect(() => {
    if (job.result && !availableChartMetrics.includes(chartMetric)) {
      setChartMetric(availableChartMetrics[0]);
    }
  }, [job.result, availableChartMetrics, chartMetric]);

  if (!effectiveDataset || effectiveDataset === "custom") {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        Select a built-in or custom dataset (not just uploaded files) to run a fairness comparison.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 p-3">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Group comparison setup</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1.5">
            <Label className="text-xs flex items-center gap-1">
              Group by
              <InfoTooltip text="The metadata column that defines the groups being compared, e.g. native_language or accent." />
            </Label>
            <Select value={groupingColumn} onValueChange={setGroupingColumn} disabled={job.isRunning}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue placeholder={columnsQuery.isLoading ? "Loading columns…" : "Select a column"} />
              </SelectTrigger>
              <SelectContent>
                {groupableColumns.map((column) => (
                  <SelectItem key={column.column} value={column.column} className="text-xs">
                    {column.column} ({column.n_values} values{column.n_missing > 0 ? `, ${column.n_missing} missing` : ""})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {columnsQuery.data && groupableColumns.length === 0 && (
              <p className="text-xs text-destructive">
                No usable grouping column found in {columnsQuery.data.n_rows} rows of this dataset.
              </p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label className="text-xs flex items-center gap-1">
                Min group size
                <InfoTooltip text={FR10_GLOSSARY.minGroupSize} />
              </Label>
              <Input
                type="number" min={2} max={1000} value={minGroupSize} disabled={job.isRunning}
                onChange={(e) => setMinGroupSize(Math.max(2, Number(e.target.value) || 2))}
                className="h-8 text-xs"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs flex items-center gap-1">
                Min speakers
                <InfoTooltip text={FR10_GLOSSARY.minSpeakers} />
              </Label>
              <Input
                type="number" min={1} max={100} value={minSpeakers} disabled={job.isRunning}
                onChange={(e) => setMinSpeakers(Math.max(1, Number(e.target.value) || 1))}
                className="h-8 text-xs"
              />
            </div>
          </div>

          <div className="flex items-center justify-between rounded-md border border-border p-3">
            <label className="flex items-center gap-2 text-sm font-medium cursor-pointer">
              <Switch checked={includeRepresentation} onCheckedChange={setIncludeRepresentation} disabled={job.isRunning} />
              Representation comparison
            </label>
          </div>
          <div className="flex items-center justify-between rounded-md border border-border p-3">
            <label className="flex items-center gap-2 text-sm font-medium cursor-pointer">
              <Switch checked={includeExplanations} onCheckedChange={setIncludeExplanations} disabled={job.isRunning} />
              Explanation fairness (saliency grounding)
            </label>
            <span className="text-xs text-muted-foreground">slower: runs saliency per sampled item</span>
          </div>

          {job.error && <p className="text-xs text-destructive">{job.error}</p>}

          <div className="flex gap-2">
            <Button
              size="sm"
              disabled={job.isRunning || job.isSubmitting || !groupingColumn}
              onClick={() => job.start()}
            >
              {job.isRunning || job.isSubmitting ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <Play className="h-4 w-4 mr-1" />
              )}
              Run fairness analysis
            </Button>
            {job.isRunning && (
              <Button size="sm" variant="outline" onClick={() => job.cancel()}>
                <X className="h-4 w-4 mr-1" />
                Cancel
              </Button>
            )}
            <Button
              size="sm"
              variant="outline"
              className="text-destructive hover:text-destructive"
              disabled={cancelAll.isPending}
              onClick={() => cancelAll.mutate()}
              title="Cancel every running fairness job in this session, including any the panel lost track of"
            >
              {cancelAll.isPending ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <X className="h-4 w-4 mr-1" />
              )}
              Cancel all fairness jobs
            </Button>
          </div>

          {job.isRunning && job.progress && (
            <div className="space-y-1">
              <Progress value={progressPct} />
              <p className="text-xs text-muted-foreground">{job.progress.message}</p>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="space-y-4">
        {job.result ? (
          <>
            <DesignBanner design={job.result.design} caveats={job.result.caveats} />

            <Card>
              <CardHeader className="pb-2 flex-row items-center justify-between space-y-0">
                <CardTitle className="text-sm">Group performance</CardTitle>
                <Select value={chartMetric} onValueChange={(v) => setChartMetric(v as FairnessMetricId)}>
                  <SelectTrigger className="h-7 w-32 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {availableChartMetrics.map((m) => (
                      <SelectItem key={m} value={m} className="text-xs">{m}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </CardHeader>
              <CardContent>
                <GroupErrorBarChart
                  groups={job.result.groups}
                  excluded={job.result.excluded_groups}
                  disparities={job.result.disparities}
                  referenceGroup={job.result.reference_group}
                  metric={chartMetric}
                />
              </CardContent>
            </Card>

            {job.result.disparities.length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Multi-metric disparity</CardTitle>
                </CardHeader>
                <CardContent>
                  <DisparityRadar disparities={job.result.disparities} referenceGroup={job.result.reference_group} />
                </CardContent>
              </Card>
            )}

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Representation &amp; grounding</CardTitle>
              </CardHeader>
              <CardContent>
                <RepresentationGroundingCard representation={job.result.representation} groups={job.result.groups} />
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-4">
                <FairnessVerdictList verdicts={job.result.verdicts} />
              </CardContent>
            </Card>

            {job.result.dataset_extensions && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    Dataset-specific analysis
                    <Badge variant="outline" className="text-[10px]">{job.result.dataset_extensions.kind}</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-xs text-muted-foreground space-y-2">
                  {job.result.dataset_extensions.caveat && (
                    <p className="text-amber-600 dark:text-amber-400">{job.result.dataset_extensions.caveat}</p>
                  )}
                  {job.result.dataset_extensions.h1_substitution_gt_addition_gt_deletion !== undefined && (
                    <p>
                      Grounding AUROC ordering (substitution &gt; addition &gt; deletion):{" "}
                      <span className="font-medium">
                        {job.result.dataset_extensions.h1_substitution_gt_addition_gt_deletion ? "holds" : "does not hold"}
                      </span>
                    </p>
                  )}
                  {job.result.dataset_extensions.age_onset_dose_response && (
                    <pre className="whitespace-pre-wrap text-[11px] bg-muted/40 rounded p-2 overflow-auto max-h-40">
                      {JSON.stringify(job.result.dataset_extensions.age_onset_dose_response, null, 2)}
                    </pre>
                  )}
                  {job.result.dataset_extensions.phone_confusion_matrix && (
                    <div className="pt-2">
                      <PhoneConfusionMatrix matrix={job.result.dataset_extensions.phone_confusion_matrix} />
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </>
        ) : (
          <div className="flex h-[280px] items-center justify-center text-sm text-muted-foreground border border-dashed border-border rounded-md">
            {job.isRunning ? "Partitioning dataset and running per-group inference…" : "Run an analysis to see the fairness report."}
          </div>
        )}
      </div>
    </div>
  );
}
