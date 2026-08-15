/**
 * Layer-wise representation analysis: controls, run orchestration, and the three
 * views.
 *
 * The panel is only ~25% of the viewport wide, so the charts are also offered
 * full-width in a dialog. Everything else lives in the hook (`use-layer-probes`)
 * and the contract module (`lib/probes`); this file is composition and controls.
 */

import { useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Maximize2, Play, X } from "lucide-react";

import { ChartCard } from "../eda/ChartCard";
import { LayerProfileChart } from "./LayerProfileChart";
import { LayerPropertyHeatmap } from "./LayerPropertyHeatmap";
import { LayerProbeSummary } from "./LayerProbeSummary";
import { ProbeConfusionMatrix } from "./ProbeConfusionMatrix";
import {
  DEFAULT_PROBE_SETTINGS,
  useLayerProbes,
  type ProbeSettings,
} from "@/hooks/use-layer-probes";
import {
  NOISE_PROPERTY,
  availableProperties,
  extractProperties,
  labelledCounts,
  propertyLabel,
} from "@/lib/probes";

interface LayerProbePanelProps {
  model: string;
  dataset: string;
  availableFiles: string[];
}

const PROFILE_EXPLANATION =
  "Each line is one property's cross-validated probe accuracy at each encoder layer. The matching dotted line is that property's majority baseline — the share of its largest class. Accuracy at or below the dotted line means NO information was found, however high the number looks. A line that peaks early and falls is information the model is discarding as it goes deeper; a line that rises with depth is information it is building.";

const HEATMAP_EXPLANATION =
  "Selectivity = probe accuracy minus the accuracy of the same probe trained on shuffled labels. High (blue) means the model genuinely encodes the property at that depth. At or below zero (grey to red) the probe was only memorising, so there is nothing there. The ringed cell is each property's peak.";

const CONFUSION_EXPLANATION =
  "Which classes the probe confuses at the property's best layer, row-normalised so classes of different size stay comparable. A bright diagonal means clean separation; a bright off-diagonal block means the model groups those classes together.";

export const LayerProbePanel = ({ model, dataset, availableFiles }: LayerProbePanelProps) => {
  const [selected, setSelected] = useState<string[]>([]);
  const [settings, setSettings] = useState<ProbeSettings>(DEFAULT_PROBE_SETTINGS);
  const [focused, setFocused] = useState<string | null>(null);

  const { metadata, metadataError, result, status, error, progress, fromCache, run, cancel } =
    useLayerProbes({ dataset, model, availableFiles });

  const offered = useMemo(() => availableProperties(metadata), [metadata]);

  // Default to everything the dataset offers, and re-derive when the dataset
  // changes -- a property from the previous dataset would produce all-null labels.
  useEffect(() => {
    setSelected(offered.map((property) => property.key));
    setFocused(null);
  }, [offered]);

  const counts = useMemo(() => {
    if (!metadata.length || !selected.length) return {};
    return labelledCounts(extractProperties(metadata, availableFiles, selected));
  }, [metadata, availableFiles, selected]);

  // Colour is assigned by position in this list, so it must be stable and must
  // match what the charts and the summary use.
  const drawn = useMemo(() => {
    if (!result) return [];
    const keys = selected.filter((key) => result.properties[key]);
    return result.properties[NOISE_PROPERTY] ? [...keys, NOISE_PROPERTY] : keys;
  }, [result, selected]);

  const toggle = (key: string) =>
    setSelected((previous) =>
      previous.includes(key) ? previous.filter((entry) => entry !== key) : [...previous, key],
    );

  const isRunning = status === "running";
  const canRun = availableFiles.length >= 2 && selected.length > 0 && !isRunning;
  const focusedProbe = focused && result ? result.properties[focused] : null;

  const charts = (
    <>
      <ChartCard title="Layer profile — accuracy by depth" explanation={PROFILE_EXPLANATION}>
        {result && <LayerProfileChart result={result} properties={drawn} />}
      </ChartCard>
      <ChartCard title="Selectivity by property and layer" explanation={HEATMAP_EXPLANATION}>
        {result && <LayerPropertyHeatmap result={result} properties={drawn} />}
      </ChartCard>
      {result && focusedProbe && focusedProbe.best_layer !== null && (
        <ChartCard
          title={`Confusion — ${propertyLabel(focused!)} at ${result.layer_names[focusedProbe.best_layer]}`}
          explanation={CONFUSION_EXPLANATION}
        >
          <ProbeConfusionMatrix
            probe={focusedProbe}
            propertyKey={focused!}
            layerName={result.layer_names[focusedProbe.best_layer]}
          />
        </ChartCard>
      )}
    </>
  );

  return (
    <div className="space-y-3">
      {availableFiles.length < 2 && (
        <div className="text-xs text-muted-foreground flex items-center gap-2 p-3 bg-muted/50 rounded-md border border-border">
          <div className="w-2 h-2 bg-muted-foreground rounded-full" />
          Layer probing compares files against each other — select a dataset with at least two.
        </div>
      )}
      {metadataError && (
        <div className="text-xs text-destructive p-3 bg-destructive/5 rounded-md border border-destructive/20">
          {metadataError}
        </div>
      )}

      {offered.length > 0 && (
        <div className="space-y-2">
          <div className="text-[11px] font-medium text-foreground">Properties to probe</div>
          <div className="grid grid-cols-2 gap-1.5">
            {offered.map((property) => (
              <label
                key={property.key}
                className="flex items-center gap-1.5 text-[11px] text-foreground cursor-pointer"
              >
                <Checkbox
                  checked={selected.includes(property.key)}
                  onCheckedChange={() => toggle(property.key)}
                  disabled={isRunning}
                  className="h-3.5 w-3.5"
                />
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="truncate">{property.label}</span>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs text-xs">{property.meaning}</TooltipContent>
                </Tooltip>
                {counts[property.key] !== undefined && (
                  <span className="text-muted-foreground ml-auto shrink-0">
                    {counts[property.key]}
                  </span>
                )}
              </label>
            ))}
          </div>
          <div className="text-[10px] text-muted-foreground">
            Numbers show how many of the {availableFiles.length} files carry that annotation.
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <Label className="text-[10px] text-muted-foreground">Probe</Label>
          <Select
            value={settings.probe}
            onValueChange={(value: ProbeSettings["probe"]) =>
              setSettings((previous) => ({ ...previous, probe: value }))
            }
            disabled={isRunning}
          >
            <SelectTrigger className="h-7 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="logreg">Logistic regression</SelectItem>
              <SelectItem value="linear_svm">Linear SVM</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label className="text-[10px] text-muted-foreground">CV folds</Label>
          <Select
            value={String(settings.cvFolds)}
            onValueChange={(value) =>
              setSettings((previous) => ({ ...previous, cvFolds: Number(value) }))
            }
            disabled={isRunning}
          >
            <SelectTrigger className="h-7 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[3, 5, 10].map((folds) => (
                <SelectItem key={folds} value={String(folds)}>
                  {folds}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label className="text-[10px] text-muted-foreground">Min class size</Label>
          <Select
            value={String(settings.minClassCount)}
            onValueChange={(value) =>
              setSettings((previous) => ({ ...previous, minClassCount: Number(value) }))
            }
            disabled={isRunning}
          >
            <SelectTrigger className="h-7 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[2, 3, 5, 10].map((size) => (
                <SelectItem key={size} value={String(size)}>
                  {size}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label className="text-[10px] text-muted-foreground">Projection</Label>
          <Select
            value={String(settings.projectDims)}
            onValueChange={(value) =>
              setSettings((previous) => ({ ...previous, projectDims: Number(value) }))
            }
            disabled={isRunning}
          >
            <SelectTrigger className="h-7 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="0">Off (full width)</SelectItem>
              <SelectItem value="64">64 dims</SelectItem>
              <SelectItem value="128">128 dims</SelectItem>
              <SelectItem value="256">256 dims</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <Switch
            id="probe-control"
            checked={settings.includeControl}
            onCheckedChange={(checked) =>
              setSettings((previous) => ({ ...previous, includeControl: checked }))
            }
            disabled={isRunning}
          />
          <Tooltip>
            <TooltipTrigger asChild>
              <Label htmlFor="probe-control" className="text-[10px] cursor-pointer">
                Control probe
              </Label>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs text-xs">
              Trains a second probe on shuffled labels at every layer. Without it there is no way to
              tell a real finding from a probe memorising the training rows. Doubles the training
              time; leave it on unless you are only exploring.
            </TooltipContent>
          </Tooltip>
        </div>
        <div className="flex items-center gap-1.5">
          <Switch
            id="probe-noise"
            checked={settings.noiseSnrDb !== null}
            onCheckedChange={(checked) =>
              setSettings((previous) => ({ ...previous, noiseSnrDb: checked ? 10 : null }))
            }
            disabled={isRunning}
          />
          <Tooltip>
            <TooltipTrigger asChild>
              <Label htmlFor="probe-noise" className="text-[10px] cursor-pointer">
                Noise probe
              </Label>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs text-xs">
              Runs every clip a second time with white noise added at 10 dB SNR and adds a
              clean-vs-noisy property. Doubles the extraction time and cannot reuse the clean-only
              cache.
            </TooltipContent>
          </Tooltip>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Button size="sm" className="h-7 text-xs flex-1" disabled={!canRun} onClick={() => run(selected, settings)}>
          <Play className="h-3 w-3 mr-1" />
          {result ? "Re-run probes" : "Run layer probes"}
        </Button>
        {isRunning && (
          <Button size="sm" variant="secondary" className="h-7 text-xs" onClick={cancel}>
            <X className="h-3 w-3" />
          </Button>
        )}
        {result && (
          <Dialog>
            <DialogTrigger asChild>
              <Button size="sm" variant="secondary" className="h-7 w-7 p-0">
                <Maximize2 className="h-3 w-3" />
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-5xl max-h-[90vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle className="text-sm">
                  Layer-wise representation analysis — {model} on {dataset}
                </DialogTitle>
              </DialogHeader>
              <div className="space-y-3">{charts}</div>
            </DialogContent>
          </Dialog>
        )}
      </div>

      {isRunning && (
        <div className="space-y-1">
          <Progress
            value={progress.total ? (progress.current / progress.total) * 100 : 0}
            className="h-1"
          />
          <div className="text-[10px] text-muted-foreground">
            {progress.message || "Working"}
            {progress.total > 0 && ` — ${progress.current}/${progress.total}`}
          </div>
        </div>
      )}

      {error && (
        <div className="text-xs text-destructive p-3 bg-destructive/5 rounded-md border border-destructive/20">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-3">
          <div className="flex items-center gap-1.5 flex-wrap">
            <Badge variant="outline" className="text-[10px]">
              {result.num_layers} layers · {result.hidden_dim}d
              {result.projected_dim !== result.hidden_dim && ` → ${result.projected_dim}d`}
            </Badge>
            <Badge variant="outline" className="text-[10px]">
              {result.n_files} files
            </Badge>
            {fromCache && (
              <Badge variant="outline" className="text-[10px]">
                cached
              </Badge>
            )}
          </div>
          {charts}
          <LayerProbeSummary
            result={result}
            properties={drawn}
            selectedProperty={focused}
            onSelectProperty={(key) => setFocused((previous) => (previous === key ? null : key))}
          />
        </div>
      )}
    </div>
  );
};
