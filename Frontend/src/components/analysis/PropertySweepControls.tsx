import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { RangeSlider } from "@/components/ui/range-slider";
import type { SweepConfig, SweepProperty } from "@/lib/linguisticAcoustic";

interface PropertyDefaults {
  property: SweepProperty;
  label: string;
  unit: string;
  min: number;
  max: number;
  defaultStart: number;
  defaultStop: number;
  step: number;
  formatValue: (value: number) => string;
  hasRepeats?: boolean;
}

export const SWEEP_DEFAULTS: PropertyDefaults[] = [
  {
    property: "pitch", label: "Pitch", unit: "semitones",
    min: -12, max: 12, defaultStart: -6, defaultStop: 6, step: 0.5,
    formatValue: (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}st`,
  },
  {
    property: "speed", label: "Speaking rate", unit: "×",
    min: 0.5, max: 2.0, defaultStart: 0.7, defaultStop: 1.4, step: 0.05,
    formatValue: (v) => `${v.toFixed(2)}×`,
  },
  {
    property: "noise", label: "Additive noise", unit: "dB SNR",
    min: -10, max: 60, defaultStart: 40, defaultStop: 0, step: 1,
    formatValue: (v) => `${v.toFixed(0)} dB`, hasRepeats: true,
  },
  {
    property: "time_mask", label: "Time masking", unit: "%",
    min: 0, max: 80, defaultStart: 0, defaultStop: 50, step: 1,
    formatValue: (v) => `${v.toFixed(0)}%`,
  },
  {
    property: "freq_mask", label: "Frequency masking", unit: "Hz width",
    min: 100, max: 8000, defaultStart: 500, defaultStop: 4000, step: 50,
    formatValue: (v) => `${v.toFixed(0)} Hz`,
  },
];

export interface SweepState {
  enabled: boolean;
  start: number;
  stop: number;
  steps: number;
  repeats: number;
}

export type SweepStateMap = Record<SweepProperty, SweepState>;

export function defaultSweepState(): SweepStateMap {
  return Object.fromEntries(
    SWEEP_DEFAULTS.map((def) => [
      def.property,
      {
        enabled: def.property === "pitch" || def.property === "speed" || def.property === "noise",
        start: def.defaultStart, stop: def.defaultStop, steps: 7, repeats: 3,
      } satisfies SweepState,
    ]),
  ) as SweepStateMap;
}

export function toSweepConfigs(state: SweepStateMap): SweepConfig[] {
  return SWEEP_DEFAULTS.filter((def) => state[def.property].enabled).map((def) => {
    const s = state[def.property];
    const config: SweepConfig = { property: def.property, start: s.start, stop: s.stop, steps: s.steps };
    if (def.hasRepeats) config.repeats = s.repeats;
    if (def.property === "freq_mask") config.band_low_hz = 0;
    return config;
  });
}

interface PropertySweepControlsProps {
  state: SweepStateMap;
  onChange: (state: SweepStateMap) => void;
  disabled?: boolean;
}

export function PropertySweepControls({ state, onChange, disabled }: PropertySweepControlsProps) {
  const update = (property: SweepProperty, patch: Partial<SweepState>) => {
    onChange({ ...state, [property]: { ...state[property], ...patch } });
  };

  return (
    <div className="space-y-4">
      {SWEEP_DEFAULTS.map((def) => {
        const s = state[def.property];
        return (
          <div key={def.property} className="rounded-md border border-border p-3 space-y-2">
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 text-sm font-medium cursor-pointer">
                <Checkbox
                  checked={s.enabled}
                  onCheckedChange={(checked) => update(def.property, { enabled: !!checked })}
                  disabled={disabled}
                />
                {def.label}
                <span className="text-xs text-muted-foreground font-normal">({def.unit})</span>
              </label>
            </div>

            {s.enabled && (
              <div className="space-y-3 pl-6">
                <RangeSlider
                  value={[s.start, s.stop]}
                  onValueChange={([start, stop]) => update(def.property, { start, stop })}
                  min={def.min}
                  max={def.max}
                  step={def.step}
                  disabled={disabled}
                  formatLabel={def.formatValue}
                />
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2 flex-1">
                    <span className="whitespace-nowrap">Steps: {s.steps}</span>
                    <Slider
                      value={[s.steps]}
                      onValueChange={([steps]) => update(def.property, { steps })}
                      min={2} max={15} step={1}
                      disabled={disabled}
                      className="flex-1"
                    />
                  </div>
                  {def.hasRepeats && (
                    <div className="flex items-center gap-2 flex-1">
                      <span className="whitespace-nowrap">Repeats: {s.repeats}</span>
                      <Slider
                        value={[s.repeats]}
                        onValueChange={([repeats]) => update(def.property, { repeats })}
                        min={1} max={10} step={1}
                        disabled={disabled}
                        className="flex-1"
                      />
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
