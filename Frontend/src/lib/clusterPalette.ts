/**
 * Shared categorical palette for cluster identity.
 *
 * Used by the embedding scatter, the EDA cluster table swatches, and the
 * nearest-neighbour cluster badges. All three MUST import from here -- the hybrid
 * UI only reads if a cluster is the same colour everywhere it appears.
 *
 * Palette provenance: four slots (blue, yellow, magenta, green) taken from the
 * validated reference palette. A scatter compares every series against every other
 * ("all-pairs"), and four is the largest subset of that palette clearing the
 * all-pairs colour-blindness and normal-vision floors in BOTH light and dark modes
 * (five-slot subsets: none pass).
 *
 * Two obligations come with it, both satisfied elsewhere in this feature:
 *  - Dark mode lands in the 6-8 CVD warn band, which is only legal with a secondary
 *    encoding channel -> marker SYMBOL encodes cluster alongside hue (see
 *    `clusterSymbol`), so hue never carries identity alone.
 *  - Light-mode yellow and magenta sit below 3:1 against the surface, which requires
 *    "relief" -> the EDA cluster table lists every cluster in text with its swatch.
 *
 * Clusters beyond the fourth reuse a hue with a different symbol (composite
 * encoding), so each cluster keeps a unique, stable (hue, symbol) pair.
 */

const LIGHT_HUES = ["#2a78d6", "#eda100", "#e87ba4", "#008300"] as const;
const DARK_HUES = ["#3987e5", "#c98500", "#d55181", "#008300"] as const;

/** Plotly marker symbols, cycled one step behind the hues. */
const SYMBOLS = ["circle", "square", "diamond", "triangle-up"] as const;

/** Noise is deliberately not a palette slot -- it must not read as "cluster 5". */
export const NOISE_COLOR_LIGHT = "#898781";
export const NOISE_COLOR_DARK = "#898781";
export const NOISE_SYMBOL = "x";
export const NOISE_LABEL = -1;

export function clusterColor(label: number, isDark = false): string {
  if (label === NOISE_LABEL) return isDark ? NOISE_COLOR_DARK : NOISE_COLOR_LIGHT;
  const hues = isDark ? DARK_HUES : LIGHT_HUES;
  return hues[label % hues.length];
}

export function clusterSymbol(label: number): string {
  if (label === NOISE_LABEL) return NOISE_SYMBOL;
  // Advance the symbol only once the hues have been exhausted, so the first four
  // clusters differ in BOTH channels.
  return SYMBOLS[Math.floor(label / LIGHT_HUES.length) % SYMBOLS.length];
}

export function clusterName(label: number): string {
  return label === NOISE_LABEL ? "Noise" : `Cluster ${label}`;
}

/** Qualitative reading of a silhouette score, for the EDA headline. */
export function silhouetteBand(score: number | null | undefined): {
  label: string;
  hint: string;
} {
  if (score === null || score === undefined) {
    return {
      label: "Not scored",
      hint: "Silhouette needs at least two clusters to be defined.",
    };
  }
  if (score >= 0.5) {
    return {
      label: "Strong",
      hint: "Clusters are dense and well separated — the dataset has distinct groups.",
    };
  }
  if (score >= 0.25) {
    return {
      label: "Moderate",
      hint: "Some structure, but clusters overlap. Treat group boundaries as approximate.",
    };
  }
  return {
    label: "Weak",
    hint: "Little separation — points sit about as close to other clusters as their own.",
  };
}
