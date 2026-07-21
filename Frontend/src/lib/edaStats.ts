// Shared statistics helpers for the Dataset EDA view — kept out of the chart
// components so the same numbers can be reused across charts, callouts, and
// the outlier list without recomputing.

export function basename(path: string): string {
  return path.split("/").pop()?.split("\\").pop() || path;
}

export function pearson(a: number[], b: number[]): number {
  const n = a.length;
  if (n === 0) return 0;
  const meanA = a.reduce((s, v) => s + v, 0) / n;
  const meanB = b.reduce((s, v) => s + v, 0) / n;
  let num = 0, denomA = 0, denomB = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    num += da * db;
    denomA += da * da;
    denomB += db * db;
  }
  const denom = Math.sqrt(denomA * denomB);
  return denom === 0 ? 0 : num / denom;
}

export function correlationMatrix(
  featuresByFile: Record<string, Record<string, number>>,
  featureKeys: string[],
): number[][] {
  const files = Object.keys(featuresByFile);
  return featureKeys.map((keyA) =>
    featureKeys.map((keyB) => {
      const a = files.map((f) => featuresByFile[f][keyA]).filter((v) => typeof v === "number");
      const b = files.map((f) => featuresByFile[f][keyB]).filter((v) => typeof v === "number");
      const n = Math.min(a.length, b.length);
      return n > 1 ? pearson(a.slice(0, n), b.slice(0, n)) : 0;
    }),
  );
}

export interface CorrelatedPair {
  a: string;
  b: string;
  r: number;
}

export function topCorrelatedPairs(matrix: number[][], keys: string[], threshold = 0.8): CorrelatedPair[] {
  const pairs: CorrelatedPair[] = [];
  for (let i = 0; i < keys.length; i++) {
    for (let j = i + 1; j < keys.length; j++) {
      const r = matrix[i]?.[j];
      if (typeof r === "number" && Math.abs(r) >= threshold) {
        pairs.push({ a: keys[i], b: keys[j], r });
      }
    }
  }
  return pairs.sort((x, y) => Math.abs(y.r) - Math.abs(x.r));
}

export function percentile(sortedValues: number[], p: number): number {
  if (sortedValues.length === 0) return 0;
  const idx = (p / 100) * (sortedValues.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedValues[lo];
  return sortedValues[lo] + (sortedValues[hi] - sortedValues[lo]) * (idx - lo);
}

export interface Quartiles {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
}

export function quartiles(values: number[]): Quartiles {
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 0) return { min: 0, q1: 0, median: 0, q3: 0, max: 0 };
  return {
    min: sorted[0],
    q1: percentile(sorted, 25),
    median: percentile(sorted, 50),
    q3: percentile(sorted, 75),
    max: sorted[sorted.length - 1],
  };
}

export interface FeatureOutlier {
  filename: string;
  feature: string;
  value: number;
  z: number;
}

export function zScores(
  featuresByFile: Record<string, Record<string, number>>,
  aggregateStatistics: Record<string, { mean: number; std: number }>,
  threshold = 3,
): FeatureOutlier[] {
  const outliers: FeatureOutlier[] = [];
  for (const [filename, features] of Object.entries(featuresByFile)) {
    for (const [feature, value] of Object.entries(features)) {
      const stats = aggregateStatistics[feature];
      if (!stats || stats.std === 0) continue;
      const z = (value - stats.mean) / stats.std;
      if (Math.abs(z) >= threshold) outliers.push({ filename, feature, value, z });
    }
  }
  return outliers.sort((a, b) => Math.abs(b.z) - Math.abs(a.z));
}

// Index of the histogram bin `value` falls into, given numpy.histogram-style
// bin edges (n+1 edges for n bins). Values at the max edge land in the last bin.
export function bucketize(value: number, edges: number[]): number {
  if (edges.length < 2) return -1;
  const last = edges.length - 2;
  if (value >= edges[edges.length - 1]) return last;
  for (let i = 0; i < last + 1; i++) {
    if (value >= edges[i] && value < edges[i + 1]) return i;
  }
  return -1;
}
