/**
 * Nearest-neighbour retrieval over raw embedding vectors.
 *
 * Entirely client-side: `EmbeddingContext` already holds the full-dimensional
 * vectors returned by the embedding job, so "find the clips that sound most like
 * this one" needs no backend call.
 */

export type SimilarityMetric = "cosine" | "euclidean";

export interface Neighbor {
  filename: string;
  /** Cosine similarity (higher = closer) or euclidean distance (lower = closer). */
  score: number;
  cluster?: number;
}

export interface NeighborIndex {
  filenames: string[];
  /** Row-major [n x dim], L2-normalised so cosine is a plain dot product. */
  normalized: Float32Array;
  /** Row-major [n x dim], original scale, for euclidean distance. */
  raw: Float32Array;
  dim: number;
  indexOf: Record<string, number>;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const delta = a[i] - b[i];
    sum += delta * delta;
  }
  return Math.sqrt(sum);
}

/**
 * Pack the embeddings into flat typed arrays once, so each subsequent query is a
 * single O(n * dim) pass instead of re-normalising on every selection change.
 */
export function buildNeighborIndex(
  embeddings: Array<{ filename: string; embedding: number[] }>
): NeighborIndex | null {
  const usable = embeddings.filter((entry) => entry.embedding?.length);
  if (usable.length === 0) return null;

  const dim = usable[0].embedding.length;
  const rows = usable.filter((entry) => entry.embedding.length === dim);
  const normalized = new Float32Array(rows.length * dim);
  const raw = new Float32Array(rows.length * dim);
  const filenames: string[] = [];
  const indexOf: Record<string, number> = {};

  rows.forEach((entry, row) => {
    const offset = row * dim;
    let norm = 0;
    for (let i = 0; i < dim; i += 1) {
      const value = entry.embedding[i];
      raw[offset + i] = value;
      norm += value * value;
    }
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < dim; i += 1) {
      normalized[offset + i] = entry.embedding[i] / norm;
    }
    filenames.push(entry.filename);
    indexOf[entry.filename] = row;
  });

  return { filenames, normalized, raw, dim, indexOf };
}

/**
 * Top-k neighbours of `query`, nearest first. The query itself is excluded.
 * Returns [] when the query is absent from the index.
 */
export function nearestNeighbors(
  query: string,
  index: NeighborIndex,
  k: number,
  metric: SimilarityMetric = "cosine",
  clusterByFilename?: Record<string, number>
): Neighbor[] {
  const queryRow = index.indexOf[query];
  if (queryRow === undefined) return [];

  const { dim, filenames } = index;
  const matrix = metric === "cosine" ? index.normalized : index.raw;
  const queryOffset = queryRow * dim;
  const scored: Neighbor[] = [];

  for (let row = 0; row < filenames.length; row += 1) {
    if (row === queryRow) continue;
    const offset = row * dim;
    let score = 0;
    if (metric === "cosine") {
      for (let i = 0; i < dim; i += 1) score += matrix[queryOffset + i] * matrix[offset + i];
    } else {
      for (let i = 0; i < dim; i += 1) {
        const delta = matrix[queryOffset + i] - matrix[offset + i];
        score += delta * delta;
      }
      score = Math.sqrt(score);
    }
    scored.push({ filename: filenames[row], score, cluster: clusterByFilename?.[filenames[row]] });
  }

  // Cosine: higher is closer. Euclidean: lower is closer.
  scored.sort((a, b) => (metric === "cosine" ? b.score - a.score : a.score - b.score));
  return scored.slice(0, k);
}
