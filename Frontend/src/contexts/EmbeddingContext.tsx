import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { materializeAudio, runJob } from '@/lib/jobs';
import { readEdaCache, writeEdaCache } from '@/lib/edaCache';

export interface EmbeddingPoint {
  filename: string;
  coordinates: number[];
  embedding?: number[];
  embedding_dim?: number;
}

export interface ClusterStat {
  label: number;
  size: number;
  mean_silhouette: number | null;
  medoid_index: number;
}

/** Mirrors the payload from `Backend/app/services/clustering_service.py`. */
export interface ClusteringResult {
  labels: number[];
  probabilities: number[];
  n_clusters: number;
  n_noise: number;
  silhouette_score: number | null;
  silhouette_samples: Array<number | null>;
  cluster_stats: ClusterStat[];
  params: { min_cluster_size: number; metric: string; pca_dims: number | null };
  skipped_reason?: string;
  /** Set instead of the above when clustering failed but the embedding job succeeded. */
  error?: string;
}

export interface EmbeddingData {
  model: string;
  dataset: string;
  reduction_method: string;
  n_components: number;
  embeddings: Array<{
    filename: string;
    embedding: number[];
    embedding_dim: number;
  }>;
  reduced_embeddings?: EmbeddingPoint[];
  total_files: number;
  original_dimension: number;
  clustering?: ClusteringResult;
  /** Derived once at fetch time so consumers never re-join labels to files. */
  clusterByFilename?: Record<string, number>;
}

interface EmbeddingContextType {
  embeddingData: EmbeddingData | null;
  isLoading: boolean;
  error: string | null;
  fetchEmbeddings: (
    model: string,
    dataset: string,
    files: string[],
    reductionMethod?: string,
    nComponents?: number,
    minClusterSize?: number
  ) => Promise<void>;
  clearEmbeddings: () => void;
}

const EmbeddingContext = createContext<EmbeddingContextType | undefined>(undefined);

export const useEmbedding = () => {
  const context = useContext(EmbeddingContext);
  if (context === undefined) {
    throw new Error('useEmbedding must be used within an EmbeddingProvider');
  }
  return context;
};

export const EmbeddingProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [embeddingData, setEmbeddingData] = useState<EmbeddingData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchEmbeddings = useCallback(async (
    model: string,
    dataset: string,
    files: string[],
    reductionMethod: string = 'pca',
    nComponents: number = 3,
    minClusterSize: number = 5
  ) => {
    if (!files || files.length === 0) {
      setError('No files provided');
      return;
    }

    // Everything that changes the result must be part of the cache identity.
    const cacheKind = `embedding:${model}:${reductionMethod}:${nComponents}:${minClusterSize}`;
    const cached = readEdaCache<EmbeddingData>(cacheKind, dataset, files);
    if (cached) {
      setEmbeddingData(cached);
      setError(null);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const assets = await Promise.all(files.map((filename) => materializeAudio(dataset, filename)));
      const result: any = await runJob({
        operation: 'embedding',
        model,
        audio_ids: assets.map((asset) => asset.audio_id),
        parameters: {
          reduction: reductionMethod,
          n_components: nComponents,
          cluster: true,
          min_cluster_size: minClusterSize,
        },
      });
      const embeddings = result.items.map((item, index) => ({
        filename: files[index], embedding: item.result, embedding_dim: item.result.length,
      }));
      // Cluster labels arrive aligned with `items`, i.e. with `files` -- the same
      // positional join `projection` already relies on.
      const clustering: ClusteringResult | undefined = result.clustering;
      const clusterByFilename = clustering?.labels
        ? Object.fromEntries(clustering.labels.map((label, index) => [files[index], label]))
        : undefined;
      const data: EmbeddingData = {
        model,
        dataset,
        reduction_method: reductionMethod,
        n_components: nComponents,
        embeddings,
        reduced_embeddings: result.projection?.map((coordinates, index) => ({
          filename: files[index], coordinates,
        })),
        total_files: files.length,
        original_dimension: embeddings[0]?.embedding_dim || 0,
        clustering,
        clusterByFilename,
      };
      setEmbeddingData(data);
      // Skipped automatically if the raw vectors are too large for sessionStorage.
      writeEdaCache(cacheKind, dataset, files, data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch embeddings';
      setError(errorMessage);
      console.error('Error fetching embeddings:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearEmbeddings = useCallback(() => {
    setEmbeddingData(null);
    setError(null);
  }, []);

  return (
    <EmbeddingContext.Provider value={{
      embeddingData,
      isLoading,
      error,
      fetchEmbeddings,
      clearEmbeddings,
    }}>
      {children}
    </EmbeddingContext.Provider>
  );
};
