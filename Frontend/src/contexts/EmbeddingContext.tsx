import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { materializeAudio, runJob } from '@/lib/jobs';

export interface EmbeddingPoint {
  filename: string;
  coordinates: number[];
  embedding?: number[];
  embedding_dim?: number;
}

interface EmbeddingData {
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
    nComponents?: number
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
    nComponents: number = 3
  ) => {
    if (!files || files.length === 0) {
      setError('No files provided');
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
        parameters: { reduction: reductionMethod, n_components: nComponents },
      });
      const embeddings = result.items.map((item, index) => ({
        filename: files[index], embedding: item.result, embedding_dim: item.result.length,
      }));
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
      };
      setEmbeddingData(data);
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
