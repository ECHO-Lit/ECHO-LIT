/**
 * Run/cache state machine for the layer-probing job.
 *
 * The panel owns the controls; this owns the async lifecycle so a dataset or
 * model switch mid-flight cannot land a stale result on screen.
 *
 * Caching is two-tier and deliberately so: the backend content-addresses the
 * expensive per-file activations, so a re-run with different probe settings is
 * already cheap server-side. The sessionStorage layer on top only saves the
 * round-trip when the user leaves the tab and comes back.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { readEdaCache, writeEdaCache } from '@/lib/edaCache';
import { materializeAudio } from '@/lib/jobs';
import {
  fetchDatasetMetadata,
  extractProperties,
  runLayerProbe,
  type LayerProbeResult,
  type MetadataRow,
} from '@/lib/probes';

export interface ProbeSettings {
  probe: 'logreg' | 'linear_svm';
  cvFolds: number;
  projectDims: number;
  minClassCount: number;
  includeControl: boolean;
  noiseSnrDb: number | null;
}

export const DEFAULT_PROBE_SETTINGS: ProbeSettings = {
  probe: 'logreg',
  cvFolds: 5,
  projectDims: 256,
  minClassCount: 5,
  includeControl: true,
  noiseSnrDb: null,
};

export type ProbeStatus = 'idle' | 'running' | 'done' | 'error';

interface UseLayerProbesArgs {
  dataset: string;
  model: string;
  availableFiles: string[];
}

function cacheKind(model: string, selected: string[], settings: ProbeSettings): string {
  const properties = [...selected].sort().join('+');
  return [
    'layer-probe',
    model,
    properties,
    settings.probe,
    settings.cvFolds,
    settings.projectDims,
    settings.minClassCount,
    settings.includeControl ? 'ctl' : 'noctl',
    settings.noiseSnrDb ?? 'clean',
  ].join(':');
}

export function useLayerProbes({ dataset, model, availableFiles }: UseLayerProbesArgs) {
  const [metadata, setMetadata] = useState<MetadataRow[]>([]);
  const [metadataError, setMetadataError] = useState<string | null>(null);
  const [result, setResult] = useState<LayerProbeResult | null>(null);
  const [status, setStatus] = useState<ProbeStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState({ current: 0, total: 0, message: '' });
  const [fromCache, setFromCache] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // A result belongs to the dataset and model it was computed from. Switching
  // either must clear it, mirroring the `availableFiles.length === 0` guard the
  // embedding tab uses -- a stale chart here would silently misattribute a
  // finding to the wrong model.
  useEffect(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setResult(null);
    setStatus('idle');
    setError(null);
    setFromCache(false);
    setProgress({ current: 0, total: 0, message: '' });
  }, [dataset, model]);

  useEffect(() => {
    if (!dataset) return;
    const controller = new AbortController();
    setMetadataError(null);
    fetchDatasetMetadata(dataset, controller.signal)
      .then(setMetadata)
      .catch((cause) => {
        if (controller.signal.aborted) return;
        setMetadata([]);
        setMetadataError(cause instanceof Error ? cause.message : 'Failed to load metadata');
      });
    return () => controller.abort();
  }, [dataset]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const run = useCallback(
    async (selected: string[], settings: ProbeSettings) => {
      if (availableFiles.length < 2 || selected.length === 0) return;

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const kind = cacheKind(model, selected, settings);
      const cached = readEdaCache<LayerProbeResult>(kind, dataset, availableFiles);
      if (cached) {
        setResult(cached);
        setStatus('done');
        setError(null);
        setFromCache(true);
        return;
      }

      setStatus('running');
      setError(null);
      setFromCache(false);
      setProgress({ current: 0, total: availableFiles.length, message: 'Preparing audio' });

      try {
        // Materialised in file order, and the labels below are built in that
        // same order. The backend joins the two positionally, so these two
        // lines must stay adjacent and must not be reordered.
        const assets = await Promise.all(
          availableFiles.map((filename) => materializeAudio(dataset, filename, controller.signal)),
        );
        const properties = extractProperties(metadata, availableFiles, selected);

        const response = await runLayerProbe(
          {
            model,
            audioIds: assets.map((asset) => asset.audio_id),
            properties,
            probe: settings.probe,
            cvFolds: settings.cvFolds,
            projectDims: settings.projectDims,
            minClassCount: settings.minClassCount,
            includeControl: settings.includeControl,
            noiseSnrDb: settings.noiseSnrDb,
          },
          {
            signal: controller.signal,
            onProgress: (job) =>
              setProgress({
                current: job.progress.current,
                total: job.progress.total,
                message: job.progress.message,
              }),
          },
        );

        if (controller.signal.aborted) return;
        setResult(response.probes);
        setStatus('done');
        writeEdaCache(kind, dataset, availableFiles, response.probes);
      } catch (cause) {
        if (controller.signal.aborted) return;
        setResult(null);
        setStatus('error');
        setError(cause instanceof Error ? cause.message : 'Layer probing failed');
      }
    },
    [availableFiles, dataset, metadata, model],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStatus('idle');
    setProgress({ current: 0, total: 0, message: '' });
  }, []);

  return { metadata, metadataError, result, status, error, progress, fromCache, run, cancel };
}
