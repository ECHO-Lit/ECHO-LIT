/**
 * Saliency faithfulness: the backend contract, the job wrapper, and the words
 * used to explain the numbers.
 *
 * A saliency map claims the model used certain regions. The faithfulness job
 * deletes those regions and checks the model actually breaks.
 *
 * Reading the numbers this module carries:
 *  - `aopc_deletion` alone means nothing. Masking audio hurts a model whichever
 *    regions you take, so part of any drop is just damage.
 *  - `faithfulness_gain` = `aopc_deletion - aopc_random` is the honest headline,
 *    and it must be read against `aopc_random_stderr`: a gain smaller than the
 *    error bar on the baseline is noise. `verdict` already applies that rule, which
 *    is why the UI shows the verdict rather than re-deriving a judgement.
 *  - `occlusion_spearman` is the number to show someone who does not want a
 *    curve: does removing a segment cost what the map said it was worth?
 *
 * Same convention as `probes.ts`: never present a score without the baseline
 * that would arise if there were no signal at all.
 */

import { runJob, type JobStatus } from './jobs';

export type FaithfulnessVerdict = 'faithful' | 'weak' | 'uninformative';

export interface CurvePoint {
  fraction: number;
  score: number;
  /** Spread across random placements; present on the random curve only. */
  std?: number;
}

export interface OcclusionPoint {
  start_time: number;
  end_time: number;
  word?: string | null;
  /** What the map claims this segment is worth. */
  saliency: number;
  /** What removing it actually cost the model. */
  drop: number;
}

export interface FaithfulnessMetrics {
  aopc_deletion: number;
  aopc_random: number;
  /** Standard error on the mean random AOPC — the bar `faithfulness_gain` must clear. */
  aopc_random_stderr: number;
  faithfulness_gain: number;
  aopc_inverse: number;
  comprehensiveness: number;
  sufficiency: number;
  auc_deletion: number;
  auc_insertion: number;
  occlusion_spearman: number | null;
  occlusion_p_value: number | null;
}

export interface SaliencySegment {
  start_time: number;
  end_time: number;
  saliency: number;
  intensity: number;
  word?: string;
}

export interface FaithfulnessComparisonBlock {
  fraction: number;
  clean_score: number;
  masked_score: number;
  /** Same duration removed at random — the control for `masked_score`. */
  random_score: number;
  removed_spans: Array<[number, number]>;
  random_spans: Array<[number, number]>;
}

export interface FaithfulnessResult {
  model: string;
  method: string;
  /** What the score tracks: a class probability, or a transcript's likelihood. */
  target: { kind: string; label: string | null };
  /** Score on unmasked audio; every drop is measured from here. */
  baseline_score: number;
  /**
   * Which estimator produced the map under test. `energy_fallback` means
   * attribution failed and the saliency service substituted an encoder energy
   * map — the verdict then describes that fallback, not the chosen method.
   */
  attribution_source: string | null;
  masked_fractions: number[];
  curves: {
    deletion_saliency: CurvePoint[];
    deletion_random: CurvePoint[];
    deletion_inverse: CurvePoint[];
    insertion_saliency: CurvePoint[];
  };
  /** The regions actually removed at `top_fraction`, with their scores. */
  comparison: FaithfulnessComparisonBlock;
  occlusion: OcclusionPoint[];
  metrics: FaithfulnessMetrics;
  n_steps: number;
  /** Frames the ranking actually ran at, after pooling to the evaluation rate. */
  eval_frames: number;
  random_repeats: number;
  seed: number;
  audio_seconds: number;
  verdict: FaithfulnessVerdict;
  skipped_reason: string | null;
  /** The map that was tested, so it can be drawn beside the verdict. */
  saliency: {
    series: number[];
    segments: SaliencySegment[];
    total_duration: number;
    emotion?: string | null;
  };
}

export interface FaithfulnessOptions {
  model: string;
  audioId: string;
  method?: 'gradcam' | 'lime' | 'shap';
  nSteps?: number;
  topFraction?: number;
  randomRepeats?: number;
  includeOcclusion?: boolean;
  seed?: number;
}

export async function runFaithfulness(
  options: FaithfulnessOptions,
  hooks: { signal?: AbortSignal; onProgress?: (status: JobStatus) => void } = {},
): Promise<FaithfulnessResult> {
  const result = await runJob<any>(
    {
      operation: 'saliency_faithfulness',
      model: options.model,
      audio_ids: [options.audioId],
      parameters: {
        method: options.method ?? 'gradcam',
        n_steps: options.nSteps ?? 9,
        top_fraction: options.topFraction ?? 0.2,
        random_repeats: options.randomRepeats ?? 3,
        include_occlusion: options.includeOcclusion ?? true,
        seed: options.seed ?? 42,
      },
    },
    hooks,
  );
  if (!result?.items?.length) throw new Error('Faithfulness job returned no result');
  return result.items[0].result as FaithfulnessResult;
}

// --- Presentation vocabulary -----------------------------------------------
// Kept here rather than in the components so every view says the same thing
// about the same number.

export const VERDICT_COPY: Record<FaithfulnessVerdict, { label: string; blurb: string }> = {
  faithful: {
    label: 'Faithful',
    blurb:
      'Removing the highlighted audio hurts this model clearly more than removing the same amount at random. The map is pointing at what the model actually uses.',
  },
  weak: {
    label: 'Weak',
    blurb:
      'The highlighted audio matters a little more than random audio, but not by much. Treat individual highlights as suggestive, not as evidence.',
  },
  uninformative: {
    label: 'Not informative',
    blurb:
      'Removing the highlighted audio costs no more than removing random audio. This map is not telling you what the model relies on — do not read meaning into its peaks.',
  },
};

export const METRIC_HELP: Record<string, string> = {
  faithfulness_gain:
    'How much more the model suffers when its most salient audio is removed, compared with removing the same duration at random. Zero means the map found nothing.',
  occlusion_spearman:
    'Rank agreement between what the map says each segment is worth and what removing that segment actually costs. 1.0 is perfect agreement, 0 is none.',
  comprehensiveness:
    'How much confidence the model loses when the top-ranked audio is removed. Higher is better — it means the map captured something the model needed.',
  sufficiency:
    'How much confidence is lost when everything EXCEPT the top-ranked audio is removed. Lower is better — it means the highlighted audio was enough on its own.',
  aopc_deletion:
    'Average confidence lost as the most salient audio is progressively removed. Read against the random baseline, never alone.',
  aopc_random:
    'The same measurement with randomly chosen regions. This is the floor any map has to beat.',
};

/** What the tracked score means for this model, in the user's terms. */
export function targetDescription(target: FaithfulnessResult['target']): string {
  switch (target.kind) {
    case 'class_prob':
      return `confidence in "${target.label}"`;
    case 'transcript_logprob':
    case 'ctc_logprob':
      return 'likelihood of the original transcript';
    default:
      return 'model output';
  }
}

/** `faithfulness_gain` as a 0-100 score, clamped. Presentation only. */
export function gainAsScore(gain: number): number {
  return Math.max(0, Math.min(100, Math.round(gain * 100)));
}

export function formatSigned(value: number, digits = 3): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}
