// Transcript accuracy metrics, mirroring the legacy backend formulas from
// /inferences/whisper-accuracy so the UI shows the same numbers the sync
// API used to return.

export interface TranscriptMetrics {
  accuracy_percentage: number;
  word_error_rate: number;
  character_error_rate: number;
  levenshtein_distance: number;
  exact_match: number;
  character_similarity: number;
  word_count_predicted: number;
  word_count_truth: number;
}

function cleanText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function levenshtein<T>(a: ArrayLike<T>, b: ArrayLike<T>): number {
  if (a.length < b.length) return levenshtein(b, a);
  if (b.length === 0) return a.length;
  let previous = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 0; i < a.length; i++) {
    const current = [i + 1];
    for (let j = 0; j < b.length; j++) {
      current.push(Math.min(
        previous[j + 1] + 1,
        current[j] + 1,
        previous[j] + (a[i] !== b[j] ? 1 : 0),
      ));
    }
    previous = current;
  }
  return previous[b.length];
}

// difflib.SequenceMatcher.ratio equivalent: 2*M/T where M = total matched
// elements (longest-common-subsequence style block matching approximated
// via edit distance: matches = (lenA + lenB - distance*?) ). Use the
// standard identity ratio = (lenA + lenB - levenshtein_substitution_free)
// — closest practical equivalent is 1 - distance / max(len), but to stay
// near difflib's behavior we compute matches from the LCS length.
function lcsLength<T>(a: ArrayLike<T>, b: ArrayLike<T>): number {
  let previous = new Array<number>(b.length + 1).fill(0);
  for (let i = 0; i < a.length; i++) {
    const current = [0];
    for (let j = 0; j < b.length; j++) {
      current.push(a[i] === b[j] ? previous[j] + 1 : Math.max(previous[j + 1], current[j]));
    }
    previous = current;
  }
  return previous[b.length];
}

function similarityRatio<T>(a: ArrayLike<T>, b: ArrayLike<T>): number {
  const total = a.length + b.length;
  if (total === 0) return 1;
  return (2 * lcsLength(a, b)) / total;
}

export function computeTranscriptMetrics(predicted: string, groundTruth: string): TranscriptMetrics {
  const predClean = cleanText(predicted);
  const truthClean = cleanText(groundTruth);
  const predWords = predClean ? predClean.split(' ') : [];
  const truthWords = truthClean ? truthClean.split(' ') : [];

  const charSimilarity = similarityRatio(predClean, truthClean);
  const wordSimilarity = similarityRatio(predWords, truthWords);
  const levDist = levenshtein(predClean, truthClean);

  const wer = truthWords.length === 0
    ? (predWords.length > 0 ? 1 : 0)
    : Math.min(levenshtein(predWords, truthWords) / truthWords.length, 1);
  const cer = truthClean.length === 0
    ? (predClean.length > 0 ? 1 : 0)
    : Math.min(levDist / truthClean.length, 1);

  return {
    accuracy_percentage: wordSimilarity * 100,
    word_error_rate: wer,
    character_error_rate: cer,
    levenshtein_distance: levDist,
    exact_match: predClean === truthClean ? 1 : 0,
    character_similarity: charSimilarity * 100,
    word_count_predicted: predWords.length,
    word_count_truth: truthWords.length,
  };
}
