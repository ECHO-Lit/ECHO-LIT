/** Word-level alignment between a reference (baseline) transcript and a
 * hypothesis (variant) transcript, for the VariantInspector heatmap.
 *
 * Standard Wagner-Fischer edit distance with backtrace, operating on word
 * tokens rather than characters, so the output is an ordered sequence of
 * match/substitute/insert/delete operations, not just a distance. This is
 * the same underlying edit-distance idea jiwer uses server-side for WER,
 * but here we need the alignment path itself, not just the count, so it's
 * a separate small client-side implementation rather than a shared one.
 */

export type DiffOp =
  | { type: "match"; word: string }
  | { type: "substitute"; refWord: string; hypWord: string }
  | { type: "insert"; word: string }
  | { type: "delete"; word: string };

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g, "")
    .split(/\s+/)
    .filter(Boolean);
}

export function wordDiff(reference: string, hypothesis: string): DiffOp[] {
  const ref = tokenize(reference);
  const hyp = tokenize(hypothesis);
  const n = ref.length;
  const m = hyp.length;

  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 0; i <= n; i++) dp[i][0] = i;
  for (let j = 0; j <= m; j++) dp[0][j] = j;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      dp[i][j] = ref[i - 1] === hyp[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]);
    }
  }

  const ops: DiffOp[] = [];
  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && ref[i - 1] === hyp[j - 1]) {
      ops.push({ type: "match", word: hyp[j - 1] });
      i -= 1; j -= 1;
    } else if (i > 0 && j > 0 && dp[i][j] === dp[i - 1][j - 1] + 1) {
      ops.push({ type: "substitute", refWord: ref[i - 1], hypWord: hyp[j - 1] });
      i -= 1; j -= 1;
    } else if (j > 0 && dp[i][j] === dp[i][j - 1] + 1) {
      ops.push({ type: "insert", word: hyp[j - 1] });
      j -= 1;
    } else {
      ops.push({ type: "delete", word: ref[i - 1] });
      i -= 1;
    }
  }
  ops.reverse();
  return ops;
}
