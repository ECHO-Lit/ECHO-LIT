/** FR-10 tooltip copy (SRS §3.7: every metric/diagnostic must be explained
 * in place). Mirrors lib/fr7Glossary.ts's role for the linguistic-acoustic panel. */
export const FR10_GLOSSARY = {
  design:
    "Whether the compared groups read the same linguistic content. 'Matched' means every group covers identical text, so a performance gap is attributable to acoustics, not sentence difficulty. 'Unmatched' means groups read different text, so part of any gap may reflect content difficulty rather than group differences.",
  referenceGroup:
    "The baseline every other group is compared against. Defaults to the largest surviving group; every disparity number is 'this group minus the reference group'.",
  minGroupSize:
    "Groups with fewer items than this are excluded from the comparison entirely and listed separately, rather than being silently dropped.",
  minSpeakers:
    "Groups with fewer distinct speakers than this are still shown, but their disparity is not one of the headline flagged results — with too few speakers, a group difference cannot be told apart from one person's voice.",
  macroWer:
    "Mean of each utterance's own word error rate, weighting every utterance equally. This is what a fairness comparison across groups is actually about, as opposed to micro (corpus-pooled) WER, which longer utterances dominate.",
  disparityCI:
    "A 95% confidence interval from a blockwise bootstrap: utterances from the same speaker are resampled together, not independently, because their errors are correlated. A narrower corpus with fewer speakers gets a wider interval, which is the honest result, not a bug.",
  pAdjusted:
    "The significance threshold after Holm-Bonferroni correction across every group/metric comparison run together. Without this correction, testing many groups and metrics at once would flag 'significant' differences from chance alone.",
  mde:
    "Minimum detectable effect: the smallest true gap this corpus could have detected at this sample size. When a comparison is not significant, check this number before concluding there is no disparity — a small corpus may simply lack the power to see a real gap.",
  silhouette:
    "How well the model's embeddings cluster by group label, from -1 to 1. Compared against a permutation null (the same statistic under random group relabeling) so a small silhouette can still be a real, non-chance signal.",
  leakage:
    "Balanced accuracy of a nearest-neighbor classifier predicting group membership from embeddings, using speaker-disjoint cross-validation so it measures accent/group structure rather than recognizing individual voices.",
  groundingLift:
    "How much more of the model's saliency attribution falls on speech versus silence, relative to what uniform attribution would give. 1.0 means no more than chance; higher means the explanation concentrates on acoustically meaningful regions.",
  attributionEntropy:
    "How spread out the saliency attribution is across the clip, from 0 (all on one frame) to 1 (perfectly uniform). A highly diffuse explanation commits to nothing and is less useful as an explanation.",
  verdict:
    "'Disparity detected' requires both statistical significance (after correction) and a practically meaningful gap size. 'Inconclusive' means either too few speakers to attribute an effect to the group, or too little data to rule out a real gap. 'No evidence of disparity' is not the same as 'proven fair'.",
  speakerConfounded:
    "This group has fewer speakers than the minimum — any performance or grounding difference could be a property of these specific speakers' voices rather than the group label.",
};
