/** Plain-language definitions for FR-7 sensitivity-profile terminology.
 * Surfaced via InfoTooltip next to the stat it explains. No em dashes,
 * per the diagnostics panel copy convention. */
export const FR7_GLOSSARY = {
  sensitivityIndex:
    "Average output degradation across the full range you swept for this property, normalized to 0 to 1. Computed as the area under the degradation curve. Higher means the model reacted more, averaged over the whole tested range, not just at the single worst point.",
  breakdownTheta:
    "The parameter value where this property's degradation curve first crosses 50 percent self-WER, or 50 percent output divergence for classification. Reported in the property's own unit, found by linear interpolation between the two nearest tested points. Left blank if the curve never crosses 50 percent within the range you swept.",
  localSlope:
    "How steeply the degradation curve rises right at the identity point, meaning no perturbation at all. A steep slope here means the model is fragile to small, realistic variation, even if the overall sensitivity index looks low because the rest of the curve stays flat.",
  asymmetry:
    "Average degradation on one side of the identity point minus the other, for sweeps that go both directions such as pitch. A nonzero value can point to a training data bias, for example a model that degrades faster when pitch is shifted up than when shifted down.",
  acousticInfluence:
    "The sensitivity index of the single most damaging property in this sweep. The headline number for how much acoustic manipulation, at worst, moved the model's output while the words stayed fixed.",
  linguisticRobustness:
    "One minus acoustic influence. How stable the model's output stayed while the spoken words were held constant across the sweep.",
  relativeToWordRemoval:
    "Acoustic influence divided by the degradation caused by the lexical destruction control, the reference sweep that actually removes about 30 percent of the words. A low ratio means acoustic manipulation barely matters compared to genuinely losing words.",
  lexicalControl:
    "A fixed reference sweep that masks about 30 percent of the audio timeline, genuinely deleting words rather than distorting sound. Used as the yardstick every other property's sensitivity index is measured against.",
  selfWer:
    "Word error rate measured against this same model's own transcript of the unperturbed baseline audio, not a separate ground truth transcript. Isolates damage caused by the perturbation from the model's ordinary transcription error on this speaker.",
  degradationBand:
    "Shaded region around a curve showing the 95 percent confidence interval across repeated runs with different random seeds. Only appears for stochastic properties such as noise. A wide band means the measurement is noisy at that point, a narrow band means it is consistent.",
  verdict:
    "Linguistically driven when acoustic influence is below 0.10. Acoustically dominated when acoustic influence reaches at least 60 percent of the lexical destruction control's degradation. Mixed otherwise.",
  wordDiffColors:
    "Plain text matched the baseline transcript exactly. Amber marks a substituted word. Red marks an inserted word not present in the baseline. Struck-through gray marks a word the variant dropped entirely.",
  interpolatedHover:
    "The line between tested points is a straight-line estimate, the same method used to compute the breakdown point. Only the marked dots were actually rendered and re-inferred through the model. Shaded bands, where present, show the 95 percent confidence interval across repeated runs with different random seeds, only for stochastic properties such as noise. The dashed horizontal line marks the 50 percent threshold that breakdown points are measured against.",
} as const;
