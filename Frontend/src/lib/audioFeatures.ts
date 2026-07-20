// Audio Feature Explanations - Shared utility for audio feature descriptions
export const AUDIO_FEATURE_EXPLANATIONS: Record<string, string> = {
  'spectral_rolloff': 'The frequency below which 85% of spectral energy is contained. Indicates brightness vs darkness of audio.',
  'spectral_centroid': 'The center of mass of the spectrum. Higher values indicate brighter sounds with more high-frequency content.',
  'spectral_bandwidth': 'The width of the spectrum. Measures how spread out the frequency content is.',
  'spectral_contrast': 'The difference in amplitude between peaks and valleys in the spectrum. Indicates audio clarity.',
  'spectral_flatness': 'Measures how noise-like vs tone-like the spectrum is. Values near 0 = tonal, near 1 = noise-like.',
  'zero_crossing_rate': 'Rate at which the audio signal changes from positive to negative. Higher for noisy/unvoiced sounds.',
  'tempo': 'The perceived speed of the music in beats per minute (BPM). Estimated from onset detection.',
  'duration': 'Length of the audio file in seconds.',
  'rms': 'Root Mean Square energy. Measures the overall loudness/power of the audio signal.',
  'mfcc_1': 'First Mel-frequency cepstral coefficient. Represents overall spectral shape and timbre.',
  'mfcc_2': 'Second MFCC. Captures the balance between low and high frequencies.',
  'mfcc_3': 'Third MFCC. Related to the slope of the spectral envelope.',
  'mfcc_4': 'Fourth MFCC. Captures finer spectral details and formant information.',
  'mfcc_5': 'Fifth MFCC. Represents additional spectral shape characteristics.',
  'mfcc_6': 'Sixth MFCC. Captures mid-frequency spectral features.',
  'mfcc_7': 'Seventh MFCC. Related to spectral fine structure.',
  'mfcc_8': 'Eighth MFCC. Represents higher-order spectral relationships.',
  'mfcc_9': 'Ninth MFCC. Captures additional timbral characteristics.',
  'mfcc_10': 'Tenth MFCC. Represents complex spectral interactions.',
  'mfcc_11': 'Eleventh MFCC. Captures additional spectral characteristics.',
  'mfcc_12': 'Twelfth MFCC. Represents fine-grain spectral features.',
  'mfcc_13': 'Thirteenth MFCC. Final coefficient capturing subtle spectral details.',
  'chroma_1': 'First chroma feature. Represents the energy in the C pitch class.',
  'chroma_2': 'Second chroma feature. Represents the energy in the C#/Db pitch class.',
  'chroma_3': 'Third chroma feature. Represents the energy in the D pitch class.',
  'chroma_4': 'Fourth chroma feature. Represents the energy in the D#/Eb pitch class.',
  'chroma_5': 'Fifth chroma feature. Represents the energy in the E pitch class.',
  'chroma_6': 'Sixth chroma feature. Represents the energy in the F pitch class.',
  'chroma_7': 'Seventh chroma feature. Represents the energy in the F#/Gb pitch class.',
  'chroma_8': 'Eighth chroma feature. Represents the energy in the G pitch class.',
  'chroma_9': 'Ninth chroma feature. Represents the energy in the G#/Ab pitch class.',
  'chroma_10': 'Tenth chroma feature. Represents the energy in the A pitch class.',
  'chroma_11': 'Eleventh chroma feature. Represents the energy in the A#/Bb pitch class.',
  'chroma_12': 'Twelfth chroma feature. Represents the energy in the B pitch class.',
  'tonnetz_1': 'First tonal centroid. Represents the position in tonal space along the circle of fifths.',
  'tonnetz_2': 'Second tonal centroid. Captures major vs minor tonality.',
  'tonnetz_3': 'Third tonal centroid. Represents the diminished chord dimension.',
  'tonnetz_4': 'Fourth tonal centroid. Additional harmonic relationship dimension.',
  'tonnetz_5': 'Fifth tonal centroid. Complex harmonic space representation.',
  'tonnetz_6': 'Sixth tonal centroid. Final tonal space dimension.',
  'rolloff': 'Alias for spectral_rolloff. The frequency below which 85% of spectral energy is contained.',
  'centroid': 'Alias for spectral_centroid. The center of mass of the spectrum.',
  'bandwidth': 'Alias for spectral_bandwidth. The width of the spectrum.',
  'contrast': 'Alias for spectral_contrast. The difference between spectral peaks and valleys.',
  'flatness': 'Alias for spectral_flatness. Measures how noise-like vs tone-like the audio is.',
  'zcr': 'Alias for zero_crossing_rate. Rate of signal sign changes.',
  // Additional common features that might appear
  'energy': 'Total energy of the audio signal. Measures overall signal power.',
  'loudness': 'Perceptual loudness measurement. Related to RMS but perceptually weighted.',
  'brightness': 'Perceptual brightness. Related to spectral centroid but perceptually calibrated.',
  'roughness': 'Perceptual roughness. Measures how harsh or smooth the audio sounds.',
  'sharpness': 'Perceptual sharpness. Measures the presence of high-frequency content.',
  'pitch': 'Fundamental frequency of the audio signal. The perceived pitch.',
  'harmonicity': 'Measure of how harmonic the audio signal is. Values near 1 indicate strong harmonics.',
  'noisiness': 'Measure of how noisy the audio signal is. Inverse of harmonicity.',
  'onset_rate': 'Rate of detected onsets in the audio. Measures rhythmic density.',
  'beat_strength': 'Strength of detected beat patterns. Measures rhythmic clarity.'
};

// Helper function to get ordinal numbers
export const getOrdinal = (num: number): string => {
  const ordinals = ['', 'First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 'Ninth', 'Tenth',
    'Eleventh', 'Twelfth', 'Thirteenth', 'Fourteenth', 'Fifteenth', 'Sixteenth', 'Seventeenth', 'Eighteenth', 'Nineteenth', 'Twentieth'];
  
  if (num <= 20) {
    return ordinals[num];
  }
  
  const suffix = ['th', 'st', 'nd', 'rd'][((num % 100) - 20) % 10] || 'th';
  return num + suffix;
};

/**
 * Get a human-readable explanation for an audio feature
 * @param featureName - The name of the audio feature (e.g., 'mfcc_1', 'spectral_centroid')
 * @returns A descriptive explanation of what the feature represents
 */
export const getFeatureExplanation = (featureName: string): string => {
  if (!featureName) return 'Unknown audio feature.';
  
  // Try exact match first
  if (AUDIO_FEATURE_EXPLANATIONS[featureName]) {
    return AUDIO_FEATURE_EXPLANATIONS[featureName];
  }
  
  // Try normalized version
  const normalizedName = featureName.toLowerCase().replace(/[_\s-]/g, '_');
  if (AUDIO_FEATURE_EXPLANATIONS[normalizedName]) {
    return AUDIO_FEATURE_EXPLANATIONS[normalizedName];
  }
  
  // Try without numbers (for numbered features like mfcc_14, chroma_13, etc.)
  const withoutNumbers = normalizedName.replace(/_\d+$/, '_1');
  if (AUDIO_FEATURE_EXPLANATIONS[withoutNumbers]) {
    const baseExplanation = AUDIO_FEATURE_EXPLANATIONS[withoutNumbers];
    const numberMatch = normalizedName.match(/_(\d+)$/);
    if (numberMatch) {
      const num = numberMatch[1];
      return baseExplanation.replace(/First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Eleventh|Twelfth|Thirteenth/, 
        `${getOrdinal(parseInt(num))}`);
    }
    return baseExplanation;
  }
  
  // Try partial matches for common patterns
  if (normalizedName.includes('mfcc')) {
    return 'Mel-frequency cepstral coefficient. Represents spectral shape and timbral characteristics of the audio.';
  }
  if (normalizedName.includes('chroma')) {
    return 'Chroma feature. Represents the energy distribution across the 12 pitch classes (musical notes).';
  }
  if (normalizedName.includes('tonnetz')) {
    return 'Tonal centroid feature. Represents the position in harmonic/tonal space.';
  }
  if (normalizedName.includes('spectral')) {
    return 'Spectral feature. Describes frequency domain characteristics of the audio signal.';
  }
  if (normalizedName.includes('temporal')) {
    return 'Temporal feature. Describes time domain characteristics of the audio signal.';
  }
  
  return 'Audio feature - extracting characteristics from the audio signal for analysis.';
};

// Plain-English explanations for dataset-level EDA charts (not per-feature).
export const EDA_CHART_EXPLANATIONS: Record<string, string> = {
  duration_histogram: 'Distribution of clip lengths. Very short or very long clips relative to the rest of the dataset may be padded or truncated differently by the model.',
  class_balance: 'Count of files per class label. Uneven counts can bias a model toward predicting the majority class, inflating accuracy without real skill.',
  transcript_length_histogram: 'Distribution of word counts per transcript. Short vs. long utterances stress a speech-to-text model differently and affect how word error rate should be read.',
  sample_rate_breakdown: 'Sample rates present in the dataset. Mixed sample rates mean some files are resampled before model input, which can introduce artifacts.',
  correlation_heatmap: 'Pairwise correlation between acoustic features. Strongly correlated (near +1 or -1) features carry redundant information.',
  feature_box_plot: 'Spread of a feature across the dataset: box = 25th-75th percentile, line = median, whiskers = min/max. Wide boxes mean high variability.',
};