/**
 * The client's one audio-type check, shared by the drop zone and the file
 * picker. It was duplicated verbatim in AudioUploader and AudioDatasetPanel
 * (3.1.3 OBS-14); once BUG-27 made the drop-zone copy reachable, the two
 * copies could drift independently.
 *
 * MIME type OR extension: Chrome reports an empty type for .flac, so the
 * extension has to be accepted as a fallback. The server re-validates both.
 */
export const ACCEPTED_AUDIO_EXTENSIONS: readonly string[] = [".wav", ".mp3", ".m4a", ".flac"];

export function isAcceptedAudioFile(file: { name: string; type: string }): boolean {
  const extension = file.name.toLowerCase().substring(file.name.lastIndexOf("."));
  return file.type.startsWith("audio/") || ACCEPTED_AUDIO_EXTENSIONS.includes(extension);
}
