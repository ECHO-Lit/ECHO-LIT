export const API_BASE: string = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface AudioReference {
  audio_id: string;
  file_id: string;
  filename: string;
  playback_url: string;
  media_type: string;
  size_bytes: number;
  duration_seconds: number;
  sample_rate?: number;
  channels?: number;
  message?: string;
}
