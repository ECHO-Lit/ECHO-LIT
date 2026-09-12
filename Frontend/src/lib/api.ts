/**
 * Where the API lives.
 *
 * `VITE_API_BASE_URL` pins it (a trailing slash is dropped: `${API_BASE}/jobs`
 * must not become `//jobs`). Unset, the API is taken to be on port 8000 of the
 * host that served this page. That keeps the API the same *site* as the page
 * — http://127.0.0.1:8080 calls http://127.0.0.1:8000, not localhost — so the
 * browser sends the SameSite=Lax session cookie with every request.
 */
export function resolveApiBase(
  configured: string | undefined,
  location: Pick<Location, "protocol" | "hostname"> | undefined,
): string {
  const value = configured?.trim();
  if (value) return value.replace(/\/+$/, "");
  if (!location?.hostname) return "http://localhost:8000";
  return `${location.protocol}//${location.hostname}:8000`;
}

export const API_BASE: string = resolveApiBase(
  import.meta.env.VITE_API_BASE_URL,
  typeof window !== "undefined" ? window.location : undefined,
);

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
