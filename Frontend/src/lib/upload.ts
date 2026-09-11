/**
 * Multipart upload with byte-level progress.
 *
 * The Fetch API exposes no upload progress, which is why 3.1.3 (BUG-22) could
 * only replace a fake progress bar with an indeterminate one. XMLHttpRequest's
 * `upload.onprogress` reports bytes actually sent, so a large dataset upload —
 * up to 100 MB per file, many files per request — can show real progress
 * (SRS US-3: "a visible progress indication") instead of a spinner that looks
 * identical whether the transfer is at 1% or stalled.
 *
 * Errors go through `describeHttpError`, exactly as the Fetch call sites do
 * (SRS US-4), and an AbortSignal cancels the transfer.
 */
import { describeHttpError } from "./httpError";

export interface UploadProgress {
  loaded: number;
  /** 0 when the browser cannot compute the body size. */
  total: number;
  /** loaded / total in [0, 1], or null when the total is unknown. */
  fraction: number | null;
}

export interface UploadOptions {
  onProgress?: (progress: UploadProgress) => void;
  signal?: AbortSignal;
  /** Prefix for error messages, as in describeHttpError. */
  context?: string;
}

const NETWORK_ERROR =
  "The upload could not reach the server. Check your connection and try again.";

export function uploadWithProgress<T = unknown>(
  url: string,
  body: FormData,
  { onProgress, signal, context }: UploadOptions = {},
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.withCredentials = true; // the session cookie, as fetch's credentials: 'include'

    xhr.upload.onprogress = (event) => {
      onProgress?.({
        loaded: event.loaded,
        total: event.lengthComputable ? event.total : 0,
        fraction: event.lengthComputable && event.total > 0 ? event.loaded / event.total : null,
      });
    };

    xhr.onload = () => {
      const response = new Response(xhr.responseText || null, {
        status: xhr.status,
        headers: { "Content-Type": xhr.getResponseHeader("Content-Type") ?? "application/json" },
      });
      if (xhr.status >= 200 && xhr.status < 300) {
        response.json().then(resolve as (value: unknown) => void, reject);
      } else {
        describeHttpError(response, context).then(reject, reject);
      }
    };
    xhr.onerror = () => reject(new Error(context ? `${context}: ${NETWORK_ERROR}` : NETWORK_ERROR));
    xhr.onabort = () => reject(new DOMException("Aborted", "AbortError"));

    signal?.addEventListener("abort", () => xhr.abort(), { once: true });
    xhr.send(body);
  });
}
