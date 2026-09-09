/**
 * One place that turns a failed Response into a message a user can act on.
 *
 * SRS US-4 (Error presentation): "Errors shall be presented as clear,
 * actionable messages in plain language rather than raw stack traces or HTTP
 * codes and shall indicate the corrective action where one exists."
 *
 * Before this helper existed there were six near-identical copies of
 * `Request failed (${response.status})` across lib/, which put a bare HTTP
 * status in front of the user — the exact thing US-4 forbids. The 401 case was
 * the worst: no 401/403 handling existed anywhere in the client, so an expired
 * session read "Request failed (401)" with no hint that reloading restores it.
 *
 * The server's own `detail` is always preferred when present: 3.1.2 verified
 * those messages are specific and actionable (e.g. "Unsupported file type:
 * .ogg. Supported formats: …"). This helper only supplies the fallback.
 */

/** Plain-language fallbacks, each naming the corrective action where one exists. */
function fallbackFor(status: number): string {
  switch (status) {
    case 400:
      return "The request was not valid. Check the values you entered and try again.";
    case 401:
      return "Your session has expired. Reload the page to start a new one — your uploaded audio will need to be uploaded again.";
    case 403:
      return "This item belongs to a different session and cannot be opened here.";
    case 404:
      return "That item no longer exists. It may have expired, or been deleted in another tab. Refresh the list and try again.";
    case 409:
      return "That result is not ready yet. Wait for the job to finish, then try again.";
    case 410:
      return "That result has expired. Results are kept for 24 hours; run the analysis again.";
    case 413:
      return "That file is too large. The limit is 100 MB and 10 minutes per file.";
    case 415:
      return "That file type is not supported. Use WAV, MP3, M4A or FLAC.";
    case 422:
      return "The request was rejected as invalid. Check the selected model, dataset and options.";
    case 429:
      return "Too many requests at once. Wait a few seconds, then try again.";
    case 503:
      return "The processing service is temporarily unavailable. Wait a moment and try again.";
    default:
      break;
  }
  if (status >= 500) {
    return "The server could not complete this request. Wait a moment and try again; if it keeps happening the analysis service may need restarting.";
  }
  return "The request could not be completed. Check your selection and try again.";
}

/**
 * FastAPI reports request-validation failures as an ARRAY of
 * `{ loc, msg, type }` entries rather than a string. Two of the call sites
 * flattened those to `msg` values before this helper existed; that behaviour is
 * preserved here so no call site loses detail by delegating.
 */
function messageFromDetail(detail: unknown): string | null {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((entry) =>
        entry && typeof entry === "object" && "msg" in entry
          ? String((entry as { msg: unknown }).msg)
          : null,
      )
      .filter((part): part is string => Boolean(part && part.trim()));
    if (parts.length) return parts.join("; ");
  }
  return null;
}

/**
 * Builds an Error from a failed Response, preferring the server's own `detail`.
 * Never puts a bare status code in the message.
 */
export async function describeHttpError(
  response: Response,
  context?: string,
): Promise<Error> {
  let detail: unknown;
  try {
    const body = await response.json();
    detail = body?.detail;
  } catch {
    detail = undefined;
  }
  const message = messageFromDetail(detail) ?? fallbackFor(response.status);
  return new Error(context ? `${context}: ${message}` : message);
}

/** Synchronous variant for callers that have already consumed the body. */
export function describeHttpStatus(status: number, context?: string): string {
  const message = fallbackFor(status);
  return context ? `${context}: ${message}` : message;
}
