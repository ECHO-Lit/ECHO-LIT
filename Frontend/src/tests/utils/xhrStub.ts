/**
 * XMLHttpRequest stand-in for §3.1.4 — the XHR counterpart of `stubFetch`.
 *
 * jsdom's own XHR performs real network I/O and emits no upload progress for a
 * FormData body, so byte-level progress (BUG-45) is unobservable through it.
 * This stand-in records every request and lets the case drive the transfer
 * explicitly: `progress(loaded, total)`, then `respond(...)` or `networkError()`.
 * Nothing happens unless the case says so, which is what makes "the bar shows
 * 25%" an assertion about the product rather than about timing.
 */
import { act } from "@testing-library/react";
import { vi } from "vitest";

export class FakeXMLHttpRequest {
  method = "";
  url = "";
  withCredentials = false;
  body: unknown = undefined;
  status = 0;
  responseText = "";
  aborted = false;
  readonly upload: { onprogress: ((event: ProgressEvent) => void) | null } = { onprogress: null };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;
  private responseHeaders: Record<string, string> = {};

  open(method: string, url: string) {
    this.method = method;
    this.url = url;
  }

  send(body: unknown) {
    this.body = body;
  }

  abort() {
    this.aborted = true;
    this.onabort?.();
  }

  getResponseHeader(name: string): string | null {
    return this.responseHeaders[name.toLowerCase()] ?? null;
  }

  // ---- controls for the test ------------------------------------------

  /** Report `loaded` of `total` bytes sent; total 0 means "not computable". */
  progress(loaded: number, total: number) {
    act(() => {
      this.upload.onprogress?.({ loaded, total, lengthComputable: total > 0 } as ProgressEvent);
    });
  }

  respond(status: number, json: unknown) {
    this.status = status;
    this.responseText = JSON.stringify(json);
    this.responseHeaders["content-type"] = "application/json";
    this.onload?.();
  }

  networkError() {
    this.onerror?.();
  }
}

export function stubXhr(): { requests: FakeXMLHttpRequest[] } {
  const requests: FakeXMLHttpRequest[] = [];
  class Recorded extends FakeXMLHttpRequest {
    constructor() {
      super();
      requests.push(this);
    }
  }
  vi.stubGlobal("XMLHttpRequest", Recorded);
  return { requests };
}
