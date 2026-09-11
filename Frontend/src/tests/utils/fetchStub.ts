/**
 * Route-table `fetch` stand-in — the frontend analogue of 3.1.2's `fake_broker`.
 *
 * Two properties matter, and both are deliberate:
 *
 * 1. **An unrouted request throws.** The datasets router is mounted at the API
 *    root and the client fires a request per panel on mount. A permissive mock
 *    returning `{}` would let a case pass because the component silently
 *    absorbed a shape it never asked for — the frontend equivalent of the
 *    invented Redis keys 3.1.1 removed. Every request a test expects must be
 *    declared.
 *
 * 2. **Failure branches are produced, never hoped for.** A 500 or a network
 *    error is an explicit route entry, never an unmocked call that happens to
 *    fail. This mirrors 3.1.2's discipline of raising from the stand-in.
 *
 * Calls are recorded so a test can assert dispatch (method, URL, body) the way
 * 3.1.2 asserts `send_task` arguments.
 */
import { vi } from "vitest";

export interface RecordedCall {
  method: string;
  url: string;
  body: BodyInit | null | undefined;
  headers: HeadersInit | undefined;
}

/**
 * A route value: a literal body (JSON-encoded), a Response, or a factory. A
 * factory may return a Promise, which holds the request open until it settles
 * — how 3.1.4/3.1.5 observe what the UI shows while a request is in flight and
 * how many requests are in flight at once.
 */
export type RouteValue =
  | { status?: number; json?: unknown; text?: string; headers?: Record<string, string> }
  | Response
  | Error
  | ((request: RecordedCall) => RouteValue | Promise<RouteValue>);

/** Key form: `"GET /jobs/abc"`, or `"POST *"` / `"GET /jobs/*"` for a prefix. */
export type RouteTable = Record<string, RouteValue>;

export interface FetchStub {
  calls: RecordedCall[];
  /** Calls matching a `"METHOD /path"` key form, prefix-matched on `*`. */
  callsFor(key: string): RecordedCall[];
  /** Replace or add routes mid-test (e.g. a job that transitions to success). */
  setRoute(key: string, value: RouteValue): void;
}

function pathOf(url: string): string {
  try {
    return new URL(url, "http://localhost:8000").pathname;
  } catch {
    return url;
  }
}

function matches(key: string, method: string, path: string): boolean {
  const [keyMethod, keyPath] = key.split(" ");
  if (keyMethod !== method) return false;
  if (keyPath === "*") return true;
  if (keyPath.endsWith("/*")) return path.startsWith(keyPath.slice(0, -1));
  return keyPath === path;
}

async function toResponse(value: RouteValue, call: RecordedCall): Promise<Response> {
  if (typeof value === "function") return toResponse(await value(call), call);
  if (value instanceof Error) throw value;
  if (value instanceof Response) return value;
  const { status = 200, json, text, headers } = value;
  const body =
    text !== undefined ? text : json !== undefined ? JSON.stringify(json) : "";
  return new Response(body, {
    status,
    headers: {
      "Content-Type": text !== undefined ? "text/plain" : "application/json",
      ...(headers ?? {}),
    },
  });
}

export function stubFetch(routes: RouteTable): FetchStub {
  const table: RouteTable = { ...routes };
  const calls: RecordedCall[] = [];

  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url =
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.toString()
            : input.url;
      const method = (init?.method ?? "GET").toUpperCase();
      const call: RecordedCall = {
        method,
        url,
        body: init?.body,
        headers: init?.headers,
      };
      calls.push(call);

      if (init?.signal?.aborted) {
        throw new DOMException("Aborted", "AbortError");
      }

      const path = pathOf(url);
      const key = Object.keys(table).find((k) => matches(k, method, path));
      if (!key) {
        throw new Error(
          `Unrouted request: ${method} ${url}\n` +
            `Declared routes: ${Object.keys(table).join(", ") || "(none)"}`,
        );
      }
      return toResponse(table[key], call);
    }),
  );

  return {
    calls,
    callsFor(key: string) {
      return calls.filter((c) => matches(key, c.method, pathOf(c.url)));
    },
    setRoute(key: string, value: RouteValue) {
      table[key] = value;
    },
  };
}
