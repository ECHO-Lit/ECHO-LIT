/**
 * RUP §3.1.8 Configuration Testing — Module CW: the client under the
 * configurations it is deployed in. Cases CF-100…CF-110.
 *
 * The client has one configuration input, `VITE_API_BASE_URL`, and one
 * environment it cannot choose: the origin the user opens it from. Both are
 * varied here. The production image and the browser baseline are checked at
 * source level (the only layer where they are observable without building
 * images or launching browsers), and every such case says so in its name.
 *
 * SRS 3.9.3: "Supported browsers are Chrome, Firefox, Safari and Edge".
 * SRS SE-2: "Session cookies shall be HttpOnly, with Secure and SameSite".
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { waitFor } from "@testing-library/react";
import { existsSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import { renderWithProviders } from "./utils/render";
import { stubFetch } from "./utils/fetchStub";
import { readComponentSource } from "./utils/source";

const wave = vi.hoisted(() => {
  const created: Record<string, unknown>[] = [];
  const loads: unknown[] = [];
  const instance = {
    on: () => {},
    un: () => {},
    destroy: () => {},
    load: (arg: unknown) => {
      loads.push(arg);
      return Promise.resolve();
    },
    play: () => {},
    pause: () => {},
    getDuration: () => 0,
    getCurrentTime: () => 0,
  };
  return { created, loads, instance };
});

vi.mock("wavesurfer.js", () => ({
  default: {
    create: (options: Record<string, unknown>) => {
      wave.created.push(options);
      return wave.instance;
    },
  },
}));

type Resolver = (
  configured: string | undefined,
  location: Pick<Location, "protocol" | "hostname"> | undefined,
) => string;

async function apiModule(): Promise<{ API_BASE: string; resolveApiBase?: Resolver }> {
  vi.resetModules();
  return (await import("@/lib/api")) as { API_BASE: string; resolveApiBase?: Resolver };
}

/** KEY=VALUE and `# KEY=VALUE` lines of an env file. */
function readEnv(relativePath: string) {
  const active = new Map<string, string>();
  const commented = new Map<string, string>();
  for (const raw of readComponentSource(relativePath).split(/\r?\n/)) {
    const line = raw.trim();
    const target = line.startsWith("#") ? commented : active;
    const match = line.replace(/^#+\s*/, "").match(/^([A-Z][A-Z0-9_]*)=(\S*)/);
    if (match) target.set(match[1], match[2]);
  }
  return { active, commented };
}

function sourceFiles(): string[] {
  const root = resolve(process.cwd(), "src");
  return (readdirSync(root, { recursive: true }) as string[])
    .map((file) => file.replace(/\\/g, "/"))
    .filter((file) => /\.(ts|tsx)$/.test(file) && !file.startsWith("tests/"))
    .map((file) => `src/${file}`);
}

/** The text of one Dockerfile stage, `FROM ... AS name` up to the next FROM. */
function stage(dockerfile: string, name: string): string {
  const match = dockerfile.match(new RegExp(`^FROM [^\\n]* AS ${name}\\n([\\s\\S]*?)(?=^FROM |$(?![\\s\\S]))`, "m"));
  return match ? match[1] : "";
}

beforeEach(() => {
  wave.created.length = 0;
  wave.loads.length = 0;
  vi.spyOn(console, "log").mockImplementation(() => {});
  vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
});

describe("TestApiBase", () => {
  it.each([
    ["http://localhost:8080", "http://localhost:8000"],
    ["http://127.0.0.1:8080", "http://127.0.0.1:8000"],
    ["https://echo.example.org", "https://echo.example.org:8000"],
  ])("CF-100 derives a same-site API base when none is configured (page at %s)", async (origin, expected) => {
    // Guards BUG-65. The client always called http://localhost:8000. From
    // http://127.0.0.1:8080 — an origin the API admits — that is another
    // site, so the Lax session cookie was never sent and every request began
    // a new, empty session.
    const { resolveApiBase } = await apiModule();
    expect(typeof resolveApiBase).toBe("function");
    expect(resolveApiBase!(undefined, new URL(origin))).toBe(expected);
  });

  it.each([
    ["http://x.test:8000/", "http://x.test:8000"],
    ["   ", "http://localhost:8000"],
    ["https://api.example.org", "https://api.example.org"],
  ])("CF-101 normalises a configured API base (%j)", async (configured, expected) => {
    // Guards BUG-65. A trailing slash produced `//jobs` paths the API does
    // not route; a blank value must mean "unset", not "same page".
    const { resolveApiBase } = await apiModule();
    expect(typeof resolveApiBase).toBe("function");
    expect(resolveApiBase!(configured, new URL("http://localhost:8080"))).toBe(expected);
  });

  it("CF-102 resolves to http://localhost:8000 under the default dev origin", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "");
    const { API_BASE } = await apiModule();
    expect(API_BASE).toBe("http://localhost:8000");
  });

  it("CF-103 gives every backend-admitted origin a same-site API base (source-level origins)", async () => {
    // Guards BUG-65. The origins the backend ships admitting are read from
    // its configuration files; the client's configured base from its own.
    const origins = new Set<string>();
    const backendExample = readEnv("../Backend/.env.example");
    for (const origin of (backendExample.active.get("ALLOWED_ORIGINS") ?? "").split(",")) {
      if (origin) origins.add(origin);
    }
    const defaults = readComponentSource("../Backend/app/core/settings.py").match(
      /ALLOWED_ORIGINS:\s*str\s*=\s*"([^"]+)"/,
    );
    for (const origin of (defaults?.[1] ?? "").split(",")) if (origin) origins.add(origin);
    expect(origins.size).toBeGreaterThan(0);

    const configured = readEnv(".env.example").active.get("VITE_API_BASE_URL");
    const { resolveApiBase } = await apiModule();
    const resolver: Resolver = resolveApiBase ?? ((value) => value || "http://localhost:8000");
    for (const origin of origins) {
      const page = new URL(origin);
      expect(new URL(resolver(configured, page)).hostname, origin).toBe(page.hostname);
    }
  });

  it("CF-104 reads only documented env keys and leaves the API base to derivation (source-level)", () => {
    // Guards BUG-65. `.env.example` pinned localhost:8000, which every copy
    // made with the README's quickstart inherited.
    const env = readEnv(".env.example");
    const documented = new Set([...env.active.keys(), ...env.commented.keys()]);
    const read = new Set<string>();
    for (const file of sourceFiles()) {
      for (const match of readComponentSource(file).matchAll(/import\.meta\.env\.([A-Z_]+)/g)) read.add(match[1]);
    }
    for (const key of read) expect(documented.has(key), key).toBe(true);
    expect(env.active.has("VITE_API_BASE_URL")).toBe(false);
  });
});

describe("TestWaveformCredentials", () => {
  it.each([
    "http://localhost:8000/audio/a1",
    "http://127.0.0.1:8000/audio/a1",
    "https://api.example.org/audio/a1",
  ])("CF-105 loads the waveform with the session cookie (%s)", async (audioUrl) => {
    // Guards BUG-66. `/audio/{id}` answers only GET, so the viewer's HEAD
    // probe always got 405; for an http:// URL it then handed the bare URL to
    // wavesurfer, which fetches without credentials — no cookie, no session,
    // 404. Only https/ngrok/colab URLs took the credentialed blob path.
    const stub = stubFetch({
      "HEAD /audio/a1": { status: 405, json: { detail: "Method Not Allowed" } },
      "GET /audio/a1": { status: 200, text: "RIFF" },
    });
    const { WaveformViewer } = await import("@/components/audio/WaveformViewer");
    renderWithProviders(<WaveformViewer audioUrl={audioUrl} />);

    await waitFor(() => expect(wave.loads.length).toBeGreaterThan(0));
    const loaded = wave.loads.at(-1);
    const options = wave.created.at(-1) as { fetchParams?: RequestInit } | undefined;
    const viaCredentialedBlob =
      typeof loaded === "string" &&
      loaded.startsWith("blob:") &&
      stub.calls.some((call) => call.method === "GET" && call.credentials === "include");
    const viaWavesurfer = loaded === audioUrl && options?.fetchParams?.credentials === "include";
    expect(viaCredentialedBlob || viaWavesurfer).toBe(true);
  });
});

describe("TestProductionImage", () => {
  // Working copies are CRLF on Windows (core.autocrlf); stages are matched by line.
  const dockerfile = readComponentSource("Dockerfile").replace(/\r\n/g, "\n");

  it("CF-106 build stage accepts the API base as a build argument (source-level)", () => {
    // Guards BUG-76. `.dockerignore` excludes `.env`, and the build stage
    // took no argument, so every production bundle called localhost:8000.
    const build = stage(dockerfile, "build");
    const argAt = build.indexOf("ARG VITE_API_BASE_URL");
    const envAt = build.indexOf("ENV VITE_API_BASE_URL");
    const buildAt = build.indexOf("RUN npm run build");
    expect(argAt).toBeGreaterThanOrEqual(0);
    expect(envAt).toBeGreaterThan(argAt);
    expect(buildAt).toBeGreaterThan(envAt);
  });

  it("CF-107 nginx serves every client route through the SPA fallback (source-level)", () => {
    // Guards BUG-76. Default nginx answered a deep link to /j-lens with 404.
    expect(existsSync(resolve(process.cwd(), "nginx.conf"))).toBe(true);
    const nginx = readComponentSource("nginx.conf");
    const root = nginx.replace(/\r\n/g, "\n").match(/location\s+\/\s*\{([^}]*)\}/);
    expect(root?.[1]).toMatch(/try_files\s+\$uri\s+\$uri\/\s+\/index\.html;/);
    expect(stage(dockerfile, "prod")).toMatch(/COPY nginx\.conf \/etc\/nginx\/conf\.d\/default\.conf/);
    const ignored = readComponentSource(".dockerignore").split(/\r?\n/).map((line) => line.trim());
    expect(ignored).not.toContain("nginx.conf");
    const routes = [...readComponentSource("src/App.tsx").matchAll(/<Route path="([^"]+)"/g)].map((m) => m[1]);
    expect(routes).toContain("/j-lens");
    for (const route of routes.filter((path) => path !== "*")) expect(route.startsWith("/")).toBe(true);
  });

  it("CF-108 the build stage installs the devDependencies vite build needs (source-level)", () => {
    // Guards BUG-77: vite.config.ts claimed the Dockerfile runs
    // `npm ci --omit=dev`; it does not, and vite is a devDependency.
    expect(dockerfile).toMatch(/^FROM deps AS build$/m);
    expect(stage(dockerfile, "deps")).toMatch(/npm ci(?! --omit=dev)/);
    const pkg = JSON.parse(readComponentSource("package.json"));
    expect(pkg.devDependencies.vite).toBeDefined();
    const config = readComponentSource("vite.config.ts");
    if (config.includes("--omit=dev") && /Dockerfile/.test(config)) {
      expect(dockerfile).toContain("--omit=dev");
    }
  });
});

describe("TestBrowserBaseline", () => {
  it("CF-109 builds for Vite 5's default browser baseline (source-level)", () => {
    // OBS-49. §3.9.3 names browsers without versions; the build therefore
    // targets Vite 5's default. A Vite major upgrade moves that floor and
    // must be a deliberate decision — this case fails when it happens.
    const vite = JSON.parse(readComponentSource("node_modules/vite/package.json"));
    expect(Number(vite.version.split(".")[0])).toBe(5);
    const constants = readComponentSource("node_modules/vite/dist/node/constants.js");
    for (const browser of ["chrome87", "edge88", "firefox78", "safari14"]) expect(constants).toContain(browser);
    expect(readComponentSource("vite.config.ts")).not.toMatch(/build:\s*\{[^}]*target/);
    expect(JSON.parse(readComponentSource("package.json")).browserslist).toBeUndefined();
  });

  it("CF-110 uses no runtime API above the baseline without a guard (source-level)", () => {
    // esbuild lowers syntax (`||=`, `?.`) to the target; it does not
    // polyfill runtime APIs. These arrived after Safari 14 / Firefox 78.
    const aboveBaseline = [
      /\bstructuredClone\(/,
      /\bcrypto\.randomUUID\(/,
      /\bObject\.hasOwn\(/,
      /\.at\(-?\d/,
      /\.findLast(Index)?\(/,
      /\.to(Sorted|Reversed|Spliced)\(/,
      /\.replaceAll\(/,
      /\bAbortSignal\.(timeout|any)\(/,
      /\brequestIdleCallback\(/,
      /\bBroadcastChannel\b/,
      /\bPromise\.any\(/,
      /\bWeakRef\b/,
    ];
    const offenders: string[] = [];
    for (const file of sourceFiles()) {
      const text = readComponentSource(file);
      for (const pattern of aboveBaseline) if (pattern.test(text)) offenders.push(`${file}: ${pattern}`);
      if (/new \(?window\.AudioContext/.test(text) && !text.includes("webkitAudioContext")) {
        offenders.push(`${file}: AudioContext without the webkit fallback`);
      }
    }
    expect(offenders).toEqual([]);
  });
});
