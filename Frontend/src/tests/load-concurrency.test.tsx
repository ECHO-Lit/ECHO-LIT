/**
 * §3.1.5 Load Testing — how much load one client puts on the API.
 * Cases LC-01…LC-07. See Backend/tests/plans/3.1.5-load-testing.md.
 *
 * SRS PE-2: "Submitting a batch of many files shall not block the interface."
 *
 * The backend modules measure the API under many users. This module measures
 * the other side of the same budget: the burst a single researcher's browser
 * sends. Loading a dataset materialised every file at once
 * (`Promise.all(files.map(materializeAudio))`) and dropping a folder started
 * every upload at once. Under HTTP/2 the browser's six-connection cap does not
 * apply, so the whole burst reached the server together — each materialise
 * probes, hashes and copies a file on the API's shared thread pool.
 *
 * The oracle is the peak number of requests in flight, measured by a route
 * stand-in that holds each request open until the case releases it.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./utils/render";
import { stubFetch, type RecordedCall, type RouteValue } from "./utils/fetchStub";
import { readComponentSource } from "./utils/source";
import { audioReference } from "./utils/fixtures";

vi.mock("react-plotly.js", () => ({ default: () => <div /> }));

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => {});
});

/** A route that holds each request open and records peak concurrency. */
function gatedRoute(respond: (call: RecordedCall) => RouteValue) {
  const waiting: Array<() => void> = [];
  const state = { inFlight: 0, peak: 0, completed: 0 };
  const route = async (call: RecordedCall): Promise<RouteValue> => {
    state.inFlight += 1;
    state.peak = Math.max(state.peak, state.inFlight);
    await new Promise<void>((release) => waiting.push(release));
    state.inFlight -= 1;
    state.completed += 1;
    return respond(call);
  };
  /** Release held requests one at a time until `total` have completed. */
  const drain = async (total: number) => {
    while (state.completed < total) {
      await waitFor(() => expect(waiting.length).toBeGreaterThan(0));
      waiting.shift()!();
      await Promise.resolve();
    }
  };
  return { route, state, drain };
}

function deferredFlush() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

describe("TestBoundedMap", () => {
  it("LC-01 never exceeds its limit and keeps input order", async () => {
    const { mapWithConcurrency } = await import("@/lib/concurrency");
    let inFlight = 0;
    let peak = 0;

    // Later items finish first, so order can only survive by index, not luck.
    const results = await mapWithConcurrency([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, async (item) => {
      inFlight += 1;
      peak = Math.max(peak, inFlight);
      await new Promise((resolve) => setTimeout(resolve, 12 - item));
      inFlight -= 1;
      return item * 10;
    });

    expect(peak).toBe(3);
    expect(results).toEqual([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
  });

  it("LC-02 starts nothing new after the first failure", async () => {
    // Promise.all rejects on the first failure but every other request it
    // already created keeps running — for a 144-file dataset, 143 wasted
    // server-side probes after the batch is already lost.
    const { mapWithConcurrency } = await import("@/lib/concurrency");
    const started: number[] = [];

    const run = mapWithConcurrency(Array.from({ length: 50 }, (_, i) => i), 2, async (item) => {
      started.push(item);
      await deferredFlush();
      if (item === 1) throw new Error("materialize failed");
      return item;
    });

    await expect(run).rejects.toThrow("materialize failed");
    await deferredFlush();
    expect(started.length).toBeLessThanOrEqual(3);
  });

  it.each([0, -1, 1.5, Number.NaN])("LC-03 refuses a nonsensical limit (%s)", async (limit) => {
    const { mapWithConcurrency } = await import("@/lib/concurrency");
    await expect(mapWithConcurrency([1], limit, async (x) => x)).rejects.toThrow(RangeError);
  });
});

describe("TestClientBurst", () => {
  it("LC-04 materialises a 60-file dataset at most four requests at a time", async () => {
    // Guards BUG-46 at the shared helper every dataset-wide call site now uses.
    const gate = gatedRoute((call) => {
      const { filename } = JSON.parse(String(call.body));
      return { status: 201, json: audioReference({ audio_id: `id-${filename}`, filename }) };
    });
    stubFetch({ "POST /audio/materialize": gate.route });
    const { materializeAll, MATERIALIZE_CONCURRENCY } = await import("@/lib/jobs");
    const filenames = Array.from({ length: 60 }, (_, i) => `clip-${String(i).padStart(3, "0")}.wav`);

    const pending = materializeAll("ravdess", filenames);
    await gate.drain(filenames.length);
    const assets = await pending;

    expect(MATERIALIZE_CONCURRENCY).toBe(4);
    expect(gate.state.peak).toBe(MATERIALIZE_CONCURRENCY);
    // The layer probe joins assets to labels positionally, so order is part
    // of the contract, not a nicety.
    expect(assets.map((asset) => asset.audio_id)).toEqual(filenames.map((f) => `id-${f}`));
  });

  it("LC-05 uploads a dropped folder two files at a time", async () => {
    // Guards BUG-46 on the drop zone: `acceptedFiles.forEach(async …)` started
    // every upload — up to 100 MB each — in the same instant.
    const gate = gatedRoute(() => ({ status: 201, json: audioReference() }));
    stubFetch({ "POST /upload": gate.route });
    const { AudioUploader } = await import("@/components/audio/AudioUploader");
    const { container } = renderWithProviders(<AudioUploader model="whisper-base" />);
    const files = Array.from({ length: 6 }, (_, i) => new File([`${i}`], `take-${i}.wav`, { type: "audio/wav" }));

    await userEvent.setup().upload(container.querySelector('input[type="file"]') as HTMLInputElement, files);
    await gate.drain(files.length);

    expect(gate.state.peak).toBe(2);
    await waitFor(() => expect(screen.getAllByText(/^Uploaded: take-/)).toHaveLength(6));
  });

  it("LC-06 keeps uploading the rest of a folder after one file fails", async () => {
    // Bounded must not mean brittle: the uploader catches per file, so a
    // rejected file does not stop the queue behind it.
    const gate = gatedRoute((call) => {
      const name = (call.body as FormData).get("file") as File;
      return name.name === "take-1.wav"
        ? { status: 413, json: { detail: "Audio exceeds the 100 MB upload limit" } }
        : { status: 201, json: audioReference({ filename: name.name }) };
    });
    stubFetch({ "POST /upload": gate.route });
    const { AudioUploader } = await import("@/components/audio/AudioUploader");
    const { container } = renderWithProviders(<AudioUploader model="whisper-base" />);
    const files = Array.from({ length: 4 }, (_, i) => new File([`${i}`], `take-${i}.wav`, { type: "audio/wav" }));

    await userEvent.setup().upload(container.querySelector('input[type="file"]') as HTMLInputElement, files);
    await gate.drain(files.length);

    await waitFor(() => expect(screen.getAllByText(/^Uploaded: take-/)).toHaveLength(3));
    expect(
      screen.getByText("Failed to upload take-1.wav: Audio exceeds the 100 MB upload limit"),
    ).toBeInTheDocument();
  });

  it("LC-07 leaves no dataset-wide fan-out unbounded (source-level)", async () => {
    // Guards BUG-46 at every call site. A runtime case per site would need
    // each panel's full job lifecycle mounted; the property itself — no
    // `Promise.all` over materializeAudio — is visible in the source. The
    // J-Lens Lab is exempt: it already materialises in fixed batches of 8.
    const sites = [
      "src/components/panels/AudioDatasetPanel.tsx",
      "src/hooks/use-layer-probes.ts",
      "src/components/panels/EmbeddingPanel.tsx",
      "src/contexts/EmbeddingContext.tsx",
      "src/components/eda/DatasetEdaView.tsx",
      "src/components/visualization/ScalersVisualization.tsx",
    ];
    for (const site of sites) {
      const src = readComponentSource(site);
      expect(src, site).not.toMatch(/Promise\.all\(\s*[\w.]+\.map\([^)]*\)\s*=>\s*materializeAudio/);
      expect(src, site).toContain("materializeAll(");
    }
    const uploader = readComponentSource("src/components/audio/AudioUploader.tsx");
    expect(uploader).not.toMatch(/acceptedFiles\.forEach\(async/);
    expect(uploader).toContain("mapWithConcurrency(acceptedFiles");
  });
});

/**
 * A job a client abandons must not keep a worker. Saliency and attention share
 * one single-process queue, so an orphan from a quick file switch made every
 * later analysis sit at "Queued" behind it.
 */
describe("TestAbandonedJobs", () => {
  const status = (job_id: string, state: string) => ({
    json: {
      job_id, operation: "saliency", status: state, cache_hit: false,
      progress: { current: 0, total: 1, message: state },
    },
  });
  const input = { operation: "saliency" as const, audio_ids: ["a1"] };

  function jobRoutes(submit: RouteValue) {
    return stubFetch({
      "POST /jobs": submit,
      "GET /jobs/job-a": status("job-a", "queued"),
      "GET /jobs/job-b": status("job-b", "success"),
      "GET /jobs/job-b/result": { json: { items: [{ result: { from: "b" } }] } },
      "DELETE /jobs/*": { status: 202, json: {} },
    });
  }

  it("LC-08 cancels the running job when a newer run replaces it", async () => {
    const { renderHook, act } = await import("@testing-library/react");
    const { useJob } = await import("@/hooks/use-job");
    const ids = ["job-a", "job-b"];
    const stub = jobRoutes(() => ({ json: { job_id: ids.shift() } }));
    const { result } = renderHook(() => useJob());

    let first!: Promise<unknown>;
    act(() => { first = result.current.start(input); });
    const firstOutcome = first.catch((error: Error) => error.name);
    await waitFor(() => expect(stub.callsFor("GET /jobs/job-a")).toHaveLength(1));

    let second!: Promise<unknown>;
    await act(async () => { second = result.current.start(input); await second; });

    expect(stub.callsFor("DELETE /jobs/job-a")).toHaveLength(1);
    expect(await firstOutcome).toBe("AbortError");
    expect(result.current.result).toEqual({ items: [{ result: { from: "b" } }] });
  });

  it("LC-09 cancels a job superseded while its submit was still in flight", async () => {
    // The id only exists once the POST answers; the cancel must follow it.
    const { renderHook, act } = await import("@testing-library/react");
    const { useJob } = await import("@/hooks/use-job");
    let releaseFirst!: () => void;
    const firstSubmit = new Promise<void>((release) => { releaseFirst = release; });
    let submits = 0;
    const stub = jobRoutes(async () => {
      submits += 1;
      if (submits === 1) {
        await firstSubmit;
        return { json: { job_id: "job-a" } };
      }
      return { json: { job_id: "job-b" } };
    });
    const { result } = renderHook(() => useJob());

    let first!: Promise<unknown>;
    act(() => { first = result.current.start(input); });
    const firstOutcome = first.catch((error: Error) => error.name);
    await waitFor(() => expect(stub.callsFor("POST /jobs")).toHaveLength(1));
    await act(async () => { await result.current.start(input); });
    releaseFirst();

    expect(await firstOutcome).toBe("AbortError");
    await waitFor(() => expect(stub.callsFor("DELETE /jobs/job-a")).toHaveLength(1));
    // Never polled: the abort is honoured before the first status read.
    expect(stub.callsFor("GET /jobs/job-a")).toHaveLength(0);
  });

  it("LC-10 never sends DELETE for a job that already finished", async () => {
    // DELETE on a terminal job removes its result (and any generated audio)
    // rather than cancelling it.
    const { renderHook, act } = await import("@testing-library/react");
    const { useJob } = await import("@/hooks/use-job");
    const stub = jobRoutes({ json: { job_id: "job-b" } });
    const { result } = renderHook(() => useJob());

    await act(async () => { await result.current.start(input); });
    await act(async () => { await result.current.start(input); });

    expect(stub.callsFor("DELETE /jobs/*")).toHaveLength(0);
  });
});
