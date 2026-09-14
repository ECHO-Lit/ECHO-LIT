/**
 * §3.1.4 Performance Profiling — client-side feedback latency.
 * Cases PF-01…PF-10. See Backend/tests/plans/3.1.4-performance-profiling.md.
 *
 * SRS US-3: "Every long-running operation shall provide immediate
 * acknowledgment (within 500 ms), a visible progress indication…"
 *
 * 3.1.3 excluded the 500 ms number because a wall-clock assertion under jsdom
 * measures the test runner, not the product. It is owned here by removing the
 * network from the question instead: the stand-in holds the request open, and
 * the case asserts the acknowledgment is on screen *while nothing has come
 * back*. An acknowledgment that exists before any server response is bounded by
 * one render, not by the API — which is what "immediate" means. The server half
 * of the budget is PE-1, profiled against the real API in the backend modules.
 *
 * Upload progress (BUG-45) is asserted through `stubXhr`, which lets the case
 * emit transfer events byte by byte.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { renderWithProviders, createTestQueryClient } from "./utils/render";
import { stubFetch, type RouteValue } from "./utils/fetchStub";
import { stubXhr } from "./utils/xhrStub";
import { groupableColumns, jobStatus } from "./utils/fixtures";

vi.mock("react-plotly.js", () => ({ default: () => <div /> }));

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => {});
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

const NEVER = () => new Promise<RouteValue>(() => {});

const accepted = {
  job_id: "job-0001", status: "queued", status_url: "/jobs/job-0001",
  result_url: "/jobs/job-0001/result", dataset: "common-voice",
  grouping_key: ["accent"], poll_after_ms: 1000, notes: [],
};

async function renderFairness(routes: Record<string, RouteValue>) {
  const stub = stubFetch({
    "GET /api/v1/analyses/fairness/groupable": { json: groupableColumns },
    ...routes,
  });
  const { FairnessPanel } = await import("@/components/fairness/FairnessPanel");
  renderWithProviders(<FairnessPanel model="whisper-base" dataset="common-voice" />);
  const run = await screen.findByRole("button", { name: /Run fairness analysis/ });
  await waitFor(() => expect(run).toBeEnabled());
  return { stub, run, user: userEvent.setup() };
}

describe("TestAcknowledgment", () => {
  it("PF-01 acknowledges a submission before the server has answered", async () => {
    // US-3 "immediate acknowledgment". The POST is held open for the whole
    // case, so everything asserted here happened with no response at all.
    const post = deferred<RouteValue>();
    const { stub, run, user } = await renderFairness({
      "POST /api/v1/analyses/fairness": () => post.promise,
    });

    await user.click(run);

    expect(stub.callsFor("POST /api/v1/analyses/fairness")).toHaveLength(1);
    expect(run).toBeDisabled();
    // A second click during submission sends nothing.
    await user.click(run);
    expect(stub.callsFor("POST /api/v1/analyses/fairness")).toHaveLength(1);
  });

  it("PF-02 stays acknowledged between acceptance and the first status poll", async () => {
    // Guards BUG-44. `isRunning` was derived from the first status response,
    // so for one full GET round trip after the 202 the mutation was no longer
    // pending and no status existed: the Run button re-enabled, the progress
    // indication vanished, and a second click submitted a duplicate job.
    const { stub, run, user } = await renderFairness({
      "POST /api/v1/analyses/fairness": { status: 202, json: accepted },
      "GET /jobs/job-0001": NEVER,
    });

    await user.click(run);
    await waitFor(() => expect(stub.callsFor("GET /jobs/job-0001")).toHaveLength(1));

    const status = await screen.findByRole("status");
    expect(within(status).getByText("Queued — waiting for a worker…")).toBeInTheDocument();
    expect(run).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled();
    await user.click(run);
    expect(stub.callsFor("POST /api/v1/analyses/fairness")).toHaveLength(1);
  });

  it("PF-03 hands over to the server's own progress once it arrives", async () => {
    const { run, user } = await renderFairness({
      "POST /api/v1/analyses/fairness": { status: 202, json: accepted },
      "GET /jobs/job-0001": {
        json: jobStatus("processing", {
          progress: { current: 2, total: 8, message: "Partitioning dataset" },
        }),
      },
    });

    await user.click(run);

    const status = await screen.findByRole("status");
    await waitFor(() => expect(within(status).getByText("Partitioning dataset")).toBeInTheDocument());
    expect(
      within(status).getByRole("progressbar", { name: "Fairness analysis progress" }),
    ).toHaveAttribute("aria-valuenow", "25");
    expect(run).toBeDisabled();
  });

  it("PF-04 does not strand a job at 'Queued' when its status cannot be read", async () => {
    // The BUG-44 fix treats "accepted, no status yet" as running. A status
    // request that FAILS must not be read the same way, or the panel would sit
    // at "Queued" forever; the failure is reported instead (US-4).
    stubFetch({ "GET /jobs/job-0001": { status: 500, json: {} } });
    const { useAnalysisJob } = await import("@/hooks/use-job-query");
    const client = createTestQueryClient();
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
    const { result } = renderHook(
      () => useAnalysisJob(async () => ({ job_id: "job-0001" }), "pf-04"),
      { wrapper },
    );

    await result.current.start(undefined);

    await waitFor(() => expect(result.current.error).toMatch(/server could not complete this request/));
    expect(result.current.isRunning).toBe(false);
  });
});

describe("TestProgressSemantics", () => {
  it("PF-10 exposes a progress bar's value to assistive technology", async () => {
    // Guards BUG-49, found by PF-03 and PF-08 rather than predicted. The
    // shadcn wrapper took `value` for the visual fill and never passed it to
    // Radix's Root, so every bar in the product — job progress, upload
    // progress — was announced as indeterminate however full it looked: a
    // screen-reader user got US-3's "visible progress indication" with the
    // progress removed.
    const { Progress } = await import("@/components/ui/progress");
    renderWithProviders(
      <>
        <Progress value={40} aria-label="determinate" />
        <Progress aria-label="indeterminate" />
      </>,
    );

    const determinate = screen.getByRole("progressbar", { name: "determinate" });
    expect(determinate).toHaveAttribute("aria-valuenow", "40");
    expect(determinate).toHaveAttribute("data-state", "loading");
    const indeterminate = screen.getByRole("progressbar", { name: "indeterminate" });
    expect(indeterminate).not.toHaveAttribute("aria-valuenow");
    expect(indeterminate).toHaveAttribute("data-state", "indeterminate");
  });
});

describe("TestUploadProgress", () => {
  it("PF-05 reports bytes sent and resolves with the server's JSON", async () => {
    // Guards BUG-45 at the transport: the helper sends the session cookie and
    // turns each transfer event into a fraction.
    const { requests } = stubXhr();
    const { uploadWithProgress } = await import("@/lib/upload");
    const seen: (number | null)[] = [];

    const pending = uploadWithProgress<{ total_files: number }>("http://api/upload/dataset/s/files", new FormData(), {
      onProgress: ({ fraction }) => seen.push(fraction),
    });
    const [xhr] = requests;
    xhr.progress(0, 0);
    xhr.progress(512, 2048);
    xhr.progress(2048, 2048);
    xhr.respond(200, { total_files: 2 });

    await expect(pending).resolves.toEqual({ total_files: 2 });
    expect(seen).toEqual([null, 0.25, 1]);
    expect(xhr.method).toBe("POST");
    expect(xhr.url).toBe("http://api/upload/dataset/s/files");
    expect(xhr.withCredentials).toBe(true);
  });

  it.each([
    [413, {}, "Upload failed: That file is too large. The limit is 100 MB and 10 minutes per file."],
    [404, { detail: "Dataset 'speech' does not exist" }, "Upload failed: Dataset 'speech' does not exist"],
  ])("PF-06 explains a %i without showing the status code", async (status, body, expected) => {
    // US-4, through the same describeHttpError the Fetch call sites use.
    const { requests } = stubXhr();
    const { uploadWithProgress } = await import("@/lib/upload");

    const pending = uploadWithProgress("http://api/u", new FormData(), { context: "Upload failed" });
    requests[0].respond(status, body);

    await expect(pending).rejects.toThrow(expected);
  });

  it("PF-07 cancels the transfer on abort and reports a lost connection plainly", async () => {
    const { requests } = stubXhr();
    const { uploadWithProgress } = await import("@/lib/upload");
    const controller = new AbortController();

    const aborted = uploadWithProgress("http://api/u", new FormData(), { signal: controller.signal });
    controller.abort();
    await expect(aborted).rejects.toMatchObject({ name: "AbortError" });
    expect(requests[0].aborted).toBe(true);

    const lost = uploadWithProgress("http://api/u", new FormData());
    requests[1].networkError();
    await expect(lost).rejects.toThrow(
      "The upload could not reach the server. Check your connection and try again.",
    );
  });

  it("PF-08 shows a dataset upload's real progress, then the server-side phase", async () => {
    // Guards BUG-45 end to end in the Custom Dataset Manager. 3.1.3's BUG-22
    // fix could only make the bar indeterminate; now it moves with the bytes,
    // and at 100% it says the server is still working rather than implying the
    // upload is done.
    const dataset = {
      dataset_name: "speech", formatted_name: "custom:sess:speech", created_at: "2026-09-10T00:00:00Z",
      session_id: "sess", files: [], total_files: 0,
    };
    stubFetch({ "GET /upload/dataset/list": { json: { datasets: [dataset] } } });
    const { requests } = stubXhr();
    const { CustomDatasetManager } = await import("@/components/dataset/CustomDatasetManager");
    renderWithProviders(<CustomDatasetManager />, { route: "/" });
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: /Manage Datasets/ }));
    screen.getByRole("tab", { name: "My Datasets" }).focus();
    await user.keyboard("{ArrowRight}{ArrowRight}");
    await user.selectOptions(await screen.findByLabelText("Select Dataset"), "speech");
    await user.upload(screen.getByLabelText("Audio Files"), [
      new File(["a"], "a.wav", { type: "audio/wav" }),
      new File(["b"], "b.wav", { type: "audio/wav" }),
    ]);
    await user.click(screen.getByRole("button", { name: "Upload Files" }));

    const [xhr] = requests;
    expect(xhr.url).toMatch(/\/upload\/dataset\/speech\/files$/);
    xhr.progress(250, 1000);
    const bar = screen.getByRole("progressbar", { name: "Upload progress" });
    expect(bar).toHaveAttribute("aria-valuenow", "25");
    expect(screen.getByText("Uploading 2 files — 25%")).toBeInTheDocument();

    xhr.progress(1000, 1000);
    expect(screen.getByText("Processing 2 files on the server…")).toBeInTheDocument();

    xhr.respond(200, {
      uploaded_files: [
        { filename: "a.wav", original_filename: "a.wav" },
        { filename: "b.wav", original_filename: "b.wav" },
      ],
      total_files: 2,
    });
    await waitFor(() => expect(screen.queryByRole("progressbar", { name: "Upload progress" })).toBeNull());
  });

  it("PF-09 keeps the upload panel usable when the browser cannot size the body", async () => {
    // lengthComputable is false for some bodies; the bar must fall back to
    // indeterminate rather than claim 0%.
    stubFetch({
      "GET /upload/dataset/list": {
        json: { datasets: [{ dataset_name: "speech", formatted_name: "x", created_at: "", session_id: "s", files: [], total_files: 0 }] },
      },
    });
    const { requests } = stubXhr();
    const { CustomDatasetManager } = await import("@/components/dataset/CustomDatasetManager");
    renderWithProviders(<CustomDatasetManager />, { route: "/" });
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /Manage Datasets/ }));
    screen.getByRole("tab", { name: "My Datasets" }).focus();
    await user.keyboard("{ArrowRight}{ArrowRight}");
    await user.selectOptions(await screen.findByLabelText("Select Dataset"), "speech");
    await user.upload(screen.getByLabelText("Audio Files"), [new File(["a"], "a.wav", { type: "audio/wav" })]);
    await user.click(screen.getByRole("button", { name: "Upload Files" }));

    requests[0].progress(300, 0);

    expect(screen.getByRole("progressbar", { name: "Upload progress" })).not.toHaveAttribute("aria-valuenow");
    expect(screen.getByText("Uploading 1 file…")).toBeInTheDocument();
  });
});
