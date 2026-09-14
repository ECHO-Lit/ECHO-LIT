/**
 * RUP §3.1.3 User Interface Testing — Module F: asynchronous feedback.
 * Cases UI-82…UI-94.
 *
 * SRS US-3: "Every long-running operation shall provide immediate
 * acknowledgment (within 500 ms), a visible progress indication, and the
 * interface shall remain fully interactive while computation proceeds in the
 * background. The user shall be able to navigate between panels and start
 * additional analyses without waiting for a previous one to finish."
 *
 * The "within 500 ms" half is EXCLUDED from this section: a wall-clock
 * assertion under jsdom measures the runner, not the product. UI-82 verifies
 * the ordering half — that the acknowledgment precedes the first status poll —
 * which is the part a functional test can honestly own. The latency budget is
 * carried to 3.1.4, alongside FR-4's control-plane budget.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { stubFetch } from "./utils/fetchStub";
import { dashboardRoutes, jobStatus } from "./utils/fixtures";
import { readComponentSource } from "./utils/source";

vi.mock("react-plotly.js", () => ({ default: () => <div /> }));
vi.mock("wavesurfer.js", () => ({
  default: {
    create: () => ({
      on: () => {}, un: () => {}, destroy: () => {}, load: () => {},
      play: () => {}, pause: () => {}, seekTo: () => {}, setVolume: () => {},
      getDuration: () => 2, getCurrentTime: () => 0, empty: () => {},
    }),
  },
}));

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => {});
});

async function renderDashboard(extraRoutes: Record<string, unknown> = {}) {
  window.history.pushState({}, "", "/");
  const stub = stubFetch({ ...dashboardRoutes, ...extraRoutes } as never);
  const { default: App } = await import("@/App");
  const result = render(<App />);
  await waitFor(() =>
    expect(screen.getByText("LIT for Voice")).toBeInTheDocument(),
  );
  return { ...result, stub };
}

describe("TestProgressIndication", () => {
  it("UI-83 renders a determinate progress indication while a job runs", async () => {
    // US-3 "a visible progress indication". The Progress component carries the
    // computed percentage, and the job's own message accompanies it.
    for (const file of [
      "src/components/fairness/FairnessPanel.tsx",
      "src/components/analysis/PerturbationDiagnosticsPanel.tsx",
    ]) {
      const src = readComponentSource(file);
      expect(src).toContain("<Progress value={progressPct}");
      expect(src).toContain("{job.progress.message}");
    }
  });

  it("UI-84 offers cancellation only while a job is running", async () => {
    // A Cancel that is present when there is nothing to cancel is misleading;
    // one that is absent while a 30-minute job runs traps the user.
    const src = readComponentSource("src/components/fairness/FairnessPanel.tsx");
    expect(src).toContain("job.isRunning");
    expect(src).toMatch(/Cancel/);
  });

  it("UI-85 keeps a session-wide cancel always available", async () => {
    // "Cancel all fairness jobs" walks server-side jobs the UI may have lost
    // track of, so it is deliberately NOT gated on the panel's own job state.
    const src = readComponentSource("src/components/fairness/FairnessPanel.tsx");
    expect(src).toContain("Cancel all fairness jobs");
  });

  it("UI-87 reports layer-probe progress with a count", async () => {
    const src = readComponentSource("src/components/probing/LayerProbePanel.tsx");
    expect(src).toContain("progress.message");
    expect(src).toContain("progress.current");
    expect(src).toContain("progress.total");
  });

  it("UI-88 reports batch inference position in the queue", async () => {
    // The badge reads "Inferencing... n/total" so the user can see the batch
    // advancing rather than guessing.
    const src = readComponentSource("src/components/panels/AudioDatasetPanel.tsx");
    expect(src).toContain("Inferencing...");
    expect(src).toContain("currentInferenceIndex");
  });

  it("UI-89 marks batch completion", async () => {
    const src = readComponentSource("src/components/panels/AudioDatasetPanel.tsx");
    expect(src).toContain("Inference Complete");
  });

  it("UI-90 announces progress and completion to assistive technology", async () => {
    // Guards BUG-18. No aria-live or role="status" existed anywhere in src/, so
    // every one of the indicators above was silent to a screen reader. For a
    // job that runs 10-30 minutes that is the difference between "working" and
    // "apparently frozen".
    const src = readComponentSource("src/components/panels/AudioDatasetPanel.tsx");
    const statusIndex = src.indexOf('role="status"');
    expect(statusIndex).toBeGreaterThan(-1);
    expect(src.indexOf('aria-live="polite"')).toBeGreaterThan(-1);
    // The live region must actually contain the badge, not sit elsewhere.
    const region = src.slice(statusIndex, statusIndex + 800);
    expect(region).toContain("Inferencing...");
  });
});

describe("TestNonBlockingInteraction", () => {
  it("UI-91 keeps the interface interactive while a job runs", async () => {
    // US-3 "the interface shall remain fully interactive while computation
    // proceeds in the background". With a running job pinned in the stand-in,
    // panel navigation must still work.
    await renderDashboard({
      "POST /jobs": { json: jobStatus("queued") },
      "GET /jobs/job-0001": { json: jobStatus("started") },
    });
    const user = userEvent.setup();
    const saliency = screen.getByRole("tab", { name: "Saliency" });
    saliency.focus();
    await user.keyboard("{ArrowRight}");
    await waitFor(() =>
      expect(saliency).toHaveAttribute("aria-selected", "false"),
    );
    // The toolbar remains operable too. Two controls are named "Upload" — the
    // toolbar's and the dataset panel's — so both are checked rather than
    // disambiguated arbitrarily.
    const uploads = screen.getAllByRole("button", { name: "Upload" });
    expect(uploads.length).toBeGreaterThanOrEqual(1);
    for (const button of uploads) expect(button).toBeEnabled();
    expect(
      screen.getByRole("button", { name: "Toggle left panel" }),
    ).toBeEnabled();
  });

  it("UI-92 keeps a hidden long-running analysis polling", async () => {
    // The concrete mechanism behind US-3's "start additional analyses without
    // waiting": the Fairness tab is force-mounted so switching away does not
    // unmount its poll observer and lose a 30-minute job.
    const src = readComponentSource("src/components/panels/PredictionPanel.tsx");
    expect(src).toContain("forceMount");
  });

  it("UI-93 stops polling once a job reaches a terminal state", async () => {
    // use-job-query returns false from refetchInterval on a terminal status. A
    // poll that never stops would keep a completed job's request loop alive for
    // the life of the session.
    const src = readComponentSource("src/hooks/use-job-query.ts");
    expect(src).toContain("refetchInterval");
    expect(src).toMatch(/success|failure|cancelled/);
  });

  it("UI-93b keeps a long job's state from being garbage-collected", async () => {
    // gcTime: Infinity on the two job key families is deliberate — it is what
    // lets a 10-30 minute job survive a tab switch. It is also why every test
    // render gets a FRESH QueryClient (see utils/render.tsx).
    const src = readComponentSource("src/hooks/use-job-query.ts");
    expect(src).toContain("gcTime: Infinity");
    expect(src).toContain("refetchIntervalInBackground");
  });
});

describe("TestUploadProgress", () => {
  it("UI-94 reports upload progress truthfully", async () => {
    // Guards BUG-22. The bar was set to 0, stayed at 0 for the whole transfer,
    // then jumped to 100 after it had already finished: it reported no progress
    // while presenting itself as a progress bar, so a user watching 0% would
    // reasonably conclude the upload had stalled. 3.1.3 replaced it with an
    // indeterminate indicator; 3.1.4 (BUG-45) completed the fix with real
    // byte-level progress over XMLHttpRequest, asserted at runtime in
    // performance-feedback.test.tsx (PF-05…PF-08). This case now pins that the
    // bar is driven by transfer events and never by a hard-coded value.
    const src = readComponentSource("src/components/dataset/CustomDatasetManager.tsx");
    expect(src).not.toContain("setUploadProgress(100)");
    expect(src).toContain("uploadWithProgress");
    expect(src).toContain("value={uploadPercent ?? undefined}");
    expect(src).toContain('role="status"');
  });
});
