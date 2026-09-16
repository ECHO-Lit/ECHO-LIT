/**
 * RUP §3.1.3 User Interface Testing — Module G: error presentation.
 * Cases UI-95…UI-109.
 *
 * SRS US-4: "Errors shall be presented as clear, actionable messages in plain
 * language rather than raw stack traces or HTTP codes and shall indicate the
 * corrective action where one exists."
 *
 * The error describer is exercised directly rather than through six components,
 * because it is the single seam every failure now passes through. Component
 * cases then confirm the seam is actually reached.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "./utils/render";
import { stubFetch } from "./utils/fetchStub";
import { baseRoutes } from "./utils/fixtures";
import { readComponentSource } from "./utils/source";

vi.mock("react-plotly.js", () => ({ default: () => <div /> }));

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => {});
});

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("TestServerMessages", () => {
  it("UI-95 surfaces a server detail verbatim", async () => {
    // The good path, asserted FIRST so the cases below cannot pass merely
    // because everything is broken. 3.1.2 verified these server messages are
    // specific and actionable, so the client must not paraphrase them.
    const { describeHttpError } = await import("@/lib/httpError");
    const detail =
      "Unsupported file type: .ogg. Supported formats: .wav, .mp3, .m4a, .flac";
    const error = await describeHttpError(jsonResponse(400, { detail }));
    expect(error.message).toBe(detail);
  });

  it("UI-96 never shows a raw HTTP status code", async () => {
    // Guards BUG-23. A failure without a JSON body used to yield
    // "Request failed (500)" — a raw code, which US-4 forbids in terms, with no
    // plain language and no corrective action.
    const { describeHttpError } = await import("@/lib/httpError");
    for (const status of [400, 404, 409, 413, 415, 422, 429, 500, 502, 503]) {
      const error = await describeHttpError(
        new Response("<html>gateway error</html>", { status }),
      );
      expect(error.message).not.toMatch(/\(\d{3}\)/);
      expect(error.message).not.toMatch(/\b(?:4|5)\d{2}\b/);
      expect(error.message.length).toBeGreaterThan(20);
      expect(error.message).toMatch(/\.$/);
    }
  });

  it("UI-97 gives every former error helper one shared contract", async () => {
    // Guards BUG-23. Five ad-hoc parseError/errorFor/readOrThrow copies existed
    // in lib/, each free to drift. All now delegate to describeHttpError.
    for (const file of [
      "src/lib/jobs.ts",
      "src/lib/models.ts",
      "src/lib/fairness.ts",
      "src/lib/linguisticAcoustic.ts",
      "src/lib/datasetLabels.ts",
    ]) {
      const src = readComponentSource(file);
      expect(src).toContain("describeHttpError");
      expect(src).not.toContain("Request failed (");
    }
  });

  it("UI-98 explains a lens-listing failure without a status code", async () => {
    // models.ts:34 was a sixth site, and not even one of the five helpers:
    // "Could not load Jacobian lenses (404)".
    const src = readComponentSource("src/lib/models.ts");
    expect(src).toContain(
      "describeHttpError(response, 'Could not load Jacobian lenses')",
    );
    const { describeHttpError } = await import("@/lib/httpError");
    const error = await describeHttpError(
      new Response("", { status: 404 }),
      "Could not load Jacobian lenses",
    );
    expect(error.message).toContain("Could not load Jacobian lenses:");
    expect(error.message).not.toMatch(/404/);
  });

  it("UI-99 explains an expired session and names the recovery", async () => {
    // Guards BUG-23b. grep for 401|403 across src/ returned NOTHING before this
    // fix, so an expired session read "Request failed (401)" with no hint that
    // reloading restores it — the single most likely error a real user meets.
    const { describeHttpError } = await import("@/lib/httpError");
    const unauthorised = await describeHttpError(new Response("", { status: 401 }));
    expect(unauthorised.message).toMatch(/session/i);
    expect(unauthorised.message).toMatch(/reload/i);
    expect(unauthorised.message).not.toMatch(/401/);

    const forbidden = await describeHttpError(new Response("", { status: 403 }));
    expect(forbidden.message).toMatch(/different session/i);
    expect(forbidden.message).not.toMatch(/403/);
  });

  it("UI-99b preserves FastAPI's array-shaped validation detail", async () => {
    // Two of the five helpers flattened a 422 detail array to its `msg` values.
    // Delegating must not lose that, or a validation failure would degrade from
    // a specific complaint to a generic sentence.
    const { describeHttpError } = await import("@/lib/httpError");
    const error = await describeHttpError(
      jsonResponse(422, {
        detail: [
          { loc: ["body", "max_items_per_group"], msg: "must be at least 2" },
          { loc: ["body", "dataset"], msg: "field required" },
        ],
      }),
    );
    expect(error.message).toBe("must be at least 2; field required");
  });
});

describe("TestSilentFailures", () => {
  it("UI-100 tells the user when the custom dataset list fails to load", async () => {
    // Guards BUG-25a. Toolbar.tsx:87 only reached console.error, so on failure
    // the dataset dropdown simply rendered with no custom datasets and no
    // explanation — the user concluded their uploads had been lost.
    stubFetch({
      ...baseRoutes,
      "GET /upload/dataset/list": { status: 500, text: "boom" },
    });
    const { Toolbar } = await import("@/components/layout/Toolbar");
    renderWithProviders(
      <Toolbar
        apiData={null}
        setApiData={() => {}}
        model="whisper-base"
        setModel={() => {}}
        dataset="common-voice"
        setDataset={() => {}}
        uploadedFiles={[]}
        selectedFile={null}
        onFileSelect={() => {}}
      />,
      { route: "/" },
    );
    await waitFor(() =>
      expect(
        screen.getByText(/Could not load your custom datasets/),
      ).toBeInTheDocument(),
    );
  });

  it("UI-101 tells the user when the registered model list fails to load", async () => {
    // Guards BUG-25b. Same shape at Toolbar.tsx:97: a registered model would
    // silently vanish from the picker.
    stubFetch({
      ...baseRoutes,
      "GET /models": { status: 503, text: "unavailable" },
    });
    const { Toolbar } = await import("@/components/layout/Toolbar");
    renderWithProviders(
      <Toolbar
        apiData={null}
        setApiData={() => {}}
        model="whisper-base"
        setModel={() => {}}
        dataset="common-voice"
        setDataset={() => {}}
        uploadedFiles={[]}
        selectedFile={null}
        onFileSelect={() => {}}
      />,
      { route: "/" },
    );
    await waitFor(() =>
      expect(
        screen.getByText(/Could not load your registered models/),
      ).toBeInTheDocument(),
    );
  });

  it("UI-102 reports a failed perturbation to the user", async () => {
    // Guards BUG-25c. PerturbationTools.tsx:206 was a user-INITIATED action
    // whose failure went only to the console: the user clicked Apply and
    // nothing whatsoever happened.
    const src = readComponentSource("src/components/analysis/PerturbationTools.tsx");
    expect(src).toContain('import { toast } from "sonner"');
    expect(src).toMatch(
      /console\.error\("Error adding perturbations:", err\);\s*\n\s*toast\.error\(/,
    );
  });

  it("UI-103 does not report a superseded request as a failure", async () => {
    // The counter-case to the three above. MainLayout and AudioDatasetPanel
    // deliberately swallow AbortError when a request is superseded; the BUG-25
    // fix must not turn those into user-visible noise.
    for (const file of [
      "src/components/layout/MainLayout.tsx",
      "src/components/panels/AudioDatasetPanel.tsx",
    ]) {
      expect(readComponentSource(file)).toMatch(/AbortError/);
    }
    const toolbar = readComponentSource("src/components/layout/Toolbar.tsx");
    // The two toasts added by BUG-25 sit on list fetches that carry no signal,
    // so they cannot fire for an aborted request.
    expect(toolbar).not.toContain("signal");
  });
});

describe("TestFabricatedOutput", () => {
  it("UI-107 shows no invented probabilities for an unrecognised model", async () => {
    // Guards BUG-15 (High). PredictionDisplay rendered Neutral 87% / Happy 8% /
    // Sad 3% / Angry 2% with "P" predicted and "T" true-label badges for EVERY
    // model that is neither whisper nor wav2vec2 — i.e. every custom model —
    // with nothing marking the numbers as fake. A researcher could screenshot
    // fabricated output from an interpretability tool and publish it.
    const { PredictionDisplay } = await import(
      "@/components/predictions/PredictionDisplay"
    );
    renderWithProviders(
      <PredictionDisplay
        selectedFile={{ file_id: "f1", filename: "clip.wav", message: "ok" }}
        model="cm-custom-model"
      />,
    );
    for (const label of ["Neutral", "Happy", "Sad", "Angry"]) {
      expect(screen.queryByText(label)).not.toBeInTheDocument();
    }
    expect(screen.queryByText("87%")).not.toBeInTheDocument();
    expect(
      screen.getByText(/No prediction view is available for this model yet/),
    ).toBeInTheDocument();
  });

  it("UI-107b keeps no hard-coded probability table in the source", async () => {
    const src = readComponentSource(
      "src/components/predictions/PredictionDisplay.tsx",
    );
    expect(src).not.toContain("probability: 0.87");
    expect(src).not.toContain('{ label: "Neutral", probability');
  });
});

describe("TestToastPlumbing", () => {
  it("UI-109 mounts exactly one toast system", async () => {
    // Guards BUG-29. App.tsx mounted BOTH the Radix <Toaster/> and <Sonner/>.
    // The Radix one was invoked by no feature code anywhere in src/, yet it
    // mounted a permanent live region and shipped TOAST_LIMIT = 1 with
    // TOAST_REMOVE_DELAY = 1000000 ms (~16.7 minutes) — booby traps for
    // whoever called it next.
    const app = readComponentSource("src/App.tsx");
    expect(app).toContain("Toaster as Sonner");
    expect(app).not.toContain('from "@/components/ui/toaster"');
    expect(() => readComponentSource("src/hooks/use-toast.ts")).toThrow();
  });
});
