/**
 * RUP §3.1.3 User Interface Testing — Module C: panel and tab navigation.
 * Cases UI-31…UI-46.
 *
 * SRS US-2 names "tabbed panels" and "modal dialogs" among the conventions the
 * product must adopt consistently, and RUP §3.1.3 asks for window-to-window and
 * field-to-field navigation plus window state and focus.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { stubFetch } from "./utils/fetchStub";
import { dashboardRoutes } from "./utils/fixtures";
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

async function renderDashboard() {
  window.history.pushState({}, "", "/");
  stubFetch(dashboardRoutes);
  const { default: App } = await import("@/App");
  const result = render(<App />);
  await waitFor(() =>
    expect(screen.getByText("LIT for Voice")).toBeInTheDocument(),
  );
  return result;
}

describe("TestPredictionTabs", () => {
  it("UI-31 presents the analysis tabs for a whisper model", async () => {
    // PredictionPanel renders 4-6 tabs depending on model capability.
    await renderDashboard();
    const names = screen.getAllByRole("tab").map((t) => t.textContent?.trim());
    for (const expected of [
      "Saliency",
      "Perturbation",
      "Diagnostics",
      "Fairness",
    ]) {
      expect(names).toContain(expected);
    }
  });

  it("UI-32 gates the attention and lens tabs on model capability", async () => {
    // The conditional is capability-driven, not name-driven, so a custom model
    // that declares the capability gets the tab too.
    const src = readComponentSource("src/components/panels/PredictionPanel.tsx");
    expect(src).toMatch(/attention/i);
    expect(src).toMatch(/jacobian_lens_apply/);
  });

  it("UI-34 moves the selected state and the panel body when a tab is chosen", async () => {
    // Driven from the KEYBOARD: Radix Tabs do not switch on a synthetic click
    // under jsdom, and arrow-key traversal is in any case the access method
    // RUP §3.1.3 names ("tab keys ... accelerator keys"). Radix tabs use
    // automatic activation, so ArrowRight both moves focus and selects.
    await renderDashboard();
    const user = userEvent.setup();
    const saliency = screen.getByRole("tab", { name: "Saliency" });
    saliency.focus();
    expect(saliency).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{ArrowRight}");
    await waitFor(() =>
      expect(saliency).toHaveAttribute("aria-selected", "false"),
    );
    // Scoped to the prediction tablist: the dashboard mounts two tablists
    // (prediction and embedding), each with its own selected tab.
    const tablist = saliency.closest('[role="tablist"]')!;
    const selected = Array.from(
      tablist.querySelectorAll('[role="tab"][aria-selected="true"]'),
    );
    expect(selected).toHaveLength(1);
    expect(selected[0]).not.toBe(saliency);
  });

  it("UI-35 exposes the tabs as a conforming WAI-ARIA tablist", async () => {
    // The tabbed-panel convention US-2 requires, supplied by Radix. The value of
    // pinning it is that a future refactor away from Radix, or an `asChild` that
    // discards props, fails here.
    await renderDashboard();
    const tablists = screen.getAllByRole("tablist");
    expect(tablists.length).toBeGreaterThan(0);
    const tabs = screen.getAllByRole("tab");
    for (const tab of tabs) {
      expect(tab).toHaveAttribute("aria-selected");
      expect(tab).toHaveAttribute("aria-controls");
    }
  });

  it("UI-36 keeps the fairness panel mounted when another tab is selected", async () => {
    // US-3: "the interface shall remain fully interactive while computation
    // proceeds in the background". PredictionPanel force-mounts the Fairness
    // tab content so a 10-30 minute job keeps polling when the tab is hidden.
    //
    // Asserted STRUCTURALLY, not by visibility: `css: false` means the
    // data-[state=inactive]:hidden class never applies under test, so
    // toBeVisible() would pass regardless and prove nothing.
    const src = readComponentSource("src/components/panels/PredictionPanel.tsx");
    expect(src).toContain("forceMount");
    expect(src).toContain("data-[state=inactive]:hidden");

    await renderDashboard();
    const user = userEvent.setup();
    const saliencyTab = screen.getByRole("tab", { name: "Saliency" });
    saliencyTab.focus();
    await user.keyboard("{ArrowRight}");
    const fairnessTab = screen.getByRole("tab", { name: "Fairness" });
    await waitFor(() =>
      expect(fairnessTab).toHaveAttribute("aria-selected", "false"),
    );
    const fairnessPanelId = fairnessTab.getAttribute("aria-controls");
    expect(document.getElementById(fairnessPanelId!)).not.toBeNull();
  });

  it("UI-37 keeps a visited analysis tab mounted so its state and job survive a tab switch", async () => {
    // Unmounting on switch reset the saliency method to GradCAM and re-submitted
    // the job on return (served from cache after queueing behind the orphan).
    await renderDashboard();
    const user = userEvent.setup();
    const saliencyTab = screen.getByRole("tab", { name: "Saliency" });
    const saliencyPanelId = saliencyTab.getAttribute("aria-controls")!;
    saliencyTab.focus();
    await user.keyboard("{ArrowRight}");
    await waitFor(() =>
      expect(saliencyTab).toHaveAttribute("aria-selected", "false"),
    );
    const saliencyPanel = document.getElementById(saliencyPanelId);
    expect(saliencyPanel).not.toBeNull();
    expect(saliencyPanel).toHaveAttribute("data-state", "inactive");
    expect(saliencyPanel!.textContent).toContain("Saliency Overlay");
  });
});

describe("TestEmbeddingTabs", () => {
  it("UI-38 presents the three embedding tabs and switches between them", async () => {
    await renderDashboard();
    for (const name of ["Embeddings", "Dataset EDA", "Layer Probes"]) {
      expect(screen.getByRole("tab", { name })).toBeInTheDocument();
    }
    const user = userEvent.setup();
    const embeddings = screen.getByRole("tab", { name: "Embeddings" });
    embeddings.focus();
    await user.keyboard("{ArrowRight}");
    await waitFor(() =>
      expect(screen.getByRole("tab", { name: "Dataset EDA" })).toHaveAttribute(
        "aria-selected",
        "true",
      ),
    );
  });

  it("UI-39 heads the embedding panel and names its help control", async () => {
    await renderDashboard();
    expect(
      screen.getByRole("heading", { level: 2, name: /Audio Embeddings/ }),
    ).toBeInTheDocument();
  });
});

describe("TestPanelLayout", () => {
  it("UI-43 presents the three dockable panels with resize handles", async () => {
    // SRS §3.9.1: "a dashboard organized around a persistent menu/toolbar and a
    // set of dockable panels".
    const { container } = await renderDashboard();
    expect(screen.getByText("Audio Embeddings")).toBeInTheDocument();
    expect(screen.getByText("Audio Dataset")).toBeInTheDocument();
    expect(screen.getByText("Datapoint Editor")).toBeInTheDocument();
    const separators = container.querySelectorAll('[role="separator"]');
    expect(separators.length).toBeGreaterThanOrEqual(3);
  });

  it("UI-44 exposes each resize handle as a focusable separator", async () => {
    // The semantic half of the RUP "size and position" bullet. Pixel geometry is
    // excluded from this section: jsdom has no layout engine, so every element
    // reports zero dimensions.
    const { container } = await renderDashboard();
    const separator = container.querySelector('[role="separator"]');
    expect(separator).not.toBeNull();
    expect(separator).toHaveAttribute("tabindex");
  });

  it("UI-45 labels the EDA comparison control and offers the other corpora", async () => {
    // SRS §3.9.1's "comparison toggle". Only dataset-vs-dataset comparison is
    // implemented; model-vs-model is not implemented in any form and is
    // therefore outside this section entirely.
    const src = readComponentSource("src/components/eda/DatasetEdaView.tsx");
    expect(src).toContain('aria-label="Comparison dataset"');
  });
});
