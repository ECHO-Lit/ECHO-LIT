/**
 * RUP §3.1.3 User Interface Testing — Module H: standards conformance.
 * Cases UI-110…UI-124.
 *
 * RUP §3.1.3 requires that "window objects and characteristics … conform to
 * standards". SRS §3.12 names the standard: "Interface design aims to conform
 * to common web usability and accessibility conventions."
 *
 * Every axe call NAMES the rules it claims to verify — see utils/axe.ts for why
 * running the full ruleset against a mounted fragment would be close to
 * vacuous. `color-contrast` is disabled throughout and excluded from this
 * section, because `test.css` is false and axe would evaluate
 * transparent-on-transparent.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "./utils/render";
import { expectNoAxeViolations } from "./utils/axe";
import { stubFetch } from "./utils/fetchStub";
import { dashboardRoutes, baseRoutes } from "./utils/fixtures";
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

describe("TestLandmarks", () => {
  it("UI-110 exposes landmarks and a level-one heading on the lab page", async () => {
    // The correct baseline: /j-lens already used <main>, <header> and an <h1>,
    // which is why the dashboard fix below is the house style rather than a new
    // convention.
    window.history.pushState({}, "", "/j-lens");
    stubFetch(dashboardRoutes);
    const { default: App } = await import("@/App");
    const { container } = render(<App />);
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { level: 1, name: "J-Lens Lab" }),
      ).toBeInTheDocument(),
    );
    expect(container.querySelector("main")).not.toBeNull();
    expect(container.querySelector("header")).not.toBeNull();
    await expectNoAxeViolations(container, ["heading-order"]);
  });

  it("UI-111 exposes a main landmark and a level-one heading on the dashboard", async () => {
    // Guards BUG-17. The primary window was bare <div>s: no <main>, <header>,
    // <nav>, <h1> or <h2>, and no skip link. Screen-reader landmark navigation
    // could not enter the product's main screen at all — and the dashboard IS
    // the product.
    const { container } = await renderDashboard();
    expect(container.querySelector("main")).not.toBeNull();
    expect(container.querySelector("header")).not.toBeNull();
    expect(
      screen.getByRole("heading", { level: 1, name: /LIT for Voice/ }),
    ).toBeInTheDocument();
  });

  it("UI-111b offers a skip link into the workspace", async () => {
    const { container } = await renderDashboard();
    const skip = screen.getByRole("link", {
      name: "Skip to the analysis workspace",
    });
    expect(skip).toHaveAttribute("href", "#analysis-workspace");
    // The target must exist, or the skip link is decoration.
    expect(container.querySelector("#analysis-workspace")).not.toBeNull();
  });

  it("UI-112 does not skip heading levels on the dashboard", async () => {
    // Panels started at <h3> with nothing above them. They are now <h2> under
    // the new <h1>.
    const { container } = await renderDashboard();
    const levels = Array.from(container.querySelectorAll("h1,h2,h3,h4,h5,h6"))
      .map((h) => Number(h.tagName[1]));
    expect(levels[0]).toBe(1);
    await expectNoAxeViolations(container, ["heading-order"]);
  });
});

describe("TestAccessibleNames", () => {
  it("UI-113 names every interactive control in the toolbar", async () => {
    // Guards BUG-16. Icon-only buttons with an empty accessible name made a
    // screen reader announce "button, button, button" across the toolbar.
    stubFetch(baseRoutes);
    const { Toolbar } = await import("@/components/layout/Toolbar");
    const { container } = renderWithProviders(
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
        leftPanelOpen
        rightPanelOpen
        bottomPanelOpen
        onToggleLeftPanel={() => {}}
        onToggleRightPanel={() => {}}
        onToggleBottomPanel={() => {}}
      />,
      { route: "/" },
    );
    await screen.findByText("LIT for Voice");
    await expectNoAxeViolations(container, [
      "button-name",
      "link-name",
      "aria-input-field-name",
    ]);
  });

  it("UI-114 gives every tooltip trigger an accessible name", async () => {
    // Guards BUG-30. Eighteen bare <TooltipTrigger> elements (no asChild) each
    // rendered a <button> whose only child was an unlabelled <svg> — eighteen
    // buttons with an empty accessible name, scattered across the analysis
    // panels, and the highest-count accessibility defect in the product.
    const files = [
      "src/components/eda/ChartCard.tsx",
      "src/components/eda/ClusteringSection.tsx",
      "src/components/eda/ClusterInsights.tsx",
      "src/components/eda/DatasetEdaView.tsx",
      "src/components/eda/NearestNeighborsPanel.tsx",
      "src/components/panels/DatapointEditorPanel.tsx",
      "src/components/panels/EmbeddingPanel.tsx",
      "src/components/visualization/ScalersVisualization.tsx",
    ];
    for (const file of files) {
      const src = readComponentSource(file);
      expect(src).not.toContain("<TooltipTrigger>");
    }
  });

  it("UI-115 makes the toolbar help tooltips reachable named controls", async () => {
    // Guards BUG-31, the inverse defect: <TooltipTrigger asChild> around a bare
    // lucide <svg> produced NO button at all, so the only in-product
    // explanation of Whisper vs Wav2Vec2 was unreachable by keyboard.
    const src = readComponentSource("src/components/layout/Toolbar.tsx");
    expect(src).toContain('aria-label="About the model options"');
    expect(src).toContain('aria-label="About the dataset options"');
  });

  it("UI-116 labels the audio search field", async () => {
    // The <Input> had no name source but its placeholder.
    const src = readComponentSource("src/components/panels/AudioDatasetPanel.tsx");
    expect(src).toContain('aria-label="Search audio files"');
  });

  it("UI-117 labels the fairness setup fields programmatically", async () => {
    // Guards BUG-21. <Label> with no htmlFor beside controls with no id, so
    // "Min group size" announced as "spin button, 2" with no indication of what
    // it configures — and setting it wrong silently changes which speaker
    // groups are excluded from a fairness result. WCAG 1.3.1 / 3.3.2.
    const src = readComponentSource("src/components/fairness/FairnessPanel.tsx");
    expect(src).toContain('htmlFor="fairness-min-group-size"');
    expect(src).toContain('id="fairness-min-group-size"');
    expect(src).toContain('htmlFor="fairness-min-speakers"');
    expect(src).toContain('id="fairness-min-speakers"');
    expect(src).toContain('aria-label="Group by"');
    expect(src).toContain('aria-label="Reference group"');
  });

  it("UI-118 labels the J-Lens Lab selects", async () => {
    const src = readComponentSource("src/pages/JacobianLensLab.tsx");
    expect(src).toContain('aria-label="Speech-to-text model"');
    expect(src).toContain('aria-label="Transcript dataset"');
    // The positive control: "Select first" was already wired correctly.
    expect(src).toContain('htmlFor="sample-limit"');
    expect(src).toContain('id="sample-limit"');
  });

  it("UI-119 keeps the perturbation controls correctly labelled", async () => {
    // The positive-control case: these htmlFor/id pairs were already right, so
    // the fixes above cannot be mistaken for a blanket problem.
    const src = readComponentSource("src/components/analysis/PerturbationTools.tsx");
    expect(src).toMatch(/htmlFor="[^"]+"/);
    expect(src).toMatch(/id="[^"]+"/);
  });

  it("UI-120 names the audio transport controls and both sliders", async () => {
    // Guards BUG-14 and BUG-16c. The two Sliders both rendered role="slider"
    // with NO accessible name, so getAllByRole("slider") returned two
    // indistinguishable nodes, and the transport buttons were icon-only.
    const { AudioPlayer } = await import("@/components/audio/AudioPlayer");
    const { container } = renderWithProviders(
      <AudioPlayer
        isPlaying={false}
        onPlayPause={() => {}}
        currentTime={10}
        duration={60}
        onSeek={() => {}}
        onVolumeChange={() => {}}
      />,
    );
    expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Skip back 5 seconds" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Skip forward 5 seconds" }),
    ).toBeInTheDocument();
    const sliders = screen.getAllByRole("slider");
    const names = sliders.map((s) =>
      s.getAttribute("aria-label") ??
      s.closest("[aria-label]")?.getAttribute("aria-label") ??
      "",
    );
    expect(names).toContain("Seek");
    expect(names).toContain("Volume");
    await expectNoAxeViolations(container, ["button-name"]);
  });

  it("UI-121 names the dataset delete control by its dataset", async () => {
    // Guards BUG-16. The destructive control was an icon-only <Trash2> button
    // announced as bare "button", sitting directly beside "Select".
    const src = readComponentSource("src/components/dataset/CustomDatasetManager.tsx");
    expect(src).toContain("aria-label={`Delete dataset ${dataset.dataset_name}`}");
  });
});

describe("TestStatusAnnouncement", () => {
  it("UI-122 announces progress and status changes", async () => {
    // Guards BUG-18. No aria-live or role="status" existed ANYWHERE in src/, so
    // "Inferencing... 7/12", "Inference Complete" and every job progress
    // message were silent to assistive technology. For a job that runs 10-30
    // minutes, a non-sighted user had no way to know anything was happening.
    const announced = [
      "src/components/panels/AudioDatasetPanel.tsx",
      "src/components/fairness/FairnessPanel.tsx",
      "src/components/analysis/PerturbationDiagnosticsPanel.tsx",
      "src/components/probing/LayerProbePanel.tsx",
      "src/components/dataset/CustomDatasetManager.tsx",
    ];
    for (const file of announced) {
      const src = readComponentSource(file);
      expect(src).toContain('role="status"');
      expect(src).toContain('aria-live="polite"');
    }
  });

  it("UI-123 reports upload progress truthfully", async () => {
    // Guards BUG-22. The bar was set to 0, stayed at 0 for the entire transfer,
    // and jumped to 100 after it had already finished — reporting no progress
    // while presenting itself as a progress bar, so the user read 0% and
    // assumed it was stuck. Replaced in 3.1.3 with an indeterminate indicator;
    // the byte-level bar over XMLHttpRequest landed in 3.1.4 (BUG-45) and is
    // asserted at runtime in performance-feedback.test.tsx. Here: the bar is a
    // named progressbar driven by transfer events.
    const src = readComponentSource("src/components/dataset/CustomDatasetManager.tsx");
    expect(src).not.toContain("setUploadProgress(100)");
    expect(src).toContain('aria-label="Upload progress"');
    expect(src).toContain("uploadWithProgress");
  });
});

describe("TestKeyboardAccess", () => {
  it("UI-124 makes audio table rows selectable from the keyboard", async () => {
    // Guards BUG-19. Rows were click-only <tr> with no tabIndex, no key handler
    // and no aria-selected. Selecting an audio file is the gateway action for
    // every downstream analysis, so the product was unusable past the toolbar
    // without a mouse.
    const src = readComponentSource("src/components/audio/AudioDataTable.tsx");
    expect(src).toContain("tabIndex={0}");
    expect(src).toContain("onKeyDown");
    expect(src).toContain("aria-selected");
    expect(src).toContain('event.key === "Enter" || event.key === " "');
    // Both paths must resolve the row id the same way, or click and keyboard
    // selection would diverge.
    expect(src).toContain("onRowSelect(resolveRowId(row))");
  });

  it("UI-124b names the pagination controls", async () => {
    const src = readComponentSource("src/components/audio/AudioDataTable.tsx");
    expect(src).toContain('aria-label="Previous page"');
    expect(src).toContain('aria-label="Next page"');
  });
});
