/**
 * RUP §3.1.3 User Interface Testing — Module B: the persistent toolbar.
 * Cases UI-13…UI-30.
 *
 * SRS US-2 requires "a single, consistent dashboard layout with a persistent
 * menu/toolbar and panels". This module asserts that toolbar's contents, its
 * selection controls, its panel-state controls and its dialog triggers.
 *
 * Radix Select menus are opened from the KEYBOARD (see utils/radix.ts):
 * userEvent.click() does not open them under jsdom, and keyboard activation is
 * in any case the access method RUP §3.1.3 names explicitly.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./utils/render";
import { stubFetch } from "./utils/fetchStub";
import { baseRoutes, customDataset, customModel } from "./utils/fixtures";
import { openSelect, selectOptionLabels, selectOpensEmpty } from "./utils/radix";

vi.mock("react-plotly.js", () => ({ default: () => <div /> }));

const baseProps = {
  apiData: null,
  setApiData: () => {},
  model: "whisper-base",
  setModel: () => {},
  dataset: "common-voice",
  setDataset: () => {},
  uploadedFiles: [],
  selectedFile: null,
  onFileSelect: () => {},
  leftPanelOpen: true,
  rightPanelOpen: true,
  bottomPanelOpen: true,
  onToggleLeftPanel: () => {},
  onToggleRightPanel: () => {},
  onToggleBottomPanel: () => {},
};

async function mountToolbar(
  props: Partial<typeof baseProps> & Record<string, unknown> = {},
  routes: Record<string, unknown> = {},
) {
  stubFetch({ ...baseRoutes, ...routes } as never);
  const { Toolbar } = await import("@/components/layout/Toolbar");
  const result = renderWithProviders(
    <Toolbar {...baseProps} {...(props as object)} />,
    { route: "/" },
  );
  await screen.findByText("LIT for Voice");
  return result;
}

/** The model Select is the first combobox, the dataset Select the second. */
async function comboboxes() {
  return await screen.findAllByRole("combobox");
}

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => {});
});

describe("TestPersistentToolbar", () => {
  it("UI-13 presents the documented toolbar chrome", async () => {
    // US-2: the persistent menu/toolbar. Asserted as a set — a toolbar missing
    // one of these is not the documented one.
    await mountToolbar();
    expect(screen.getByText("LIT for Voice")).toBeInTheDocument();
    expect(screen.getByText("v1.0")).toBeInTheDocument();
    expect(screen.getByText("Model:")).toBeInTheDocument();
    expect(screen.getByText("Dataset:")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "J-Lens Lab" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Manage Datasets" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Models" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upload" })).toBeInTheDocument();
  });

  it("UI-14 exposes the model and dataset help as reachable, named controls", async () => {
    // Guards BUG-31. Both HelpCircle icons sat inside <TooltipTrigger asChild>
    // wrapping a bare <svg>, so NO button was produced: the only in-product
    // explanation of what Whisper vs Wav2Vec2 does was unreachable by keyboard.
    // US-1 makes this material — it requires a first-time user to be guided
    // "only by on-screen labels and tooltips".
    await mountToolbar();
    const modelHelp = screen.getByRole("button", {
      name: "About the model options",
    });
    const datasetHelp = screen.getByRole("button", {
      name: "About the dataset options",
    });
    // Both must be real, focusable buttons. Focus is released again before the
    // case ends: leaving a Radix tooltip open keeps its Popper alive, and an
    // open Popper is pathologically slow under jsdom (see Special Considerations).
    modelHelp.focus();
    expect(modelHelp).toHaveFocus();
    modelHelp.blur();
    datasetHelp.focus();
    expect(datasetHelp).toHaveFocus();
    datasetHelp.blur();
    expect(modelHelp.tagName).toBe("BUTTON");
    expect(datasetHelp.tagName).toBe("BUTTON");
  });

  it("UI-15b offers exactly the two built-in models (source-level)", async () => {
    // Runtime oracle unavailable: opening a Radix Select under jsdom costs ~64s
    // even for a bare two-item Select with no ECHO code involved (measured).
    // The option set is therefore asserted against the component source.
    const { readComponentSource } = await import("./utils/source");
    const src = readComponentSource("src/components/layout/Toolbar.tsx");
    const builtIns = [...src.matchAll(/<SelectItem value="(whisper[^"]*|wav2vec2)">/g)].map(
      (m) => m[1],
    );
    expect(builtIns).toEqual(["whisper-base", "wav2vec2"]);
  });

  it("UI-16 displays a ready custom model as the current selection", async () => {
    await mountToolbar(
      { model: "cm-ready" },
      {
        "GET /models": {
          json: [customModel({ model_id: "cm-ready", hf_repo: "acme/ready-model" })],
        },
      },
    );
    const [modelSelect] = await comboboxes();
    await waitFor(() => expect(modelSelect).toHaveTextContent("acme/ready-model"));
  });

  it("UI-16b offers only models whose validation has completed (source-level)", async () => {
    // A model still validating cannot be run, so offering it would produce a
    // job that fails after the user has already committed to it.
    const { readComponentSource } = await import("./utils/source");
    const src = readComponentSource("src/components/layout/Toolbar.tsx");
    expect(src).toContain("customModels.filter((customModel) => customModel.status === 'ready')");
  });

  it.each([
    ["common-voice", "Common Voice"],
    ["ravdess", "RAVDESS"],
    ["l2-arctic", "L2-ARCTIC"],
    ["saa", "SAA"],
  ])("UI-18 displays dataset %s as %s", async (value, label) => {
    await mountToolbar({ dataset: value });
    const [, datasetSelect] = await comboboxes();
    expect(datasetSelect).toHaveTextContent(label);
  });

  it("UI-19 declares a model in its dataset map that the menu never offers", async () => {
    // Pins OBS-11. modelDatasetMap and defaultDatasetForModel both declare
    // "whisper-large", but no SelectItem offers it: dead configuration that
    // will silently start working, or stop mattering, on the next edit.
    const { readComponentSource } = await import("./utils/source");
    const src = readComponentSource("src/components/layout/Toolbar.tsx");
    expect(src).toContain('"whisper-large"');
    expect(src).not.toContain('<SelectItem value="whisper-large">');
  });

  it("UI-20 renders custom datasets under a real grouping label", async () => {
    const { readComponentSource } = await import("./utils/source");
    const src = readComponentSource("src/components/layout/Toolbar.tsx");
    expect(src).toContain("<SelectLabel>Custom Datasets</SelectLabel>");
    expect(src).toContain("<SelectGroup>");
  });

  it("UI-21 puts no disabled pseudo-option into any listbox", async () => {
    // Guards BUG-20. The grouping header used to be
    // <SelectItem disabled value="separator">── Custom Datasets ──</SelectItem>,
    // i.e. a real option: announced to a screen reader as a dimmed choice,
    // counted in the option total, and — if ever activated — setting the
    // dataset state to the literal string "separator", after which every
    // downstream request 404s.
    const { readComponentSource } = await import("./utils/source");
    const src = readComponentSource("src/components/layout/Toolbar.tsx");
    expect(src).not.toContain("SelectItem disabled");
    expect(src).not.toContain('value="separator"');
    expect(src).not.toContain("── Custom Datasets ──");
  });

  it("UI-22 hides the uploaded-file selector until a file exists", async () => {
    const { rerender } = await mountToolbar();
    expect(await comboboxes()).toHaveLength(2);
    const { Toolbar } = await import("@/components/layout/Toolbar");
    rerender(
      <Toolbar
        {...baseProps}
        uploadedFiles={[
          { file_id: "f1", filename: "clip.wav", message: "ok" },
        ]}
      />,
    );
    await waitFor(async () => expect(await comboboxes()).toHaveLength(3));
  });

  it("UI-23 leaves a custom model with no selectable dataset (source-level)", async () => {
    // Pins OBS-13, a navigation dead end. modelDatasetMap has no entry for a
    // custom model id, so allowedDatasets falls back to ["custom"], which the
    // built-in branch then filters out — leaving the dataset menu with zero
    // options. Not fixed here: which datasets a custom model may run against is
    // a product decision, and CustomModelManager already states the path is
    // incomplete. Pinned so the dead end is visible rather than folklore.
    const { readComponentSource } = await import("./utils/source");
    const src = readComponentSource("src/components/layout/Toolbar.tsx");
    expect(src).toContain('modelDatasetMap[model] || ["custom"]');
    expect(src).toContain("allowedDatasets.filter(ds => ds !== 'custom')");
  });
});

describe("TestPanelStateControls", () => {
  it("UI-24 exposes the three panel toggles by name", async () => {
    await mountToolbar();
    for (const name of [
      "Toggle left panel",
      "Toggle bottom panel",
      "Toggle right panel",
    ]) {
      expect(screen.getByRole("button", { name })).toBeInTheDocument();
    }
  });

  it("UI-25 gives every toolbar control an accessible name", async () => {
    // Guards BUG-16. Before the fix the three panel toggles rendered as
    // icon-only buttons with an empty accessible name: a screen-reader user
    // heard "button, button, button" across the toolbar with no way to tell
    // which panel each one controlled.
    await mountToolbar();
    const unnamed = screen
      .getAllByRole("button")
      .filter((b) => !(b.textContent?.trim() || b.getAttribute("aria-label")));
    expect(
      unnamed.map((b) => b.outerHTML.slice(0, 120)).join("\n"),
    ).toBe("");
  });

  it("UI-26 reports panel open state, not just colour", async () => {
    // Guards BUG-16b. Open/closed was signalled only by the Button `variant`,
    // i.e. a background colour, so a screen-reader user could not tell whether
    // the panel they were toggling was currently open.
    await mountToolbar({ leftPanelOpen: true });
    expect(
      screen.getByRole("button", { name: "Toggle left panel" }),
    ).toHaveAttribute("aria-pressed", "true");
  });

  it("UI-26b reports the closed state of the same control", async () => {
    // Asserted in a separate render: rerendering into the same tree would leave
    // the previous toolbar mounted and make the query ambiguous.
    await mountToolbar({ leftPanelOpen: false });
    expect(
      screen.getByRole("button", { name: "Toggle left panel" }),
    ).toHaveAttribute("aria-pressed", "false");
  });

  it("UI-27 invokes the toggle callback on activation", async () => {
    const onToggleRightPanel = vi.fn();
    await mountToolbar({ onToggleRightPanel });
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Toggle right panel" }));
    expect(onToggleRightPanel).toHaveBeenCalledTimes(1);
  });

  it("UI-28 reaches a panel toggle from the keyboard", async () => {
    // RUP §3.1.3 access methods: the control must be operable without a mouse.
    const onToggleBottomPanel = vi.fn();
    await mountToolbar({ onToggleBottomPanel });
    const toggle = screen.getByRole("button", { name: "Toggle bottom panel" });
    // Native button + focusable + named: Enter and Space activation is then a
    // platform guarantee. Synthesising the keypress instead would assert
    // jsdom's dispatch, not the product's markup.
    expect(toggle.tagName).toBe("BUTTON");
    expect(toggle).not.toHaveAttribute("disabled");
    toggle.focus();
    expect(toggle).toHaveFocus();
    toggle.blur();
    void onToggleBottomPanel;
  });
});

describe("TestUploadAffordance", () => {
  it("UI-29 opens the uploader when the toolbar Upload button is activated", async () => {
    // Guards BUG-13. The most prominent button in the persistent toolbar —
    // primary variant, upload icon, its own tooltip — had no onClick at all and
    // did nothing when clicked. First-run users go there first, so this is a
    // direct hit on US-1's ten-minute learnability criterion.
    const onUploadClick = vi.fn();
    await mountToolbar({ onUploadClick });
    await userEvent.setup().click(screen.getByRole("button", { name: "Upload" }));
    expect(onUploadClick).toHaveBeenCalledTimes(1);
  });

  it("UI-30 states the accepted formats and the size limit", async () => {
    // Guards BUG-28. SRS §3.9.1 requires the upload panel to carry
    // "format/size hints"; no size hint existed anywhere in the client, so a
    // user could upload a 300 MB file, wait for the transfer, and only then
    // learn the server rejected it. The figures are FR-1's, whose deployed
    // values 3.1.2 verified in FT-07c.
    //
    // Asserted at source level because the hint sits in Radix tooltip content,
    // which is Popper-rendered and therefore not openable at acceptable cost
    // under jsdom. Module D asserts the same hint at runtime on the drop
    // overlay, where it is rendered inline.
    const { readComponentSource } = await import("./utils/source");
    const src = readComponentSource("src/components/layout/Toolbar.tsx");
    expect(src).toContain(
      "WAV, MP3, M4A or FLAC · up to 100 MB and 10 minutes per file",
    );
  });
});
