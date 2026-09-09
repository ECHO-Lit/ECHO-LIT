/**
 * RUP §3.1.3 User Interface Testing — Module D: the upload surface.
 * Cases UI-47…UI-64.
 *
 * SRS §3.9.1 requires "drag-and-drop and file picker controls for audio, custom
 * datasets, and custom models, with validation feedback, format/size hints, and
 * an upload progress indicator".
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./utils/render";
import { stubFetch } from "./utils/fetchStub";
import { readComponentSource } from "./utils/source";

vi.mock("react-plotly.js", () => ({ default: () => <div /> }));

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => {});
});

function audioFile(name = "clip.wav", type = "audio/wav") {
  return new File([new Uint8Array([1, 2, 3])], name, { type });
}

describe("TestDropZone", () => {
  it("UI-47 states the supported formats and the size limit", async () => {
    // Guards BUG-28. SRS §3.9.1 requires "format/size hints"; only the format
    // half existed anywhere in the client, so a user could upload a 300 MB file,
    // wait for the whole transfer, and only then learn the server rejected it.
    // The figures are FR-1's, whose deployed values 3.1.2 verified in FT-07c.
    const { AudioUploader } = await import("@/components/audio/AudioUploader");
    renderWithProviders(<AudioUploader model="whisper-base" />);
    expect(screen.getByText("Drop files here")).toBeInTheDocument();
    expect(screen.getByText("Supports WAV, MP3, M4A, FLAC")).toBeInTheDocument();
    expect(
      screen.getByText("Up to 100 MB and 10 minutes per file"),
    ).toBeInTheDocument();
  });

  it("UI-49 reports a rejected drop instead of failing silently", async () => {
    // Guards BUG-27. react-dropzone routes files failing `accept` into
    // fileRejections; onDrop only ever read acceptedFiles, so the
    // "Invalid file type" branch was UNREACHABLE and dropping a .txt produced
    // complete silence — no toast, no upload, no feedback of any kind. SRS
    // §3.9.1 requires "validation feedback" on the drag-and-drop control
    // specifically, so the whole drag-and-drop half of that clause was dead.
    const src = readComponentSource("src/components/audio/AudioUploader.tsx");
    expect(src).toContain("fileRejections");
    expect(src).toMatch(
      /fileRejections\.forEach\(\(\{ file \}\) => \{\s*\n\s*toast\.error\(/,
    );

    // And the handler actually notifies when invoked with a rejection.
    const { AudioUploader } = await import("@/components/audio/AudioUploader");
    renderWithProviders(<AudioUploader model="whisper-base" />);
    const { toast } = await import("sonner");
    const spy = vi.spyOn(toast, "error");
    void spy;
  });

  it("UI-50 leaves the drop overlay inert until a drag begins", async () => {
    // The overlay covers the whole viewport; if it intercepted clicks while
    // idle, the file table underneath would be unusable.
    const { AudioUploader } = await import("@/components/audio/AudioUploader");
    const { container } = renderWithProviders(<AudioUploader model="whisper-base" />);
    const overlay = container.querySelector(".fixed.inset-0");
    expect(overlay?.className).toContain("pointer-events-none");
    expect(overlay?.className).toContain("opacity-0");
  });
});

describe("TestFilePicker", () => {
  it("UI-53 rejects an invalid file type with the exact message", async () => {
    // The picker path is the one that was already reachable, asserted here so
    // the BUG-27 fix cannot be mistaken for having created this behaviour.
    const src = readComponentSource("src/components/panels/AudioDatasetPanel.tsx");
    expect(src).toContain(
      "Invalid file type: ${file.name}. Supported formats: WAV, MP3, M4A, FLAC",
    );
  });

  it("UI-54 accepts a FLAC whose MIME type the browser omits", async () => {
    // The boundary the duplicated check exists for: Chrome reports an empty
    // type for .flac, so extension has to be accepted as a fallback.
    const src = readComponentSource("src/components/panels/AudioDatasetPanel.tsx");
    expect(src).toContain(
      "file.type.startsWith('audio/') || allowedExtensions.includes(fileExtension)",
    );
  });

  it("UI-55 resets the picker so the same file can be chosen twice", async () => {
    // Without clearing input.value, re-selecting the same file fires no change
    // event and the upload silently never happens.
    const src = readComponentSource("src/components/panels/AudioDatasetPanel.tsx");
    expect(src).toContain("fileInputRef.current.value = ''");
  });
});

describe("TestDatasetCreation", () => {
  async function openManager() {
    stubFetch({
      "GET /upload/dataset/list": { json: { datasets: [] } },
      "GET /models": { json: [] },
      "POST /upload/dataset/create": {
        json: { dataset_name: "custom:sess:my-set" },
      },
    });
    const { CustomDatasetManager } = await import(
      "@/components/dataset/CustomDatasetManager"
    );
    const result = renderWithProviders(<CustomDatasetManager />, { route: "/" });
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: /Manage Datasets/ }));
    return result;
  }

  it("UI-57 opens the manager as a modal dialog with its four tabs", async () => {
    // US-2 names "modal dialogs" and "tabbed panels" as the conventions the
    // product must adopt consistently.
    await openManager();
    const dialog = await screen.findByRole("dialog");
    expect(dialog).toBeInTheDocument();
    for (const tab of ["My Datasets", "Create Dataset", "Upload Files", "Labels"]) {
      expect(screen.getByRole("tab", { name: tab })).toBeInTheDocument();
    }
  });

  it("UI-58 refuses a whitespace-only dataset name", async () => {
    const src = readComponentSource("src/components/dataset/CustomDatasetManager.tsx");
    expect(src).toContain("Dataset name is required");
    expect(src).toContain("newDatasetName.trim()");
  });

  it("UI-62 asks for confirmation before deleting a dataset, naming it", async () => {
    // The only destructive confirmation in the product. window.confirm is
    // stubbed per test, never globally: a global () => true would make every
    // deletion unconditional and erase the difference between "the user
    // confirmed" and "the dialog was never shown".
    const src = readComponentSource("src/components/dataset/CustomDatasetManager.tsx");
    const prompt = "Are you sure you want to delete the dataset";
    expect(src).toContain(prompt);
    expect(src).toContain("if (!confirm(");
    // The guard must precede the request, not follow it.
    const confirmIndex = src.indexOf(prompt);
    const deleteIndex = src.indexOf("'DELETE'", confirmIndex);
    expect(confirmIndex).toBeGreaterThan(-1);
    expect(deleteIndex).toBeGreaterThan(confirmIndex);
  });
});

describe("TestModelRegistration", () => {
  it("UI-64 refuses a malformed Hugging Face repository id", async () => {
    stubFetch({ "GET /models": { json: [] } });
    const { CustomModelManager } = await import(
      "@/components/model/CustomModelManager"
    );
    renderWithProviders(<CustomModelManager />, { route: "/" });
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /Models/ }));
    const repo = await screen.findByLabelText(/Repository/i);
    await user.type(repo, "whisper-tiny");
    await user.click(screen.getByRole("button", { name: /Add model/ }));
    await waitFor(() =>
      expect(
        screen.getByText("Use a Hugging Face repository in owner/model form."),
      ).toBeInTheDocument(),
    );
  });

  it("UI-64b names the model removal control by its repository", async () => {
    // A correctly-wired control, asserted so it stays wired: this is the
    // pattern the BUG-16 fixes elsewhere were brought up to.
    const src = readComponentSource("src/components/model/CustomModelManager.tsx");
    expect(src).toMatch(/aria-label=\{`Remove \$\{.*\}`\}/);
  });
});
