/**
 * RUP §3.1.3 User Interface Testing — Module E: the audio panel.
 * Cases UI-65…UI-81.
 *
 * SRS §3.9.1 requires "a waveform viewer with playback controls (play/pause,
 * seek …)". Waveform ZOOM and the saliency/highlighted-span overlays are not
 * implemented in any form and are therefore outside this section entirely, per
 * the standing decision on unimplemented requirements.
 *
 * The wavesurfer stub is an EVENT SOURCE with a real listener registry, not a
 * replacement for WaveformViewer: the component's own state machine, its own
 * strings and its own Retry handler are what get asserted.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./utils/render";

const listeners = new Map<string, ((...args: unknown[]) => void)[]>();
const wavesurferInstance = {
  on: (event: string, handler: (...args: unknown[]) => void) => {
    listeners.set(event, [...(listeners.get(event) ?? []), handler]);
  },
  un: () => {},
  destroy: () => {},
  load: vi.fn(),
  play: () => {},
  pause: () => {},
  seekTo: () => {},
  setVolume: () => {},
  getDuration: () => 2,
  getCurrentTime: () => 0,
  empty: () => {},
};

vi.mock("wavesurfer.js", () => ({
  default: { create: () => wavesurferInstance },
}));
vi.mock("react-plotly.js", () => ({ default: () => <div /> }));

beforeEach(() => {
  listeners.clear();
  vi.spyOn(console, "error").mockImplementation(() => {});
});

describe("TestPlayerTransport", () => {
  async function renderPlayer(props: Record<string, unknown> = {}) {
    const { AudioPlayer } = await import("@/components/audio/AudioPlayer");
    const onSeek = vi.fn();
    const onPlayPause = vi.fn();
    const onVolumeChange = vi.fn();
    const result = renderWithProviders(
      <AudioPlayer
        isPlaying={false}
        onPlayPause={onPlayPause}
        currentTime={10}
        duration={60}
        onSeek={onSeek}
        onVolumeChange={onVolumeChange}
        {...props}
      />,
    );
    return { ...result, onSeek, onPlayPause, onVolumeChange };
  }

  it("UI-69 exposes play and pause under the right name for the state", async () => {
    // Guards BUG-16. The control was an icon-only button with no accessible
    // name at either state.
    const { rerender } = await renderPlayer();
    expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
    const { AudioPlayer } = await import("@/components/audio/AudioPlayer");
    rerender(
      <AudioPlayer isPlaying onPlayPause={() => {}} currentTime={10} duration={60} />,
    );
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument();
  });

  it("UI-70 reports play/pause activation to its owner", async () => {
    const { onPlayPause } = await renderPlayer();
    await userEvent.setup().click(screen.getByRole("button", { name: "Play" }));
    expect(onPlayPause).toHaveBeenCalledTimes(1);
  });

  it("UI-71 moves the playhead when the skip controls are used", async () => {
    // Guards BUG-14. Both skip buttons had NO onClick at all — two permanently
    // inert controls in a transport bar, which SRS §3.9.1 requires to provide
    // "standard media playback controls". The onSeek prop they needed was
    // already supplied by DatapointEditorPanel.
    const { onSeek } = await renderPlayer({ currentTime: 30, duration: 60 });
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Skip back 5 seconds" }));
    expect(onSeek).toHaveBeenLastCalledWith(25);
    await user.click(screen.getByRole("button", { name: "Skip forward 5 seconds" }));
    expect(onSeek).toHaveBeenLastCalledWith(35);
  });

  it("UI-72 clamps skipping at both ends of the clip", async () => {
    // The boundary the fix has to get right: skipping must not seek to a
    // negative time or past the end of the audio.
    const { onSeek } = await renderPlayer({ currentTime: 2, duration: 60 });
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Skip back 5 seconds" }));
    expect(onSeek).toHaveBeenLastCalledWith(0);

    const second = await renderPlayer({ currentTime: 59, duration: 60 });
    await user.click(
      screen.getAllByRole("button", { name: "Skip forward 5 seconds" }).at(-1)!,
    );
    expect(second.onSeek).toHaveBeenLastCalledWith(60);
  });

  it("UI-73 disables the skip controls when seeking is unavailable", async () => {
    // Without an onSeek the controls cannot do anything; disabling them is
    // honest, where the previous silent no-op was not.
    const { AudioPlayer } = await import("@/components/audio/AudioPlayer");
    renderWithProviders(
      <AudioPlayer isPlaying={false} onPlayPause={() => {}} currentTime={0} duration={0} />,
    );
    expect(
      screen.getByRole("button", { name: "Skip back 5 seconds" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Skip forward 5 seconds" }),
    ).toBeDisabled();
  });

  it("UI-74 distinguishes the seek and volume sliders by name", async () => {
    // Guards BUG-16c. Both Sliders rendered role="slider" with NO accessible
    // name, so getAllByRole("slider") returned two indistinguishable nodes and
    // a screen-reader user heard "slider, slider".
    await renderPlayer();
    const named = screen
      .getAllByRole("slider")
      .map(
        (s) =>
          s.getAttribute("aria-label") ??
          s.closest("[aria-label]")?.getAttribute("aria-label") ??
          "",
      );
    expect(named).toContain("Seek");
    expect(named).toContain("Volume");
  });

  it("UI-75 formats elapsed and total time", async () => {
    // Catches a padStart regression: 65s must read 1:05, not 1:5.
    await renderPlayer({ currentTime: 65, duration: 125 });
    expect(screen.getByText("1:05")).toBeInTheDocument();
    expect(screen.getByText("2:05")).toBeInTheDocument();
  });
});

describe("TestWaveform", () => {
  async function renderWaveform(audioUrl = "http://localhost:8000/audio/a1") {
    const { WaveformViewer } = await import("@/components/audio/WaveformViewer");
    return renderWithProviders(<WaveformViewer audioUrl={audioUrl} />);
  }

  function emit(event: string, ...args: unknown[]) {
    (listeners.get(event) ?? []).forEach((handler) => handler(...args));
  }

  it("UI-76 shows no-file state when nothing is selected", async () => {
    const { WaveformViewer } = await import("@/components/audio/WaveformViewer");
    renderWithProviders(<WaveformViewer audioUrl={undefined} />);
    expect(screen.getByText(/No audio file selected/)).toBeInTheDocument();
  });

  it("UI-77 reports a load failure in plain language with a recovery action", async () => {
    // US-4's "indicate the corrective action where one exists": the viewer's
    // error state offers a working Retry rather than a dead end.
    await renderWaveform();
    emit("error", new Error("decode failed"));
    const retry = await screen.findByRole("button", { name: /Retry/ });
    expect(screen.getByText(/Error loading audio/)).toBeInTheDocument();
    expect(retry).toBeEnabled();
  });
});
