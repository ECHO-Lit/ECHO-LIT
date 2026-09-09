/**
 * RUP §3.1.3 User Interface Testing — Module A: application navigation.
 * Cases UI-01…UI-12.
 *
 * These render the REAL <App/>. BrowserRouter lives inside App.tsx:17, not
 * main.tsx, so wrapping a copy of the route table in a MemoryRouter would test
 * the copy — and a copy cannot detect a regression in the real table. The URL is
 * set with window.history.pushState before render, which drives the real
 * BrowserRouter through jsdom's real history. environmentOptions.jsdom.url
 * fixes the origin so this is deterministic.
 *
 * Only third-party rendering engines are stubbed (plotly, wavesurfer). No ECHO
 * component is ever replaced.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { stubFetch } from "./utils/fetchStub";
import { dashboardRoutes } from "./utils/fixtures";

vi.mock("react-plotly.js", () => ({
  default: () => <div data-testid="plotly-stub" />,
}));
vi.mock("wavesurfer.js", () => ({
  default: {
    create: () => ({
      on: () => {},
      un: () => {},
      destroy: () => {},
      load: () => {},
      play: () => {},
      pause: () => {},
      seekTo: () => {},
      setVolume: () => {},
      getDuration: () => 2,
      getCurrentTime: () => 0,
      empty: () => {},
    }),
  },
}));

const appRoutes = { ...dashboardRoutes };

function go(path: string) {
  window.history.pushState({}, "", path);
}

async function renderApp(path: string) {
  go(path);
  stubFetch(appRoutes);
  const { default: App } = await import("@/App");
  return render(<App />);
}

beforeEach(() => {
  // console.error is load-bearing evidence in Module G; keep the surface quiet
  // here so a real failure stays visible in the run output.
  vi.spyOn(console, "error").mockImplementation(() => {});
});
afterEach(() => {
  go("/");
});

describe("TestRouting", () => {
  it("UI-01 renders the dashboard shell at the root route", async () => {
    await renderApp("/");
    await waitFor(() =>
      expect(screen.getByText("LIT for Voice")).toBeInTheDocument(),
    );
    // The three dockable panels of SRS §3.9.1, asserted together: a shell that
    // renders only one of them is not the documented dashboard.
    expect(screen.getByText("Audio Embeddings")).toBeInTheDocument();
    expect(screen.getByText("Audio Dataset")).toBeInTheDocument();
    expect(screen.getByText("Datapoint Editor")).toBeInTheDocument();
  });

  it("UI-02 renders the J-Lens Lab at /j-lens", async () => {
    await renderApp("/j-lens");
    expect(
      screen.getByRole("heading", { level: 1, name: "J-Lens Lab" }),
    ).toBeInTheDocument();
    expect(screen.getByText("1. Choose a model and dataset")).toBeInTheDocument();
    expect(screen.getByText("2. Select fitting samples")).toBeInTheDocument();
    expect(screen.getByText("3. Fit the lens")).toBeInTheDocument();
  });

  it("UI-03 renders the 404 page for an unknown path", async () => {
    await renderApp("/nope");
    expect(
      screen.getByRole("heading", { level: 1, name: "404" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Oops! Page not found")).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Return to Home" }),
    ).toBeInTheDocument();
  });

  it.each(["/nope", "/j-lens/extra", "/upload"])(
    "UI-04 reaches the 404 page for %s",
    async (path) => {
      await renderApp(path);
      expect(
        screen.getByRole("heading", { level: 1, name: "404" }),
      ).toBeInTheDocument();
    },
  );

  it("UI-05 returns home from 404 without a full document reload", async () => {
    // Guards BUG-26. NotFound.tsx:19 is a plain <a href="/">, which unloads the
    // document: the bundle is re-fetched and all React state is destroyed.
    // Every other in-app navigation uses <Link>, so this is also a US-2 break.
    await renderApp("/nope");
    const user = userEvent.setup();
    await user.click(screen.getByRole("link", { name: "Return to Home" }));
    await waitFor(() =>
      expect(screen.getByText("LIT for Voice")).toBeInTheDocument(),
    );
    expect(window.location.pathname).toBe("/");
  });
});

describe("TestWindowToWindow", () => {
  it("UI-06 navigates from the dashboard to the J-Lens Lab via the toolbar", async () => {
    await renderApp("/");
    const user = userEvent.setup();
    await user.click(await screen.findByRole("link", { name: /J-Lens Lab/ }));
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { level: 1, name: "J-Lens Lab" }),
      ).toBeInTheDocument(),
    );
  });

  it("UI-07 returns from the lab to the analysis workspace", async () => {
    await renderApp("/j-lens");
    const user = userEvent.setup();
    await user.click(screen.getByRole("link", { name: /Analysis workspace/ }));
    await waitFor(() =>
      expect(screen.getByText("Audio Embeddings")).toBeInTheDocument(),
    );
  });

  it("UI-08 completes the round trip using the keyboard alone", async () => {
    // RUP §3.1.3 "use of access methods (tab keys …)". No mouse event is fired.
    await renderApp("/");
    const user = userEvent.setup();
    const toLab = await screen.findByRole("link", { name: /J-Lens Lab/ });
    toLab.focus();
    expect(toLab).toHaveFocus();
    await user.keyboard("{Enter}");
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { level: 1, name: "J-Lens Lab" }),
      ).toBeInTheDocument(),
    );
    const back = screen.getByRole("link", { name: /Analysis workspace/ });
    back.focus();
    expect(back).toHaveFocus();
    await user.keyboard("{Enter}");
    await waitFor(() =>
      expect(screen.getByText("Audio Embeddings")).toBeInTheDocument(),
    );
  });

  it("UI-09 restores the previous window on browser back", async () => {
    await renderApp("/");
    const user = userEvent.setup();
    await user.click(await screen.findByRole("link", { name: /J-Lens Lab/ }));
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { level: 1, name: "J-Lens Lab" }),
      ).toBeInTheDocument(),
    );
    window.history.back();
    await waitFor(() =>
      expect(screen.getByText("Audio Embeddings")).toBeInTheDocument(),
    );
  });
});

describe("TestProviderContract", () => {
  it("UI-10 mounts the documented provider stack in one render", async () => {
    // The contract renderWithProviders reproduces: QueryClientProvider ->
    // TooltipProvider -> Toaster + Sonner -> BrowserRouter (App.tsx:13-17).
    await renderApp("/");
    await waitFor(() =>
      expect(screen.getByText("LIT for Voice")).toBeInTheDocument(),
    );
    // A router-dependent read, proving BrowserRouter is mounted.
    expect(screen.getByRole("link", { name: /J-Lens Lab/ })).toBeInTheDocument();
    // A Radix tooltip renders, proving TooltipProvider is mounted: an unwrapped
    // Tooltip throws "`Tooltip` must be used within `TooltipProvider`".
    expect(screen.getByText("Model:")).toBeInTheDocument();
    // The toast surface is live — asserted by actually notifying, because
    // sonner mounts its viewport lazily and querying for the container proves
    // nothing about whether a message would reach the user.
    const { toast } = await import("sonner");
    toast.success("provider contract probe");
    await waitFor(() =>
      expect(screen.getByText("provider contract probe")).toBeInTheDocument(),
    );
  });

  it("UI-11 fails loudly when an embedding consumer is mounted outside its provider", async () => {
    stubFetch(appRoutes);
    const NearestNeighborsPanel = (
      await import("@/components/eda/NearestNeighborsPanel")
    ).NearestNeighborsPanel;
    expect(() => render(<NearestNeighborsPanel selectedFile={null} />)).toThrow(
      "useEmbedding must be used within an EmbeddingProvider",
    );
  });

  it("UI-12 contains a render-time throw and explains it", async () => {
    // Guards BUG-24. No ErrorBoundary, componentDidCatch,
    // getDerivedStateFromError or Suspense exists anywhere in src/, so a single
    // render-time throw unmounts the entire tree and leaves a blank white page
    // with no message and no reload affordance.
    const { ErrorBoundary } = await import("@/components/ErrorBoundary");
    const Boom = () => {
      throw new Error("synthetic render failure");
    };
    render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>,
    );
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Reload the page" }),
    ).toBeInTheDocument();
    // Plain language: the raw Error text must not be the user-facing message.
    expect(screen.queryByText("synthetic render failure")).not.toBeInTheDocument();
  });
});
