/**
 * Global vitest setup — RUP Master Test Plan §3.1.3 User Interface Testing.
 *
 * Sections 3.1.1 and 3.1.2 avoid global fixtures on principle. That is not
 * achievable here: jsdom does not implement browser APIs that Radix and
 * react-resizable-panels call unconditionally during *mount*, before any test
 * body runs, so there is no per-module seam at which to inject them.
 *
 * This file therefore contains ONLY:
 *   - absent-browser-API shims (a missing one is a hard TypeError, not a
 *     softer test),
 *   - jest-dom matchers,
 *   - per-test cleanup.
 *
 * It contains NO application state, NO network stand-in, NO window.confirm and
 * NO component mock. Everything that changes application behaviour is scoped
 * per module or per test, which is where those decisions belong.
 */
import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

/**
 * Defines a property only when it is absent, tolerating jsdom prototypes that
 * expose non-writable accessors (`??=` throws on those).
 *
 * The presence test is deliberately on the VALUE, not on `key in target`.
 * jsdom declares several unimplemented APIs as accessors that resolve to
 * `undefined` — `window.matchMedia` is one — so an `in` check reports them as
 * present and silently skips the shim, producing a "not a function" TypeError
 * at mount instead.
 */
function definePolyfill(target: object, key: string, value: unknown): void {
  if ((target as Record<string, unknown>)[key] !== undefined) return;
  Object.defineProperty(target, key, {
    value,
    configurable: true,
    writable: true,
  });
}

class NoopObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords() {
    return [];
  }
}

// react-resizable-panels PanelGroup; Radix ScrollArea, Select and Popper.
//
// A no-op is correct here. An observer that actually delivers a measurement was
// tried and re-entered: the callback re-renders, the re-render re-observes, and
// the run never terminates.
definePolyfill(globalThis, "ResizeObserver", NoopObserver);
definePolyfill(globalThis, "IntersectionObserver", NoopObserver);

// Radix Select scrolls the highlighted item into view when it opens.
definePolyfill(Element.prototype, "scrollIntoView", function () {});

// Radix Select/Slider/Switch use pointer capture; jsdom has no PointerEvent API.
definePolyfill(Element.prototype, "hasPointerCapture", () => false);
definePolyfill(Element.prototype, "setPointerCapture", function () {});
definePolyfill(Element.prototype, "releasePointerCapture", function () {});

// hooks/use-mobile.tsx and ui/sonner.tsx (via next-themes) read matchMedia.
definePolyfill(window, "matchMedia", (query: string) => ({
  matches: false,
  media: query,
  onchange: null,
  addListener() {},
  removeListener() {},
  addEventListener() {},
  removeEventListener() {},
  dispatchEvent() {
    return false;
  },
}));

// WaveformViewer's blob fallback path and its `new Audio()` probe.
definePolyfill(URL, "createObjectURL", () => "blob:echo-test");
definePolyfill(URL, "revokeObjectURL", () => {});
definePolyfill(HTMLMediaElement.prototype, "play", function () {
  return Promise.resolve();
});
definePolyfill(HTMLMediaElement.prototype, "pause", function () {});
