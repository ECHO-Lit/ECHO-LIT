/**
 * Radix interaction helpers for jsdom.
 *
 * `userEvent.click()` does NOT open a Radix Select under jsdom, with or without
 * `pointerEventsCheck: 0` — Radix opens on a pointerdown carrying pointer
 * geometry jsdom's synthetic events do not supply, and the click silently does
 * nothing (measured: the listbox never mounts and the test times out).
 *
 * Keyboard activation works and is used instead. This is a better oracle for
 * this section anyway: RUP §3.1.3 names "use of access methods (tab keys,
 * mouse movements, accelerator keys)" explicitly, so driving the menus from the
 * keyboard asserts a requirement rather than merely working around a limitation.
 * Cases that must prove mouse activation specifically say so and assert the
 * handler directly.
 */
import { fireEvent, screen, waitFor } from "@testing-library/react";

/** Opens a Radix Select trigger via keyboard and waits for its listbox. */
export async function openSelect(trigger: HTMLElement): Promise<HTMLElement[]> {
  fireEvent.keyDown(trigger, { key: "Enter", code: "Enter" });
  await waitFor(() => {
    if (screen.queryAllByRole("option").length === 0) {
      throw new Error("Radix Select did not open: no option role present");
    }
  });
  return screen.getAllByRole("option");
}

/**
 * Reads the option labels of a Select.
 *
 * The menu is left OPEN deliberately: Radix restores focus and unmounts the
 * listbox asynchronously on close, and reopening the same trigger inside one
 * test races that teardown (measured: repeated open/close cycles time out).
 * Each case renders fresh, so a single open per Select is all that is needed.
 */
export async function selectOptionLabels(trigger: HTMLElement): Promise<string[]> {
  const options = await openSelect(trigger);
  return options.map((o) => o.textContent ?? "");
}

/** Asserts a Select opens to an EMPTY listbox, without waiting for options. */
export async function selectOpensEmpty(trigger: HTMLElement): Promise<boolean> {
  fireEvent.keyDown(trigger, { key: "Enter", code: "Enter" });
  await new Promise((resolve) => setTimeout(resolve, 50));
  return screen.queryAllByRole("option").length === 0;
}

/** Opens a Select and activates the option whose text matches exactly. */
export async function chooseOption(trigger: HTMLElement, label: string): Promise<void> {
  const options = await openSelect(trigger);
  const match = options.find((o) => o.textContent?.trim() === label);
  if (!match) {
    throw new Error(
      `Option "${label}" not offered. Present: ${options
        .map((o) => JSON.stringify(o.textContent))
        .join(", ")}`,
    );
  }
  fireEvent.click(match);
}
