/**
 * Source-level assertions.
 *
 * Used ONLY where a runtime oracle is unavailable and the alternative would be
 * no coverage at all. Radix renders a Select's items in a portal that exists
 * only while the menu is open, and opening a Select under jsdom costs ~64
 * seconds even for a bare two-item Select with no application code involved
 * (measured; see the plan doc's Special Considerations). Menu CONTENT is
 * therefore asserted against the component source instead, and every such case
 * says so in its name.
 *
 * This is the same device 3.1.2 used for its control-plane import-purity check:
 * an honest assertion made at the only layer where the property is observable,
 * labelled as such, rather than a runtime assertion that cannot be made.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

export function readComponentSource(relativePath: string): string {
  return readFileSync(resolve(process.cwd(), relativePath), "utf8");
}
