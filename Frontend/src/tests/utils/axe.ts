/**
 * axe-core assertion helper — RUP §3.1.3, SRS §3.12 ("Interface design aims to
 * conform to common web usability and accessibility conventions").
 *
 * Two deliberate constraints:
 *
 * 1. Every call names the rules it claims to verify. Running the full ruleset
 *    against a mounted fragment is close to vacuous — a fragment is not a page,
 *    so page-level rules either fire spuriously or pass for the wrong reason.
 *    The plan doc's Test Case Matrix records the rule ids per case.
 *
 * 2. `color-contrast` is disabled globally. `test.css` is false, so no Tailwind
 *    declarations exist and axe would evaluate transparent-on-transparent.
 *    Colour contrast is excluded from this section (see the plan doc's
 *    Deliberately Excluded table) rather than asserted dishonestly.
 */
import axe, { type Result, type RuleObject } from "axe-core";
import { expect } from "vitest";

export async function axeViolations(
  target: Element | Document,
  ruleIds: string[],
): Promise<Result[]> {
  const rules: RuleObject = { "color-contrast": { enabled: false } };
  const results = await axe.run(target as Element, {
    runOnly: { type: "rule", values: ruleIds },
    rules,
    resultTypes: ["violations"],
  });
  return results.violations;
}

/**
 * Asserts no violations, formatting any that occur as the offending HTML.
 * The `toBe("")` shape is chosen so a failure prints the actual markup, which
 * is the only thing that makes an axe failure actionable.
 */
export async function expectNoAxeViolations(
  target: Element | Document,
  ruleIds: string[],
): Promise<void> {
  const violations = await axeViolations(target, ruleIds);
  const report = violations
    .map(
      (v) =>
        `${v.id} (${v.impact}): ${v.help}\n` +
        v.nodes
          .map((n) => `  - ${n.target.join(" ")}\n    ${n.html}`)
          .join("\n"),
    )
    .join("\n\n");
  expect(report).toBe("");
}
