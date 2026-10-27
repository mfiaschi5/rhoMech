// Shared helpers of the spec-v9 Predict and Correct e2e tests: capability skips, the chemistry
// log line, and the pool canvas read as atoms, bonds and screen positions (PoolCanvas's
// data-atom-xy, data-elements, data-charges and data-bonds).
import { expect, test, type APIRequestContext, type Locator, type Page } from "@playwright/test";

export interface CapabilityLike {
  enabled: boolean;
  reason: string | null;
}

export async function capability(request: APIRequestContext, name: string): Promise<CapabilityLike> {
  const meta = (await (await request.get("/api/meta")).json()) as { capabilities?: Record<string, CapabilityLike> };
  return meta.capabilities?.[name] ?? { enabled: true, reason: null };
}

/** Skip the current test, with the server's reason, when a capability it needs is off. */
export async function skipUnless(request: APIRequestContext, ...names: string[]): Promise<void> {
  for (const name of names) {
    const cap = await capability(request, name);
    test.skip(!cap.enabled, `capability ${name} is off: ${cap.reason ?? ""}`);
  }
}

export async function logChemistry(request: APIRequestContext, tag: string): Promise<string> {
  const meta = (await (await request.get("/api/meta")).json()) as { chemistry: string };
  console.log(`[${tag}] /api/meta.chemistry = ${meta.chemistry}`);
  test.info().annotations.push({ type: "chemistry", description: meta.chemistry });
  return meta.chemistry;
}

export type XY = [number, number];

export interface CanvasInfo {
  /** Page coordinates of pool atom i. */
  at(i: number): XY;
  /** Page coordinates of the point at t along i -> j. */
  along(i: number, j: number, t?: number): XY;
  elements: Record<string, string>;
  charges: Record<string, number>;
  bonds: [number, number, number][];
  /** The pool atoms of an element. */
  atomsOf(el: string): number[];
  /** The atoms bonded to i. */
  neighbours(i: number): number[];
}

/** The pool canvas `testId` once drawn, read through its data attributes. */
export async function readCanvas(page: Page, testId: string): Promise<CanvasInfo> {
  const c = page.getByTestId(testId);
  await expect(c).not.toHaveAttribute("data-atom-xy", "{}", { timeout: 60_000 });
  const box = await c.boundingBox();
  if (!box) throw new Error(`${testId} has no box`);
  const xy = JSON.parse((await c.getAttribute("data-atom-xy")) ?? "{}") as Record<string, XY>;
  const elements = JSON.parse((await c.getAttribute("data-elements")) ?? "{}") as Record<string, string>;
  const charges = JSON.parse((await c.getAttribute("data-charges")) ?? "{}") as Record<string, number>;
  const bonds = ((await c.getAttribute("data-bonds")) ?? "")
    .split(",")
    .filter(Boolean)
    .map((s) => {
      const [pair, order] = s.split(":");
      const [i, j] = pair.split("-").map(Number);
      return [i, j, Number(order)] as [number, number, number];
    });
  const at = (i: number): XY => [box.x + xy[i][0], box.y + xy[i][1]];
  return {
    at,
    along: (i, j, t = 0.5) => [box.x + xy[i][0] * (1 - t) + xy[j][0] * t, box.y + xy[i][1] * (1 - t) + xy[j][1] * t],
    elements,
    charges,
    bonds,
    atomsOf: (el) => Object.keys(elements).filter((k) => elements[k] === el).map(Number),
    neighbours: (i) => bonds.filter(([a, b]) => a === i || b === i).map(([a, b]) => (a === i ? b : a)),
  };
}

/** A mouse drag in a few steps (the arrow tool). */
export async function drag(page: Page, from: XY, to: XY): Promise<void> {
  await page.mouse.move(from[0], from[1]);
  await page.mouse.down();
  await page.mouse.move((from[0] + to[0]) / 2, (from[1] + to[1]) / 2, { steps: 4 });
  await page.mouse.move(to[0], to[1], { steps: 4 });
  await page.mouse.up();
}

export const pair = (i: number, j: number) => (i < j ? `${i}-${j}` : `${j}-${i}`);

/** The number of R entries the Correct page's table lists. */
export async function entryCount(page: Page): Promise<number> {
  return Number(await page.getByTestId("correct-entries").getAttribute("data-count"));
}

export async function waitRdkit(page: Page): Promise<void> {
  await expect(page.locator("body[data-rdkit=ready]")).toBeAttached({ timeout: 90_000 });
}

export function testIdOf(loc: Locator): Promise<string> {
  return loc.getAttribute("data-testid").then((v) => v ?? "");
}
