// Playwright test of the molecule workbench (#/workbench), against the server e2e/serve.mjs
// starts (cached GFN2 chemistry when ethanol and the smoke reactants are in the shared cache,
// else synthetic EHT, which always computes without xTB):
//   draw ethanol by clicks (bond tool: a click on the canvas gives atoms 1-2, a click on atom 2
//   grows atom 4, "o" on atom 4 makes it O), Compute, wait for ready, switch to sticks, click the
//   HOMO level -> its orbital surface layer is drawn (or, without WebGL, its cube fetched), switch
//   the set to pm_mulliken -> the orbital rows are Pipek-Mezey LMOs, add a density layer, open
//   the atoms tab and click the O row -> the 3D view selects it and the sketch highlights O.
// The layout test imports camphor through the sketcher's import panel and presses Clean (the
// server's depiction, POST /api/depict2d): textbook gives RDKit's ring-template drawing, then
// perspective the norbornane template with depth and crossing breaks, the wedges keeping the
// stereo. It computes camphor, turns the 3D view by a mouse drag and presses 3D->2D: a match3d
// layout with the view's depth, crossing breaks, the record's stereo (the page stays linked),
// and the quaternion convention matching 3Dmol's own screen positions
// (data-projection-residual). Undo steps back through the three layouts. With
// KSMECH_E2E_SHOTS=<dir> it saves screenshots there.
// The overflow test computes ethanol and checks, at 1024 and 1440 px, that no button, select or
// input of the workbench sticks out of its panel and that no panel scrolls sideways (only the
// tables scroll inside theirs); a flip keeps the page linked. The transform test imports
// L-alanine and flips it (the wedge becomes a hash, the SMILES is unchanged), turns it with the
// selection frame's rotate handle (one undo step), rotates by the R key and scales from the "⋯"
// menu, the canonical isomeric SMILES unchanged throughout.
// Skipped, with the server's reason, while the workbench capability is off.
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type Locator, type Page } from "@playwright/test";

import { browserPath, NO_BROWSER_REASON } from "./browser";

test.skip(!browserPath, NO_BROWSER_REASON);

const LAYER_DONE = /^(ready|fetched)$/;

async function drawEthanol(page: Page): Promise<void> {
  const sketch = page.getByTestId("sketch");
  await expect(sketch).toHaveAttribute("data-tool", "bond");
  const canvas = page.getByTestId("sketch-canvas");
  const box = await canvas.boundingBox();
  if (!box) throw new Error("the sketch canvas has no box");
  await canvas.click({ position: { x: box.width / 2 - 40, y: box.height / 2 } });
  await page.getByTestId("sketch-atom-2").click();
  await page.getByTestId("sketch-atom-4").hover();
  await page.keyboard.press("o");
  await expect(page.locator('[data-testid=sketch][data-smiles="CCO"][data-valid=true]')).toBeVisible();
}

/** A surface layer drawn in 3D, or with the viewer without WebGL, at least its cube fetched. */
async function expectLayerDone(page: Page, layer: Locator, label: string): Promise<void> {
  await expect(layer).toHaveAttribute("data-state", LAYER_DONE, { timeout: 120_000 });
  const viewerState = await page.getByTestId("wb-viewer").getAttribute("data-state");
  if (viewerState === "ready") await expect(layer).toHaveAttribute("data-state", "ready");
  else await expect(layer).toHaveAttribute("data-fetched", "true");
  console.log(`[workbench] ${label}: layer ${await layer.getAttribute("data-state")}, viewer ${viewerState}`);
}

async function skipWithoutWorkbench(request: import("@playwright/test").APIRequestContext): Promise<string> {
  const meta = (await (await request.get("/api/meta")).json()) as {
    chemistry: string;
    capabilities?: Record<string, { enabled: boolean; reason: string | null }>;
  };
  const cap = meta.capabilities?.workbench ?? { enabled: true, reason: null };
  test.skip(!cap.enabled, `capability workbench is off: ${cap.reason ?? ""}`);
  console.log(`[workbench] /api/meta.chemistry = ${meta.chemistry}`);
  test.info().annotations.push({ type: "chemistry", description: meta.chemistry });
  return meta.chemistry;
}

test("draw ethanol, compute, orbitals, surfaces and linked selection", async ({ page, request }) => {
  test.setTimeout(300_000);
  await skipWithoutWorkbench(request);

  await page.goto("/#/workbench");
  await expect(page.locator("body[data-rdkit=ready]")).toBeAttached({ timeout: 90_000 });
  await drawEthanol(page);

  await page.getByTestId("wb-compute").click();
  await expect(page.locator("[data-testid=wb-status][data-status=ready]")).toBeVisible({ timeout: 180_000 });
  await expect(page).toHaveURL(/[?&]m=m[0-9a-f]{20}/);
  const viewer = page.getByTestId("wb-viewer");
  await expect(viewer).toHaveAttribute("data-state", /^(ready|webgl-unavailable)$/);
  await expect(viewer).toHaveAttribute("data-has-geometry", "true", { timeout: 60_000 });

  await page.getByTestId("rep-select").selectOption("sticks");
  await expect(viewer).toHaveAttribute("data-rep", "sticks");

  const homo = page.locator("[data-testid^=level-][data-frontier=homo]").first();
  await expect(homo).toBeAttached({ timeout: 120_000 });
  const homoIndex = ((await homo.getAttribute("data-testid")) ?? "").replace("level-", "");
  await homo.click();
  const orbital = page.locator(`[data-testid=wb-surface-layer][data-kind=orbital][data-set=canonical][data-index="${homoIndex}"]`);
  await expectLayerDone(page, orbital, `HOMO (MO ${homoIndex})`);

  await page.getByTestId("set-select").selectOption("pm_mulliken");
  const pmRows = page.locator("[data-testid=orbital-row][data-set=pm_mulliken]");
  await expect(pmRows.first()).toBeVisible({ timeout: 120_000 });
  expect(await pmRows.count()).toBeGreaterThan(0);
  await expect(page.locator("[data-testid=orbital-row]:not([data-set=pm_mulliken])")).toHaveCount(0);

  await page.getByTestId("add-layer-kind").selectOption("density");
  await page.getByTestId("wb-add-layer").click();
  await expectLayerDone(page, page.locator("[data-testid=wb-surface-layer][data-kind=density]"), "density");

  await page.getByTestId("tab-atoms").click();
  const oxygen = page.locator("[data-testid=atom-row][data-element=O]");
  await expect(oxygen).toHaveCount(1);
  const oIndex = (await oxygen.getAttribute("data-record-index")) ?? "";
  await oxygen.click();
  await expect(viewer).toHaveAttribute("data-selected-atoms", new RegExp(`(^|,)${oIndex}(,|$)`));
  await expect(page.getByTestId("workbench")).toHaveAttribute("data-linked", "true");
  await expect(page.getByTestId("sketch-atom-4")).toHaveAttribute("data-highlighted", "true");
});

const CAMPHOR = "C[C@@]12CC[C@@H](C1(C)C)CC2=O";

/** Every drawn atom's screen position, as one comparable string. */
async function atomPositions(page: Page): Promise<string> {
  return page.locator("[data-testid^=sketch-atom-] circle.sk-hit").evaluateAll((els) => els.map((e) => `${e.getAttribute("cx")},${e.getAttribute("cy")}`).join(" "));
}

async function shot(page: Page, testId: string, name: string): Promise<void> {
  const dir = process.env.KSMECH_E2E_SHOTS;
  if (!dir) return;
  mkdirSync(dir, { recursive: true });
  await page.getByTestId(testId).screenshot({ path: join(dir, name) });
}

async function expectReport(page: Page, attrs: Record<string, string | RegExp>): Promise<Locator> {
  const report = page.getByTestId("sketch-depict-report");
  for (const [k, v] of Object.entries(attrs)) await expect(report).toHaveAttribute(k, v, { timeout: 60_000 });
  console.log(`[workbench] ${(await report.textContent()) ?? ""}`);
  return report;
}

const MARKS = "[data-testid^=sketch-bond-][data-stereo=wedge], [data-testid^=sketch-bond-][data-stereo=hash]";

test.describe("layout", () => {
  test.use({ deviceScaleFactor: 2 }); // sharper screenshots

  test("camphor: server Clean (ring template, perspective template), compute, 3D->2D (match3d), stereo and link kept", async ({ page, request }) => {
    test.setTimeout(300_000);
    await skipWithoutWorkbench(request);
    await page.goto("/#/workbench");
    await expect(page.locator("body[data-rdkit=ready]")).toBeAttached({ timeout: 90_000 });
    await page.getByTestId("sketch-import").click();
    await page.getByTestId("sketch-import-text").fill(CAMPHOR);
    await page.getByTestId("sketch-import-load").click();
    const sketch = page.getByTestId("sketch");
    await expect(sketch).toHaveAttribute("data-atoms", "11");
    await expect(sketch).toHaveAttribute("data-smiles", /@/);
    const smiles = (await sketch.getAttribute("data-smiles")) ?? "";
    const project = page.getByTestId("sketch-project");
    await expect(project).toBeDisabled();
    await expect(project).toHaveAttribute("data-reason", "compute the molecule first");

    // Clean, textbook: RDKit's ring templates (the server's choice over CoordGen for camphor)
    await expect(page.getByTestId("sketch-clean-style")).toHaveValue("textbook");
    await page.getByTestId("sketch-clean").click();
    await expectReport(page, { "data-kind": "clean", "data-method": "rdkit-templates", "data-stereo": "kept", "data-fallback": "false" });
    await expect(sketch).toHaveAttribute("data-has-depth", "false");
    await expect(sketch).toHaveAttribute("data-smiles", smiles);
    await shot(page, "sketch", "camphor-1-textbook.png");

    // Clean, perspective: the norbornane template, its depth, the crossing drawn broken
    await page.getByTestId("sketch-clean-style").selectOption("perspective");
    await expect(sketch).toHaveAttribute("data-clean-style", "perspective");
    await page.getByTestId("sketch-clean").click();
    await expectReport(page, { "data-kind": "clean", "data-template": "norbornane", "data-stereo": "kept", "data-label-overlaps": "0" });
    await expect(sketch).toHaveAttribute("data-has-depth", "true");
    await expect(sketch).toHaveAttribute("data-break-crossings", "true");
    await expect(sketch).toHaveAttribute("data-smiles", smiles);
    await expect(page.locator(MARKS)).toHaveCount(2);
    if (Number(await sketch.getAttribute("data-crossings")) > 0) await expect(page.locator("[data-testid^=sketch-bond-][data-broken]").first()).toBeAttached();
    await shot(page, "sketch", "camphor-2-perspective.png");

    await page.getByTestId("wb-compute").click();
    await expect(page.locator("[data-testid=wb-status][data-status=ready]")).toBeVisible({ timeout: 180_000 });
    const viewer = page.getByTestId("wb-viewer");
    await expect(viewer).toHaveAttribute("data-has-geometry", "true", { timeout: 60_000 });
    const wb = page.getByTestId("workbench");
    await expect(wb).toHaveAttribute("data-linked", "true");
    if ((await viewer.getAttribute("data-state")) !== "ready") {
      await expect(project).toHaveAttribute("data-reason", /no 3D view/);
      test.skip(true, "no WebGL in this browser: 3D->2D needs the 3D view");
    }
    await expect(project).toBeEnabled({ timeout: 30_000 });

    // turn the molecule: a drag on the 3Dmol canvas
    const box = await page.locator(".wb-canvas canvas").first().boundingBox();
    if (!box) throw new Error("the 3D canvas has no box");
    const [cx, cy] = [box.x + box.width / 2, box.y + box.height / 2];
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    await page.mouse.move(cx + 70, cy + 45, { steps: 8 });
    await page.mouse.up();
    await shot(page, "wb-viewer", "camphor-3-view3d.png");

    const before = await atomPositions(page);
    await project.click();
    await expectReport(page, { "data-kind": "project", "data-method": /^match3d-/, "data-stereo": "kept" });
    await expect.poll(() => atomPositions(page)).not.toBe(before);
    await expect(sketch).toHaveAttribute("data-has-depth", "true");
    await expect(sketch).toHaveAttribute("data-break-crossings", "true");
    await expect(page.getByTestId("sketch-break-toggle")).toHaveAttribute("data-active", "true");
    const residual = Number(await wb.getAttribute("data-projection-residual"));
    console.log(`[workbench] quaternion convention vs 3Dmol modelToScreen: residual ${residual}`);
    expect(residual).toBeLessThan(0.1);
    await expect(page.locator(MARKS)).toHaveCount(2);
    await expect(wb).toHaveAttribute("data-linked", "true");
    await expect(page.getByTestId("wb-stale-note")).toHaveCount(0);
    await shot(page, "sketch", "camphor-4-match3d.png");

    // one undo step per layout: back to the perspective, then the textbook drawing
    await page.getByTestId("sketch-undo").click();
    await expect(page.getByTestId("sketch-depict-report")).toHaveCount(0);
    await expect(sketch).toHaveAttribute("data-has-depth", "true");
    await page.getByTestId("sketch-undo").click();
    await expect(sketch).toHaveAttribute("data-has-depth", "false");
    await expect(sketch).toHaveAttribute("data-break-crossings", "false");
  });
});

/** Buttons, selects and inputs of the workbench outside their panel, and panels that scroll
 *  sideways; the tables (.wb-table-wrap, .wb-orbital-list) and popovers may scroll or overhang. */
async function layoutOverflow(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const bad: string[] = [];
    for (const panel of document.querySelectorAll<HTMLElement>(".workbench .panel")) {
      const r = panel.getBoundingClientRect();
      const name = panel.dataset.testid || panel.className;
      if (panel.scrollWidth > panel.clientWidth + 1) bad.push(`${name} scrolls sideways (${panel.scrollWidth} > ${panel.clientWidth})`);
      for (const el of panel.querySelectorAll<HTMLElement>("button, select, input")) {
        if (el.closest(".wb-table-wrap, .wb-orbital-list, .sk-popover")) continue;
        const b = el.getBoundingClientRect();
        if (b.width === 0) continue;
        if (b.left < r.left - 0.5 || b.right > r.right + 0.5) bad.push(`${name}: ${el.tagName} ${el.dataset.testid ?? el.textContent?.trim()} [${b.left}, ${b.right}] outside [${r.left}, ${r.right}]`);
      }
    }
    if (document.documentElement.scrollWidth > window.innerWidth) bad.push("the page scrolls sideways");
    return bad;
  });
}

async function importSmiles(page: Page, smiles: string): Promise<void> {
  await page.getByTestId("sketch-import").click();
  await page.getByTestId("sketch-import-text").fill(smiles);
  await page.getByTestId("sketch-import-load").click();
}

test("no control outside its panel at 1024 and 1440 px; a flip keeps the link", async ({ page, request }) => {
  test.setTimeout(300_000);
  await skipWithoutWorkbench(request);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("/#/workbench");
  await expect(page.locator("body[data-rdkit=ready]")).toBeAttached({ timeout: 90_000 });
  await importSmiles(page, "CCO");
  await expect(page.locator('[data-testid=sketch][data-smiles="CCO"]')).toBeVisible();
  await page.getByTestId("wb-compute").click();
  await expect(page.locator("[data-testid=wb-status][data-status=ready]")).toBeVisible({ timeout: 180_000 });
  await expect(page.getByTestId("wb-orbitals")).toBeVisible({ timeout: 60_000 });
  await page.getByTestId("tab-atoms").click();
  await expect(page.locator("[data-testid=atom-row]").first()).toBeVisible({ timeout: 60_000 });
  for (const width of [1024, 1440]) {
    await page.setViewportSize({ width, height: 1000 });
    await expect.poll(() => layoutOverflow(page), { message: `overflow at ${width} px` }).toEqual([]);
    await shot(page, "workbench", `layout-${width}.png`);
  }
  const wb = page.getByTestId("workbench");
  await expect(wb).toHaveAttribute("data-linked", "true");
  const before = await atomPositions(page);
  await page.getByTestId("sketch-flip-h").click();
  await expect.poll(() => atomPositions(page)).not.toBe(before);
  await expect(wb).toHaveAttribute("data-linked", "true");
  await expect(page.getByTestId("wb-stale-note")).toHaveCount(0);
});

/** data-smiles once the analysis describes the drawing on screen. */
async function settledSmiles(page: Page): Promise<string> {
  const sketch = page.getByTestId("sketch");
  await expect(sketch).toHaveAttribute("data-status", "ready");
  return (await sketch.getAttribute("data-smiles")) ?? "";
}

test("sketch transforms: flip, rotate handle, R key and scale keep the stereoisomer", async ({ page, request }) => {
  test.setTimeout(180_000);
  await skipWithoutWorkbench(request);
  await page.goto("/#/workbench");
  await expect(page.locator("body[data-rdkit=ready]")).toBeAttached({ timeout: 90_000 });
  await importSmiles(page, "C[C@H](N)C(=O)O");
  const sketch = page.getByTestId("sketch");
  await expect(sketch).toHaveAttribute("data-atoms", "6");
  const smiles = await settledSmiles(page);
  expect(smiles).toContain("@");
  const mark = page.locator(MARKS).first();
  const stereo = (await mark.getAttribute("data-stereo")) ?? "";

  // flip, nothing selected: the whole drawing; the wedge becomes a hash (or back)
  let before = await atomPositions(page);
  await page.getByTestId("sketch-flip-h").click();
  await expect.poll(() => atomPositions(page)).not.toBe(before);
  await expect(mark).toHaveAttribute("data-stereo", stereo === "wedge" ? "hash" : "wedge");
  expect(await settledSmiles(page)).toBe(smiles);

  // select all, drag the rotate handle a quarter turn about the frame
  await page.getByTestId("sketch-canvas").hover();
  await page.keyboard.press("Control+a");
  const frame = page.getByTestId("sketch-frame");
  await expect(frame).toBeVisible();
  const fb = await frame.boundingBox();
  const hb = await page.getByTestId("sketch-frame-rotate").boundingBox();
  if (!fb || !hb) throw new Error("the selection frame has no box");
  const [hx, hy] = [hb.x + hb.width / 2, hb.y + hb.height / 2];
  const [cx, cy] = [fb.x + fb.width / 2, fb.y + fb.height / 2 + 11];
  const flipped = await atomPositions(page);
  await page.mouse.move(hx, hy);
  await page.mouse.down();
  await page.mouse.move(cx + (cy - hy), cy, { steps: 10 });
  await page.mouse.up();
  await expect.poll(() => atomPositions(page)).not.toBe(flipped);
  expect(await settledSmiles(page)).toBe(smiles);
  await page.getByTestId("sketch-undo").click();
  await expect.poll(() => atomPositions(page)).toBe(flipped);

  // R rotates 15 degrees, the menu scales by 10 %
  before = await atomPositions(page);
  await page.getByTestId("sketch-canvas").hover();
  await page.keyboard.press("r");
  await expect.poll(() => atomPositions(page)).not.toBe(before);
  before = await atomPositions(page);
  await page.getByTestId("sketch-more").click();
  await page.getByTestId("sketch-scale-up").click();
  await expect(page.getByTestId("sketch-more-menu")).toHaveCount(0);
  await expect.poll(() => atomPositions(page)).not.toBe(before);
  expect(await settledSmiles(page)).toBe(smiles);
  await shot(page, "sketch", "transform.png");
});
