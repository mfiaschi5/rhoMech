// Playwright test of the molecule workbench (#/workbench), against the server e2e/serve.mjs
// starts (cached GFN2 chemistry when ethanol and the smoke reactants are in the shared cache,
// else synthetic EHT, which always computes without xTB):
//   draw ethanol by clicks (bond tool: a click on the canvas gives atoms 1-2, a click on atom 2
//   grows atom 4, "o" on atom 4 makes it O), Compute, wait for ready, switch to sticks, click the
//   HOMO level -> its orbital surface layer is drawn (or, without WebGL, its cube fetched), switch
//   the set to pm_mulliken -> the orbital rows are Pipek-Mezey LMOs, add a density layer, open
//   the atoms tab and click the O row -> the 3D view selects it and the sketch highlights O.
// The projection test imports camphor (a bicycle RDKit's 2D clean-up draws flat) through the
// sketcher's import panel, computes it, turns the 3D view by a mouse drag and presses 3D->2D:
// the atoms move, the depth and the crossing breaks are on, the fix report shows no label
// overlap, the wedges still draw the record's stereo (the page stays linked), and the
// projection matches 3Dmol's own screen positions (data-projection-residual). With
// KSMECH_E2E_SHOTS=<dir> it saves before/after screenshots there.
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

test.describe("projection", () => {
  test.use({ deviceScaleFactor: 2 }); // sharper screenshots

  test("3D -> 2D projection of camphor: depth, crossing breaks, fix report, stereo kept", async ({ page, request }) => {
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
    const project = page.getByTestId("sketch-project");
    await expect(project).toBeDisabled();
    await expect(project).toHaveAttribute("data-reason", "compute the molecule first");
    await shot(page, "sketch", "camphor-1-clean2d.png");

    await page.getByTestId("wb-compute").click();
    await expect(page.locator("[data-testid=wb-status][data-status=ready]")).toBeVisible({ timeout: 180_000 });
    const viewer = page.getByTestId("wb-viewer");
    await expect(viewer).toHaveAttribute("data-has-geometry", "true", { timeout: 60_000 });
    const wb = page.getByTestId("workbench");
    await expect(wb).toHaveAttribute("data-linked", "true");
    if ((await viewer.getAttribute("data-state")) !== "ready") {
      await expect(project).toHaveAttribute("data-reason", /no 3D view/);
      test.skip(true, "no WebGL in this browser: the projection needs the 3D view");
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
    await shot(page, "wb-viewer", "camphor-2-view3d.png");

    const before = await atomPositions(page);
    await project.click();
    await expect.poll(() => atomPositions(page)).not.toBe(before);
    await expect(sketch).toHaveAttribute("data-has-depth", "true");
    await expect(sketch).toHaveAttribute("data-break-crossings", "true");
    await expect(page.getByTestId("sketch-break-toggle")).toHaveAttribute("data-active", "true");
    const report = page.getByTestId("sketch-fix-report");
    await expect(report).toHaveAttribute("data-kind", "project");
    await expect(report).toHaveAttribute("data-label-overlaps", "0");
    await expect(report).toHaveAttribute("data-stereo", /^(kept|repaired)$/);
    console.log(`[workbench] projection: ${(await report.textContent()) ?? ""}`);
    const residual = Number(await wb.getAttribute("data-projection-residual"));
    console.log(`[workbench] projection vs 3Dmol modelToScreen: residual ${residual}`);
    expect(residual).toBeLessThan(0.1);
    const crossings = Number(await sketch.getAttribute("data-crossings"));
    if (crossings > 0) await expect(page.locator("[data-testid^=sketch-bond-][data-broken]").first()).toBeAttached();
    await expect(page.locator("[data-testid^=sketch-bond-][data-stereo=wedge], [data-testid^=sketch-bond-][data-stereo=hash]")).toHaveCount(2);
    await expect(wb).toHaveAttribute("data-linked", "true");
    await expect(page.getByTestId("wb-stale-note")).toHaveCount(0);
    await shot(page, "sketch", "camphor-3-projected.png");

    await page.getByTestId("sketch-fix").click();
    await expect(report).toHaveAttribute("data-kind", "fix");
    await expect(report).toHaveAttribute("data-label-overlaps", "0");
    await expect(wb).toHaveAttribute("data-linked", "true");

    // one undo step per action: back to the imported drawing
    await page.getByTestId("sketch-undo").click();
    await page.getByTestId("sketch-undo").click();
    await expect(sketch).toHaveAttribute("data-has-depth", "false");
    await expect(sketch).toHaveAttribute("data-break-crossings", "false");
  });
});
