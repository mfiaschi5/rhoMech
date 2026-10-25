// Playwright test of the molecule workbench (#/workbench), against the server e2e/serve.mjs
// starts (cached GFN2 chemistry when ethanol and the smoke reactants are in the shared cache,
// else synthetic EHT, which always computes without xTB):
//   draw ethanol by clicks (bond tool: a click on the canvas gives atoms 1-2, a click on atom 2
//   grows atom 4, "o" on atom 4 makes it O), Compute, wait for ready, switch to sticks, click the
//   HOMO level -> its orbital surface layer is drawn (or, without WebGL, its cube fetched), switch
//   the set to pm_mulliken -> the orbital rows are Pipek-Mezey LMOs, add a density layer, open
//   the atoms tab and click the O row -> the 3D view selects it and the sketch highlights O.
// Skipped, with the server's reason, while the workbench capability is off.
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

test("draw ethanol, compute, orbitals, surfaces and linked selection", async ({ page, request }) => {
  test.setTimeout(300_000);
  const meta = (await (await request.get("/api/meta")).json()) as {
    chemistry: string;
    capabilities?: Record<string, { enabled: boolean; reason: string | null }>;
  };
  const cap = meta.capabilities?.workbench ?? { enabled: true, reason: null };
  test.skip(!cap.enabled, `capability workbench is off: ${cap.reason ?? ""}`);
  console.log(`[workbench] /api/meta.chemistry = ${meta.chemistry}`);
  test.info().annotations.push({ type: "chemistry", description: meta.chemistry });

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
