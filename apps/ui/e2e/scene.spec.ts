// The scene route, #/scene?src=...: the example scenes of public/scenes (a step, a network and
// a molecule exported by `npm run scene:export`) drawn by @ksmech/viz with every /api/ request
// aborted, so the views provably need no server (what the wiki will embed). Each test also
// asserts that the page did not even try the API (no /api/meta of the app shell).
import { expect, test, type Page } from "@playwright/test";

import { browserPath, NO_BROWSER_REASON } from "./browser";

test.skip(!browserPath, NO_BROWSER_REASON);

/** Abort every /api/ request; the list of the URLs that were tried. */
async function offline(page: Page): Promise<string[]> {
  const tried: string[] = [];
  await page.route("**/api/**", (route) => {
    tried.push(route.request().url());
    return route.abort();
  });
  return tried;
}

async function openScene(page: Page, file: string): Promise<void> {
  await page.goto(`/#/scene?src=scenes/${file}`);
  await expect(page.getByTestId("scene-page")).toHaveAttribute("data-state", "ready");
  await expect(page.getByTestId("scene-provenance")).toContainText("chemistry real");
}

test("step scene: the textbook, animated and orbital views offline", async ({ page }) => {
  const tried = await offline(page);
  await openScene(page, "sn2-step.json");
  await expect(page.getByTestId("scene-view")).toHaveAttribute("data-type", "step");
  // textbook: the scheme with its two curved arrows, the product and the orbital caption
  await expect(page.getByTestId("scene-step-canvas-arrow-0")).toHaveAttribute("data-kind", "lone_pair_to_bond");
  await expect(page.getByTestId("scene-step-canvas-arrow-1")).toHaveAttribute("data-kind", "bond_to_lone_pair");
  await expect(page.getByTestId("scene-step-product-drawing")).toHaveAttribute("data-smiles", "COC.[Cl-]");
  await expect(page.getByTestId("scene-step-caption-0")).toContainText("σ*(C–Cl)");
  // animated: the morph's canvas and its R card
  await page.getByTestId("scene-step-variant-animated").click();
  await expect(page.getByTestId("scene-step-view")).toHaveAttribute("data-variant", "animated");
  await expect(page.getByTestId("scene-step-canvas")).toHaveAttribute("data-variant", "animated");
  await expect(page.getByTestId("scene-step-card-entry-0")).toBeVisible();
  // orbitals: the energy diagram of the reading and both cubes read from the scene's assets
  await page.getByTestId("scene-step-variant-orbital").click();
  await expect(page.getByTestId("scene-step-energy")).toHaveAttribute("data-reading", "1-5");
  await expect(page.getByTestId("scene-step-orbital-viewer")).toHaveAttribute("data-cubes", "ready");
  expect(tried).toEqual([]);
});

test("network scene: the scheme and the pathways offline", async ({ page }) => {
  const tried = await offline(page);
  await openScene(page, "sn2-network.json");
  const net = page.getByTestId("predict-network");
  await expect(net).toHaveAttribute("data-variant", "layered");
  await expect(net).toHaveAttribute("data-node-count", "5");
  await expect(net).toHaveAttribute("data-edge-count", "6");
  // the cards carry their structure drawings (RDKit.js, no server)
  await expect(page.locator(".nv-card-state img")).toHaveCount(5);
  await page.locator('[data-testid^="predict-net-node-"]').nth(1).click();
  await expect(page.getByTestId("scene-network-selection")).not.toHaveAttribute("data-node", "");
  await page.getByTestId("scene-network-variant-explorer").click();
  await expect(net).toHaveAttribute("data-variant", "explorer");
  await expect(page.locator('[data-testid^="predict-net-card-"]').first()).toBeVisible();
  expect(tried).toEqual([]);
});

test("molecule scene: 2D, 3D with the HOMO cube, the level diagram offline", async ({ page }) => {
  const tried = await offline(page);
  await openScene(page, "acrolein.json");
  await expect(page.getByTestId("scene-molecule-2d")).toHaveAttribute("data-state", "ready");
  await expect(page.getByTestId("scene-molecule-3d")).toBeVisible();
  const orbital = page.getByTestId("scene-molecule-orbital");
  await expect(orbital).toHaveAttribute("data-cube", "ready");
  await expect(orbital).toContainText("HOMO");
  await expect(page.getByTestId("level-diagram")).toBeVisible();
  // the LUMO's level: its cube is in the scene too
  const homo = await orbital.getAttribute("data-index");
  await page.getByTestId(`level-${Number(homo) + 1}`).click();
  await expect(orbital).toHaveAttribute("data-index", String(Number(homo) + 1));
  await expect(orbital).toHaveAttribute("data-cube", "ready");
  expect(tried).toEqual([]);
});

test("the scene index, an invalid document and another origin are reported", async ({ page }) => {
  const tried = await offline(page);
  await page.goto("/#/scene");
  await expect(page.getByTestId("scene-index")).toHaveAttribute("data-state", "ready");
  await expect(page.getByTestId("scene-link-sn2-step.json")).toBeVisible();
  await page.goto("/#/scene?src=scenes/index.json");
  await expect(page.getByTestId("scene-page")).toHaveAttribute("data-state", "error");
  await expect(page.getByTestId("scene-errors")).toContainText("(document): expected an object");
  await page.goto("/#/scene?src=https://example.org/x.json");
  await expect(page.getByTestId("scene-error")).toContainText("only scenes of this site");
  expect(tried).toEqual([]);
});
