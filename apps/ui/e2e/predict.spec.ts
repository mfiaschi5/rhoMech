// Playwright test of the spec-v9 Predict page (#/predict), against the server e2e/serve.mjs
// starts (cached GFN2 or synthetic chemistry; the predictors /api/predictors lists):
//   MeO- + MeCl: the reactant rows with a Draw round trip through the workbench sketcher, the
//   predictor select (unavailable predictors greyed with their reason, the "scores, not
//   probabilities" badge for the floor), beam and max steps, the job, the network and its list,
//   the ranked pathways, a step drawn as its reaction matrix (R marks, Bézier arrows, product),
//   its score in the predictor's unit, the orbital reading with the 3D orbitals, and "Correct
//   this step" (the Correct page with the step's R preloaded). The job view has no control
//   sticking out of its panel at 1024, 1440 and 1920 px.
// Skipped, with the server's reason, while the predict capability is off.
import { expect, test, type Page } from "@playwright/test";

import { browserPath, NO_BROWSER_REASON } from "./browser";
import { entryCount, logChemistry, skipUnless, waitRdkit } from "./helpers";

test.skip(!browserPath, NO_BROWSER_REASON);

const JOB_HASH = /#\/predict\/job\/([^/?]+)/;

interface PredictorLike {
  name: string;
  kind: string;
  available: boolean;
  reason: string | null;
  scores_are_probabilities: boolean;
}

interface EdgeLike {
  id: string;
  matrix: { bonds: [number, number, number][]; lone: [number, number][] };
  arrows: unknown[];
  readings: unknown[];
}

async function fillReactants(page: Page): Promise<void> {
  await page.goto("/#/predict");
  await waitRdkit(page);
  await page.getByTestId("predict-smiles-0").fill("C[O-]");
  await page.getByTestId("predict-add-reactant").click();
  await page.getByTestId("predict-smiles-1").fill("C");
  // Draw: the sketcher opens on the row's molecule and "Use" writes its canonical SMILES back;
  // closing without "Use" leaves the row alone
  await page.getByTestId("predict-draw-1").click();
  const modal = page.getByTestId("predict-draw-modal");
  await expect(modal.locator("[data-testid=sketch][data-smiles=C]")).toBeVisible({ timeout: 30_000 });
  await page.getByTestId("predict-draw-modal-close").click();
  await expect(modal).toHaveCount(0);
  await expect(page.getByTestId("predict-smiles-1")).toHaveValue("C");
  await page.getByTestId("predict-smiles-1").fill("ClC");
  await page.getByTestId("predict-draw-1").click();
  await expect(page.getByTestId("predict-draw-smiles")).toHaveText("CCl", { timeout: 30_000 });
  await page.getByTestId("predict-draw-use").click();
  await expect(page.getByTestId("predict-smiles-1")).toHaveValue("CCl");
  await expect(page.getByTestId("predict-depiction-0").locator("svg")).toBeVisible();
  await expect(page.locator("[data-testid=predict-preflight-0][data-ok=true]")).toBeVisible({ timeout: 60_000 });
  await expect(page.locator("[data-testid=predict-preflight-1][data-ok=true]")).toBeVisible({ timeout: 60_000 });
}

async function checkPredictors(page: Page, list: PredictorLike[]): Promise<PredictorLike> {
  for (const p of list) {
    await expect(page.getByTestId(`predict-predictor-option-${p.name}`)).toHaveAttribute("data-available", String(p.available));
    if (!p.available) await expect(page.getByTestId(`predict-predictor-reason-${p.name}`)).toContainText(p.reason ?? "unavailable");
  }
  const chosen = list.find((p) => p.kind === "floor" && p.available) ?? list.find((p) => p.available);
  if (!chosen) throw new Error("no predictor is available");
  await page.getByTestId("predict-predictor").selectOption(chosen.name);
  const badge = page.getByTestId("predict-score-badge");
  if (chosen.scores_are_probabilities) await expect(badge).toHaveCount(0);
  else await expect(badge).toContainText("not probabilities");
  return chosen;
}

/** No control of a .pv-panel sticks out of it and the page does not scroll sideways. */
async function expectNoOverflow(page: Page, label: string): Promise<void> {
  const problems = await page.evaluate(() => {
    const out: string[] = [];
    if (document.documentElement.scrollWidth > window.innerWidth + 1) out.push(`page ${document.documentElement.scrollWidth} > ${window.innerWidth}`);
    for (const panel of document.querySelectorAll(".pv-panel")) {
      const pr = panel.getBoundingClientRect();
      for (const el of panel.querySelectorAll("button, select, input, a.btn")) {
        const r = el.getBoundingClientRect();
        if (r.width === 0 || el.closest(".pv-table-wrap, .network-legend-panel, .pathway-list")) continue;
        if (r.right > pr.right + 1 || r.left < pr.left - 1) out.push(`${el.getAttribute("data-testid") ?? el.tagName} sticks out`);
      }
    }
    return out;
  });
  expect(problems, label).toEqual([]);
}

test("MeO- + MeCl: network, pathways, a step as R with its arrows and reading, then correct it", async ({ page, request }) => {
  test.setTimeout(300_000);
  await skipUnless(request, "predict");
  await logChemistry(request, "predict");
  const predictors = (await (await request.get("/api/predictors")).json()) as PredictorLike[];
  await fillReactants(page);
  const chosen = await checkPredictors(page, predictors);
  await page.getByTestId("predict-beam").fill("3");
  await page.getByTestId("predict-max-steps").fill("2");
  await page.getByTestId("predict-submit").click();
  await page.waitForURL(JOB_HASH, { timeout: 60_000 });
  const jobId = JOB_HASH.exec(page.url())?.[1] ?? "";
  await expect(page.locator("[data-testid=predict-job-status][data-status=done]")).toBeVisible({ timeout: 240_000 });
  await expect(page.getByTestId("predict-network")).not.toHaveAttribute("data-edge-count", "0");
  await expect(page.locator("[data-testid^=predict-network-edge-]").first()).toBeVisible();

  // the first pathway is shown; its first step opens the step panel
  await expect(page.locator("[data-testid=predict-pathway-1][data-selected=true]")).toBeVisible();
  await page.getByTestId("predict-pathway-step-1-0").click();
  const panel = page.getByTestId("predict-step-panel");
  await expect(panel).toBeVisible();
  const edgeId = (await panel.getAttribute("data-edge")) ?? "";
  const job = (await (await request.get(`/api/jobs/${jobId}`)).json()) as { result: { edges: EdgeLike[] } };
  const edge = job.result.edges.find((e) => e.id === edgeId);
  expect(edge, `edge ${edgeId} in the job's network`).toBeTruthy();
  const nEntries = edge!.matrix.bonds.length + edge!.matrix.lone.length;
  await expect(page.locator("[data-testid^=predict-step-canvas-r-]")).toHaveCount(nEntries, { timeout: 60_000 });
  await expect(page.locator("[data-testid^=predict-step-canvas-arrow-]")).toHaveCount(edge!.arrows.length);
  await expect(page.locator("[data-testid^=predict-step-arrow-text-]")).toHaveCount(edge!.arrows.length);
  for (const a of await page.locator("[data-testid^=predict-step-canvas-arrow-] .pv-arrow-line").all()) expect(await a.getAttribute("d")).toMatch(/^M [-\d.]+ [-\d.]+ C /);
  await expect(page.getByTestId("predict-step-product")).not.toHaveAttribute("data-smiles", "");
  await expect(page.getByTestId("predict-step-score")).toHaveAttribute("data-unit", chosen.scores_are_probabilities ? "logp" : "score");

  console.log(`[predict] step ${edgeId}: ${nEntries} R entries, ${edge!.arrows.length} arrows, ${edge!.readings.length} readings`);
  if (edge!.readings.length) {
    await expect(page.locator("[data-testid^=predict-step-reading-]").first()).toBeVisible();
    await page.getByTestId("predict-step-reading-3d-0").click();
    const viewer = page.getByTestId("predict-orbital-viewer");
    await expect(viewer).toHaveAttribute("data-state", /^(ready|webgl-unavailable)$/, { timeout: 30_000 });
    await expect(viewer).toHaveAttribute("data-cubes", "ready", { timeout: 180_000 });
    console.log(`[predict] orbital viewer: ${await viewer.getAttribute("data-state")}`);
    await page.getByTestId("predict-orbital-modal-close").click();
  }

  for (const width of [1024, 1440, 1920]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.waitForTimeout(300);
    await expectNoOverflow(page, `job view at ${width} px`);
  }
  await page.setViewportSize({ width: 1400, height: 1000 });

  await page.getByTestId("predict-correct-step").click();
  await page.waitForURL(new RegExp(`#/correct/${jobId}/edge/${edgeId}`));
  await expect(page.getByTestId("correct-editor")).toBeVisible();
  await expect.poll(() => entryCount(page)).toBe(nEntries);
});
