// Playwright test of the spec-v9 Predict page (#/predict), against the server e2e/serve.mjs
// starts (cached GFN2 or synthetic chemistry; the predictors /api/predictors lists):
//   MeO- + MeCl: the reactant rows with a Draw round trip through the workbench sketcher, the
//   predictor select (unavailable predictors greyed with their reason, the "scores, not
//   probabilities" badge for the floor), beam and max steps, the job, the network and its list,
//   the ranked pathways, a step drawn as its reaction matrix (R marks, Bézier arrows, product),
//   its score in the predictor's unit, the orbital reading with the 3D orbitals, and "Correct
//   this step" (the Correct page with the step's R preloaded). The job view has no control
//   sticking out of its panel at 1024, 1440 and 1920 px.
//   The network: a tiny random checkpoint (tests/serve_fake_model.py) under the server's data
//   root, chosen in the checkpoint picker (the default: the best real run), its job with the
//   checkpoint in the header, probabilities (no "not probabilities" badge, log p scores), and an
//   expansion with the same predictor and checkpoint.
// Skipped, with the server's reason, while the predict capability is off.
import { execFileSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

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

/** The two network viewports behind the header's switch (remembered across a reload): the
 *  scheme's cards never overlap; its double-click on a state without steps out expands it
 *  (POST /jobs/{id}/expand) and the network reloads; the pathway explorer's cards. */
async function checkVariants(page: Page, jobId: string, request: import("@playwright/test").APIRequestContext): Promise<void> {
  const net = page.getByTestId("predict-network");
  await page.getByTestId("predict-network-variant-layered").click();
  await expect(net).toHaveAttribute("data-variant", "layered");
  const cards = page.locator("[data-testid=predict-network] .nv-card-state");
  await expect(cards.first()).toBeVisible();
  const boxes = await cards.evaluateAll((els) => els.map((e) => e.getBoundingClientRect()).map((r) => [r.left, r.top, r.right, r.bottom]));
  for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
    const [a, b] = [boxes[i], boxes[j]];
    expect(a[0] < b[2] - 0.5 && b[0] < a[2] - 0.5 && a[1] < b[3] - 0.5 && b[1] < a[3] - 0.5, `cards ${i} and ${j} overlap`).toBe(false);
  }
  await page.getByTestId("predict-network-variant-explorer").click();
  await expect(net).toHaveAttribute("data-variant", "explorer");
  await expect(page.locator("[data-testid^=predict-net-card-]").first()).toBeVisible();
  await page.reload();
  await expect(net).toHaveAttribute("data-variant", "explorer");
  await page.getByTestId("predict-network-variant-layered").click();
  const job = (await (await request.get(`/api/jobs/${jobId}`)).json()) as { result: { root: string; nodes: { id: string; depth: number }[]; edges: { source: string; target: string }[] } };
  const depth = new Map(job.result.nodes.map((n) => [n.id, n.depth] as const));
  const leaf = job.result.nodes.find((n) => n.id !== job.result.root && !job.result.edges.some((e) => e.source === n.id && (depth.get(e.target) ?? 0) > n.depth));
  if (!leaf) return;
  const before = Number(await net.getAttribute("data-edge-count"));
  await page.getByTestId(`predict-net-node-${leaf.id}`).dblclick();
  await page.waitForURL(/[?&]rev=\d+/, { timeout: 180_000 });
  const after = Number(await net.getAttribute("data-edge-count"));
  console.log(`[predict] expanded ${leaf.id}: ${before} -> ${after} steps`);
  expect(after).toBeGreaterThanOrEqual(before);
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
  await page.getByTestId("predict-network-list").locator("summary").click();
  await expect(page.locator("[data-testid^=predict-network-edge-]").first()).toBeVisible();
  await checkVariants(page, jobId, request);

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

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");

/** The fake run `e2e/tiny` (and a smoke run beside it) under the server's data root. */
async function writeFakeRuns(request: APIRequestContext): Promise<void> {
  const meta = (await (await request.get("/api/meta")).json()) as { data_root: string };
  for (const run of ["tiny", "smoke_tiny"]) {
    const runDir = join(meta.data_root, "data", "models", "e2e", run);
    const out = execFileSync("uv", ["run", "--no-sync", "--project", REPO, "python", "-m", "tests.serve_fake_model", runDir], { cwd: REPO, encoding: "utf8", timeout: 240_000 });
    console.log(`[predict/network] ${out.trim().replace(/\n/g, "; ")}`);
  }
}

interface NetJob {
  status: string;
  result: { edges: { id: string; logp: number | null; predictor_version: string | null; source: string; target: string }[]; nodes: { id: string; status: string; is_root: boolean }[]; provenance: { predictor_version: string; checkpoint: { run: string; file: string } | null }; scores_are_probabilities: boolean } | null;
}

test("the network predictor: checkpoint picker, probabilities, the job's checkpoint, an expansion", async ({ page, request }) => {
  test.setTimeout(300_000);
  await skipUnless(request, "predict");
  await logChemistry(request, "predict/network");
  await writeFakeRuns(request);
  const predictors = (await (await request.get("/api/predictors")).json()) as PredictorLike[];
  const net = predictors.find((p) => p.name === "network");
  test.skip(!net?.available, `the network is unavailable: ${net?.reason}`);

  await page.goto("/#/predict");
  await waitRdkit(page);
  await page.getByTestId("predict-smiles-0").fill("C[O-]");
  await page.getByTestId("predict-add-reactant").click();
  await page.getByTestId("predict-smiles-1").fill("CCl");
  await expect(page.locator("[data-testid=predict-preflight-1][data-ok=true]")).toBeVisible({ timeout: 60_000 });
  await page.getByTestId("predict-predictor").selectOption("network");
  await expect(page.getByTestId("predict-score-badge")).toHaveCount(0);
  const picker = page.getByTestId("predict-checkpoint");
  await expect(picker).toHaveValue("e2e/tiny/best"); // the best real run, not the smoke one
  await expect(page.getByTestId("predict-checkpoint-option-e2e-smoke_tiny-best")).toHaveAttribute("data-smoke", "true");
  await expect(page.getByTestId("predict-checkpoint-option-e2e-smoke_tiny-best")).toContainText("smoke");
  await picker.selectOption("e2e/tiny/last");
  await page.getByTestId("predict-beam").fill("2");
  await page.getByTestId("predict-max-steps").fill("2");
  await page.getByTestId("predict-submit").click();
  await page.waitForURL(JOB_HASH, { timeout: 60_000 });
  const jobId = JOB_HASH.exec(page.url())?.[1] ?? "";
  await expect(page.locator("[data-testid=predict-job-status][data-status=done]")).toBeVisible({ timeout: 240_000 });
  const header = page.getByTestId("predict-job-header");
  await expect(header).toHaveAttribute("data-probabilities", "true");
  await expect(page.getByTestId("predict-job-checkpoint")).toHaveAttribute("data-run", "tiny");
  await expect(page.getByTestId("predict-job-checkpoint")).toHaveAttribute("data-file", "last");
  await expect(header.getByTestId("predict-score-badge")).toHaveCount(0);

  const job = (await (await request.get(`/api/jobs/${jobId}`)).json()) as NetJob;
  expect(job.result?.scores_are_probabilities).toBe(true);
  expect(job.result?.provenance.checkpoint?.file).toBe("last");
  const edges = job.result?.edges ?? [];
  console.log(`[predict/network] ${edges.length} steps, logp ${edges.map((e) => e.logp?.toFixed(2)).join(", ")}`);
  test.skip(!edges.length, "the tiny random model proposed no valid step");
  expect(edges.every((e) => e.logp !== null)).toBe(true);
  await page.goto(`/#/predict/job/${jobId}?edge=${edges[0].id}`);
  await expect(page.getByTestId("predict-step-panel")).toHaveAttribute("data-edge", edges[0].id, { timeout: 60_000 });
  await expect(page.getByTestId("predict-step-score")).toHaveAttribute("data-unit", "logp");
  await expect(page.getByTestId("predict-step-score")).toContainText("log p");

  // a double-click on a state without steps out expands it with the same checkpoint
  await page.goto(`/#/predict/job/${jobId}`);
  await page.getByTestId("predict-network-variant-layered").click();
  const sources = new Set(edges.map((e) => e.source));
  const leaf = job.result?.nodes.find((n) => !n.is_root && !sources.has(n.id));
  test.skip(!leaf, "no state without steps out to expand");
  await page.getByTestId(`predict-net-node-${leaf!.id}`).dblclick();
  await page.waitForURL(/[?&]rev=\d+/, { timeout: 180_000 });
  const grown = (await (await request.get(`/api/jobs/${jobId}`)).json()) as NetJob;
  const versions = new Set((grown.result?.edges ?? []).map((e) => e.predictor_version));
  expect([...versions]).toEqual([job.result?.provenance.predictor_version]);
  await page.setViewportSize({ width: 1400, height: 1000 });
});
