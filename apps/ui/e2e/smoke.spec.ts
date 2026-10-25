// Playwright smoke test of ksmech-serve (design §13.3), against the server e2e/serve.mjs starts:
//   Test 1: MeO- + MeCl with the floor -> the O -> C SN2 arrow from the candidates list, its
//           physics panel and the contact placement in 3D.
//   Test 2: an override of the first root edge to another acceptor re-applies and continues.
//   Test 3: while /api/meta reports predict or correct off (spec v8 port), the nav greys those
//           pages and they show the capability notice with the reason.
// Tests 1 and 2 are skipped, with the server's reason, while their capability is off.
// Each test logs /api/meta.chemistry (cached when both reactants have real GFN2 tokens in the
// shared cache, else synthetic EHT). The SN2 tuple is found through the candidates list, so the
// tests do not depend on the untrained floor's rank of it.
import { expect, test, type APIRequestContext, type Locator, type Page } from "@playwright/test";

import { browserPath, NO_BROWSER_REASON } from "./browser";

const NEAR_PX = 20;
const MAX_CANDIDATE_PAGES = 30;
const JOB_HASH = /#\/predict\/job\/(j[0-9a-f]{16})/;

test.skip(!browserPath, NO_BROWSER_REASON);

interface CandidateLike {
  acceptor: number;
  acceptor_centre_elements: string[];
}

interface CapabilityLike {
  enabled: boolean;
  reason: string | null;
}

/** /api/meta.capabilities[name] (an unknown name reads as enabled). */
async function capability(request: APIRequestContext, name: string): Promise<CapabilityLike> {
  const meta = (await (await request.get("/api/meta")).json()) as {
    capabilities?: Record<string, CapabilityLike>;
  };
  return meta.capabilities?.[name] ?? { enabled: true, reason: null };
}

/** Skip the current test, with the server's reason, when a capability it needs is off. */
async function skipUnless(request: APIRequestContext, ...names: string[]): Promise<void> {
  for (const name of names) {
    const cap = await capability(request, name);
    test.skip(!cap.enabled, `capability ${name} is off: ${cap.reason ?? ""}`);
  }
}

async function logChemistry(request: APIRequestContext): Promise<string> {
  const meta = (await (await request.get("/api/meta")).json()) as { chemistry: string };
  console.log(`[smoke] /api/meta.chemistry = ${meta.chemistry}`);
  test.info().annotations.push({ type: "chemistry", description: meta.chemistry });
  return meta.chemistry;
}

/** Fill the Predict form with MeO- + MeCl and the floor, submit, wait for done. Returns the job id. */
async function runPredict(page: Page, beam: number, maxSteps: number): Promise<string> {
  await page.goto("/#/predict");
  await expect(page.locator("body[data-rdkit=ready]")).toBeAttached({ timeout: 90_000 });
  await page.getByTestId("smiles-input-0").fill("C[O-]");
  await page.getByTestId("add-reactant").click();
  await page.getByTestId("smiles-input-1").fill("CCl");
  await expect(page.getByTestId("input-depiction-0").locator("svg")).toBeVisible();
  await expect(page.getByTestId("input-depiction-1").locator("svg")).toBeVisible();
  await expect(page.locator("[data-testid=preflight-0][data-ok=true]")).toBeVisible();
  await expect(page.locator("[data-testid=preflight-1][data-ok=true]")).toBeVisible();
  await expect(page.locator("[data-testid=predictor-option-floor][data-available=true]")).toBeAttached();
  await page.getByTestId("predictor-select").selectOption("floor");
  await page.getByTestId("beam-input").fill(String(beam));
  await page.getByTestId("max-steps-input").fill(String(maxSteps));
  await page.getByTestId("predict-submit").click();
  await page.waitForURL(JOB_HASH, { timeout: 60_000 });
  const jobId = JOB_HASH.exec(page.url())?.[1] ?? "";
  expect(jobId).toMatch(/^j[0-9a-f]{16}$/);
  await expect(page.locator("[data-testid=job-status][data-status=done]")).toBeVisible({ timeout: 180_000 });
  return jobId;
}

/** The first row matching `selector`, clicking "show more" until it appears. */
async function findCandidate(page: Page, selector: string): Promise<Locator> {
  const rows = page.locator(selector);
  await expect(page.locator("[data-testid=candidate-row]").first()).toBeVisible({ timeout: 120_000 });
  for (let i = 0; i < MAX_CANDIDATE_PAGES && (await rows.count()) === 0; i++) {
    const more = page.getByTestId("candidate-more");
    if ((await more.count()) === 0) break;
    await more.click();
    await page.waitForTimeout(300);
  }
  await expect(rows.first()).toBeAttached();
  return rows.first();
}

function near(a: [number, number], b: [number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

async function num(loc: Locator, attr: string): Promise<number> {
  return Number(await loc.getAttribute(attr));
}

/** The O -> C overlay of `scope` lies on the O and C atoms of its depiction (within 20 px). */
async function checkArrowGeometry(scope: Locator, label: string): Promise<Locator> {
  const overlay = scope
    .locator("[data-testid=arrow-overlay][data-role=main][data-donor-element=O][data-acceptor-element=C]")
    .first();
  await expect(overlay).toBeAttached({ timeout: 60_000 });
  expect(await overlay.locator(".arrow-line").getAttribute("d")).toBeTruthy();
  const dep = scope.locator("[data-testid=mol-depiction][data-atom-xy]").first();
  await expect(dep).toHaveAttribute("data-render-state", "ready", { timeout: 60_000 });
  const xy = JSON.parse((await dep.getAttribute("data-atom-xy")) ?? "{}") as Record<string, [number, number]>;
  const dMap = (await overlay.getAttribute("data-donor-site-map")) ?? "";
  const aMap = (await overlay.getAttribute("data-acceptor-site-map")) ?? "";
  expect(xy[dMap], `${label}: donor map ${dMap} has a position`).toBeTruthy();
  expect(xy[aMap], `${label}: acceptor map ${aMap} has a position`).toBeTruthy();
  const start: [number, number] = [await num(overlay, "data-x1"), await num(overlay, "data-y1")];
  const end: [number, number] = [await num(overlay, "data-x2"), await num(overlay, "data-y2")];
  expect(near(start, xy[dMap])).toBeLessThanOrEqual(NEAR_PX);
  expect(near(end, xy[aMap])).toBeLessThanOrEqual(NEAR_PX);
  console.log(`[smoke] ${label}: coords method ${await dep.getAttribute("data-coords-method")}`);
  return overlay;
}

async function rootNodeId(page: Page): Promise<string> {
  const root = page.locator("[data-testid^=network-node-][data-root=true]").first();
  const testId = (await root.getAttribute("data-testid")) ?? "";
  return testId.replace("network-node-", "");
}

test("MeO- + MeCl with the floor", async ({ page, request }) => {
  test.setTimeout(240_000);
  await skipUnless(request, "predict", "candidates", "physics", "contact");
  await logChemistry(request);
  await runPredict(page, 3, 1);
  const view = page.getByTestId("network-view");
  await expect(view).toBeVisible();
  expect(await num(view, "data-node-count")).toBeGreaterThanOrEqual(2);
  await expect(page.getByTestId("pathway-item-1")).toBeVisible();

  await page.locator("[data-testid^=network-node-][data-root=true]").first().click();
  await page.getByTestId("node-candidates-toggle").click();
  const sn2 =
    '[data-testid=candidate-row][data-donor-species="C[O-]"][data-donor-site-element=O]' +
    "[data-acceptor-species=CCl][data-acceptor-site-element=C]" +
    '[data-acceptor-centre-elements="C,Cl"][data-valid=true]';
  const row = await findCandidate(page, sn2);
  await row.getByTestId("candidate-preview-btn").click();

  const preview = page.getByTestId("candidate-preview");
  const overlay = await checkArrowGeometry(preview, "candidate preview");
  await overlay.getByTestId("arrow-handle").click();
  const panel = page.getByTestId("physics-panel");
  await expect(panel).toBeVisible();
  await expect(panel).toHaveAttribute("data-loaded", "true", { timeout: 120_000 });
  const ks3 = Number(await panel.getByTestId("physics-ks3").getAttribute("data-value"));
  expect(Number.isFinite(ks3)).toBe(true);
  await expect(panel.getByTestId("floor-rank")).toHaveText(/^\d+ \/ \d+$/);
  await expect(panel.getByTestId("dominant-mo-pair")).toBeAttached();
  await expect(panel.getByTestId("energy-levels")).toBeAttached();

  const strip = page.locator(
    "[data-testid=strip-step-0] [data-testid=arrow-overlay][data-role=main][data-donor-element=O][data-acceptor-element=C]",
  );
  if ((await strip.count()) > 0) {
    await checkArrowGeometry(page.getByTestId("strip-step-0"), "pathway strip (optional)");
  } else {
    console.log("[smoke] pathway strip step 0 has no O -> C arrow (optional check skipped)");
  }

  await page.getByTestId("open-3d").click();
  const threed = page.locator("[data-testid=threed-panel][data-placement-loaded=true]");
  await expect(threed).toBeVisible({ timeout: 90_000 });
  await expect(threed).toHaveAttribute("data-state", /^(ready|webgl-unavailable)$/, { timeout: 90_000 });
  await expect(page.locator("[data-testid=contact-status][data-consistent=true]")).toBeVisible();
  console.log(`[smoke] 3D panel state ${await threed.getAttribute("data-state")}`);
});

/** C,H unless the original edge's acceptor already is a C-H antibond, then C,Cl (§13.3). */
async function overrideCentres(request: APIRequestContext, jobId: string, edgeId: string): Promise<string> {
  const job = await (await request.get(`/api/jobs/${jobId}`)).json();
  const edge = job.result.edges.find((e: { id: string }) => e.id === edgeId);
  const url = `/api/pool/${edge.source_state_id}/candidates?limit=10000&include_excluded=true`;
  const cands = (await (await request.get(url)).json()).candidates as CandidateLike[];
  const centres = new Map(cands.map((c) => [c.acceptor, c.acceptor_centre_elements.join(",")]));
  const original = edge.arrows.map((a: { acceptor: number }) => centres.get(a.acceptor));
  return original.includes("C,H") ? "C,Cl" : "C,H";
}

test("override continues", async ({ page, request }) => {
  test.setTimeout(300_000);
  await skipUnless(request, "predict", "correct", "candidates");
  const chemistry = await logChemistry(request);
  const jobId = await runPredict(page, 2, 1);
  const before = await num(page.getByTestId("network-view"), "data-node-count");
  const root = await rootNodeId(page);
  const first = page.locator(`[data-testid^=network-edge-][data-source="${root}"]`).first();
  const edgeId = ((await first.getAttribute("data-testid")) ?? "").replace("network-edge-", "");
  expect(edgeId).toMatch(/^e[0-9a-f]{16}$/);

  await page.goto(`/#/correct/${jobId}/edge/${edgeId}`);
  await expect(page.locator("[data-testid=correct-panel][data-target]")).toBeVisible();
  const centres = await overrideCentres(request, jobId, edgeId);
  const pick =
    "[data-testid=candidate-row][data-valid=true][data-donor-site-element=O]" +
    `[data-acceptor-species=CCl][data-acceptor-site-element=C][data-acceptor-centre-elements="${centres}"]`;
  const row = await findCandidate(page, pick);
  await row.getByTestId("candidate-select-btn").click();
  await expect(page.getByTestId("override-arrow-0")).toBeVisible();
  await page.getByTestId("override-continue").fill("1");
  await page.getByTestId("override-submit").click();
  await expect(page.getByTestId("override-result")).toBeVisible({ timeout: 60_000 });

  await page.waitForURL(/#\/predict\/job\/.*override=/, { timeout: 240_000 });
  await expect(page.locator("[data-testid=override-result][data-status=done]")).toBeVisible({ timeout: 60_000 });
  const corrected = page.locator("[data-testid^=network-edge-][data-kind=corrected]").first();
  await expect(corrected).toBeAttached({ timeout: 60_000 });
  await expect(page.locator(`[data-testid=network-edge-${edgeId}][data-status=superseded]`)).toBeAttached();

  const target = (await corrected.getAttribute("data-target")) ?? "";
  const node = page.getByTestId(`network-node-${target}`);
  const status = (await node.getAttribute("data-status")) ?? "";
  const reason = (await node.getAttribute("data-terminal-reason")) ?? "";
  console.log(`[smoke] corrected target ${target}: ${status} ${reason}`);
  const continued = status === "expanded" || status === "failed" || (status === "terminal" && reason !== "MAX_STEPS");
  expect(continued, `target status ${status} (${reason})`).toBe(true);
  if (chemistry === "synthetic") {
    expect(await num(page.getByTestId("network-view"), "data-node-count")).toBeGreaterThan(before);
  }
});

test("disabled pages show the capability notice", async ({ page, request }) => {
  const off: string[] = [];
  for (const name of ["predict", "correct"]) {
    if (!(await capability(request, name)).enabled) off.push(name);
  }
  test.skip(off.length === 0, "predict and correct are both enabled");
  await page.goto("/#/browse");
  for (const name of off) {
    const link = page.getByTestId(`nav-${name}`);
    await expect(link).toHaveAttribute("data-disabled", "true");
    await expect(link).toHaveAttribute("title", /disabled/);
    await page.goto(`/#/${name}`);
    const notice = page.locator(`[data-testid=capability-notice][data-capability=${name}]`);
    await expect(notice).toBeVisible();
    const reason = (await capability(request, name)).reason ?? "";
    await expect(notice.getByTestId("capability-reason")).toHaveText(reason);
  }
});
