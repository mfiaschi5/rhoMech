// Playwright test of the embedding trace (features/embedding), against the server e2e/serve.mjs
// starts. A tiny random checkpoint (tests/serve_fake_model.py) is written under the server's
// data root first, which turns the embedding capability on; the pool is MeO- + MeCl.
//   Predict: a floor job, one of its steps, the step panel's Embedding variant: the checkpoint
//   picker, the trace job, the page in an iframe (its tokens drawn), its view hash kept across a
//   reload, "open in new tab" with that hash. Workbench: "Model trace" greyed until a molecule
//   is computed.
// Skipped, with the reason, when the predict capability is off or the pool gives no trace
// sample (the job's error code says why).
import { execFileSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type APIRequestContext } from "@playwright/test";

import { browserPath, NO_BROWSER_REASON } from "./browser";
import { logChemistry, skipUnless } from "./helpers";

test.skip(!browserPath, NO_BROWSER_REASON);

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const SN2 = ["C[O-]", "CCl"];

interface JobLike {
  job_id: string;
  status: string;
  error: { code: string; message: string } | null;
  result: Record<string, unknown> | null;
}

async function waitJob(request: APIRequestContext, jobId: string, timeoutMs = 240_000): Promise<JobLike> {
  const end = Date.now() + timeoutMs;
  for (;;) {
    const job = (await (await request.get(`/api/jobs/${jobId}`)).json()) as JobLike;
    if (["done", "failed", "cancelled"].includes(job.status)) return job;
    if (Date.now() > end) throw new Error(`job ${jobId} still ${job.status}`);
    await new Promise((r) => setTimeout(r, 500));
  }
}

/** The fake run `e2e/tiny` under the server's data root (GET /api/meta.data_root). */
async function writeFakeCheckpoint(request: APIRequestContext): Promise<void> {
  const meta = (await (await request.get("/api/meta")).json()) as { data_root: string };
  const runDir = join(meta.data_root, "data", "models", "e2e", "tiny");
  const out = execFileSync("uv", ["run", "--no-sync", "--project", REPO, "python", "-m", "tests.serve_fake_model", runDir], { cwd: REPO, encoding: "utf8", timeout: 240_000 });
  console.log(`[embedding] ${out.trim().replace(/\n/g, "; ")}`);
}

test("a step's parent pool through a checkpoint: the trace page in the step panel", async ({ page, request }) => {
  test.setTimeout(300_000);
  await skipUnless(request, "predict");
  await logChemistry(request, "embedding");
  await writeFakeCheckpoint(request);
  const models = (await (await request.get("/api/models")).json()) as { embedding: { enabled: boolean; reason: string | null } };
  test.skip(!models.embedding.enabled, `embedding is off: ${models.embedding.reason}`);

  // the trace of the pool through the API first: skip cleanly if it gives no sample
  const created = await (await request.post("/api/embedding", { data: { smiles: SN2.join("."), run: "tiny" } })).json();
  const probe = await waitJob(request, created.job_id as string);
  test.skip(probe.status !== "done", `no trace sample for ${SN2.join(".")}: ${probe.error?.code} ${probe.error?.message}`);

  const predicted = await (await request.post("/api/predict", { data: { reactants_smiles: SN2, predictor: "floor", beam: 2, max_steps: 1 } })).json();
  const job = await waitJob(request, predicted.job_id as string);
  expect(job.status, JSON.stringify(job.error)).toBe("done");
  const edges = (job.result?.edges ?? []) as { id: string; source_state_id: string }[];
  test.skip(!edges.length, "the floor proposed no step for MeO- + MeCl");
  const edge = edges[0];

  await page.goto(`/#/predict/job/${job.job_id}?edge=${edge.id}`);
  const panel = page.getByTestId("predict-step-panel");
  await expect(panel).toHaveAttribute("data-edge", edge.id, { timeout: 60_000 });
  const tab = page.getByTestId("predict-step-variant-embedding");
  await expect(tab).toBeEnabled();
  await tab.click();
  const view = page.getByTestId("predict-step-embedding");
  await expect(view).toHaveAttribute("data-status", "ready", { timeout: 240_000 });
  await expect(page.getByTestId("predict-step-embedding-checkpoint")).toHaveValue("e2e/tiny/best");
  await expect(page.getByTestId("predict-step-embedding-facts")).toContainText("dense encoder");
  const eid = (await view.getAttribute("data-embedding-id")) ?? "";
  expect(eid).toMatch(/^b[0-9a-f]{24}$/);

  // the page in the iframe draws the pool's tokens; its view goes into its hash
  const frame = page.frameLocator("[data-testid=predict-step-embedding-frame]");
  await expect(frame.locator("#map")).toBeVisible({ timeout: 30_000 });
  await frame.locator("#layer").evaluate((el: HTMLInputElement) => {
    el.value = "1";
    el.dispatchEvent(new Event("input"));
  });
  await expect(page.getByTestId("predict-step-embedding-frame")).toHaveAttribute("data-hash", /layer=1/, { timeout: 10_000 });
  const open = page.getByTestId("predict-step-embedding-open");
  await expect(open).toHaveAttribute("href", new RegExp(`^/api/embedding/${eid}/page(\\?theme=(light|dark))?#.*layer=1`));
  await expect(open).toHaveAttribute("target", "_blank");

  // a reload answers the stored trace and reopens the page at the kept view
  await page.reload();
  await page.getByTestId("predict-step-variant-embedding").click();
  await expect(view).toHaveAttribute("data-status", "ready", { timeout: 60_000 });
  await expect(page.getByTestId("predict-step-embedding-facts")).toHaveAttribute("data-cached", "1");
  await expect(page.getByTestId("predict-step-embedding-frame")).toHaveAttribute("src", /#.*layer=1/);
  await expect(frame.locator("#layer")).toHaveValue("1");
  const html = await (await request.get(`/api/embedding/${eid}/page`)).text();
  expect(html).not.toContain("/*PAYLOAD*/");

  // the Workbench's entry point is greyed until a molecule is computed
  await page.goto("/#/workbench");
  const wb = page.getByTestId("wb-embedding-open");
  await expect(wb).toBeDisabled();
  await expect(page.getByTestId("wb-embedding-wrap")).toHaveAttribute("data-reason", /compute the molecule first/);
});
