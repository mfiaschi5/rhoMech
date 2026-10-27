// Playwright test of the Curate page's export (spec v9), against the server e2e/serve.mjs
// starts: a mechanism is built through the API (a user network for MeO- + MeCl, the SN2 step as
// an override on the root's pool atoms, POST /curate validate), then #/curate's export downloads
// GET /curate/export/pmechdb (text/csv) and shows the X-Ksmech-Export-Report counts; unchecking a
// status warns that its mechanisms are left out.
// Skipped, with the server's reason, while correct, curate_edit or export is off.
import { readFileSync } from "node:fs";
import { expect, test, type APIRequestContext } from "@playwright/test";

import { browserPath, NO_BROWSER_REASON } from "./browser";
import { logChemistry, skipUnless } from "./helpers";

test.skip(!browserPath, NO_BROWSER_REASON);

interface DepictionLike {
  molblock: string;
  atoms: { depiction_idx: number; pool_atom: number; element: string }[];
}

/** The SN2 matrix on the pool atoms of a MeO- + MeCl depiction (bonds read from its molblock). */
function sn2Matrix(dep: DepictionLike): { bonds: [number, number, number][]; lone: [number, number][] } {
  const lines = dep.molblock.split("\n");
  const [nAtoms, nBonds] = [Number(lines[3].slice(0, 3)), Number(lines[3].slice(3, 6))];
  const pool = (pos: number) => dep.atoms.find((a) => a.depiction_idx === pos)!;
  const bonds = lines.slice(4 + nAtoms, 4 + nAtoms + nBonds).map((l) => [pool(Number(l.slice(0, 3)) - 1), pool(Number(l.slice(3, 6)) - 1)]);
  const O = dep.atoms.find((a) => a.element === "O")!.pool_atom;
  const Cl = dep.atoms.find((a) => a.element === "Cl")!.pool_atom;
  const ccl = bonds.find(([a, b]) => a.element === "Cl" || b.element === "Cl")!;
  const C = (ccl[0].element === "C" ? ccl[0] : ccl[1]).pool_atom;
  const p = (i: number, j: number, d: number): [number, number, number] => (i < j ? [i, j, d] : [j, i, d]);
  return { bonds: [p(O, C, 1), p(C, Cl, -1)], lone: [[O, -2], [Cl, 2]] };
}

async function buildMechanism(request: APIRequestContext): Promise<number> {
  const created = (await (await request.post("/api/predict", { data: { reactants_smiles: ["C[O-]", "CCl"], stoichiometry: [1, 1], predictor: "user", max_steps: 1 } })).json()) as { job_id: string };
  let root = "";
  let state = "";
  await expect
    .poll(async () => {
      const j = (await (await request.get(`/api/jobs/${created.job_id}`)).json()) as { status: string; result: { root: string; nodes: { id: string; state_id: string }[] } | null };
      root = j.result?.root ?? "";
      state = j.result?.nodes.find((n) => n.id === root)?.state_id ?? "";
      return j.status;
    }, { timeout: 120_000 })
    .toBe("done");
  const dep = (await (await request.get(`/api/pool/${state}/depiction`)).json()) as DepictionLike;
  const res = await request.post(`/api/jobs/${created.job_id}/override`, { data: { parent_state_id: state, matrix: sn2Matrix(dep) } });
  expect(res.status(), await res.text()).toBe(200);
  const { edge } = (await res.json()) as { edge: { id: string } };
  const m = await request.post("/api/curate", { data: { job_id: created.job_id, edge_ids: [edge.id], action: "validate", notes: "e2e export" } });
  expect(m.status(), await m.text()).toBe(201);
  return ((await m.json()) as { id: number }).id;
}

test("export: the curated steps download as PMechDB CSV with the report's counts", async ({ page, request }) => {
  test.setTimeout(300_000);
  await skipUnless(request, "correct", "curate_edit", "export");
  await logChemistry(request, "curate");
  const mid = await buildMechanism(request);
  console.log(`[curate] mechanism ${mid}`);
  await page.goto("/#/curate");
  const panel = page.getByTestId("export-panel");
  await expect(panel).toBeVisible();
  await page.getByTestId("export-status-corrected").uncheck();
  await expect(page.getByTestId("export-withdraw-warning")).toHaveAttribute("data-unchecked", "corrected");
  await page.getByTestId("export-status-corrected").check();
  await expect(page.getByTestId("export-withdraw-warning")).toHaveCount(0);
  await expect(page.getByTestId("export-link")).toHaveAttribute("href", /\/api\/curate\/export\/pmechdb\?status=validated&status=corrected/);
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByTestId("export-button").click()]);
  expect(download.suggestedFilename()).toBe("ksmech_curated_pmechdb.csv");
  const csv = readFileSync((await download.path())!, "utf8");
  const report = page.getByTestId("export-report");
  await expect(report).toBeVisible();
  const written = Number(await report.getAttribute("data-written"));
  const considered = Number(await report.getAttribute("data-considered"));
  console.log(`[curate] export: ${written} written of ${considered}, ${csv.split("\n").filter(Boolean).length} CSV lines`);
  expect(considered).toBeGreaterThanOrEqual(1);
  expect(csv.split("\n").filter(Boolean).length).toBe(written);
});
