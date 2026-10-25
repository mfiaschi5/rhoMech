// Playwright test of the spec-v9 Correct page (#/correct), against the server e2e/serve.mjs
// starts:
//   From bare reactants (MeO- + MeCl, no prediction): the arrow tool compiles two drags into
//   the SN2 matrix (lone pair O -> C: R_OO -2, R_OC +1; bond C-Cl -> Cl: R_CCl -1, R_ClCl +2),
//   the live /step/check gives a valid step, the product COC.[Cl-] and the formal charges;
//   undo / redo by keys; the lone-pair tool's mask (no -2 on a carbon without a lone pair) and
//   the issue it raises, whose hover halos its atoms; the bond tool's cycle; the eraser on an
//   arrow; Delete on a selected entry; Enter applies and opens the product state with a
//   one-step breadcrumb. On the product state (a new atom order: pool atoms from the depiction's
//   pool_atom), an entry on a hidden H draws that H; the reverse SN2 (Cl- -> C, C-O -> O) is a
//   valid step, Apply gives a two-step breadcrumb, and Save to curation stores both steps.
//   From a prediction: #/correct/<job>/edge/<edge> preloads the edge's R; Apply adds a
//   corrected step.
// Skipped, with the server's reason, while the correct capability is off.
import { expect, test, type Page } from "@playwright/test";

import { browserPath, NO_BROWSER_REASON } from "./browser";
import { drag, entryCount, logChemistry, pair, readCanvas, skipUnless, waitRdkit, type CanvasInfo } from "./helpers";

test.skip(!browserPath, NO_BROWSER_REASON);

interface Sn2Atoms {
  O: number;
  C: number;
  Cl: number;
}

function sn2Atoms(c: CanvasInfo): Sn2Atoms {
  const O = c.atomsOf("O")[0];
  const Cl = c.atomsOf("Cl")[0];
  const C = c.neighbours(Cl).find((i) => c.elements[i] === "C");
  if (O === undefined || Cl === undefined || C === undefined) throw new Error(`no O / Cl / C-Cl in ${JSON.stringify(c.elements)}`);
  return { O, C, Cl };
}

async function drawSn2(page: Page, c: CanvasInfo, a: Sn2Atoms): Promise<void> {
  await page.keyboard.press("a");
  await expect(page.getByTestId("correct-tool-arrow")).toHaveAttribute("aria-pressed", "true");
  await drag(page, c.at(a.O), c.at(a.C));
  await expect(page.getByTestId(`correct-canvas-r-bond-${pair(a.O, a.C)}`)).toHaveAttribute("data-delta", "1");
  await expect(page.getByTestId(`correct-canvas-r-lone-${a.O}`)).toHaveAttribute("data-delta", "-2");
  await expect(page.getByTestId("correct-canvas-arrow-0")).toHaveAttribute("data-kind", "lone_pair_to_bond");
  await drag(page, c.along(a.C, a.Cl, 0.5), c.at(a.Cl));
  await expect(page.getByTestId(`correct-canvas-r-bond-${pair(a.C, a.Cl)}`)).toHaveAttribute("data-delta", "-1");
  await expect(page.getByTestId(`correct-canvas-r-lone-${a.Cl}`)).toHaveAttribute("data-delta", "2");
  await expect(page.getByTestId("correct-canvas-arrow-1")).toHaveAttribute("data-kind", "bond_to_lone_pair");
  expect(await entryCount(page)).toBe(4);
}

async function expectValidSn2(page: Page, a: Sn2Atoms): Promise<void> {
  await expect(page.getByTestId("correct-check")).toHaveAttribute("data-state", "ok", { timeout: 60_000 });
  const product = page.getByTestId("correct-preview-product");
  await expect(product).toHaveAttribute("data-smiles", /\[Cl-\]/);
  await expect(product).toHaveAttribute("data-smiles", /COC/);
  await expect(page.getByTestId(`correct-charge-${a.Cl}`)).toHaveAttribute("data-charge", "-1");
  await expect(page.getByTestId(`correct-charge-${a.O}`)).toHaveAttribute("data-charge", "0");
  await expect(page.getByTestId("correct-reading-arrow-0")).toBeVisible();
}

/** Masks, issues, the bond cycle, the eraser and Delete; every change undone again. */
async function editAndUndo(page: Page, c: CanvasInfo, a: Sn2Atoms): Promise<void> {
  await page.keyboard.press("Control+z");
  await expect.poll(() => entryCount(page)).toBe(2);
  await page.keyboard.press("Control+y");
  await expect.poll(() => entryCount(page)).toBe(4);

  await page.keyboard.press("l");
  await page.mouse.click(...c.at(a.C));
  await expect(page.getByTestId(`correct-entry-l-${a.C}`)).toHaveAttribute("data-delta", "2");
  await expect(page.getByTestId("correct-notice")).toContainText("no lone pair");
  const issue = page.locator("[data-testid^=correct-issue-][data-code=CHARGE_NOT_CONSERVED]");
  await expect(issue).toBeVisible();
  await issue.hover();
  await expect(page.getByTestId(`correct-canvas-halo-${a.C}`)).toBeVisible();
  await page.keyboard.press("Control+z");

  await page.keyboard.press("b");
  await page.mouse.click(...c.along(a.C, a.Cl, 0.5));
  await expect(page.getByTestId(`correct-entry-b-${pair(a.C, a.Cl)}`)).toHaveAttribute("data-delta", "1");
  await page.keyboard.press("Control+z");

  await page.keyboard.press("e");
  const arrow = page.getByTestId("correct-canvas-arrow-1");
  const box = await page.getByTestId("correct-canvas").boundingBox();
  await page.mouse.click(box!.x + Number(await arrow.getAttribute("data-mx")), box!.y + Number(await arrow.getAttribute("data-my")));
  await expect.poll(() => entryCount(page)).toBe(2);
  await page.keyboard.press("Control+z");

  await page.getByTestId(`correct-entry-select-l-${a.O}`).check();
  await page.keyboard.press("Delete");
  await expect.poll(() => entryCount(page)).toBe(3);
  await page.keyboard.press("Control+z");
  await expect.poll(() => entryCount(page)).toBe(4);
}

/** An entry on a hidden H (added through the table) draws that H on the canvas; undone. */
async function hiddenHydrogen(page: Page): Promise<void> {
  const hidden = ((await page.getByTestId("correct-canvas").getAttribute("data-hidden-h")) ?? "").split(",").filter(Boolean);
  expect(hidden.length, "the product depiction hides its H").toBeGreaterThan(0);
  const h = hidden[0];
  await expect(page.getByTestId(`correct-canvas-atom-${h}`)).toHaveCount(0);
  await page.getByTestId("correct-add-kind").selectOption("lone");
  await page.getByTestId("correct-add-i").fill(h);
  await page.getByTestId("correct-add-value").selectOption("2");
  await page.getByTestId("correct-add").click();
  await expect(page.getByTestId(`correct-canvas-r-lone-${h}`)).toBeVisible();
  await expect(page.getByTestId(`correct-canvas-atom-${h}`)).toHaveCount(1);
  await page.keyboard.press("Control+z");
  await expect.poll(() => entryCount(page)).toBe(0);
  await expect(page.getByTestId(`correct-canvas-atom-${h}`)).toHaveCount(0);
}

/** Cl- back onto a methyl of the product: lone pair Cl -> C, bond C-O -> O; applied. */
async function reverseSn2(page: Page): Promise<void> {
  const c = await readCanvas(page, "correct-canvas");
  const Cl = c.atomsOf("Cl")[0];
  const O = c.atomsOf("O")[0];
  const C = c.neighbours(O).find((i) => c.elements[i] === "C");
  if (C === undefined) throw new Error(`no C on O in ${JSON.stringify(c.elements)}`);
  await page.keyboard.press("a");
  await drag(page, c.at(Cl), c.at(C));
  await drag(page, c.along(C, O, 0.5), c.at(O));
  await expect(page.getByTestId(`correct-canvas-r-bond-${pair(Cl, C)}`)).toHaveAttribute("data-delta", "1");
  await expect(page.getByTestId(`correct-canvas-r-bond-${pair(C, O)}`)).toHaveAttribute("data-delta", "-1");
  await expect(page.getByTestId("correct-check")).toHaveAttribute("data-state", "ok", { timeout: 60_000 });
  const product = page.getByTestId("correct-preview-product");
  await expect(product).toHaveAttribute("data-smiles", /Cl/);
  await expect(product).toHaveAttribute("data-smiles", /\[O-\]/);
  await expect(page.getByTestId("correct-readings-status")).toHaveAttribute("data-status", /^(ready|pending|unavailable)$/);
  await page.getByTestId("correct-apply").click();
  await expect(page.getByTestId("correct-breadcrumb")).toHaveAttribute("data-steps", "2", { timeout: 60_000 });
}

test("from bare reactants: arrows to R, live check, masks, undo, apply, breadcrumb, curation", async ({ page, request }) => {
  test.setTimeout(300_000);
  await skipUnless(request, "correct");
  await logChemistry(request, "correct");
  await page.goto("/#/correct?r=C%5BO-%5D&n=1&r=CCl&n=1");
  await waitRdkit(page);
  await expect(page.getByTestId("correct-smiles-0")).toHaveValue("C[O-]");
  await page.getByTestId("correct-start-submit").click();
  await expect(page.getByTestId("correct-editor")).toBeVisible({ timeout: 120_000 });
  const c = await readCanvas(page, "correct-canvas");
  const a = sn2Atoms(c);
  await drawSn2(page, c, a);
  await expectValidSn2(page, a);
  await editAndUndo(page, c, a);
  await expectValidSn2(page, a);

  await page.keyboard.press("Enter");
  await page.waitForURL(/#\/correct\/[^/]+\/node\/[^/?]+\?path=/, { timeout: 60_000 });
  await expect(page.getByTestId("correct-breadcrumb")).toHaveAttribute("data-steps", "1");
  const after = await readCanvas(page, "correct-canvas");
  expect(Object.values(after.charges)).toContain(-1);
  expect(await entryCount(page)).toBe(0);
  await hiddenHydrogen(page);
  await reverseSn2(page);

  await page.getByTestId("correct-save").click();
  await page.waitForURL(/#\/curate\/m\/\d+/, { timeout: 60_000 });
  const mid = /#\/curate\/m\/(\d+)/.exec(page.url())?.[1];
  const m = (await (await request.get(`/api/curate/${mid}`)).json()) as { n_steps: number };
  expect(m.n_steps).toBe(2);
});

test("from a prediction: the edge's R is preloaded and Apply adds a corrected step", async ({ page, request }) => {
  test.setTimeout(300_000);
  await skipUnless(request, "predict", "correct");
  const created = (await (await request.post("/api/predict", { data: { reactants_smiles: ["C[O-]", "CCl"], stoichiometry: [1, 1], beam: 2, max_steps: 1 } })).json()) as { job_id: string };
  let edges: { id: string; matrix: { bonds: unknown[]; lone: unknown[] } }[] = [];
  await expect
    .poll(
      async () => {
        const j = (await (await request.get(`/api/jobs/${created.job_id}`)).json()) as { status: string; result: { edges?: typeof edges } | null };
        edges = j.result?.edges ?? [];
        return j.status;
      },
      { timeout: 240_000, intervals: [1000] },
    )
    .toBe("done");
  test.skip(edges.length === 0, "the predictor proposed no step for MeO- + MeCl");
  const edge = edges[0];
  await page.goto(`/#/correct/${created.job_id}/edge/${edge.id}`);
  await waitRdkit(page);
  await expect(page.getByTestId("correct-breadcrumb")).toContainText("editing step");
  await expect.poll(() => entryCount(page), { timeout: 60_000 }).toBe(edge.matrix.bonds.length + edge.matrix.lone.length);
  await expect(page.getByTestId("correct-check")).toHaveAttribute("data-state", /^(ok|issues)$/, { timeout: 60_000 });
  const state = await page.getByTestId("correct-check").getAttribute("data-state");
  console.log(`[correct] preloaded edge ${edge.id}: check ${state}`);
  if (state !== "ok") return;
  await page.getByTestId("correct-apply").click();
  await page.waitForURL(/#\/correct\/[^/]+\/node\//, { timeout: 60_000 });
  await expect(page.getByTestId("correct-breadcrumb")).toHaveAttribute("data-steps", "1");
});
