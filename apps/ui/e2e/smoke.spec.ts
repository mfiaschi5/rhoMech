// Playwright smoke test of ksmech-serve (design §13.3), against the server e2e/serve.mjs starts:
// ChemRho's nav (the brand, the four sections in order, ClotoMech's tab row of predict, correct
// and curate, the section-name hash aliases), and, while /api/meta reports predict or correct
// off, the ClotoMech tab row greys those pages and they show the capability notice with the
// reason. The spec-v9 Predict and Correct flows are in predict.spec.ts and correct.spec.ts.
import { expect, test, type APIRequestContext } from "@playwright/test";

import { browserPath, NO_BROWSER_REASON } from "./browser";

test.skip(!browserPath, NO_BROWSER_REASON);

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

test("ChemRho nav: sections, the ClotoMech tab row and the hash aliases", async ({ page }) => {
  await page.goto("/#/workbench");
  await expect(page.getByTestId("brand")).toHaveText("ChemRho");
  await expect(page).toHaveTitle("ChemRho · ChemPtah");
  const items = page.locator("nav.nav .nav-link");
  await expect(items).toHaveText(["ChemPtah", "ClotoMech", "ChemThot", "Wiki"]);
  await expect(page.getByTestId("nav-workbench")).toHaveAttribute("aria-current", "page");
  await expect(page.getByTestId("subnav-clotomech")).toHaveCount(0);
  await page.getByTestId("nav-clotomech").click();
  await expect(page).toHaveURL(/#\/predict$/);
  await expect(page.getByTestId("nav-clotomech")).toHaveAttribute("aria-current", "page");
  await expect(page.getByTestId("subnav-clotomech").locator("a")).toHaveText(["Predict", "Correct", "Curate"]);
  await expect(page.getByTestId("subnav-predict")).toHaveAttribute("aria-current", "page");
  await page.getByTestId("subnav-curate").click();
  await expect(page).toHaveURL(/#\/curate$/);
  await expect(page.getByTestId("nav-clotomech")).toHaveAttribute("aria-current", "page");
  await expect(page.getByTestId("subnav-curate")).toHaveAttribute("aria-current", "page");
  await expect(page).toHaveTitle("ChemRho · ClotoMech · Curate");
  await page.goto("/#/chemthot");
  await expect(page.getByTestId("browse-page")).toBeVisible();
  await expect(page.getByTestId("nav-browse")).toHaveAttribute("aria-current", "page");
  await page.goto("/#/chemptah");
  await expect(page.getByTestId("nav-workbench")).toHaveAttribute("aria-current", "page");
  await page.goto("/#/clotomech");
  await expect(page.getByTestId("subnav-predict")).toHaveAttribute("aria-current", "page");
  await page.getByTestId("nav-wiki").click();
  await expect(page.getByTestId("wiki-page")).toBeVisible();
});

test("disabled pages show the capability notice", async ({ page, request }) => {
  const off: string[] = [];
  for (const name of ["predict", "correct"]) {
    if (!(await capability(request, name)).enabled) off.push(name);
  }
  test.skip(off.length === 0, "predict and correct are both enabled");
  await page.goto("/#/curate");
  for (const name of off) {
    const link = page.getByTestId(`subnav-${name}`);
    await expect(link).toHaveAttribute("data-disabled", "true");
    await expect(link).toHaveAttribute("title", /disabled/);
    await page.goto(`/#/${name}`);
    const notice = page.locator(`[data-testid=capability-notice][data-capability=${name}]`);
    await expect(notice).toBeVisible();
    const reason = (await capability(request, name)).reason ?? "";
    await expect(notice.getByTestId("capability-reason")).toHaveText(reason);
  }
});
