// Playwright smoke test of ksmech-serve (design §13.3), against the server e2e/serve.mjs starts:
// while /api/meta reports predict or correct off, the nav greys those pages and they show the
// capability notice with the reason. The spec-v9 Predict and Correct flows are in predict.spec.ts
// and correct.spec.ts.
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
