// Playwright smoke test of ksmech-serve (design §13.3): `npm --prefix apps/ui run e2e`.
//
// Ordering facts this config is built around: Playwright starts webServer before globalSetup
// and empties outputDir (test-results/) at the start of a run. So there is no globalSetup: the
// build, the data-root reset (apps/ui/.e2e-data, outside outputDir) and the chemistry choice all
// happen inside the webServer command, e2e/serve.mjs, which then runs
// `ksmech-serve --port 8765` with one compute worker and never real xTB.
//
// The browser is the system Chrome or Chromium (e2e/browser.ts); without one, webServer is
// omitted, every test skips with the reason text of e2e/browser.ts, and the config prints that
// reason itself, because the list reporter never prints skip annotations.
import { defineConfig } from "@playwright/test";

import { browserPath, E2E_BASE_URL, NO_BROWSER_REASON } from "./e2e/browser";

/** Print why every test skips when there is no system browser; true when it printed. */
export function warnNoBrowser(path: string | undefined, warn: (message: string) => void = console.warn): boolean {
  if (path) return false;
  warn(`[e2e] skipping: ${NO_BROWSER_REASON}`);
  return true;
}

warnNoBrowser(browserPath);

export default defineConfig({
  testDir: "e2e",
  workers: 1,
  fullyParallel: false,
  retries: 0,
  outputDir: "test-results",
  reporter: [["list"]],
  timeout: 300_000,
  expect: { timeout: 30_000 },
  use: {
    baseURL: E2E_BASE_URL,
    browserName: "chromium",
    headless: true,
    viewport: { width: 1400, height: 1000 },
    launchOptions: browserPath
      ? { executablePath: browserPath, args: ["--disable-dev-shm-usage"] }
      : undefined,
    screenshot: "only-on-failure",
    trace: "off",
  },
  webServer: browserPath
    ? {
        command: "node e2e/serve.mjs",
        url: `${E2E_BASE_URL}/api/health`,
        reuseExistingServer: false,
        timeout: 300_000,
        stdout: "pipe",
        stderr: "pipe",
        gracefulShutdown: { signal: "SIGTERM", timeout: 15_000 },
      }
    : undefined,
});
