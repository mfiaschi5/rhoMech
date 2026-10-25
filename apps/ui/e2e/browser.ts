/// <reference types="node" />
// The system browser of the e2e smoke test (design §13.3). This repo never downloads Playwright
// browsers (`npx playwright install` is never used): the first existing executable below is
// used through launchOptions.executablePath. With none, playwright.config.ts omits webServer
// (nothing is built or started) and every test skips with NO_BROWSER_REASON.
import { existsSync } from "node:fs";

export const BROWSER_CANDIDATES = [
  "/usr/bin/google-chrome",
  "/usr/bin/chromium-browser",
  "/usr/bin/chromium",
] as const;

export const NO_BROWSER_REASON =
  "no system Chrome/Chromium at /usr/bin/google-chrome, /usr/bin/chromium-browser or " +
  "/usr/bin/chromium; this repo never downloads Playwright browsers";

export const browserPath: string | undefined = BROWSER_CANDIDATES.find((p) => existsSync(p));

export const E2E_PORT = 8765;
export const E2E_BASE_URL = `http://127.0.0.1:${E2E_PORT}`;
