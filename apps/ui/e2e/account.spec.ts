// Accounts in the browser (serve/accounts), on the lab server of e2e/serve.mjs: without a session
// the header names the local admin with no job limits; the admin adds a user on #/users, signs
// in as it (the session cookie), sees its role and quota on the account page, and signs out
// back to the local admin. A wrong password shows LOGIN_FAILED.
import { expect, test } from "@playwright/test";

import { browserPath, NO_BROWSER_REASON } from "./browser";

test.skip(!browserPath, NO_BROWSER_REASON);

const LOGIN = `e2e-${Date.now().toString(36)}`;
const PASSWORD = "a long e2e password";

test("an admin adds a user, who signs in and out", async ({ page }) => {
  await page.goto("/#/workbench");
  const menu = page.getByTestId("user-menu");
  await expect(menu).toHaveAttribute("data-signed-in", "false");
  await expect(page.getByTestId("user-name")).toHaveText("local");
  await expect(page.getByTestId("user-role")).toHaveText("admin · no limits");

  await page.getByTestId("nav-users").click();
  await expect(page.getByTestId("users-page")).toBeVisible();
  await page.getByTestId("user-new-login").fill(LOGIN);
  await page.getByTestId("user-new-name").fill("E2E User");
  await page.getByTestId("user-new-role").selectOption("user");
  await page.getByTestId("user-new-password").fill(PASSWORD);
  await page.getByTestId("user-new-submit").click();
  await expect(page.getByTestId(`user-row-${LOGIN}`)).toBeVisible();
  await expect(page.getByTestId(`user-role-${LOGIN}`)).toHaveValue("user");

  await page.goto("/#/workbench");
  await page.getByTestId("sign-in").click();
  await expect(page.getByTestId("login-page")).toBeVisible();
  await page.getByTestId("login-login").fill(LOGIN);
  await page.getByTestId("login-password").fill("not the password");
  await page.getByTestId("login-submit").click();
  await expect(page.getByTestId("login-error")).toHaveAttribute("data-code", "LOGIN_FAILED");
  await page.getByTestId("login-password").fill(PASSWORD);
  await page.getByTestId("login-submit").click();

  await expect(page).toHaveURL(/#\/workbench$/);
  await expect(menu).toHaveAttribute("data-signed-in", "true");
  await expect(page.getByTestId("user-name")).toHaveText("E2E User");
  await expect(page.getByTestId("user-role")).toHaveText("user");
  await expect(page.getByTestId("nav-users")).toHaveCount(0);
  await page.getByTestId("user-name").click();
  await expect(page.getByTestId("account-role")).toHaveText("user");
  await expect(page.getByTestId("account-quota")).toContainText("of 100 jobs in the last 24 h");

  await page.getByTestId("sign-out").click();
  await expect(menu).toHaveAttribute("data-signed-in", "false");
  await expect(page.getByTestId("user-name")).toHaveText("local");
});
