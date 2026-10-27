// Playwright test of the Wiki section (#/wiki, #/wiki/<slug>) with GET /api/wiki and
// /api/wiki/{slug} mocked by page.route (the pages are the backend's; the UI's part is the list,
// the search, the article, its contents, its source view and its scene figures): the index
// lists two pages, the search narrows it (q in the hash), a page shows its title, tags and table
// of contents, its <figure class="wiki-scene" data-scene-src="scenes/sn2-step.json"> is hydrated
// with the scene viewer (the committed example scene of public/scenes, no API call), an in-text
// link #/wiki/<slug> opens the other page in the app, "View Markdown source" shows the Markdown,
// and an unknown slug shows the 404 code.
import { expect, test, type Page } from "@playwright/test";

import { browserPath, NO_BROWSER_REASON } from "./browser";

test.skip(!browserPath, NO_BROWSER_REASON);

const PAGES = [
  { slug: "sn2", title: "The SN2 step", summary: "A lone pair into a σ* orbital.", tags: ["mechanism", "substitution"] },
  { slug: "orbitals", title: "Localized orbitals", summary: "Pipek-Mezey orbitals as tokens.", tags: ["orbitals"] },
];

const SN2_HTML = [
  '<h2 id="the-step">The step</h2>',
  '<p>Methoxide attacks chloromethane; see <a href="#/wiki/orbitals">localized orbitals</a> and <a href="#why">why</a>.</p>',
  '<figure class="wiki-scene" data-scene-src="scenes/sn2-step.json"><figcaption>The SN2 step</figcaption></figure>',
  '<h3 id="why">Why it works</h3>',
  "<p>The gap between n(O) and σ*(C–Cl) is small.</p>",
].join("\n");

function pageOut(slug: string) {
  const p = PAGES.find((x) => x.slug === slug);
  if (!p) return null;
  const sn2 = slug === "sn2";
  return {
    ...p,
    html: sn2 ? SN2_HTML : '<h2 id="tokens">Tokens</h2><p>Orbital tokens.</p>',
    markdown: sn2 ? "## The step\n\nMethoxide attacks chloromethane.\n" : "## Tokens\n",
    scenes: sn2 ? ["scenes/sn2-step.json"] : [],
    toc: sn2
      ? [
          { level: 2, id: "the-step", text: "The step" },
          { level: 3, id: "why", text: "Why it works" },
        ]
      : [{ level: 2, id: "tokens", text: "Tokens" }],
    updated: "2026-10-27",
  };
}

async function mockWiki(page: Page): Promise<void> {
  await page.route(
    (url) => url.pathname === "/api/wiki" || url.pathname.startsWith("/api/wiki/"),
    async (route) => {
      const url = new URL(route.request().url());
      if (url.pathname === "/api/wiki") {
        const q = (url.searchParams.get("q") ?? "").toLowerCase();
        return route.fulfill({ json: { pages: PAGES.filter((p) => !q || `${p.title} ${p.summary} ${p.tags.join(" ")}`.toLowerCase().includes(q)) } });
      }
      const slug = decodeURIComponent(url.pathname.slice("/api/wiki/".length));
      const out = pageOut(slug);
      if (out) return route.fulfill({ json: out });
      return route.fulfill({ status: 404, json: { error: { code: "WIKI_PAGE_NOT_FOUND", message: `no wiki page "${slug}"`, upstream: null, detail: null, request_id: "e2e" } } });
    },
  );
}

test("wiki: list, search, article with contents, a hydrated scene, in-app links and the source", async ({ page }) => {
  await mockWiki(page);
  await page.goto("/#/wiki");
  await expect(page.getByTestId("nav-wiki")).toHaveAttribute("aria-current", "page");
  await expect(page.getByTestId("wiki-list")).toHaveAttribute("data-count", "2");
  await expect(page.getByTestId("wiki-index")).toContainText("The SN2 step");
  await page.getByTestId("wiki-search").fill("pipek");
  await expect(page.getByTestId("wiki-list")).toHaveAttribute("data-count", "1");
  await expect(page).toHaveURL(/#\/wiki\?q=pipek$/);
  await page.getByTestId("wiki-search").fill("");
  await expect(page.getByTestId("wiki-list")).toHaveAttribute("data-count", "2");
  await page.getByTestId("wiki-link-sn2").click();
  const article = page.getByTestId("wiki-article");
  await expect(article).toHaveAttribute("data-state", "ready");
  await expect(page.getByTestId("wiki-title")).toHaveText("The SN2 step");
  await expect(page.getByTestId("wiki-tags")).toContainText("substitution");
  await expect(page.getByTestId("wiki-toc").locator("button")).toHaveText(["The step", "Why it works"]);
  const scene = page.locator("figure.wiki-scene").getByTestId("wiki-scene-view");
  await expect(scene).toHaveAttribute("data-state", "ready");
  await expect(scene).toHaveAttribute("data-src", "scenes/sn2-step.json");
  await expect(scene.getByTestId("scene-view")).toHaveAttribute("data-type", "step");
  await expect(page.locator("figure.wiki-scene figcaption")).toHaveText("The SN2 step");
  // a heading link scrolls; the route stays
  await page.getByTestId("wiki-html").getByRole("link", { name: "why" }).click();
  await expect(page).toHaveURL(/#\/wiki\/sn2$/);
  await page.getByTestId("wiki-toc-why").click();
  await expect(page).toHaveURL(/#\/wiki\/sn2$/);
  // the source view
  await page.getByTestId("wiki-source-link").click();
  await expect(page.getByTestId("wiki-markdown")).toContainText("Methoxide attacks chloromethane.");
  await page.getByTestId("wiki-source-link").click();
  await expect(page.getByTestId("wiki-html")).toBeVisible();
  // an in-text wiki link opens the other page in the app
  await page.getByTestId("wiki-html").getByRole("link", { name: "localized orbitals" }).click();
  await expect(page).toHaveURL(/#\/wiki\/orbitals$/);
  await expect(page.getByTestId("wiki-title")).toHaveText("Localized orbitals");
  await expect(page.getByTestId("wiki-link-orbitals")).toHaveAttribute("aria-current", "page");
});

test("wiki: an unknown page shows WIKI_PAGE_NOT_FOUND", async ({ page }) => {
  await mockWiki(page);
  await page.goto("/#/wiki/nope");
  await expect(page.getByTestId("wiki-article")).toHaveAttribute("data-state", "error");
  await expect(page.getByTestId("error-banner")).toHaveAttribute("data-code", "WIKI_PAGE_NOT_FOUND");
});
