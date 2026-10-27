// @ts-nocheck -- run by `npm test` (node --experimental-strip-types --test).
// ChemRho's sections: the nav groups (registry.NAV_SECTIONS), the section of each page, and the
// hash aliases of the section names (router.PAGE_ALIASES) with the wiki routes.
import test from "node:test";
import assert from "node:assert/strict";
import { NAV_PAGE_IDS, NAV_SECTIONS, PRODUCT_NAME, sectionOf } from "./registry.ts";
import { formatRoute, parseHash, wikiHash } from "./router.ts";

test("four sections in order; ClotoMech groups predict, correct and curate", () => {
  assert.equal(PRODUCT_NAME, "ChemRho");
  assert.deepEqual(NAV_SECTIONS.map((s) => s.label), ["ChemPtah", "ClotoMech", "ChemThot", "Wiki"]);
  assert.deepEqual(NAV_SECTIONS.map((s) => s.id), ["workbench", "clotomech", "browse", "wiki"]);
  assert.deepEqual(sectionOf("correct").pages, ["predict", "correct", "curate"]);
  assert.equal(sectionOf("workbench").label, "ChemPtah");
  assert.equal(sectionOf("login"), undefined);
  assert.deepEqual(NAV_SECTIONS.flatMap((s) => s.pages).sort(), [...NAV_PAGE_IDS].sort(), "every nav page in one section");
});

test("section names are hash aliases of the page ids, with parts and query kept", () => {
  assert.equal(parseHash("#/chemptah").page, "workbench");
  assert.equal(parseHash("#/ClotoMech").page, "predict");
  const r = parseHash("#/chemthot/step/pmechdb:12?q=CCO");
  assert.deepEqual([r.page, r.parts, r.params.get("q")], ["browse", ["step", "pmechdb:12"], "CCO"]);
  assert.equal(parseHash("#/predict").page, "predict", "the old routes are unchanged");
  assert.equal(parseHash("").page, "workbench");
});

test("wiki routes: the index, a page by slug, the search", () => {
  assert.equal(wikiHash(), "#/wiki");
  assert.equal(wikiHash("sn2-reaction"), "#/wiki/sn2-reaction");
  const r = parseHash(formatRoute({ page: "wiki", parts: ["a b"], params: { q: "pi*" } }));
  assert.deepEqual([r.page, r.parts, r.params.get("q")], ["wiki", ["a b"], "pi*"]);
});
