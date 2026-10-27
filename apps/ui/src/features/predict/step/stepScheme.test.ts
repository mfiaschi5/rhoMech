// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// The textbook scheme on a real pool (MeO- + MeCl -> Me2O + Cl-): the row layout, lone pairs on
// the touched atoms, routed arrows without crossings, the publication SVG.
import test from "node:test";
import assert from "node:assert/strict";
import { poolStructure } from "../poolStructure.ts";
import { alignProduct, productIndexMap, sideFragments, sideOf, xyOf } from "./stepGeometry.ts";
import { PRODUCT_ID, drawScheme, schemeGeometry } from "./stepScheme.ts";
import { fitScheme } from "./stepScene.ts";
import { arrowCrossings } from "./stepRoute.ts";
import { schemeSvg } from "./stepExport.ts";
import { ACS_1996_STYLE } from "../../workbench/sketch/style.ts";

const mol = (atoms, bonds, chg) => `\n     RDKit          2D\n\n${String(atoms.length).padStart(3)}${String(bonds.length).padStart(3)}  0  0  0  0  0  0  0  0999 V2000\n${atoms.map(([x, y, el]) => `${x.toFixed(4).padStart(10)}${y.toFixed(4).padStart(10)}    0.0000 ${el.padEnd(3)} 0  0  0  0  0  0  0  0  0  0  0  0`).join("\n")}\n${bonds.map(([a, b]) => `${String(a).padStart(3)}${String(b).padStart(3)}  1  0`).join("\n")}\n${chg}M  END\n`;
const A = (depiction_idx, pool_atom, element, hidden, map) => ({ depiction_idx, pool_atom, hidden_h_atoms: hidden, global_site: pool_atom, map, element, fragment: 0, hidden_h: hidden.length });
const SRC_DEP = { ref: "s", molblock: mol([[-0.5, -0.5, "C"], [0.5, -0.5, "O"], [-0.5, 0.5, "C"], [0.5, 0.5, "Cl"]], [[1, 2], [3, 4]], "M  CHG  1   2  -1\n"), atoms: [A(0, 0, "C", [2, 3, 4], 1), A(1, 1, "O", [], 2), A(2, 5, "C", [7, 8, 9], 6), A(3, 6, "Cl", [], 7)] };
const PROD_DEP = { ref: "p", molblock: mol([[-0.8666, 0.0865, "C"], [0.0012, 0.5833, "O"], [0.8654, 0.0801, "C"], [0, -0.75, "Cl"]], [[1, 2], [2, 3]], "M  CHG  1   4  -1\n"), atoms: [A(0, 0, "C", [6, 7, 8], 1), A(1, 1, "O", [], 2), A(2, 2, "C", [3, 4, 5], 6), A(3, 9, "Cl", [], 7)] };
const SRC = "[C:1]([O-:2])([H:3])([H:4])[H:5].[C:6]([Cl:7])([H:8])([H:9])[H:10]";
const PROD = "[C:1]([O:2][C:6]([H:8])([H:9])[H:10])([H:3])([H:4])[H:5].[Cl-:7]";
const matrix = { bonds: [[1, 5, 1], [5, 6, -1]], lone: [[1, -2], [6, 2]] };
const arrows = [{ kind: "lone_pair_to_bond", source: [1], target: [1, 5] }, { kind: "bond_to_lone_pair", source: [5, 6], target: [6] }];
const STYLE = { ...ACS_1996_STYLE, bondLengthPt: 17, chargePlacement: "circled", labelMethyls: true };

function input() {
  const s = poolStructure(SRC_DEP, { label: new Set([1, 5, 6]), facing: [[1, 5]] });
  const source = sideOf(s);
  const toSource = productIndexMap(SRC, PROD);
  const toProduct = new Map([...toSource].map(([k, i]) => [i, k]));
  const product = sideOf(poolStructure(PROD_DEP, { label: new Set([1, 5, 6].map((i) => toProduct.get(i))) }), toSource);
  const aligned = alignProduct(xyOf(source), xyOf(product), sideFragments(product), 2.1);
  return { source, product, aligned, matrix, arrows };
}

test("scheme layout: product to the right of the reactants, the reaction arrow between", () => {
  const inp = input();
  const g = schemeGeometry(inp);
  const rx = Math.max(...[...g.react.values()].map((p) => p[0]));
  const px = Math.min(...[...g.prod.values()].map((p) => p[0]));
  assert.ok(px > rx + 3, "a clear gap for the arrow");
  assert.ok(g.rxn.from[0] > rx && g.rxn.to[0] < px && g.rxn.to[0] > g.rxn.from[0]);
  assert.equal(g.prod.size, inp.product.atoms.size);
});

test("scheme drawing: lone pairs where the step touches, two routed arrows without crossings", () => {
  const inp = input();
  const g = schemeGeometry(inp);
  const { view } = fitScheme(g.box, 1200);
  const d = drawScheme(inp, g, view, STYLE, false, false);
  assert.equal(d.pairsR.get(1).angles.length, 3, "O- has three lone pairs");
  assert.equal(d.pairsR.get(6).angles.length, 3, "Cl has three");
  assert.equal(d.pairsP.get(6 + PRODUCT_ID).angles.length, 4, "Cl- has four in the product");
  assert.ok(!d.pairsR.has(5), "carbon has none");
  assert.equal(d.arrows.filter(Boolean).length, 2);
  assert.equal(arrowCrossings(d.arrows), 0);
  // the lone-pair arrow leaves from O's pair that faces C5
  const o = d.geomR.get(1).at;
  const c = d.geomR.get(5).at;
  const tail = d.arrows[0].tail;
  assert.ok(Math.hypot(tail[0] - c[0], tail[1] - c[1]) < Math.hypot(o[0] - c[0], o[1] - c[1]));
});

test("export: an SVG in points with true curves and the scheme's text", () => {
  const inp = input();
  const f = schemeSvg(inp, STYLE, { above: "floor", below: "score 1.28", background: "#ffffff", labelCarbons: false });
  assert.match(f.svg, /^<svg xmlns="http:\/\/www.w3.org\/2000\/svg"/);
  assert.equal((f.svg.match(/<path d="M [-\d.]+ [-\d.]+ C /g) ?? []).length, 2, "two cubic arrows");
  assert.match(f.svg, />floor</);
  assert.match(f.svg, />score 1.28</);
  assert.ok(f.widthPt > f.heightPt && f.widthPt > 100);
  assert.ok(!/NaN/.test(f.svg));
});
