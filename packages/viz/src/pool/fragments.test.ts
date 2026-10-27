// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// Fragment separation and fit: the real C[O-] + CCl depiction (CoordGen stacked the two species
// one bond apart) gets one box per fragment side by side, apart drawings are kept, many
// fragments wrap into rows; the view then fits the drawing at a capped bond length, centred.
import test from "node:test";
import assert from "node:assert/strict";
import { FRAGMENT_GAP, boxOf, fragmentsCrowded, fragmentsOf, separateFragments } from "./fragments.ts";
import { poolStructure } from "./poolStructure.ts";
import { MAX_SCALE, poolView } from "./poolView.ts";

/** GET /pool/s187ab3f19a88fa83ed9a/depiction of the live server (C[O-] + CCl, H hidden). */
const STACKED = {
  ref: "s187ab3f19a88fa83ed9a",
  molblock: [
    "     RDKit          2D",
    "",
    "  4  2  0  0  0  0  0  0  0  0999 V2000",
    "   -0.5000   -0.5000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
    "    0.5000   -0.5000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0",
    "   -0.5000    0.5000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
    "    0.5000    0.5000    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0",
    "  1  2  1  0",
    "  3  4  1  0",
    "M  CHG  1   2  -1",
    "M  END",
  ].join("\n"),
  atoms: [
    { depiction_idx: 0, pool_atom: 0, hidden_h_atoms: [2, 3, 4], global_site: 0, map: 1, element: "C", fragment: 0, hidden_h: 3 },
    { depiction_idx: 1, pool_atom: 1, hidden_h_atoms: [], global_site: 1, map: 2, element: "O", fragment: 0, hidden_h: 0 },
    { depiction_idx: 2, pool_atom: 5, hidden_h_atoms: [7, 8, 9], global_site: 5, map: 6, element: "C", fragment: 1, hidden_h: 3 },
    { depiction_idx: 3, pool_atom: 6, hidden_h_atoms: [], global_site: 6, map: 7, element: "Cl", fragment: 1, hidden_h: 0 },
  ],
  server_svg: null,
  server_coords: null,
};

const atom = (id, x, y) => ({ id, el: "C", x, y, charge: 0 });
const bond = (id, a, b) => ({ id, a, b, order: 1, stereo: "none" });

test("stacked fragments get their own boxes, side by side with the gap", () => {
  const s = poolStructure(STACKED);
  const frags = fragmentsOf(s.model);
  assert.equal(frags.length, 2);
  const [a, b] = frags.map((f) => boxOf(s.model, f)).sort((p, q) => p.x0 - q.x0);
  assert.ok(b.x0 - a.x1 >= FRAGMENT_GAP - 1e-9, `gap ${b.x0 - a.x1}`);
  assert.ok(!fragmentsCrowded([a, b]));
  assert.ok(Math.abs((a.y0 + a.y1) / 2 - (b.y0 + b.y1) / 2) < 1e-9, "both on one row");
  const bondLen = (i, j) => Math.hypot(s.atom(i).x - s.atom(j).x, s.atom(i).y - s.atom(j).y);
  assert.ok(Math.abs(bondLen(0, 1) - 1.5) < 1e-6 && Math.abs(bondLen(5, 6) - 1.5) < 1e-6, "bonds keep their length");
});

test("fragments already apart are kept; more than three wrap into rows", () => {
  const apart = { atoms: [atom(1, 0, 0), atom(2, 1.5, 0), atom(3, 10, 0), atom(4, 11.5, 0)], bonds: [bond(5, 1, 2), bond(6, 3, 4)], nextId: 7 };
  assert.equal(separateFragments(apart), apart);
  const five = { atoms: [0, 1, 2, 3, 4].map((k) => atom(k + 1, 0, 0.1 * k)), bonds: [], nextId: 6 };
  const out = separateFragments(five);
  const ys = [...new Set(out.atoms.map((a) => a.y.toFixed(3)))];
  assert.equal(ys.length, 2, "3 + 2 in two rows");
  assert.ok(!fragmentsCrowded(fragmentsOf(out).map((f) => boxOf(out, f))));
});

test("the view fits the separated drawing, centred, at a capped bond length", () => {
  const s = poolStructure(STACKED);
  const pv = poolView(s, 900, 380);
  assert.ok(pv.bondPx <= MAX_SCALE * 1.5 + 1e-9 && pv.bondPx >= 40, `bond ${pv.bondPx} px`);
  const xs = [...pv.xy.values()].map((p) => p[0]);
  const ys = [...pv.xy.values()].map((p) => p[1]);
  assert.ok(Math.abs((Math.min(...xs) + Math.max(...xs)) / 2 - 450) < 1);
  assert.ok(Math.abs((Math.min(...ys) + Math.max(...ys)) / 2 - 190) < 1);
  const small = poolView(s, 260, 180);
  assert.ok(small.bondPx < pv.bondPx, "a narrow canvas scales down");
});

test("the atoms R touches are drawn with their symbol", () => {
  const s = poolStructure(STACKED, { label: new Set([5]) });
  const c5 = s.model.atoms.find((a) => a.id === s.atom(5).sketchId);
  const c0 = s.model.atoms.find((a) => a.id === s.atom(0).sketchId);
  assert.equal(c5.explicitH, true);
  assert.ok(!c0.explicitH);
});

test("a forming bond between fragments draws them face to face, level", () => {
  // C[O-] + CCl with R forming O1-C5: MeO- on the left with O on its right, C5 on the left of CCl
  const s = poolStructure(STACKED, { facing: [[1, 5]] });
  const [c0, o1, c5, cl6] = [0, 1, 5, 6].map((i) => s.atom(i));
  assert.ok(c0.x < o1.x && o1.x < c5.x && c5.x < cl6.x, `order ${[c0, o1, c5, cl6].map((a) => a.x.toFixed(2))}`);
  assert.ok(Math.abs(o1.y - c5.y) < 1e-9, "the forming bond is level");
  assert.ok(c5.x - o1.x >= FRAGMENT_GAP - 1e-9);
});

test("showing a hidden H never moves a fragment", () => {
  const a = poolStructure(STACKED, { facing: [[1, 5]] });
  const b = poolStructure(STACKED, { facing: [[1, 5]], show: new Set([7, 8]) });
  for (const i of [0, 1, 5, 6]) assert.ok(Math.abs(a.atom(i).x - b.atom(i).x) < 1e-9 && Math.abs(a.atom(i).y - b.atom(i).y) < 1e-9);
});
