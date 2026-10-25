// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
import test from "node:test";
import assert from "node:assert/strict";
import { fromMolblock, looksLikeMolfile, toMolblock } from "./molfile.ts";
import { addAtom, addBond, emptySketch, setBondStereo } from "./model.ts";

function zwitterion() {
  // C[NH2+]C(C)C(=O)[O-] with a wedge and a hash at the stereocentre
  let m = emptySketch();
  const ids = [];
  const atoms = [
    ["C", -2.6, 0.75, 0],
    ["N", -1.3, 0, 1],
    ["C", 0, 0.75, 0],
    ["C", 0, 2.25, 0],
    ["C", 1.3, 0, 0],
    ["O", 1.3, -1.5, 0],
    ["O", 2.6, 0.75, -1],
    ["F", -0.9, 1.6, 0],
  ];
  for (const [el, x, y, charge] of atoms) {
    let id;
    [m, id] = addAtom(m, el, x, y, { charge });
    ids.push(id);
  }
  const bonds = [[0, 1, 1], [1, 2, 1], [2, 3, 1], [2, 4, 1], [4, 5, 2], [4, 6, 1], [2, 7, 1]];
  const bondIds = [];
  for (const [i, j, o] of bonds) {
    let id;
    [m, id] = addBond(m, ids[i], ids[j], o);
    bondIds.push(id);
  }
  m = setBondStereo(m, bondIds[2], "wedge");
  m = setBondStereo(m, bondIds[6], "hash");
  return m;
}

test("V2000 writer: counts line, fixed columns, M  CHG, stereo flags", () => {
  const mb = toMolblock(zwitterion());
  const lines = mb.split("\n");
  assert.equal(lines[3], "  8  7  0  0  0  0  0  0  0  0999 V2000");
  assert.equal(lines[5], "   -1.3000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0");
  assert.ok(lines.includes("  3  4  1  1"), "wedge from the stereocentre");
  assert.ok(lines.includes("  3  8  1  6"), "hash from the stereocentre");
  assert.ok(lines.includes("  5  6  2  0"));
  assert.ok(lines.includes("M  CHG  2   2   1   7  -1"));
  assert.ok(mb.endsWith("M  END\n"));
});

test("molfile round trip keeps elements, coordinates, charges, orders and wedges", () => {
  const m = zwitterion();
  const back = fromMolblock(toMolblock(m));
  assert.equal(back.atoms.length, m.atoms.length);
  m.atoms.forEach((a, i) => {
    const b = back.atoms[i];
    assert.equal(b.el, a.el);
    assert.equal(b.charge, a.charge);
    assert.ok(Math.abs(b.x - a.x) < 1e-4 && Math.abs(b.y - a.y) < 1e-4);
  });
  const idx = new Map(m.atoms.map((a, i) => [a.id, i]));
  const bidx = new Map(back.atoms.map((a, i) => [a.id, i]));
  m.bonds.forEach((b, k) => {
    const c = back.bonds[k];
    assert.deepEqual([bidx.get(c.a), bidx.get(c.b), c.order, c.stereo], [idx.get(b.a), idx.get(b.b), b.order, b.stereo]);
  });
  assert.equal(toMolblock(back), toMolblock(m), "writer is a fixed point of the reader");
});

test("reader: old-style charges, M  CHG precedence, SDF records, many charges", () => {
  const atom = (el, chg) => `    0.0000    0.0000    0.0000 ${el.padEnd(3)} 0${String(chg).padStart(3)}  0  0  0  0  0  0  0  0  0  0`;
  const base = ["", "  x", "", "  2  0  0  0  0  0  0  0  0  0999 V2000", atom("N", 3), atom("O", 5)];
  const old = fromMolblock([...base, "M  END", "> <name>", "x", "", "$$$$", "next record"].join("\n"));
  assert.deepEqual(old.atoms.map((a) => a.charge), [1, -1]);
  const chg = fromMolblock([...base, "M  CHG  1   2  -2", "M  END"].join("\n"));
  assert.deepEqual(chg.atoms.map((a) => a.charge), [0, -2], "M  CHG resets the atom block");
  let m = emptySketch();
  for (let i = 0; i < 10; i++) [m] = addAtom(m, "N", i * 1.5, 0, { charge: 1 });
  const ten = toMolblock(m);
  assert.equal(ten.split("\n").filter((l) => l.startsWith("M  CHG")).length, 2);
  assert.deepEqual(fromMolblock(ten).atoms.map((a) => a.charge), Array(10).fill(1));
});

test("reader refuses aromatic bonds and V3000, normalizes bond length on request", () => {
  const head = ["", "", "", "  2  1  0  0  0  0  0  0  0  0999 V2000"];
  const atoms = ["    0.0000    0.0000    0.0000 C   0  0", "    1.0000    0.0000    0.0000 C   0  0"];
  assert.throws(() => fromMolblock([...head, ...atoms, "  1  2  4  0", "M  END"].join("\n")), /type 4/);
  assert.throws(() => fromMolblock("\n\n\n  0  0  0  0  0  0  0  0  0  0999 V3000\nM  END"), /V3000/);
  const scaled = fromMolblock([...head, ...atoms, "  1  2  1  0", "M  END"].join("\n"), { normalize: true });
  assert.ok(Math.abs(scaled.atoms[1].x - scaled.atoms[0].x - 1.5) < 1e-9);
  assert.ok(looksLikeMolfile([...head, "M  END"].join("\n")));
  assert.ok(!looksLikeMolfile("CCO"));
});
