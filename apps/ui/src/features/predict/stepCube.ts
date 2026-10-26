// The pool's 3D scene for the orbital variant (pure; node --test runs it). GET
// /pool/{state}/geometry gives each species in the frame of its own QM run plus a display offset
// that lays the species side by side; an orbital cube (GET /pool/{state}/orbital/{token}) is in
// the frame of its species. So the scene shifts each molblock and each cube by its species'
// offset (speciesOfCube finds the species by the cube's atoms), and maps pool atoms to 3D atoms
// (heavy atoms by global site, hidden H by their parent, in order) for the 2D -> 3D hover.
// Cube headers carry the LMO energy (eV) on their first line ("... E_r=-9.078568").
import type { DepictionOut, GeometryOut } from "../../api/types.gen";

export const BOHR = 0.529177210903;

export function cubeEnergy(text: string): number | null {
  const m = /E_r=(-?[\d.]+(?:e-?\d+)?)/i.exec(text.slice(0, text.indexOf("\n")));
  return m ? Number(m[1]) : null;
}

export interface CubeAtom {
  z: number;
  /** Position in Å. */
  at: [number, number, number];
}

/** The atoms of a cube's header (positions converted from bohr to Å). */
export function cubeAtoms(text: string): CubeAtom[] {
  const lines = text.split("\n");
  const n = Math.abs(Number.parseInt(lines[2]?.trim().split(/\s+/)[0] ?? "0", 10));
  const out: CubeAtom[] = [];
  for (let k = 0; k < n; k++) {
    const f = (lines[6 + k] ?? "").trim().split(/\s+/).map(Number);
    if (f.length >= 5) out.push({ z: f[0], at: [f[2] * BOHR, f[3] * BOHR, f[4] * BOHR] });
  }
  return out;
}

/** The species (index into geometry.species) whose atoms the cube carries, or null. */
export function speciesOfCube(text: string, g: GeometryOut, tol = 0.05): number | null {
  const atoms = cubeAtoms(text);
  if (!atoms.length) return null;
  const k = g.species.findIndex(
    (s) => s.atoms.length === atoms.length && s.atoms.every((a, i) => a.Z === atoms[i].z && Math.hypot(a.x - atoms[i].at[0], a.y - atoms[i].at[1], a.z - atoms[i].at[2]) < tol),
  );
  return k >= 0 ? k : null;
}

const num = (v: number, w: number, d: number) => v.toFixed(d).padStart(w);

/** The cube with its origin and atoms moved by `offset` (Å). */
export function offsetCube(text: string, offset: readonly [number, number, number]): string {
  if (!offset[0] && !offset[1] && !offset[2]) return text;
  const lines = text.split("\n");
  const shift = (line: string, from: number) => {
    const f = line.trim().split(/\s+/);
    const out = f.map((x, i) => (i >= from && i < from + 3 ? num(Number(x) + offset[i - from] / BOHR, 12, 6) : x));
    return " " + out.join(" ");
  };
  const n = Math.abs(Number.parseInt(lines[2].trim().split(/\s+/)[0], 10));
  lines[2] = shift(lines[2], 1);
  for (let k = 0; k < n; k++) lines[6 + k] = shift(lines[6 + k], 2);
  return lines.join("\n");
}

/** A V2000 molblock with its atoms moved by `offset` (Å). */
export function offsetMolblock(mol: string, offset: readonly [number, number, number]): string {
  if (!offset[0] && !offset[1] && !offset[2]) return mol;
  const lines = mol.split("\n");
  const n = Number.parseInt(lines[3]?.slice(0, 3) ?? "0", 10);
  for (let k = 0; k < n; k++) {
    const l = lines[4 + k];
    const xyz = [l.slice(0, 10), l.slice(10, 20), l.slice(20, 30)].map(Number);
    lines[4 + k] = xyz.map((v, i) => num(v + offset[i], 10, 4)).join("") + l.slice(30);
  }
  return lines.join("\n");
}

/** Where a pool atom is in the scene: species index, atom index in that species, position (Å,
 *  offset applied). */
export interface Atom3D {
  species: number;
  index: number;
  at: [number, number, number];
}

/** Pool atom -> scene atom: heavy atoms by global site; a hidden H by its parent's H neighbours
 *  in the geometry, in the order the depiction lists them. */
export function poolAtoms3D(dep: DepictionOut, g: GeometryOut): Map<number, Atom3D> {
  const site = new Map<number, Atom3D>();
  g.species.forEach((s, k) =>
    s.atoms.forEach((a, i) => a.global_site !== null && site.set(a.global_site, { species: k, index: i, at: [a.x + s.display_offset[0], a.y + s.display_offset[1], a.z + s.display_offset[2]] })),
  );
  const out = new Map<number, Atom3D>();
  for (const a of dep.atoms) {
    const me = a.global_site !== null ? site.get(a.global_site) : undefined;
    if (!me) continue;
    out.set(a.pool_atom, me);
    const s = g.species[me.species];
    const hs = s.bonds
      .filter(([i, j]) => i === me.index || j === me.index)
      .map(([i, j]) => (i === me.index ? j : i))
      .filter((i) => s.atoms[i]?.Z === 1)
      .sort((x, y) => x - y);
    (a.hidden_h_atoms ?? []).forEach((h, n) => {
      const i = hs[n];
      if (i !== undefined) out.set(h, { species: me.species, index: i, at: [s.atoms[i].x + s.display_offset[0], s.atoms[i].y + s.display_offset[1], s.atoms[i].z + s.display_offset[2]] });
    });
  }
  return out;
}
