// Fragment layout of a pool drawing (pure; node --test runs it). A server depiction may draw the
// species of a pool on top of each other (CoordGen with the H removed puts C[O-] and CCl one
// bond length apart, stacked). When two fragments' boxes, grown by half the gap, overlap, every
// fragment gets its own box: they are laid out left to right in their original x order (rows of
// ceil(sqrt(n)) when there are more than three), with FRAGMENT_GAP between boxes, each centred
// on its row. A drawing whose fragments are already apart is kept as it is. With `facing` pairs
// (the atoms of a bond R forms between two fragments) the first such pair is drawn face to face:
// its fragments come first, each turned by 180° (a rotation keeps the stereo drawing) when its
// atom is not on the side that faces the other, and the second moved level with the first.
// Boxes leave out the `ignore` atoms (drawn hidden H), so showing an H never moves a fragment.
import type { SketchModel } from "../workbench/sketch/model";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { BOND_LENGTH } from "../workbench/sketch/model.ts";

/** Space between fragment boxes (model units): two bond lengths, room for a forming bond's
 *  arrow to curve between them. */
export const FRAGMENT_GAP = 2 * BOND_LENGTH;
/** The smallest box a fragment takes (a lone ion still needs room for its label). */
const MIN_EXTENT = 0.6 * BOND_LENGTH;

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** The connected components of a sketch, as lists of atom ids (in atom order). */
export function fragmentsOf(m: SketchModel): number[][] {
  const parent = new Map(m.atoms.map((a) => [a.id, a.id] as const));
  const find = (x: number): number => {
    while (parent.get(x) !== x) x = parent.get(x)!;
    return x;
  };
  for (const b of m.bonds) parent.set(find(b.a), find(b.b));
  const groups = new Map<number, number[]>();
  for (const a of m.atoms) groups.set(find(a.id), [...(groups.get(find(a.id)) ?? []), a.id]);
  return [...groups.values()];
}

export function boxOf(m: SketchModel, ids: readonly number[]): Box {
  const pts = m.atoms.filter((a) => ids.includes(a.id));
  const xs = pts.map((a) => a.x);
  const ys = pts.map((a) => a.y);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const hw = Math.max((Math.max(...xs) - Math.min(...xs)) / 2, MIN_EXTENT / 2);
  const hh = Math.max((Math.max(...ys) - Math.min(...ys)) / 2, MIN_EXTENT / 2);
  return { x0: cx - hw, y0: cy - hh, x1: cx + hw, y1: cy + hh };
}

const overlap = (a: Box, b: Box, pad: number) => a.x0 - pad < b.x1 + pad && b.x0 - pad < a.x1 + pad && a.y0 - pad < b.y1 + pad && b.y0 - pad < a.y1 + pad;

/** Do any two fragments come closer than the gap? */
export function fragmentsCrowded(boxes: readonly Box[]): boolean {
  for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) if (overlap(boxes[i], boxes[j], FRAGMENT_GAP / 2 - 1e-6)) return true;
  return false;
}

/** The sketch with every fragment in its own box when they crowd each other (see the header). */
export function separateFragments(m0: SketchModel, facing: readonly (readonly [number, number])[] = [], ignore: ReadonlySet<number> = new Set()): SketchModel {
  const all = fragmentsOf(m0);
  if (all.length < 2) return m0;
  const frags = all.map((f) => f.filter((id) => !ignore.has(id)));
  const members = all;
  if (!fragmentsCrowded(frags.map((f) => boxOf(m0, f)))) return m0;
  const { m, first } = faceToFace(m0, frags, members, facing);
  const boxes = frags.map((f) => boxOf(m, f));
  const byX = (a: number, b: number) => boxes[a].x0 + boxes[a].x1 - (boxes[b].x0 + boxes[b].x1) || boxes[b].y1 - boxes[a].y1;
  const order = [...first, ...frags.map((_, k) => k).filter((k) => !first.includes(k)).sort(byX)];
  const perRow = frags.length > 3 ? Math.ceil(Math.sqrt(frags.length)) : frags.length;
  const shift = new Map<number, [number, number]>();
  let rowTop = 0;
  for (let r = 0; r * perRow < order.length; r++) {
    const row = order.slice(r * perRow, (r + 1) * perRow);
    const height = Math.max(...row.map((k) => boxes[k].y1 - boxes[k].y0));
    const cy = rowTop - height / 2;
    let x = 0;
    for (const k of row) {
      const b = boxes[k];
      const d: [number, number] = [x - b.x0, cy - (b.y0 + b.y1) / 2];
      for (const id of members[k]) shift.set(id, d);
      x += b.x1 - b.x0 + FRAGMENT_GAP;
    }
    rowTop -= height + FRAGMENT_GAP;
  }
  const out = { ...m, atoms: m.atoms.map((a) => ({ ...a, x: a.x + shift.get(a.id)![0], y: a.y + shift.get(a.id)![1] })) };
  return first.length === 2 ? level(out, facing, frags, members) : out;
}

/** Turn a fragment (all its members) by 180° about the centre of its box (of `ids`). */
function turn(m: SketchModel, ids: readonly number[], members: readonly number[]): SketchModel {
  const b = boxOf(m, ids);
  const [cx, cy] = [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2];
  const set = new Set(members);
  return { ...m, atoms: m.atoms.map((a) => (set.has(a.id) ? { ...a, x: 2 * cx - a.x, y: 2 * cy - a.y } : a)) };
}

/** The first inter-fragment pair: its two fragments (left, right) and their atoms. */
function firstPair(frags: number[][], facing: readonly (readonly [number, number])[]): [number, number, number, number] | null {
  const fragOf = (id: number) => frags.findIndex((f) => f.includes(id));
  for (const [a, b] of facing) {
    const [fa, fb] = [fragOf(a), fragOf(b)];
    if (fa >= 0 && fb >= 0 && fa !== fb) return [fa, fb, a, b];
  }
  return null;
}

/** Turn the paired fragments so their atoms face each other (left fragment's atom on its right). */
function faceToFace(m: SketchModel, frags: number[][], members: number[][], facing: readonly (readonly [number, number])[]): { m: SketchModel; first: number[] } {
  const p = firstPair(members, facing);
  if (!p) return { m, first: [] };
  const [fa, fb, a, b] = p;
  const x = (mm: SketchModel, id: number) => mm.atoms.find((t) => t.id === id)!.x;
  const centre = (mm: SketchModel, k: number) => (boxOf(mm, frags[k]).x0 + boxOf(mm, frags[k]).x1) / 2;
  let out = m;
  if (x(out, a) < centre(out, fa) - 1e-9) out = turn(out, frags[fa], members[fa]);
  if (x(out, b) > centre(out, fb) + 1e-9) out = turn(out, frags[fb], members[fb]);
  return { m: out, first: [fa, fb] };
}

/** Move the second paired fragment level with the first's atom (a straight forming bond). */
function level(m: SketchModel, facing: readonly (readonly [number, number])[], frags: number[][], members: number[][]): SketchModel {
  const p = firstPair(members, facing);
  if (!p || !frags.length) return m;
  const [, fb, a, b] = p;
  const dy = m.atoms.find((t) => t.id === a)!.y - m.atoms.find((t) => t.id === b)!.y;
  const set = new Set(members[fb]);
  return { ...m, atoms: m.atoms.map((t) => (set.has(t.id) ? { ...t, y: t.y + dy } : t)) };
}
