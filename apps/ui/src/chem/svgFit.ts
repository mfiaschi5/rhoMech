// Tier-2 atom positions of an RDKit SVG (design §11.4), as pure functions without DOM or
// imports so they can be checked outside the browser.
//
// The fit is X = s x + tx, Y = -s y + ty from the molblock's 2D coordinates to the drawing, on
// exact observations only: the ends of single-stroke bonds (one line, possibly split in two
// colour halves) at atoms without a label. Double and triple bonds are skipped (their offset
// strokes are trimmed and do not end on the atoms). It needs two distinct exact atoms and is
// accepted when their RMS is <= FIT_MAX_RMS_PX and every labelled atom lands inside its own
// label's glyph bbox (+ LABEL_SLACK_PX). Label bbox centres are not used as observations: charge
// signs and H counts move them by up to 6 px (checked against RDKit's GetDrawCoords).

export type Point = [number, number];

/** One drawn element: its tag, class attribute, path data and text anchor (if any). */
export interface SvgPrimitive {
  tag: string;
  cls: string;
  d?: string;
  x?: number;
  y?: number;
}

export const FIT_MAX_RMS_PX = 1.5;
export const LABEL_SLACK_PX = 1.5;
const COLLINEAR_PX = 0.75;

export interface Observation {
  idx: number;
  X: number;
  Y: number;
  w: number;
}

export type Box = [number, number, number, number]; // x0, y0, x1, y1

export interface FitResult {
  positions: Map<number, Point>;
  rms: number;
  s: number;
  tx: number;
  ty: number;
  nExact: number;
}

/** Molblock (V2000 or V3000) atom coordinates, by atom index. */
export function molblockCoords(molblock: string): Point[] {
  const lines = molblock.split(/\r?\n/);
  const v3000 = lines.findIndex((l) => l.startsWith("M  V30 BEGIN ATOM"));
  const out: Point[] = [];
  if (v3000 >= 0) {
    for (let i = v3000 + 1; i < lines.length && !lines[i].startsWith("M  V30 END ATOM"); i++) {
      const f = lines[i].slice(7).trim().split(/\s+/);
      out.push([Number.parseFloat(f[2]), Number.parseFloat(f[3])]);
    }
    return out;
  }
  const n = Number.parseInt((lines[3] ?? "").slice(0, 3), 10);
  for (let i = 0; i < (Number.isFinite(n) ? n : 0); i++) {
    const line = lines[4 + i] ?? "";
    out.push([Number.parseFloat(line.slice(0, 10)), Number.parseFloat(line.slice(10, 20))]);
  }
  return out;
}

export function pathPoints(d: string): Point[] {
  const nums = (d.match(/-?\d*\.?\d+(?:e[-+]?\d+)?/gi) ?? []).map(Number);
  const pts: Point[] = [];
  for (let i = 0; i + 1 < nums.length; i += 2) pts.push([nums[i], nums[i + 1]]);
  return pts;
}

export function classIndices(cls: string, prefix: string): number[] {
  const re = new RegExp(`(?:^|\\s)${prefix}-(\\d+)(?=\\s|$)`, "g");
  return Array.from(cls.matchAll(re), (m) => Number.parseInt(m[1], 10));
}

/** The glyph bbox of every atom label, by atom index (the atoms that carry a label). */
export function labelBoxes(prims: readonly SvgPrimitive[]): Map<number, Box> {
  const boxes = new Map<number, Box>();
  for (const p of prims) {
    if (/(?:^|\s)bond-\d+/.test(p.cls)) continue;
    const atoms = classIndices(p.cls, "atom");
    const pts = p.d ? pathPoints(p.d) : p.x !== undefined && p.y !== undefined ? [[p.x, p.y] as Point] : [];
    if (atoms.length !== 1 || pts.length === 0) continue;
    const b = boxes.get(atoms[0]) ?? [Infinity, Infinity, -Infinity, -Infinity];
    for (const [x, y] of pts) {
      b[0] = Math.min(b[0], x);
      b[1] = Math.min(b[1], y);
      b[2] = Math.max(b[2], x);
      b[3] = Math.max(b[3], y);
    }
    boxes.set(atoms[0], b);
  }
  return boxes;
}

function bondGroups(prims: readonly SvgPrimitive[]): Map<number, { atoms: number[]; segs: [Point, Point][] }> {
  const groups = new Map<number, { atoms: number[]; segs: [Point, Point][] }>();
  for (const p of prims) {
    if (p.tag !== "path" || !p.d) continue;
    const [bond] = classIndices(p.cls, "bond");
    const atoms = classIndices(p.cls, "atom");
    const pts = pathPoints(p.d);
    if (bond === undefined || atoms.length !== 2) continue;
    const g = groups.get(bond) ?? { atoms, segs: [] };
    if (pts.length === 2) g.segs.push([pts[0], pts[1]]);
    else g.segs.push([pts[0] ?? [NaN, NaN], [NaN, NaN]]); // wedges, wavy bonds: unusable group
    groups.set(bond, g);
  }
  return groups;
}

function collinear(segs: [Point, Point][]): boolean {
  const [a, b] = segs[0];
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  if (!(len > 1e-6)) return false;
  const off = (p: Point) => Math.abs((p[0] - a[0]) * dy - (p[1] - a[1]) * dx) / len;
  return segs.every(([p, q]) => off(p) <= COLLINEAR_PX && off(q) <= COLLINEAR_PX);
}

/** Ends of single-stroke bonds at unlabelled atoms. Ends are matched to atoms through the
 *  molblock direction of the bond under the y flip (X = s x, Y = -s y, s > 0). */
export function bondObservations(
  prims: readonly SvgPrimitive[],
  coords: Point[],
  labelled: ReadonlySet<number> | ReadonlyMap<number, unknown>,
): Observation[] {
  const obs: Observation[] = [];
  for (const { atoms, segs } of bondGroups(prims).values()) {
    const [i, j] = atoms;
    const ci = coords[i];
    const cj = coords[j];
    if (!ci || !cj || !segs.length || !collinear(segs)) continue;
    const ux = cj[0] - ci[0];
    const uy = -(cj[1] - ci[1]);
    const ends = segs.flat();
    const proj = ends.map(([x, y]) => x * ux + y * uy);
    const lo = ends[proj.indexOf(Math.min(...proj))];
    const hi = ends[proj.indexOf(Math.max(...proj))];
    if (!labelled.has(i)) obs.push({ idx: i, X: lo[0], Y: lo[1], w: 1 });
    if (!labelled.has(j)) obs.push({ idx: j, X: hi[0], Y: hi[1], w: 1 });
  }
  return obs;
}

function solveFit(obs: readonly Observation[], coords: Point[]): { s: number; tx: number; ty: number } | null {
  let W = 0;
  let mx = 0;
  let my = 0;
  let mX = 0;
  let mY = 0;
  for (const o of obs) {
    const [x, y] = coords[o.idx];
    W += o.w;
    mx += o.w * x;
    my += o.w * y;
    mX += o.w * o.X;
    mY += o.w * o.Y;
  }
  if (!(W > 0)) return null;
  mx /= W;
  my /= W;
  mX /= W;
  mY /= W;
  let num = 0;
  let den = 0;
  for (const o of obs) {
    const dx = coords[o.idx][0] - mx;
    const dy = coords[o.idx][1] - my;
    num += o.w * (dx * (o.X - mX) - dy * (o.Y - mY));
    den += o.w * (dx * dx + dy * dy);
  }
  if (!(den > 1e-12) || !(num / den > 0)) return null;
  const s = num / den;
  return { s, tx: mX - s * mx, ty: mY + s * my };
}

function inside(p: Point, b: Box, slack: number): boolean {
  return p[0] >= b[0] - slack && p[0] <= b[2] + slack && p[1] >= b[1] - slack && p[1] <= b[3] + slack;
}

/** Drawing positions of every atom from the fit, or null when it is not accepted. */
export function fitAtomPositions(prims: readonly SvgPrimitive[], coords: Point[], nAtoms: number): FitResult | null {
  if (coords.length !== nAtoms || coords.some(([x, y]) => !Number.isFinite(x) || !Number.isFinite(y))) return null;
  const labels = labelBoxes(prims);
  const exact = bondObservations(prims, coords, labels);
  if (new Set(exact.map((o) => o.idx)).size < 2) return null;
  const fit = solveFit(exact, coords);
  if (!fit) return null;
  const at = (i: number): Point => [fit.s * coords[i][0] + fit.tx, -fit.s * coords[i][1] + fit.ty];
  const sq = exact.map((o) => {
    const [X, Y] = at(o.idx);
    return (X - o.X) ** 2 + (Y - o.Y) ** 2;
  });
  const rms = Math.sqrt(sq.reduce((a, b) => a + b, 0) / sq.length);
  if (!(rms <= FIT_MAX_RMS_PX)) return null;
  for (const [idx, box] of labels) {
    if (idx < nAtoms && !inside(at(idx), box, LABEL_SLACK_PX)) return null;
  }
  const positions = new Map<number, Point>();
  coords.forEach((_, i) => positions.set(i, at(i)));
  return { positions, rms, ...fit, nExact: exact.length };
}
