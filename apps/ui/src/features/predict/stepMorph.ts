// The animated step (StepAnimated) as a pure function of time (node --test runs it): the plan
// of a morph from the reactant side to the product side (stepGeometry.ts; atoms keep their
// identity by source pool atom) and the frame at t in [0, 1].
//
// Timeline (PHASES, fractions of the run):
//   arrows      0.00-0.22  the curved arrows draw on (stroke progress 0 -> 1)
//   loneOut     0.24-0.45  the lone pair an arrow takes fades out
//   move        0.26-0.85  atoms move from their reactant to their product position (smoothstep)
//   breakFade   0.30-0.60  breaking bonds (order down) fade out
//   formFade    0.45-0.80  forming bonds (order up) fade in
//   charge      0.55       labels switch to the product's charges and H counts
//   loneIn      0.60-0.85  new lone pairs fade in
//   arrowsOut   0.50-0.72  the arrows fade away (their ends follow the moving atoms)
// A bond whose order changes (1 -> 2) cross-fades its two drawings over breakFade / formFade.
import type { Vec } from "./arrowPath";
import type { Side, XY } from "./stepGeometry";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { pairKey } from "./stepGeometry.ts";

export type Span = readonly [number, number];

export const PHASES = {
  arrows: [0, 0.22] as Span,
  loneOut: [0.24, 0.45] as Span,
  move: [0.26, 0.85] as Span,
  breakFade: [0.3, 0.6] as Span,
  formFade: [0.45, 0.8] as Span,
  charge: 0.55,
  loneIn: [0.6, 0.85] as Span,
  arrowsOut: [0.5, 0.72] as Span,
};

export const clamp01 = (x: number): number => Math.max(0, Math.min(1, x));
/** Progress through a span (0 before, 1 after). */
export const phase = (t: number, [a, b]: Span): number => clamp01((t - a) / Math.max(b - a, 1e-9));
/** Smoothstep: zero slope at both ends. */
export const ease = (u: number): number => u * u * (3 - 2 * u);

export interface MorphBond {
  key: string;
  i: number;
  j: number;
  before: number;
  after: number;
}

export interface MorphPlan {
  /** Atoms drawn on either side (source pool atoms). */
  atoms: number[];
  from: XY;
  to: XY;
  bonds: MorphBond[];
  before: Side;
  after: Side;
}

/** The union of both sides' bonds with their orders; an atom missing on one side stays put. */
export function morphPlan(before: Side, after: Side, to: XY): MorphPlan {
  const from: XY = new Map([...before.atoms.values()].map((a) => [a.index, [a.x, a.y] as Vec]));
  const atoms = [...new Set([...before.atoms.keys(), ...after.atoms.keys()])].sort((a, b) => a - b);
  const target: XY = new Map(atoms.map((i) => [i, to.get(i) ?? from.get(i) ?? [0, 0]] as const));
  for (const i of atoms) if (!from.has(i)) from.set(i, target.get(i)!);
  const keys = new Set([...before.bonds.keys(), ...after.bonds.keys()]);
  const bonds = [...keys].map((key) => {
    const b = before.bonds.get(key) ?? after.bonds.get(key)!;
    return { key, i: b.i, j: b.j, before: before.bonds.get(key)?.order ?? 0, after: after.bonds.get(key)?.order ?? 0 };
  });
  return { atoms, from, to: target, bonds, before, after };
}

export interface BondAlpha {
  /** Opacity of the bond drawn with its reactant order. */
  before: number;
  /** Opacity of the bond drawn with its product order. */
  after: number;
}

export interface MorphFrame {
  t: number;
  xy: XY;
  bonds: Map<string, BondAlpha>;
  /** Stroke progress of the arrows (0-1) and their opacity. */
  arrowDraw: number;
  arrowAlpha: number;
  /** Labels (charges, H counts) of the product side. */
  productLabels: boolean;
  loneOut: number;
  loneIn: number;
}

const lerp = (p: Vec, q: Vec, u: number): Vec => [p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u];

/** The opacities of one bond at t: unchanged bonds are solid throughout; a broken bond fades
 *  out over breakFade, a new one in over formFade, an order change cross-fades over both. */
export function bondAlpha(b: MorphBond, t: number): BondAlpha {
  if (b.before === b.after) return { before: 1, after: 0 };
  if (b.after === 0) return { before: 1 - phase(t, PHASES.breakFade), after: 0 };
  if (b.before === 0) return { before: 0, after: phase(t, PHASES.formFade) };
  const c = phase(t, [PHASES.breakFade[0], PHASES.formFade[1]]);
  return { before: 1 - c, after: c };
}

export function morphFrame(plan: MorphPlan, t0: number): MorphFrame {
  const t = clamp01(t0);
  const u = ease(phase(t, PHASES.move));
  const xy: XY = new Map(plan.atoms.map((i) => [i, lerp(plan.from.get(i)!, plan.to.get(i)!, u)] as const));
  return {
    t,
    xy,
    bonds: new Map(plan.bonds.map((b) => [b.key, bondAlpha(b, t)] as const)),
    arrowDraw: ease(phase(t, PHASES.arrows)),
    arrowAlpha: 1 - phase(t, PHASES.arrowsOut),
    productLabels: t >= PHASES.charge,
    loneOut: 1 - phase(t, PHASES.loneOut),
    loneIn: phase(t, PHASES.loneIn),
  };
}

/** The bond key list of a plan that changes (the side card's bond rows). */
export const changedBonds = (plan: MorphPlan): MorphBond[] => plan.bonds.filter((b) => b.before !== b.after);

export { pairKey };
