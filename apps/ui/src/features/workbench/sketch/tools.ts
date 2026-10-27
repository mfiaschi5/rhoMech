// The sketcher's tools as pure model edits: what a click does with each tool, what a key does
// on the hovered atom or bond (ChemDraw hotkeys; the whole key map is shortcuts.ts), and the
// hit test that finds that object.
// Radicals: the radical tool gives an atom one unpaired electron, the diradical (carbene) tool
// two; a second click with the same tool clears them. The "." key cycles none -> one -> two.
import type { BondOrder, SketchModel } from "@ksmech/viz/sketch/model";
import type { Vec, View } from "@ksmech/viz/sketch/geometry";
import type { RingKind } from "./templates";
import { addAtom, adjustCharge, atomById, cycleBondOrder, cycleRadical, removeAtoms, removeBonds, removeSelection, setBondOrder, setBondStereo, setCharge, setElement, toggleExplicitH, toggleRadical } from "@ksmech/viz/sketch/model";
import { isElement } from "@ksmech/viz/sketch/elements";
import { dist, segmentDistance, toScreen } from "@ksmech/viz/sketch/geometry";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { addRing, bondFromPoint, CYCLOALKANE_OF_SIZE, growBond } from "./templates.ts";

export const TOOLS = [
  "select",
  "atom",
  "bond",
  "wedge",
  "hash",
  "chain",
  "benzene",
  "cyclohexane",
  "cyclopentane",
  "charge-plus",
  "charge-minus",
  "explicit-h",
  "radical",
  "diradical",
  "eraser",
] as const;
export type ToolName = (typeof TOOLS)[number];

export const RING_TOOLS: Readonly<Partial<Record<ToolName, RingKind>>> = {
  benzene: "benzene",
  cyclohexane: "cyclohexane",
  cyclopentane: "cyclopentane",
};

export type Hit = { kind: "atom"; id: number } | { kind: "bond"; id: number } | null;

export interface Selection {
  atoms: ReadonlySet<number>;
  bonds: ReadonlySet<number>;
}

export const NO_SELECTION: Selection = { atoms: new Set(), bonds: new Set() };

/** The atom within `atomPx` of the screen point (closest first), else the bond within `bondPx`;
 *  atoms in `exclude` (and their bonds) are ignored. */
export function hitTest(
  m: SketchModel,
  view: View,
  p: Vec,
  exclude: ReadonlySet<number> = new Set(),
  atomPx = 10,
  bondPx = 6,
): Hit {
  let best: Hit = null;
  let bestD = atomPx;
  for (const a of m.atoms) {
    if (exclude.has(a.id)) continue;
    const d = dist(p, toScreen(view, [a.x, a.y]));
    if (d <= bestD) {
      best = { kind: "atom", id: a.id };
      bestD = d;
    }
  }
  if (best) return best;
  bestD = bondPx;
  for (const b of m.bonds) {
    const pa = atomById(m, b.a);
    const pb = atomById(m, b.b);
    if (!pa || !pb || exclude.has(b.a) || exclude.has(b.b)) continue;
    const d = segmentDistance(p, toScreen(view, [pa.x, pa.y]), toScreen(view, [pb.x, pb.y]));
    if (d <= bestD) {
      best = { kind: "bond", id: b.id };
      bestD = d;
    }
  }
  return best;
}

/** The edit of a click with `tool` on `hit` (or on empty canvas at model point `at`). */
export function applyClick(m: SketchModel, tool: ToolName, element: string, hit: Hit, at: Vec): SketchModel {
  const ring = RING_TOOLS[tool];
  if (ring) {
    const place = hit?.kind === "atom" ? hit : hit?.kind === "bond" ? { ...hit, at } : { kind: "empty" as const, at };
    return addRing(m, ring, place).model;
  }
  switch (tool) {
    case "atom":
      if (hit?.kind === "atom") return setElement(m, [hit.id], element);
      return hit ? m : addAtom(m, element, at[0], at[1])[0];
    case "bond":
    case "wedge":
    case "hash":
    case "chain":
      return bondClick(m, tool, hit, at);
    case "charge-plus":
    case "charge-minus":
      return hit?.kind === "atom" ? adjustCharge(m, [hit.id], tool === "charge-plus" ? 1 : -1) : m;
    case "explicit-h":
      return hit?.kind === "atom" ? toggleExplicitH(m, [hit.id]) : m;
    case "radical":
    case "diradical":
      return hit?.kind === "atom" ? toggleRadical(m, [hit.id], tool === "radical" ? 1 : 2) : m;
    case "eraser":
      if (hit?.kind === "atom") return removeAtoms(m, [hit.id]);
      return hit?.kind === "bond" ? removeBonds(m, [hit.id]) : m;
    default:
      return m;
  }
}

function bondClick(m: SketchModel, tool: "bond" | "wedge" | "hash" | "chain", hit: Hit, at: Vec): SketchModel {
  const stereo = tool === "wedge" || tool === "hash" ? tool : "none";
  if (hit?.kind === "atom") return growBond(m, hit.id, 1, stereo).model;
  if (hit?.kind === "bond") return stereo === "none" ? cycleBondOrder(m, hit.id) : setBondStereo(m, hit.id, stereo);
  return bondFromPoint(m, at, 1, stereo).model;
}

/** ChemDraw element hotkeys (on a hovered atom). */
export const KEY_ELEMENTS: Readonly<Record<string, string>> = {
  c: "C",
  n: "N",
  o: "O",
  s: "S",
  f: "F",
  p: "P",
  l: "Cl",
  b: "Br",
  i: "I",
  h: "H",
};

/** ChemDraw bond hotkeys (on a hovered bond): wedge and hash; again flips the narrow end. */
export const KEY_BOND_STEREO: Readonly<Record<string, "wedge" | "hash">> = { w: "wedge", h: "hash" };

/** The key that cycles an atom's unpaired electrons (none -> one -> two -> none). */
export const RADICAL_KEY = ".";
/** ChemDraw: Space on an atom removes its label (back to a plain carbon). */
export const CARBON_KEY = " ";

function atomKey(m: SketchModel, k: string, id: number, order: BondOrder | null, delta: number, del: boolean): SketchModel | null {
  if (KEY_ELEMENTS[k]) return setElement(m, [id], KEY_ELEMENTS[k]);
  if (k === CARBON_KEY) return setCharge(setElement(m, [id], "C"), id, 0);
  if (delta) return adjustCharge(m, [id], delta);
  if (order) return growBond(m, id, order).model;
  if (k === RADICAL_KEY) return cycleRadical(m, [id]);
  return del ? removeAtoms(m, [id]) : null;
}

function bondKey(m: SketchModel, k: string, id: number, order: BondOrder | null, del: boolean): SketchModel | null {
  if (order) return setBondOrder(m, id, order);
  if (KEY_BOND_STEREO[k]) return setBondStereo(m, id, KEY_BOND_STEREO[k]);
  return del ? removeBonds(m, [id]) : null;
}

function selectionKey(m: SketchModel, k: string, sel: Selection, order: BondOrder | null, delta: number, del: boolean): SketchModel | null {
  if (del) return removeSelection(m, sel.atoms, sel.bonds);
  if (KEY_ELEMENTS[k] && sel.atoms.size) return setElement(m, sel.atoms, KEY_ELEMENTS[k]);
  if (delta && sel.atoms.size) return adjustCharge(m, sel.atoms, delta);
  if (k === RADICAL_KEY && sel.atoms.size) return cycleRadical(m, sel.atoms);
  if (order && sel.bonds.size) return [...sel.bonds].reduce((acc, id) => setBondOrder(acc, id, order), m);
  const stereo = KEY_BOND_STEREO[k];
  if (stereo && sel.bonds.size) return [...sel.bonds].reduce((acc, id) => setBondStereo(acc, id, stereo), m);
  return null;
}

/** The edit of a key on the hovered object, or on the selection when nothing is hovered; null
 *  when the key means nothing there (the event is then left alone). */
export function applyKey(m: SketchModel, key: string, hit: Hit, sel: Selection): SketchModel | null {
  const k = key.length === 1 ? key.toLowerCase() : key;
  const del = k === "Delete" || k === "Backspace";
  const delta = k === "+" || k === "=" ? 1 : k === "-" || k === "_" ? -1 : 0;
  const order = k === "1" || k === "2" || k === "3" ? (Number(k) as BondOrder) : null;
  if (hit?.kind === "atom") return atomKey(m, k, hit.id, order, delta, del);
  if (hit?.kind === "bond") return bondKey(m, k, hit.id, order, del);
  if (sel.atoms.size === 0 && sel.bonds.size === 0) return null;
  return selectionKey(m, k, sel, order, delta, del);
}

/** The key event fields the ring hotkeys read. */
export interface RingKeyLike {
  key: string;
  code?: string;
  shiftKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  altKey: boolean;
}

/** The ring template of a key: Shift+3 ... Shift+8 a cycloalkane of that size, Shift+A benzene
 *  (by physical key, since Shift+digit gives a different character on every layout); null for
 *  any other key. Not ChemDraw defaults (ChemDraw has no ring hotkeys): Marvin-like additions. */
export function ringKey(e: RingKeyLike): RingKind | null {
  if (!e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return null;
  const digit = e.code?.startsWith("Digit") ? Number(e.code.slice(5)) : !e.code && /^[3-8]$/.test(e.key) ? Number(e.key) : NaN;
  if (CYCLOALKANE_OF_SIZE[digit]) return CYCLOALKANE_OF_SIZE[digit];
  return e.code === "KeyA" || (!e.code && e.key.toLowerCase() === "a") ? "benzene" : null;
}

/** A ring template on the hovered atom (spiro) or bond (fused); null with nothing hovered. */
export function applyRingKey(m: SketchModel, ring: RingKind, hit: Hit): SketchModel | null {
  if (!hit) return null;
  return addRing(m, ring, hit.kind === "atom" ? hit : { kind: "bond", id: hit.id }).model;
}

/** An atom label typed in the label editor (ChemDraw: Enter on an atom): an element symbol,
 *  case-insensitive, with an optional charge ("N+", "O-", "Fe3+", "S2-"); null when it is not
 *  one. */
export function parseAtomLabel(text: string): { el: string; charge: number } | null {
  const m = /^\s*([A-Za-z]{1,2})\s*(?:(\d?)([+-]))?\s*$/.exec(text.replace(/\u2212/g, "-"));
  if (!m) return null;
  const el = m[1][0].toUpperCase() + m[1].slice(1).toLowerCase();
  if (!isElement(el)) return null;
  const charge = m[3] ? (m[3] === "+" ? 1 : -1) * Number(m[2] || 1) : 0;
  return { el, charge };
}

/** The atom `id` relabelled (element and charge). */
export function relabelAtom(m: SketchModel, id: number, label: { el: string; charge: number }): SketchModel {
  return setCharge(setElement(m, [id], label.el), id, label.charge);
}

/** The atoms a selection moves: its atoms plus the ends of its bonds. */
export function selectionAtoms(m: SketchModel, sel: Selection): Set<number> {
  const out = new Set(sel.atoms);
  for (const b of m.bonds) if (sel.bonds.has(b.id)) out.add(b.a).add(b.b);
  return out;
}

/** Everything inside a model-frame rectangle: atoms inside, bonds with both ends inside. */
export function selectInRect(m: SketchModel, inside: (p: Vec) => boolean): Selection {
  const atoms = new Set(m.atoms.filter((a) => inside([a.x, a.y])).map((a) => a.id));
  const bonds = new Set(m.bonds.filter((b) => atoms.has(b.a) && atoms.has(b.b)).map((b) => b.id));
  return { atoms, bonds };
}
