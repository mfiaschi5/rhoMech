// The R editor of the Correct page as a pure reducer (node --test runs it): the matrix being
// edited, the arrows the arrow tool drew, the tool, the bond tool's first atom, the selected
// entries and the undo / redo history.
//
// Tools and their cycles (masked by the reactant structure, spec "Edit decoder" masks):
//   bond   click a bond (or a forming-bond overlay): delta 0 -> -1 -> +1 -> 0; -1 needs a bond,
//          +1 an order below 3. Click two atoms: the pair's delta cycles the same way (a
//          non-bonded pair: 0 <-> +1).
//   lone   click an atom: 0 -> -2 -> +2 -> 0; -2 needs a lone pair on the atom.
//   arrow  a drag compiles to R by rmatrix.compileArrow and is added (the arrow is kept for
//          drawing while it still adds up to R).
//   eraser click an atom (its lone entry and every bond entry at it), a bond (its entry) or an
//          arrow (its matrix is subtracted).
// Every change of the matrix is one undo step; the tool, the pending atom and the selection
// are not history.
import type { Tool } from "../predict/PoolCanvas";
import type { ArrowReadingOut, MatrixIO } from "../predict/v9types";
import type { BondContext, Entry } from "../predict/rmatrix";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { addMatrix, bondDelta, compileArrow, compileArrows, EMPTY_MATRIX, entryKey, loneDelta, MAX_BOND_ORDER, normalize, removeEntries, sameMatrix, setEntry } from "../predict/rmatrix.ts";

export const TOOLS: readonly Tool[] = ["bond", "lone", "arrow", "eraser"];
export const HISTORY_CAP = 200;

export interface Snapshot {
  matrix: MatrixIO;
  arrows: ArrowReadingOut[];
}

export interface EditorState extends Snapshot {
  tool: Tool;
  /** The bond tool's first clicked atom, waiting for the second. */
  pending: number | null;
  /** Selected entry keys (rmatrix.entryKey). */
  selection: ReadonlySet<string>;
  past: Snapshot[];
  future: Snapshot[];
  /** Why the last action did nothing (masked, invalid gesture); cleared by the next one. */
  notice: string | null;
}

export type EditorAction =
  | { type: "tool"; tool: Tool }
  | { type: "load"; matrix: MatrixIO; arrows?: ArrowReadingOut[] }
  | { type: "clickAtom"; i: number }
  | { type: "clickBond"; i: number; j: number }
  | { type: "clickArrow"; index: number }
  | { type: "arrow"; arrow: ArrowReadingOut }
  | { type: "notice"; message: string | null }
  | { type: "setEntry"; entry: Entry; value: number }
  | { type: "select"; keys: string[]; additive?: boolean }
  | { type: "deleteSelection" }
  | { type: "clear" }
  | { type: "undo" }
  | { type: "redo" };

export function initEditor(matrix: MatrixIO = EMPTY_MATRIX, arrows: ArrowReadingOut[] = [], tool: Tool = "bond"): EditorState {
  return { matrix: normalize(matrix), arrows, tool, pending: null, selection: new Set(), past: [], future: [], notice: null };
}

/** The next value of a bond cycle 0 -> -1 -> +1 -> 0 for a pair of reactant order `order`. */
export function nextBondDelta(current: number, order: number): number {
  const allowed = [0, -1, 1].filter((v) => v === 0 || (v === -1 ? order >= 1 : order + 1 <= MAX_BOND_ORDER));
  const at = allowed.indexOf(current);
  return allowed[(at + 1) % allowed.length] ?? 0;
}

/** The next value of a lone-pair cycle 0 -> -2 -> +2 -> 0 (lonePairs null: unknown, unmasked). */
export function nextLoneDelta(current: number, lonePairs: number | null): number {
  const allowed = [0, -2, 2].filter((v) => v !== -2 || lonePairs === null || lonePairs >= 1);
  const at = allowed.indexOf(current);
  return allowed[(at + 1) % allowed.length] ?? 0;
}

/** Arrows that still add up to R are kept for drawing; otherwise none are (the server reads R). */
function keptArrows(matrix: MatrixIO, arrows: ArrowReadingOut[]): ArrowReadingOut[] {
  const c = compileArrows(arrows);
  return c.ok && sameMatrix(c.matrix, matrix) ? arrows : [];
}

/** Commit a new snapshot as one undo step (no-op when nothing changed). */
function commit(s: EditorState, matrix: MatrixIO, arrows: ArrowReadingOut[], notice: string | null = null): EditorState {
  const m = normalize(matrix);
  const kept = keptArrows(m, arrows);
  if (sameMatrix(m, s.matrix) && JSON.stringify(kept) === JSON.stringify(s.arrows)) return { ...s, pending: null, notice };
  const past = [...s.past, { matrix: s.matrix, arrows: s.arrows }].slice(-HISTORY_CAP);
  return { ...s, matrix: m, arrows: kept, past, future: [], pending: null, notice };
}

function cycleBond(s: EditorState, ctx: BondContext | null, i: number, j: number): EditorState {
  if (i === j) return { ...s, pending: null };
  const order = ctx?.bondOrder(i, j) ?? 1;
  const cur = bondDelta(s.matrix, i, j);
  return commit(s, setEntry(s.matrix, { kind: "bond", i, j, delta: 0 }, nextBondDelta(cur, order)), s.arrows);
}

function cycleLone(s: EditorState, ctx: BondContext | null, i: number): EditorState {
  const cur = loneDelta(s.matrix, i);
  const lp = ctx?.lonePairs(i) ?? null;
  const next = nextLoneDelta(cur, lp);
  const notice = lp === 0 && next !== -2 && cur === 0 ? `atom ${i} has no lone pair: only +2` : null;
  return commit(s, setEntry(s.matrix, { kind: "lone", i, delta: 0 }, next), s.arrows, notice);
}

function eraseAtom(s: EditorState, i: number): EditorState {
  const m = normalize({ bonds: s.matrix.bonds.filter(([a, b]) => a !== i && b !== i), lone: s.matrix.lone.filter(([a]) => a !== i) });
  return commit(s, m, s.arrows);
}

function clickAtom(s: EditorState, ctx: BondContext | null, i: number): EditorState {
  if (s.tool === "lone") return cycleLone(s, ctx, i);
  if (s.tool === "eraser") return eraseAtom(s, i);
  if (s.tool !== "bond") return s;
  if (s.pending === null) return { ...s, pending: i, notice: null };
  if (s.pending === i) return { ...s, pending: null, notice: null };
  return cycleBond(s, ctx, s.pending, i);
}

function clickBond(s: EditorState, ctx: BondContext | null, i: number, j: number): EditorState {
  if (s.tool === "bond") return cycleBond(s, ctx, i, j);
  if (s.tool === "eraser") return commit(s, setEntry(s.matrix, { kind: "bond", i, j, delta: 0 }, 0), s.arrows);
  return s;
}

function eraseArrow(s: EditorState, index: number): EditorState {
  const a = s.arrows[index];
  if (!a || s.tool !== "eraser") return s;
  const c = compileArrow(a);
  if (!c.ok) return s;
  return commit(s, addMatrix(s.matrix, c.matrix, -1), s.arrows.filter((_, k) => k !== index));
}

function addArrow(s: EditorState, arrow: ArrowReadingOut): EditorState {
  const c = compileArrow(arrow);
  if (!c.ok) return { ...s, notice: c.error };
  return commit(s, addMatrix(s.matrix, c.matrix), [...s.arrows, arrow]);
}

function undoRedo(s: EditorState, dir: "undo" | "redo"): EditorState {
  const from = dir === "undo" ? s.past : s.future;
  const snap = from[from.length - 1];
  if (!snap) return s;
  const cur = { matrix: s.matrix, arrows: s.arrows };
  const rest = from.slice(0, -1);
  const other = [...(dir === "undo" ? s.future : s.past), cur];
  return {
    ...s,
    ...snap,
    past: dir === "undo" ? rest : other,
    future: dir === "undo" ? other : rest,
    pending: null,
    selection: new Set(),
    notice: null,
  };
}

function select(s: EditorState, keys: string[], additive: boolean): EditorState {
  const next = new Set(additive ? s.selection : []);
  for (const k of keys) {
    if (additive && next.has(k)) next.delete(k);
    else next.add(k);
  }
  return { ...s, selection: next };
}

/** The editor reducer; `ctx` is the reactant structure the masks read (null: unmasked). */
export function editorReducer(s: EditorState, a: EditorAction, ctx: BondContext | null = null): EditorState {
  switch (a.type) {
    case "tool":
      return { ...s, tool: a.tool, pending: null, notice: null };
    case "load":
      return initEditor(a.matrix, a.arrows ?? [], s.tool);
    case "clickAtom":
      return clickAtom(s, ctx, a.i);
    case "clickBond":
      return clickBond(s, ctx, a.i, a.j);
    case "clickArrow":
      return eraseArrow(s, a.index);
    case "arrow":
      return addArrow(s, a.arrow);
    case "notice":
      return { ...s, notice: a.message };
    case "setEntry":
      return commit(s, setEntry(s.matrix, a.entry, a.value), s.arrows);
    case "select":
      return select(s, a.keys, !!a.additive);
    case "deleteSelection":
      return s.selection.size ? { ...commit(s, removeEntries(s.matrix, s.selection), s.arrows), selection: new Set() } : s;
    case "clear":
      return { ...commit(s, EMPTY_MATRIX, []), selection: new Set() };
    case "undo":
    case "redo":
      return undoRedo(s, a.type);
  }
}

export const canUndo = (s: EditorState) => s.past.length > 0;
export const canRedo = (s: EditorState) => s.future.length > 0;

/** The selection keys that still name an entry of R. */
export function liveSelection(s: EditorState): string[] {
  const keys = new Set([...s.matrix.bonds.map(([i, j, delta]) => entryKey({ kind: "bond", i, j, delta })), ...s.matrix.lone.map(([i, delta]) => entryKey({ kind: "lone", i, delta }))]);
  return [...s.selection].filter((k) => keys.has(k));
}

const TOOL_KEYS: Record<string, Tool> = { b: "bond", l: "lone", a: "arrow", e: "eraser" };

/** The editor's keyboard (useEditorKeys.ts): b / l / a / e tools, Ctrl+Z undo, Ctrl+Y and
 *  Ctrl+Shift+Z redo, Delete / Backspace the selection, Enter "apply"; null for other keys. */
export function keyAction(e: { key: string; ctrlKey: boolean; metaKey: boolean; shiftKey: boolean; altKey: boolean }): EditorAction | "apply" | null {
  const mod = e.ctrlKey || e.metaKey;
  const k = e.key.toLowerCase();
  if (mod && k === "z") return { type: e.shiftKey ? "redo" : "undo" };
  if (mod && k === "y") return { type: "redo" };
  if (mod || e.altKey) return null;
  if (TOOL_KEYS[k]) return { type: "tool", tool: TOOL_KEYS[k] };
  if (e.key === "Delete" || e.key === "Backspace") return { type: "deleteSelection" };
  if (e.key === "Enter") return "apply";
  return null;
}
