// The sketcher's whole key map as data: the "?" cheat sheet (KeyHelp.tsx) lists it, and the
// tests check that every entry names a key the handlers know. The handlers themselves are
// tools.applyKey / ringKey (hover keys), transform.transformKey and useSketchKeys.ts.
//
// `origin` says where a binding comes from:
//   chemdraw  a ChemDraw default (hotkeys cheat sheet, user guide or support notes);
//   standard  the usual editor key (copy, cut, undo, ...), also ChemDraw's;
//   chemrho   ChemRho's own (ChemDraw has none there, or ChemRho maps it differently: `note`).

export type ShortcutScope = "atom" | "bond" | "selection" | "drawing" | "drag";
export type ShortcutOrigin = "chemdraw" | "standard" | "chemrho";

export interface Shortcut {
  keys: string;
  action: string;
  scope: ShortcutScope;
  origin: ShortcutOrigin;
  /** What ChemDraw does instead (shown as "ChemDraw: <note>"). */
  note?: string;
}

export const SCOPE_LABEL: Record<ShortcutScope, string> = {
  atom: "On the atom under the pointer (else on the selected atoms)",
  bond: "On the bond under the pointer (else on the selected bonds)",
  selection: "On the selection, or on the whole drawing",
  drawing: "Anywhere in the sketcher",
  drag: "While dragging",
};

export const SHORTCUTS: readonly Shortcut[] = [
  { keys: "C N O S F P", action: "element", scope: "atom", origin: "chemdraw" },
  { keys: "L  B  I  H", action: "Cl, Br, I, H", scope: "atom", origin: "chemdraw" },
  { keys: "Enter", action: "type an atom label (element, charge: N+, O-, Fe3+)", scope: "atom", origin: "chemdraw" },
  { keys: "Space", action: "plain carbon (removes the label)", scope: "atom", origin: "chemdraw" },
  { keys: "+  -", action: "charge +1 / -1", scope: "atom", origin: "chemrho" },
  { keys: "1  2  3", action: "grow a bond of that order", scope: "atom", origin: "chemrho", note: "adds n-Bu, s-Bu, t-Bu (this sketcher has no abbreviations)" },
  { keys: ".", action: "radical: none, one dot, two dots", scope: "atom", origin: "chemrho", note: "attachment point" },
  { keys: "Shift+3 … Shift+8", action: "cycloalkane ring of that size (spiro on an atom, fused on a bond)", scope: "atom", origin: "chemrho" },
  { keys: "Shift+A", action: "benzene ring (spiro on an atom, fused on a bond)", scope: "atom", origin: "chemrho" },
  { keys: "1  2  3", action: "single, double, triple", scope: "bond", origin: "chemdraw" },
  { keys: "W", action: "wedge (again: flip the narrow end)", scope: "bond", origin: "chemdraw" },
  { keys: "H", action: "hashed wedge (again: flip)", scope: "bond", origin: "chemdraw" },
  { keys: "Delete  Backspace", action: "erase", scope: "atom", origin: "standard" },
  { keys: "Ctrl+C  Ctrl+X", action: "copy / cut as molfile (pastes into ChemDraw) and SMILES", scope: "selection", origin: "standard" },
  { keys: "Ctrl+V", action: "paste a SMILES, a molfile or an image of a structure (OCR)", scope: "drawing", origin: "standard" },
  { keys: "Ctrl+D", action: "duplicate beside the drawing", scope: "selection", origin: "chemdraw" },
  { keys: "Ctrl+A", action: "select all", scope: "drawing", origin: "standard" },
  { keys: "Ctrl+Z  Ctrl+Shift+Z / Ctrl+Y", action: "undo / redo", scope: "drawing", origin: "standard" },
  { keys: "Ctrl+Shift+K", action: "Clean structure (server 2D layout)", scope: "drawing", origin: "chemdraw" },
  { keys: "Ctrl+Shift+H  Ctrl+Shift+V", action: "flip horizontal / vertical (same stereoisomer)", scope: "selection", origin: "chemdraw" },
  { keys: "Alt+H  Alt+V", action: "flip horizontal / vertical", scope: "selection", origin: "chemrho" },
  { keys: "R  Shift+R", action: "rotate 15° clockwise / counter-clockwise", scope: "selection", origin: "chemrho" },
  { keys: "Ctrl+=  Ctrl+-", action: "scale up / down 10 %", scope: "selection", origin: "chemrho" },
  { keys: "Shift+D  Shift+X", action: "3D → 2D, break crossings", scope: "drawing", origin: "chemrho" },
  { keys: "Space", action: "back to the previous tool (nothing under the pointer)", scope: "drawing", origin: "chemrho" },
  { keys: "?", action: "this list", scope: "drawing", origin: "chemrho", note: "bond properties" },
  { keys: "Esc", action: "cancel, clear the selection, close panels", scope: "drawing", origin: "standard" },
  { keys: "drag", action: "fixed length (one standard bond) and fixed angles (30° steps), toggles in the toolbar", scope: "drag", origin: "chemdraw", note: "15° angle steps" },
  { keys: "Alt or Shift + drag", action: "free length and angle for this drag", scope: "drag", origin: "chemdraw" },
  { keys: "Alt + drag on empty canvas, middle drag", action: "pan", scope: "drag", origin: "chemrho" },
  { keys: "Ctrl + wheel", action: "zoom", scope: "drag", origin: "standard" },
];

/** The shortcuts of one scope, in table order. */
export const shortcutsOf = (scope: ShortcutScope): Shortcut[] => SHORTCUTS.filter((s) => s.scope === scope);
