// The sketcher's window listeners: keydown (the key map of shortcuts.ts), paste (text, or an
// image to OCR: imageImport.pickPaste), copy and cut (the selection or the whole drawing as a
// molfile, clipboard.ts). They are live while the pointer is over the sketcher or focus is inside
// it, never while typing in a text field; the handlers read the latest state through a ref.
import { useEffect, useLayoutEffect, useRef, type RefObject } from "react";
import { applyKey, applyRingKey, NO_SELECTION, ringKey } from "./tools";
import { transformKey, type TransformAction } from "./transform";
import { duplicateSelection, isEmptySelection, partOf } from "./clipboard";
import { pickPaste } from "./imageImport";
import type { SketchEditor } from "./useSketchEditor";
import type { SketchPointer } from "./useSketchPointer";
import type { LayoutActions } from "./useLayoutActions";
import type { SketchModel } from "@ksmech/viz/sketch/model";

export interface KeyDeps {
  ed: SketchEditor;
  pointer: SketchPointer;
  rootRef: RefObject<HTMLDivElement | null>;
  readOnly: boolean;
  closePanel(): void;
  /** The "?" key: the shortcut cheat sheet. */
  toggleKeys(): void;
  importText(text: string): string | null;
  importImage(file: File): void;
  layout: LayoutActions;
  transform(a: TransformAction): void;
  /** Enter on a hovered atom: the label editor. */
  editLabel(atomId: number): void;
  /** The clipboard texts of a part of the drawing. */
  clipText(part: SketchModel): { molfile: string; smiles: string };
}

function isTextInput(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  return t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName);
}

/** Text selected on the page (a copy then copies it, not the drawing). */
const pageSelection = () => (window.getSelection?.()?.toString() ?? "").length > 0;

export function useSketchKeys(deps: KeyDeps) {
  const latest = useRef(deps);
  useLayoutEffect(() => {
    latest.current = deps;
  });
  useEffect(() => {
    const live = (e: Event) => {
      const { pointer, rootRef, readOnly } = latest.current;
      const focused = rootRef.current?.contains(document.activeElement) ?? false;
      return !readOnly && (pointer.inside.current || focused) && !isTextInput(e.target);
    };
    const onKey = (e: KeyboardEvent) => {
      if (live(e) && handleKey(e, latest.current)) e.preventDefault();
    };
    const onPaste = (e: ClipboardEvent) => {
      if (!live(e)) return;
      const pick = pickPaste(e.clipboardData);
      if (!pick) return;
      e.preventDefault();
      if (pick.kind === "image") return latest.current.importImage(pick.file);
      const err = latest.current.importText(pick.text);
      if (err) console.warn("sketch paste:", err);
    };
    const onCopy = (cut: boolean) => (e: ClipboardEvent) => {
      if (!live(e) || pageSelection() || !e.clipboardData) return;
      if (copyPart(e.clipboardData, latest.current, cut)) e.preventDefault();
    };
    const copy = onCopy(false);
    const cut = onCopy(true);
    window.addEventListener("keydown", onKey);
    window.addEventListener("paste", onPaste);
    window.addEventListener("copy", copy);
    window.addEventListener("cut", cut);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("paste", onPaste);
      window.removeEventListener("copy", copy);
      window.removeEventListener("cut", cut);
    };
  }, []);
}

/** Copy (and for a cut, erase) the selection or the whole drawing; false on an empty one. */
function copyPart(data: DataTransfer, { ed, clipText }: KeyDeps, cut: boolean): boolean {
  const part = partOf(ed.committed, ed.selection);
  if (!part.atoms.length) return false;
  const { molfile, smiles } = clipText(part);
  data.setData("text/plain", molfile);
  if (smiles) data.setData("chemical/x-daylight-smiles", smiles);
  if (cut) {
    const sel = ed.selection;
    ed.commit(isEmptySelection(sel) ? { atoms: [], bonds: [], nextId: ed.committed.nextId } : partRemoved(ed.committed, part));
    ed.setSelection(NO_SELECTION);
  }
  return true;
}

function partRemoved(m: SketchModel, part: SketchModel): SketchModel {
  const ids = new Set(part.atoms.map((a) => a.id));
  return { ...m, atoms: m.atoms.filter((a) => !ids.has(a.id)), bonds: m.bonds.filter((b) => !ids.has(b.a) && !ids.has(b.b)) };
}

/** Ctrl / Cmd shortcuts; true when handled. Ctrl+C / X / V are left to the copy, cut and paste
 *  events. */
function modKey(e: KeyboardEvent, { ed, layout }: KeyDeps): boolean {
  const k = e.key.toLowerCase();
  if (k === "z") {
    if (e.shiftKey) ed.redo();
    else ed.undo();
    return true;
  }
  if (k === "y") {
    ed.redo();
    return true;
  }
  if (k === "a" && !e.shiftKey) {
    ed.setSelection({ atoms: new Set(ed.committed.atoms.map((a) => a.id)), bonds: new Set(ed.committed.bonds.map((b) => b.id)) });
    return true;
  }
  if (k === "d" && !e.shiftKey) {
    const dup = duplicateSelection(ed.committed, ed.selection);
    if (dup) {
      ed.commit(dup.model);
      ed.setSelection(dup.selection);
    }
    return true;
  }
  if (e.code === "KeyK" && e.shiftKey) {
    if (layout.canClean) void layout.clean();
    return true;
  }
  return false;
}

/** Shift+D / Shift+X: 3D -> 2D, crossing breaks (not element keys). */
function layoutKey(k: string, layout: LayoutActions): boolean {
  if (k === "d") {
    if (layout.canProject) void layout.project();
    return true;
  }
  if (k === "x") {
    layout.toggleBreaks();
    return true;
  }
  return false;
}

/** Keys that are no model edit: Escape, "?", Enter (label editor), Space off any atom (back to
 *  the previous tool); true when handled. */
function uiKey(e: KeyboardEvent, deps: KeyDeps): boolean {
  const { ed, pointer, closePanel } = deps;
  if (e.key === "Escape") {
    pointer.cancel();
    ed.setSelection(NO_SELECTION);
    closePanel();
    return true;
  }
  if (e.key === "?") {
    deps.toggleKeys();
    return true;
  }
  if (e.key === "Enter" && ed.hover?.kind === "atom") {
    deps.editLabel(ed.hover.id);
    return true;
  }
  if (e.key === " " && ed.hover?.kind !== "atom") return ed.swapTool() || ed.hover === null;
  return false;
}

/** One keydown; true when it was handled. */
export function handleKey(e: KeyboardEvent, deps: KeyDeps): boolean {
  const { ed, layout, transform } = deps;
  const mod = e.ctrlKey || e.metaKey;
  const tf = transformKey(e);
  if (tf) {
    transform(tf);
    return true;
  }
  if (mod && !e.altKey) return modKey(e, deps);
  if (e.altKey) return false;
  const ring = ringKey(e);
  if (ring) {
    const next = applyRingKey(ed.committed, ring, ed.hover);
    if (next && next !== ed.committed) ed.commit(next);
    return next !== null;
  }
  if (e.shiftKey && layoutKey(e.key.toLowerCase(), layout)) return true;
  if (uiKey(e, deps)) return true;
  const next = applyKey(ed.committed, e.key, ed.hover, ed.selection);
  if (!next) return false;
  if (next !== ed.committed) ed.commit(next);
  return true;
}
