// The sketcher's editing state: undo/redo history of committed models, the drag preview, the
// selection, the hovered object, the current tool and element. Commits call onChange; a new
// `value` from the parent (by reference) is pushed into the history without calling it back.
import { useCallback, useEffect, useRef, useState } from "react";
import { emptySketch, type SketchModel } from "@ksmech/viz/sketch/model";
import { canRedo, canUndo, initHistory, pushHistory, redo, undo, type History } from "./history";
import { NO_SELECTION, type Hit, type Selection, type ToolName } from "./tools";

export interface SketchEditor {
  /** The model on screen: the drag preview if any, else the committed present. */
  model: SketchModel;
  committed: SketchModel;
  /** Bumps whenever a model comes from outside the drawing gestures (value prop, import). */
  externalVersion: number;
  commit(next: SketchModel, opts?: { external?: boolean }): void;
  setPreview(m: SketchModel | null): void;
  undo(): void;
  redo(): void;
  canUndo: boolean;
  canRedo: boolean;
  selection: Selection;
  setSelection(s: Selection): void;
  hover: Hit;
  setHover(h: Hit): void;
  tool: ToolName;
  setTool(t: ToolName): void;
  /** Back to the tool used before the current one (the Space key); false when there is none. */
  swapTool(): boolean;
  element: string;
  setElement(el: string): void;
}

/** Drop selected and hovered ids that no longer exist. */
function pruneSelection(s: Selection, m: SketchModel): Selection {
  const atoms = new Set(m.atoms.map((a) => a.id));
  const bonds = new Set(m.bonds.map((b) => b.id));
  const keepA = [...s.atoms].filter((id) => atoms.has(id));
  const keepB = [...s.bonds].filter((id) => bonds.has(id));
  if (keepA.length === s.atoms.size && keepB.length === s.bonds.size) return s;
  return { atoms: new Set(keepA), bonds: new Set(keepB) };
}

function hitExists(h: Hit, m: SketchModel): boolean {
  if (!h) return true;
  return h.kind === "atom" ? m.atoms.some((a) => a.id === h.id) : m.bonds.some((b) => b.id === h.id);
}

export function useSketchEditor(value: SketchModel | undefined, onChange?: (m: SketchModel) => void): SketchEditor {
  const [hist, setHist] = useState<History<SketchModel>>(() => initHistory(value ?? emptySketch()));
  const histRef = useRef(hist);
  const onChangeRef = useRef(onChange);
  useEffect(() => {
    onChangeRef.current = onChange;
  });
  const [preview, setPreview] = useState<SketchModel | null>(null);
  const [selection, setSelectionState] = useState<Selection>(NO_SELECTION);
  const [hover, setHover] = useState<Hit>(null);
  const [tools, setTools] = useState<{ tool: ToolName; previous: ToolName | null }>({ tool: "bond", previous: null });
  const [element, setElement] = useState("C");
  const [externalVersion, setExternalVersion] = useState(0);

  const install = useCallback((next: History<SketchModel>, notify: boolean) => {
    if (next === histRef.current) return;
    histRef.current = next;
    setHist(next);
    setPreview(null);
    setSelectionState((s) => pruneSelection(s, next.present));
    setHover((h) => (hitExists(h, next.present) ? h : null));
    if (notify) onChangeRef.current?.(next.present);
  }, []);

  const commit = useCallback(
    (next: SketchModel, opts: { external?: boolean } = {}) => {
      setPreview(null);
      install(pushHistory(histRef.current, next), true);
      if (opts.external) setExternalVersion((v) => v + 1);
    },
    [install],
  );

  useEffect(() => {
    if (!value || value === histRef.current.present) return;
    install(pushHistory(histRef.current, value), false);
    setExternalVersion((v) => v + 1);
  }, [value, install]);

  return {
    model: preview ?? hist.present,
    committed: hist.present,
    externalVersion,
    commit,
    setPreview,
    undo: useCallback(() => install(undo(histRef.current), true), [install]),
    redo: useCallback(() => install(redo(histRef.current), true), [install]),
    canUndo: canUndo(hist),
    canRedo: canRedo(hist),
    selection,
    setSelection: setSelectionState,
    hover,
    setHover,
    tool: tools.tool,
    setTool: useCallback((t: ToolName) => setTools((s) => (s.tool === t ? s : { tool: t, previous: s.tool })), []),
    swapTool: () => {
      if (!tools.previous) return false;
      setTools({ tool: tools.previous, previous: tools.tool });
      return true;
    },
    element,
    setElement,
  };
}
