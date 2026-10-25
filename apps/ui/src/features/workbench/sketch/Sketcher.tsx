// Sketcher: a ChemDraw-like 2D molecule editor in SVG (workbench phase 4). Self-contained: its
// own immutable model (model.ts), undo/redo, tools, hotkeys, RDKit validation, clean-up,
// import and export.
//
// Props:
//   value          SketchModel   optional; a new value (by reference) replaces the drawing as an
//                                undoable step, without calling onChange back
//   onChange(m)                  every committed edit (gesture end, key, undo/redo, import, clean)
//   highlightAtoms number[]      sketch atom ids drawn with the highlight halo
//   readOnly       boolean       no toolbar, no edits (hover, pan and zoom still work)
//   height         number        canvas height in px (default 360); the width follows the parent
//   projection     ProjectionSource  optional; the 3D -> 2D projection (useLayoutActions.ts):
//                                why it is off, or the drawn atoms in the 3D view's frame
//
// DOM: <div data-testid="sketch" data-smiles (RDKit canonical, "" when invalid/empty)
//      data-valid data-atoms data-bonds data-tool data-status (useSketchInfo status)
//      data-crossings (bond crossings in 2D) data-break-crossings data-has-depth>
//   the toolbar (Toolbar.tsx), <svg data-testid="sketch-canvas"> (SketchCanvas.tsx), the
//   import/export popovers (ImportExport.tsx) and <div data-testid="sketch-info"> (InfoBar.tsx).
//
// Keys (while the pointer is over the sketcher or focus is inside it; never while
// typing in a text field), on the hovered atom / bond, else on the selection:
//   c n o s f p, l=Cl, b=Br, i=I, h=H   element of the atom
//   1 2 3                               bond order (on an atom: grow a bond of that order)
//   + (or =)  -                         charge
//   Delete / Backspace                  erase
//   Ctrl+Z   Ctrl+Shift+Z / Ctrl+Y      undo / redo
//   Ctrl+A                              select all
//   Ctrl+V                              import the clipboard (SMILES or molfile)
//   Shift+D  Shift+G  Shift+X           3D -> 2D projection, fix layout, crossing breaks
//   Esc                                 cancel the gesture, clear the selection, close popovers
//
// Demo / test harness (not a page; for a scratch mount or a Playwright test):
//   const [m, setM] = useState<SketchModel>();
//   <Sketcher value={m} onChange={setM} highlightAtoms={[1]} height={400} />
//   e2e: click tool-bond, click the canvas, click sketch-atom-2, hover the third atom, press
//   "o"; expect sketch[data-smiles="CCO"].
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type DragEvent, type RefObject } from "react";
import { useRDKit, type RDKitModule } from "../../../chem/rdkit";
import { setElement, type SketchModel } from "./model";
import { fitView, type Vec, type View } from "./geometry";
import { applyKey, NO_SELECTION } from "./tools";
import { cleanUpSketch, exportSketch, importSketch } from "./rdkitSketch";
import { useSketchEditor, type SketchEditor } from "./useSketchEditor";
import { useSketchInfo } from "./useSketchInfo";
import { useSketchPointer, type SketchPointer } from "./useSketchPointer";
import { SketchCanvas } from "./SketchCanvas";
import { Toolbar, type SketchPanel } from "./Toolbar";
import { ExportPanel, ImportPanel, readDroppedText } from "./ImportExport";
import { InfoBar } from "./InfoBar";
import { useLayoutActions, type LayoutActions, type ProjectionSource } from "./useLayoutActions";
import { findCrossings } from "./crossings";
import { hasDepth } from "./model";
import "./sketch.css";

export interface SketcherProps {
  value?: SketchModel;
  onChange?(m: SketchModel): void;
  highlightAtoms?: readonly number[];
  readOnly?: boolean;
  height?: number;
  projection?: ProjectionSource;
}

/** Pixels per model unit at 100 %: a 1.5 bond is 37.5 px. */
export const DEFAULT_SCALE = 25;

function isTextInput(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  return t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName);
}

function useWidth(ref: RefObject<HTMLElement | null>): number {
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setW(el.getBoundingClientRect().width);
    const ro = new ResizeObserver((entries) => setW(entries[0]?.contentRect.width ?? 0));
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

interface KeyDeps {
  ed: SketchEditor;
  pointer: SketchPointer;
  rootRef: RefObject<HTMLDivElement | null>;
  readOnly: boolean;
  closePanel(): void;
  importText(text: string): string | null;
  layout: LayoutActions;
}

/** Window keydown/paste listeners, live while the pointer is over the sketcher or focus is in the
 *  sketcher; the handler is read from a ref so it always sees the latest state. */
function useSketchKeys(deps: KeyDeps) {
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
      if (!live(e)) return;
      if (handleKey(e, latest.current)) e.preventDefault();
    };
    const onPaste = (e: ClipboardEvent) => {
      const text = e.clipboardData?.getData("text") ?? "";
      if (!live(e) || !text.trim()) return;
      e.preventDefault();
      const err = latest.current.importText(text);
      if (err) console.warn("sketch paste:", err);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("paste", onPaste);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("paste", onPaste);
    };
  }, []);
}

/** One keydown; true when it was handled. */
function handleKey(e: KeyboardEvent, { ed, pointer, closePanel, layout }: KeyDeps): boolean {
  const mod = e.ctrlKey || e.metaKey;
  const k = e.key.toLowerCase();
  if (e.shiftKey && !mod && !e.altKey && layoutKey(k, layout)) return true;
  if (mod && k === "z") {
    if (e.shiftKey) ed.redo();
    else ed.undo();
    return true;
  }
  if (mod && k === "y") {
    ed.redo();
    return true;
  }
  if (mod && k === "a") {
    ed.setSelection({ atoms: new Set(ed.committed.atoms.map((a) => a.id)), bonds: new Set(ed.committed.bonds.map((b) => b.id)) });
    return true;
  }
  if (e.key === "Escape") {
    pointer.cancel();
    ed.setSelection(NO_SELECTION);
    closePanel();
    return true;
  }
  if (mod || e.altKey) return false;
  const next = applyKey(ed.committed, e.key, ed.hover, ed.selection);
  if (!next) return false;
  if (next !== ed.committed) ed.commit(next);
  return true;
}

/** Shift+D / Shift+G / Shift+X: projection, fix layout, crossing breaks (not element keys). */
function layoutKey(k: string, layout: LayoutActions): boolean {
  if (k === "d") {
    if (layout.canProject) layout.project();
    return true;
  }
  if (k === "g") {
    layout.fix();
    return true;
  }
  if (k === "x") {
    layout.toggleBreaks();
    return true;
  }
  return false;
}

interface ViewState {
  view: View;
  setView(v: View): void;
  fit(): void;
}

/** The screen transform: centred on the origin until the user pans, zooms or fits; refitted when
 *  a drawing arrives from outside (value prop, import, clean-up) and once the width is known. */
function useView(ed: SketchEditor, width: number, height: number): ViewState {
  const [viewState, setView] = useState<View | null>(null);
  const view: View = viewState ?? { s: DEFAULT_SCALE, ox: width / 2, oy: height / 2 };
  const fit = () => {
    if (width > 0) setView(fitView(ed.committed.atoms.map((a): Vec => [a.x, a.y]), width, height, DEFAULT_SCALE));
  };
  const measured = width > 0;
  useEffect(() => {
    if (measured && ed.committed.atoms.length) fit();
    // refit only on those two events, not on every edit
  }, [ed.externalVersion, measured]);
  return { view, setView, fit };
}

/** The toolbar and drop actions that need RDKit. */
function useSketchActions(ed: SketchEditor, rdkit: RDKitModule | null, readOnly: boolean) {
  const importText = (text: string): string | null => {
    try {
      ed.commit(importSketch(rdkit, text), { external: true });
      return null;
    } catch (err) {
      return err instanceof Error ? err.message : String(err);
    }
  };
  const clean = () => {
    const m = rdkit ? cleanUpSketch(rdkit, ed.committed) : null;
    if (m) ed.commit(m, { external: true });
  };
  const onElement = (el: string) => {
    ed.setElement(el);
    ed.setTool("atom");
    if (ed.selection.atoms.size) ed.commit(setElement(ed.committed, ed.selection.atoms, el));
  };
  const onDrop = async (e: DragEvent<HTMLDivElement>) => {
    if (readOnly) return;
    e.preventDefault();
    const text = await readDroppedText(e);
    const err = text ? importText(text) : null;
    if (err) console.warn("sketch drop:", err);
  };
  const clear = () => ed.commit({ atoms: [], bonds: [], nextId: ed.committed.nextId });
  return { importText, clean, onElement, onDrop, clear };
}

type SketchActions = ReturnType<typeof useSketchActions>;

function EditorToolbar(p: {
  ed: SketchEditor;
  act: SketchActions;
  layout: LayoutActions;
  projection: ProjectionSource | undefined;
  canClean: boolean;
  onFit(): void;
  panel: SketchPanel;
  onPanel(p: SketchPanel): void;
}) {
  const { ed, act, layout } = p;
  return (
    <Toolbar
      tool={ed.tool}
      onTool={ed.setTool}
      element={ed.element}
      onElement={act.onElement}
      onUndo={ed.undo}
      onRedo={ed.redo}
      canUndo={ed.canUndo}
      canRedo={ed.canRedo}
      onClean={act.clean}
      canClean={p.canClean}
      onFit={p.onFit}
      onClear={act.clear}
      projection={p.projection ? { enabled: layout.canProject, reason: p.projection.disabledReason, onProject: layout.project } : null}
      onFix={layout.fix}
      canFix={ed.committed.atoms.length > 1}
      breakCrossings={ed.committed.breakCrossings ?? false}
      onToggleBreaks={layout.toggleBreaks}
      panel={p.panel}
      onPanel={p.onPanel}
    />
  );
}

function ExportFor({ rdkit, model, onClose }: { rdkit: RDKitModule | null; model: SketchModel; onClose(): void }) {
  const out = useMemo(
    () => ({
      smiles: exportSketch(rdkit, model, "smiles"),
      inchi: exportSketch(rdkit, model, "inchi"),
      molfile: exportSketch(rdkit, model, "molfile"),
    }),
    [rdkit, model],
  );
  return <ExportPanel {...out} onClose={onClose} />;
}

export function Sketcher({ value, onChange, highlightAtoms, readOnly = false, height = 360, projection }: SketcherProps) {
  const ed = useSketchEditor(value, onChange);
  const info = useSketchInfo(ed.committed);
  const { rdkit } = useRDKit();
  const rootRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const { view, setView, fit } = useView(ed, useWidth(stageRef), height);
  const [panel, setPanel] = useState<SketchPanel>(null);
  const pointer = useSketchPointer(ed, svgRef, view, setView, readOnly);
  const act = useSketchActions(ed, rdkit, readOnly);
  const fresh = !info.stale && ed.model === ed.committed && info.valid;
  const layout = useLayoutActions(ed, rdkit, projection, fresh ? info.hCounts : null);
  const closePanel = () => setPanel(null);
  useSketchKeys({ ed, pointer, rootRef, readOnly, closePanel, importText: act.importText, layout });
  const highlight = useMemo(() => new Set(highlightAtoms ?? []), [highlightAtoms]);
  const invalid = useMemo(() => new Set(info.badAtoms), [info.badAtoms]);
  const crossings = useMemo(() => findCrossings(ed.committed).length, [ed.committed]);
  const setInside = (v: boolean) => () => {
    pointer.inside.current = v;
  };
  return (
    <div
      ref={rootRef}
      className="sketch"
      data-testid="sketch"
      data-smiles={info.smiles}
      data-valid={info.valid}
      data-atoms={ed.committed.atoms.length}
      data-bonds={ed.committed.bonds.length}
      data-tool={ed.tool}
      data-status={info.status}
      data-readonly={readOnly}
      data-crossings={crossings}
      data-break-crossings={ed.committed.breakCrossings ? "true" : "false"}
      data-has-depth={hasDepth(ed.committed) ? "true" : "false"}
      onPointerEnter={setInside(true)}
      onPointerLeave={setInside(false)}
    >
      {!readOnly && (
        <EditorToolbar ed={ed} act={act} layout={layout} projection={projection} canClean={!!rdkit && ed.committed.atoms.length > 1} onFit={fit} panel={panel} onPanel={setPanel} />
      )}
      <div className="sk-stage" ref={stageRef} onDragOver={(e) => e.preventDefault()} onDrop={act.onDrop}>
        <SketchCanvas
          model={ed.model}
          view={view}
          hCounts={fresh ? info.hCounts : null}
          hover={ed.hover}
          selection={ed.selection}
          highlight={highlight}
          invalid={invalid}
          rubber={pointer.rubber}
          height={height}
          pointer={pointer}
          svgRef={svgRef}
          readOnly={readOnly}
        />
        {panel === "import" && <ImportPanel onImport={act.importText} onClose={closePanel} />}
        {panel === "export" && <ExportFor rdkit={rdkit} model={ed.committed} onClose={closePanel} />}
      </div>
      <InfoBar info={info} report={layout.report} />
    </div>
  );
}
