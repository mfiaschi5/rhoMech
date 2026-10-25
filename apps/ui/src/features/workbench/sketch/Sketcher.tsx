// Sketcher: a ChemDraw-like 2D molecule editor in SVG (workbench phase 4). Self-contained: its
// own immutable model (model.ts), undo/redo, tools, hotkeys, RDKit validation, import and
// export; the 2D layouts (Clean, 3D->2D) come from the server through the page's Depictor.
//
// Props:
//   value          SketchModel   optional; a new value (by reference) replaces the drawing as an
//                                undoable step, without calling onChange back
//   onChange(m)                  every committed edit (gesture end, key, undo/redo, import, clean)
//   highlightAtoms number[]      sketch atom ids drawn with the highlight halo
//   readOnly       boolean       no toolbar, no edits (hover, pan and zoom still work)
//   height         number        canvas height in px (default 360); the width follows the parent
//   projection     ProjectionSource  optional; 3D -> 2D (useLayoutActions.ts): why it is off,
//                                or the computed molecule's id and the 3D viewer's quaternion
//   depictor       Depictor      optional; POST /api/depict2d (depiction.ts). Clean uses it with
//                                the style picked next to the button (textbook | perspective,
//                                kept per tab in sessionStorage); without it, or when the server
//                                fails, Clean is RDKit.js's CoordGen layout with a notice
//   display        SketchDisplay optional; label font size, bond line width, atom colouring
//                                and label style (the page's preferences), the drawing style
//                                (styleId, customStyle; style.ts)
//   onStyle(id, custom)          optional; a style picked or edited in the toolbar (the page
//                                persists it); without it the Sketcher keeps the style itself
//
// Drawing styles (style.ts, StyleControls.tsx): Screen, ACS Document 1996, Total synthesis or
// a custom style, render-only (the model, its coordinates, stereo and history never change);
// the zoom maps points to pixels for the publication styles. The export panel's image section
// (ImageExport.tsx) writes SVG / PNG / PDF at the style's true size, copies the image, the SVG,
// a LaTeX figure or chemfig code; the drag handle in the canvas corner carries the files.
//
// Labels are placed collision-aware (labelLayout.ts). Separate fragments (salts, counter-ions)
// that come too close are spread apart by "tidy fragments" (fragments.ts): after every Clean as
// its own undoable step, and on the Tidy button.
//
// The R/S toggle (kept per tab in sessionStorage) draws RDKit's CIP labels by the stereocentres
// and double bonds and lists them in the info bar; they follow every edit (wedges, rotations,
// flips: a flip keeps the stereoisomer, so the labels stay). Radicals (the radical and diradical
// tools, the "." key) are drawn as dots and reach RDKit as M  RAD.
//
// DOM: <div data-testid="sketch" data-smiles (RDKit canonical, "" when invalid/empty)
//      data-valid data-atoms data-bonds data-tool data-status (useSketchInfo status)
//      data-radicals (unpaired electrons drawn) data-show-cip
//      data-crossings (bond crossings in 2D) data-break-crossings data-has-depth
//      data-busy (a layout is waiting for the server) data-clean-style data-style (the drawing
//      style id) data-fragments (connected fragments) data-fragments-touch (some fragments are
//      closer than the tidy clearance: the Tidy button is enabled)>
//   the toolbar (Toolbar.tsx, ActionBar.tsx), <svg data-testid="sketch-canvas">
//   (SketchCanvas.tsx; with two or more atoms selected, the selection frame with its rotate and
//   scale handles, frame.ts), the
//   import/export popovers (ImportExport.tsx) and <div data-testid="sketch-info"> (InfoBar.tsx).
//
// Keys (while the pointer is over the sketcher or focus is inside it; never while
// typing in a text field), on the hovered atom / bond, else on the selection:
//   c n o s f p, l=Cl, b=Br, i=I, h=H   element of the atom
//   1 2 3                               bond order (on an atom: grow a bond of that order)
//   + (or =)  -                         charge
//   .                                   radical: none -> one dot -> two dots -> none
//   Delete / Backspace                  erase
//   Ctrl+Z   Ctrl+Shift+Z / Ctrl+Y      undo / redo
//   Ctrl+A                              select all
//   Ctrl+V                              import the clipboard (SMILES or molfile)
//   Shift+D  Shift+X                    3D -> 2D, crossing breaks
//   R  Shift+R                          rotate 15 degrees clockwise / counter-clockwise
//   Alt+H  Alt+V                        flip horizontal / vertical (wedges swap: same isomer)
//   Ctrl+=  Ctrl+-                      scale up / down by 10 %
//                                       (transforms: the selection, else the whole drawing)
//   Esc                                 cancel the gesture, clear the selection, close popovers
//
// Demo / test harness (not a page; for a scratch mount or a Playwright test):
//   const [m, setM] = useState<SketchModel>();
//   <Sketcher value={m} onChange={setM} highlightAtoms={[1]} height={400} />
//   e2e: click tool-bond, click the canvas, click sketch-atom-2, hover the third atom, press
//   "o"; expect sketch[data-smiles="CCO"].
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type DragEvent, type ReactNode, type RefObject } from "react";
import { useRDKit, type RDKitModule } from "../../../chem/rdkit";
import { setElement, type SketchModel } from "./model";
import { fitView, type Vec, type View } from "./geometry";
import { applyKey, NO_SELECTION, selectionAtoms } from "./tools";
import { applyTransform, transformKey, transformTargets, type TransformAction } from "./transform";
import { selectionFrame } from "./frame";
import { exportSketch, importSketch } from "./rdkitSketch";
import { useSketchEditor, type SketchEditor } from "./useSketchEditor";
import { useSketchInfo } from "./useSketchInfo";
import { useSketchPointer, type SketchPointer } from "./useSketchPointer";
import { SketchCanvas } from "./SketchCanvas";
import { Toolbar, type SketchPanel } from "./Toolbar";
import { ExportPanel, ImportPanel, readDroppedText } from "./ImportExport";
import { InfoBar } from "./InfoBar";
import { useLayoutActions, type LayoutActions, type ProjectionSource } from "./useLayoutActions";
import { findCrossings } from "./crossings";
import { fragments, tidyFragments } from "./fragments";
import { hasDepth, radicalElectrons } from "./model";
import { cipItems } from "./cip";
import type { LabelStyle } from "./drawing";
import { actualSizeScale, drawMetrics, pxPerPt, resolveStyle, ACS_1996_STYLE, TOTAL_SYNTHESIS_STYLE, type DrawingStyle, type StyleId } from "./style";
import { setChainAngle } from "./geometry";
import type { StyleControlProps } from "./StyleControls";
import { DEFAULT_IMAGE_OPTIONS, SCHEME_DRAG_TYPE, DragHandle, ImageExportSection, useSchemeFiles, type ImageOptions, type SchemeFiles } from "./ImageExport";
import { restoreCleanStyle, saveCleanStyle, type CleanStyle, type Depictor } from "./depiction";
import "./sketch.css";

export interface SketcherProps {
  value?: SketchModel;
  onChange?(m: SketchModel): void;
  highlightAtoms?: readonly number[];
  readOnly?: boolean;
  height?: number;
  projection?: ProjectionSource;
  depictor?: Depictor;
  display?: SketchDisplay;
  onStyle?(id: StyleId, custom: DrawingStyle): void;
}

/** How the sketch draws (the page's preferences); every field optional. */
export interface SketchDisplay {
  fontPx?: number;
  bondWidth?: number;
  /** Element colours on the labels (default on). */
  colourAtoms?: boolean;
  labelStyle?: LabelStyle;
  /** The drawing style (default Screen) and the custom style. */
  styleId?: StyleId;
  customStyle?: DrawingStyle;
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
  transform(a: TransformAction): void;
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
function handleKey(e: KeyboardEvent, { ed, pointer, closePanel, layout, transform }: KeyDeps): boolean {
  const mod = e.ctrlKey || e.metaKey;
  const k = e.key.toLowerCase();
  const tf = transformKey(e);
  if (tf) {
    transform(tf);
    return true;
  }
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

interface ViewState {
  view: View;
  setView(v: View): void;
  fit(): void;
}

/** The screen transform: centred on the origin until the user pans, zooms or fits; refitted when
 *  a drawing arrives from outside (value prop, import, clean-up), once the width is known, and
 *  when the canvas changes size (a layout splitter or preset) while the view is still a fit, i.e.
 *  the user has not panned or zoomed since. */
function useView(ed: SketchEditor, width: number, height: number): ViewState {
  const [viewState, setViewState] = useState<View | null>(null);
  const fitted = useRef(false);
  const view: View = viewState ?? { s: DEFAULT_SCALE, ox: width / 2, oy: height / 2 };
  const fit = () => {
    if (width <= 0) return;
    fitted.current = true;
    setViewState(fitView(ed.committed.atoms.map((a): Vec => [a.x, a.y]), width, height, DEFAULT_SCALE));
  };
  const setView = (v: View) => {
    fitted.current = false;
    setViewState(v);
  };
  const measured = width > 0;
  useEffect(() => {
    if (measured && ed.committed.atoms.length) fit();
    // refit only on those two events, not on every edit
  }, [ed.externalVersion, measured]);
  useEffect(() => {
    if (fitted.current && ed.committed.atoms.length) fit();
  }, [width, height]);
  return { view, setView, fit };
}

/** The toolbar and drop actions that need RDKit (Clean is a layout action). */
function useSketchActions(ed: SketchEditor, rdkit: RDKitModule | null, readOnly: boolean) {
  const importText = (text: string): string | null => {
    try {
      ed.commit(importSketch(rdkit, text), { external: true });
      return null;
    } catch (err) {
      return err instanceof Error ? err.message : String(err);
    }
  };
  const onElement = (el: string) => {
    ed.setElement(el);
    ed.setTool("atom");
    if (ed.selection.atoms.size) ed.commit(setElement(ed.committed, ed.selection.atoms, el));
  };
  const onDrop = async (e: DragEvent<HTMLDivElement>) => {
    if (readOnly || e.dataTransfer.types.includes(SCHEME_DRAG_TYPE)) return;
    e.preventDefault();
    const text = await readDroppedText(e);
    const err = text ? importText(text) : null;
    if (err) console.warn("sketch drop:", err);
  };
  const clear = () => ed.commit({ atoms: [], bonds: [], nextId: ed.committed.nextId });
  /** One undoable step; coordinates (and, for a flip, wedges and depth) only. */
  const transform = (a: TransformAction) => {
    const next = applyTransform(ed.committed, transformTargets(ed.committed, ed.selection), a);
    if (next !== ed.committed) ed.commit(next);
  };
  return { importText, onElement, onDrop, clear, transform };
}

type SketchActions = ReturnType<typeof useSketchActions>;

function EditorToolbar(p: {
  ed: SketchEditor;
  act: SketchActions;
  layout: LayoutActions;
  projection: ProjectionSource | undefined;
  cleanStyle: CleanStyle;
  onCleanStyle(s: CleanStyle): void;
  onFit(): void;
  panel: SketchPanel;
  onPanel(p: SketchPanel): void;
  showCip: boolean;
  onToggleCip(): void;
  styleControls: StyleControlProps;
  canTidy: boolean;
  fragments: number;
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
      onClean={() => void layout.clean()}
      canClean={layout.canClean}
      onTidy={layout.tidy}
      canTidy={p.canTidy}
      fragments={p.fragments}
      cleanStyle={p.cleanStyle}
      onCleanStyle={p.onCleanStyle}
      busy={layout.busy}
      onFit={p.onFit}
      onClear={act.clear}
      projection={p.projection ? { enabled: layout.canProject, reason: p.projection.disabledReason, onProject: () => void layout.project() } : null}
      breakCrossings={ed.committed.breakCrossings ?? false}
      onToggleBreaks={layout.toggleBreaks}
      panel={p.panel}
      onPanel={p.onPanel}
      onTransform={act.transform}
      hasSelection={ed.selection.atoms.size + ed.selection.bonds.size > 0}
      showCip={p.showCip}
      onToggleCip={p.onToggleCip}
      styleControls={p.styleControls}
    />
  );
}

function ExportFor({ rdkit, model, onClose, image }: { rdkit: RDKitModule | null; model: SketchModel; onClose(): void; image: ReactNode }) {
  const out = useMemo(
    () => ({
      smiles: exportSketch(rdkit, model, "smiles"),
      inchi: exportSketch(rdkit, model, "inchi"),
      molfile: exportSketch(rdkit, model, "molfile"),
    }),
    [rdkit, model],
  );
  return (
    <ExportPanel {...out} onClose={onClose}>
      {image}
    </ExportPanel>
  );
}

const CIP_KEY = "ksmech.sketch.showCip";

/** The R/S toggle, kept per tab (sessionStorage; off when unavailable). */
function useShowCip(): [boolean, () => void] {
  const [on, setOn] = useState<boolean>(() => {
    try {
      return window.sessionStorage.getItem(CIP_KEY) === "1";
    } catch {
      return false;
    }
  });
  const toggle = () =>
    setOn((v) => {
      try {
        window.sessionStorage.setItem(CIP_KEY, v ? "0" : "1");
      } catch {
        // storage unavailable: the toggle lasts as long as the page
      }
      return !v;
    });
  return [on, toggle];
}

interface StyleState {
  styleId: StyleId;
  style: DrawingStyle;
  custom: DrawingStyle;
  onStyle(id: StyleId, custom?: DrawingStyle): void;
}

/** The drawing style: the page's (display + onStyle) or the Sketcher's own state. The chain
 *  angle of the style shapes new bonds (geometry.setChainAngle). */
function useDrawingStyle(display: SketchDisplay | undefined, onStyleProp: SketcherProps["onStyle"]): StyleState {
  const [own, setOwn] = useState<{ id: StyleId; custom: DrawingStyle }>({ id: "screen", custom: ACS_1996_STYLE });
  const controlled = !!onStyleProp;
  const styleId = (controlled ? display?.styleId : own.id) ?? "screen";
  const custom = (controlled ? display?.customStyle : own.custom) ?? ACS_1996_STYLE;
  const style = useMemo(
    () => resolveStyle(styleId, custom, { fontPx: display?.fontPx, bondWidth: display?.bondWidth, colourAtoms: display?.colourAtoms }),
    [styleId, custom, display?.fontPx, display?.bondWidth, display?.colourAtoms],
  );
  useEffect(() => setChainAngle(style.chainAngleDeg), [style.chainAngleDeg]);
  const onStyle = (id: StyleId, next?: DrawingStyle) => {
    const c = next ?? custom;
    if (onStyleProp) onStyleProp(id, c);
    else setOwn({ id, custom: c });
  };
  return { styleId, style, custom, onStyle };
}

/** "Actual size": the zoom at which a standard bond is the style's length on screen, about the
 *  canvas centre. */
function actualSizeView(view: View, style: DrawingStyle, width: number, height: number): View {
  const s = actualSizeScale(style);
  const cx = (width / 2 - view.ox) / view.s;
  const cy = (view.oy - height / 2) / view.s;
  return { s, ox: width / 2 - s * cx, oy: height / 2 + s * cy };
}

function useCleanStyle(): [CleanStyle, (s: CleanStyle) => void] {
  const [style, setStyle] = useState<CleanStyle>(restoreCleanStyle);
  const set = (s: CleanStyle) => {
    setStyle(s);
    saveCleanStyle(s);
  };
  return [style, set];
}

/** The image files of the drawing in its style (pre-rendered for drag and copy) and the
 *  export panel's image section. */
function useSketchImage(ed: SketchEditor, info: ReturnType<typeof useSketchInfo>, style: DrawingStyle, labelStyle: LabelStyle | undefined, enabled: boolean) {
  const [options, setOptions] = useState<ImageOptions>(DEFAULT_IMAGE_OPTIONS);
  const hCounts = !info.stale && info.valid ? info.hCounts : null;
  const files: SchemeFiles | null = useSchemeFiles({ model: ed.committed, hCounts, style, labelStyle, stereo: info.valid ? info.stereo : null, options, enabled });
  const section = <ImageExportSection files={files} options={options} onOptions={setOptions} />;
  return { files, section };
}

/** The publication presets: the tidy keeps their clearances whatever style is on screen. */
const PUBLICATION_STYLES = [ACS_1996_STYLE, TOTAL_SYNTHESIS_STYLE];

/** The fragment tidy in the drawing style and the publication presets (fragments.ts), with
 *  RDKit's H counts (atom ids are kept by Clean, so the counts of the drawing before it still
 *  apply). */
function useTidy(ed: SketchEditor, info: ReturnType<typeof useSketchInfo>, style: DrawingStyle, labelStyle: LabelStyle | undefined) {
  const hCounts = info.valid ? info.hCounts : null;
  const opts = { style, alsoFor: PUBLICATION_STYLES, labelStyle };
  const run = (m: SketchModel) => tidyFragments(m, hCounts, opts);
  const frags = useMemo(() => fragments(ed.committed).length, [ed.committed]);
  const canTidy = useMemo(() => frags > 1 && tidyFragments(ed.committed, hCounts, { style, alsoFor: PUBLICATION_STYLES, labelStyle }) !== ed.committed, [frags, ed.committed, hCounts, style, labelStyle]);
  return { run, canTidy, fragments: frags };
}

export function Sketcher({ value, onChange, highlightAtoms, readOnly = false, height = 360, projection, depictor, display, onStyle }: SketcherProps) {
  const ed = useSketchEditor(value, onChange);
  const info = useSketchInfo(ed.committed);
  const { rdkit } = useRDKit();
  const rootRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const width = useWidth(stageRef);
  const { view, setView, fit } = useView(ed, width, height);
  const ds = useDrawingStyle(display, onStyle);
  const metrics = useMemo(() => drawMetrics(ds.style, pxPerPt(ds.style, view.s)), [ds.style, view.s]);
  const image = useSketchImage(ed, info, ds.style, display?.labelStyle, !readOnly);
  const [panel, setPanel] = useState<SketchPanel>(null);
  const frame = useMemo(() => selectionFrame(ed.model, [...selectionAtoms(ed.committed, ed.selection)], view), [ed.model, ed.committed, ed.selection, view]);
  const pointer = useSketchPointer(ed, svgRef, view, setView, readOnly, frame);
  const act = useSketchActions(ed, rdkit, readOnly);
  const [cleanStyle, setCleanStyle] = useCleanStyle();
  const tidy = useTidy(ed, info, ds.style, display?.labelStyle);
  const layout = useLayoutActions(ed, rdkit, projection, depictor, cleanStyle, tidy.run);
  const fresh = !info.stale && ed.model === ed.committed && info.valid;
  const closePanel = () => setPanel(null);
  useSketchKeys({ ed, pointer, rootRef, readOnly, closePanel, importText: act.importText, layout, transform: act.transform });
  const highlight = useMemo(() => new Set(highlightAtoms ?? []), [highlightAtoms]);
  const invalid = useMemo(() => new Set(info.badAtoms), [info.badAtoms]);
  const crossings = useMemo(() => findCrossings(ed.committed).length, [ed.committed]);
  const [showCip, toggleCip] = useShowCip();
  const cip = showCip && info.valid ? info.stereo : null;
  const cipList = useMemo(() => (cip ? cipItems(ed.committed, cip) : null), [cip, ed.committed]);
  const radicals = radicalElectrons(ed.committed);
  const setInside = (v: boolean) => () => {
    pointer.inside.current = v;
  };
  return (
    <div
      ref={rootRef}
      className={`sketch${ds.style.colour ? "" : " sk-mono"}`}
      data-testid="sketch"
      data-style={ds.styleId}
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
      data-busy={layout.busy ? "true" : "false"}
      data-clean-style={cleanStyle}
      data-radicals={radicals}
      data-show-cip={showCip ? "true" : "false"}
      data-fragments={tidy.fragments}
      data-fragments-touch={tidy.canTidy ? "true" : "false"}
      onPointerEnter={setInside(true)}
      onPointerLeave={setInside(false)}
    >
      {!readOnly && (
        <EditorToolbar
          ed={ed}
          act={act}
          layout={layout}
          projection={projection}
          cleanStyle={cleanStyle}
          onCleanStyle={setCleanStyle}
          onFit={fit}
          panel={panel}
          onPanel={setPanel}
          showCip={showCip}
          onToggleCip={toggleCip}
          canTidy={tidy.canTidy}
          fragments={tidy.fragments}
          styleControls={{ styleId: ds.styleId, style: ds.style, onStyle: ds.onStyle, onActualSize: () => setView(actualSizeView(view, ds.style, width, height)) }}
        />
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
          frame={frame}
          height={height}
          pointer={pointer}
          svgRef={svgRef}
          readOnly={readOnly}
          cip={cip}
          display={{ labelStyle: display?.labelStyle, metrics }}
          styleId={ds.styleId}
        />
        {!readOnly && ed.committed.atoms.length > 0 && <DragHandle files={image.files} />}
        {panel === "import" && <ImportPanel onImport={act.importText} onClose={closePanel} />}
        {panel === "export" && <ExportFor rdkit={rdkit} model={ed.committed} onClose={closePanel} image={image.section} />}
      </div>
      <InfoBar info={info} report={layout.report} cip={cipList} radicals={radicals} />
    </div>
  );
}
