// Pointer handling of the sketch canvas: hover by geometric hit test, clicks through
// tools.applyClick, drags through gestures.dragModel (preview, committed on release), the
// select tool's rubber band, the selection frame's handles (frame.ts: rotate and scale the
// selection, any tool; one commit on release), middle-button (or Alt) panning and Ctrl/Cmd +
// wheel zoom.
import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type RefObject } from "react";
import { dist, inRect, rectFrom, toModel, toScreen, type Rect, type Vec, type View } from "@ksmech/viz/sketch/geometry";
import { dragModel, finishMove, movedAtoms, type DragStart } from "./gestures";
import { frameHandleAt, handleDragModel, type HandleDrag, type SelectionFrame } from "./frame";
import { applyClick, hitTest, NO_SELECTION, selectInRect, type Hit, type Selection } from "./tools";
import type { SketchEditor } from "./useSketchEditor";

const DRAG_START_PX = 4;
const MIN_SCALE = 6;
const MAX_SCALE = 80;

interface Press {
  start: DragStart;
  screen: Vec;
  moved: boolean;
  additive: boolean;
  pan: View | null;
  /** A drag of a selection-frame handle. */
  handle: HandleDrag | null;
}

export interface SketchPointer {
  onPointerDown(e: ReactPointerEvent<SVGSVGElement>): void;
  onPointerMove(e: ReactPointerEvent<SVGSVGElement>): void;
  onPointerUp(e: ReactPointerEvent<SVGSVGElement>): void;
  onPointerLeave(): void;
  /** Rubber band in screen pixels while the select tool drags on empty canvas. */
  rubber: Rect | null;
  /** True while the pointer is over the sketcher (keyboard shortcuts are live); the Sketcher
   *  root sets it. */
  inside: RefObject<boolean>;
  cancel(): void;
}

function toggle(sel: Selection, hit: NonNullable<Hit>, additive: boolean): Selection {
  if (!additive) return hit.kind === "atom" ? { atoms: new Set([hit.id]), bonds: new Set() } : { atoms: new Set(), bonds: new Set([hit.id]) };
  const atoms = new Set(sel.atoms);
  const bonds = new Set(sel.bonds);
  const set = hit.kind === "atom" ? atoms : bonds;
  if (set.has(hit.id)) set.delete(hit.id);
  else set.add(hit.id);
  return { atoms, bonds };
}

function union(a: Selection, b: Selection): Selection {
  return { atoms: new Set([...a.atoms, ...b.atoms]), bonds: new Set([...a.bonds, ...b.bonds]) };
}

/** A press released without moving: the select tool's (de)selection, else the tool's click. */
function clickPress(ed: SketchEditor, view: View, d: Press, p: Vec) {
  const { hit, tool } = d.start;
  if (tool === "select") {
    ed.setSelection(hit ? toggle(ed.selection, hit, d.additive) : d.additive ? ed.selection : NO_SELECTION);
    return;
  }
  const next = applyClick(ed.committed, tool, ed.element, hit, toModel(view, p));
  if (next !== ed.committed) ed.commit(next);
  ed.setHover(hitTest(next, view, p));
}

/** The selection of a rubber band from the press point to p (Shift adds to the old one). */
function rubberSelection(view: View, d: Press, p: Vec): Selection {
  const r = rectFrom(d.screen, p);
  const found = selectInRect(d.start.base, (q) => inRect(r, toScreen(view, q)));
  return d.additive ? union(d.start.selection, found) : found;
}

/** Commit the model of a drag released at p (a select-drag may merge its atom on drop). */
function finishDrag(ed: SketchEditor, view: View, d: Press, p: Vec, freeAngle: boolean) {
  const exclude = new Set(d.start.tool === "select" ? movedAtoms(d.start) : []);
  const toHit = hitTest(d.start.base, view, p, exclude);
  const m = dragModel(d.start, toModel(view, p), toHit, freeAngle);
  if (!m) ed.setPreview(null);
  else ed.commit(d.start.tool === "select" ? finishMove(d.start, m, toHit) : m);
}

/** The handle drag a press at p starts, or null when p is on no handle of the frame. */
function startHandle(frame: SelectionFrame | null, p: Vec, ed: SketchEditor): HandleDrag | null {
  const handle = frameHandleAt(frame, p);
  return handle && frame ? { handle, ids: frame.ids, centre: frame.centre, from: p, base: ed.committed } : null;
}

export function useSketchPointer(
  ed: SketchEditor,
  svgRef: RefObject<SVGSVGElement | null>,
  view: View,
  setView: (v: View) => void,
  readOnly: boolean,
  frame: SelectionFrame | null,
): SketchPointer {
  const press = useRef<Press | null>(null);
  const inside = useRef(false);
  const [rubber, setRubber] = useState<Rect | null>(null);

  const point = (e: { clientX: number; clientY: number }): Vec => {
    const r = svgRef.current?.getBoundingClientRect();
    return r ? [e.clientX - r.left, e.clientY - r.top] : [e.clientX, e.clientY];
  };

  const cancel = useCallback(() => {
    press.current = null;
    setRubber(null);
    ed.setPreview(null);
  }, [ed]);

  const onPointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    const p = point(e);
    const pan = e.button === 1 || (e.button === 0 && e.altKey);
    if (!pan && (e.button !== 0 || readOnly)) return;
    e.preventDefault(); // no text selection; it also keeps focus where it was, so move it here
    svgRef.current?.focus({ preventScroll: true });
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      // synthetic events without an active pointer
    }
    const hit = hitTest(ed.committed, view, p);
    const start: DragStart = { tool: ed.tool, hit, at: toModel(view, p), base: ed.committed, selection: ed.selection };
    const handle = pan ? null : startHandle(frame, p, ed);
    press.current = { start, screen: p, moved: false, additive: e.shiftKey, pan: pan ? view : null, handle };
  };

  const onPointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    const p = point(e);
    const d = press.current;
    if (!d) {
      ed.setHover(hitTest(ed.committed, view, p));
      return;
    }
    if (!d.moved && dist(p, d.screen) < DRAG_START_PX) return;
    d.moved = true;
    if (d.handle) {
      ed.setPreview(handleDragModel(d.handle, view, p, e.shiftKey));
      return;
    }
    if (d.pan) {
      setView({ ...d.pan, ox: d.pan.ox + p[0] - d.screen[0], oy: d.pan.oy + p[1] - d.screen[1] });
      return;
    }
    if (d.start.tool === "select" && !d.start.hit) {
      setRubber(rectFrom(d.screen, p));
      return;
    }
    const exclude = new Set(d.start.tool === "select" ? movedAtoms(d.start) : []);
    const toHit = hitTest(d.start.base, view, p, exclude);
    ed.setHover(toHit);
    ed.setPreview(dragModel(d.start, toModel(view, p), toHit, e.shiftKey));
  };

  const onPointerUp = (e: ReactPointerEvent<SVGSVGElement>) => {
    const d = press.current;
    press.current = null;
    if (!d || d.pan) return;
    const p = point(e);
    if (d.handle) {
      if (!d.moved) return;
      const next = handleDragModel(d.handle, view, p, e.shiftKey);
      if (next !== ed.committed) ed.commit(next);
      else ed.setPreview(null);
      return;
    }
    if (!d.moved) clickPress(ed, view, d, p);
    else if (d.start.tool === "select" && !d.start.hit) {
      ed.setSelection(rubberSelection(view, d, p));
      setRubber(null);
    } else finishDrag(ed, view, d, p, e.shiftKey);
  };

  useZoom(svgRef, view, setView);

  return {
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onPointerLeave: () => {
      if (!press.current) ed.setHover(null);
    },
    rubber,
    inside,
    cancel,
  };
}

/** Ctrl/Cmd + wheel (and trackpad pinch) zooms about the pointer; a plain wheel scrolls the page. */
function useZoom(svgRef: RefObject<SVGSVGElement | null>, view: View, setView: (v: View) => void) {
  const viewRef = useRef(view);
  useEffect(() => {
    viewRef.current = view;
  });
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      const v = viewRef.current;
      const r = svg.getBoundingClientRect();
      const px = e.clientX - r.left;
      const py = e.clientY - r.top;
      const s = Math.max(MIN_SCALE, Math.min(MAX_SCALE, v.s * Math.exp(-e.deltaY * 0.002)));
      setView({ s, ox: px - ((px - v.ox) * s) / v.s, oy: py - ((py - v.oy) * s) / v.s });
    };
    svg.addEventListener("wheel", onWheel, { passive: false });
    return () => svg.removeEventListener("wheel", onWheel);
  }, [svgRef, setView]);
}
