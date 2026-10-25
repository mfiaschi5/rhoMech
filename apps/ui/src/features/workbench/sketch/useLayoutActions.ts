// The sketcher's layout actions: the 3D -> 2D projection (from a ProjectionSource the page
// provides: the sketch never talks to the 3D viewer itself), "Fix layout" and the crossing
// breaks toggle, with the report of the last projection or fix (shown in the info bar while it
// describes the drawing on screen).
//
// Projection: applyProjection (coordinates and depth from the source), RDKit's reading of the
// 3D stereo (the target), projectedLayout (wedges from the depth, layout fix, tilts), then
// repairStereo as the safety net; one undoable commit. Fix: fixLayout with the hand anchor (the
// projection anchor when the drawing has depth), RDKit's reading of the drawing before as the
// stereo target.
import { useState } from "react";
import type { RDKitModule } from "../../../chem/rdkit";
import { hasDepth, setBreakCrossings, type SketchModel } from "./model";
import { applyProjection, type Projection3D } from "./projection";
import { projectedLayout } from "./projectLayout";
import { ANCHOR_HAND, ANCHOR_PROJECTION, fixLayout, type LayoutMetrics } from "./layoutFix";
import { drawnStereo, repairStereo, stereoFromDepth, type StereoReading } from "./rdkitSketch";
import type { SketchEditor } from "./useSketchEditor";

/** What the page offers the sketcher for the projection. */
export interface ProjectionSource {
  /** Null when the projection can run; else why not (the button's tooltip). */
  disabledReason: string | null;
  /** The drawn atoms in the viewer's current frame (Angstrom), or null. */
  project(): Projection3D | null;
}

export type StereoOutcome = "none" | "kept" | "repaired" | "mismatch";

export interface LayoutReport {
  kind: "project" | "fix";
  before: LayoutMetrics;
  after: LayoutMetrics;
  tilt: readonly [number, number] | null;
  stereo: StereoOutcome;
  forModel: SketchModel;
}

/** Repair the drawing's stereo towards `target` (RDKit), with the outcome. */
function checkStereo(rdkit: RDKitModule | null, m: SketchModel, target: StereoReading | null): { model: SketchModel; stereo: StereoOutcome } {
  if (!rdkit || !target || target.centres.size === 0) return { model: m, stereo: "none" };
  const r = repairStereo(rdkit, m, target);
  return { model: r.model, stereo: !r.ok ? "mismatch" : r.model !== m ? "repaired" : "kept" };
}

export function useLayoutActions(ed: SketchEditor, rdkit: RDKitModule | null, source: ProjectionSource | undefined, hCounts: ReadonlyMap<number, number> | null) {
  const [report, setReport] = useState<LayoutReport | null>(null);
  const canProject = !!source && source.disabledReason === null && ed.committed.atoms.length > 0;
  const project = () => {
    const p = canProject ? source?.project() : null;
    const m1 = p ? applyProjection(ed.committed, p) : null;
    if (!m1) return;
    const target = rdkit ? stereoFromDepth(rdkit, m1) : null;
    const r = projectedLayout(m1, target ? target.centres.keys() : [], { hCounts });
    const { model, stereo } = checkStereo(rdkit, r.model, target);
    ed.commit(model, { external: true });
    setReport({ kind: "project", before: r.before, after: r.after, tilt: r.tilt, stereo, forModel: model });
  };
  const fix = () => {
    const m = ed.committed;
    if (m.atoms.length < 2) return;
    const target = rdkit ? drawnStereo(rdkit, m) : null;
    const r = fixLayout(m, { anchor: hasDepth(m) ? ANCHOR_PROJECTION : ANCHOR_HAND, hCounts });
    const { model, stereo } = checkStereo(rdkit, r.model, target);
    if (model !== m) ed.commit(model);
    setReport({ kind: "fix", before: r.before, after: r.after, tilt: null, stereo, forModel: model });
  };
  const toggleBreaks = () => ed.commit(setBreakCrossings(ed.committed, !ed.committed.breakCrossings));
  const shown = report && report.forModel === ed.committed ? report : null;
  return { canProject, project, fix, toggleBreaks, report: shown };
}

export type LayoutActions = ReturnType<typeof useLayoutActions>;
