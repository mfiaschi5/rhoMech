// The sketcher's layout actions: Clean, 3D->2D and the crossing-breaks toggle, with the report
// of the last one (shown in the info bar while it describes the drawing on screen).
//
// Both layouts come from the server (POST /api/depict2d through the page's Depictor; the sketch
// never fetches by itself): Clean sends the molblock with mode "clean" and the chosen style
// (textbook: RDKit's ring templates or CoordGen, whichever has fewer defects; perspective: the
// chemist-style template library); 3D->2D sends mode "match3d" with the computed molecule's id
// and the 3D viewer's quaternion (ProjectionSource.view(), the page reads it from 3Dmol). The
// answer is applied by atom and bond index (depiction.applyDepiction) as one undoable commit,
// unless the drawing changed while it was on its way; then, when separate fragments (a salt's
// ions) come too close in the drawing style, "tidy fragments" (fragments.ts) spreads them apart
// as a second undoable step (the report says so; the Tidy button runs it alone). When the server
// cannot be reached (or
// refuses), Clean falls back to RDKit.js's CoordGen layout in the browser and the report says
// so; 3D->2D has no fallback.
import { useLayoutEffect, useRef, useState } from "react";
import type { Depict2DOut } from "../../../api/types.gen";
import type { RDKitModule } from "@ksmech/viz/chem/rdkit";
import { setBreakCrossings, type SketchModel } from "@ksmech/viz/sketch/model";
import type { Quat } from "./projection";
import { toMolblock } from "@ksmech/viz/sketch/molfile";
import { cleanUpSketch } from "./rdkitSketch";
import { applyDepiction, type CleanStyle, type Depictor } from "./depiction";
import type { SketchEditor } from "./useSketchEditor";

/** What the page offers the sketcher for 3D->2D. */
export interface ProjectionSource {
  /** Null when 3D->2D can run; else why not (the button's tooltip). */
  disabledReason: string | null;
  /** The computed molecule and the viewer's rotation quaternion [x, y, z, w] now, or null. */
  view(): { moleculeId: string; quaternion: Quat } | null;
}

export interface LayoutReport {
  kind: "clean" | "project";
  /** The server's answer (null for a fallback or an error). */
  result: Depict2DOut | null;
  /** Why the browser layout was used instead of the server's. */
  fallback: string | null;
  error: string | null;
  forModel: SketchModel;
  /** The fragments were spread apart after the layout (its own undo step). */
  tidied?: boolean;
}

const message = (err: unknown): string => (err instanceof Error ? err.message : String(err));

/** The fragment tidy of the drawing style: the model, or a copy with its fragments apart. */
export type Tidy = (m: SketchModel) => SketchModel;

export function useLayoutActions(ed: SketchEditor, rdkit: RDKitModule | null, source: ProjectionSource | undefined, depictor: Depictor | undefined, style: CleanStyle, tidy?: Tidy) {
  const [report, setReport] = useState<LayoutReport | null>(null);
  const [busy, setBusy] = useState(false);
  const latest = useRef(ed.committed);
  useLayoutEffect(() => {
    latest.current = ed.committed;
  });
  const n = ed.committed.atoms.length;
  const canProject = !!source && !!depictor && source.disabledReason === null && n > 0 && !busy;
  const canClean = n > 1 && !busy && (!!depictor || !!rdkit);

  /** Commit a layout, then its fragment tidy as a second step; the model on screen after. */
  const commitTidied = (next: SketchModel, kind: LayoutReport["kind"]): { model: SketchModel; tidied: boolean } => {
    ed.commit(next, { external: true });
    const t = kind === "clean" && tidy ? tidy(next) : next;
    if (t !== next) ed.commit(t, { external: true });
    return { model: t, tidied: t !== next };
  };

  const browserClean = (m: SketchModel, why: string) => {
    const next = rdkit ? cleanUpSketch(rdkit, m) : null;
    const done = next ? commitTidied(next, "clean") : { model: m, tidied: false };
    setReport({ kind: "clean", result: null, fallback: why, error: next ? null : "RDKit.js is not loaded", forModel: done.model, tidied: done.tidied });
  };

  /** Send `m` to the server and apply the answer; false when the drawing moved on meanwhile. */
  const run = async (kind: LayoutReport["kind"], m: SketchModel, req: Parameters<Depictor>[0]) => {
    const out = await (depictor as Depictor)(req);
    if (latest.current !== m) return false;
    const next = applyDepiction(m, out);
    if (!next) throw new Error("the server's drawing does not fit the sketch");
    const done = commitTidied(next, kind);
    setReport({ kind, result: out, fallback: null, error: null, forModel: done.model, tidied: done.tidied });
    return true;
  };

  const clean = async () => {
    const m = ed.committed;
    if (!canClean) return;
    if (!depictor) return browserClean(m, "no server: browser CoordGen layout");
    setBusy(true);
    try {
      await run("clean", m, { molblock: toMolblock(m), mode: "clean", style });
    } catch (err) {
      if (latest.current === m) browserClean(m, `server unavailable (${message(err)}): browser CoordGen layout`);
    } finally {
      setBusy(false);
    }
  };

  const project = async () => {
    const m = ed.committed;
    const v = canProject ? source?.view() : null;
    if (!v) return;
    setBusy(true);
    try {
      const view = { quaternion: [...v.quaternion], convention: "3dmol-rotationGroup" as const };
      await run("project", m, { molblock: toMolblock(m), mode: "match3d", style, molecule_id: v.moleculeId, view });
    } catch (err) {
      setReport({ kind: "project", result: null, fallback: null, error: message(err), forModel: m });
    } finally {
      setBusy(false);
    }
  };

  const toggleBreaks = () => ed.commit(setBreakCrossings(ed.committed, !ed.committed.breakCrossings));
  /** The Tidy button: spread touching fragments apart as one undoable step. */
  const tidyNow = () => {
    const t = tidy ? tidy(ed.committed) : ed.committed;
    if (t !== ed.committed) ed.commit(t, { external: true });
  };
  const shown = report && report.forModel === ed.committed ? report : null;
  return { canProject, canClean, busy, clean, project, toggleBreaks, tidy: tidyNow, report: shown };
}

export type LayoutActions = ReturnType<typeof useLayoutActions>;
