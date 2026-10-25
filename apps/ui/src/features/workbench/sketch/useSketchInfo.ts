// useSketchInfo(model): the RDKit judgement of a drawing, debounced (250 ms after the last
// edit; the first analysis runs at once). Returns the latest SketchAnalysis plus a status:
//   "empty"        nothing drawn
//   "loading"      RDKit.js still loading (H counts are the default-valence fallback)
//   "unavailable"  RDKit.js failed to load
//   "pending"      the drawing changed since the analysis shown (it is `stale`)
//   "ready"        the analysis describes this exact model
import { useEffect, useRef, useState } from "react";
import { useRDKit } from "../../../chem/rdkit";
import type { SketchModel } from "./model";
import { analyzeSketch, type SketchAnalysis } from "./rdkitSketch";

export type SketchInfoStatus = "empty" | "loading" | "unavailable" | "pending" | "ready";

export interface SketchInfo extends SketchAnalysis {
  status: SketchInfoStatus;
  /** True when the fields describe an earlier drawing than `model`. */
  stale: boolean;
}

export const SKETCH_INFO_DELAY_MS = 250;

function placeholder(model: SketchModel, error: string | null): SketchAnalysis {
  return {
    forModel: model,
    empty: model.atoms.length === 0,
    valid: false,
    smiles: "",
    inchi: "",
    descriptors: null,
    formula: "",
    error,
    badAtoms: [],
    hCounts: new Map(),
  };
}

export function useSketchInfo(model: SketchModel, delayMs = SKETCH_INFO_DELAY_MS): SketchInfo {
  const { rdkit, error } = useRDKit();
  const [result, setResult] = useState<SketchAnalysis | null>(null);
  const analyzed = useRef(false);
  useEffect(() => {
    if (!rdkit) return;
    const run = () => {
      analyzed.current = true;
      setResult(analyzeSketch(rdkit, model));
    };
    const t = window.setTimeout(run, analyzed.current ? delayMs : 0);
    return () => window.clearTimeout(t);
  }, [rdkit, model, delayMs]);

  if (model.atoms.length === 0) return { ...placeholder(model, null), status: "empty", stale: false };
  if (!rdkit) {
    const msg = error ? `RDKit.js unavailable: ${error.message}` : null;
    return { ...placeholder(model, msg), status: error ? "unavailable" : "loading", stale: false };
  }
  if (!result) return { ...placeholder(model, null), status: "pending", stale: true };
  const stale = result.forModel !== model;
  return { ...result, status: stale ? "pending" : "ready", stale };
}
