// React glue of the step viewport variants: the variant choice (remembered in localStorage),
// the shown step (the edge, or one of its alternatives with its arrows and readings from POST
// /step/check) and the two sides of a step from the loaded depictions. The pure parts (the
// sides, the hover set, the container width) are stepSides.ts.
import { useMemo, useState } from "react";
import { useDrawnStructure, usePool, useStepCheck } from "../hooks";
import { EMPTY_MATRIX } from "@ksmech/viz/pool/rmatrix";
import { productIndexMap } from "@ksmech/viz/step/stepGeometry";
import { predictedStep, sidesOf, touchKeyOf, type ShownStep, type StepSides } from "@ksmech/viz/step/stepSides";
import type { EdgeOut } from "@ksmech/viz/v9types";

export const STEP_VARIANTS = ["textbook", "animated", "orbital", "embedding"] as const;
export type StepVariant = (typeof STEP_VARIANTS)[number];
export const STEP_VARIANT_KEY = "ksmech.predict.stepVariant";
export const STEP_VARIANT_TITLES: Record<StepVariant, string> = { textbook: "Textbook", animated: "Animated", orbital: "Orbitals", embedding: "Embedding" };

function readVariant(): StepVariant {
  try {
    const v = window.localStorage.getItem(STEP_VARIANT_KEY);
    return (STEP_VARIANTS as readonly string[]).includes(v ?? "") ? (v as StepVariant) : "textbook";
  } catch {
    return "textbook";
  }
}

export function useStepVariant(): [StepVariant, (v: StepVariant) => void] {
  const [v, setV] = useState<StepVariant>(readVariant);
  const set = (next: StepVariant) => {
    setV(next);
    try {
      window.localStorage.setItem(STEP_VARIANT_KEY, next);
    } catch {
      // storage blocked: the choice lasts for this page only
    }
  };
  return [v, set];
}

export function useShownStep(edge: EdgeOut, index: number): ShownStep {
  const alt = index >= 0 ? edge.alternatives[index] : undefined;
  const check = useStepCheck(alt ? edge.source_state_id : null, alt?.matrix ?? EMPTY_MATRIX);
  const charges = useMemo(() => new Map(edge.formal_charges), [edge.formal_charges]);
  if (!alt) return predictedStep(edge, charges);
  const res = check.current ? check.result : null;
  const note = check.error ? `no reading for this alternative: ${check.error.message}` : !res ? "reading the alternative…" : null;
  return { index, matrix: alt.matrix, arrows: res?.arrows ?? [], readings: res?.readings ?? [], chargesAfter: res ? new Map(res.formal_charges) : null, score: alt.score, logp: alt.logp, pending: !res, note };
}

export function useStepSides(edge: EdgeOut, step: ShownStep): StepSides {
  const pool = usePool(edge.source_state_id);
  const prodPool = usePool(edge.product_state_id || null);
  const structure = useDrawnStructure(pool, step.matrix, step.arrows);
  const toSource = useMemo(() => productIndexMap(edge.source_mapped_smiles, edge.product_mapped_smiles), [edge.source_mapped_smiles, edge.product_mapped_smiles]);
  const touchKey = touchKeyOf(step);
  return useMemo(() => {
    const sides = sidesOf(structure, prodPool.depiction, toSource, touchKey);
    return { ...sides, loading: pool.loading || prodPool.loading, error: sides.error ?? pool.error?.message ?? null };
  }, [structure, prodPool.depiction, toSource, touchKey, pool.loading, prodPool.loading, pool.error]);
}
