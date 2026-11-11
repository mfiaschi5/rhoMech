// The target product of the Predict form as request fields (TargetFields renders it): with a
// target the predict job searches the pathways to it (PredictIn.target_smiles).
import type { PredictIn } from "@ksmech/viz/v9types";

export interface TargetSettings {
  smiles: string;
  skeleton: boolean; // also accept the SKELETON level
  k: number | null; // pathways to find (n_pathways)
  maxExpansions: number | null;
}

export const DEFAULT_K = 5;
export const DEFAULT_EXPANSIONS = 50;
export const DEFAULT_TARGET: TargetSettings = { smiles: "", skeleton: true, k: DEFAULT_K, maxExpansions: DEFAULT_EXPANSIONS };

/** The PredictIn fields of a target; none when the target is empty (a beam expansion). */
export function targetRequest(t: TargetSettings): Partial<PredictIn> {
  const smiles = t.smiles.trim();
  if (!smiles) return {};
  return {
    target_smiles: smiles,
    target_levels: t.skeleton ? ["EXACT", "SKELETON"] : ["EXACT"],
    n_pathways: t.k ?? DEFAULT_K,
    max_expansions: t.maxExpansions ?? DEFAULT_EXPANSIONS,
  };
}
