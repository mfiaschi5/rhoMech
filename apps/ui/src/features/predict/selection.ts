// Selection state of the Predict job view, all of it in the hash (router.ts):
//   #/predict/job/<jobId>?pathway=<pid>&node=<nid>&edge=<eid>
import { useCallback } from "react";
import { navigate, withParams, type Route } from "../../router";

export interface JobSelection {
  pathway: string | null;
  node: string | null;
  edge: string | null;
}

export function parseSelection(route: Route): JobSelection {
  const p = route.params;
  return { pathway: p.get("pathway") || null, node: p.get("node") || null, edge: p.get("edge") || null };
}

export type SelectionUpdate = Record<string, string | number | null | undefined>;

/** A setter for selection params: rewrites the current hash entry (no history spam). */
export function useSelect(route: Route): (updates: SelectionUpdate) => void {
  return useCallback((updates: SelectionUpdate) => navigate(withParams(route, updates), { replace: true }), [route]);
}
