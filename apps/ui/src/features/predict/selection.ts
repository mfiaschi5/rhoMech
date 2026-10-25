// Selection state of the Predict job view, all of it in the hash (design §11.2):
//   #/predict/job/<jobId>?pathway=<pid>&node=<nid>&edge=<eid>&arrow=<i>&cand=<d>,<ds>,<a>,<as>
// plus these feature-local params:
//   phys=cand     the physics drawer shows the previewed candidate tuple (`cand`) of `node`
//   threed=1      the 3D contact panel is open in the drawer
//   cands=1       the node's candidates panel is open
//   cdonor=<id>   candidate donor filter;  cexcl=1  show excluded tuples with their reasons
//   cycles=1      highlight every cycle of the network
//   override=<j>  the override job whose result is shown (set when Correct returns here)
// `arrow=<i>` opens the physics drawer for arrow i of the selected edge.
import { useCallback } from "react";
import { navigate, parseCand, withParams, type Route } from "../../router";

export interface JobSelection {
  pathway: string | null;
  node: string | null;
  edge: string | null;
  arrow: number | null;
  cand: [number, number, number, number] | null;
  candPhysics: boolean;
  threed: boolean;
  cands: boolean;
  candDonor: number | null;
  candExcluded: boolean;
  cycles: boolean;
  override: string | null;
}

function intOrNull(v: string | null): number | null {
  if (v === null || v === "") return null;
  const n = Number.parseInt(v, 10);
  return Number.isFinite(n) ? n : null;
}

export function parseSelection(route: Route): JobSelection {
  const p = route.params;
  return {
    pathway: p.get("pathway") || null,
    node: p.get("node") || null,
    edge: p.get("edge") || null,
    arrow: intOrNull(p.get("arrow")),
    cand: parseCand(p.get("cand")),
    candPhysics: p.get("phys") === "cand",
    threed: p.get("threed") === "1",
    cands: p.get("cands") === "1",
    candDonor: intOrNull(p.get("cdonor")),
    candExcluded: p.get("cexcl") === "1",
    cycles: p.get("cycles") === "1",
    override: p.get("override") || null,
  };
}

export type SelectionUpdate = Record<string, string | number | null | undefined>;

/** Params cleared whenever the selected node or edge changes (the candidates panel, its
 *  filters and preview belong to one node; the drawer to one arrow). */
export const CLEAR_TARGET: SelectionUpdate = {
  arrow: null,
  cand: null,
  phys: null,
  threed: null,
  cands: null,
  cdonor: null,
  cexcl: null,
};

/** A setter for selection params: rewrites the current hash entry (no history spam). */
export function useSelect(route: Route): (updates: SelectionUpdate) => void {
  return useCallback(
    (updates: SelectionUpdate) => navigate(withParams(route, updates), { replace: true }),
    [route],
  );
}

/** "<d>,<ds>,<a>,<as>" for the cand param. */
export function candParam(t: { donor: number; donor_site: number; acceptor: number; acceptor_site: number }) {
  return `${t.donor},${t.donor_site},${t.acceptor},${t.acceptor_site}`;
}
