// Pure helpers over NetworkOut / JobOut (no React, no DOM).
import type {
  ArrowOut,
  CycleOut,
  EdgeOut,
  JobOut,
  NetworkOut,
  NodeOut,
  OverrideResultOut,
  PathwayOut,
} from "../../api/types.gen";
import type { TokenArrowLike } from "../../chem/arrowGeom";

/** The network a predict (or override) job carries, if any (also partial ones while running). */
export function networkOf(job: JobOut | null | undefined): NetworkOut | null {
  const r = job?.result as Partial<NetworkOut> | null | undefined;
  if (!r || typeof r !== "object") return null;
  if (job?.result_kind === "network" || (Array.isArray(r.nodes) && Array.isArray(r.edges) && "root" in r)) {
    return r as NetworkOut;
  }
  return null;
}

export interface NetIndex {
  nodes: Map<string, NodeOut>;
  edges: Map<string, EdgeOut>;
  cycles: Map<string, CycleOut>;
  root: NodeOut | undefined;
}

export function indexNetwork(net: NetworkOut): NetIndex {
  const nodes = new Map(net.nodes.map((n) => [n.id, n] as const));
  const edges = new Map(net.edges.map((e) => [e.id, e] as const));
  const cycles = new Map(net.cycles.map((c) => [c.id, c] as const));
  return { nodes, edges, cycles, root: nodes.get(net.root) };
}

/** "C[O-] + CCl", with counts ("2 S + Cat"). */
export function speciesLabel(node: NodeOut): string {
  if (!node.species.length) return node.display_smiles || node.pool_key;
  return node.species.map((s) => (s.count > 1 ? `${s.count} ${s.display_smiles}` : s.display_smiles)).join(" + ");
}

/** "O1→C7" for one arrow (element and map number of both sites). */
export function arrowText(a: TokenArrowLike): string {
  return `${a.donor_element}${a.donor_site_map}→${a.acceptor_element}${a.acceptor_site_map}`;
}

export function edgeArrowsText(e: EdgeOut): string {
  return e.arrows.length ? e.arrows.map(arrowText).join("; ") : "(no arrows)";
}

/** Map numbers a depiction must keep (explicit H sites, bond partners, sink partners). */
export function arrowMaps(arrows: readonly TokenArrowLike[]): number[] {
  const maps = new Set<number>();
  for (const a of arrows) {
    maps.add(a.donor_site_map);
    maps.add(a.acceptor_site_map);
    if (a.source_partner_map != null) maps.add(a.source_partner_map);
    if (a.sink_partner_map != null) maps.add(a.sink_partner_map);
  }
  return [...maps].filter((m) => Number.isFinite(m) && m > 0).sort((x, y) => x - y);
}

/** The pathway shown in the strip: the selected one, else rank 1. */
export function currentPathway(net: NetworkOut, selected: string | null): PathwayOut | null {
  if (selected) {
    const hit = net.pathways.find((p) => p.id === selected);
    if (hit) return hit;
  }
  return net.pathways.length ? [...net.pathways].sort((a, b) => a.rank - b.rank)[0] : null;
}

/** Edge ids to draw in the cycle colour: every cycle (toggle on), else the cycles of the
 *  selected edge and of the shown pathway. */
export function cycleEdgeIds(net: NetworkOut, edge: EdgeOut | undefined, pathway: PathwayOut | null, all: boolean) {
  const ids = new Set<string>();
  const wanted = new Set<string>(all ? net.cycles.map((c) => c.id) : [...(edge?.cycle_ids ?? []), ...(pathway?.cycle_ids ?? [])]);
  for (const c of net.cycles) if (wanted.has(c.id)) for (const e of c.edge_ids) ids.add(e);
  return { edgeIds: ids, cycleIds: wanted };
}

/** The id of the cytoscape virtual edge of a turnover cycle. */
export const virtualEdgeId = (cycleId: string) => `virtual-${cycleId}`;

/** The outcome of an override job: its OverrideResultOut, or (when the job result is the
 *  network itself) the newest corrected edge that replaced `edgeId` / leaves `nodeId`. */
export function overrideOutcome(
  job: JobOut,
  target: { edgeId?: string | null; nodeId?: string | null } = {},
): OverrideResultOut | null {
  const r = job.result as Partial<OverrideResultOut> | null;
  if (r && typeof r === "object" && typeof r.corrected_edge_id === "string") return r as OverrideResultOut;
  const net = networkOf(job);
  if (!net) return null;
  const corrected = net.edges
    .filter((e) => e.kind === "corrected")
    .filter((e) => (target.edgeId ? e.original?.edge_id === target.edgeId : target.nodeId ? e.source === target.nodeId : true))
    .sort((a, b) => b.revision - a.revision);
  const edge = corrected[0];
  if (!edge) return null;
  return {
    parent_job_id: job.parent_job_id ?? job.network_id ?? "",
    network_revision: net.revision,
    corrected_edge_id: edge.id,
    target_node: edge.target,
    continued_from: null,
    issues: [],
  };
}

/** The 1-based position of an arrow in its edge, for labels ("arrow 2 of 3"). */
export function arrowOrdinal(arrows: readonly ArrowOut[], index: number): string {
  return arrows.length > 1 ? `arrow ${index + 1} of ${arrows.length}` : "arrow";
}
