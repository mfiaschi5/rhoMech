// Pure helpers over the spec-v9 NetworkOut (no React, no DOM).
import { fmt, fmtSig } from "../../components/format";
import type { EdgeOut, MatrixIO, NetworkOut, NodeOut, PathwayOut } from "./v9types";

/** "C[O-] + CCl", with counts ("2 S + Cat"). */
export function speciesLabel(node: NodeOut): string {
  if (!node.species.length) return node.display_smiles || node.pool_key;
  return node.species.map((s) => (s.count > 1 ? `${s.count} ${s.display_smiles}` : s.display_smiles)).join(" + ");
}

/** The pathway shown: the selected one, else rank 1. */
export function currentPathway(net: NetworkOut, selected: string | null): PathwayOut | null {
  if (selected) {
    const hit = net.pathways.find((p) => p.id === selected);
    if (hit) return hit;
  }
  return net.pathways.length ? [...net.pathways].sort((a, b) => a.rank - b.rank)[0] : null;
}

/** Edge ids of the cycles of the selected edge and of the shown pathway. */
export function cycleEdgeIds(net: NetworkOut, edge: EdgeOut | undefined, pathway: PathwayOut | null): Set<string> {
  const wanted = new Set<string>([...(edge?.cycle_ids ?? []), ...(pathway?.cycle_ids ?? [])]);
  const ids = new Set<string>();
  for (const c of net.cycles) if (wanted.has(c.id)) for (const e of c.edge_ids) ids.add(e);
  return ids;
}

/** The id of the cytoscape virtual edge of a turnover cycle. */
export const virtualEdgeId = (cycleId: string) => `virtual-${cycleId}`;

/** A 0..1 weight for the edge width: exp(logp) when the predictor gives log-probabilities, else
 *  the score over the largest |score| of the network. */
export function edgeWeights(net: NetworkOut): Map<string, number> {
  const max = Math.max(1e-12, ...net.edges.map((e) => Math.abs(e.score ?? 0)));
  return new Map(
    net.edges.map((e) => {
      const w = e.logp !== null && Number.isFinite(e.logp) ? Math.exp(e.logp) : Math.abs(e.score ?? 0) / max;
      return [e.id, Math.max(0, Math.min(1, w))] as const;
    }),
  );
}

/** The predictor's rank of a step, or null for a user step (rank 0 or null). */
export const rankOf = (e: EdgeOut): number | null => (e.rank && e.rank > 0 ? e.rank : null);

/** "#2", or the kind ("corrected") for a user step. */
export const rankLabel = (e: EdgeOut): string => (rankOf(e) !== null ? `#${rankOf(e)}` : e.kind);

/** "p 0.412" (probabilities) or "score 3.21" (the floor), "user" for a corrected step. */
export function scoreText(e: EdgeOut, probabilities: boolean): string {
  if (e.kind === "corrected") return "user";
  if (probabilities && e.logp !== null) return `p ${fmt(Math.exp(e.logp), 3)}`;
  return `score ${fmtSig(e.score, 3)}`;
}

/** "+C1–O0 −C1–Cl2 lp O0 −2": R as compact text; `label(i)` names an atom. */
export function matrixText(m: MatrixIO, label: (i: number) => string = String): string {
  const bonds = m.bonds.map(([i, j, d]) => `${d > 0 ? "+" : "−"}${label(i)}–${label(j)}`);
  const lone = m.lone.map(([i, d]) => `lp ${label(i)} ${d > 0 ? "+" : "−"}${Math.abs(d)}`);
  return [...bonds, ...lone].join(" ") || "(empty)";
}

/** The edges from the root to `nodeId` along best parents (for a breadcrumb). */
export function pathTo(net: NetworkOut, nodeId: string): EdgeOut[] {
  const nodes = new Map(net.nodes.map((n) => [n.id, n] as const));
  const edges = new Map(net.edges.map((e) => [e.id, e] as const));
  const out: EdgeOut[] = [];
  let cur = nodes.get(nodeId);
  const seen = new Set<string>();
  while (cur && cur.best_parent_edge && !seen.has(cur.id)) {
    seen.add(cur.id);
    const e = edges.get(cur.best_parent_edge);
    if (!e) break;
    out.unshift(e);
    cur = nodes.get(e.source);
  }
  return out;
}
