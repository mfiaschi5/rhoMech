// C. Pathway explorer: the ranked pathways as columns of structure cards (the scheme's cards),
// their shared prefixes merged into one trunk (pathwayTree / treeRows of netLayout.ts): each
// column is packed from the top in depth-first order, so the best pathway is the straight top
// lane, ribbons never cross and never run far. Ribbons are as wide as the weight of the pathways
// through them and labelled with the step's score; branches fold and unfold; a breadcrumb
// (fixed above the scrolling lanes) follows the selected path; a click on a card selects the
// step into it (the step panel below shows it), on the reactants their state.
//
// DOM: <div data-testid="predict-network" data-variant="explorer" data-node-count data-edge-count
// tabIndex=0> with predict-net-breadcrumb, cards div[data-testid=predict-net-card-<key>]
// [data-node][data-edge][data-selected], their fold buttons predict-net-fold-<key>, ribbons
// path[data-testid=predict-net-ribbon-<key>][data-weight].
import { useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useRDKit } from "../../../chem/rdkit";
import type { NetworkOut } from "../v9types";
import { backEdges, pathwayTree, treeRows, type TreeRow } from "./netLayout";
import { speciesLabel, stepInfo, typeVar, useSymbols, type NetViewProps, type StepInfo } from "./shared";
import { cardSize, StateCard, THUMB_MAX_H, THUMB_MAX_W } from "./StateCard";
import { cardThumb, type CardThumb } from "./thumbs";

const COL_GAP = 92;
const ROW_GAP = 14;
const PAD = 18;
const FOOT = 22;

interface Placed extends TreeRow {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Columns by depth, each as wide as its widest card. Cards are placed in depth-first order
 *  (treeRows' order), each at its parent's height when its column is free there, else just below
 *  the column's last card: lanes stay straight where they can, ribbons stay short, and a column
 *  never overlaps itself. */
export function place(rows: TreeRow[], size: (r: TreeRow) => { w: number; h: number }): Placed[] {
  const depths = [...new Set(rows.map((r) => r.node.depth))].sort((a, b) => a - b);
  const colX = new Map<number, number>();
  const colW = new Map<number, number>();
  let x = PAD;
  for (const d of depths) {
    const w = Math.max(...rows.filter((r) => r.node.depth === d).map((r) => size(r).w));
    colX.set(d, x);
    colW.set(d, w);
    x += w + COL_GAP;
  }
  const nextY = new Map<number, number>();
  const yOf = new Map<string, number>();
  const out: Placed[] = [];
  for (const r of rows) {
    const d = r.node.depth;
    const h = size(r).h;
    const parentY = r.parent !== null ? (yOf.get(r.parent) ?? PAD) : PAD;
    const y = Math.max(nextY.get(d) ?? PAD, parentY);
    yOf.set(r.node.key, y);
    nextY.set(d, y + h + ROW_GAP);
    out.push({ ...r, x: colX.get(d)!, y, w: colW.get(d)!, h });
  }
  return out;
}

function ribbon(a: Placed, b: Placed): string {
  const [x0, y0, x1, y1] = [a.x + a.w, a.y + a.h / 2, b.x, b.y + b.h / 2];
  const mx = (x0 + x1) / 2;
  return `M ${x0} ${y0} C ${mx} ${y0} ${mx} ${y1} ${x1} ${y1}`;
}

/** The tree key whose step is the selected edge (the first lane through it), else "". */
function selectedKey(rows: TreeRow[], edgeId: string | null, nodeId: string | null): string | null {
  if (edgeId) return rows.find((r) => r.node.edgeId === edgeId)?.node.key ?? null;
  if (nodeId) return rows.find((r) => r.node.nodeId === nodeId)?.node.key ?? null;
  return null;
}

function Breadcrumb({ net, placed, sel, info, onKey }: { net: NetworkOut; placed: Placed[]; sel: string | null; info: Map<string, StepInfo>; onKey: (key: string) => void }) {
  if (sel === null) return <div className="nv-crumbs muted" data-testid="predict-net-breadcrumb">Pick a card to follow its pathway.</div>;
  const chain: Placed[] = [];
  for (let key: string | null = sel; key !== null; ) {
    const r = placed.find((q) => q.node.key === key);
    if (!r) break;
    chain.unshift(r);
    key = r.parent;
  }
  const nodes = new Map(net.nodes.map((n) => [n.id, n] as const));
  return (
    <nav className="nv-crumbs" data-testid="predict-net-breadcrumb" aria-label="selected pathway">
      {chain.map((r, k) => (
        <span key={r.node.key} className="nv-crumb">
          {k > 0 && <span className="nv-crumb-step" style={{ color: typeVar(info.get(r.node.edgeId ?? "")?.type ?? "other") }}>→ {info.get(r.node.edgeId ?? "")?.type}</span>}
          <button type="button" className="legend-button" onClick={() => onKey(r.node.key)}>
            {nodes.get(r.node.nodeId) ? speciesLabel(nodes.get(r.node.nodeId)!) : r.node.nodeId}
          </button>
        </span>
      ))}
    </nav>
  );
}

export function PathwayExplorer(p: NetViewProps) {
  const { net } = p;
  const { rdkit } = useRDKit();
  const tree = useMemo(() => pathwayTree(net), [net]);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const rows = useMemo(() => treeRows(tree, collapsed), [tree, collapsed]);
  const nodesById = useMemo(() => new Map(net.nodes.map((n) => [n.id, n] as const)), [net]);
  const thumbs = useMemo(() => {
    const m = new Map<string, CardThumb | null>();
    for (const n of net.nodes) m.set(n.id, rdkit ? cardThumb(rdkit, n.species, n.display_smiles, THUMB_MAX_W, THUMB_MAX_H) : null);
    return m;
  }, [net, rdkit]);
  const placed = useMemo(() => place(rows, (r) => cardSize(thumbs.get(r.node.nodeId) ?? null, FOOT)), [rows, thumbs]);
  const back = useMemo(() => backEdges(net), [net]);
  const cycles = useMemo(() => {
    const m = new Map<string, number>();
    for (const e of net.edges) if (back.has(e.id)) m.set(e.source, (m.get(e.source) ?? 0) + 1);
    return m;
  }, [net, back]);
  const symbols = useSymbols(net);
  const info = useMemo(() => new Map(net.edges.map((e) => [e.id, stepInfo(e, symbols(e), p.probabilities)] as const)), [net, symbols, p.probabilities]);
  const sel = selectedKey(rows, p.selectedEdge, p.selectedNode);
  const [focus, setFocus] = useState<string | null>(null);
  const scroller = useRef<HTMLDivElement | null>(null);
  const byKey = new Map(placed.map((r) => [r.node.key, r] as const));
  const pick = (key: string) => {
    const r = byKey.get(key);
    if (!r) return;
    setFocus(key);
    if (r.node.edgeId) p.onSelectEdge(r.node.edgeId);
    else p.onSelectNode(r.node.nodeId);
  };
  const fold = (key: string) =>
    setCollapsed((s) => {
      const next = new Set(s);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  const width = Math.max(...placed.map((r) => r.x + r.w)) + PAD;
  const height = Math.max(...placed.map((r) => r.y + r.h)) + PAD;
  const onKeyDown = (e: KeyboardEvent) => {
    const cur = byKey.get(focus ?? sel ?? "") ?? placed[0];
    if (!cur) return;
    let next: Placed | undefined;
    if (e.key === "ArrowRight") next = placed.filter((r) => r.parent === cur.node.key).sort((a, b) => a.row - b.row)[0];
    else if (e.key === "ArrowLeft") next = cur.parent !== null ? byKey.get(cur.parent) : undefined;
    else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      const col = placed.filter((r) => r.node.depth === cur.node.depth && (e.key === "ArrowDown" ? r.row > cur.row : r.row < cur.row));
      next = col.sort((a, b) => Math.abs(a.row - cur.row) - Math.abs(b.row - cur.row))[0];
    } else if (e.key === "Enter") return void (e.preventDefault(), pick(cur.node.key));
    else if (e.key === "f" || e.key === "F") return void (e.preventDefault(), scroller.current?.scrollTo({ left: 0, top: 0, behavior: "smooth" }));
    else return;
    e.preventDefault();
    if (next) {
      setFocus(next.node.key);
      document.querySelector(`[data-testid="predict-net-card-${next.node.key || "root"}"]`)?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
    }
  };
  return (
    <div className="nv-stage nv-explorer-stage" data-testid="predict-network" data-variant="explorer" data-node-count={net.nodes.length} data-edge-count={net.edges.length} tabIndex={0} onKeyDown={onKeyDown} aria-label="ranked pathways as lanes; arrows move, Enter selects">
      <Breadcrumb net={net} placed={placed} sel={sel} info={info} onKey={pick} />
      <div ref={scroller} className="nv-explorer">
        <div className="nv-explorer-canvas" style={{ width, height }}>
          <svg className="nv-ribbons" width={width} height={height}>
            {placed
              .filter((r) => r.parent !== null)
              .map((r) => {
                const a = byKey.get(r.parent!)!;
                const s = info.get(r.node.edgeId ?? "");
                const on = sel !== null && (sel === r.node.key || sel.startsWith(`${r.node.key},`));
                const mid = { x: (a.x + a.w + r.x) / 2, y: (a.y + a.h / 2 + r.y + r.h / 2) / 2 };
                const w = 2 + 16 * r.node.weight;
                return (
                  <g key={r.node.key} className={`nv-ribbon${on ? " on" : ""}`}>
                    <path d={ribbon(a, r)} data-testid={`predict-net-ribbon-${r.node.key}`} data-weight={r.node.weight.toFixed(4)} style={{ strokeWidth: w, stroke: typeVar(s?.type ?? "other") }} />
                    <text x={mid.x} y={mid.y - w / 2 - 5} textAnchor="middle" className="nv-ribbon-label">
                      {s?.score}
                    </text>
                  </g>
                );
              })}
          </svg>
          {placed.map((r) => {
            const n = nodesById.get(r.node.nodeId);
            const key = r.node.key || "root";
            if (!n) return null;
            return (
              <StateCard
                key={key}
                n={n}
                root={r.parent === null}
                thumb={thumbs.get(n.id) ?? null}
                cycles={cycles.get(n.id) ?? 0}
                style={{ left: r.x, top: r.y, width: r.w, height: r.h }}
                testId={`predict-net-card-${key}`}
                className={focus === r.node.key ? "focus" : undefined}
                selected={sel === r.node.key}
                onClick={() => pick(r.node.key)}
              >
                <div className="nv-card-foot" data-edge={r.node.edgeId ?? ""}>
                  <span className="nv-ranks" title={`pathways ${r.node.ranks.map((k) => `#${k}`).join(", ")}`}>
                    {r.node.ranks.slice(0, 5).map((k) => (
                      <span key={k} className="nv-rank">
                        #{k}
                      </span>
                    ))}
                    {r.node.ranks.length > 5 && <span className="nv-rank nv-rank-more">+{r.node.ranks.length - 5}</span>}
                  </span>
                  {r.node.children.length > 0 && (
                    <button type="button" className="nv-fold" data-testid={`predict-net-fold-${key}`} aria-expanded={!r.collapsed} onClick={(e) => (e.stopPropagation(), fold(r.node.key))} title={r.collapsed ? `unfold ${r.hidden} hidden` : "fold this branch"}>
                      {r.collapsed ? `▸ ${r.hidden}` : "▾"}
                    </button>
                  )}
                </div>
              </StateCard>
            );
          })}
        </div>
      </div>
    </div>
  );
}
