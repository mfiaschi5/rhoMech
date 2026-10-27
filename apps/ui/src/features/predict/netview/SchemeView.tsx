// A. Scheme: the network as a reaction scheme, left to right by depth. Card sizes come first (the
// structures at one bond length, cardThumb), then schemeLayout stacks every column with fixed
// gaps in barycentre order (sub-columns when tall) and centres it: no card overlaps another.
// Steps are smooth curves from a card's right mid-height to the next card's left, as wide as their
// strength (beam weight), with one arrowhead; hovering one shows its score and reading. Cycle
// closures are hidden behind a "↺ n" chip on the card where they close and drawn (dashed arcs above
// the cards) on hover, on selection or with "Show cycles". Hovering a card lights its in/out
// paths; a picked pathway lights up while the rest fades. Column bands with sticky headers, pan
// (drag), zoom (wheel, buttons), animated fit (F), a minimap, SVG and PNG export; a double-click
// on a card with no step out expands it (POST /jobs/{id}/expand) and the network reloads.
//
// DOM: <div data-testid="predict-network" data-variant="layered" data-node-count data-edge-count
// tabIndex=0> with the column headers predict-net-col-<layer>, cards predict-net-node-<id>
// (StateCard), step paths predict-net-edge-<id>[data-back], predict-net-tooltip, predict-net-fit,
// predict-net-zoom-in/out, predict-net-cycles[aria-pressed], predict-net-export-svg/png,
// predict-net-minimap (once zoomed in past the fit), predict-net-expand-status.
import { useEffect, useMemo, useRef, useState, type PointerEvent as RPointerEvent, type WheelEvent } from "react";
import { toApiError } from "../../../api/client";
import { pollJob } from "../../../api/poll";
import { useRDKit } from "../../../chem/rdkit";
import { expandBody } from "../checkpoints";
import { expandNode } from "../v9api";
import type { NetworkOut } from "../v9types";
import { backEdges, edgeStrength, schemeLayout, type Rect, type Scheme } from "./netLayout";
import { cssVar, download, speciesLabel, stepInfo, useGraphKeys, useSymbols, type NetViewProps, type StepInfo } from "./shared";
import { cardSize, StateCard, THUMB_MAX_H, THUMB_MAX_W } from "./StateCard";
import { cardThumb, type CardThumb } from "./thumbs";

interface View {
  k: number;
  x: number;
  y: number;
  animate: boolean;
}

const HEADER_H = 30;

/** Forward-step neighbourhood of a card: everything upstream and downstream of it. */
function pathsThrough(net: NetworkOut, id: string, back: Set<string>): Set<string> {
  const fwd = net.edges.filter((e) => !back.has(e.id));
  const out = new Set<string>([id]);
  const walk = (start: string, dir: "up" | "down") => {
    const stack = [start];
    while (stack.length) {
      const cur = stack.pop()!;
      for (const e of fwd) {
        const [from, to] = dir === "down" ? [e.source, e.target] : [e.target, e.source];
        if (from !== cur) continue;
        out.add(e.id);
        if (!out.has(to)) {
          out.add(to);
          stack.push(to);
        }
      }
    }
  };
  walk(id, "down");
  walk(id, "up");
  return out;
}

function forwardPath(a: Rect, b: Rect): { d: string; mid: [number, number] } {
  const [x0, y0, x1, y1] = [a.x + a.w, a.y + a.h / 2, b.x - 7, b.y + b.h / 2];
  const dx = Math.max(40, (x1 - x0) * 0.5);
  return { d: `M ${x0} ${y0} C ${x0 + dx} ${y0} ${x1 - dx} ${y1} ${x1} ${y1}`, mid: [(x0 + x1) / 2, (y0 + y1) / 2] };
}

function backPath(a: Rect, b: Rect): { d: string; mid: [number, number] } {
  const [x0, y0, x1, y1] = [a.x + a.w / 2, a.y, b.x + b.w / 2, b.y - 7];
  const lift = 50 + Math.abs(x1 - x0) * 0.18;
  const top = Math.min(y0, y1) - lift;
  return { d: `M ${x0} ${y0} C ${x0} ${top} ${x1} ${top} ${x1} ${y1}`, mid: [(x0 + x1) / 2, top + lift * 0.25] };
}

function usePanZoom(stage: React.RefObject<HTMLDivElement | null>, bounds: Rect) {
  const [view, setView] = useState<View>({ k: 1, x: 40, y: 200, animate: false });
  const fitK = useRef(1);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const fit = (animate = true) => {
    const el = stage.current;
    if (!el || !bounds.w) return;
    const [W, H] = [el.clientWidth, el.clientHeight - HEADER_H];
    const k = Math.min(1.1, (W - 48) / bounds.w, (H - 40) / bounds.h);
    fitK.current = k;
    setView({ k, x: (W - bounds.w * k) / 2 - bounds.x * k, y: HEADER_H + (H - bounds.h * k) / 2 - bounds.y * k, animate });
  };
  const zoomAt = (factor: number, cx: number, cy: number, animate = false) =>
    setView((v) => {
      const k = Math.max(0.15, Math.min(2.5, v.k * factor));
      return { k, x: cx - ((cx - v.x) * k) / v.k, y: cy - ((cy - v.y) * k) / v.k, animate };
    });
  const onWheel = (e: WheelEvent) => {
    const r = stage.current!.getBoundingClientRect();
    zoomAt(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.clientX - r.left, e.clientY - r.top);
  };
  const onPointerDown = (e: RPointerEvent) => {
    if ((e.target as Element).closest(".nv-card-state, .nv-tools, .nv-minimap, .nv-edge-hit")) return;
    drag.current = { x: e.clientX, y: e.clientY };
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: RPointerEvent) => {
    const d = drag.current;
    if (!d) return;
    setView((v) => ({ ...v, x: v.x + e.clientX - d.x, y: v.y + e.clientY - d.y, animate: false }));
    [d.x, d.y] = [e.clientX, e.clientY];
  };
  const onPointerUp = () => void (drag.current = null);
  return { view, setView, fit, zoomAt, fitK, handlers: { onWheel, onPointerDown, onPointerMove, onPointerUp } };
}

/** The scheme as a standalone SVG (bands, steps, cards with their structures and captions). */
function schemeSvg(net: NetworkOut, sch: Scheme, thumbs: Map<string, CardThumb | null>, strength: Map<string, number>, back: Set<string>): string {
  const c = { fg: cssVar("--fg", "#1c2330"), muted: cssVar("--muted", "#5d6878"), border: cssVar("--border", "#d5dbe3"), panel: cssVar("--panel", "#ffffff"), card: cssVar("--depiction-bg", "#ffffff"), accent: cssVar("--accent", "#2463c7") };
  const b = sch.bounds;
  const [x0, y0, w, h] = [b.x - 40, b.y - 60, b.w + 80, b.h + 100];
  const esc = (t: string) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const edges = net.edges
    .filter((e) => !back.has(e.id))
    .map((e) => `<path d="${forwardPath(sch.rects.get(e.source)!, sch.rects.get(e.target)!).d}" fill="none" stroke="${c.muted}" stroke-width="${(1.2 + 2.6 * (strength.get(e.id) ?? 0.3)).toFixed(1)}" opacity="0.7"/>`);
  const heads = sch.columns.map((col) => `<text x="${col.x0}" y="${b.y - 30}" font-family="sans-serif" font-size="12" font-weight="600" fill="${c.muted}">${col.layer === 0 ? "REACTANTS" : `STEP ${col.layer}`} (${col.count})</text>`);
  const cards = net.nodes.map((n) => {
    const r = sch.rects.get(n.id)!;
    const t = thumbs.get(n.id);
    const tx = r.x + (r.w - (t?.w ?? 0)) / 2;
    return `<g><rect x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" rx="8" fill="${c.card}" stroke="${n.id === net.root ? c.accent : c.border}" stroke-width="1.5"/>${t ? `<image href="${t.uri}" x="${tx}" y="${r.y + 24}" width="${t.w}" height="${t.h}"/>` : ""}<text x="${r.x + r.w / 2}" y="${r.y + r.h - 7}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#5d6878">${esc(speciesLabel(n))}</text></g>`;
  });
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${x0} ${y0} ${w} ${h}" width="${w}" height="${h}"><rect x="${x0}" y="${y0}" width="${w}" height="${h}" fill="${c.panel}"/>${heads.join("")}${edges.join("")}${cards.join("")}</svg>`;
}

function exportPng(svg: string): void {
  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = img.width * 2;
    canvas.height = img.height * 2;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(2, 2);
    ctx.drawImage(img, 0, 0);
    canvas.toBlob((b) => b && download(b, "ksmech-network.png"), "image/png");
  };
  img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function useExpand(p: NetViewProps) {
  const [status, setStatus] = useState<string | null>(null);
  const expand = async (id: string) => {
    setStatus(`expanding ${id}…`);
    try {
      const created = await expandNode(p.jobId, expandBody(p.net.provenance, id, 1));
      const done = await pollJob(created.job_id);
      setStatus(done.status === "done" ? null : `expansion ${done.status}`);
      if (done.status === "done") p.onNetworkChanged();
    } catch (err) {
      setStatus(`cannot expand: ${toApiError(err).message}`);
    }
  };
  return { status, expand };
}

export function SchemeView(p: NetViewProps) {
  const { net } = p;
  const { rdkit } = useRDKit();
  const thumbs = useMemo(() => new Map(net.nodes.map((n) => [n.id, rdkit ? cardThumb(rdkit, n.species, n.display_smiles, THUMB_MAX_W, THUMB_MAX_H) : null] as const)), [net, rdkit]);
  const sch = useMemo(() => schemeLayout(net, (id) => cardSize(thumbs.get(id) ?? null), { colGap: 78, subGap: 20, minWidth: 136 }), [net, thumbs]);
  const back = useMemo(() => backEdges(net), [net]);
  const strength = useMemo(() => edgeStrength(net), [net]);
  const symbols = useSymbols(net);
  const info = useMemo(() => new Map(net.edges.map((e) => [e.id, stepInfo(e, symbols(e), p.probabilities)] as const)), [net, symbols, p.probabilities]);
  const cycles = useMemo(() => {
    const m = new Map<string, number>();
    for (const e of net.edges) if (back.has(e.id)) m.set(e.source, (m.get(e.source) ?? 0) + 1);
    return m;
  }, [net, back]);
  const stage = useRef<HTMLDivElement | null>(null);
  const pz = usePanZoom(stage, sch.bounds);
  const [hoverNode, setHoverNode] = useState<string | null>(null);
  const [hoverEdge, setHoverEdge] = useState<string | null>(null);
  const [showCycles, setShowCycles] = useState(false);
  const { status, expand } = useExpand(p);
  const keys = useGraphKeys(net, sch.pos, p.selectedNode, p.onSelectNode, () => pz.fit());
  useEffect(() => {
    const t = setTimeout(() => pz.fit(false), 0);
    return () => clearTimeout(t);
    // a new layout (network, thumbnails) is fitted once
  }, [sch]);
  const lit = useMemo(() => (hoverNode ? pathsThrough(net, hoverNode, back) : null), [net, hoverNode, back]);
  const inPath = new Set([...(p.pathway?.node_ids ?? []), ...(p.pathway?.edge_ids ?? [])]);
  const faded = (id: string) => (lit ? !lit.has(id) : p.pathwayFocus && !inPath.has(id));
  const focusNode = hoverNode ?? p.selectedNode;
  const showBack = (eid: string) => {
    const e = net.edges.find((x) => x.id === eid)!;
    return showCycles || p.selectedEdge === eid || e.source === focusNode || e.target === focusNode;
  };
  const hasOut = (id: string) => net.edges.some((e) => e.source === id && !back.has(e.id));
  const hoverInfo = hoverEdge ? info.get(hoverEdge) : undefined;
  const hoverMid = hoverEdge ? edgeMid(net, sch, back, hoverEdge) : null;
  const { view } = pz;
  return (
    <div ref={stage} className="nv-stage nv-scheme" data-testid="predict-network" data-variant="layered" data-node-count={net.nodes.length} data-edge-count={net.edges.length} tabIndex={0} onKeyDown={keys.onKeyDown} {...pz.handlers} aria-label="reaction network, left to right by depth; arrows move, Enter selects, F fits">
      <div className={`nv-world${view.animate ? " nv-animate" : ""}`} style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.k})` }}>
        {sch.columns.map((c, k) => (
          <div key={c.layer} className={`nv-band${k % 2 ? " nv-band-alt" : ""}`} style={{ left: c.x0 - 30, width: c.x1 - c.x0 + 60, top: sch.bounds.y - 4000, height: sch.bounds.h + 8000 }} />
        ))}
        <svg className="nv-edges" style={{ left: 0, top: 0 }} width={1} height={1}>
          <defs>
            <marker id="nv-arrow" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" className="nv-arrowhead" />
            </marker>
            <marker id="nv-arrow-cycle" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" className="nv-arrowhead-cycle" />
            </marker>
          </defs>
          {net.edges.map((e) => {
            const isBack = back.has(e.id);
            if (isBack && !showBack(e.id)) return null;
            const a = sch.rects.get(e.source);
            const b = sch.rects.get(e.target);
            if (!a || !b) return null;
            const { d } = isBack ? backPath(a, b) : forwardPath(a, b);
            const s = strength.get(e.id) ?? 0.3;
            const cls = `nv-step${isBack ? " back" : ""}${inPath.has(e.id) ? " inpath" : ""}${p.selectedEdge === e.id ? " selected" : ""}${lit?.has(e.id) || hoverEdge === e.id ? " hl" : ""}${faded(e.id) ? " faded" : ""}`;
            return (
              <g key={e.id} className={cls}>
                <path d={d} className="nv-edge-hit" onClick={() => p.onSelectEdge(e.id)} onPointerEnter={() => setHoverEdge(e.id)} onPointerLeave={() => setHoverEdge(null)} />
                <path d={d} className="nv-step-line" data-testid={`predict-net-edge-${e.id}`} data-back={isBack ? "true" : "false"} style={{ strokeWidth: isBack ? 1.4 : 1.2 + 2.6 * s, opacity: isBack ? undefined : 0.38 + 0.45 * s }} markerEnd={isBack ? "url(#nv-arrow-cycle)" : "url(#nv-arrow)"} />
              </g>
            );
          })}
          {hoverInfo && hoverMid && (
            <text x={hoverMid[0]} y={hoverMid[1] - 8} textAnchor="middle" className="nv-step-label">
              {hoverInfo.score}
            </text>
          )}
        </svg>
        {net.nodes.map((n) => {
          const r = sch.rects.get(n.id)!;
          return (
            <StateCard
              key={n.id}
              n={n}
              root={n.id === net.root}
              thumb={thumbs.get(n.id) ?? null}
              cycles={cycles.get(n.id) ?? 0}
              style={{ left: r.x, top: r.y, width: r.w, height: r.h }}
              testId={`predict-net-node-${n.id}`}
              className={`${inPath.has(n.id) ? "inpath" : ""}${keys.focus === n.id ? " focus" : ""}${faded(n.id) ? " faded" : ""}${lit?.has(n.id) ? " hl" : ""}`}
              selected={p.selectedNode === n.id}
              onClick={() => p.onSelectNode(n.id)}
              onDoubleClick={hasOut(n.id) ? undefined : () => void expand(n.id)}
              onHover={(on) => setHoverNode(on ? n.id : null)}
            />
          );
        })}
      </div>
      <ColumnHeaders sch={sch} view={view} />
      <Tools pz={pz} showCycles={showCycles} setShowCycles={setShowCycles} nCycles={back.size} onSvg={() => download(new Blob([schemeSvg(net, sch, thumbs, strength, back)], { type: "image/svg+xml" }), "ksmech-network.svg")} onPng={() => exportPng(schemeSvg(net, sch, thumbs, strength, back))} />
      {hoverInfo && <StepTip s={hoverInfo} />}
      {status && (
        <div className="nv-status" data-testid="predict-net-expand-status">
          {status}
        </div>
      )}
      {view.k > pz.fitK.current * 1.15 && <Minimap sch={sch} view={view} stage={stage.current} onCentre={(x, y) => pz.setView((v) => ({ ...v, x: (stage.current?.clientWidth ?? 0) / 2 - x * v.k, y: (stage.current?.clientHeight ?? 0) / 2 - y * v.k, animate: true }))} />}
    </div>
  );
}

function edgeMid(net: NetworkOut, sch: Scheme, back: Set<string>, id: string): [number, number] | null {
  const e = net.edges.find((x) => x.id === id);
  const a = e && sch.rects.get(e.source);
  const b = e && sch.rects.get(e.target);
  if (!e || !a || !b) return null;
  return (back.has(id) ? backPath(a, b) : forwardPath(a, b)).mid;
}

function ColumnHeaders({ sch, view }: { sch: Scheme; view: View }) {
  return (
    <div className="nv-col-heads" aria-hidden="true">
      {sch.columns.map((c) => (
        <div key={c.layer} className="nv-col-head" data-testid={`predict-net-col-${c.layer}`} style={{ left: (c.x0 - 30) * view.k + view.x, width: (c.x1 - c.x0 + 60) * view.k }}>
          <span className="nv-col-title">{c.layer === 0 ? "Reactants" : `Step ${c.layer}`}</span>
          <span className="nv-count">{c.count}</span>
        </div>
      ))}
    </div>
  );
}

function Tools({ pz, showCycles, setShowCycles, nCycles, onSvg, onPng }: { pz: ReturnType<typeof usePanZoom>; showCycles: boolean; setShowCycles: (v: boolean) => void; nCycles: number; onSvg: () => void; onPng: () => void }) {
  const centre = (f: number) => pz.zoomAt(f, 400, 260, true);
  return (
    <div className="nv-tools">
      <button type="button" className={`btn btn-small${showCycles ? " btn-primary" : ""}`} data-testid="predict-net-cycles" aria-pressed={showCycles} disabled={!nCycles} onClick={() => setShowCycles(!showCycles)} title="draw every cycle closure (otherwise on hover or selection)">
        ↺ cycles {nCycles}
      </button>
      <button type="button" className="btn btn-small" data-testid="predict-net-fit" onClick={() => pz.fit()} title="fit (F)">
        Fit
      </button>
      <button type="button" className="btn btn-small" data-testid="predict-net-zoom-in" onClick={() => centre(1.25)} aria-label="zoom in">
        +
      </button>
      <button type="button" className="btn btn-small" data-testid="predict-net-zoom-out" onClick={() => centre(0.8)} aria-label="zoom out">
        −
      </button>
      <button type="button" className="btn btn-small" data-testid="predict-net-export-svg" onClick={onSvg}>
        SVG
      </button>
      <button type="button" className="btn btn-small" data-testid="predict-net-export-png" onClick={onPng}>
        PNG
      </button>
    </div>
  );
}

function StepTip({ s }: { s: StepInfo }) {
  return (
    <div className="nv-tooltip nv-tip-corner" data-testid="predict-net-tooltip">
      <strong>
        {s.type} · {s.score}
      </strong>
      <div className="muted">{s.reading || "(no reading)"}</div>
      <div className="muted">R: {s.r}</div>
    </div>
  );
}

function Minimap({ sch, view, stage, onCentre }: { sch: Scheme; view: View; stage: HTMLDivElement | null; onCentre: (x: number, y: number) => void }) {
  const b = sch.bounds;
  const W = 132;
  const H = 76;
  const s = Math.min(W / (b.w + 80), H / (b.h + 80));
  const mx = (x: number) => (x - b.x + 40) * s;
  const my = (y: number) => (y - b.y + 40) * s;
  const vw = (stage?.clientWidth ?? 0) / view.k;
  const vh = (stage?.clientHeight ?? 0) / view.k;
  return (
    <svg className="nv-minimap" data-testid="predict-net-minimap" width={W} height={H} onPointerDown={(e) => {
      const r = e.currentTarget.getBoundingClientRect();
      onCentre((e.clientX - r.left) / s + b.x - 40, (e.clientY - r.top) / s + b.y - 40);
    }}>
      {[...sch.rects.values()].map((r, k) => (
        <rect key={k} x={mx(r.x)} y={my(r.y)} width={r.w * s} height={r.h * s} className="nv-mini-node" />
      ))}
      <rect className="nv-mini-view" x={mx(-view.x / view.k)} y={my(-view.y / view.k)} width={vw * s} height={vh * s} />
    </svg>
  );
}
