// NetworkPanel: the Predict job's network viewport, in one of two variants picked by the
// segmented control in its header (remembered in localStorage): Scheme (the network as a
// reaction scheme) and Pathways (the ranked pathways as merged lanes). The viewport has a fixed
// height (60 % of the window at first) that the splitter under it resizes; the network list (the
// DOM mirror of every state and step) stays below as a collapsible secondary view.
//
// DOM: <section class="panel pv-panel"> with div[data-testid=predict-network-variant] >
// button[data-testid=predict-network-variant-<layered|explorer>][aria-pressed], the variant's
// <div data-testid="predict-network" data-variant>, the splitter predict-network-resize
// (role=separator, arrow keys too), and the NetworkLegend <details data-testid="predict-network-list">.
import { useRef, type KeyboardEvent, type PointerEvent as RPointerEvent } from "react";
import { NetworkLegend } from "../NetworkLegend";
import { Panel } from "../Panel";
import { PathwayExplorer } from "./PathwayExplorer";
import { SchemeView } from "./SchemeView";
import { useVariant, VARIANTS, type NetViewProps } from "./shared";
import "./netview.css";

export interface NetworkPanelProps extends NetViewProps {
  height: number;
  onHeight: (h: number) => void;
}

function Splitter({ height, onHeight }: { height: number; onHeight: (h: number) => void }) {
  const start = useRef<{ y: number; h: number } | null>(null);
  const down = (e: RPointerEvent<HTMLDivElement>) => {
    start.current = { y: e.clientY, h: height };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const move = (e: RPointerEvent<HTMLDivElement>) => start.current && onHeight(start.current.h + e.clientY - start.current.y);
  const key = (e: KeyboardEvent) => {
    if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      e.preventDefault();
      onHeight(height + (e.key === "ArrowDown" ? 40 : -40));
    }
  };
  return <div className="nv-splitter" role="separator" aria-orientation="horizontal" aria-label="network height" aria-valuenow={height} tabIndex={0} data-testid="predict-network-resize" onPointerDown={down} onPointerMove={move} onPointerUp={() => (start.current = null)} onKeyDown={key} />;
}

export function NetworkPanel(p: NetworkPanelProps) {
  const [variant, setVariant] = useVariant();
  const { net } = p;
  const tools = (
    <span className="nv-switch" role="group" aria-label="network view" data-testid="predict-network-variant" data-variant={variant}>
      {VARIANTS.map((v) => (
        <button key={v.id} type="button" className={`nv-seg${variant === v.id ? " on" : ""}`} data-testid={`predict-network-variant-${v.id}`} aria-pressed={variant === v.id} title={v.title} onClick={() => setVariant(v.id)}>
          {v.label}
        </button>
      ))}
    </span>
  );
  const note = `revision ${net.revision}${net.complete ? "" : ", incomplete"} · ${net.nodes.length} states, ${net.edges.length} steps`;
  return (
    <Panel title="Network" note={note} tools={tools} className="pv-network-panel">
      <div className="nv-frame" style={{ height: p.height }}>
        {variant === "layered" ? <SchemeView {...p} /> : <PathwayExplorer {...p} />}
      </div>
      <Splitter height={p.height} onHeight={p.onHeight} />
      <NetworkLegend net={net} selectedNode={p.selectedNode} selectedEdge={p.selectedEdge} pathway={p.pathway} onSelectNode={p.onSelectNode} onSelectEdge={p.onSelectEdge} probabilities={p.probabilities} />
    </Panel>
  );
}
