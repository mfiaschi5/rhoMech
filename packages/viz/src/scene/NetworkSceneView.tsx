// NetworkSceneView: a network scene (scene/types.ts NetworkScene) in the network views of the
// Predict page, Scheme (SchemeView) and Pathways (PathwayExplorer), with the selection kept here
// (no expansion: a scene is a snapshot) and one line on the selected state or step.
//
// DOM: <section data-testid="scene-network" data-variant> with
// button[data-testid=scene-network-variant-<layered|explorer>][aria-pressed], the view's
// <div data-testid="predict-network"> and <div data-testid="scene-network-selection">.
import { useMemo, useState } from "react";
import { PathwayExplorer } from "../network/PathwayExplorer";
import { SchemeView } from "../network/SchemeView";
import { mappedSymbols } from "../network/netLayout";
import { speciesLabel, stateLine, stepInfo, VARIANTS, type NetViewProps, type Variant } from "../network/shared";
import type { NetworkScene } from "./types";

interface Selection {
  node: string | null;
  edge: string | null;
  pathway: string | null;
}

function SelectionLine({ scene, sel }: { scene: NetworkScene; sel: Selection }) {
  const net = scene.network;
  const edge = net.edges.find((e) => e.id === sel.edge);
  const node = net.nodes.find((n) => n.id === (edge?.target ?? sel.node));
  const info = edge ? stepInfo(edge, mappedSymbols(edge.source_mapped_smiles), net.scores_are_probabilities) : null;
  return (
    <div className="muted pv-wrap scene-network-selection" data-testid="scene-network-selection" data-node={node?.id ?? ""} data-edge={edge?.id ?? ""}>
      {node ? <b>{speciesLabel(node)}</b> : "click a state or a step"}
      {node && !info && <> · {stateLine(node)}</>}
      {info && (
        <>
          {" "}
          · {info.type} · {info.score} · {info.reading || "(no reading)"} · R: {info.r}
        </>
      )}
    </div>
  );
}

export function NetworkSceneView({ scene, height = 520 }: { scene: NetworkScene; height?: number }) {
  const net = scene.network;
  const [variant, setVariant] = useState<Variant>(scene.variant ?? "layered");
  const [sel, setSel] = useState<Selection>({ node: null, edge: null, pathway: null });
  const pathway = useMemo(() => net.pathways.find((w) => w.id === sel.pathway) ?? net.pathways[0] ?? null, [net, sel.pathway]);
  const p: NetViewProps = {
    net,
    selectedNode: sel.node,
    selectedEdge: sel.edge,
    pathway,
    pathwayFocus: sel.pathway !== null,
    probabilities: net.scores_are_probabilities,
    onSelectNode: (id) => setSel((s) => ({ ...s, node: id, edge: null })),
    onSelectEdge: (id) => setSel((s) => ({ ...s, edge: id, node: net.edges.find((e) => e.id === id)?.target ?? s.node })),
    onSelectPathway: (id) => setSel((s) => ({ ...s, pathway: id })),
  };
  return (
    <section className="scene-network stack" data-testid="scene-network" data-variant={variant}>
      <span className="nv-switch" role="group" aria-label="network view">
        {VARIANTS.map((v) => (
          <button key={v.id} type="button" className={`nv-seg${variant === v.id ? " on" : ""}`} data-testid={`scene-network-variant-${v.id}`} aria-pressed={variant === v.id} title={v.title} onClick={() => setVariant(v.id)}>
            {v.label}
          </button>
        ))}
      </span>
      <div className="nv-frame" style={{ height }}>
        {variant === "layered" ? <SchemeView {...p} /> : <PathwayExplorer {...p} />}
      </div>
      <SelectionLine scene={scene} sel={sel} />
    </section>
  );
}
