// What the network viewports share: their props, the variants, the keyboard (arrows through neighbours, Enter selects, F fits), the per-state facts a tooltip
// shows, and the colours of the step types (CSS variables of netview.css, light and dark).
import { useEffect, useMemo, useState, type KeyboardEvent } from "react";
import { fmt, fmtSig } from "../format";
import { matrixText, speciesLabel } from "./network";
import type { EdgeOut, NetworkOut, NodeOut, PathwayOut } from "../v9types";
import { mappedSymbols, neighbour, readingLabel, stepType, type Direction, type StepType, type XY } from "./netLayout";
import "./netview.css";

export interface NetViewProps {
  net: NetworkOut;
  selectedNode: string | null;
  selectedEdge: string | null;
  pathway: PathwayOut | null;
  /** The user picked the pathway (not the default rank 1): everything else fades. */
  pathwayFocus: boolean;
  probabilities: boolean;
  onSelectNode: (id: string) => void;
  onSelectEdge: (id: string) => void;
  onSelectPathway: (id: string) => void;
  /** Expand a state with no step out (a double-click on its card; none: no expansion). */
  onExpand?: (nodeId: string) => void;
  /** What the expansion is doing ("expanding n3…", an error), shown over the scheme. */
  expandStatus?: string | null;
}

export type Variant = "layered" | "explorer";
export const VARIANTS: { id: Variant; label: string; title: string }[] = [
  { id: "layered", label: "Scheme", title: "The network as a reaction scheme, left to right by depth" },
  { id: "explorer", label: "Pathways", title: "The ranked pathways as merged lanes" },
];
const ARROWS: Record<string, Direction> = { ArrowLeft: "left", ArrowRight: "right", ArrowUp: "up", ArrowDown: "down" };

/** Keyboard focus over a layered position map: arrows move it, Enter selects it, F fits. */
export function useGraphKeys(net: NetworkOut, pos: ReadonlyMap<string, XY & { layer: number }>, selected: string | null, onSelect: (id: string) => void, onFit: () => void) {
  const [focus, setFocus] = useState<string | null>(null);
  useEffect(() => {
    if (selected) setFocus(selected);
  }, [selected]);
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "f" || e.key === "F") return void (e.preventDefault(), onFit());
    const cur = focus ?? net.root;
    if (e.key === "Enter") return void (e.preventDefault(), onSelect(cur));
    const dir = ARROWS[e.key];
    if (!dir) return;
    e.preventDefault();
    const next = focus ? neighbour(net, pos, cur, dir) : net.root;
    if (next) setFocus(next);
  };
  return { focus, setFocus, onKeyDown };
}

/** Element symbols of each state's pool atoms, from its edges' mapped SMILES. */
export function useSymbols(net: NetworkOut): (e: EdgeOut) => string[] {
  return useMemo(() => {
    const cache = new Map<string, string[]>();
    return (e: EdgeOut) => {
      if (!cache.has(e.source_state_id)) cache.set(e.source_state_id, mappedSymbols(e.source_mapped_smiles));
      return cache.get(e.source_state_id)!;
    };
  }, [net]);
}

export interface StepInfo {
  type: StepType;
  reading: string;
  score: string;
  r: string;
}

export function stepInfo(e: EdgeOut, symbols: string[], probabilities: boolean): StepInfo {
  const label = (i: number) => `${symbols[i] ?? ""}${i}`;
  const score = e.kind === "corrected" ? "user step" : probabilities && e.logp !== null ? `p ${fmt(Math.exp(e.logp), 3)}` : `score ${fmtSig(e.score, 3)}`;
  return { type: stepType(e.arrows, symbols), reading: readingLabel(e.arrows, symbols), score, r: matrixText(e.matrix, label) };
}

/** "depth 2 · terminal (RECURRENCE)" for a state. */
export function stateLine(n: NodeOut): string {
  return `${n.is_root ? "reactants" : `depth ${n.depth}`} · ${n.status}${n.terminal_reason ? ` (${n.terminal_reason})` : ""}`;
}

export { speciesLabel };

/** The colour token of a step type (netview.css). */
export const typeVar = (t: StepType) => `var(--st-${t.replace(" ", "-")})`;

/** A CSS variable's current value (cytoscape and canvas need plain colours). */
export function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** Hand a blob to the browser as a download. */
export function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
