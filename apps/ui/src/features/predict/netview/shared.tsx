// What the three network viewports share: their props, the variant preference (localStorage),
// the keyboard (arrows through neighbours, Enter selects, F fits), the per-state facts a tooltip
// shows, and the colours of the step types (CSS variables of netview.css, light and dark).
import { useEffect, useMemo, useState, type KeyboardEvent } from "react";
import { fmt, fmtSig } from "../../../components/format";
import { matrixText, speciesLabel } from "../network";
import type { EdgeOut, NetworkOut, NodeOut, PathwayOut } from "../v9types";
import { mappedSymbols, neighbour, readingLabel, stepType, type Direction, type StepType, type XY } from "./netLayout";

export interface NetViewProps {
  jobId: string;
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
  /** The network changed on the server (an expansion): reload it. */
  onNetworkChanged: () => void;
}

export type Variant = "layered" | "explorer";
export const VARIANTS: { id: Variant; label: string; title: string }[] = [
  { id: "layered", label: "Scheme", title: "The network as a reaction scheme, left to right by depth" },
  { id: "explorer", label: "Pathways", title: "The ranked pathways as merged lanes" },
];
const KEY = "ksmech.predict.networkVariant";
const HEIGHT_KEY = "ksmech.predict.networkHeight";
export const MIN_NET_H = 320;
export const MAX_NET_H = 1200;

/** The network viewport's height (px): 60 % of the window at first, then what the user dragged
 *  the splitter to (localStorage). */
export function useNetHeight(): [number, (h: number) => void] {
  const [h, setH] = useState<number>(() => {
    let v = Number.NaN;
    try {
      v = Number(localStorage.getItem(HEIGHT_KEY));
    } catch {
      // no storage: the default
    }
    return Number.isFinite(v) && v >= MIN_NET_H ? Math.min(MAX_NET_H, v) : Math.max(MIN_NET_H, Math.round(window.innerHeight * 0.6));
  });
  const set = (next: number) => {
    const c = Math.round(Math.max(MIN_NET_H, Math.min(MAX_NET_H, next)));
    setH(c);
    try {
      localStorage.setItem(HEIGHT_KEY, String(c));
    } catch {
      // private window
    }
  };
  return [h, set];
}

export function useVariant(): [Variant, (v: Variant) => void] {
  const [v, setV] = useState<Variant>(() => {
    try {
      const s = localStorage.getItem(KEY);
      return VARIANTS.some((x) => x.id === s) ? (s as Variant) : "layered";
    } catch {
      return "layered";
    }
  });
  const set = (next: Variant) => {
    setV(next);
    try {
      localStorage.setItem(KEY, next);
    } catch {
      // private window: the choice lasts for this page only
    }
  };
  return [v, set];
}

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
