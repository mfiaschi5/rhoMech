// StateCard: one pool state as the scheme (A) and the pathway explorer (C) draw it: a header of
// subtle chips (reactants, the terminal reason, pruned, failed, "↺ n" cycles closing here), the
// species at one bond length with "+" between them (cardThumb), and the SMILES caption (ellipsis,
// the full text on hover). The size comes from cardSize, so layouts can place cards before they
// render.
//
// DOM: <div class="nv-card-state status-<s>" data-testid=<testId> data-node data-selected> with
// span.nv-chip[data-chip] chips, the structure <img>, div.nv-caption[title], then children.
import type { ReactNode } from "react";
import type { NodeOut } from "../v9types";
import { speciesLabel } from "./shared";
import type { CardThumb } from "./thumbs";

export const CARD_HEAD = 22;
export const CARD_CAPTION = 20;
export const CARD_PAD = 10;
export const THUMB_MAX_W = 300;
export const THUMB_MAX_H = 120;

/** The card size of a thumbnail (width before the column makes it uniform). */
export function cardSize(t: { w: number; h: number } | null, footer = 0): { w: number; h: number } {
  const w = Math.max(150, (t?.w ?? 120) + 2 * CARD_PAD);
  const h = CARD_HEAD + Math.max(t?.h ?? 50, 40) + 8 + CARD_CAPTION + footer;
  return { w, h };
}

export interface Chip {
  key: string;
  text: string;
}

export function stateChips(n: NodeOut, root: boolean, cycles: number): Chip[] {
  const out: Chip[] = [];
  if (root) out.push({ key: "root", text: "reactants" });
  if (n.status === "terminal") out.push({ key: "terminal", text: (n.terminal_reason ?? "terminal").toLowerCase().replace(/_/g, " ") });
  if (n.status === "pruned") out.push({ key: "pruned", text: "pruned" });
  if (n.status === "failed") out.push({ key: "failed", text: n.failure?.code ?? "failed" });
  if (n.status === "frontier") out.push({ key: "frontier", text: "frontier" });
  if (cycles > 0) out.push({ key: "cycle", text: `↺ ${cycles}` });
  return out;
}

export interface StateCardProps {
  n: NodeOut;
  root: boolean;
  thumb: CardThumb | null;
  cycles: number;
  style: React.CSSProperties;
  testId: string;
  className?: string;
  selected?: boolean;
  onClick?: () => void;
  onDoubleClick?: () => void;
  onHover?: (on: boolean) => void;
  children?: ReactNode;
}

export function StateCard(p: StateCardProps) {
  const chips = stateChips(p.n, p.root, p.cycles);
  const label = speciesLabel(p.n);
  return (
    <div
      className={`nv-card-state status-${p.n.status}${p.root ? " root" : ""}${p.selected ? " selected" : ""}${p.className ? ` ${p.className}` : ""}`}
      style={p.style}
      data-testid={p.testId}
      data-node={p.n.id}
      data-selected={p.selected ? "true" : "false"}
      onClick={p.onClick}
      onDoubleClick={p.onDoubleClick}
      onPointerEnter={() => p.onHover?.(true)}
      onPointerLeave={() => p.onHover?.(false)}
    >
      <div className="nv-card-head">
        {chips.map((c) => (
          <span key={c.key} className={`nv-chip nv-chip-${c.key}`} data-chip={c.key}>
            {c.text}
          </span>
        ))}
      </div>
      <div className="nv-card-body">{p.thumb ? <img src={p.thumb.uri} alt={label} width={p.thumb.w} height={p.thumb.h} draggable={false} /> : <span className="muted">{label}</span>}</div>
      <div className="nv-caption" title={`${label}\n${p.n.display_smiles}`}>
        {label}
      </div>
      {p.children}
    </div>
  );
}
