// Collision-aware placement of the R overlay's badges (lone-pair ±2, formal charges), in screen
// px (pure; node --test runs it). Each badge tries rings of positions around its atom, starting
// at its preferred direction and turning by 45° both ways, nearest ring first; the first box that
// touches no obstacle (atom labels, points sampled along bonds and arrows, badges already placed)
// wins, else the one with the least overlap.
import type { Vec } from "./arrowPath";

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface BadgeRequest {
  key: string;
  at: Vec;
  w: number;
  h: number;
  /** Preferred direction from the atom (radians, screen y down). */
  prefer: number;
}

export const RINGS = [0.6, 0.85, 1.1];
const TURNS = [0, 45, -45, 90, -90, 135, -135, 180].map((d) => (d * Math.PI) / 180);

export const boxAround = (p: Vec, hw: number, hh: number): Box => ({ x0: p[0] - hw, y0: p[1] - hh, x1: p[0] + hw, y1: p[1] + hh });

function overlapArea(a: Box, b: Box): number {
  const w = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0);
  const h = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0);
  return w > 0 && h > 0 ? w * h : 0;
}

/** Small boxes every `step` px along p-q (a bond or a curve piece as obstacles). */
export function segmentBoxes(p: Vec, q: Vec, step = 4, r = 1.5): Box[] {
  const n = Math.max(1, Math.ceil(Math.hypot(q[0] - p[0], q[1] - p[1]) / step));
  return Array.from({ length: n + 1 }, (_, k) => boxAround([p[0] + ((q[0] - p[0]) * k) / n, p[1] + ((q[1] - p[1]) * k) / n], r, r));
}

/** Badge centres by key. */
export function placeBadges(reqs: readonly BadgeRequest[], obstacles: readonly Box[], bondPx: number): Map<string, Vec> {
  const taken = [...obstacles];
  const out = new Map<string, Vec>();
  for (const r of reqs) {
    let best: { at: Vec; cost: number } | null = null;
    search: for (const ring of RINGS) {
      for (const t of TURNS) {
        const a = r.prefer + t;
        const at: Vec = [r.at[0] + Math.cos(a) * ring * bondPx, r.at[1] + Math.sin(a) * ring * bondPx];
        const box = boxAround(at, r.w / 2, r.h / 2);
        const cost = taken.reduce((s, o) => s + overlapArea(box, o), 0);
        if (!best || cost < best.cost) best = { at, cost };
        if (cost === 0) break search;
      }
    }
    out.set(r.key, best!.at);
    taken.push(boxAround(best!.at, r.w / 2 + 1, r.h / 2 + 1));
  }
  return out;
}
