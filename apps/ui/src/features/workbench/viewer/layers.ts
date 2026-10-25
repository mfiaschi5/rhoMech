// Surface layers of the 3D view: what to fetch (LayerSpec, in the hash) plus the client-side
// look (isovalue, opacity, solid or mesh, ± colours, visibility), which never refetches.
import type { CubeKind } from "../../../api/client";
import { normalizeSpec, type LayerSpec } from "../params";

export interface SurfaceLayer {
  id: string;
  spec: LayerSpec;
  iso: number;
  opacity: number;
  mesh: boolean;
  pos: string;
  neg: string;
  visible: boolean;
  /** The selected-orbital layer: follows the orbital picked in the list or the diagram. */
  follow: boolean;
  /** ESP only: the ± colour range in a.u. (null: automatic from the cube). */
  range: number | null;
}

export interface KindLook {
  iso: number;
  isoMin: number;
  isoMax: number;
  isoStep: number;
  opacity: number;
  pos: string;
  neg: string;
  /** Two surfaces (+iso and -iso) for a signed field. */
  signed: boolean;
}

const SMALL = { isoMin: 0.0002, isoMax: 0.05, isoStep: 0.0002 };

export const KIND_LOOK: Record<CubeKind, KindLook> = {
  orbital: { iso: 0.05, isoMin: 0.005, isoMax: 0.2, isoStep: 0.005, opacity: 0.85, pos: "#2463c7", neg: "#c0362c", signed: true },
  orbital_density: { iso: 0.0025, ...SMALL, opacity: 0.8, pos: "#7b3fb8", neg: "#7b3fb8", signed: false },
  density: { iso: 0.002, ...SMALL, opacity: 0.45, pos: "#8a94a3", neg: "#8a94a3", signed: false },
  fukui_minus: { iso: 0.003, ...SMALL, opacity: 0.8, pos: "#e07b00", neg: "#e07b00", signed: false },
  fukui_plus: { iso: 0.003, ...SMALL, opacity: 0.8, pos: "#1f8a4c", neg: "#1f8a4c", signed: false },
  dual: { iso: 0.002, ...SMALL, opacity: 0.8, pos: "#2463c7", neg: "#c0362c", signed: true },
  esp: { iso: 0.002, ...SMALL, opacity: 0.9, pos: "#ffffff", neg: "#ffffff", signed: false },
};

let counter = 0;

/** A new layer with its kind's look; `iso` overrides the kind's default isovalue. */
export function newLayer(spec: LayerSpec, follow = false, iso?: number): SurfaceLayer {
  const s = normalizeSpec(spec);
  const look = KIND_LOOK[s.kind];
  counter += 1;
  return {
    id: `${follow ? "follow" : "pin"}-${counter}`,
    spec: s,
    iso: iso ?? look.iso,
    opacity: look.opacity,
    mesh: false,
    pos: look.pos,
    neg: look.neg,
    visible: true,
    follow,
    range: null,
  };
}
