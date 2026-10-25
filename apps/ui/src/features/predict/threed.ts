// 3Dmol.js helpers for the contact panel (design §8.3, §11.7). 3Dmol is the window global set by
// the classic script /vendor/3Dmol-min.js; nothing from it is bundled (types only).
//
// The scene is in the donor fragment's record frame (Angstrom): the donor fragment, the placed
// acceptor cap (alternate colours, opacity 0.8), the faded acceptor context, the two lobe arrows
// (u at the donor site a, -u at the placed acceptor site b, 1.2 Angstrom long), the dashed a-b
// cylinder labelled with the contact distance, and each orbital cube as a +iso / -iso pair of
// isosurfaces (donor blue / red, acceptor green / orange).
import type { GLShape, GLViewer, VolumeData } from "3dmol";
import type { ContactOut, LobeOut } from "../../api/types.gen";

export type ThreeDState = "loading" | "ready" | "error" | "webgl-unavailable";

export type ThreeDmol = NonNullable<Window["$3Dmol"]>;

export const LOBE_LEN_ANG = 1.2;
export const ISO_MIN = 0.005;
export const ISO_MAX = 0.2;
export const ISO_STEP = 0.005;

export const SURFACE_COLORS = {
  donor: { pos: "#2463c7", neg: "#c0362c" },
  acceptor: { pos: "#1f8a4c", neg: "#e07b00" },
} as const;

type Vec = [number, number, number];
const xyz = (v: readonly number[]) => ({ x: v[0], y: v[1], z: v[2] });
const along = (o: readonly number[], d: readonly number[], len: number): Vec => [o[0] + len * d[0], o[1] + len * d[1], o[2] + len * d[2]];

let probed: { ok: boolean; reason: string | null } | null = null;

/** Can this browser create a WebGL context? Probed once per page (the probe context is
 *  released at once; browsers cap live contexts). Never throws. */
export function probeWebGL(): { ok: boolean; reason: string | null } {
  if (probed) return probed;
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
    probed = gl ? { ok: true, reason: null } : { ok: false, reason: "this browser cannot create a WebGL context" };
  } catch (err) {
    probed = { ok: false, reason: `WebGL probe failed: ${String(err)}` };
  }
  return probed;
}

export function threeDmol(): ThreeDmol | null {
  return window.$3Dmol ?? null;
}

function addLobe(viewer: GLViewer, lobe: LobeOut, color: string, radius: number): void {
  viewer.addArrow({ start: xyz(lobe.origin), end: xyz(along(lobe.origin, lobe.direction, LOBE_LEN_ANG)), radius, radiusRatio: 2.2, mid: 0.72, color });
}

/** Models, lobes and the a-b distance of one contact placement. */
export function drawContact(viewer: GLViewer, c: ContactOut): void {
  const donor = viewer.addModel(c.donor_fragment.molblock, "sdf");
  donor.setStyle({}, { stick: { radius: 0.14 }, sphere: { scale: 0.22 } });
  const cap = viewer.addModel(c.cap.molblock, "sdf");
  cap.setStyle({}, { stick: { radius: 0.14, colorscheme: "greenCarbon", opacity: 0.8 }, sphere: { scale: 0.22, colorscheme: "greenCarbon", opacity: 0.8 } });
  if (c.acceptor_context) {
    const ctx = viewer.addModel(c.acceptor_context.molblock, "sdf");
    ctx.setStyle({}, { stick: { radius: 0.08, colorscheme: "greenCarbon", opacity: 0.25 } });
  }
  addLobe(viewer, c.donor_lobe, SURFACE_COLORS.donor.pos, 0.07);
  for (const t of c.donor_lobe.tied) addLobe(viewer, { ...c.donor_lobe, direction: t }, "#8aa6d8", 0.035);
  addLobe(viewer, c.acceptor_lobe_placed, SURFACE_COLORS.acceptor.pos, 0.07);
  const a = c.donor_lobe.origin;
  const b = c.acceptor_lobe_placed.origin;
  viewer.addCylinder({ start: xyz(a), end: xyz(b), radius: 0.03, dashed: true, dashLength: 0.12, gapLength: 0.1, color: "#555555", fromCap: "none", toCap: "none" });
  const mid: Vec = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
  viewer.addLabel(`${c.distance_ang.toFixed(2)} Å`, { position: xyz(mid), fontSize: 12, fontColor: "black", backgroundColor: "white", backgroundOpacity: 0.75, inFront: true });
}

/** A Gaussian cube parsed once (3Dmol VolumeData), reused for every isovalue. */
export function parseCube(lib: ThreeDmol, text: string): VolumeData {
  return new lib.VolumeData(text, "cube");
}

/** The +iso and -iso isosurfaces of one cube; returns the shapes so they can be replaced. */
export function addSurfaces(viewer: GLViewer, vol: VolumeData, iso: number, colors: { pos: string; neg: string }): GLShape[] {
  return [
    viewer.addIsosurface(vol, { isoval: iso, color: colors.pos, opacity: 0.85 }),
    viewer.addIsosurface(vol, { isoval: -iso, color: colors.neg, opacity: 0.85 }),
  ];
}

export function removeShapes(viewer: GLViewer, shapes: readonly GLShape[]): void {
  for (const s of shapes) viewer.removeShape(s);
}
