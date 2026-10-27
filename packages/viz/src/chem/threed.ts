// 3Dmol.js helpers shared by the pages (design §8.3, §11.7): the WebGL probe, cube parsing,
// ±iso orbital surfaces and a scalar mapped on an isosurface (the ESP on the density). 3Dmol
// is the window global set by the classic script /vendor/3Dmol-min.js; nothing from it is
// bundled (types only). Scene-specific drawing stays in its feature (predict/contactScene.ts).
import type { GLShape, GLViewer, VolumeData } from "3dmol";

export type ThreeDState = "loading" | "ready" | "error" | "webgl-unavailable";

export type ThreeDmol = NonNullable<Window["$3Dmol"]>;

export const ISO_MIN = 0.005;
export const ISO_MAX = 0.2;
export const ISO_STEP = 0.005;

export const SURFACE_COLORS = {
  donor: { pos: "#2463c7", neg: "#c0362c" },
  acceptor: { pos: "#1f8a4c", neg: "#e07b00" },
} as const;

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

/** A Gaussian cube parsed once (3Dmol VolumeData), reused for every isovalue. */
export function parseCube(lib: ThreeDmol, text: string): VolumeData {
  return new lib.VolumeData(text, "cube");
}

export interface SurfaceStyle {
  opacity?: number;
  /** Mesh instead of a solid surface. */
  wireframe?: boolean;
}

/** The +iso and -iso isosurfaces of one cube; returns the shapes so they can be replaced. */
export function addSurfaces(
  viewer: GLViewer,
  vol: VolumeData,
  iso: number,
  colors: { pos: string; neg: string },
  style: SurfaceStyle = {},
): GLShape[] {
  const common = { opacity: style.opacity ?? 0.85, wireframe: style.wireframe ?? false };
  return [
    viewer.addIsosurface(vol, { isoval: iso, color: colors.pos, ...common }),
    viewer.addIsosurface(vol, { isoval: -iso, color: colors.neg, ...common }),
  ];
}

/** One isosurface of a positive scalar (density, |ψ|², f±) at +iso. */
export function addSurface(viewer: GLViewer, vol: VolumeData, iso: number, color: string, style: SurfaceStyle = {}): GLShape {
  return viewer.addIsosurface(vol, { isoval: iso, color, opacity: style.opacity ?? 0.85, wireframe: style.wireframe ?? false });
}

/** A handle for whatever addMappedSurface drew: an isosurface shape or a 3Dmol surface id. */
export type MappedSurface = { kind: "shape"; shape: GLShape } | { kind: "surface"; id: number };

/** The density isosurface at `iso`, coloured by `espVol` with red-white-blue over
 *  [-range, +range] (red negative). Falls back to a van der Waals surface coloured by the same
 *  volume when the isosurface colouring throws. */
export function addMappedSurface(
  viewer: GLViewer,
  densityVol: VolumeData,
  espVol: VolumeData,
  iso: number,
  range: number,
  style: SurfaceStyle = {},
): MappedSurface {
  const lib = threeDmol();
  if (!lib) throw new Error("3Dmol.js is not loaded");
  const scheme = new lib.Gradient.RWB(-range, range);
  const opacity = style.opacity ?? 0.85;
  try {
    const shape = viewer.addIsosurface(densityVol, { isoval: iso, voldata: espVol, volscheme: scheme, opacity, wireframe: style.wireframe ?? false });
    return { kind: "shape", shape };
  } catch {
    const surf = viewer.addSurface(lib.SurfaceType.VDW, { opacity, voldata: espVol, volscheme: scheme }, {});
    const id = typeof surf === "number" ? surf : Number((surf as { surfid?: number })?.surfid ?? -1);
    return { kind: "surface", id };
  }
}

export function removeMapped(viewer: GLViewer, m: MappedSurface): void {
  if (m.kind === "shape") viewer.removeShape(m.shape);
  else if (m.id >= 0) viewer.removeSurface(m.id);
}

export function removeShapes(viewer: GLViewer, shapes: readonly GLShape[]): void {
  for (const s of shapes) viewer.removeShape(s);
}
