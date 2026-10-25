// The point group of the computed molecule (GET /molecule/{id}/symmetry) shared by the
// Properties Global tab (tables/PointGroupSection.tsx: the group, the tolerance, the element
// and class lists) and the 3D view (the elements drawn on the record geometry, the
// equivalent-atom classes as atom colours). The view state lives in a small store: the
// tolerance (Å), whether every element is drawn, the highlighted element (drawn even when the
// others are not) and the class colouring.
//
// Drawing: a rotation axis is a rod through the centre of mass labelled C_n (gold when
// highlighted), an improper axis a dashed rod labelled S_n, a mirror plane a translucent disc
// labelled σ, the inversion centre a small sphere labelled i.
import type { GLViewer } from "3dmol";
import { getMoleculeSymmetry } from "../../../api/client";
import { useDebounced, usePendingCached, type PendingCached } from "../../../api/hooks";
import type { PointGroupOut, SymmetryElementOut } from "../../../api/types.gen";
import { createStore, useStore } from "../../../state/store";
import { useOverlay, type Overlay } from "./useGLViewer";

export const TOLERANCE_DEFAULT = 0.1;
export const TOLERANCE_MIN = 0.01;
export const TOLERANCE_MAX = 0.5;

export interface SymmetryView {
  tolerance: number;
  /** Draw every element in the 3D view. */
  show: boolean;
  /** The highlighted element id (drawn even when `show` is off). */
  highlight: string | null;
  /** Colour the atoms by equivalence class. */
  classes: boolean;
}

export const symmetryStore = createStore<SymmetryView>({ tolerance: TOLERANCE_DEFAULT, show: false, highlight: null, classes: false });

export function setSymmetryView(patch: Partial<SymmetryView>): void {
  symmetryStore.update((s) => ({ ...s, ...patch }));
}

export function useSymmetryView(): SymmetryView {
  return useStore(symmetryStore);
}

/** A tolerance within the server's range, rounded to 0.01 Å (null: not a number). */
export function clampTolerance(t: number): number | null {
  if (!Number.isFinite(t)) return null;
  return Math.round(Math.min(TOLERANCE_MAX, Math.max(TOLERANCE_MIN, t)) * 100) / 100;
}

/** The point group of molecule `mid` at the store's tolerance (debounced while typing). */
export function useSymmetry(mid: string | null): PendingCached<PointGroupOut> {
  const { tolerance } = useSymmetryView();
  const tol = useDebounced(clampTolerance(tolerance) ?? TOLERANCE_DEFAULT, 250);
  const key = mid ? `/molecule/${mid}/symmetry?tolerance=${tol}` : null;
  return usePendingCached(key, (o) => getMoleculeSymmetry(mid ?? "", tol, o));
}

const CLASS_COLOURS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#9a6324", "#469990", "#808000", "#000075", "#bfef45"];

export const classColour = (k: number): string => CLASS_COLOURS[k % CLASS_COLOURS.length];

/** 3Dmol colorfunc colouring each atom by its equivalence class (null: classes off). */
export function classColourFn(data: PointGroupOut | undefined, on: boolean): ((atom: { index?: number }) => string) | null {
  if (!on || !data) return null;
  const byAtom = new Map<number, string>();
  data.classes.forEach((c, k) => c.atoms.forEach((a) => byAtom.set(a, classColour(k))));
  return (atom) => byAtom.get(atom.index ?? -1) ?? "#bbbbbb";
}

/** The elements to draw: all with `show`, else only the highlighted one. */
export function drawnElements(data: PointGroupOut | undefined, view: SymmetryView): SymmetryElementOut[] {
  if (!data) return [];
  return data.elements.filter((e) => view.show || e.id === view.highlight);
}

type V3 = [number, number, number];
const xyz = (p: readonly number[]) => ({ x: p[0], y: p[1], z: p[2] });
const add = (a: readonly number[], b: readonly number[], s: number): V3 => [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]];
const cross = (a: readonly number[], b: readonly number[]): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: readonly number[]) => Math.hypot(a[0], a[1], a[2]);
const unit = (a: readonly number[]): V3 => {
  const n = norm(a) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
};

/** The structure's radius about the centre (Å), from the symmetrized coordinates. */
function radius(data: PointGroupOut): number {
  let r = 0;
  for (const p of data.symmetrized_xyz) r = Math.max(r, norm(add(p, data.centre, -1)));
  return r;
}

const AXIS_COLOURS: Record<number, string> = { 2: "#2463c7", 3: "#1f8a4c", 4: "#d9730d", 5: "#7b3fb8", 6: "#c62828" };
const PLANE_COLOURS: Record<string, string> = { "σh": "#2463c7", "σv": "#1f8a4c", "σv′": "#1f8a4c", "σd": "#d9730d" };
const GOLD = "#f2c200";
const LABEL = { fontSize: 12, fontColor: "#1c2330", backgroundColor: "white", backgroundOpacity: 0.75, inFront: true, alignment: "center" } as const;

/** A two-sided translucent disc of radius r normal to n at c. */
function disc(v: GLViewer, o: Overlay, c: readonly number[], n: readonly number[], r: number, color: string, opacity: number): void {
  const nu = unit(n);
  const a = unit(cross(nu, Math.abs(nu[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
  const b = cross(nu, a);
  const steps = 48;
  const ring = Array.from({ length: steps }, (_, k) => {
    const t = (2 * Math.PI * k) / steps;
    return add(add(c, a, r * Math.cos(t)), b, r * Math.sin(t));
  });
  const vertexArr = [c, ...ring, c, ...ring].map(xyz);
  const normalArr = [...Array(steps + 1).fill(nu), ...Array(steps + 1).fill(add([0, 0, 0], nu, -1))].map(xyz);
  const faceArr: number[] = [];
  for (let k = 0; k < steps; k++) {
    const k1 = (k + 1) % steps;
    faceArr.push(0, 1 + k, 1 + k1);
    faceArr.push(steps + 1, steps + 2 + k1, steps + 2 + k);
  }
  o.shapes.push(v.addCustom({ vertexArr, normalArr, faceArr, color, opacity }));
}

function drawElement(v: GLViewer, o: Overlay, e: SymmetryElementOut, c: readonly number[], r: number, lit: boolean): void {
  if (e.kind === "i" || !e.vector) {
    o.shapes.push(v.addSphere({ center: xyz(c), radius: lit ? 0.16 : 0.1, color: lit ? GOLD : "#c2185b", opacity: 0.95 }));
    o.labels.push(v.addLabel("i", { ...LABEL, position: xyz(c) }));
    return;
  }
  if (e.kind === "sigma") {
    disc(v, o, c, e.vector, r + 0.6, lit ? GOLD : (PLANE_COLOURS[e.label] ?? "#7b3fb8"), lit ? 0.45 : 0.2);
    const edge = unit(cross(e.vector, Math.abs(e.vector[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
    if (lit) o.labels.push(v.addLabel(e.label, { ...LABEL, position: xyz(add(c, edge, r + 0.6)) }));
    return;
  }
  const len = r + (e.kind === "S" ? 1.0 : 1.4);
  const color = lit ? GOLD : e.kind === "S" ? "#666666" : (AXIS_COLOURS[e.order ?? 0] ?? "#00838f");
  const start = xyz(add(c, e.vector, -len));
  const end = xyz(add(c, e.vector, len));
  const radiusRod = lit ? 0.07 : e.principal ? 0.05 : 0.035;
  const dashes = e.kind === "S" ? { dashed: true, dashLength: 0.15, gapLength: 0.12 } : {};
  o.shapes.push(v.addCylinder({ start, end, radius: radiusRod, color, fromCap: 1, toCap: 1, ...dashes }));
  if (e.kind === "C" || lit) o.labels.push(v.addLabel(e.label, { ...LABEL, fontColor: lit ? "#7a5c00" : color, position: xyz(add(c, e.vector, len + 0.25)) }));
}

/** Draw the shown symmetry elements on the viewer (replaced whenever the inputs change). */
export function useSymmetryElements(viewer: GLViewer | null, data: PointGroupOut | undefined, view: SymmetryView): void {
  const { show, highlight } = view;
  useOverlay(
    viewer,
    (v, o) => {
      if (!data) return;
      const r = radius(data);
      for (const e of drawnElements(data, view)) drawElement(v, o, e, data.centre, r, e.id === highlight);
    },
    [data, show, highlight],
  );
}
