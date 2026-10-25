// The molecule in the 3D view: the record's molblock as one 3Dmol model (record atom order, so
// a 3Dmol atom index is a record index), its representation and colours, and the overlays drawn
// on it: atom labels, Wiberg bond labels, orbital centroids, selection halos and measurements.
import type { GLModel, GLViewer } from "3dmol";
import { useEffect, useRef, useState } from "react";
import type { AtomPropsOut, BondPropsOut, MoleculeGeometryOut, OrbitalRowOut } from "../../../api/types.gen";
import { fmt } from "../../../components/format";
import type { Rep } from "../params";
import type { AtomPropDef } from "../atomProps";
import { colourAt } from "./colour";
import { measureText, type Measurement } from "./measure";
import { useOverlay, type Overlay } from "./useGLViewer";

export type LabelMode = "none" | "element" | "index" | "element-index" | "mulliken" | "property";

export const LABEL_MODES: { value: LabelMode; label: string }[] = [
  { value: "none", label: "no labels" },
  { value: "element", label: "element" },
  { value: "index", label: "record index" },
  { value: "element-index", label: "element + index" },
  { value: "mulliken", label: "Mulliken charge" },
  { value: "property", label: "colour property" },
];

export interface AtomColouring {
  def: AtomPropDef;
  range: [number, number];
  values: (number | null)[];
}

const xyz = (a: { x: number; y: number; z: number }) => ({ x: a.x, y: a.y, z: a.z });

type ColourFn = ((atom: { index?: number }) => string) | undefined;

function colourFn(c: AtomColouring | null): ColourFn {
  if (!c) return undefined;
  const scale = c.def.signed ? "rwb" : "seq";
  return (atom) => {
    const v = c.values[atom.index ?? -1];
    return v === null || v === undefined ? "#bbbbbb" : colourAt(scale, v, c.range[0], c.range[1]);
  };
}

/** The 3Dmol style of a representation (colorfunc: colour by property, else element colours). */
export function repStyle(rep: Rep, colorfunc: ColourFn): Record<string, unknown> {
  const c = colorfunc ? { colorfunc } : {};
  switch (rep) {
    case "sticks":
      return { stick: { radius: 0.17, ...c } };
    case "spacefill":
      return { sphere: { scale: 1.0, ...c } };
    case "wireframe":
      return { stick: { radius: 0.04, ...c } };
    case "lines":
      return { line: { linewidth: 2, ...c } };
    default:
      return { stick: { radius: 0.13, ...c }, sphere: { scale: 0.25, ...c } };
  }
}

/** The model of the geometry (replaced when the geometry changes; zoomed once). */
export function useModel(viewer: GLViewer | null, geometry: MoleculeGeometryOut | undefined): GLModel | null {
  const [model, setModel] = useState<GLModel | null>(null);
  useEffect(() => {
    if (!viewer || !geometry) return;
    let m: GLModel | null = null;
    try {
      m = viewer.addModel(geometry.molblock, "sdf", { keepH: true });
      viewer.zoomTo();
      viewer.render();
      setModel(m);
    } catch (err) {
      console.warn("workbench: cannot load the geometry into 3Dmol", err);
    }
    return () => {
      setModel(null);
      try {
        if (m) viewer.removeModel(m);
        viewer.render();
      } catch {
        // cleared viewer
      }
    };
  }, [viewer, geometry]);
  return model;
}

export function useStyle(viewer: GLViewer | null, model: GLModel | null, rep: Rep, showH: boolean, colouring: AtomColouring | null): void {
  useEffect(() => {
    if (!viewer || !model) return;
    model.setStyle({}, repStyle(rep, colourFn(colouring)));
    if (!showH) model.setStyle({ elem: "H" }, {});
    viewer.render();
  }, [viewer, model, rep, showH, colouring]);
}

/** Click handling on the model's atoms; `onAtom` is read from a ref so it may change freely. */
export function useAtomClicks(viewer: GLViewer | null, model: GLModel | null, onAtom: (index: number) => void): void {
  const ref = useRef(onAtom);
  ref.current = onAtom;
  useEffect(() => {
    if (!viewer || !model) return;
    model.setClickable({}, true, (atom: { index?: number }) => {
      if (typeof atom.index === "number") ref.current(atom.index);
    });
    viewer.render();
  }, [viewer, model]);
}

function labelText(mode: LabelMode, i: number, el: string, props: AtomPropsOut[] | undefined, colouring: AtomColouring | null): string {
  switch (mode) {
    case "element":
      return el;
    case "index":
      return String(i);
    case "element-index":
      return `${el}${i}`;
    case "mulliken":
      return fmt(props?.[i]?.mulliken_charge, 2);
    case "property": {
      const v = colouring?.values[i];
      return v === null || v === undefined ? "–" : fmt(v, colouring?.def.digits ?? 3);
    }
    default:
      return "";
  }
}

const LABEL_STYLE = { fontSize: 11, fontColor: "black", backgroundColor: "white", backgroundOpacity: 0.7, inFront: true, alignment: "center" } as const;

export function useAtomLabels(viewer: GLViewer | null, geometry: MoleculeGeometryOut | undefined, mode: LabelMode, showH: boolean, props: AtomPropsOut[] | undefined, colouring: AtomColouring | null): void {
  useOverlay(
    viewer,
    (v, o) => {
      if (!geometry || mode === "none") return;
      for (const a of geometry.atoms) {
        if (!showH && a.symbol === "H") continue;
        const text = labelText(mode, a.record_index, a.symbol, props, colouring);
        if (text) o.labels.push(v.addLabel(text, { ...LABEL_STYLE, position: xyz(a) }));
      }
    },
    [geometry, mode, showH, props, colouring],
  );
}

function bondOverlay(v: GLViewer, o: Overlay, geometry: MoleculeGeometryOut, b: BondPropsOut): void {
  const a = geometry.atoms[b.i];
  const c = geometry.atoms[b.j];
  if (!a || !c) return;
  const mid = { x: (a.x + c.x) / 2, y: (a.y + c.y) / 2, z: (a.z + c.z) / 2 };
  if (!b.bonded) {
    o.shapes.push(v.addCylinder({ start: xyz(a), end: xyz(c), radius: 0.03, dashed: true, dashLength: 0.1, gapLength: 0.1, color: "#7b3fb8", fromCap: "none", toCap: "none" }));
  }
  const style = { ...LABEL_STYLE, fontSize: 10, fontColor: b.bonded ? "#1c2330" : "#7b3fb8", position: mid };
  o.labels.push(v.addLabel(fmt(b.wiberg, 2), style));
}

/** Wiberg indices at the bond midpoints; non-bonded pairs above the threshold as dashed lines. */
export function useBondLabels(viewer: GLViewer | null, geometry: MoleculeGeometryOut | undefined, bonds: BondPropsOut[] | undefined, show: boolean): void {
  useOverlay(
    viewer,
    (v, o) => {
      if (!show || !geometry || !bonds) return;
      for (const b of bonds) bondOverlay(v, o, geometry, b);
    },
    [geometry, bonds, show],
  );
}

/** Orbital centroids as small spheres (the selected one larger, gold); click picks an orbital. */
export function useCentroids(viewer: GLViewer | null, rows: OrbitalRowOut[] | undefined, selected: number | null, show: boolean, onPick: (index: number) => void): void {
  const pick = useRef(onPick);
  pick.current = onPick;
  useOverlay(
    viewer,
    (v, o) => {
      if (!show || !rows) return;
      for (const r of rows) {
        if (!r.centroid) continue;
        const sel = r.index === selected;
        const color = sel ? "#f2c200" : r.kind === "occupied" ? "#2463c7" : "#1f8a4c";
        const center = { x: r.centroid[0], y: r.centroid[1], z: r.centroid[2] };
        o.shapes.push(v.addSphere({ center, radius: sel ? 0.2 : 0.1, color, opacity: sel ? 0.95 : 0.7, clickable: true, callback: () => pick.current(r.index) }));
      }
    },
    [rows, selected, show],
  );
}

/** Halos on the selected (gold), hovered (violet) and measure-pending (green) atoms. */
export function useHalos(viewer: GLViewer | null, geometry: MoleculeGeometryOut | undefined, selected: readonly number[], hover: number | null, picks: readonly number[]): void {
  useOverlay(
    viewer,
    (v, o) => {
      if (!geometry) return;
      const halo = (i: number, color: string, radius: number, opacity: number) => {
        const a = geometry.atoms[i];
        if (a) o.shapes.push(v.addSphere({ center: xyz(a), radius, color, opacity }));
      };
      for (const i of selected) halo(i, "#f2c200", 0.6, 0.45);
      if (hover !== null && !selected.includes(hover)) halo(hover, "#7b3fb8", 0.55, 0.3);
      for (const i of picks) halo(i, "#1f8a4c", 0.5, 0.45);
    },
    [geometry, selected, hover, picks],
  );
}

export function useMeasurements(viewer: GLViewer | null, geometry: MoleculeGeometryOut | undefined, list: readonly Measurement[]): void {
  useOverlay(
    viewer,
    (v, o) => {
      if (!geometry) return;
      for (const m of list) {
        const pts = m.atoms.map((i) => geometry.atoms[i]).filter(Boolean);
        for (let k = 0; k + 1 < pts.length; k++) {
          o.shapes.push(v.addCylinder({ start: xyz(pts[k]), end: xyz(pts[k + 1]), radius: 0.025, dashed: true, dashLength: 0.08, gapLength: 0.08, color: "#1f8a4c", fromCap: "none", toCap: "none" }));
        }
        const n = pts.length || 1;
        const c = pts.reduce((s, p) => ({ x: s.x + p.x / n, y: s.y + p.y / n, z: s.z + p.z / n }), { x: 0, y: 0, z: 0 });
        o.labels.push(v.addLabel(measureText(m), { ...LABEL_STYLE, fontColor: "#1f8a4c", position: c }));
      }
    },
    [geometry, list],
  );
}
