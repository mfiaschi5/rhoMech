// ViewerPanel: the workbench's one 3Dmol view of the computed molecule (record geometry and
// atom order): representation, labels, colour by property with its colour bar, Wiberg labels,
// orbital centroids, selection halos, click-to-measure and exports. The surface layers are their
// own panel (SurfaceControls.tsx, placed by the page's layout) drawing on the same viewer.
//
// onViewer(v) hands the 3Dmol viewer (null when gone) to the page, which reads its orientation
// for the sketcher's 3D -> 2D projection (viewProjection.ts) and gives it to the surfaces.
// `height` is the view's height in px (the layout's row splitter); the viewer is resized
// whenever its box changes (splitters, a panel shown again).
//
// The point group's elements (axes, mirror discs, inversion centre) and its equivalent-atom
// class colours are drawn from viewer/symmetry.ts's store (toolbar: symmetry-toggle,
// symmetry-classes-toggle; the Global tab's element list highlights one).
//
// DOM: <div data-testid="wb-viewer-panel">, <div data-testid="wb-viewer"
// data-state="loading|ready|error|webgl-unavailable" data-rep data-selected-atoms="0,2"
// data-has-geometry data-symmetry-drawn=<elements drawn> data-symmetry-classes>, measurement
// rows wb-measure and the toolbar.
import { useEffect, useMemo, useState } from "react";
import type { GLViewer } from "3dmol";
import type { CubeQuality } from "../../../api/workbench";
import type { MoleculeGeometryOut, MoleculePropertiesOut, OrbitalSetOut, PointGroupOut } from "../../../api/types.gen";
import { atomPropDefs, findProp, propRange } from "../atomProps";
import type { Rep } from "../params";
import { setHover, toggleAtom, useSelection } from "../store";
import { ColourBar } from "./ColourBar";
import { MEASURE_ATOMS, measure, measureText, type Measurement, type P3 } from "./measure";
import { useAtomClicks, useAtomLabels, useBondLabels, useCentroids, useHalos, useMeasurements, useModel, useStyle, type AtomColouring } from "./scene";
import { classColourFn, drawnElements, useSymmetryElements, useSymmetryView } from "./symmetry";
import { useGLViewer, type ViewerHandle } from "./useGLViewer";
import { DEFAULT_VIEW, ViewerToolbar, type ViewOptions } from "./ViewerToolbar";

export interface ViewerPanelProps {
  mid: string | null;
  geometry: MoleculeGeometryOut | undefined;
  properties: MoleculePropertiesOut | undefined;
  orbitals: OrbitalSetOut | undefined;
  selectedOrbital: number | null;
  onPickOrbital(index: number): void;
  rep: Rep;
  onRep(r: Rep): void;
  quality: CubeQuality;
  onQuality(q: CubeQuality): void;
  fileStem: string;
  onViewer?(v: GLViewer | null): void;
  /** Height of the 3D view (px). */
  height?: number;
  /** The point group (drawn elements, class colours; viewer/symmetry.ts). */
  symmetry?: PointGroupOut;
}

/** Resize the viewer to its element whenever the element's box changes. */
function useViewerResize(viewer: GLViewer | null, el: HTMLDivElement | null): void {
  useEffect(() => {
    if (!viewer || !el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      try {
        if (el.clientWidth > 0 && el.clientHeight > 0) {
          viewer.resize();
          viewer.render();
        }
      } catch {
        // the viewer is going away
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [viewer, el]);
}

function download(name: string, href: string): void {
  const a = document.createElement("a");
  a.href = href;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function downloadText(name: string, text: string, type: string): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  download(name, url);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Clicks in measure mode collect atoms; a full set becomes a measurement. */
function useMeasureTool(geometry: MoleculeGeometryOut | undefined, mode: ViewOptions["measure"]) {
  const [picks, setPicks] = useState<number[]>([]);
  const [list, setList] = useState<Measurement[]>([]);
  const pick = (i: number) => {
    if (mode === "select" || !geometry) return;
    const next = [...picks, i];
    if (next.length < MEASURE_ATOMS[mode]) return setPicks(next);
    const pts = next.map((k): P3 => [geometry.atoms[k].x, geometry.atoms[k].y, geometry.atoms[k].z]);
    setList((l) => [...l, { mode, atoms: next, value: measure(mode, pts) }]);
    setPicks([]);
  };
  const clear = () => {
    setPicks([]);
    setList([]);
  };
  return { picks, list, pick, clear };
}

function useColouring(properties: MoleculePropertiesOut | undefined, key: string | null): AtomColouring | null {
  return useMemo(() => {
    const def = findProp(atomPropDefs(properties), key);
    if (!def || !properties) return null;
    const range = propRange(def, properties.atoms);
    return range ? { def, range, values: properties.atoms.map(def.get) } : null;
  }, [properties, key]);
}

function ViewerStatus({ handle }: { handle: ViewerHandle }) {
  if (handle.state === "webgl-unavailable") return <div className="banner banner-warn">3D view unavailable: {handle.message}. Tables, orbitals and cube downloads still work.</div>;
  if (handle.state === "error") return <div className="banner banner-error">{handle.message}</div>;
  return null;
}

function Measurements({ m, onClear }: { m: ReturnType<typeof useMeasureTool>; onClear(): void }) {
  if (!m.list.length && !m.picks.length) return null;
  return (
    <div className="wb-measures row">
      {m.list.map((x, i) => (
        <span key={i} className="badge" data-testid="wb-measure" data-mode={x.mode} data-value={x.value}>
          {x.mode} {x.atoms.join("-")}: {measureText(x)}
        </span>
      ))}
      {m.picks.length > 0 && <span className="muted wb-small">picked {m.picks.join(", ")}…</span>}
      <button type="button" className="btn btn-small" onClick={onClear}>
        clear
      </button>
    </div>
  );
}

export function ViewerPanel(p: ViewerPanelProps) {
  const [el, setEl] = useState<HTMLDivElement | null>(null);
  const handle = useGLViewer(el);
  const viewer = handle.viewer;
  useViewerResize(viewer, el);
  const { onViewer } = p;
  useEffect(() => {
    onViewer?.(viewer);
    return () => onViewer?.(null);
  }, [viewer, onViewer]);
  const [view, setView] = useState<ViewOptions>(DEFAULT_VIEW);
  const sel = useSelection();
  const defs = useMemo(() => atomPropDefs(p.properties), [p.properties]);
  const colouring = useColouring(p.properties, view.colour);
  const tool = useMeasureTool(p.geometry, view.measure);
  const model = useModel(viewer, p.geometry);
  const symView = useSymmetryView();
  const symmetry = p.symmetry && p.geometry && p.symmetry.molecule_id === p.mid ? p.symmetry : undefined;
  const classColours = useMemo(() => classColourFn(symmetry, symView.classes), [symmetry, symView.classes]);
  useStyle(viewer, model, p.rep, view.showH, colouring, classColours);
  useSymmetryElements(viewer, symmetry, symView);
  useAtomClicks(viewer, model, (i) => (view.measure === "select" ? toggleAtom(i) : tool.pick(i)));
  useAtomLabels(viewer, p.geometry, view.labels, view.showH, p.properties?.atoms, colouring);
  useBondLabels(viewer, p.geometry, p.properties?.bonds, view.wiberg);
  useCentroids(viewer, p.orbitals?.orbitals, p.selectedOrbital, view.centroids, p.onPickOrbital);
  useHalos(viewer, p.geometry, sel.atoms, sel.hover, tool.picks);
  useMeasurements(viewer, p.geometry, tool.list);
  const onExport = (what: "png" | "xyz" | "mol") => {
    if (!p.geometry) return;
    if (what === "png" && viewer) download(`${p.fileStem}.png`, viewer.pngURI());
    if (what === "xyz") downloadText(`${p.fileStem}.xyz`, p.geometry.xyz, "chemical/x-xyz");
    if (what === "mol") downloadText(`${p.fileStem}.mol`, p.geometry.molblock, "chemical/x-mdl-molfile");
  };
  return (
    <div className="wb-viewer-panel" data-testid="wb-viewer-panel">
      <ViewerToolbar rep={p.rep} onRep={p.onRep} quality={p.quality} onQuality={p.onQuality} view={view} onView={setView} props={defs} canExport={!!p.geometry} onExport={onExport} />
      <ViewerStatus handle={handle} />
      <div
        className="wb-viewer"
        data-testid="wb-viewer"
        data-state={handle.state}
        data-rep={p.rep}
        data-selected-atoms={sel.atoms.join(",")}
        data-has-geometry={p.geometry ? "true" : "false"}
        data-symmetry-drawn={drawnElements(symmetry, symView).length}
        data-symmetry-classes={classColours ? "true" : "false"}
        style={p.height ? { height: p.height } : undefined}
        onPointerLeave={() => setHover(null)}
      >
        <div className="wb-canvas" ref={setEl} />
        {!p.geometry && handle.state !== "webgl-unavailable" && <div className="wb-viewer-empty muted">Draw a molecule and press Compute.</div>}
      </div>
      {colouring && !classColours && (
        <ColourBar role="atoms" label={colouring.def.label} scale={colouring.def.signed ? "rwb" : "seq"} min={colouring.range[0]} max={colouring.range[1]} note={colouring.def.title} />
      )}
      <Measurements m={tool} onClear={tool.clear} />
    </div>
  );
}
