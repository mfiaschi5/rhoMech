// ViewerPanel: the workbench's one 3Dmol view of the computed molecule (record geometry and
// atom order): representation, labels, colour by property with its colour bar, Wiberg labels,
// orbital centroids, selection halos, click-to-measure, exports and the surface layers.
//
// DOM: <section data-testid="wb-viewer-panel">, <div data-testid="wb-viewer"
// data-state="loading|ready|error|webgl-unavailable" data-rep data-selected-atoms="0,2"
// data-has-geometry>, measurement rows wb-measure, the toolbar and SurfaceControls.
import { useMemo, useState } from "react";
import type { CubeQuality } from "../../../api/client";
import type { MoleculeGeometryOut, MoleculePropertiesOut, OrbitalSetOut } from "../../../api/types.gen";
import { atomPropDefs, findProp, propRange } from "../atomProps";
import type { LayerSpec, Rep } from "../params";
import { setHover, toggleAtom, useSelection } from "../store";
import { ColourBar } from "./ColourBar";
import type { SurfaceLayer } from "./layers";
import { MEASURE_ATOMS, measure, measureText, type Measurement, type P3 } from "./measure";
import { useAtomClicks, useAtomLabels, useBondLabels, useCentroids, useHalos, useMeasurements, useModel, useStyle, type AtomColouring } from "./scene";
import { SurfaceControls } from "./SurfaceControls";
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
  layers: SurfaceLayer[];
  orbitalSpec: LayerSpec | null;
  onLayerChange(l: SurfaceLayer): void;
  onLayerRemove(id: string): void;
  onLayerPin(l: SurfaceLayer): void;
  onLayerAdd(s: LayerSpec): void;
  fileStem: string;
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
  const [view, setView] = useState<ViewOptions>(DEFAULT_VIEW);
  const sel = useSelection();
  const defs = useMemo(() => atomPropDefs(p.properties), [p.properties]);
  const colouring = useColouring(p.properties, view.colour);
  const tool = useMeasureTool(p.geometry, view.measure);
  const model = useModel(viewer, p.geometry);
  useStyle(viewer, model, p.rep, view.showH, colouring);
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
    <section className="panel wb-viewer-panel" data-testid="wb-viewer-panel">
      <ViewerToolbar rep={p.rep} onRep={p.onRep} quality={p.quality} onQuality={p.onQuality} view={view} onView={setView} props={defs} canExport={!!p.geometry} onExport={onExport} />
      <ViewerStatus handle={handle} />
      <div
        className="wb-viewer"
        data-testid="wb-viewer"
        data-state={handle.state}
        data-rep={p.rep}
        data-selected-atoms={sel.atoms.join(",")}
        data-has-geometry={p.geometry ? "true" : "false"}
        onPointerLeave={() => setHover(null)}
      >
        <div className="wb-canvas" ref={setEl} />
        {!p.geometry && handle.state !== "webgl-unavailable" && <div className="wb-viewer-empty muted">Draw a molecule and press Compute.</div>}
      </div>
      {colouring && (
        <ColourBar role="atoms" label={colouring.def.label} scale={colouring.def.signed ? "rwb" : "seq"} min={colouring.range[0]} max={colouring.range[1]} note={colouring.def.title} />
      )}
      <Measurements m={tool} onClear={tool.clear} />
      {p.mid && (
        <SurfaceControls
          viewer={viewer}
          mid={p.mid}
          quality={p.quality}
          layers={p.layers}
          orbital={p.orbitalSpec}
          onChange={p.onLayerChange}
          onRemove={p.onLayerRemove}
          onPin={p.onLayerPin}
          onAdd={p.onLayerAdd}
        />
      )}
    </section>
  );
}
