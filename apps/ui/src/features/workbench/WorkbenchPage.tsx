// WorkbenchPage (#/workbench): draw a molecule, compute it (POST /molecule: the QM record and
// tokens, display only), then look at it: the 3D view with surfaces, the orbital sets and their
// energy levels, and the property tables with the point group (viewer/symmetry.ts, drawn in 3D
// too). The panels (sketch, 3D view, orbitals, surfaces, properties) are placed by the
// customizable layout (layout/: columns with splitters, a wide row, presets, show / hide /
// collapse / move, kept in localStorage with the preferences: sketch font and bond width, atom
// colours, label style, default representation and orbital isovalue, theme). Radicals can be drawn but not computed: Compute is off with the reason
// while the drawing has unpaired electrons (the server answers RADICAL_NOT_SUPPORTED).
//
// Hash params (params.ts): smiles, m, set, start, orb, rep, layers, q. The drawing persists in
// sessionStorage (session.ts). Selections link the tables, the 3D view and the sketch through
// the POST's sketch mapping, only while the sketch is unchanged since the compute (topology, and
// wedges unless they draw the record's configuration: session.sketchChanged; coordinates never
// count). The sketcher's Clean and 3D -> 2D are server layouts (POST /api/depict2d, the page's
// Depictor); 3D -> 2D reads the viewer's quaternion through a ProjectionSource
// (viewer/viewProjection.ts), so the sketch module never sees 3Dmol. On each 3D -> 2D the page
// also checks the quaternion convention: the plain projection of the record by that quaternion
// against 3Dmol's own modelToScreen.
//
// DOM: <div data-testid="workbench" data-molecule-id data-linked data-projection-residual
// (screen-fit residual of that check, viewProjection.screenResidual) data-preset data-theme>,
// the layout bar (layout/LayoutMenu.tsx), the grid (layout/LayoutGrid.tsx), wb-compute (inside
// wb-compute-wrap, whose title says why it is off: data-reason), wb-status
// [data-status=idle|posting|running|ready|qm_only|failed|missing|error], wb-stale-note,
// wb-radical-note, wb-chemistry-banner[data-chemistry], wb-mapping-note, wb-embedding-open (the
// computed molecule traced through a trained checkpoint: features/embedding/EmbeddingDialog;
// greyed with the reason while the embedding capability is off).
import { useEffect, useMemo, useRef, useState, type Dispatch } from "react";
import type { GLViewer } from "3dmol";
import type { PageProps } from "../../registry";
import { navigate, parseHash, withParams, type Route } from "../../router";
import { useRDKit } from "../../chem/rdkit";
import { CapabilityNotice } from "../../components/CapabilityNotice";
import { ErrorBanner } from "../../components/ErrorBanner";
import { PendingNote } from "../../components/PendingNote";
import { useCapability } from "../../state/store";
import { EmbeddingDialog } from "../embedding/EmbeddingDialog";
import type { MoleculeGeometryOut, MoleculeOut } from "../../api/types.gen";
import { depict2d } from "../../api/workbench";
import { useGeometry, useOrbitalSet, useProperties } from "./data";
import { OrbitalPanel } from "./orbitals/OrbitalPanel";
import type { Window } from "./orbitals/levels";
import { CANONICAL, DEFAULT_SET, DEFAULT_START, isMoleculeId, isOrbitalKind, parseIndex, parseLayers, parseQuality, parseRep, formatLayers, type LayerSpec } from "./params";
import { restoreComputed, restoreSketch, saveComputed, saveSketch, sketchChanged, sketchTopology } from "./session";
import { emptySketch, exportSketch, fromMolblock, importSketch, molblockSmiles, radicalElectrons, Sketcher, toMolblock, useSketchInfo, type Depictor, type ProjectionSource, type SketchModel, type DrawingStyle, type StyleId } from "./sketch";
import { clearSelection, useSelection } from "./store";
import { PropertyTabs } from "./tables/PropertyTabs";
import { useMolecule, type ComputeState } from "./useMolecule";
import { useWorkbenchLayers } from "./useWorkbenchLayers";
import { ViewerPanel } from "./viewer/ViewerPanel";
import { setSymmetryView, useSymmetry } from "./viewer/symmetry";
import { SurfaceControls } from "./viewer/SurfaceControls";
import { LayoutGrid, type PanelContent } from "./layout/LayoutGrid";
import { LayoutBar } from "./layout/LayoutMenu";
import { useLayoutState, usePrefs, type PrefsState } from "./layout/useWorkbenchLayout";
import type { LayoutAction, LayoutState, PanelId } from "./layout/layoutState";
import { screenResidual, viewProjection, viewQuaternion } from "./viewer/viewProjection";
import "./workbench.css";
import "./layout/layout.css";

type Update = Record<string, string | number | null>;

/** Set some hash params. Reads the live hash, so two updates in one tick both apply. */
function update(u: Update): void {
  const live = parseHash();
  if (live.page !== "workbench") return;
  navigate(withParams(live, u), { replace: true });
}

/** The drawing: restored from the session, a `smiles` param imported once RDKit is ready. */
function useSketchState(smiles: string | null, update: (u: Update) => void) {
  const [model, setModel] = useState<SketchModel>(() => restoreSketch() ?? emptySketch());
  const { rdkit } = useRDKit();
  useEffect(() => {
    if (!rdkit || !smiles) return;
    try {
      setModel(importSketch(rdkit, smiles));
    } catch (err) {
      console.warn("workbench: cannot import the smiles param", err);
    }
    update({ smiles: null });
  }, [rdkit, smiles]);
  useEffect(() => saveSketch(model), [model]);
  return { model, setModel };
}

/** The compute flow plus the hash `m`: restore on load, write the id once known. */
function useComputeSync(m: string | null, update: (u: Update) => void) {
  const mol = useMolecule();
  const { state } = mol;
  useEffect(() => {
    if (!isMoleculeId(m)) return;
    const saved = restoreComputed();
    if (saved && saved.m === m) {
      try {
        return void mol.compute(fromMolblock(saved.molblock), saved.molblock);
      } catch {
        // an unreadable saved drawing: load by id
      }
    }
    mol.load(m);
    // on mount only: later `m` changes come from this page
  }, []);
  const id = state.molecule?.molecule_id ?? null;
  useEffect(() => {
    if (!id) return;
    if (state.molblock) saveComputed({ m: id, molblock: state.molblock });
    if (id !== m) update({ m: id, ...(isMoleculeId(m) ? { orb: null } : {}) });
  }, [id]);
  return mol;
}

function statusText(s: ComputeState): string {
  switch (s.status) {
    case "idle":
      return "not computed";
    case "posting":
      return "submitting";
    case "running":
      return "featurizing (QM record and tokens)";
    case "ready":
      return "ready";
    case "qm_only":
      return "QM record only (tokens failed): geometry, orbitals and cubes still work";
    case "failed":
      return "featurize failed";
    case "missing":
      return "not featurized";
    default:
      return "error";
  }
}

function ChemistryBanner({ molecule }: { molecule: MoleculeOut | null }) {
  if (!molecule || molecule.chemistry === "real") return null;
  if (molecule.chemistry === "cached") {
    return (
      <div className="banner banner-warn" data-testid="wb-chemistry-banner" data-chemistry="cached" role="status">
        Cached chemistry ({molecule.method}): only molecules with a GFN2-xTB record in the shared cache compute; a new molecule fails with its reason code.
      </div>
    );
  }
  // synthetic / stub: the shell's SyntheticBanner already explains it; one line here
  return (
    <div className="muted wb-small" data-testid="wb-chemistry-banner" data-chemistry={molecule.chemistry} role="status">
      <span className="badge badge-warn">{molecule.chemistry}</span> {molecule.method}: orbitals, densities and properties of this molecule are not GFN2-xTB.
    </div>
  );
}

const RADICAL_REASON = "Radicals can be drawn but not computed yet: the QM featurization is closed-shell and the server refuses unpaired electrons (RADICAL_NOT_SUPPORTED). Clear the dots (radical tools or the \".\" key) to compute.";

/** Why Compute is off (null: it can run). */
function computeBlocker(valid: boolean, stale: boolean, radicals: number): string | null {
  if (radicals > 0) return RADICAL_REASON;
  if (!valid) return "Draw a valid molecule first";
  return stale ? "Wait for RDKit to check the drawing" : null;
}

function FailureBanner({ failure }: { failure: NonNullable<MoleculeOut["error"]> }) {
  const radical = failure.code === "RADICAL_NOT_SUPPORTED";
  const text = radical ? `${failure.message}. ${RADICAL_REASON}` : failure.message;
  return <ErrorBanner error={text} title={`Featurize failed: ${failure.code}`} testId="wb-failure" />;
}

/** "Model trace": the computed molecule through a trained checkpoint, in a dialog. */
function TraceButton({ state }: { state: ComputeState }) {
  const cap = useCapability("embedding");
  const [open, setOpen] = useState(false);
  const key = state.status === "ready" ? (state.molecule?.key ?? null) : null;
  const why = cap.known && !cap.enabled ? `embedding trace unavailable: ${cap.reason ?? ""}` : key ? "trace the molecule through a trained spec-v9 checkpoint" : "compute the molecule first (QM record and tokens)";
  return (
    <>
      <span title={why} data-testid="wb-embedding-wrap" data-reason={!key || !cap.enabled ? why : ""}>
        <button type="button" className="btn btn-small" data-testid="wb-embedding-open" disabled={!key || !cap.enabled} onClick={() => setOpen(true)}>
          Model trace
        </button>
      </span>
      {open && key && <EmbeddingDialog smiles={key} onClose={() => setOpen(false)} />}
    </>
  );
}

function ComputeBar(p: { state: ComputeState; blocker: string | null; radicals: number; stale: boolean; onCompute(retry?: boolean): void }) {
  const { state } = p;
  const busy = state.status === "posting" || state.status === "running";
  const failure = state.molecule?.error ?? null;
  return (
    <div className="wb-compute-bar">
      <div className="row">
        <span className="wb-compute-wrap" data-testid="wb-compute-wrap" data-reason={p.blocker ?? ""} title={p.blocker ?? "Compute the QM record and tokens of the drawing"}>
          <button type="button" className="btn btn-primary" data-testid="wb-compute" disabled={p.blocker !== null || busy} onClick={() => p.onCompute()}>
            {state.status === "idle" ? "Compute" : "Recompute"}
          </button>
        </span>
        <span className="wb-status" data-testid="wb-status" data-status={state.status} data-molecule-id={state.molecule?.molecule_id ?? ""}>
          {statusText(state)}
          {state.molecule && <code className="muted"> {state.molecule.key}</code>}
        </span>
        {busy && <PendingNote label={state.status === "posting" ? "registering the molecule" : "featurize: QM record and tokens"} />}
        {state.status === "failed" && (
          <button type="button" className="btn btn-small" data-testid="wb-retry" onClick={() => p.onCompute(true)}>
            retry
          </button>
        )}
        <TraceButton state={state} />
      </div>
      {p.radicals > 0 && (
        <div className="banner banner-warn wb-small" data-testid="wb-radical-note" data-electrons={p.radicals} role="status">
          {p.radicals} unpaired electron{p.radicals === 1 ? "" : "s"} drawn. {RADICAL_REASON}
        </div>
      )}
      {p.stale && (
        <div className="muted wb-small" data-testid="wb-stale-note">
          The sketch changed since the compute: selections are no longer linked to it. Recompute to link them again.
        </div>
      )}
      <ErrorBanner error={state.error} title="Cannot compute the molecule" />
      {failure && <FailureBanner failure={failure} />}
      {state.molecule?.warnings.map((w) => (
        <div key={w} className="banner banner-warn">
          {w}
        </div>
      ))}
      {state.molecule && !state.mapping && (
        <div className="muted wb-small" data-testid="wb-mapping-note">
          {state.molecule.mapping_error
            ? `The drawing does not map onto the record (${state.molecule.mapping_error}); the sketch is not linked.`
            : "No sketch mapping for this molecule in this tab (opened by id): recompute the drawing to link it."}
        </div>
      )}
    </div>
  );
}

/** Sketch atom ids to highlight for the selected record atoms (through record_parent_sketch). */
function sketchHighlight(state: ComputeState, model: SketchModel, linked: boolean, atoms: readonly number[]): number[] {
  const map = state.mapping;
  if (!linked || !map) return [];
  const ids = atoms.map((r) => map.record_parent_sketch[r]).map((i) => (i === null || i === undefined ? undefined : model.atoms[i]?.id));
  return [...new Set(ids.filter((x): x is number => x !== undefined))];
}

/** Whether the sketch still links to the computed record (see session.sketchChanged). */
function useStale(model: SketchModel, state: ComputeState, geometry: MoleculeGeometryOut | undefined): boolean {
  const { rdkit } = useRDKit();
  const recordSmiles = useMemo(() => (rdkit && geometry ? molblockSmiles(rdkit, geometry.molblock) : ""), [rdkit, geometry]);
  return useMemo(
    () =>
      sketchChanged(model, state.model, () =>
        rdkit && state.model
          ? { drawn: exportSketch(rdkit, model, "smiles"), computed: exportSketch(rdkit, state.model, "smiles"), record: recordSmiles }
          : null,
      ),
    [model, state.model, rdkit, recordSmiles],
  );
}

/** Why the projection is off, or null. */
function projectionBlocker(state: ComputeState, model: SketchModel, geometry: MoleculeGeometryOut | undefined, viewer: GLViewer | null): string | null {
  const ready = state.status === "ready" || state.status === "qm_only";
  if (!state.molecule || !ready) return "compute the molecule first";
  if (!state.mapping || !state.model) return "no sketch mapping for this molecule: recompute the drawing";
  if (sketchTopology(model) !== sketchTopology(state.model)) return "sketch changed since compute";
  if (!geometry) return "the 3D geometry is still loading";
  if (!viewer) return "no 3D view (WebGL unavailable)";
  return null;
}

/** The sketcher's projection source: the computed molecule and the 3D view's quaternion now
 *  (and the convention check against 3Dmol's screen positions as a side effect). */
function useProjectionSource(state: ComputeState, model: SketchModel, geometry: MoleculeGeometryOut | undefined, viewer: GLViewer | null) {
  const [residual, setResidual] = useState<number | null>(null);
  const source: ProjectionSource = {
    disabledReason: projectionBlocker(state, model, geometry, viewer),
    view: () => {
      const mid = state.molecule?.molecule_id;
      const q = viewer ? viewQuaternion(viewer) : null;
      if (!viewer || !geometry || !state.mapping || !mid || !q) return null;
      const p = viewProjection(viewer, model, state.mapping, geometry);
      if (p) setResidual(screenResidual(viewer, geometry, model, state.mapping, p));
      return { moleculeId: mid, quaternion: q };
    },
  };
  return { source, residual };
}

const depictor: Depictor = (req, signal) => depict2d(req, { signal });

/** Everything the panels need, gathered by the page. */
interface PageState {
  route: Route;
  prefs: PrefsState["prefs"];
  setPrefs: PrefsState["setPrefs"];
  layout: LayoutState;
  dispatch: Dispatch<LayoutAction>;
}

const placeholder = (text: string) => <div className="muted wb-small wb-placeholder">{text}</div>;

/** The page's state: the drawing, the compute, the fetched data, the selection, the layers. */
function useWorkbenchData({ route, prefs }: Pick<PageState, "route" | "prefs">) {
  const prm = route.params;
  const set = prm.get("set") || DEFAULT_SET;
  const start = prm.get("start") || DEFAULT_START;
  const orb = parseIndex(prm.get("orb"));
  const rep = parseRep(prm.get("rep"), prefs.rep);
  const quality = parseQuality(prm.get("q"));
  const { model, setModel } = useSketchState(prm.get("smiles"), update);
  const info = useSketchInfo(model);
  const mol = useComputeSync(prm.get("m"), update);
  const { state } = mol;
  const ready = state.status === "ready" || state.status === "qm_only";
  const mid = ready ? (state.molecule?.molecule_id ?? null) : null;
  const geometry = useGeometry(mid);
  const properties = useProperties(mid);
  const symmetry = useSymmetry(mid);
  useEffect(() => setSymmetryView({ highlight: null }), [mid]);
  const shown = useOrbitalSet(mid, set, start);
  const canonical = useOrbitalSet(mid, CANONICAL, "");
  const [win, setWin] = useState<Window>("frontier");
  const sel = useSelection();
  const stale = useStale(model, state, geometry.data);
  const linked = !!state.mapping && !stale;
  const [viewer, setViewer] = useState<GLViewer | null>(null);
  const projection = useProjectionSource(state, model, geometry.data, viewer);
  const orbitalSpec: LayerSpec | null = mid && orb !== null ? { kind: "orbital", set, start: set === CANONICAL ? null : start, index: orb } : null;
  const layers = useWorkbenchLayers(prm.get("layers"), orbitalSpec, (v) => update({ layers: v }), prefs.orbitalIso);
  useOrbitalLayerReset(mid);
  const radicals = radicalElectrons(model);
  const onCompute = (retry = false) => {
    clearSelection();
    mol.compute(model, toMolblock(model), retry);
  };
  const highlight = useMemo(() => sketchHighlight(state, model, linked, sel.atoms), [state, model, linked, sel.atoms]);
  const pick = (i: number) => update({ orb: i });
  return { set, start, orb, rep, quality, model, setModel, info, state, ready, mid, geometry, properties, symmetry, shown, canonical, win, setWin, stale, linked, viewer, setViewer, projection, layers, radicals, onCompute, highlight, pick };
}

type WorkbenchData = ReturnType<typeof useWorkbenchData>;

function sketchPanel(d: WorkbenchData, prefs: PageState["prefs"], setPrefs: PageState["setPrefs"], height: number): PanelContent {
  const display = {
    fontPx: prefs.fontPx,
    bondWidth: prefs.bondWidth,
    colourAtoms: prefs.colourAtoms,
    labelStyle: prefs.labelStyle,
    styleId: prefs.drawingStyle,
    customStyle: prefs.customStyle,
  };
  const onStyle = (drawingStyle: StyleId, customStyle: DrawingStyle) => setPrefs({ drawingStyle, customStyle });
  const { model, setModel, highlight, projection, state, info, radicals, stale, onCompute } = d;
  return {
    body: (
      <>
        <Sketcher value={model} onChange={setModel} highlightAtoms={highlight} height={height} projection={projection.source} depictor={depictor} display={display} onStyle={onStyle} />
        <ComputeBar state={state} blocker={computeBlocker(info.valid, info.stale, radicals)} radicals={radicals} stale={stale} onCompute={onCompute} />
      </>
    ),
  };
}

/** The five panels' contents. */
function workbenchPanels(d: WorkbenchData, { prefs, setPrefs, layout, dispatch }: Omit<PageState, "route">): Record<PanelId, PanelContent> {
  const { set, start, orb, rep, quality, state, ready, mid, geometry, properties, symmetry, shown, canonical, win, setWin, viewer, setViewer, layers, info, pick } = d;
  return {
    sketch: sketchPanel(d, prefs, setPrefs, layout.heights.sketch),
    viewer: {
      body: (
        <>
          <ErrorBanner error={geometry.error} title="Cannot load the geometry" />
          {geometry.loading && <PendingNote label={geometry.pending?.label ?? "loading the geometry"} worker={geometry.pending?.worker} />}
          <ViewerPanel mid={mid} geometry={geometry.data} properties={properties.data} orbitals={shown.data} selectedOrbital={orb} onPickOrbital={pick} rep={rep} onRep={(r) => update({ rep: r === prefs.rep ? null : r })} quality={quality} onQuality={(q) => update({ q: q === "normal" ? null : q })} onViewer={setViewer} height={layout.heights.viewer} symmetry={symmetry.data} fileStem={`ksmech-${(state.molecule?.key ?? "molecule").replace(/[^A-Za-z0-9]+/g, "_").slice(0, 40)}`} />
        </>
      ),
    },
    orbitals: {
      note: mid ? null : "not computed",
      body: mid ? (
        <OrbitalPanel set={set} start={start} onSet={(s) => update({ set: s === DEFAULT_SET ? null : s, orb: null })} onStart={(s) => update({ start: s === DEFAULT_START ? null : s, orb: null })} shown={shown} canonical={canonical} selected={orb} onPick={pick} onPickCanonical={(i) => update({ set: null, orb: i })} window={win} onWindow={setWin} infoOpen={layout.orbitalInfo} onInfoToggle={() => dispatch({ type: "orbitalInfo" })} height={layout.heights.orbitals} degeneracyEv={prefs.degeneracyEv} onDegeneracy={(v) => setPrefs({ degeneracyEv: v })} />
      ) : (
        placeholder("Compute a molecule to see its orbital sets and energy levels.")
      ),
    },
    surfaces: {
      note: mid ? `${layers.layers.length} layer${layers.layers.length === 1 ? "" : "s"}` : "not computed",
      body: mid ? (
        <SurfaceControls viewer={viewer} mid={mid} quality={quality} layers={layers.layers} orbital={layers.orbitalSpec} onChange={layers.onChange} onRemove={layers.onRemove} onPin={layers.onPin} onAdd={layers.onAdd} />
      ) : (
        placeholder("Compute a molecule to draw orbitals, densities, Fukui functions and the ESP.")
      ),
    },
    properties: {
      body: <PropertyTabs molecule={ready ? state.molecule : null} properties={properties} orbitals={shown} selectedOrbital={orb} onPickOrbital={pick} descriptors={info.descriptors} symmetry={symmetry} />,
    },
  };
}

/** Another molecule: the orbital layers of the previous one do not apply. */
function useOrbitalLayerReset(mid: string | null): void {
  const lastMid = useRef(mid);
  useEffect(() => {
    const prev = lastMid.current;
    if (mid) lastMid.current = mid;
    if (!mid || !prev || prev === mid) return;
    const kept = parseLayers(parseHash().params.get("layers")).filter((s) => !isOrbitalKind(s.kind));
    update({ layers: kept.length ? formatLayers(kept) : null });
  }, [mid]);
}

function Workbench({ route }: { route: Route }) {
  const [layout, dispatch] = useLayoutState();
  const prefsState = usePrefs();
  const { prefs } = prefsState;
  const wb = useWorkbenchData({ route, prefs });
  const panels = workbenchPanels(wb, { prefs, setPrefs: prefsState.setPrefs, layout, dispatch });
  return (
    <div
      className="workbench"
      data-testid="workbench"
      data-molecule-id={wb.mid ?? ""}
      data-linked={wb.linked ? "true" : "false"}
      data-projection-residual={wb.projection.residual === null ? undefined : wb.projection.residual.toFixed(4)}
      data-preset={layout.preset}
      data-theme={prefs.theme}
    >
      <ChemistryBanner molecule={wb.state.molecule} />
      <LayoutBar layout={layout} dispatch={dispatch} prefs={prefs} setPrefs={prefsState.setPrefs} resetPrefs={prefsState.resetPrefs} />
      <LayoutGrid layout={layout} dispatch={dispatch} panels={panels} />
    </div>
  );
}

export function WorkbenchPage({ route }: PageProps) {
  const cap = useCapability("workbench");
  if (cap.known && !cap.enabled) return <CapabilityNotice name="workbench" title="The molecule workbench is disabled" />;
  return <Workbench route={route} />;
}
