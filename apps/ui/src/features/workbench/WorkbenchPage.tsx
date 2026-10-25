// WorkbenchPage (#/workbench): draw a molecule, compute it (POST /molecule: the QM record and
// tokens, display only), then look at it: the 3D view with surfaces, the orbital sets and their
// energy levels, and the property tables. The sketcher is on the left with the orbital panel,
// the 3D view on the right, the tables below.
//
// Hash params (params.ts): smiles, m, set, start, orb, rep, layers, q. The drawing persists in
// sessionStorage (session.ts). Selections link the tables, the 3D view and the sketch through
// the POST's sketch mapping, only while the sketch is unchanged since the compute.
//
// DOM: <div data-testid="workbench" data-molecule-id data-linked>, wb-compute, wb-status
// [data-status=idle|posting|running|ready|qm_only|failed|missing|error], wb-stale-note,
// wb-chemistry-banner[data-chemistry], wb-mapping-note.
import { useEffect, useMemo, useRef, useState } from "react";
import type { PageProps } from "../../registry";
import { navigate, parseHash, withParams, type Route } from "../../router";
import { useRDKit } from "../../chem/rdkit";
import { CapabilityNotice } from "../../components/CapabilityNotice";
import { ErrorBanner } from "../../components/ErrorBanner";
import { PendingNote } from "../../components/PendingNote";
import { useCapability } from "../../state/store";
import type { MoleculeOut } from "../../api/types.gen";
import { useGeometry, useOrbitalSet, useProperties } from "./data";
import { OrbitalPanel } from "./orbitals/OrbitalPanel";
import type { Window } from "./orbitals/levels";
import { CANONICAL, DEFAULT_SET, DEFAULT_START, isMoleculeId, isOrbitalKind, parseIndex, parseLayers, parseQuality, parseRep, formatLayers, type LayerSpec } from "./params";
import { restoreComputed, restoreSketch, saveComputed, saveSketch, sketchSignature } from "./session";
import { emptySketch, fromMolblock, importSketch, Sketcher, toMolblock, useSketchInfo, type SketchModel } from "./sketch";
import { clearSelection, useSelection } from "./store";
import { PropertyTabs } from "./tables/PropertyTabs";
import { useMolecule, type ComputeState } from "./useMolecule";
import { useWorkbenchLayers } from "./useWorkbenchLayers";
import { ViewerPanel } from "./viewer/ViewerPanel";
import "./workbench.css";

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

function ComputeBar(p: { state: ComputeState; canCompute: boolean; stale: boolean; onCompute(retry?: boolean): void }) {
  const { state } = p;
  const busy = state.status === "posting" || state.status === "running";
  const failure = state.molecule?.error ?? null;
  return (
    <div className="wb-compute-bar">
      <div className="row">
        <button type="button" className="btn btn-primary" data-testid="wb-compute" disabled={!p.canCompute || busy} onClick={() => p.onCompute()}>
          {state.status === "idle" ? "Compute" : "Recompute"}
        </button>
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
      </div>
      {p.stale && (
        <div className="muted wb-small" data-testid="wb-stale-note">
          The sketch changed since the compute: selections are no longer linked to it. Recompute to link them again.
        </div>
      )}
      <ErrorBanner error={state.error} title="Cannot compute the molecule" />
      {failure && <ErrorBanner error={`${failure.message}`} title={`Featurize failed: ${failure.code}`} testId="wb-failure" />}
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

function Workbench({ route }: { route: Route }) {
  const prm = route.params;
  const set = prm.get("set") || DEFAULT_SET;
  const start = prm.get("start") || DEFAULT_START;
  const orb = parseIndex(prm.get("orb"));
  const rep = parseRep(prm.get("rep"));
  const quality = parseQuality(prm.get("q"));
  const { model, setModel } = useSketchState(prm.get("smiles"), update);
  const info = useSketchInfo(model);
  const mol = useComputeSync(prm.get("m"), update);
  const { state } = mol;
  const ready = state.status === "ready" || state.status === "qm_only";
  const mid = ready ? (state.molecule?.molecule_id ?? null) : null;
  const geometry = useGeometry(mid);
  const properties = useProperties(mid);
  const shown = useOrbitalSet(mid, set, start);
  const canonical = useOrbitalSet(mid, CANONICAL, "");
  const [win, setWin] = useState<Window>("frontier");
  const sel = useSelection();
  const signature = useMemo(() => sketchSignature(model), [model]);
  const computedSignature = useMemo(() => (state.model ? sketchSignature(state.model) : null), [state.model]);
  const stale = computedSignature !== null && computedSignature !== signature;
  const linked = !!state.mapping && !stale;
  const orbitalSpec: LayerSpec | null = mid && orb !== null ? { kind: "orbital", set, start: set === CANONICAL ? null : start, index: orb } : null;
  const layers = useWorkbenchLayers(prm.get("layers"), orbitalSpec, (v) => update({ layers: v }));
  const onCompute = (retry = false) => {
    clearSelection();
    mol.compute(model, toMolblock(model), retry);
  };
  const lastMid = useRef(mid);
  useEffect(() => {
    // another molecule: the orbital layers of the previous one do not apply
    const prev = lastMid.current;
    if (mid) lastMid.current = mid;
    if (!mid || !prev || prev === mid) return;
    const kept = parseLayers(parseHash().params.get("layers")).filter((s) => !isOrbitalKind(s.kind));
    update({ layers: kept.length ? formatLayers(kept) : null });
  }, [mid]);
  const highlight = useMemo(() => sketchHighlight(state, model, linked, sel.atoms), [state, model, linked, sel.atoms]);
  const pick = (i: number) => update({ orb: i });
  return (
    <div className="workbench" data-testid="workbench" data-molecule-id={mid ?? ""} data-linked={linked ? "true" : "false"}>
      <ChemistryBanner molecule={state.molecule} />
      <div className="wb-grid">
        <div className="wb-left">
          <section className="panel wb-sketch-panel">
            <Sketcher value={model} onChange={setModel} highlightAtoms={highlight} height={340} />
            <ComputeBar state={state} canCompute={info.valid && !info.stale} stale={stale} onCompute={onCompute} />
          </section>
          {mid && (
            <OrbitalPanel
              set={set}
              start={start}
              onSet={(s) => update({ set: s === DEFAULT_SET ? null : s, orb: null })}
              onStart={(s) => update({ start: s === DEFAULT_START ? null : s, orb: null })}
              shown={shown}
              canonical={canonical}
              selected={orb}
              onPick={pick}
              onPickCanonical={(i) => update({ set: null, orb: i })}
              window={win}
              onWindow={setWin}
            />
          )}
        </div>
        <div className="wb-right">
          <ErrorBanner error={geometry.error} title="Cannot load the geometry" />
          {geometry.loading && <PendingNote label={geometry.pending?.label ?? "loading the geometry"} worker={geometry.pending?.worker} />}
          <ViewerPanel
            mid={mid}
            geometry={geometry.data}
            properties={properties.data}
            orbitals={shown.data}
            selectedOrbital={orb}
            onPickOrbital={pick}
            rep={rep}
            onRep={(r) => update({ rep: r === "ball-and-stick" ? null : r })}
            quality={quality}
            onQuality={(q) => update({ q: q === "normal" ? null : q })}
            layers={layers.layers}
            orbitalSpec={layers.orbitalSpec}
            onLayerChange={layers.onChange}
            onLayerRemove={layers.onRemove}
            onLayerPin={layers.onPin}
            onLayerAdd={layers.onAdd}
            fileStem={`ksmech-${(state.molecule?.key ?? "molecule").replace(/[^A-Za-z0-9]+/g, "_").slice(0, 40)}`}
          />
        </div>
      </div>
      <PropertyTabs molecule={ready ? state.molecule : null} properties={properties} orbitals={shown} selectedOrbital={orb} onPickOrbital={pick} descriptors={info.descriptors} />
    </div>
  );
}

export function WorkbenchPage({ route }: PageProps) {
  const cap = useCapability("workbench");
  if (cap.known && !cap.enabled) return <CapabilityNotice name="workbench" title="The molecule workbench is disabled" />;
  return <Workbench route={route} />;
}
