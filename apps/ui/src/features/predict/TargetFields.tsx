// TargetFields: the optional product of the Predict form, and SearchLine, the outcome of a
// search on the job page. With a target the job is a search towards it instead of the beam
// expansion (serve.predict.search, PredictIn.target_smiles): the k best pathways from the
// reactants to a pool that holds the target (by-products allowed), each at most "max steps"
// steps, within a budget of node expansions (each one a model call, and QM for new molecules).
// Arrival EXACT compares the species keys; with "up to protons" also SKELETON, the heavy-atom
// skeletons with their bond orders, so a product drawn neutral is reached by its ion.
//
// TargetFields props: value TargetSettings, onChange(patch), maxExpansions (the request limit).
// SearchLine props: net NetworkOut (renders nothing without net.search).
//
// DOM: predict-target (the section), predict-target-smiles-0 and predict-target-depiction-0
// (SmilesInput), predict-target-draw (DrawModal ids predict-target-draw-*), predict-target-clear,
// predict-target-skeleton, predict-target-k, predict-target-expansions; predict-search
// [data-reason] with predict-search-target.
import { useState } from "react";
import { NumberField } from "../../components/NumberField";
import { SmilesInput } from "../../components/SmilesInput";
import type { NetworkOut } from "@ksmech/viz/v9types";
import { DrawModal } from "./DrawModal";
import type { TargetSettings } from "./target";

export function TargetFields({ value, onChange, maxExpansions = 300 }: { value: TargetSettings; onChange: (patch: Partial<TargetSettings>) => void; maxExpansions?: number }) {
  const [drawing, setDrawing] = useState(false);
  const on = value.smiles.trim().length > 0;
  return (
    <fieldset className="pv-target stack" data-testid="predict-target">
      <legend className="field-label">
        target product <span className="muted">(optional: search the pathways to it instead of exploring)</span>
      </legend>
      {drawing && (
        <DrawModal
          prefix="predict-target"
          initial={value.smiles}
          onClose={() => setDrawing(false)}
          onUse={(smiles) => {
            onChange({ smiles });
            setDrawing(false);
          }}
        />
      )}
      <div className="row pv-inline-row">
        <SmilesInput index={0} prefix="predict-target" label="product SMILES" placeholder="e.g. CC(C)([O-])OC" value={value.smiles} onChange={(smiles) => onChange({ smiles })} />
        <div className="stack">
          <button type="button" className="btn btn-small" data-testid="predict-target-draw" onClick={() => setDrawing(true)} title="draw the product in the sketcher">
            Draw
          </button>
          {on && (
            <button type="button" className="btn btn-small" data-testid="predict-target-clear" onClick={() => onChange({ smiles: "" })}>
              clear
            </button>
          )}
        </div>
      </div>
      {on && (
        <div className="row pv-inline-row">
          <label className="field-inline pv-small">
            <input type="checkbox" data-testid="predict-target-skeleton" checked={value.skeleton} onChange={(e) => onChange({ skeleton: e.target.checked })} />
            also up to protons (the heavy-atom skeleton)
          </label>
          <NumberField label="pathways" testId="predict-target-k" value={value.k} min={1} max={50} onChange={(k) => onChange({ k })} />
          <NumberField label="max expansions" testId="predict-target-expansions" value={value.maxExpansions} min={1} max={maxExpansions} hint="each one a model call" onChange={(maxExpansions) => onChange({ maxExpansions })} />
        </div>
      )}
    </fieldset>
  );
}

const REASONS: Record<string, string> = {
  K_FOUND: "found the pathways asked for",
  EXHAUSTED: "nothing left to expand",
  POP_BUDGET: "stopped at the path budget",
  START_IS_GOAL: "the reactants already hold the target",
  CANCELLED: "cancelled",
};

function reasonText(s: NonNullable<NetworkOut["search"]>): string {
  if (s.reason === null) return "searching…";
  if (s.reason === "EXHAUSTED" && (s.cuts.EXPANSION_BUDGET ?? 0) > 0) return `stopped at the budget of ${s.max_expansions} expansions`;
  return REASONS[s.reason] ?? s.reason.toLowerCase().replace(/_/g, " ");
}

export function SearchLine({ net }: { net: NetworkOut }) {
  const s = net.search;
  if (!s) return null;
  const arrived = Object.keys(s.arrivals).length;
  return (
    <span className="pv-wrap" data-testid="predict-search" data-reason={s.reason ?? ""}>
      → <code className="code" data-testid="predict-search-target">{s.target_smiles}</code>{" "}
      <span className="muted">
        ({s.levels.join(" or ").toLowerCase()}): {reasonText(s)}; {net.pathways.length} pathway{net.pathways.length === 1 ? "" : "s"}, {arrived} state{arrived === 1 ? "" : "s"} at the target, {s.expanded} / {s.max_expansions} expansions
      </span>
    </span>
  );
}
