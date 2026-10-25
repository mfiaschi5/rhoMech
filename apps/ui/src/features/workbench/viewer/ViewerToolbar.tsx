// ViewerToolbar: representation, hydrogens, labels, colour by property, overlays, the measure
// tool, cube quality and exports of the workbench 3D view.
//
// DOM: rep-select, h-toggle, label-select, colour-select, wiberg-toggle, centroid-toggle,
// measure-select, quality-select, export-png, export-xyz, export-mol.
import type { CubeQuality } from "../../../api/client";
import type { AtomPropDef } from "../atomProps";
import { QUALITIES, REPS, type Rep } from "../params";
import type { MeasureMode } from "./measure";
import { LABEL_MODES, type LabelMode } from "./scene";

export interface ViewOptions {
  showH: boolean;
  labels: LabelMode;
  colour: string | null;
  wiberg: boolean;
  centroids: boolean;
  measure: MeasureMode;
}

export const DEFAULT_VIEW: ViewOptions = { showH: true, labels: "none", colour: null, wiberg: false, centroids: false, measure: "select" };

export interface ViewerToolbarProps {
  rep: Rep;
  onRep(r: Rep): void;
  quality: CubeQuality;
  onQuality(q: CubeQuality): void;
  view: ViewOptions;
  onView(v: ViewOptions): void;
  props: AtomPropDef[];
  canExport: boolean;
  onExport(what: "png" | "xyz" | "mol"): void;
}

const REP_LABELS: Record<Rep, string> = {
  "ball-and-stick": "ball and stick",
  sticks: "sticks",
  spacefill: "spacefill",
  wireframe: "wireframe",
  lines: "lines",
};

function Check({ testId, label, checked, onChange }: { testId: string; label: string; checked: boolean; onChange(v: boolean): void }) {
  return (
    <label className="wb-inline">
      <input type="checkbox" data-testid={testId} checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );
}

export function ViewerToolbar(p: ViewerToolbarProps) {
  const v = p.view;
  const set = (patch: Partial<ViewOptions>) => p.onView({ ...v, ...patch });
  return (
    <div className="wb-toolbar row">
      <select data-testid="rep-select" value={p.rep} onChange={(e) => p.onRep(e.target.value as Rep)} aria-label="representation">
        {REPS.map((r) => (
          <option key={r} value={r}>
            {REP_LABELS[r]}
          </option>
        ))}
      </select>
      <Check testId="h-toggle" label="H" checked={v.showH} onChange={(showH) => set({ showH })} />
      <select data-testid="label-select" value={v.labels} onChange={(e) => set({ labels: e.target.value as LabelMode })} aria-label="atom labels">
        {LABEL_MODES.map((m) => (
          <option key={m.value} value={m.value} disabled={m.value === "property" && !v.colour}>
            {m.label}
          </option>
        ))}
      </select>
      <select data-testid="colour-select" value={v.colour ?? ""} onChange={(e) => set({ colour: e.target.value || null })} aria-label="colour atoms by">
        <option value="">colour: element</option>
        {p.props.map((d) => (
          <option key={d.key} value={d.key} title={d.title}>
            colour: {d.label}
          </option>
        ))}
      </select>
      <Check testId="wiberg-toggle" label="Wiberg" checked={v.wiberg} onChange={(wiberg) => set({ wiberg })} />
      <Check testId="centroid-toggle" label="centroids" checked={v.centroids} onChange={(centroids) => set({ centroids })} />
      <select data-testid="measure-select" value={v.measure} onChange={(e) => set({ measure: e.target.value as MeasureMode })} aria-label="click mode" title="what a click on an atom does">
        <option value="select">click: select</option>
        <option value="distance">measure distance</option>
        <option value="angle">measure angle</option>
        <option value="dihedral">measure dihedral</option>
      </select>
      <select data-testid="quality-select" value={p.quality} onChange={(e) => p.onQuality(e.target.value as CubeQuality)} aria-label="cube quality" title="grid of the cubes (draft ≈ 120k, normal ≈ 350k, fine ≈ 1.2M points)">
        {QUALITIES.map((q) => (
          <option key={q} value={q}>
            grid: {q}
          </option>
        ))}
      </select>
      <span className="wb-spacer" />
      <button type="button" className="btn btn-small" data-testid="export-png" disabled={!p.canExport} onClick={() => p.onExport("png")}>
        PNG
      </button>
      <button type="button" className="btn btn-small" data-testid="export-xyz" disabled={!p.canExport} onClick={() => p.onExport("xyz")}>
        xyz
      </button>
      <button type="button" className="btn btn-small" data-testid="export-mol" disabled={!p.canExport} onClick={() => p.onExport("mol")}>
        molfile
      </button>
    </div>
  );
}
