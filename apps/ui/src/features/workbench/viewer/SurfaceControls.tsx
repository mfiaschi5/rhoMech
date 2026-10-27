// SurfaceControls: the surface layers of the 3D view, several at once. The first row is the
// selected orbital (it follows the list and the diagram; "pin" keeps a copy); pinned layers
// are in the hash. Isovalue, opacity, solid or mesh and colours change on the client only.
//
// DOM: <div data-testid="wb-surfaces">, one <div data-testid="wb-surface-layer" data-kind
// data-set data-index data-state="loading|pending|fetched|ready|error" data-fetched> per layer,
// the add controls add-layer-kind (select) and wb-add-layer (button).
import type { GLViewer } from "3dmol";
import { useState } from "react";
import { moleculeCubeUrl, type CubeKind, type CubeQuality } from "../../../api/workbench";
import { ErrorBanner } from "../../../components/ErrorBanner";
import { fmt, fmtSig } from "@ksmech/viz/format";
import { PendingNote } from "../../../components/PendingNote";
import { CUBE_KINDS, isOrbitalKind, KIND_LABELS, type LayerSpec } from "../params";
import { ColourBar } from "./ColourBar";
import { KIND_LOOK, type SurfaceLayer } from "./layers";
import { useLayerSurface } from "./useLayerSurface";

export interface SurfaceControlsProps {
  viewer: GLViewer | null;
  mid: string;
  quality: CubeQuality;
  layers: SurfaceLayer[];
  /** The selected orbital, for adding orbital layers (null: none selected). */
  orbital: LayerSpec | null;
  onChange(layer: SurfaceLayer): void;
  onRemove(id: string): void;
  onPin(layer: SurfaceLayer): void;
  onAdd(spec: LayerSpec): void;
}

function specText(s: LayerSpec): string {
  if (!isOrbitalKind(s.kind)) return KIND_LABELS[s.kind];
  return `${KIND_LABELS[s.kind]} · ${s.set}${s.start ? `/${s.start}` : ""} #${s.index}`;
}

function IntegralNote({ meta }: { meta: ReturnType<typeof useLayerSurface>["meta"] }) {
  if (!meta || meta.integral === null) return null;
  return (
    <span className="muted wb-small" data-testid="cube-integral" title={`units ${meta.units}; grid spacing ${fmt(meta.spacing, 3)} Å`}>
      ∫ {fmtSig(meta.integral, 4)}
      {meta.expected !== null ? ` (expected ${fmtSig(meta.expected, 4)})` : ""}
    </span>
  );
}

function LookControls({ layer, onChange }: { layer: SurfaceLayer; onChange(l: SurfaceLayer): void }) {
  const look = KIND_LOOK[layer.spec.kind];
  const set = (patch: Partial<SurfaceLayer>) => onChange({ ...layer, ...patch });
  return (
    <div className="wb-layer-look">
      <label className="wb-inline" title="isovalue (cube units)">
        iso
        <input type="range" data-testid="layer-iso" min={look.isoMin} max={look.isoMax} step={look.isoStep} value={layer.iso} onChange={(e) => set({ iso: Number(e.target.value) })} />
        <input type="number" className="wb-num-input" min={0} step={look.isoStep} value={layer.iso} onChange={(e) => Number(e.target.value) > 0 && set({ iso: Number(e.target.value) })} />
      </label>
      <label className="wb-inline">
        opacity
        <input type="range" min={0.1} max={1} step={0.05} value={layer.opacity} onChange={(e) => set({ opacity: Number(e.target.value) })} />
      </label>
      <select value={layer.mesh ? "mesh" : "solid"} onChange={(e) => set({ mesh: e.target.value === "mesh" })} aria-label="surface style">
        <option value="solid">solid</option>
        <option value="mesh">mesh</option>
      </select>
      {layer.spec.kind !== "esp" && <input type="color" value={layer.pos} onChange={(e) => set({ pos: e.target.value })} title={look.signed ? "+ colour" : "colour"} />}
      {look.signed && <input type="color" value={layer.neg} onChange={(e) => set({ neg: e.target.value })} title="− colour" />}
    </div>
  );
}

function EspRange({ layer, range, onChange }: { layer: SurfaceLayer; range: number; onChange(l: SurfaceLayer): void }) {
  return (
    <>
      <label className="wb-inline" title="ESP colour range ± (a.u.); empty: automatic">
        ±
        <input
          type="number"
          className="wb-num-input"
          min={0}
          step={0.005}
          placeholder="auto"
          value={layer.range ?? ""}
          onChange={(e) => onChange({ ...layer, range: e.target.value === "" || Number(e.target.value) <= 0 ? null : Number(e.target.value) })}
        />
      </label>
      <ColourBar role="esp" label="ESP on ρ" scale="rwb" min={-range} max={range} units="a.u." note="approximate: Mulliken point charges, mapped on the valence density" />
    </>
  );
}

function LayerRow(p: { layer: SurfaceLayer } & Omit<SurfaceControlsProps, "layers" | "orbital" | "onAdd">) {
  const { layer, mid, quality, onChange } = p;
  const surf = useLayerSurface(p.viewer, mid, layer, quality);
  const s = layer.spec;
  const url = moleculeCubeUrl(mid, { ...s, quality }, true);
  return (
    <div
      className="wb-layer"
      data-testid="wb-surface-layer"
      data-kind={s.kind}
      data-set={s.set ?? ""}
      data-index={s.index ?? ""}
      data-state={surf.state}
      data-fetched={surf.load.state === "fetched" ? "true" : "false"}
      data-follow={layer.follow ? "true" : "false"}
    >
      <div className="wb-layer-head">
        <input type="checkbox" checked={layer.visible} onChange={(e) => onChange({ ...layer, visible: e.target.checked })} aria-label="visible" />
        <strong>{layer.follow ? "selected: " : ""}</strong>
        <span>{specText(s)}</span>
        <IntegralNote meta={surf.meta} />
        <span className="wb-spacer" />
        {surf.load.state === "fetched" && (
          <a className="btn btn-small" href={url} download data-testid="cube-download">
            cube
          </a>
        )}
        {layer.follow ? (
          <button type="button" className="btn btn-small" onClick={() => p.onPin(layer)} title="keep this orbital as its own layer">
            pin
          </button>
        ) : (
          <button type="button" className="btn btn-small" onClick={() => p.onRemove(layer.id)} aria-label="remove layer">
            ×
          </button>
        )}
      </div>
      <LookControls layer={layer} onChange={onChange} />
      {s.kind === "esp" && surf.range !== null && <EspRange layer={layer} range={surf.range} onChange={onChange} />}
      {(surf.state === "loading" || surf.state === "pending") && <PendingNote label={surf.load.pending?.label ?? `loading the ${KIND_LABELS[s.kind]} cube`} worker={surf.load.pending?.worker} />}
      <ErrorBanner error={surf.load.error} title="Cannot load the cube" />
      {surf.drawError && <div className="banner banner-error">{surf.drawError}</div>}
    </div>
  );
}

function AddLayer({ orbital, onAdd }: { orbital: LayerSpec | null; onAdd(s: LayerSpec): void }) {
  const [kind, setKind] = useState<CubeKind>("density");
  const needsOrbital = isOrbitalKind(kind) && (!orbital || orbital.index === null);
  const add = () => onAdd(isOrbitalKind(kind) && orbital ? { ...orbital, kind } : { kind, set: null, start: null, index: null });
  return (
    <div className="row wb-add-layer">
      <select data-testid="add-layer-kind" value={kind} onChange={(e) => setKind(e.target.value as CubeKind)} aria-label="layer kind">
        {CUBE_KINDS.map((k) => (
          <option key={k} value={k}>
            {KIND_LABELS[k]}
            {isOrbitalKind(k) ? " (selected orbital)" : ""}
          </option>
        ))}
      </select>
      <button type="button" className="btn btn-small" data-testid="wb-add-layer" onClick={add} disabled={needsOrbital} title={needsOrbital ? "select an orbital first" : undefined}>
        + add layer
      </button>
    </div>
  );
}

export function SurfaceControls(props: SurfaceControlsProps) {
  const { layers, orbital, onAdd, ...rest } = props;
  return (
    <div className="wb-surfaces" data-testid="wb-surfaces">
      {layers.length === 0 && <p className="muted wb-small">Pick an orbital in the list or the level diagram, or add a density, Fukui, dual or ESP layer.</p>}
      {layers.map((l) => (
        <LayerRow key={l.id} layer={l} {...rest} />
      ))}
      <AddLayer orbital={orbital} onAdd={onAdd} />
    </div>
  );
}
