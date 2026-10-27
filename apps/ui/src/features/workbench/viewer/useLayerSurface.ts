// One surface layer's data and drawing: the cube(s) it needs are loaded (LRU, AbortController)
// when the molecule, the spec or the quality change; the isosurfaces are redrawn on the client
// when the look changes. ESP layers load the ESP and the density cube and paint the ESP on the
// density isosurface.
import type { GLShape, GLViewer } from "3dmol";
import { useEffect, useMemo, useRef, useState } from "react";
import { isAbort, toApiError, type ApiError } from "../../../api/http";
import { type CubeMeta, type CubeQuality } from "../../../api/workbench";
import type { PendingInfo } from "../../../api/hooks";
import { addMappedSurface, addSurface, addSurfaces, removeMapped, removeShapes, type MappedSurface } from "../../../chem/threed";
import type { LayerSpec } from "../params";
import { KIND_LOOK, type SurfaceLayer } from "./layers";
import { espRange, loadVolume, peekVolume, volumeKey, type LoadedVolume } from "./volumes";

export type LayerState = "loading" | "pending" | "fetched" | "ready" | "error";

export interface LayerLoad {
  state: "loading" | "pending" | "fetched" | "error";
  main: LoadedVolume | null;
  /** ESP: the density the ESP is painted on. */
  density: LoadedVolume | null;
  error: ApiError | null;
  pending: PendingInfo | null;
}

const DENSITY: LayerSpec = { kind: "density", set: null, start: null, index: null };

function initial(mid: string, spec: LayerSpec, quality: CubeQuality): LayerLoad {
  const main = peekVolume(mid, spec, quality) ?? null;
  const density = spec.kind === "esp" ? (peekVolume(mid, DENSITY, quality) ?? null) : null;
  const done = main !== null && (spec.kind !== "esp" || density !== null);
  return { state: done ? "fetched" : "loading", main: done ? main : null, density: done ? density : null, error: null, pending: null };
}

/** Load the cube(s) of a spec; reloads only when the molecule, spec or quality change. */
export function useLayerLoad(mid: string, spec: LayerSpec, quality: CubeQuality): LayerLoad {
  const key = volumeKey(mid, spec, quality);
  const [load, setLoad] = useState<{ key: string; load: LayerLoad }>(() => ({ key, load: initial(mid, spec, quality) }));
  useEffect(() => {
    const first = initial(mid, spec, quality);
    setLoad({ key, load: first });
    if (first.state === "fetched") return;
    const ctl = new AbortController();
    const onPending = (label: string, worker: PendingInfo["worker"]) =>
      setLoad((s) => (s.key === key ? { key, load: { ...s.load, state: "pending", pending: { label, worker } } } : s));
    const o = { signal: ctl.signal, onPending };
    const both = Promise.all([loadVolume(mid, spec, quality, o), spec.kind === "esp" ? loadVolume(mid, DENSITY, quality, o) : Promise.resolve(null)]);
    both.then(
      ([main, density]) => setLoad({ key, load: { state: "fetched", main, density, error: null, pending: null } }),
      (err: unknown) => {
        if (!isAbort(err)) setLoad({ key, load: { state: "error", main: null, density: null, error: toApiError(err), pending: null } });
      },
    );
    return () => ctl.abort();
    // `key` identifies (mid, spec, quality)
  }, [key]);
  return load.key === key ? load.load : initial(mid, spec, quality);
}

type Drawn = { shapes: GLShape[]; mapped: MappedSurface | null };

function draw(viewer: GLViewer, layer: SurfaceLayer, load: LayerLoad, range: number): Drawn {
  const vol = load.main?.vol;
  const style = { opacity: layer.opacity, wireframe: layer.mesh };
  if (!vol || !layer.visible) return { shapes: [], mapped: null };
  if (layer.spec.kind === "esp") {
    const rho = load.density?.vol;
    return rho ? { shapes: [], mapped: addMappedSurface(viewer, rho, vol, layer.iso, range, style) } : { shapes: [], mapped: null };
  }
  if (KIND_LOOK[layer.spec.kind].signed) return { shapes: addSurfaces(viewer, vol, layer.iso, { pos: layer.pos, neg: layer.neg }, style), mapped: null };
  return { shapes: [addSurface(viewer, vol, layer.iso, layer.pos, style)], mapped: null };
}

export interface LayerSurface {
  load: LayerLoad;
  state: LayerState;
  /** ESP: the colour range in use (a.u.). */
  range: number | null;
  meta: CubeMeta | null;
  drawError: string | null;
}

export function useLayerSurface(viewer: GLViewer | null, mid: string, layer: SurfaceLayer, quality: CubeQuality): LayerSurface {
  const load = useLayerLoad(mid, layer.spec, quality);
  const drawn = useRef<Drawn>({ shapes: [], mapped: null });
  const [drawState, setDrawState] = useState<{ ok: boolean; error: string | null }>({ ok: false, error: null });
  const autoRange = useMemo(() => {
    const rho = load.density?.vol;
    const esp = load.main?.vol;
    return layer.spec.kind === "esp" && rho && esp ? espRange(rho, esp, KIND_LOOK.density.iso) : null;
  }, [load.main, load.density, layer.spec.kind]);
  const range = layer.range ?? autoRange ?? 0.05;
  useEffect(() => {
    if (!viewer || load.state !== "fetched") return void setDrawState({ ok: false, error: null });
    try {
      drawn.current = draw(viewer, layer, load, range);
      viewer.render();
      setDrawState({ ok: !!load.main?.vol, error: load.main?.parseError ?? null });
    } catch (err) {
      setDrawState({ ok: false, error: `cannot draw the surface: ${String(err)}` });
    }
    return () => {
      try {
        removeShapes(viewer, drawn.current.shapes);
        if (drawn.current.mapped) removeMapped(viewer, drawn.current.mapped);
        viewer.render();
      } catch {
        // cleared viewer
      }
      drawn.current = { shapes: [], mapped: null };
    };
  }, [viewer, load, layer.iso, layer.opacity, layer.mesh, layer.pos, layer.neg, layer.visible, range]);
  const state: LayerState = load.state === "fetched" && drawState.ok ? "ready" : load.state;
  return { load, state, range: layer.spec.kind === "esp" ? range : null, meta: load.main?.meta ?? null, drawError: drawState.error };
}
