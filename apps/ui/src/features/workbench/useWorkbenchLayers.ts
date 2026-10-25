// The surface layers of the page: the selected-orbital layer (follows set, start and orb) and
// the pinned layers of the hash `layers` param. A layer's look (iso, opacity, style, colours)
// is kept per spec on the client, so editing it never touches the hash or refetches.
import { useMemo, useRef, useState } from "react";
import { formatLayers, parseLayers, specKey, type LayerSpec } from "./params";
import { newLayer, type SurfaceLayer } from "./viewer/layers";

export interface WorkbenchLayers {
  layers: SurfaceLayer[];
  orbitalSpec: LayerSpec | null;
  onChange(l: SurfaceLayer): void;
  onRemove(id: string): void;
  onPin(l: SurfaceLayer): void;
  onAdd(s: LayerSpec): void;
}

/** `layersParam`: the hash value; `setParam` writes a new one. */
export function useWorkbenchLayers(layersParam: string | null, orbitalSpec: LayerSpec | null, setParam: (v: string | null) => void): WorkbenchLayers {
  const specs = useMemo(() => parseLayers(layersParam), [layersParam]);
  const looks = useRef(new Map<string, SurfaceLayer>());
  const [version, setVersion] = useState(0);
  const [follow, setFollow] = useState<SurfaceLayer>(() => newLayer({ kind: "orbital", set: null, start: null, index: 0 }, true));
  const pinned = useMemo(
    () =>
      specs.map((s) => {
        const k = specKey(s);
        let l = looks.current.get(k);
        if (!l) {
          l = newLayer(s);
          looks.current.set(k, l);
        }
        return l;
      }),
    // `version` bumps when a look changes
    [specs, version],
  );
  const followLayer = orbitalSpec ? { ...follow, spec: orbitalSpec } : null;
  const duplicate = followLayer && pinned.some((l) => specKey(l.spec) === specKey(followLayer.spec));
  const layers = followLayer && !duplicate ? [followLayer, ...pinned] : pinned;
  const writeSpecs = (next: LayerSpec[]) => setParam(next.length ? formatLayers(next) : null);
  const addSpec = (s: LayerSpec, look?: SurfaceLayer) => {
    const k = specKey(s);
    if (look) looks.current.set(k, { ...look, id: newLayer(s).id, spec: s, follow: false });
    if (!specs.some((x) => specKey(x) === k)) writeSpecs([...specs, s]);
    setVersion((v) => v + 1);
  };
  return {
    layers,
    orbitalSpec,
    onChange: (l) => {
      if (l.follow) return setFollow(l);
      looks.current.set(specKey(l.spec), l);
      setVersion((v) => v + 1);
    },
    onRemove: (id) => writeSpecs(pinned.filter((l) => l.id !== id).map((l) => l.spec)),
    onPin: (l) => addSpec(l.spec, l),
    onAdd: (s) => addSpec(s),
  };
}
