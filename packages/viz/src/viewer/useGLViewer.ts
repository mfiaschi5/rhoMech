// The one 3Dmol GLViewer of the workbench: created when its element mounts (after a WebGL
// probe, in try/catch), cleared on unmount. useOverlay draws a set of shapes and labels that is
// replaced as a whole whenever its inputs change (halos, labels, centroids, measurements), so
// every drawing effect removes only what it added itself.
import type { GLShape, GLViewer } from "3dmol";
import { useEffect, useState } from "react";
import { probeWebGL, threeDmol, type ThreeDState } from "../chem/threed";

type Label = ReturnType<GLViewer["addLabel"]>;

export interface ViewerHandle {
  viewer: GLViewer | null;
  state: ThreeDState;
  message: string | null;
}

export function useGLViewer(el: HTMLDivElement | null): ViewerHandle {
  const [handle, setHandle] = useState<ViewerHandle>({ viewer: null, state: "loading", message: null });
  useEffect(() => {
    if (!el) return;
    const probe = probeWebGL();
    if (!probe.ok) return void setHandle({ viewer: null, state: "webgl-unavailable", message: probe.reason });
    const lib = threeDmol();
    if (!lib) return void setHandle({ viewer: null, state: "error", message: "3Dmol.js is not loaded (/vendor/3Dmol-min.js)" });
    let v: GLViewer | undefined;
    try {
      v = lib.createViewer(el, { backgroundColor: "white", antialias: true });
    } catch (err) {
      return void setHandle({ viewer: null, state: "webgl-unavailable", message: String(err) });
    }
    if (!v) return void setHandle({ viewer: null, state: "webgl-unavailable", message: "3Dmol could not create a viewer" });
    setHandle({ viewer: v, state: "ready", message: null });
    const created = v;
    return () => {
      setHandle({ viewer: null, state: "loading", message: null });
      try {
        created.clear();
      } catch {
        // the WebGL context may already be gone
      }
      el.replaceChildren();
    };
  }, [el]);
  return handle;
}

export interface Overlay {
  shapes: GLShape[];
  labels: Label[];
}

export const emptyOverlay = (): Overlay => ({ shapes: [], labels: [] });

function removeOverlay(viewer: GLViewer, o: Overlay): void {
  try {
    for (const s of o.shapes) viewer.removeShape(s);
    for (const l of o.labels) viewer.removeLabel(l);
    viewer.render();
  } catch {
    // the viewer was cleared first (unmount)
  }
}

/** Run `draw` on the viewer whenever `deps` change, removing the previous overlay first. */
export function useOverlay(viewer: GLViewer | null, draw: (v: GLViewer, o: Overlay) => void, deps: readonly unknown[]): void {
  useEffect(() => {
    if (!viewer) return;
    const o = emptyOverlay();
    try {
      draw(viewer, o);
      viewer.render();
    } catch (err) {
      console.warn("workbench 3D overlay failed", err);
    }
    return () => removeOverlay(viewer, o);
    // `draw` is a fresh closure every render; `deps` decide when to redraw
  }, [viewer, ...deps]);
}
