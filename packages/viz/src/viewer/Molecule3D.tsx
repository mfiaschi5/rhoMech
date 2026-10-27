// Molecule3D: one molecule in 3Dmol (the record geometry's molblock, viewer/scene.ts's model and
// representation) with the ±iso surfaces of one cube on it (an orbital: blue / red when occupied,
// green / orange when virtual). Props only: the cube is text the caller loaded.
//
// DOM: <div data-testid=<testId> data-state="loading|ready|error|webgl-unavailable"
// data-surface="none|ready|error">.
import type { GLShape, GLViewer } from "3dmol";
import { useEffect, useState } from "react";
import type { MoleculeGeometryOut } from "../apiTypes";
import { addSurfaces, parseCube, removeShapes, SURFACE_COLORS, threeDmol } from "../chem/threed";
import { DEFAULT_REP, type Rep } from "./rep";
import { useModel, useStyle } from "./scene";
import { useGLViewer } from "./useGLViewer";

export interface Molecule3DProps {
  geometry: MoleculeGeometryOut;
  /** A Gaussian cube to draw as ±iso surfaces (null: none). */
  cube?: string | null;
  /** Occupied (donor colours) or virtual (acceptor colours). */
  occupied?: boolean;
  iso?: number;
  rep?: Rep;
  height?: number;
  testId?: string;
}

type SurfaceState = "none" | "ready" | "error";

function useCubeSurfaces(viewer: GLViewer | null, cube: string | null, iso: number, occupied: boolean): SurfaceState {
  const [state, setState] = useState<SurfaceState>("none");
  useEffect(() => {
    const lib = threeDmol();
    if (!viewer || !lib || !cube) return void setState("none");
    let shapes: GLShape[] = [];
    try {
      shapes = addSurfaces(viewer, parseCube(lib, cube), iso, occupied ? SURFACE_COLORS.donor : SURFACE_COLORS.acceptor, { opacity: 0.8 });
      viewer.render();
      setState("ready");
    } catch (err) {
      console.warn("orbital surface failed", err);
      setState("error");
    }
    return () => {
      removeShapes(viewer, shapes);
      try {
        viewer.render();
      } catch {
        // the viewer is gone (unmount)
      }
    };
  }, [viewer, cube, iso, occupied]);
  return state;
}

export function Molecule3D({ geometry, cube = null, occupied = true, iso = 0.05, rep = DEFAULT_REP, height = 360, testId = "molecule-3d" }: Molecule3DProps) {
  const [el, setEl] = useState<HTMLDivElement | null>(null);
  const handle = useGLViewer(el);
  const model = useModel(handle.viewer, geometry);
  useStyle(handle.viewer, model, rep, true, null);
  const surface = useCubeSurfaces(handle.viewer, cube, iso, occupied);
  return (
    <div className="viz-3d" data-testid={testId} data-state={handle.state} data-surface={surface} style={{ height }}>
      <div ref={setEl} className="viz-3d-canvas" />
      {handle.state !== "ready" && handle.state !== "loading" && <div className="viz-3d-note muted">{handle.message ?? handle.state}</div>}
    </div>
  );
}
