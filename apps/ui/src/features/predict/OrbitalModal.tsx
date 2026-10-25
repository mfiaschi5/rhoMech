// OrbitalModal: the donor and acceptor orbitals of one orbital reading in 3D (the workbench's
// GL viewer hook, chem/threed.ts surfaces): the pool geometry (GET /pool/{state}/geometry) and
// the localized orbitals' cubes (GET /pool/{state}/orbital/{token}) as ±iso surfaces, donor in
// blue / red, acceptor in green / orange, with an isovalue slider.
//
// Props: stateId, donor, acceptor (token ids), title, onClose.
// DOM: predict-orbital-modal (Modal) > predict-orbital-viewer[data-state="loading|ready|error|
// webgl-unavailable"][data-cubes="loading|ready|error"], predict-orbital-show-{donor|acceptor|both},
// predict-orbital-iso.
import type { GLShape } from "3dmol";
import { useEffect, useRef, useState } from "react";
import { getGeometry, getOrbitalCube } from "../../api/client";
import { usePendingCached } from "../../api/hooks";
import { addSurfaces, ISO_MAX, ISO_MIN, ISO_STEP, parseCube, removeShapes, SURFACE_COLORS, threeDmol } from "../../chem/threed";
import { ErrorBanner } from "../../components/ErrorBanner";
import { PendingNote } from "../../components/PendingNote";
import { useGLViewer } from "../workbench/viewer/useGLViewer";
import { Modal } from "./Modal";

export interface OrbitalModalProps {
  stateId: string;
  donor: number;
  acceptor: number;
  title: string;
  onClose: () => void;
}

type Show = "donor" | "acceptor" | "both";

function useCube(stateId: string, token: number) {
  return usePendingCached<string>(`/pool/${stateId}/orbital/${token}`, (o) => getOrbitalCube(stateId, token, {}, o));
}

export function OrbitalModal({ stateId, donor, acceptor, title, onClose }: OrbitalModalProps) {
  const [el, setEl] = useState<HTMLDivElement | null>(null);
  const handle = useGLViewer(el);
  const [show, setShow] = useState<Show>("both");
  const [iso, setIso] = useState(0.05);
  const geometry = usePendingCached(`/pool/${stateId}/geometry`, (o) => getGeometry(stateId, o));
  const dCube = useCube(stateId, donor);
  const aCube = useCube(stateId, acceptor);
  const shapes = useRef<GLShape[]>([]);
  const [drawError, setDrawError] = useState<string | null>(null);
  const { viewer } = handle;
  useEffect(() => {
    if (!viewer || !geometry.data) return;
    viewer.removeAllModels();
    for (const s of geometry.data.species) viewer.addModel(s.molblock, "sdf").setStyle({}, { stick: { radius: 0.14 }, sphere: { scale: 0.22 } });
    viewer.zoomTo();
    viewer.render();
  }, [viewer, geometry.data]);
  useEffect(() => {
    const lib = threeDmol();
    if (!viewer || !lib) return;
    try {
      removeShapes(viewer, shapes.current);
      shapes.current = [];
      if (show !== "acceptor" && dCube.data) shapes.current.push(...addSurfaces(viewer, parseCube(lib, dCube.data), iso, SURFACE_COLORS.donor));
      if (show !== "donor" && aCube.data) shapes.current.push(...addSurfaces(viewer, parseCube(lib, aCube.data), iso, SURFACE_COLORS.acceptor));
      viewer.render();
      setDrawError(null);
    } catch (err) {
      setDrawError(`cannot draw the orbitals: ${String(err)}`);
    }
  }, [viewer, dCube.data, aCube.data, show, iso]);
  const cubes = dCube.error || aCube.error ? "error" : dCube.data && aCube.data ? "ready" : "loading";
  const pending = geometry.pending ?? dCube.pending ?? aCube.pending;
  return (
    <Modal title={title} onClose={onClose} testId="predict-orbital-modal" wide>
      <div className="row pv-inline-row">
        {(["donor", "acceptor", "both"] as Show[]).map((s) => (
          <button key={s} type="button" className={`btn btn-small${show === s ? " btn-primary" : ""}`} data-testid={`predict-orbital-show-${s}`} aria-pressed={show === s} onClick={() => setShow(s)}>
            {s === "both" ? "both" : `${s} ${s === "donor" ? donor : acceptor}`}
          </button>
        ))}
        <label className="wb-inline">
          iso
          <input type="range" data-testid="predict-orbital-iso" min={ISO_MIN} max={ISO_MAX} step={ISO_STEP} value={iso} onChange={(e) => setIso(Number(e.target.value))} />
          <code className="code">{iso.toFixed(3)}</code>
        </label>
      </div>
      {pending && <PendingNote label={pending.label} worker={pending.worker} />}
      <div className="pv-viewer" data-testid="predict-orbital-viewer" data-state={handle.state} data-cubes={cubes}>
        <div ref={setEl} className="pv-viewer-canvas" />
        {handle.state !== "ready" && handle.state !== "loading" && <div className="pv-viewer-note muted">{handle.message ?? handle.state}</div>}
      </div>
      <ErrorBanner error={geometry.error ?? dCube.error ?? aCube.error} title="Cannot load the orbitals" />
      {drawError && <div className="banner banner-error">{drawError}</div>}
    </Modal>
  );
}
