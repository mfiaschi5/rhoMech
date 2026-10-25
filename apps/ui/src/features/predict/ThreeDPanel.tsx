// ThreeDPanel: the oriented contact placement the KS3 kernel evaluated, in 3Dmol.js
// (GET /contact/{state}/{donor}/{site}/{acceptor}/{site}, design §8.3, §11.7): the donor
// fragment, the placed acceptor cap and its faded context, both lobes, the dashed a-b distance
// and both orbital cubes as ±iso surfaces, with an isovalue slider. WebGL is probed first, in a
// try/catch; without it the placement data and checks are still shown.
//
// Props:
//   target    ContactRef           state and tuple
//   onClose   () => void
//
// DOM: <section data-testid="threed-panel" data-state="loading|ready|error|webgl-unavailable"
// data-placement-loaded="true|false" data-cubes="loading|ready|error">, contact-status[data-consistent],
// iso-slider.
import type { GLShape, GLViewer, VolumeData } from "3dmol";
import { useEffect, useRef, useState } from "react";
import { getContact, getContactCube, getTextUrl, toApiError, type ApiError, type ContactRef, type RequestOptions } from "../../api/client";
import type { ContactOut } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";
import { PendingNote } from "../../components/PendingNote";
import { ProvenanceBadge } from "../../components/ProvenanceBadge";
import { exact, fmt, fmtSig } from "./format";
import { usePendingCached, type PendingInfo } from "./hooks";
import { addSurfaces, drawContact, ISO_MAX, ISO_MIN, ISO_STEP, parseCube, probeWebGL, removeShapes, SURFACE_COLORS, threeDmol, type ThreeDState } from "./threed";

export interface ThreeDPanelProps {
  target: ContactRef;
  onClose: () => void;
}

const contactKey = (t: ContactRef) => `/contact/${t.stateId}/${t.donor}/${t.donorSite}/${t.acceptor}/${t.acceptorSite}`;

interface ViewState {
  state: ThreeDState;
  message: string | null;
}

interface Cubes {
  donor: string | null;
  acceptor: string | null;
  error: ApiError | null;
  pending: PendingInfo | null;
}

function cubeText(c: ContactOut, ref: ContactRef, which: "donor" | "acceptor", o: RequestOptions): Promise<string> {
  const inline = c.cubes?.[which];
  if (inline) return Promise.resolve(inline);
  const url = c.cube_urls[which];
  return url ? getTextUrl(url, o) : getContactCube(ref, which, o);
}

/** Both cube texts of a placement (pending-aware). */
function useCubes(c: ContactOut | undefined, ref: ContactRef): Cubes {
  const [cubes, setCubes] = useState<Cubes>({ donor: null, acceptor: null, error: null, pending: null });
  useEffect(() => {
    if (!c) return;
    let live = true;
    const o: RequestOptions = { onPending: (label, worker) => live && setCubes((s) => ({ ...s, pending: { label, worker } })) };
    setCubes({ donor: null, acceptor: null, error: null, pending: null });
    Promise.all([cubeText(c, ref, "donor", o), cubeText(c, ref, "acceptor", o)]).then(
      ([donor, acceptor]) => live && setCubes({ donor, acceptor, error: null, pending: null }),
      (err: unknown) => live && setCubes({ donor: null, acceptor: null, error: toApiError(err), pending: null }),
    );
    return () => {
      live = false;
    };
    // the placement identifies the cubes
  }, [c]);
  return cubes;
}

/** The 3Dmol viewer: created once the placement arrived, torn down on unmount. */
function useViewer(el: HTMLDivElement | null, c: ContactOut | undefined) {
  const [view, setView] = useState<ViewState>({ state: "loading", message: null });
  const viewer = useRef<GLViewer | null>(null);
  useEffect(() => {
    if (!c || !el) return;
    const probe = probeWebGL();
    if (!probe.ok) return void setView({ state: "webgl-unavailable", message: probe.reason });
    const lib = threeDmol();
    if (!lib) return void setView({ state: "error", message: "3Dmol.js is not loaded (/vendor/3Dmol-min.js)" });
    let v: GLViewer | undefined;
    try {
      v = lib.createViewer(el, { backgroundColor: "white", antialias: true });
    } catch (err) {
      return void setView({ state: "webgl-unavailable", message: String(err) });
    }
    if (!v) return void setView({ state: "webgl-unavailable", message: "3Dmol could not create a viewer" });
    try {
      drawContact(v, c);
      v.zoomTo();
      v.render();
      viewer.current = v;
      setView({ state: "ready", message: null });
    } catch (err) {
      setView({ state: "error", message: `drawing the placement failed: ${String(err)}` });
    }
    const created = v;
    return () => {
      viewer.current = null;
      try {
        created.clear();
      } catch {
        // the WebGL context may already be gone
      }
      el.replaceChildren();
    };
  }, [c, el]);
  return { view, viewer };
}

/** (Re)draw both cubes' ±iso surfaces whenever the cubes or the isovalue change. */
function useSurfaces(viewer: GLViewer | null, cubes: Cubes, iso: number) {
  const shapes = useRef<{ owner: GLViewer | null; list: GLShape[] }>({ owner: null, list: [] });
  const vols = useRef<{ donor: VolumeData; acceptor: VolumeData } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    vols.current = null;
    const lib = threeDmol();
    if (!lib || !cubes.donor || !cubes.acceptor) return;
    try {
      vols.current = { donor: parseCube(lib, cubes.donor), acceptor: parseCube(lib, cubes.acceptor) };
      setError(null);
    } catch (err) {
      setError(`cannot parse the cube files: ${String(err)}`);
    }
  }, [cubes.donor, cubes.acceptor]);
  useEffect(() => {
    if (!viewer || !vols.current) return;
    // shapes are removed by position: only ever remove this viewer's own shapes
    try {
      if (shapes.current.owner === viewer) removeShapes(viewer, shapes.current.list);
      shapes.current = { owner: viewer, list: [] };
      const v = vols.current;
      shapes.current.list = [...addSurfaces(viewer, v.donor, iso, SURFACE_COLORS.donor), ...addSurfaces(viewer, v.acceptor, iso, SURFACE_COLORS.acceptor)];
      viewer.render();
    } catch (err) {
      setError(`cannot draw the isosurfaces: ${String(err)}`);
    }
  }, [viewer, cubes.donor, cubes.acceptor, iso]);
  return error;
}

function ContactStatus({ c }: { c: ContactOut }) {
  return (
    <div className="contact-status" data-testid="contact-status" data-consistent={c.consistent ? "true" : "false"}>
      <span className={`badge ${c.consistent ? "badge-ok" : "badge-error"}`}>{c.consistent ? "consistent" : "INCONSISTENT"}</span>{" "}
      <span className="muted">
        tier {c.tier}, roll {c.roll_step} ({fmt(c.roll_deg, 0)}°), slots {c.slots.join("/")}
        {c.is_contact_argmax ? ", the contact (argmax |S|) site pair" : ", not the contact site pair"}
        {c.same_fragment ? ", intramolecular (the cap is a displaced copy)" : ""}
      </span>
      <div className="contact-numbers">
        d {fmt(c.distance_ang, 2)} Å (clash limit {fmt(c.clash_limit_ang, 2)} Å), S {fmtSig(c.S, 5)}, β {fmtSig(c.beta, 5)} eV, KS3 at these sites {fmtSig(c.ks3_site, 5)} eV, G{" "}
        {fmtSig(c.G, 5)} 1/eV{c.clash ? ", CLASH" : ""}
      </div>
      <details>
        <summary>checks</summary>
        <dl className="kv">
          {Object.entries(c.checks).map(([k, v]) => (
            <div key={k} className="kv-row">
              <dt>{k}</dt>
              <dd>{typeof v === "boolean" ? (v ? "pass" : "FAIL") : exact(v)}</dd>
            </div>
          ))}
        </dl>
      </details>
      {c.acceptor_context && <div className="muted">faded: context only; the kernel placed only the cap</div>}
      <div className="muted">{c.donor_fragment.frame_note}</div>
    </div>
  );
}

export function ThreeDPanel({ target, onClose }: ThreeDPanelProps) {
  const [el, setEl] = useState<HTMLDivElement | null>(null);
  const res = usePendingCached<ContactOut>(contactKey(target), (o) => getContact(target, { cubes: "urls" }, o));
  const c = res.data;
  const { view, viewer } = useViewer(el, c);
  const cubes = useCubes(c, target);
  const [iso, setIso] = useState<number | null>(null);
  const isoValue = iso ?? c?.iso_default ?? 0.05;
  const surfaceError = useSurfaces(view.state === "ready" ? viewer.current : null, cubes, isoValue);
  const state: ThreeDState = res.error ? "error" : c ? view.state : "loading";
  const cubeState = cubes.error ? "error" : cubes.donor && cubes.acceptor ? "ready" : "loading";
  return (
    <section className="panel threed-panel" data-testid="threed-panel" data-state={state} data-placement-loaded={c ? "true" : "false"} data-cubes={c ? cubeState : "loading"}>
      <div className="row drawer-head">
        <h3>3D contact placement</h3>
        <button type="button" className="btn btn-small" onClick={onClose}>
          close 3D
        </button>
      </div>
      {res.loading && <PendingNote label={res.pending?.label ?? "loading the placement"} worker={res.pending?.worker} />}
      <ErrorBanner error={res.error} title="Cannot reconstruct the contact" />
      {c && <ContactStatus c={c} />}
      {view.state === "webgl-unavailable" && c && <div className="banner banner-warn">3D view unavailable: {view.message}</div>}
      {view.state === "error" && c && <div className="banner banner-error">{view.message}</div>}
      <div className="threed-viewer" ref={setEl} data-testid="threed-viewer" />
      {c && (
        <div className="row">
          <label className="field-inline">
            isovalue
            <input type="range" data-testid="iso-slider" min={ISO_MIN} max={ISO_MAX} step={ISO_STEP} value={isoValue} onChange={(e) => setIso(Number(e.target.value))} />
            <span className="num">{isoValue.toFixed(3)}</span>
          </label>
          {cubeState === "loading" && <PendingNote label={cubes.pending?.label ?? "computing both orbital cubes"} worker={cubes.pending?.worker} />}
          <span className="muted">donor blue/red, acceptor green/orange</span>
        </div>
      )}
      <ErrorBanner error={cubes.error} title="Cannot load the orbital cubes" />
      {surfaceError && <div className="banner banner-error">{surfaceError}</div>}
      {c && <ProvenanceBadge provenance={c.provenance} compact />}
    </section>
  );
}
