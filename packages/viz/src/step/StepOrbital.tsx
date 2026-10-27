// StepOrbital (step viewport variant C): the scheme on the left (StepScheme, screen style), the
// pool in 3D on the right (the workbench GL viewer) with the donor and acceptor orbitals of the
// readings switched on as ±iso surfaces (donor blue / red, acceptor green / orange; the
// dominant channel on by default), and between them a two-level energy diagram of the shown
// reading: the donor and acceptor LMOs (E_r, from the cubes), the dominant canonical pair k -> m
// (HOMO -> LUMO, with its share of the term) and the gap. Hovering an atom or bond in 2D (or a
// reading) marks its atoms in 3D. The species are laid side by side as the geometry's display
// offsets say; each cube moves with its species (stepCube.ts). The 3D data comes in as
// `orbital` (OrbitalData): the pool geometry, the source depiction and a cube loader (the app
// fetches them, apps/ui/src/features/predict/step/useOrbitalData.tsx; a scene reads its assets).
//
// DOM: predict-step-orbital > the StepScheme ids, predict-step-energy (svg, data-reading),
// predict-step-orbital-viewer[data-state][data-cubes="loading|ready|error|none"],
// predict-step-orbital-reading-<k> (checkbox), predict-step-orbital-show-<donor|acceptor|both>,
// predict-step-orbital-iso.
import type { GLShape, GLViewer } from "3dmol";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { DepictionOut, GeometryOut } from "../apiTypes";
import { addSurfaces, ISO_MAX, ISO_MIN, ISO_STEP, parseCube, removeShapes, SURFACE_COLORS, threeDmol } from "../chem/threed";
import { SCREEN_STYLE, type DrawingStyle } from "../sketch/style";
import { useGLViewer } from "../viewer/useGLViewer";
import { cubeEnergy, offsetCube, offsetMolblock, poolAtoms3D, speciesOfCube } from "./stepCube";
import type { HoverSet } from "./stepSides";
import { StepScheme } from "./StepScheme";
import { captionParts, diagSymbol, dominantReading, signedNum } from "./stepText";
import type { StepVariantProps } from "./StepTextbook";
import { useLoaded } from "../useLoaded";
import type { OrbitalReadingOut } from "../v9types";

export const ORBITAL_STYLE: DrawingStyle = { ...SCREEN_STYLE, bondLengthPt: 30, fontSizePt: 12, lineWidthPt: 1.2, chargePlacement: "circled", labelMethyls: true, scaleWithZoom: true };

type Show = "donor" | "acceptor" | "both";

/** What the 3D side reads: the pool's geometry, its depiction (a 2D atom found in 3D) and the
 *  cube text of one orbital token. `key` names the pool (cubes are loaded by key and token). */
export interface OrbitalData {
  key: string;
  geometry: GeometryOut | undefined;
  depiction: DepictionOut | null | undefined;
  loadCube: (token: number) => Promise<string>;
  /** Shown over the viewer while the geometry loads (the app: its pending note). */
  pending?: ReactNode;
  /** Shown under the viewer (the app: why the geometry did not load). */
  error?: ReactNode;
}

/** Theme background for the GL canvas (the panel colour, so dark mode stays dark). */
function panelColour(el: HTMLElement | null): string {
  if (!el) return "#ffffff";
  const c = getComputedStyle(el).getPropertyValue("--panel").trim();
  return c || "#ffffff";
}

interface Viewer3DProps {
  orbital: OrbitalData;
  readings: readonly OrbitalReadingOut[];
  on: ReadonlySet<number>;
  show: Show;
  iso: number;
  hover: HoverSet;
  onEnergies: (e: Map<number, number>) => void;
  prefix: string;
}

function useSpecies(viewer: GLViewer | null, geometry: GeometryOut | undefined, host: HTMLElement | null) {
  useEffect(() => {
    if (!viewer || !geometry) return;
    viewer.setBackgroundColor(panelColour(host), 1);
    viewer.removeAllModels();
    for (const s of geometry.species) viewer.addModel(offsetMolblock(s.molblock, s.display_offset), "sdf").setStyle({}, { stick: { radius: 0.13 }, sphere: { scale: 0.2 } });
    viewer.zoomTo();
    viewer.zoom(1.12);
    viewer.render();
  }, [viewer, geometry, host]);
}

/** The surfaces of one token's cube in its species' place. */
function CubeLayer({ viewer, orbital, token, geometry, colours, iso, onState, onEnergy }: { viewer: GLViewer | null; orbital: OrbitalData; token: number; geometry: GeometryOut | undefined; colours: { pos: string; neg: string }; iso: number; onState: (token: number, s: string) => void; onEnergy: (token: number, e: number | null) => void }) {
  const cube = useLoaded<string>(`${orbital.key}/orbital/${token}`, () => orbital.loadCube(token));
  const shapes = useRef<GLShape[]>([]);
  const placed = useMemo(() => {
    if (!cube.data || !geometry) return null;
    const k = speciesOfCube(cube.data, geometry);
    return offsetCube(cube.data, k === null ? [0, 0, 0] : geometry.species[k].display_offset);
  }, [cube.data, geometry]);
  useEffect(() => onState(token, cube.error ? "error" : cube.data ? "ready" : "loading"), [cube.error, cube.data, token]);
  useEffect(() => onEnergy(token, cube.data ? cubeEnergy(cube.data) : null), [cube.data, token]);
  useEffect(() => {
    const lib = threeDmol();
    if (!viewer || !lib || !placed) return;
    try {
      shapes.current = addSurfaces(viewer, parseCube(lib, placed), iso, colours, { opacity: 0.78 });
      viewer.render();
    } catch (err) {
      console.warn("orbital surface failed", err);
    }
    return () => {
      removeShapes(viewer, shapes.current);
      shapes.current = [];
      try {
        viewer.render();
      } catch {
        // the viewer is gone (unmount)
      }
    };
  }, [viewer, placed, iso, colours]);
  return null;
}

function useHoverSpheres(viewer: GLViewer | null, depiction: DepictionOut | null | undefined, geometry: GeometryOut | undefined, hover: HoverSet) {
  const map = useMemo(() => (depiction && geometry ? poolAtoms3D(depiction, geometry) : null), [depiction, geometry]);
  useEffect(() => {
    if (!viewer || !map || !hover.atoms.size) return;
    const shapes: GLShape[] = [];
    for (const i of hover.atoms) {
      const a = map.get(i);
      if (a) shapes.push(viewer.addSphere({ center: { x: a.at[0], y: a.at[1], z: a.at[2] }, radius: 0.85, color: "#ffc400", alpha: 0.85 }));
    }
    viewer.render();
    return () => {
      removeShapes(viewer, shapes);
      try {
        viewer.render();
      } catch {
        // unmounted
      }
    };
  }, [viewer, map, hover]);
}

function Viewer3D({ orbital, readings, on, show, iso, hover, onEnergies, prefix }: Viewer3DProps) {
  const [el, setEl] = useState<HTMLDivElement | null>(null);
  const handle = useGLViewer(el);
  const geometry = orbital.geometry;
  useSpecies(handle.viewer, geometry, el);
  useHoverSpheres(handle.viewer, orbital.depiction, geometry, hover);
  const [states, setStates] = useState<Map<number, string>>(new Map());
  const energies = useRef(new Map<number, number>());
  const tokens = new Map<number, "donor" | "acceptor">();
  for (const k of on) {
    const r = readings[k];
    if (!r) continue;
    if (show !== "acceptor") tokens.set(r.donor_token, "donor");
    if (show !== "donor") tokens.set(r.acceptor_token, "acceptor");
  }
  const vals = [...tokens.keys()].map((t) => states.get(t) ?? "loading");
  const cubes = !tokens.size ? "none" : vals.includes("error") ? "error" : vals.every((v) => v === "ready") ? "ready" : "loading";
  return (
    <div className="ss-viewer" data-testid={`${prefix}-orbital-viewer`} data-state={handle.state} data-cubes={cubes}>
      <div ref={setEl} className="ss-viewer-canvas" />
      {[...tokens].map(([t, kind]) => (
        <CubeLayer key={`${t}-${kind}`} viewer={handle.viewer} orbital={orbital} token={t} geometry={geometry} colours={SURFACE_COLORS[kind]} iso={iso} onState={(tok, s) => setStates((m) => new Map(m).set(tok, s))} onEnergy={(tok, e) => {
          if (e === null || energies.current.get(tok) === e) return;
          energies.current.set(tok, e);
          onEnergies(new Map(energies.current));
        }} />
      ))}
      {handle.state !== "ready" && handle.state !== "loading" && <div className="ss-viewer-note muted">{handle.message ?? handle.state}</div>}
      {orbital.pending}
      <div className="ss-viewer-legend">
        <span className="ss-swatch" style={{ background: SURFACE_COLORS.donor.pos }} />
        <span className="ss-swatch" style={{ background: SURFACE_COLORS.donor.neg }} /> donor
        <span className="ss-swatch" style={{ background: SURFACE_COLORS.acceptor.pos }} />
        <span className="ss-swatch" style={{ background: SURFACE_COLORS.acceptor.neg }} /> acceptor
      </div>
      {orbital.error}
    </div>
  );
}

const EW = 214;
const EH = 300;

/** The two-level diagram of one reading (see the header). */
export function StepEnergy({ r, energies, prefix, label }: { r: OrbitalReadingOut | null; energies: ReadonlyMap<number, number>; prefix: string; label: (i: number) => string }) {
  if (!r) return <div className="ss-energy muted pv-small" data-testid={`${prefix}-energy`}>no reading</div>;
  const c = r.canonical;
  const eD = energies.get(r.donor_token) ?? null;
  const eA = energies.get(r.acceptor_token) ?? null;
  const vals = [c?.e_k_ev, c?.e_m_ev, eD, eA].filter((v): v is number => v !== null && v !== undefined);
  const lo = Math.min(...vals) - 0.8;
  const hi = Math.max(...vals) + 0.8;
  const y = (e: number) => 26 + ((hi - e) / Math.max(hi - lo, 1e-6)) * (EH - 70);
  const level = (x: number, e: number, cls: string, text: string, sub: string, side: "l" | "r") => (
    <g className={cls}>
      <line x1={x - 22} x2={x + 22} y1={y(e)} y2={y(e)} />
      <text x={side === "l" ? x - 26 : x + 26} y={y(e) - 2} textAnchor={side === "l" ? "end" : "start"} className="ss-e-label">
        {text}
      </text>
      <text x={side === "l" ? x - 26 : x + 26} y={y(e) + 10} textAnchor={side === "l" ? "end" : "start"} className="ss-e-sub">
        {sub}
      </text>
    </g>
  );
  const [xc, xl] = [74, 142];
  const p = captionParts(r);
  return (
    <figure className="ss-energy" data-testid={`${prefix}-energy`} data-reading={`${r.i}-${r.j}`}>
      <svg width={EW} height={EH} viewBox={`0 0 ${EW} ${EH}`} role="img" aria-label={`energy levels of ${p.channel}`}>
        <text x={xc} y={12} textAnchor="middle" className="ss-e-head">canonical</text>
        <text x={xl} y={12} textAnchor="middle" className="ss-e-head">LMO</text>
        {c && eD !== null && <line className="ss-e-link" x1={xc + 22} y1={y(c.e_k_ev)} x2={xl - 22} y2={y(eD)} />}
        {c && eA !== null && <line className="ss-e-link" x1={xc + 22} y1={y(c.e_m_ev)} x2={xl - 22} y2={y(eA)} />}
        {c && (
          <>
            {level(xc, c.e_k_ev, "ss-e-occ", c.k_label.replace("-", "−"), signedNum(c.e_k_ev, 2), "l")}
            {level(xc, c.e_m_ev, "ss-e-virt", c.m_label.replace("-", "−"), signedNum(c.e_m_ev, 2), "l")}
            <path className="ss-e-electrons" d={`M ${xc - 6} ${y(c.e_k_ev) + 6} v -12 l -3 4 M ${xc + 6} ${y(c.e_k_ev) - 6} v 12 l 3 -4`} />
            <line className="ss-e-gap" x1={xc} y1={y(c.e_m_ev) + 4} x2={xc} y2={y(c.e_k_ev) - 8} markerEnd="url(#ss-e-arrow)" markerStart="url(#ss-e-arrow)" />
            <text className="ss-e-gaptext" x={xc + 4} y={(y(c.e_m_ev) + y(c.e_k_ev)) / 2 + 3}>
              Δ {(c.e_m_ev - c.e_k_ev).toFixed(2)}
            </text>
          </>
        )}
        {eD !== null && level(xl, eD, "ss-e-donor", diagSymbol(r.donor_diag) ?? `r ${r.donor_token}`, signedNum(eD, 2), "r")}
        {eA !== null && level(xl, eA, "ss-e-acceptor", diagSymbol(r.acceptor_diag) ?? `s ${r.acceptor_token}`, signedNum(eA, 2), "r")}
        <defs>
          <marker id="ss-e-arrow" viewBox="0 0 6 6" refX="3" refY="3" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M0,0 L6,3 L0,6 z" className="ss-e-marker" />
          </marker>
        </defs>
        <text x={EW / 2} y={EH - 30} textAnchor="middle" className="ss-e-foot">
          {label(r.i)}–{label(r.j)} · eV
        </text>
        <text x={EW / 2} y={EH - 16} textAnchor="middle" className="ss-e-foot">
          G(r,s) {signedNum(r.g_rs, 3)} · term {signedNum(r.term, 3)}
        </text>
        <text x={EW / 2} y={EH - 2} textAnchor="middle" className="ss-e-foot">
          {c?.share != null ? `pair share ${Math.round(100 * c.share)} % · ` : ""}G^atom {signedNum(r.g_atom, 2)}
        </text>
      </svg>
    </figure>
  );
}

export function StepOrbital(props: StepVariantProps & { orbital: OrbitalData }) {
  const { sides, step, prefix, orbital, label } = props;
  const top = dominantReading(step.readings);
  const [on, setOn] = useState<Set<number>>(() => new Set(top >= 0 ? [top] : []));
  const [show, setShow] = useState<Show>("both");
  const [iso, setIso] = useState(0.05);
  const [energies, setEnergies] = useState<Map<number, number>>(new Map());
  const [showR, setShowR] = useState(false);
  const key = JSON.stringify(step.matrix) + step.readings.length;
  useEffect(() => setOn(new Set(top >= 0 ? [top] : [])), [key]);
  const shown = [...on].sort((a, b) => a - b)[0] ?? top;
  return (
    <div className="ss-orbital" data-testid={`${prefix}-orbital`}>
      <div className="ss-frame ss-orb-scheme">
        {sides.source ? (
          <StepScheme sides={sides} step={step} prefix={prefix} style={ORBITAL_STYLE} showR={showR} hover={props.hover} onHover={props.onHover} above={props.above} below={props.below} productSmiles={props.productSmiles.join(".")} variant="orbital" fit={{ maxBondPx: 50, maxHeight: 340, minHeight: 200 }} />
        ) : (
          <div className="pv-canvas-placeholder muted">{sides.loading ? "loading the depiction…" : (sides.error ?? "no depiction")}</div>
        )}
        <ul className="ss-orb-readings">
          {step.readings.map((r, k) => {
            const p = captionParts(r);
            return (
              <li key={k} onPointerEnter={() => props.onHover({ atoms: new Set([r.i, r.j]), bonds: new Set([r.i < r.j ? `${r.i}-${r.j}` : `${r.j}-${r.i}`]) })} onPointerLeave={() => props.onHover(null)}>
                <label>
                  <input type="checkbox" data-testid={`${prefix}-orbital-reading-${k}`} checked={on.has(k)} onChange={(e) => setOn((s) => { const n = new Set(s); if (e.target.checked) n.add(k); else n.delete(k); return n; })} />
                  <b>
                    {label(r.i)}–{label(r.j)}
                  </b>{" "}
                  <span className="ss-cap-channel">{p.channel}</span> <span className="muted">{p.g}</span>
                </label>
              </li>
            );
          })}
          {!step.readings.length && <li className="muted">{step.note ?? "no forming bond to read"}</li>}
          <li className="ss-orb-r">
            <label className="pv-small">
              <input type="checkbox" data-testid={`${prefix}-show-r`} checked={showR} onChange={(e) => setShowR(e.target.checked)} />
              R marks on the reactants
            </label>
          </li>
        </ul>
      </div>
      <StepEnergy r={step.readings[shown] ?? null} energies={energies} prefix={prefix} label={label} />
      <div className="ss-frame ss-orb-3d">
        <div className="ss-orb-tools">
          <span className="ss-seg" role="group" aria-label="which orbitals">
            {(["donor", "acceptor", "both"] as Show[]).map((v) => (
              <button key={v} type="button" className={`ss-seg-btn${show === v ? " ss-seg-on" : ""}`} aria-pressed={show === v} data-testid={`${prefix}-orbital-show-${v}`} onClick={() => setShow(v)}>
                {v}
              </button>
            ))}
          </span>
          <label className="wb-inline pv-small">
            iso
            <input type="range" data-testid={`${prefix}-orbital-iso`} min={ISO_MIN} max={ISO_MAX} step={ISO_STEP} value={iso} onChange={(e) => setIso(Number(e.target.value))} />
            <code className="code">{iso.toFixed(3)}</code>
          </label>
        </div>
        <Viewer3D orbital={orbital} readings={step.readings} on={on} show={show} iso={iso} hover={props.hover} onEnergies={setEnergies} prefix={prefix} />
      </div>
    </div>
  );
}
