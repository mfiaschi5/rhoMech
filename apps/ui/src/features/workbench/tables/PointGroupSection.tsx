// PointGroupSection: the point group of the computed molecule in the Properties Global tab
// (GET /molecule/{id}/symmetry, viewer/symmetry.ts): the Schoenflies symbol and order, the
// tolerance (Å; an operation counts when every atom lands within it of an atom of the same
// element), the RMS of the symmetrized structure, the hint of a higher group at a looser
// tolerance, the symmetry elements (a click highlights one in the 3D view, again clears it) and
// the symmetry-equivalent atom classes (a click selects the class's atoms; colourable in 3D).
//
// DOM: <section data-testid="pointgroup-section">, pointgroup [data-point-group data-order
// data-tolerance], pointgroup-tolerance (number input), pointgroup-hint (button applying the
// looser tolerance), pointgroup-show, pointgroup-classes (checkboxes shared with the 3D
// toolbar), symmetry-element [data-id data-label data-kind aria-pressed], symmetry-class
// [data-element data-size].
import { useEffect, useState, type ReactNode } from "react";
import type { ApiError } from "../../../api/http";
import type { PendingCached } from "../../../api/hooks";
import type { PointGroupOut, SymmetryElementOut } from "../../../api/types.gen";
import { ErrorBanner } from "../../../components/ErrorBanner";
import { fmt } from "../../../components/format";
import { PendingNote } from "../../../components/PendingNote";
import { selectAtoms } from "../store";
import { classColour, clampTolerance, setSymmetryView, TOLERANCE_MAX, TOLERANCE_MIN, useSymmetryView } from "../viewer/symmetry";
import "./pointgroup.css";

/** A Schoenflies symbol or element label with its subscript: D6h -> D<sub>6h</sub>,
 *  C2′ -> C<sub>2</sub>′, σv -> σ<sub>v</sub>, ∞σv -> ∞σ<sub>v</sub>. */
export function Schoenflies({ text }: { text: string }): ReactNode {
  const m = /^(∞?)([A-Zσ])([^′″]*)([′″]*)$/u.exec(text);
  if (!m || !m[3]) return text;
  return (
    <>
      {m[1]}
      {m[2]}
      <sub>{m[3]}</sub>
      {m[4]}
    </>
  );
}

function ToleranceInput() {
  const { tolerance } = useSymmetryView();
  const [text, setText] = useState(tolerance.toFixed(2));
  useEffect(() => setText(tolerance.toFixed(2)), [tolerance]);
  const commit = (raw: string) => {
    setText(raw);
    const t = clampTolerance(Number.parseFloat(raw));
    if (t !== null && raw.trim() !== "") setSymmetryView({ tolerance: t });
  };
  return (
    <label className="wb-inline wb-pg-tol" title={`an operation counts when every atom lands within this distance of an atom of the same element (${TOLERANCE_MIN}–${TOLERANCE_MAX} Å)`}>
      tolerance
      <input type="number" data-testid="pointgroup-tolerance" min={TOLERANCE_MIN} max={TOLERANCE_MAX} step={0.01} value={text} onChange={(e) => commit(e.target.value)} onBlur={() => setText(tolerance.toFixed(2))} aria-label="point-group tolerance (Å)" />
      Å
    </label>
  );
}

function ElementChip({ e, lit }: { e: SymmetryElementOut; lit: boolean }) {
  const where = e.kind === "i" ? "the centre of mass" : e.kind === "sigma" ? "plane" : "axis";
  const title = `${e.label}${e.principal ? " (principal axis)" : ""}: ${e.atoms.length} atom${e.atoms.length === 1 ? "" : "s"} on the ${where}${e.atoms.length ? ` (${e.atoms.join(", ")})` : ""}; largest residual ${fmt(e.deviation_ang, 3)} Å`;
  return (
    <button type="button" className={`wb-pg-chip${lit ? " active" : ""}${e.principal ? " principal" : ""}`} data-testid="symmetry-element" data-id={e.id} data-label={e.label} data-kind={e.kind} aria-pressed={lit} title={title} onClick={() => setSymmetryView({ highlight: lit ? null : e.id })}>
      <Schoenflies text={e.label} />
      {e.atoms.length > 0 && <span className="muted"> ·{e.atoms.length}</span>}
    </button>
  );
}

function Check({ testId, label, checked, onChange }: { testId: string; label: string; checked: boolean; onChange(v: boolean): void }) {
  return (
    <label className="wb-inline">
      <input type="checkbox" data-testid={testId} checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );
}

function Classes({ data, coloured }: { data: PointGroupOut; coloured: boolean }) {
  return (
    <div className="wb-pg-chips" role="list" aria-label="symmetry-equivalent atoms">
      {data.classes.map((c, k) => (
        <button key={c.atoms[0]} type="button" role="listitem" className="wb-pg-chip" data-testid="symmetry-class" data-element={c.element} data-size={c.atoms.length} title={`atoms ${c.atoms.join(", ")}: select them`} onClick={(ev) => selectAtoms(c.atoms, ev.shiftKey || ev.ctrlKey || ev.metaKey)}>
          {coloured && <span className="wb-pg-swatch" style={{ background: classColour(k) }} />}
          {c.atoms.length}×{c.element}
        </button>
      ))}
    </div>
  );
}

function Hint({ data }: { data: PointGroupOut }) {
  const h = data.looser;
  if (!h) return null;
  return (
    <div className="wb-small">
      would be{" "}
      <button type="button" className="btn btn-small" data-testid="pointgroup-hint" data-point-group={h.point_group} onClick={() => setSymmetryView({ tolerance: h.tolerance, highlight: null })} title="use this tolerance">
        <Schoenflies text={h.point_group} /> at {h.tolerance.toFixed(2)} Å
      </button>
    </div>
  );
}

function PointGroupBody({ data }: { data: PointGroupOut }) {
  const view = useSymmetryView();
  const order = data.order === null ? "∞" : String(data.order);
  return (
    <>
      <div className="wb-pg-head">
        <span className="wb-pg-symbol" data-testid="pointgroup" data-point-group={data.point_group} data-order={order} data-tolerance={data.tolerance}>
          <Schoenflies text={data.point_group} />
        </span>
        <span className="muted wb-small">
          order {order} · {data.top} top · RMS {fmt(data.rms_ang, 3)} Å (max {fmt(data.max_deviation_ang, 3)})
        </span>
      </div>
      <Hint data={data} />
      {data.notes.map((n) => (
        <div key={n} className="muted wb-small">
          {n}
        </div>
      ))}
      <div className="wb-pg-chips" role="list" aria-label="symmetry elements">
        {data.elements.length === 0 && <span className="muted wb-small">no symmetry element besides the identity</span>}
        {data.elements.map((e) => (
          <ElementChip key={e.id} e={e} lit={view.highlight === e.id} />
        ))}
      </div>
      <div className="wb-small muted">symmetry-equivalent atoms ({data.classes.length} classes)</div>
      <Classes data={data} coloured={view.classes} />
    </>
  );
}

export function PointGroupSection({ res }: { res: PendingCached<PointGroupOut> }) {
  const view = useSymmetryView();
  return (
    <section className="wb-pg" data-testid="pointgroup-section">
      <div className="wb-pg-bar">
        <strong>Point group</strong>
        <ToleranceInput />
        <Check testId="pointgroup-show" label="show in 3D" checked={view.show} onChange={(show) => setSymmetryView({ show })} />
        <Check testId="pointgroup-classes" label="colour classes" checked={view.classes} onChange={(classes) => setSymmetryView({ classes })} />
      </div>
      {res.loading && <PendingNote label={res.pending?.label ?? "finding the point group"} worker={res.pending?.worker} />}
      <ErrorBanner error={res.error as ApiError | null} title="Cannot find the point group" />
      {res.data !== undefined && <PointGroupBody data={res.data} />}
    </section>
  );
}
