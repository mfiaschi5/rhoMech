// OrbitalPanel: the orbital set and LMO start pickers, the energy-level diagram and the orbital
// list of the shown set. Picking an orbital (list, diagram, a centroid in 3D) shows it as the
// selected-orbital surface layer.
//
// DOM: set-select, start-select, window-select, <div data-testid="orbital-list"> with
// <tr data-testid="orbital-row" data-set data-index data-selected data-frontier>.
import type { ApiError } from "../../../api/client";
import type { PendingCached } from "../../../api/hooks";
import type { OrbitalRowOut, OrbitalSetOut } from "../../../api/types.gen";
import { ErrorBanner } from "../../../components/ErrorBanner";
import { fmt, moLabel } from "../../../components/format";
import { PendingNote } from "../../../components/PendingNote";
import { CANONICAL, FALLBACK_SETS, FALLBACK_STARTS } from "../params";
import { LevelDiagram } from "./LevelDiagram";
import { frontierLabel, type Window } from "./levels";

export interface OrbitalPanelProps {
  set: string;
  start: string;
  onSet(set: string): void;
  onStart(start: string): void;
  shown: PendingCached<OrbitalSetOut>;
  canonical: PendingCached<OrbitalSetOut>;
  selected: number | null;
  onPick(index: number): void;
  onPickCanonical(index: number): void;
  window: Window;
  onWindow(w: Window): void;
}

const SET_LABELS: Record<string, string> = { canonical: "canonical MOs", ibo: "IBO", pm_mulliken: "Pipek-Mezey (Mulliken)" };

export function centresText(r: OrbitalRowOut, n = 3): string {
  const parts = r.centres.slice(0, n).map((c) => `${c.element}${c.atom} ${fmt(c.weight, 2)}`);
  return parts.join(", ") + (r.centres.length > n ? ` +${r.centres.length - n}` : "");
}

function rowLabel(r: OrbitalRowOut): string {
  if (r.label) return moLabel(r.label);
  return r.kind === "occupied" ? "occ" : "virt";
}

function OrbitalList({ data, selected, onPick }: { data: OrbitalSetOut; selected: number | null; onPick(i: number): void }) {
  return (
    <div className="wb-orbital-list" data-testid="orbital-list">
      <table className="table">
        <thead>
          <tr>
            <th>#</th>
            <th>{data.set === CANONICAL ? "label" : "block"}</th>
            <th className="num">occ</th>
            <th className="num">{data.set === CANONICAL ? "E (eV)" : "E_r (eV)"}</th>
            <th>centres (record atoms)</th>
            {data.set !== CANONICAL && (
              <th className="diagnostic" title="diagnostic: never model input">
                type
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {data.orbitals.map((r) => (
            <tr
              key={r.index}
              data-testid="orbital-row"
              data-set={data.set}
              data-index={r.index}
              data-frontier={r.frontier ?? ""}
              data-selected={r.index === selected ? "true" : "false"}
              className={`wb-click${r.index === selected ? " selected" : ""}`}
              onClick={() => onPick(r.index)}
            >
              <td className="num">{r.index}</td>
              <td>{rowLabel(r)}</td>
              <td className="num">{fmt(r.occupation, 2)}</td>
              <td className="num">{fmt(r.energy_ev, 3)}</td>
              <td className="wb-small">{centresText(r)}</td>
              {data.set !== CANONICAL && (
                <td className="diagnostic" title="diagnostic: never model input">
                  {r.diagnostic?.type ?? "–"}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Loading({ res, what }: { res: PendingCached<OrbitalSetOut>; what: string }) {
  return (
    <>
      {res.loading && <PendingNote label={res.pending?.label ?? `loading the ${what}`} worker={res.pending?.worker} />}
      <ErrorBanner error={res.error as ApiError | null} title={`Cannot load the ${what}`} />
    </>
  );
}

function SetStats({ data }: { data: OrbitalSetOut }) {
  const st = data.stats;
  return (
    <div className="muted wb-small" data-testid="orbital-set-stats">
      {data.is_model_set && <span className="badge badge-ok">model input set</span>} {st.n_occupied} occupied, {st.n_virtual} virtual
      {st.converged !== null ? `, ${st.converged ? "converged" : "NOT converged"} (${st.sweeps_occ ?? "–"}/${st.sweeps_virt ?? "–"} sweeps)` : ""}; ‖CᵀSC − I‖ {st.orthonormality_error.toExponential(1)}
      {data.notes.map((n) => (
        <div key={n}>{n}</div>
      ))}
    </div>
  );
}

export function OrbitalPanel(p: OrbitalPanelProps) {
  const data = p.shown.data;
  const sets = data?.sets ?? p.canonical.data?.sets ?? FALLBACK_SETS;
  const starts = data?.starts ?? p.canonical.data?.starts ?? FALLBACK_STARTS;
  const lmo = p.set !== CANONICAL;
  const homo = p.canonical.data?.orbitals.find((r) => r.frontier === "homo");
  return (
    <section className="panel wb-orbitals" data-testid="wb-orbitals">
      <div className="row wb-orbital-pickers">
        <label className="wb-inline">
          set
          <select data-testid="set-select" value={p.set} onChange={(e) => p.onSet(e.target.value)}>
            {sets.map((s) => (
              <option key={s} value={s}>
                {SET_LABELS[s] ?? s}
              </option>
            ))}
          </select>
        </label>
        <label className="wb-inline" title="the start of the localization">
          start
          <select data-testid="start-select" value={p.start} disabled={!lmo} onChange={(e) => p.onStart(e.target.value)}>
            {starts.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <select data-testid="window-select" value={p.window} onChange={(e) => p.onWindow(e.target.value as Window)} aria-label="energy window">
          <option value="frontier">frontier window</option>
          <option value="all">all levels</option>
        </select>
        {homo && <span className="muted wb-small">{frontierLabel("homo")} {fmt(homo.energy_ev, 2)} eV</span>}
      </div>
      <Loading res={p.canonical} what="canonical orbitals" />
      {p.canonical.data && (
        <LevelDiagram canonical={p.canonical.data.orbitals} lmos={lmo ? (data?.set === p.set ? data.orbitals : null) : null} selected={p.selected} window={p.window} onPickCanonical={p.onPickCanonical} onPickLmo={p.onPick} />
      )}
      {lmo && <Loading res={p.shown} what={`${SET_LABELS[p.set] ?? p.set} orbitals`} />}
      {data && <SetStats data={data} />}
      {data && <OrbitalList data={data} selected={p.selected} onPick={p.onPick} />}
    </section>
  );
}
