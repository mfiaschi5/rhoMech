// ReadingTable: the orbital reading of a step (spec "Orbital interpretation"): per forming bond
// (i, j) its pooled gap factor G^atom, the dominant channel (donor token r -> acceptor token s,
// its term g_rs), the dominant canonical MO pair (k, m) with their energies, and the tokens'
// diagnostic labels (greyed: display only, never model input). "3D" opens both orbitals.
//
// Columns: G^atom; the channel r -> s with G(r, s) and its term of G^atom (1/eV); the dominant
// canonical pair as frontier labels (HOMO-1 -> LUMO) with k, m, energies and the pair's share.
// Props: readings, structure (atom labels), stateId (null: no 3D button).
// DOM: <table data-testid="<prefix>-readings"> > tr[data-testid=<prefix>-reading-<k>][data-i][data-j]
// [data-g-atom], button[data-testid=<prefix>-reading-3d-<k>]; the OrbitalModal.
import { useState } from "react";
import { exact, fmt, fmtSig, moLabel } from "../../components/format";
import { OrbitalModal } from "./OrbitalModal";
import type { PoolStructure } from "./poolStructure";
import type { OrbitalReadingOut } from "./v9types";

export interface ReadingTableProps {
  readings: readonly OrbitalReadingOut[];
  structure: PoolStructure | null;
  stateId: string | null;
  prefix: string;
}

function Diag({ text }: { text: string | null }) {
  return text ? (
    <span className="diagnostic" title="diagnostic label: display only, never a model input">
      {text}
    </span>
  ) : null;
}

export function ReadingTable({ readings, structure, stateId, prefix }: ReadingTableProps) {
  const [open, setOpen] = useState<number | null>(null);
  const label = (i: number) => structure?.label(i) ?? String(i);
  if (!readings.length) return <p className="muted" data-testid={`${prefix}-readings-empty`}>No forming bond to read (a step without a +1 entry).</p>;
  const r = open !== null ? readings[open] : null;
  return (
    <div className="pv-table-wrap">
      <table className="table" data-testid={`${prefix}-readings`}>
        <thead>
          <tr>
            <th>bond</th>
            <th className="num">G^atom</th>
            <th>dominant channel r → s</th>
            <th className="num">G(r, s)</th>
            <th className="num">term</th>
            <th>canonical pair</th>
            <th className="num">E_k / E_m (eV)</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {readings.map((x, k) => (
            <tr key={k} data-testid={`${prefix}-reading-${k}`} data-i={x.i} data-j={x.j} data-g-atom={exact(x.g_atom)}>
              <td>
                {label(x.i)}–{label(x.j)}
              </td>
              <td className="num" title={exact(x.g_atom)}>
                {fmtSig(x.g_atom, 4)}
              </td>
              <td>
                <code className="code">{x.donor_token}</code> <Diag text={x.donor_diag} /> → <code className="code">{x.acceptor_token}</code> <Diag text={x.acceptor_diag} />
              </td>
              <td className="num" title={exact(x.g_rs)}>
                {fmtSig(x.g_rs, 4)}
              </td>
              <td className="num" title={exact(x.term)} data-testid={`${prefix}-reading-term-${k}`}>
                {fmtSig(x.term, 4)}
              </td>
              <td data-testid={`${prefix}-reading-canonical-${k}`}>
                {x.canonical ? (
                  <span title={`k ${x.canonical.k}, m ${x.canonical.m} (${x.canonical.index_source}); term ${fmtSig(x.canonical.term, 4)}`}>
                    {moLabel(x.canonical.k_label)} → {moLabel(x.canonical.m_label)} <span className="muted">({x.canonical.k}, {x.canonical.m}{x.canonical.share != null ? `; ${fmt(100 * x.canonical.share, 0)} %` : ""})</span>
                  </span>
                ) : (
                  "–"
                )}
              </td>
              <td className="num">{x.canonical ? `${fmt(x.canonical.e_k_ev, 2)} / ${fmt(x.canonical.e_m_ev, 2)}` : "–"}</td>
              <td>
                {stateId && (
                  <button type="button" className="btn btn-small" data-testid={`${prefix}-reading-3d-${k}`} onClick={() => setOpen(k)} title="the donor and acceptor orbitals in 3D">
                    3D
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {r && stateId && <OrbitalModal stateId={stateId} donor={r.donor_token} acceptor={r.acceptor_token} title={`${label(r.i)}–${label(r.j)}: orbital ${r.donor_token} → ${r.acceptor_token}`} onClose={() => setOpen(null)} />}
    </div>
  );
}
