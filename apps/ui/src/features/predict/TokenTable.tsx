// TokenTable: every token of a pool (GET /pool/{state}/tokens): orbital tokens with their
// continuous model-input descriptors, and atom tokens with theirs. The classifier's labels
// (orbital type, hybridization, aromaticity, functional group) are shown greyed with a
// "diagnostic" tooltip: they are never model input (CLAUDE.md, "No chemist's taxonomy").
//
// Props:
//   stateId   string
//
// DOM: <div data-testid="token-table"> > table.orbital-tokens tr[data-token][data-kind], table.atom-tokens.
import type { AtomTokenOut, OrbitalTokenOut } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";
import { PendingNote } from "../../components/PendingNote";
import { DIAGNOSTIC_TITLE } from "./CandidateTable";
import { centreText, fmt } from "./format";
import { usePoolTokens } from "./hooks";

export interface TokenTableProps {
  stateId: string;
}

function OrbitalRow({ o }: { o: OrbitalTokenOut }) {
  const m = o.model_input;
  const masks = [o.masks.donor_ok && "donor", o.masks.acceptor_ok && "acceptor", o.masks.d_type && "d-type", o.masks.fractional && "fractional", o.masks.somo && "SOMO"].filter(Boolean);
  return (
    <tr data-token={o.token_id} data-kind={o.kind}>
      <td className="num">{o.token_id}</td>
      <td>{o.kind}</td>
      <td className="num">{fmt(m.occupation, 2)}</td>
      <td>{o.bookkeeping.centres.map(centreText).join(", ")}</td>
      <td className="num">{m.center_weights.map((w) => fmt(w, 2)).join(" ")}</td>
      <td className="num">{m.center_spd.slice(0, 2).map((spd) => spd.map((v) => fmt(v, 2)).join("/")).join("; ")}</td>
      <td className="num">
        {fmt(m.cos_a, 2)} / {fmt(m.cos_b, 2)}
      </td>
      <td className="num">
        {fmt(m.overlap_population, 3)} ({m.overlap_sign > 0 ? "+" : m.overlap_sign < 0 ? "-" : "0"})
      </td>
      <td className="num">{fmt(m.distance_ab, 2)}</td>
      <td className="num">{fmt(m.radius_gyration, 2)}</td>
      <td className="num">
        {m.spectral_set
          .slice(0, 3)
          .map(([e, w]) => `${fmt(e, 2)}:${fmt(w, 2)}`)
          .join(" ")}
      </td>
      <td className="muted">{masks.join(", ")}</td>
      <td>
        <span className="diagnostic" title={DIAGNOSTIC_TITLE}>
          {o.diagnostic.token_class || o.diagnostic.type}
        </span>
      </td>
    </tr>
  );
}

function AtomRow({ a }: { a: AtomTokenOut }) {
  const mi = a.model_input;
  const diag = Object.entries(a.diagnostic)
    .filter(([, v]) => v !== null && v !== "")
    .map(([k, v]) => `${k} ${typeof v === "number" ? fmt(v, 3) : String(v)}`)
    .join(", ");
  const val = (k: string) => {
    const v = mi[k];
    if (Array.isArray(v)) return v.map((x) => (x === null ? "–" : fmt(x, 2))).join(" ");
    return typeof v === "number" ? fmt(v, 3) : "–";
  };
  return (
    <tr data-token={a.token_id} data-map={a.map}>
      <td className="num">{a.token_id}</td>
      <td>
        {a.element}
        {a.map}
      </td>
      <td className="num">{val("formal_charge")}</td>
      <td className="num">{val("mulliken_charge")}</td>
      <td className="num">
        {val("f_plus_fmo")} / {val("f_minus_fmo")}
      </td>
      <td className="num">{val("dual")}</td>
      <td className="num">{val("fmo_weights")}</td>
      <td className="num">{val("vbur")}</td>
      <td>
        <span className="diagnostic" title={DIAGNOSTIC_TITLE}>
          {diag}
        </span>
      </td>
    </tr>
  );
}

const ORBITAL_COLUMNS = ["id", "block", "occ", "centres", "weights", "s/p/d (a; b)", "|cos| a/b", "overlap pop.", "d_ab Å", "r_gyr", "spectral (E:w)", "masks"];
const ATOM_COLUMNS = ["id", "atom", "formal q", "Mulliken q", "f+ / f- (FMO)", "dual", "frontier weights", "%Vbur"];

function Head({ columns }: { columns: string[] }) {
  return (
    <thead>
      <tr>
        {columns.map((c) => (
          <th key={c}>{c}</th>
        ))}
        <th className="diagnostic" title={DIAGNOSTIC_TITLE}>
          diagnostic
        </th>
      </tr>
    </thead>
  );
}

export function TokenTable({ stateId }: TokenTableProps) {
  const tokens = usePoolTokens(stateId);
  const t = tokens.data;
  const fragments = t?.fragments
    .map((f) => `${f.token_key}${f.fermi_smeared ? " (Fermi-smeared)" : ""}${f.pm_converged ? "" : " (PM not converged)"}${f.qm_warnings ? ` [${f.qm_warnings}]` : ""}`)
    .join("; ");
  return (
    <div className="token-table" data-testid="token-table" data-state-id={stateId}>
      {tokens.loading && <PendingNote label={tokens.pending?.label ?? "loading tokens"} worker={tokens.pending?.worker} />}
      <ErrorBanner error={tokens.error} title="Cannot load the tokens" />
      {t && (
        <>
          <div className="muted">
            {t.n_atoms} atom tokens, {t.n_orbitals} orbital tokens; {fragments}
          </div>
          <div className="table-scroll">
            <table className="table orbital-tokens">
              <Head columns={ORBITAL_COLUMNS} />
              <tbody>
                {t.orbitals.map((o) => (
                  <OrbitalRow key={o.token_id} o={o} />
                ))}
              </tbody>
            </table>
          </div>
          <div className="table-scroll">
            <table className="table atom-tokens">
              <Head columns={ATOM_COLUMNS} />
              <tbody>
                {t.atoms.map((a) => (
                  <AtomRow key={a.token_id} a={a} />
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
