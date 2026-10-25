// CandidateTable: the (donor, donor site, acceptor, acceptor site) tuples of a pool ranked by a
// predictor (GET /pool/{state}/candidates, design §11.8). Model-input descriptors are plain text;
// the classifier's class names (lone pair, σ*, ...) are greyed with a "diagnostic" tooltip:
// they are display only, never model input. "show excluded" adds the filtered tuples with their
// reason codes. Pages of 200 tuples; "more" loads the next one.
//
// Props:
//   stateId, predictor, temperature, scope   the query
//   donor            number | null            donor filter (token id)
//   onDonorChange    (donor | null) => void
//   showExcluded     boolean
//   onShowExcludedChange (on) => void
//   activeTuple      [d, ds, a, as] | null    the previewed / selected row (highlighted)
//   onPreview        (c) => void              shows candidate-preview-btn
//   onSelect         (c) => void              shows candidate-select-btn (Correct page)
//   selectLabel      string                   default "use"
//
// DOM: candidates-panel > candidate-donor-filter, candidate-show-excluded, candidate-row[data-*].
import { useState } from "react";
import { getCandidates, type CandidatesQuery } from "../../api/client";
import type { CandidateOut, CandidatesOut, OrbitalTokenOut } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";
import { PendingNote } from "../../components/PendingNote";
import { ProvenanceBadge } from "../../components/ProvenanceBadge";
import { centreText, descText, exact, fmt, fmtProb, fmtSig } from "../../components/format";
import { usePendingCached } from "../../api/hooks";
import { usePoolTokens } from "./hooks";

export const CANDIDATE_PAGE = 200;

export interface CandidateTableProps {
  stateId: string;
  predictor: string;
  temperature: number | null;
  scope: "all" | "inter";
  donor: number | null;
  onDonorChange: (donor: number | null) => void;
  showExcluded: boolean;
  onShowExcludedChange: (on: boolean) => void;
  activeTuple?: readonly [number, number, number, number] | null;
  onPreview?: (c: CandidateOut) => void;
  onSelect?: (c: CandidateOut) => void;
  selectLabel?: string;
}

type Query = Omit<CandidatesQuery, "offset" | "limit"> & { stateId: string };

function queryKey(q: Query, offset: number): string {
  const t = q.temperature ?? "";
  return `/pool/${q.stateId}/candidates?predictor=${q.predictor}&donor=${q.donor ?? ""}&excl=${q.includeExcluded ? 1 : 0}&T=${t}&scope=${q.scope}&offset=${offset}&limit=${CANDIDATE_PAGE}`;
}

function useCandidatePage(q: Query, offset: number) {
  return usePendingCached<CandidatesOut>(queryKey(q, offset), (o) =>
    getCandidates(q.stateId, { ...q, offset, limit: CANDIDATE_PAGE }, o),
  );
}

export const DIAGNOSTIC_TITLE = "diagnostic: the classifier's name, display only; never a model input";

function Diagnostic({ name }: { name: string | undefined }) {
  if (!name) return null;
  return (
    <span className="diagnostic" title={DIAGNOSTIC_TITLE} data-diagnostic="true">
      {name}
    </span>
  );
}

const sameTuple = (c: CandidateOut, t: readonly number[] | null | undefined) =>
  !!t && c.donor === t[0] && c.donor_site === t[1] && c.acceptor === t[2] && c.acceptor_site === t[3];

function rowAttrs(c: CandidateOut): Record<string, string> {
  return {
    "data-rank": c.rank === null ? "" : String(c.rank),
    "data-donor": String(c.donor),
    "data-donor-site": String(c.donor_site),
    "data-acceptor": String(c.acceptor),
    "data-acceptor-site": String(c.acceptor_site),
    "data-donor-site-map": String(c.donor_site_map),
    "data-acceptor-site-map": String(c.acceptor_site_map),
    "data-donor-species": c.donor_species,
    "data-acceptor-species": c.acceptor_species,
    "data-donor-site-element": c.donor_site_element,
    "data-acceptor-site-element": c.acceptor_site_element,
    "data-acceptor-centre-elements": c.acceptor_centre_elements.join(","),
    "data-valid": c.valid ? "true" : "false",
    "data-reason": c.reason ?? "",
    "data-contact": c.is_contact_site_pair ? "true" : "false",
  };
}

interface RowProps {
  c: CandidateOut;
  active: boolean;
  onPreview?: (c: CandidateOut) => void;
  onSelect?: (c: CandidateOut) => void;
  selectLabel: string;
}

function CandidateRow({ c, active, onPreview, onSelect, selectLabel }: RowProps) {
  return (
    <tr data-testid="candidate-row" className={`candidate-row${c.valid ? "" : " invalid"}${active ? " active" : ""}`} {...rowAttrs(c)}>
      <td className="num">{c.rank ?? "–"}</td>
      <td>
        <div>
          <strong>#{c.donor}</strong> at {c.donor_site_element}
          {c.donor_site_map} <span className="muted">in {c.donor_species}</span> <Diagnostic name={c.diagnostic.donor_class} />
        </div>
        <div className="desc">{descText(c.donor_desc)}</div>
      </td>
      <td>
        <div>
          <strong>#{c.acceptor}</strong> at {c.acceptor_site_element}
          {c.acceptor_site_map} <span className="muted">in {c.acceptor_species}</span> <Diagnostic name={c.diagnostic.acceptor_class} />
          <span className="muted"> ({c.acceptor_centre_elements.join("–")})</span>
        </div>
        <div className="desc">{descText(c.acceptor_desc)}</div>
      </td>
      <td className="num" title={`score ${exact(c.score)}; probability ${exact(c.probability)}`}>
        {fmtSig(c.score)}
        <div className="muted">p {fmtProb(c.probability)}</div>
      </td>
      <td className="num" title={`KS3 ${exact(c.ks3)} eV; KS2 ${exact(c.ks2)} eV; KS1 ${exact(c.ks1)} eV`}>
        {fmt(c.ks3, 4)}
        <div className="muted">
          {fmt(c.ks2, 3)} / {fmt(c.ks1, 3)}
        </div>
      </td>
      <td>
        {c.is_contact_site_pair ? <span className="badge badge-ok">contact</span> : <span className="badge" title={`contact sites ${c.contact_sites.join(", ")}`}>off-contact</span>}
        {c.clash && <span className="badge badge-error">CLASH</span>}
        {!c.valid && <span className="badge badge-warn">{c.reason ?? "excluded"}</span>}
      </td>
      <td className="row-actions">
        {onPreview && (
          <button type="button" className="btn btn-small" data-testid="candidate-preview-btn" onClick={() => onPreview(c)}>
            preview
          </button>
        )}
        {onSelect && (
          <button type="button" className="btn btn-small btn-primary" data-testid="candidate-select-btn" onClick={() => onSelect(c)}>
            {selectLabel}
          </button>
        )}
      </td>
    </tr>
  );
}

function PageRows({ q, offset, rowProps }: { q: Query; offset: number; rowProps: Omit<RowProps, "c" | "active"> & { activeTuple?: readonly number[] | null } }) {
  const page = useCandidatePage(q, offset);
  const { activeTuple, ...handlers } = rowProps;
  if (page.error) {
    return (
      <tr>
        <td colSpan={7}>
          <ErrorBanner error={page.error} title={`Cannot load tuples ${offset + 1}…`} />
        </td>
      </tr>
    );
  }
  if (!page.data) {
    return (
      <tr>
        <td colSpan={7} className="muted">
          loading tuples {offset + 1}… <PendingNote label={page.pending?.label} worker={page.pending?.worker} />
        </td>
      </tr>
    );
  }
  return (
    <>
      {page.data.candidates.map((c) => (
        <CandidateRow key={`${c.donor}-${c.donor_site}-${c.acceptor}-${c.acceptor_site}`} c={c} active={sameTuple(c, activeTuple)} {...handlers} />
      ))}
    </>
  );
}

function donorLabel(o: OrbitalTokenOut): string {
  const centres = o.bookkeeping.centres.slice(0, 2).map(centreText).join(", ");
  return `#${o.token_id} occ ${fmt(o.model_input.occupation, 2)} · ${centres}`;
}

function DonorFilter({ props, first }: { props: CandidateTableProps; first: CandidatesOut | undefined }) {
  const tokens = usePoolTokens(props.stateId);
  const fromTokens = tokens.data?.orbitals.filter((o) => o.kind === "occupied" && o.masks.donor_ok) ?? [];
  const fromRows = [...new Set(first?.candidates.map((c) => c.donor) ?? [])].sort((a, b) => a - b);
  return (
    <label className="field">
      <span className="field-label">donor</span>
      <select
        data-testid="candidate-donor-filter"
        value={props.donor ?? ""}
        onChange={(e) => props.onDonorChange(e.target.value === "" ? null : Number(e.target.value))}
      >
        <option value="">all donors</option>
        {fromTokens.length
          ? fromTokens.map((o) => (
              <option key={o.token_id} value={o.token_id}>
                {donorLabel(o)}
              </option>
            ))
          : fromRows.map((d) => (
              <option key={d} value={d}>
                #{d}
              </option>
            ))}
        {props.donor !== null && !fromTokens.some((o) => o.token_id === props.donor) && !fromRows.includes(props.donor) && (
          <option value={props.donor}>#{props.donor}</option>
        )}
      </select>
    </label>
  );
}

function Summary({ first }: { first: CandidatesOut }) {
  const excluded = Object.entries(first.excluded_counts).filter(([, n]) => n > 0);
  return (
    <div className="row muted candidates-summary">
      <span>
        {first.total} tuples shown of {first.n_tuples}; {first.n_valid_pairs} valid token pairs
      </span>
      <span>
        {first.predictor}, T {fmt(first.temperature, 3)}, scope {first.scope}
      </span>
      {excluded.length > 0 && <span>excluded: {excluded.map(([k, n]) => `${k} ${n}`).join(", ")}</span>}
      <ProvenanceBadge provenance={first.provenance} compact />
    </div>
  );
}

export function CandidateTable(props: CandidateTableProps) {
  const q: Query = {
    stateId: props.stateId,
    predictor: props.predictor,
    donor: props.donor,
    includeExcluded: props.showExcluded,
    temperature: props.temperature,
    scope: props.scope,
  };
  const qk = queryKey(q, 0);
  const [pages, setPages] = useState<{ key: string; n: number }>({ key: qk, n: 1 });
  const nPages = pages.key === qk ? pages.n : 1;
  const first = useCandidatePage(q, 0);
  const total = first.data?.total ?? 0;
  const rowProps = { onPreview: props.onPreview, onSelect: props.onSelect, selectLabel: props.selectLabel ?? "use", activeTuple: props.activeTuple };
  return (
    <section className="candidates-panel" data-testid="candidates-panel" data-state-id={props.stateId} data-total={total}>
      <div className="row candidates-controls">
        <DonorFilter props={props} first={first.data} />
        <label className="field-inline">
          <input type="checkbox" data-testid="candidate-show-excluded" checked={props.showExcluded} onChange={(e) => props.onShowExcludedChange(e.target.checked)} />
          show excluded (with reasons)
        </label>
        {first.data && <Summary first={first.data} />}
      </div>
      <div className="table-scroll">
        <table className="table candidates-table">
          <thead>
            <tr>
              <th>rank</th>
              <th>donor (model input)</th>
              <th>acceptor (model input)</th>
              <th>score</th>
              <th title="KS3, then KS2 / KS1 (eV)">KS3 (KS2 / KS1)</th>
              <th>sites</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: nPages }, (_, k) => (
              <PageRows key={`${qk}#${k}`} q={q} offset={k * CANDIDATE_PAGE} rowProps={rowProps} />
            ))}
          </tbody>
        </table>
      </div>
      {total > nPages * CANDIDATE_PAGE && (
        <button type="button" className="btn btn-small" data-testid="candidate-more" onClick={() => setPages({ key: qk, n: nPages + 1 })}>
          show {Math.min(CANDIDATE_PAGE, total - nPages * CANDIDATE_PAGE)} more of {total - nPages * CANDIDATE_PAGE}
        </button>
      )}
    </section>
  );
}
