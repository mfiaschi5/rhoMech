// ResultsTable (design §10, §11.8): one page of SearchOut, ordered by the server (source, then
// step id or (mechanism, step)). A row opens the step view (#/browse/step/<ref>, the search kept
// in the hash). Substructure hits say which recorded form of a species matched.
// DOM: div[data-testid=browse-results][data-total][data-offset][data-truncated] holding
// tr[data-testid=browse-result-{ref}][data-source][data-status].
import { useMemo } from "react";
import type { SearchOut, StepSummaryOut } from "../../api/types.gen";
import { useRDKit } from "../../chem/rdkit";
import { navigate, type Route } from "../../router";
import { browseHash, PAGE_SIZE, type SearchState } from "./searchParams";
import { speciesCounts, speciesText } from "./smiles";

export interface ResultsTableProps {
  route: Route;
  search: SearchState;
  out: SearchOut;
  selectedRef: string | null;
}

export function stepLabel(r: StepSummaryOut): string {
  if (r.source === "user_curated") return `mechanism #${r.mechanism_id ?? "?"} step ${(r.step_index ?? 0) + 1}`;
  return r.step_id ?? r.ref;
}

export function SplitBadges({ splits }: { splits: Record<string, string> | null }) {
  if (!splits) return null;
  return (
    <>
      {Object.entries(splits).map(([name, value]) => (
        <span key={name} className="badge br-split" data-testid={`browse-split-${name}`} data-split={value} title={`${name} split`}>
          {name}: {value}
        </span>
      ))}
    </>
  );
}

function Row({ r, route, search, selected }: { r: StepSummaryOut; route: Route; search: SearchState; selected: boolean }) {
  const { rdkit } = useRDKit();
  const sides = useMemo(
    () => [speciesText(speciesCounts(rdkit, r.reactants_smiles)), speciesText(speciesCounts(rdkit, r.products_smiles))],
    [rdkit, r.reactants_smiles, r.products_smiles],
  );
  const href = browseHash(route, search, r.ref);
  return (
    <tr
      data-testid={`browse-result-${r.ref}`}
      data-ref={r.ref}
      data-source={r.source}
      data-status={r.status}
      className={`br-click-row${selected ? " br-row-current" : ""}`}
      onClick={() => navigate(href)}
    >
      <td>
        <a href={href} onClick={(e) => e.stopPropagation()}>
          {stepLabel(r)}
        </a>
      </td>
      <td>
        <span className="badge">{r.source === "pmechdb" ? "PMechDB" : "curated"}</span>{" "}
        <span className={r.status === "rejected" ? "badge badge-error" : r.status === "dataset" ? "badge" : "badge badge-ok"}>
          {r.status}
        </span>
      </td>
      <td className="br-smiles" title={`${r.reactants_smiles}>>${r.products_smiles}`}>
        {sides[0]} → {sides[1]}
      </td>
      <td className="num">{r.n_arrows}</td>
      <td>{r.pmechdb_code ? <code>{r.pmechdb_code}</code> : "–"}</td>
      <td>
        {r.resonance_step && <span className="badge badge-warn">resonance</span>} <SplitBadges splits={r.splits} />
      </td>
      <td className="br-smiles">{r.matched_forms.length ? `via ${r.matched_forms.join(", ")}` : ""}</td>
    </tr>
  );
}

function Pager({ total, offset, shown, truncated, onPage }: {
  total: number;
  offset: number;
  shown: number;
  truncated: boolean;
  onPage: (offset: number) => void;
}) {
  const last = offset + shown;
  return (
    <div className="row br-pager">
      <strong>
        {total} step{total === 1 ? "" : "s"}
      </strong>
      <span className="muted">
        {total ? offset + 1 : 0}–{last}
      </span>
      <button type="button" className="btn btn-small" data-testid="browse-page-prev" disabled={offset === 0} onClick={() => onPage(offset - PAGE_SIZE)}>
        previous
      </button>
      <button type="button" className="btn btn-small" data-testid="browse-page-next" disabled={last >= total} onClick={() => onPage(offset + PAGE_SIZE)}>
        next
      </button>
      {truncated && (
        <span className="badge badge-warn" title="the substructure verification cap (20,000 matches) was reached">
          truncated
        </span>
      )}
    </div>
  );
}

export function ResultsTable({ route, search, out, selectedRef }: ResultsTableProps) {
  const page = (offset: number) => navigate(browseHash(route, { ...search, offset: Math.max(0, offset) }, selectedRef));
  return (
    <div
      className="br-results"
      data-testid="browse-results"
      data-total={out.total}
      data-offset={search.offset}
      data-truncated={out.truncated ? "true" : "false"}
    >
      <Pager total={out.total} offset={search.offset} shown={out.results.length} truncated={out.truncated} onPage={page} />
      {out.results.length === 0 ? (
        <p className="muted">No step matches.</p>
      ) : (
        <div className="br-table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>step</th>
                <th>source / status</th>
                <th>reactants → products</th>
                <th className="num">arrows</th>
                <th>PMechDB code</th>
                <th>flags</th>
                <th>matched</th>
              </tr>
            </thead>
            <tbody>
              {out.results.map((r) => (
                <Row key={r.ref} r={r} route={route} search={search} selected={r.ref === selectedRef} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
