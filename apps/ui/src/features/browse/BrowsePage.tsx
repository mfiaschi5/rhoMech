// The Browse page (design §10, §11.8): search PMechDB and curated steps by exact species or
// substructure, filter by source, status and resonance, page through the results and view a step.
// Routes (all state in the hash, see searchParams.ts): #/browse?q=&mode=&side=&src=&st=&res=&off=
// and #/browse/step/<ref>?<search>. A search never starts the index job: a 409 INDEX_NOT_READY
// (pmechdb requested while its index is not ready) is shown in place with the way out (build the
// index, or search curated steps only); curated-only searches always work.
import { getBrowseStatus, searchSteps, type BrowseSource } from "../../api/client";
import type { IndexStatusOut, SearchOut } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";
import { PendingNote } from "../../components/PendingNote";
import type { PageProps } from "../../registry";
import { navigate, withParams, type Route } from "../../router";
import { IndexStatus } from "./IndexStatus";
import { ResultsTable } from "./ResultsTable";
import { SearchBar } from "./SearchBar";
import { browseHash, defaultSources, PAGE_SIZE, readSearch, searchKey, type SearchState } from "./searchParams";
import { StepView } from "./StepView";
import { useLoad, type LoadState } from "./useLoad";

function NotReady({ route, search, detail }: { route: Route; search: SearchState; detail: Record<string, unknown> | null }) {
  const state = typeof detail?.state === "string" ? detail.state : "not ready";
  const curatedOnly = () => navigate(browseHash(route, { ...search, sources: ["user_curated"], offset: 0 }));
  return (
    <div className="banner banner-warn" data-testid="browse-index-not-ready" data-code="INDEX_NOT_READY" data-state={state}>
      <strong>The PMechDB index is {state}.</strong> PMechDB steps cannot be searched until it is built (the button above
      starts it; it takes a few minutes on one worker).{" "}
      <button type="button" className="btn btn-small" data-testid="browse-curated-only" onClick={curatedOnly}>
        Search curated steps only
      </button>
    </div>
  );
}

function Results(props: { route: Route; search: SearchState; results: LoadState<SearchOut>; stepRef: string | null }) {
  const { route, search, results, stepRef } = props;
  if (!search.submitted) {
    return (
      <p className="muted" data-testid="browse-hint">
        Enter a SMILES (exact species) or a SMARTS (substructure) and search; an empty query lists every step of the chosen
        sources.
      </p>
    );
  }
  const err = results.error;
  if (search.sources && search.sources.length === 0) {
    return (
      <p className="banner banner-warn" data-testid="browse-no-source">
        No source chosen: tick PMechDB or curated and search again.
      </p>
    );
  }
  return (
    <div className="br-results-area">
      {results.pending && <PendingNote label={results.pending.label} worker={results.pending.worker} />}
      {results.loading && !results.data && <p className="muted">searching…</p>}
      {err?.code === "INDEX_NOT_READY" ? (
        <NotReady route={route} search={search} detail={err.detail} />
      ) : (
        <ErrorBanner error={err} title={err?.code === "QUERY_INVALID" ? "Query not understood" : "Search failed"} testId="browse-search-error" />
      )}
      {results.data && !err && <ResultsTable route={route} search={search} out={results.data} selectedRef={stepRef} />}
    </div>
  );
}

function useSearch(search: SearchState, status: LoadState<IndexStatusOut>) {
  const settled = !!status.data || !!status.error;
  const fallback = defaultSources(status.data?.pmechdb.state);
  const sources: BrowseSource[] | null = search.sources ?? (settled ? fallback : null);
  const key = search.submitted && sources?.length ? searchKey(search, sources) : null;
  const results = useLoad<SearchOut>(key, (o) =>
    searchSteps(
      {
        q: search.q,
        mode: search.mode,
        side: search.side,
        sources: sources ?? [],
        status: search.status || undefined,
        resonance: search.resonance === "" ? undefined : search.resonance === "true",
        limit: PAGE_SIZE,
        offset: search.offset,
      },
      o,
    ),
  );
  return { results, fallback };
}

export function BrowsePage({ route }: PageProps) {
  const search = readSearch(route.params);
  const stepRef = route.parts[0] === "step" && route.parts[1] ? route.parts[1] : null;
  const status = useLoad<IndexStatusOut>("/browse/status", (o) => getBrowseStatus(o));
  const pm = status.data?.pmechdb;
  const indexJob = route.params.get("index") ?? (pm?.state === "building" ? pm.job_id : null);
  const { results, fallback } = useSearch(search, status);
  const resultRow = results.data?.results.find((r) => r.ref === stepRef);
  const barKey = JSON.stringify(search);
  return (
    <div className="br-page stack" data-testid="browse-page">
      <section className="panel">
        <h2 className="br-title">Browse steps</h2>
        <SearchBar
          key={barKey}
          initial={search}
          defaultSources={fallback}
          indexState={pm?.state}
          onSubmit={(s) => navigate(browseHash(route, s, null))}
        />
        <IndexStatus
          status={status}
          jobId={indexJob}
          onStarted={(jobId) => {
            navigate(withParams(route, { index: jobId }), { replace: true });
            status.reload();
          }}
          onFinished={() => {
            status.reload();
            if (results.error?.code === "INDEX_NOT_READY") results.reload();
          }}
        />
      </section>
      {stepRef && <StepView key={stepRef} stepRef={stepRef} resultRow={resultRow} />}
      <section className="panel">
        <Results route={route} search={search} results={results} stepRef={stepRef} />
      </section>
    </div>
  );
}
