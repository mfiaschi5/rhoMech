// SearchBar (design §10, §11.8): query, mode (exact | substructure), side, the two source
// checkboxes, status and resonance filters. The form is local until submitted; submitting writes
// the search into the hash (BrowsePage reads it from there), so every search is deep-linkable.
// The pmechdb checkbox follows the default (checked only when the index is ready) until the user
// changes it. Test ids: browse-query, browse-mode, browse-side, browse-source-{pmechdb|user_curated},
// browse-status, browse-resonance, browse-submit.
import { useState, type FormEvent } from "react";
import type { BrowseSource } from "../../api/browse";
import { SIDES, SOURCES, STEP_STATUSES, type Mode, type Resonance, type SearchState, type Side } from "./searchParams";

export interface SearchBarProps {
  initial: SearchState;
  defaultSources: readonly BrowseSource[];
  indexState: string | undefined;
  onSubmit: (s: SearchState) => void;
}

const SOURCE_LABEL: Record<BrowseSource, string> = { pmechdb: "PMechDB", user_curated: "curated" };

function SourceChecks(props: {
  effective: readonly BrowseSource[];
  indexState: string | undefined;
  onToggle: (s: BrowseSource, on: boolean) => void;
}) {
  const { effective, indexState, onToggle } = props;
  const notReady = indexState !== undefined && indexState !== "ready";
  return (
    <>
      {SOURCES.map((s) => (
        <label key={s} className="br-check" title={s === "pmechdb" && notReady ? `the PMechDB index is ${indexState}` : undefined}>
          <input
            type="checkbox"
            data-testid={`browse-source-${s}`}
            checked={effective.includes(s)}
            onChange={(e) => onToggle(s, e.target.checked)}
          />{" "}
          {SOURCE_LABEL[s]}
          {s === "pmechdb" && notReady ? <span className="muted"> (index {indexState})</span> : null}
        </label>
      ))}
    </>
  );
}

function FilterSelects(props: {
  status: string;
  resonance: Resonance;
  onStatus: (v: string) => void;
  onResonance: (v: Resonance) => void;
}) {
  return (
    <>
      <select data-testid="browse-status" value={props.status} onChange={(e) => props.onStatus(e.target.value)} aria-label="status">
        <option value="">any status</option>
        {STEP_STATUSES.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>
      <select
        data-testid="browse-resonance"
        value={props.resonance}
        onChange={(e) => props.onResonance(e.target.value as Resonance)}
        aria-label="resonance steps"
      >
        <option value="">with resonance steps</option>
        <option value="false">without resonance steps</option>
        <option value="true">resonance steps only</option>
      </select>
    </>
  );
}

export function SearchBar({ initial, defaultSources, indexState, onSubmit }: SearchBarProps) {
  const [q, setQ] = useState(initial.q);
  const [mode, setMode] = useState<Mode>(initial.mode);
  const [side, setSide] = useState<Side>(initial.side);
  const [sources, setSources] = useState<BrowseSource[] | null>(initial.sources);
  const [status, setStatus] = useState(initial.status);
  const [resonance, setResonance] = useState<Resonance>(initial.resonance);
  const effective = sources ?? [...defaultSources];
  const toggle = (s: BrowseSource, on: boolean) =>
    setSources(SOURCES.filter((x) => (x === s ? on : effective.includes(x))));
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!effective.length) return;
    // untouched checkboxes stay "default" (src absent), so they keep following the index state
    onSubmit({ submitted: true, q: q.trim(), mode, side, sources, status, resonance, offset: 0 });
  };
  return (
    <form className="br-search" onSubmit={submit} role="search">
      <input
        data-testid="browse-query"
        className="br-query"
        value={q}
        placeholder={mode === "exact" ? "SMILES, e.g. COC (empty: every step)" : "SMARTS or SMILES, e.g. C=O"}
        onChange={(e) => setQ(e.target.value)}
        aria-label="query"
      />
      <select data-testid="browse-mode" value={mode} onChange={(e) => setMode(e.target.value as Mode)} aria-label="mode">
        <option value="exact">exact species</option>
        <option value="substructure">substructure</option>
      </select>
      <select data-testid="browse-side" value={side} onChange={(e) => setSide(e.target.value as Side)} aria-label="side">
        {SIDES.map((s) => (
          <option key={s} value={s}>
            {s === "either" ? "either side" : s}
          </option>
        ))}
      </select>
      <SourceChecks effective={effective} indexState={indexState} onToggle={toggle} />
      <FilterSelects status={status} resonance={resonance} onStatus={setStatus} onResonance={setResonance} />
      <button type="submit" className="btn btn-primary" data-testid="browse-submit" disabled={!effective.length}>
        Search
      </button>
      {!effective.length && <span className="muted">choose at least one source</span>}
    </form>
  );
}
