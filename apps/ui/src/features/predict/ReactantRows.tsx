// ReactantRows: the reactant inputs of the Predict page (design §11.8): SMILES, count, a live
// depiction and the /normalize preflight badges of each row, plus "add reactant".
//
// Props:
//   rows       ReactantRow[]              the rows (SMILES and count)
//   onChange   (rows) => void
//   preflight  Preflight                  from usePreflight(rows)
//   maxRows    number                     meta.limits.max_species (default 6)
//   maxCount   number                     meta.limits.max_count (default 4)
//
// Each row has a "Draw" button that opens the workbench sketcher (DrawModal) and takes its SMILES.
//
// DOM per row i (prefix: the prop, default "predict"): <prefix>-smiles-{i}, <prefix>-count-{i},
// <prefix>-draw-{i}, <prefix>-remove-{i}, <prefix>-depiction-{i} and
// <prefix>-preflight-{i}[data-ok="true|false|pending|empty|unchecked"] (unchecked: another field
// failed the whole request); <prefix>-add-reactant; the DrawModal's <prefix>-draw-* ids.
import { useEffect, useState } from "react";
import { ApiError, isAbort, normalize, toApiError } from "../../api/client";
import type { NormalizeOut, SpeciesPreflightOut } from "../../api/types.gen";
import { useDebounced } from "../../api/hooks";
import { DrawModal } from "./DrawModal";
import { NumberField } from "./NumberField";
import { SmilesInput } from "./SmilesInput";

export interface ReactantRow {
  smiles: string;
  count: number;
}

export type RowState = "empty" | "pending" | "ok" | "error" | "unchecked";

export interface RowPreflight {
  state: RowState;
  species: SpeciesPreflightOut[];
  message: string | null;
}

export interface Preflight {
  rows: RowPreflight[];
  result: NormalizeOut | null;
  error: ApiError | null;
  /** Every filled row is ok and the current rows were checked. */
  ok: boolean;
}

export const PREFLIGHT_DEBOUNCE_MS = 250;

const rowsKey = (rows: readonly ReactantRow[]) => JSON.stringify(rows.map((r) => [r.smiles.trim(), r.count]));

interface Checked {
  key: string;
  result: NormalizeOut | null;
  error: ApiError | null;
  /** Input row index of each field sent to /normalize. */
  fields: number[];
}

/** /normalize preflight of the rows, 250 ms after the last change; stale answers are dropped. */
export function usePreflight(rows: readonly ReactantRow[]): Preflight {
  const settled = useDebounced(rows, PREFLIGHT_DEBOUNCE_MS);
  const settledKey = rowsKey(settled);
  const [checked, setChecked] = useState<Checked | null>(null);
  useEffect(() => {
    const filled = settled.map((r, i) => ({ smiles: r.smiles.trim(), count: r.count, i })).filter((r) => r.smiles);
    if (!filled.length) {
      setChecked({ key: settledKey, result: null, error: null, fields: [] });
      return;
    }
    const ctl = new AbortController();
    const body = { reactants_smiles: filled.map((r) => r.smiles), stoichiometry: filled.map((r) => r.count) };
    const fields = filled.map((r) => r.i);
    normalize(body, { signal: ctl.signal }).then(
      (result) => setChecked({ key: settledKey, result, error: null, fields }),
      (err: unknown) => {
        if (!isAbort(err)) setChecked({ key: settledKey, result: null, error: toApiError(err), fields });
      },
    );
    return () => ctl.abort();
    // settledKey stands for `settled`
  }, [settledKey]);
  return summarize(rows, checked);
}

/** A 422 of the whole request: the field it names (detail.field) is the failing row; the
 *  others were not checked. Without a field, every row carries the error. */
function requestError(checked: Checked, row: number): RowPreflight {
  const err = checked.error!;
  const field = typeof err.detail?.field === "number" ? err.detail.field : null;
  const message = `${err.code}: ${err.message}`;
  if (field === null || checked.fields[field] === row) return { state: "error", species: [], message };
  return { state: "unchecked", species: [], message: `not checked: field ${field + 1} failed` };
}

function summarize(rows: readonly ReactantRow[], checked: Checked | null): Preflight {
  const current = checked !== null && checked.key === rowsKey(rows);
  const out = rows.map((r, i): RowPreflight => {
    if (!r.smiles.trim()) return { state: "empty", species: [], message: null };
    if (!current || !checked) return { state: "pending", species: [], message: null };
    if (checked.error) return requestError(checked, i);
    const field = checked.fields.indexOf(i);
    const species = checked.result?.species.filter((s) => s.field === field) ?? [];
    const bad = species.length === 0 || species.some((s) => s.errors.length > 0);
    return { state: bad ? "error" : "ok", species, message: species.length ? null : "no species parsed" };
  });
  const filled = out.filter((r) => r.state !== "empty");
  const ok = current && filled.length > 0 && filled.every((r) => r.state === "ok") && !!checked?.result?.ok;
  return { rows: out, result: current ? (checked?.result ?? null) : null, error: current ? (checked?.error ?? null) : null, ok };
}

function SpeciesBadges({ s }: { s: SpeciesPreflightOut }) {
  const charge = s.charge === null ? "?" : s.charge > 0 ? `+${s.charge}` : String(s.charge);
  return (
    <div className="preflight-species">
      <code className="code">{s.token_key ?? s.input}</code>
      {s.count > 1 && <span className="badge">×{s.count}</span>}
      <span className="badge">charge {charge}</span>
      {s.multiplicity !== null && <span className="badge">mult {s.multiplicity}</span>}
      {s.n_atoms !== null && <span className="badge">{s.n_atoms} atoms</span>}
      <span className={`badge ${s.tokens_cached ? "badge-ok" : s.qm_cached ? "" : "badge-warn"}`}>
        {s.tokens_cached ? "tokens cached" : s.qm_cached ? "QM cached" : "needs QM"}
      </span>
      {s.errors.map((e, k) => (
        <span key={k} className="badge badge-error" title={e.message}>
          {e.code}: {e.message}
        </span>
      ))}
    </div>
  );
}

function PreflightBadge({ index, row, prefix }: { index: number; row: RowPreflight; prefix: string }) {
  return (
    <div className={`preflight preflight-${row.state}`} data-testid={`${prefix}-preflight-${index}`} data-ok={row.state === "ok" ? "true" : row.state === "error" ? "false" : row.state}>
      {row.state === "empty" && <span className="muted">enter a SMILES</span>}
      {row.state === "pending" && <span className="muted">checking…</span>}
      {row.state === "ok" && <span className="badge badge-ok">ok</span>}
      {row.message && <span className={`badge ${row.state === "unchecked" ? "badge-warn" : "badge-error"}`}>{row.message}</span>}
      {row.species.map((s, k) => (
        <SpeciesBadges key={k} s={s} />
      ))}
    </div>
  );
}

export interface ReactantRowsProps {
  rows: ReactantRow[];
  onChange: (rows: ReactantRow[]) => void;
  preflight: Preflight;
  maxRows?: number;
  maxCount?: number;
  prefix?: string;
}

export function ReactantRows({ rows, onChange, preflight, maxRows = 6, maxCount = 4, prefix = "predict" }: ReactantRowsProps) {
  const set = (i: number, patch: Partial<ReactantRow>) => onChange(rows.map((r, k) => (k === i ? { ...r, ...patch } : r)));
  const remove = (i: number) => onChange(rows.filter((_, k) => k !== i));
  const [drawing, setDrawing] = useState<number | null>(null);
  return (
    <div className="reactant-rows stack">
      {drawing !== null && (
        <DrawModal
          prefix={prefix}
          initial={rows[drawing]?.smiles ?? ""}
          onClose={() => setDrawing(null)}
          onUse={(smiles) => {
            set(drawing, { smiles });
            setDrawing(null);
          }}
        />
      )}
      {rows.map((row, i) => (
        <div key={i} className="reactant-row row" data-row={i}>
          <SmilesInput index={i} prefix={prefix} value={row.smiles} onChange={(smiles) => set(i, { smiles })} />
          <div className="stack">
            <NumberField
              label="count"
              testId={`${prefix}-count-${i}`}
              value={row.count}
              min={1}
              max={maxCount}
              onChange={(count) => count !== null && set(i, { count })}
            />
            <button type="button" className="btn btn-small" data-testid={`${prefix}-draw-${i}`} onClick={() => setDrawing(i)} title="draw this reactant in the sketcher">
              Draw
            </button>
            {rows.length > 1 && (
              <button type="button" className="btn btn-small" onClick={() => remove(i)} data-testid={`${prefix}-remove-${i}`}>
                remove
              </button>
            )}
          </div>
          <PreflightBadge index={i} prefix={prefix} row={preflight.rows[i] ?? { state: "pending", species: [], message: null }} />
        </div>
      ))}
      <div className="row">
        <button
          type="button"
          className="btn"
          data-testid={`${prefix}-add-reactant`}
          disabled={rows.length >= maxRows}
          title={rows.length >= maxRows ? `at most ${maxRows} species` : undefined}
          onClick={() => onChange([...rows, { smiles: "", count: 1 }])}
        >
          + add reactant
        </button>
        {preflight.result?.pool_key && (
          <span className="muted">
            pool <code className="code">{preflight.result.pool_key}</code>
            {preflight.result.n_atoms !== null && `, ${preflight.result.n_atoms} atoms`}
          </span>
        )}
      </div>
    </div>
  );
}
