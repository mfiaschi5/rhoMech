// EntryTable: the entries of R as an editable table: each row's value is a select (bonds
// −1 / 0 / +1, lone pairs −2 / 0 / +2; 0 removes it), a checkbox selects it (Delete removes the
// selection), and a last row adds an entry by atom indices.
//
// Props: state, dispatch, structure (labels, atom count).
// DOM: <table data-testid="correct-entries"> > tr[data-testid="correct-entry-b-<i>-<j>" |
// "correct-entry-l-<i>"][data-delta][data-selected], select correct-entry-value-..., checkbox
// correct-entry-select-...; correct-delete-selected; the add row correct-add-kind, correct-add-i,
// correct-add-j, correct-add-value, correct-add.
import { useState, type Dispatch } from "react";
import type { EditorAction, EditorState } from "./editor";
import type { PoolStructure } from "@ksmech/viz/pool/poolStructure";
import { entriesOf, entryKey, type Entry } from "@ksmech/viz/pool/rmatrix";

const idOf = (e: Entry) => (e.kind === "bond" ? `b-${e.i}-${e.j}` : `l-${e.i}`);
const VALUES = { bond: [-1, 0, 1], lone: [-2, 0, 2] } as const;
const signed = (v: number) => (v > 0 ? `+${v}` : String(v));

function AddRow({ dispatch, n }: { dispatch: Dispatch<EditorAction>; n: number }) {
  const [kind, setKind] = useState<"bond" | "lone">("bond");
  const [i, setI] = useState("");
  const [j, setJ] = useState("");
  const [value, setValue] = useState(1);
  const ai = Number.parseInt(i, 10);
  const aj = Number.parseInt(j, 10);
  const inRange = (x: number) => Number.isInteger(x) && x >= 0 && x < n;
  const ok = inRange(ai) && (kind === "lone" || (inRange(aj) && aj !== ai));
  const add = () => {
    const entry: Entry = kind === "bond" ? { kind, i: ai, j: aj, delta: 0 } : { kind, i: ai, delta: 0 };
    dispatch({ type: "setEntry", entry, value });
  };
  return (
    <tr className="pv-add-row">
      <td>
        <select data-testid="correct-add-kind" value={kind} onChange={(e) => {
          const k = e.target.value === "lone" ? "lone" : "bond";
          setKind(k);
          setValue(k === "lone" ? -2 : 1);
        }}>
          <option value="bond">bond</option>
          <option value="lone">lone</option>
        </select>
      </td>
      <td>
        <input className="pv-atom-input" data-testid="correct-add-i" inputMode="numeric" placeholder="i" value={i} onChange={(e) => setI(e.target.value)} />
        {kind === "bond" && <input className="pv-atom-input" data-testid="correct-add-j" inputMode="numeric" placeholder="j" value={j} onChange={(e) => setJ(e.target.value)} />}
      </td>
      <td>
        <select data-testid="correct-add-value" value={value} onChange={(e) => setValue(Number(e.target.value))}>
          {VALUES[kind].filter((v) => v !== 0).map((v) => (
            <option key={v} value={v}>
              {signed(v)}
            </option>
          ))}
        </select>
      </td>
      <td>
        <button type="button" className="btn btn-small" data-testid="correct-add" disabled={!ok} onClick={add}>
          add
        </button>
      </td>
    </tr>
  );
}

export function EntryTable({ state, dispatch, structure }: { state: EditorState; dispatch: Dispatch<EditorAction>; structure: PoolStructure }) {
  const entries = entriesOf(state.matrix);
  const label = (i: number) => structure.label(i);
  return (
    <div className="pv-table-wrap">
      <table className="table pv-entries" data-testid="correct-entries" data-count={entries.length}>
        <thead>
          <tr>
            <th>entry</th>
            <th>atoms</th>
            <th>Δ</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => {
            const key = entryKey(e);
            const id = idOf(e);
            const selected = state.selection.has(key);
            return (
              <tr key={key} data-testid={`correct-entry-${id}`} data-delta={e.delta} data-selected={selected ? "true" : "false"} className={selected ? "pv-row-selected" : undefined}>
                <td>{e.kind === "bond" ? "bond" : "lone"}</td>
                <td>{e.kind === "bond" ? `${label(e.i)}–${label(e.j)}` : label(e.i)}</td>
                <td>
                  <select data-testid={`correct-entry-value-${id}`} value={e.delta} onChange={(ev) => dispatch({ type: "setEntry", entry: e, value: Number(ev.target.value) })}>
                    {[...new Set([...VALUES[e.kind], e.delta])].map((v) => (
                      <option key={v} value={v}>
                        {signed(v)}
                      </option>
                    ))}
                  </select>
                </td>
                <td>
                  <input type="checkbox" data-testid={`correct-entry-select-${id}`} checked={selected} aria-label={`select ${key}`} onChange={() => dispatch({ type: "select", keys: [key], additive: true })} />
                </td>
              </tr>
            );
          })}
          <AddRow dispatch={dispatch} n={structure.n} />
        </tbody>
      </table>
      <button type="button" className="btn btn-small" data-testid="correct-delete-selected" disabled={state.selection.size === 0} onClick={() => dispatch({ type: "deleteSelection" })} title="remove the selected entries (Delete)">
        Delete selected
      </button>
    </div>
  );
}
