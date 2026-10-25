// SortableTable: a plain table sorted by clicking a column header (again: reversed; missing
// values last). Diagnostic columns are greyed and say "never model input" on hover.
//
// DOM: <table data-testid={testId} data-sort="<key>:<asc|desc>">, headers
// <th data-testid="sort-<key>" data-diagnostic>, rows <tr data-testid={rowTestId} ...rowAttrs>.
import { useMemo, useState, type MouseEvent, type ReactNode } from "react";
import { fmt } from "../../../components/format";

export type Cell = number | string | boolean | null | undefined;

export interface Column<R> {
  key: string;
  label: string;
  title?: string;
  value(r: R): Cell;
  render?(r: R): ReactNode;
  digits?: number;
  diagnostic?: boolean;
}

export interface SortableTableProps<R> {
  testId: string;
  rowTestId: string;
  rows: readonly R[];
  columns: readonly Column<R>[];
  rowKey(r: R): string | number;
  rowAttrs?(r: R): Record<string, string>;
  isSelected?(r: R): boolean;
  onRowClick?(r: R, e: MouseEvent): void;
  onRowHover?(r: R | null): void;
}

export const DIAGNOSTIC_TITLE = "diagnostic: never model input";

function compare(a: Cell, b: Cell): number {
  const na = a === null || a === undefined || (typeof a === "number" && !Number.isFinite(a));
  const nb = b === null || b === undefined || (typeof b === "number" && !Number.isFinite(b));
  if (na || nb) return na === nb ? 0 : na ? 1 : -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

function cellText<R>(c: Column<R>, r: R): ReactNode {
  if (c.render) return c.render(r);
  const v = c.value(r);
  if (v === null || v === undefined) return "–";
  if (typeof v === "number") return c.digits === undefined ? String(v) : fmt(v, c.digits);
  if (typeof v === "boolean") return v ? "yes" : "no";
  return v;
}

export function SortableTable<R>(p: SortableTableProps<R>) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);
  const rows = useMemo(() => {
    const col = sort && p.columns.find((c) => c.key === sort.key);
    if (!col || !sort) return p.rows;
    return [...p.rows].sort((a, b) => {
      const va = col.value(a);
      const vb = col.value(b);
      const missing = va === null || va === undefined || vb === null || vb === undefined;
      return missing ? compare(va, vb) : sort.dir * compare(va, vb);
    });
  }, [p.rows, p.columns, sort]);
  const onSort = (key: string) => setSort((s) => (s?.key === key ? { key, dir: s.dir === 1 ? -1 : 1 } : { key, dir: 1 }));
  return (
    <div className="wb-table-wrap">
      <table className="table wb-table" data-testid={p.testId} data-sort={sort ? `${sort.key}:${sort.dir === 1 ? "asc" : "desc"}` : ""}>
        <thead>
          <tr>
            {p.columns.map((c) => (
              <th
                key={c.key}
                data-testid={`sort-${c.key}`}
                data-diagnostic={c.diagnostic ? "true" : undefined}
                className={`wb-sortable${c.diagnostic ? " diagnostic" : ""}${c.digits !== undefined ? " num" : ""}`}
                title={c.diagnostic ? `${c.title ? `${c.title}; ` : ""}${DIAGNOSTIC_TITLE}` : c.title}
                onClick={() => onSort(c.key)}
                aria-sort={sort?.key === c.key ? (sort.dir === 1 ? "ascending" : "descending") : undefined}
              >
                {c.label}
                {sort?.key === c.key ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody onPointerLeave={() => p.onRowHover?.(null)}>
          {rows.map((r) => {
            const sel = p.isSelected?.(r) ?? false;
            return (
              <tr
                key={p.rowKey(r)}
                data-testid={p.rowTestId}
                data-selected={sel ? "true" : "false"}
                {...p.rowAttrs?.(r)}
                className={`${p.onRowClick ? "wb-click" : ""}${sel ? " selected" : ""}`}
                onClick={p.onRowClick ? (e) => p.onRowClick?.(r, e) : undefined}
                onPointerEnter={p.onRowHover ? () => p.onRowHover?.(r) : undefined}
              >
                {p.columns.map((c) => (
                  <td key={c.key} className={`${c.diagnostic ? "diagnostic" : ""}${c.digits !== undefined ? " num" : ""}`} title={c.diagnostic ? DIAGNOSTIC_TITLE : undefined}>
                    {cellText(c, r)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
