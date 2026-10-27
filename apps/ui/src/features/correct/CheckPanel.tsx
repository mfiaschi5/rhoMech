// CheckPanel: what the live /step/check says about the R being edited: the issues (local ones
// first; hovering one highlights its atoms on the canvas), the product preview with its sigma
// keys, the formal charges after R and the arrow reading with its orbital channels.
//
// Props: check (useStepCheck), local (rmatrix.localIssues), structure, stateId, onHover(atoms).
// DOM: correct-check[data-state="idle|checking|ok|issues|error"], correct-issues >
// li[data-testid=correct-issue-<k>][data-code][data-atoms], correct-preview-product (ProductView),
// correct-charges > li[data-testid=correct-charge-<atom>][data-charge], the reading list
// correct-reading-arrow-<k>, correct-readings-status[data-status="ready|pending|unavailable"]
// and the ReadingTable "correct".
import { ErrorBanner } from "../../components/ErrorBanner";
import type { StepCheck } from "../predict/hooks";
import type { PoolStructure } from "@ksmech/viz/pool/poolStructure";
import { ReadingTable } from "../predict/ReadingTable";
import { arrowText, type LocalIssue } from "@ksmech/viz/pool/rmatrix";
import { ProductView } from "@ksmech/viz/step/ProductView";

export interface CheckPanelProps {
  check: StepCheck & { current: boolean };
  local: LocalIssue[];
  structure: PoolStructure;
  stateId: string;
  empty: boolean;
  onHover: (atoms: number[] | null) => void;
}

const READINGS_TEXT: Record<string, string> = {
  ready: "ready",
  pending: "pending (the pool's analysis is being computed; checked again in a few seconds)",
  unavailable: "unavailable (this pool has no tokens)",
};

const signed = (v: number) => (v > 0 ? `+${v}` : String(v));

function checkState(p: CheckPanelProps): string {
  if (p.empty) return "idle";
  if (p.check.error) return "error";
  if (!p.check.current || !p.check.result) return p.local.some((i) => i.code === "MATRIX_MALFORMED") ? "issues" : "checking";
  return p.check.result.ok && p.local.length === 0 ? "ok" : "issues";
}

function Issues({ issues, onHover, structure }: { issues: LocalIssue[]; onHover: CheckPanelProps["onHover"]; structure: PoolStructure }) {
  if (!issues.length) return null;
  return (
    <ul className="pv-issues" data-testid="correct-issues">
      {issues.map((x, k) => (
        <li key={k} data-testid={`correct-issue-${k}`} data-code={x.code} data-atoms={x.atoms.join(",")} onPointerEnter={() => onHover(x.atoms)} onPointerLeave={() => onHover(null)}>
          <code className="code">{x.code}</code> {x.message}
          {x.atoms.length > 0 && <span className="muted"> ({x.atoms.map((i) => structure.label(i)).join(", ")})</span>}
        </li>
      ))}
    </ul>
  );
}

export function CheckPanel(props: CheckPanelProps) {
  const { check, local, structure, stateId } = props;
  const res = check.current ? check.result : null;
  const state = checkState(props);
  // the server's issues, less the codes the local check already lists (with their atoms)
  const localCodes = new Set(local.map((i) => i.code));
  const issues = [...local, ...(res?.issues ?? []).filter((i) => !localCodes.has(i.code))];
  const charges = (res?.formal_charges ?? []).filter(([i, q]) => q !== 0 || (structure.atom(i)?.charge ?? 0) !== 0);
  return (
    <div className="pv-check stack" data-testid="correct-check" data-state={state}>
      <div className="row pv-inline-row">
        <span className={`badge pv-check-badge pv-check-${state}`}>{state === "idle" ? "empty R" : state === "checking" ? "checking…" : state === "ok" ? "valid step" : state === "error" ? "check failed" : "issues"}</span>
      </div>
      <Issues issues={issues} onHover={props.onHover} structure={structure} />
      <ErrorBanner error={check.error} title="Cannot check the step" testId="correct-check-error" />
      <ProductView smiles={res?.product_smiles ?? []} sigmaKeys={res?.sigma_keys} prefix="correct-preview" />
      {charges.length > 0 && (
        <ul className="pv-charges" data-testid="correct-charges">
          {charges.map(([i, q]) => (
            <li key={i} data-testid={`correct-charge-${i}`} data-charge={q}>
              {structure.label(i)}: {signed(structure.atom(i)?.charge ?? 0)} → {signed(q)}
            </li>
          ))}
        </ul>
      )}
      {res && res.arrows.length > 0 && (
        <ol className="pv-arrow-list">
          {res.arrows.map((a, k) => (
            <li key={k} data-testid={`correct-reading-arrow-${k}`} data-kind={a.kind}>
              {arrowText(a, structure.label)} <span className="muted">({a.kind.replace(/_/g, " ")})</span>
            </li>
          ))}
        </ol>
      )}
      {res && (
        <div className={`muted pv-readings-${res.readings_status}`} data-testid="correct-readings-status" data-status={res.readings_status}>
          orbital readings: {READINGS_TEXT[res.readings_status]}
        </div>
      )}
      {res && res.readings.length > 0 && <ReadingTable readings={res.readings} structure={structure} stateId={stateId} prefix="correct" />}
    </div>
  );
}
