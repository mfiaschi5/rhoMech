// StepView: one elementary step drawn as its reaction matrix: the reactant pool (PoolCanvas)
// with R overlaid and the curved arrows of its reading, then the product next to it (RDKit.js
// drawing of the product SMILES, one per species) with the sigma keys under it.
//
// Props:
//   stateId        string             the reactant pool state
//   matrix         MatrixIO
//   arrows         ArrowReadingOut[]
//   productSmiles  string[]
//   sigmaKeys      string[]
//   chargesAfter   Map<atom, charge>  formal charges after R (when known)
//   highlight      Set<atom>
//   prefix         string             test id prefix ("predict-step", "correct-preview")
//   height         number             px (default 300)
//
// Hidden H that R or an arrow touches are drawn (poolStructure's `show`).
//
// DOM: <div data-testid="<prefix>"> with the PoolCanvas <prefix>-canvas, the arrow list
// <prefix>-arrow-text-<k>, <prefix>-product (data-smiles) and <prefix>-sigma-<k>.
import { ErrorBanner } from "../../../components/ErrorBanner";
import { useDrawnStructure, usePool } from "../hooks";
import { PoolCanvas } from "../PoolCanvas";
import { arrowText } from "@ksmech/viz/pool/rmatrix";
import { ProductView } from "@ksmech/viz/step/ProductView";
import type { ArrowReadingOut, MatrixIO } from "@ksmech/viz/v9types";

export interface StepViewProps {
  stateId: string;
  matrix: MatrixIO;
  arrows: readonly ArrowReadingOut[];
  productSmiles: readonly string[];
  sigmaKeys?: readonly string[];
  chargesAfter?: ReadonlyMap<number, number> | null;
  highlight?: ReadonlySet<number>;
  prefix: string;
  height?: number;
}

export function StepView(props: StepViewProps) {
  const { stateId, matrix, arrows, prefix } = props;
  const pool = usePool(stateId);
  const s = useDrawnStructure(pool, matrix, arrows);
  return (
    <div className="pv-step" data-testid={prefix}>
      <div className="pv-step-grid">
        <div className="pv-step-reactant stack">
          {s ? <PoolCanvas structure={s} testId={`${prefix}-canvas`} height={props.height ?? 300} matrix={matrix} arrows={arrows} chargesAfter={props.chargesAfter} highlight={props.highlight} /> : <div className="pv-canvas-placeholder muted">{pool.loading ? "loading the depiction…" : "no depiction"}</div>}
          <ErrorBanner error={pool.error} title="Cannot draw the pool" />
          <ol className="pv-arrow-list">
            {arrows.map((a, k) => (
              <li key={k} data-testid={`${prefix}-arrow-text-${k}`} data-kind={a.kind}>
                {arrowText(a, (i) => s?.label(i) ?? String(i))} <span className="muted">({a.kind.replace(/_/g, " ")})</span>
              </li>
            ))}
          </ol>
        </div>
        <div className="pv-step-arrow" aria-hidden="true">
          →
        </div>
        <ProductView smiles={props.productSmiles} sigmaKeys={props.sigmaKeys} prefix={prefix} />
      </div>
    </div>
  );
}
