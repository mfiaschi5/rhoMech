// The step lists of the Curate page (design §11.8):
//   PathwayStepList   the steps of a predicted pathway (spec-v9 EdgeOut): the source state's
//                     depiction, the step's reaction matrix and its arrow reading as text, and an
//                     "independent of previous step" toggle on every step after the first (the
//                     order groups of POST /curate);
//   MechanismStepList the steps of a stored mechanism (CuratedStepOut + its StepDoc), with the
//                     original arrows of every corrected step and the latest re-check row.
// DOM: <ol> of <li data-testid="curate-step-{i}" data-order-group data-corrected data-edge>, each
// with input[data-testid="independent-toggle-{i}"] for i >= 1.
import type { ReactNode } from "react";
import type { ArrowReadingOut, CuratedStepOut, DatasetArrowOut, EdgeOut, MatrixIO, NetworkOut, PathwayOut } from "../../api/types.gen";
import type { TokenArrowLike } from "../../chem/arrowGeom";
import { correctHash } from "../../router";
import type { IndependenceHint } from "./orderGroups";
import { StateDepiction } from "./StateDepiction";
import { arrowMaps, arrowText, datasetArrowText, fmtNum, fmtProb, parseStepDoc, type StepDocView } from "./stepdoc";

export interface ToggleProps {
  checked: boolean;
  disabled: boolean;
  onChange: (value: boolean) => void;
}

interface StepCardProps {
  index: number;
  group: number;
  corrected: boolean;
  edgeId?: string;
  stateId: string | null;
  fallbackSmiles: string;
  arrows: readonly TokenArrowLike[];
  datasetArrows?: readonly DatasetArrowOut[] | null;
  original?: { arrows: readonly TokenArrowLike[]; note: string } | null;
  toggle?: ToggleProps | null;
  hints?: readonly IndependenceHint[];
  children?: ReactNode;
}

function HintList({ hints }: { hints: readonly IndependenceHint[] }) {
  if (!hints.length) return null;
  return (
    <ul className="cu-hints">
      {hints.map((h) => (
        <li key={`${h.code}-${h.other}`} className={h.code === "NOT_INDEPENDENT" ? "cu-hint-error" : "cu-hint-warn"} data-code={h.code}>
          <code>{h.code}</code>{" "}
          {h.code === "NOT_INDEPENDENT"
            ? `shares maps ${h.maps.join(", ")} with step ${h.other + 1}: the server will refuse this grouping`
            : `maps are not continuous from step ${h.other + 1} (a merge renumbered them): accepted unchecked`}
        </li>
      ))}
    </ul>
  );
}

function IndependentToggle({ index, toggle }: { index: number; toggle: ToggleProps }) {
  return (
    <label className="cu-toggle">
      <input
        type="checkbox"
        data-testid={`independent-toggle-${index}`}
        checked={toggle.checked}
        disabled={toggle.disabled}
        onChange={(e) => toggle.onChange(e.target.checked)}
      />
      independent of previous step
    </label>
  );
}

interface OriginalArrowsProps {
  index: number;
  stateId: string | null;
  fallbackSmiles: string;
  original: { arrows: readonly TokenArrowLike[]; note: string };
  keep: readonly number[];
}

/** The prediction a corrected step replaced, faded on the same source state. */
function OriginalArrows({ index, stateId, fallbackSmiles, original, keep }: OriginalArrowsProps) {
  return (
    <div className="cu-original" data-testid={`curate-original-${index}`}>
      <div className="muted">{original.note}</div>
      <StateDepiction
        stateId={stateId}
        fallbackSmiles={fallbackSmiles}
        arrows={original.arrows}
        extraMaps={keep}
        faded
        width={220}
        height={150}
        testId={`curate-original-depiction-${index}`}
      />
      <ArrowTextList arrows={original.arrows} />
    </div>
  );
}

function StepCard(p: StepCardProps) {
  // one keep_maps set for both drawings of the step, so they share one cached depiction
  const keep = [...new Set([...arrowMaps(p.arrows, p.datasetArrows), ...arrowMaps(p.original?.arrows)])];
  return (
    <li
      className={`cu-step${p.corrected ? " cu-step-corrected" : ""}`}
      data-testid={`curate-step-${p.index}`}
      data-order-group={p.group}
      data-corrected={p.corrected ? "true" : "false"}
      data-edge={p.edgeId ?? ""}
    >
      <div className="cu-step-head">
        <strong>Step {p.index + 1}</strong>
        <span className="badge" title="order group: steps sharing one are independent of each other">
          group {p.group}
        </span>
        {p.corrected && <span className="badge cu-badge-corrected">corrected</span>}
      </div>
      {p.toggle && <IndependentToggle index={p.index} toggle={p.toggle} />}
      <HintList hints={p.hints ?? []} />
      <StateDepiction
        stateId={p.stateId}
        fallbackSmiles={p.fallbackSmiles}
        arrows={p.datasetArrows ? null : p.arrows}
        datasetArrows={p.datasetArrows}
        extraMaps={keep}
        edgeId={p.edgeId}
      />
      {p.original && <OriginalArrows index={p.index} stateId={p.stateId} fallbackSmiles={p.fallbackSmiles} original={p.original} keep={keep} />}
      <div className="cu-step-body">{p.children}</div>
    </li>
  );
}

function ArrowTextList({ arrows }: { arrows: readonly TokenArrowLike[] }) {
  return (
    <ul className="cu-arrow-text">
      {arrows.map((a, k) => (
        <li key={k}>{arrowText(a)}</li>
      ))}
    </ul>
  );
}

interface EdgeDetailsProps {
  index: number;
  jobId: string;
  edge: EdgeOut;
  sourceKey: string;
}

/** "+3–5 −5–9 lp 3 −2": R on the pool atoms. */
function matrixText(m: MatrixIO): string {
  const bonds = m.bonds.map(([i, j, d]) => `${d > 0 ? "+" : "−"}${i}–${j}`);
  const lone = m.lone.map(([i, d]) => `lp ${i} ${d > 0 ? "+" : "−"}${Math.abs(d)}`);
  return [...bonds, ...lone].join(" ") || "(empty)";
}

const readingText = (a: ArrowReadingOut) =>
  `${a.source.length === 1 ? `lp ${a.source[0]}` : a.source.join("–")} → ${a.target.length === 1 ? `lp ${a.target[0]}` : a.target.join("–")}`;

function EdgeDetails({ index, jobId, edge, sourceKey }: EdgeDetailsProps) {
  return (
    <>
      <ul className="cu-arrow-text">
        {edge.arrows.map((a, k) => (
          <li key={k}>{readingText(a)}</li>
        ))}
      </ul>
      <dl className="kv cu-kv">
        <dt>R</dt>
        <dd>
          <code>{matrixText(edge.matrix)}</code>
        </dd>
        <dt>log p</dt>
        <dd className="num">{fmtNum(edge.logp)}</dd>
        <dt>score</dt>
        <dd className="num">{fmtNum(edge.score)}</dd>
        <dt>rank</dt>
        <dd>{edge.rank > 0 ? edge.rank : "user"}{edge.n_alternatives > 0 ? ` (${edge.n_alternatives} alternatives)` : ""}</dd>
        <dt>pools</dt>
        <dd>
          <code>{sourceKey}</code> → <code>{edge.target_pool_key}</code>
        </dd>
        <dt>product</dt>
        <dd>
          <code>{edge.product_smiles.join(".")}</code>
        </dd>
        <dt>PMechDB code</dt>
        <dd>{edge.pmechdb_code ? <code>{edge.pmechdb_code}</code> : "–"}</dd>
      </dl>
      <a className="btn btn-small" href={correctHash(jobId, { edge: edge.id })} data-testid={`curate-correct-link-${index}`}>
        correct this step
      </a>
    </>
  );
}

export interface PathwayStepListProps {
  jobId: string;
  network: NetworkOut;
  pathway: PathwayOut;
  groups: readonly number[];
  independent: readonly boolean[];
  hints: readonly IndependenceHint[];
  disabled: boolean;
  onToggle: (index: number, value: boolean) => void;
}

export function PathwayStepList(p: PathwayStepListProps) {
  const edges = new Map(p.network.edges.map((e) => [e.id, e]));
  const nodes = new Map(p.network.nodes.map((n) => [n.id, n]));
  const last = edges.get(p.pathway.edge_ids[p.pathway.edge_ids.length - 1] ?? "");
  return (
    <ol className="cu-steps" data-testid="curate-steps" data-n-steps={p.pathway.edge_ids.length}>
      {p.pathway.edge_ids.map((id, i) => {
        const edge = edges.get(id);
        if (!edge) {
          return (
            <li key={id} className="cu-step" data-testid={`curate-step-${i}`} data-missing="true">
              <div className="cu-hint-error">
                <code>EDGE_NOT_FOUND</code> edge {id} of this pathway is not in the network
              </div>
            </li>
          );
        }
        const source = nodes.get(edge.source);
        return (
          <StepCard
            key={id}
            index={i}
            group={p.groups[i] ?? i}
            corrected={edge.kind === "corrected"}
            edgeId={edge.id}
            stateId={edge.source_state_id}
            fallbackSmiles={source?.display_smiles ?? ""}
            arrows={[]}
            original={null}
            hints={p.hints.filter((h) => h.step === i)}
            toggle={i > 0 ? { checked: !!p.independent[i], disabled: p.disabled, onChange: (v) => p.onToggle(i, v) } : null}
          >
            <EdgeDetails index={i} jobId={p.jobId} edge={edge} sourceKey={source?.pool_key ?? edge.source} />
          </StepCard>
        );
      })}
      {last && (
        <li className="cu-step cu-product" data-testid="curate-product">
          <div className="cu-step-head">
            <strong>Product</strong> <code>{last.target_pool_key}</code>
          </div>
          <StateDepiction stateId={last.product_state_id} fallbackSmiles={last.product_smiles.join(".")} />
        </li>
      )}
    </ol>
  );
}

function RecheckBrief({ step }: { step: CuratedStepOut }) {
  const r = step.recheck;
  if (!r) return <div className="muted">not re-checked yet</div>;
  return (
    <div className="cu-recheck-brief" data-testid={`curate-step-recheck-${step.step_index}`} data-top1={String(r.top1)}>
      re-check {r.predictor}: rank {r.first_arrow_rank ?? "–"} / {r.n_valid ?? "–"}, top1 {String(r.top1 ?? "–")},
      product {r.product_match}
      {r.reason && (
        <>
          {" "}
          <code>{r.reason}</code>
        </>
      )}
    </div>
  );
}

function DocDetails({ step, doc }: { step: CuratedStepOut; doc: StepDocView }) {
  return (
    <>
      <ArrowTextList arrows={doc.arrows} />
      {doc.badArrows > 0 && <div className="cu-hint-error">{doc.badArrows} arrow(s) of the StepDoc cannot be drawn</div>}
      <dl className="kv cu-kv">
        <dt>probability</dt>
        <dd className="num">{fmtProb(doc.probability)}</dd>
        <dt>rank</dt>
        <dd>{doc.rank ?? "–"}</dd>
        <dt>pools</dt>
        <dd>
          <code>{step.reactant_pool_key}</code> → <code>{step.product_pool_key}</code>
        </dd>
        <dt>dataset arrows</dt>
        <dd>{doc.datasetArrows ? doc.datasetArrows.map(datasetArrowText).join("; ") : "–"}</dd>
        <dt>PMechDB code</dt>
        <dd>{doc.pmechdbCode ? <code>{doc.pmechdbCode}</code> : "–"}</dd>
        <dt>touched maps</dt>
        <dd>{doc.touchedMaps.join(", ") || "–"}</dd>
      </dl>
      {doc.encodeReason && (
        <div className="cu-hint-warn">
          <code>ARROW_ENCODING_FAILED</code> {doc.encodeReason}: an export skips this step
        </div>
      )}
      <RecheckBrief step={step} />
    </>
  );
}

export interface MechanismStepListProps {
  steps: readonly CuratedStepOut[];
  groups: readonly number[];
  independent: readonly boolean[];
  hints: readonly IndependenceHint[];
  disabled: boolean;
  drawDataset: boolean;
  onToggle: (index: number, value: boolean) => void;
}

export function MechanismStepList(p: MechanismStepListProps) {
  return (
    <ol className="cu-steps" data-testid="curate-steps" data-n-steps={p.steps.length}>
      {p.steps.map((step, i) => {
        const doc = parseStepDoc(step.step);
        const orig = step.original ? parseStepDoc(step.original) : null;
        const original = step.was_corrected
          ? { arrows: orig?.arrows ?? [], note: orig ? `original prediction (p ${fmtProb(orig.probability)})` : "original prediction not stored" }
          : null;
        return (
          <StepCard
            key={step.step_index}
            index={i}
            group={p.groups[i] ?? step.order_group}
            corrected={step.was_corrected}
            edgeId={doc.edgeId ?? undefined}
            stateId={doc.reactantStateId}
            fallbackSmiles={doc.reactantsMapped}
            arrows={doc.arrows}
            datasetArrows={p.drawDataset ? (doc.datasetArrows ?? []) : null}
            original={original}
            hints={p.hints.filter((h) => h.step === i)}
            toggle={i > 0 ? { checked: !!p.independent[i], disabled: p.disabled, onChange: (v) => p.onToggle(i, v) } : null}
          >
            <DocDetails step={step} doc={doc} />
          </StepCard>
        );
      })}
    </ol>
  );
}
