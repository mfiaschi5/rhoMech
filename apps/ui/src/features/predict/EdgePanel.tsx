// EdgePanel: the selected network edge (one elementary step): its arrows drawn on the source
// state (a click opens the physics drawer), the per-arrow physics brief, score, probability and
// rank, the product, the dataset encoding, cycles, alternatives and a "correct" link. A corrected
// edge also shows the original prediction it replaced, faded, and what the predictor thought of
// each user arrow; a superseded edge links to its correction.
//
// Props:
//   jobId     string
//   net       NetworkOut
//   edge      EdgeOut
//   sel       JobSelection
//   select    (updates) => void
//
// DOM: edge-panel[data-edge][data-kind][data-status], edge-correct-link, edge-arrow-{i}.
import type { ArrowOut, EdgeOut, NetworkOut } from "../../api/types.gen";
import { correctHash } from "../../router";
import { fmt, fmtProb, fmtSig, moLabel, shortId } from "../../components/format";
import { arrowText, edgeArrowsText, speciesLabel } from "./network";
import { CLEAR_TARGET, type JobSelection, type SelectionUpdate } from "./selection";
import { StateDepiction } from "./StateDepiction";

export interface EdgePanelProps {
  jobId: string;
  net: NetworkOut;
  edge: EdgeOut;
  sel: JobSelection;
  select: (updates: SelectionUpdate) => void;
}

function ArrowBrief({ a, i, onOpen, active }: { a: ArrowOut; i: number; onOpen: () => void; active: boolean }) {
  const p = a.physics;
  return (
    <li className={`edge-arrow${active ? " selected" : ""}`} data-testid={`edge-arrow-${i}`} data-site-source={a.site_source}>
      <button type="button" className="legend-button" aria-pressed={active} onClick={onOpen}>
        <span className="legend-kind">{i + 1}.</span>
        <span className="legend-main">
          <code className="code">{arrowText(a)}</code> #{a.donor} → #{a.acceptor}{" "}
          <span className="muted">
            ({a.source_kind}
            {a.sink_partner_map !== null ? `, releases ${a.acceptor_element}${a.acceptor_site_map}–${a.sink_partner_map}` : ""}; sites {a.site_source})
          </span>{" "}
          <span className="diagnostic" title="diagnostic: the classifier's names, display only; never a model input">
            {[a.diagnostic.donor_class, a.diagnostic.acceptor_class].filter(Boolean).join(" → ")}
          </span>
        </span>
        {p && (
          <span className="num muted">
            KS3 {fmtSig(p.ks3, 4)} eV, floor rank {p.floor_rank ?? "–"} / {p.n_valid}
            {p.dominant ? `, ${moLabel(p.dominant.k_label)}→${moLabel(p.dominant.m_label)}` : ""}
            {p.clash ? ", CLASH" : ""}
            {!p.pair_valid ? `, invalid: ${p.reason ?? "?"}` : ""}
          </span>
        )}
      </button>
    </li>
  );
}

function Summary({ edge, net, select }: { edge: EdgeOut; net: NetworkOut; select: (u: SelectionUpdate) => void }) {
  const nodes = new Map(net.nodes.map((n) => [n.id, n] as const));
  const src = nodes.get(edge.source);
  const tgt = nodes.get(edge.target);
  const goNode = (id: string) => select({ node: id, edge: null, ...CLEAR_TARGET });
  return (
    <dl className="kv">
      <dt>from / to</dt>
      <dd>
        <button type="button" className="link-button" onClick={() => goNode(edge.source)}>
          {src ? speciesLabel(src) : shortId(edge.source)}
        </button>{" "}
        →{" "}
        <button type="button" className="link-button" onClick={() => goNode(edge.target)}>
          {tgt ? speciesLabel(tgt) : shortId(edge.target)}
        </button>
      </dd>
      <dt>product</dt>
      <dd>
        <code className="code">{edge.product_smiles}</code>
      </dd>
      <dt>score / p / rank</dt>
      <dd>
        {edge.kind === "corrected" ? "user correction (log p 0)" : `${fmtSig(edge.score, 5)} / ${fmtProb(edge.probability)} / ${edge.rank ?? "–"}`}
        <span className="muted">
          {" "}
          log p {fmt(edge.logp, 3)}; {edge.n_alternatives} alternative{edge.n_alternatives === 1 ? "" : "s"}; {edge.predictor ?? "no predictor"} {edge.predictor_version ?? ""}
        </span>
      </dd>
      <dt>dataset arrows</dt>
      <dd>
        {edge.pmechdb_code ? <code className="code">{edge.pmechdb_code}</code> : <span className="muted">not encoded{edge.encode_reason ? `: ${edge.encode_reason}` : ""}</span>}
      </dd>
      {(edge.recurrence || edge.closes_cycle || edge.cycle_ids.length > 0) && (
        <>
          <dt>cycles</dt>
          <dd>
            {edge.recurrence && <span className="badge">recurrence</span>} {edge.closes_cycle && <span className="badge legend-cycle">closes a turnover cycle</span>} {edge.cycle_ids.join(", ")}
          </dd>
        </>
      )}
    </dl>
  );
}

function Original({ edge }: { edge: EdgeOut }) {
  const o = edge.original;
  if (!o) return <div className="muted">corrected from a node: no original edge</div>;
  return (
    <div className="edge-original" data-testid="edge-original" data-original-edge={o.edge_id}>
      <strong>Original prediction</strong> <span className="muted">({o.predictor ?? "?"}, rank {o.rank ?? "–"}, p {fmtProb(o.probability)})</span>
      <StateDepiction stateId={edge.source_state_id} arrows={o.arrows} edgeId={o.edge_id} fadedArrows width={260} height={180} caption={o.arrows.map(arrowText).join("; ") || "(no arrows)"} />
      {edge.predictor_view && (
        <ul className="muted">
          {edge.predictor_view.map((v, i) => (
            <li key={i}>
              arrow {i + 1}: predictor score {fmtSig(v.score, 4)}, p {fmtProb(v.probability)}, rank {v.rank ?? "–"}
              {v.pair_valid ? "" : `, invalid: ${v.reason ?? "?"}`}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function EdgePanel({ jobId, net, edge, sel, select }: EdgePanelProps) {
  const openArrow = (k: number) => select({ arrow: k, phys: null, threed: null });
  return (
    <section className={`panel edge-panel kind-${edge.kind}`} data-testid="edge-panel" data-edge={edge.id} data-kind={edge.kind} data-status={edge.status}>
      <div className="row">
        <h3>
          Step <code className="code">{shortId(edge.id, 12)}</code>
        </h3>
        <span className={`badge kind-${edge.kind}${edge.kind === "corrected" ? " badge-corrected" : ""}`}>{edge.kind}</span>
        <span className="badge">{edge.status}</span>
        {edge.superseded_by && (
          <button type="button" className="btn btn-small" onClick={() => select({ edge: edge.superseded_by, ...CLEAR_TARGET })}>
            show its correction
          </button>
        )}
        <a className="btn btn-small" data-testid="edge-correct-link" href={correctHash(jobId, { edge: edge.id })}>
          correct this step
        </a>
      </div>
      <div className="row">
        <StateDepiction
          stateId={edge.source_state_id}
          arrows={edge.arrows}
          edgeId={edge.id}
          onArrowClick={openArrow}
          selectedArrow={sel.edge === edge.id ? sel.arrow : null}
          width={340}
          height={230}
          caption={edgeArrowsText(edge)}
        />
        <Summary edge={edge} net={net} select={select} />
      </div>
      <ol className="edge-arrows">
        {edge.arrows.map((a, i) => (
          <ArrowBrief key={i} a={a} i={i} active={sel.arrow === i} onOpen={() => openArrow(i)} />
        ))}
      </ol>
      {edge.kind === "corrected" && <Original edge={edge} />}
      {edge.alternatives.length > 0 && (
        <details>
          <summary>{edge.alternatives.length} alternative arrow sets to the same product</summary>
          <ul>
            {edge.alternatives.map((alt) => (
              <li key={alt.rank}>
                rank {alt.rank}: {alt.arrows.map(arrowText).join("; ")} (score {fmtSig(alt.score, 4)}, p {fmtProb(alt.probability)})
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
