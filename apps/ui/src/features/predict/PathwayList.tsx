// PathwayList: the ranked pathways of a network (spec v9 "Inference": ranked paths through the
// network). A click shows the pathway (thickened in the graph, its steps listed) and selects its
// first step; each item links to the Curate page.
//
// Props:
//   jobId, net, current (the shown pathway), onSelect(pathwayId), probabilities (score units),
//   selectedEdge, onEdge(edgeId)
//
// DOM: <ol data-testid="predict-pathways"> > <li data-testid="predict-pathway-{rank}"
// data-pathway data-selected>, with its step buttons predict-pathway-step-{rank}-{k} and
// <a data-testid="predict-curate-link-{rank}">.
import { fmt } from "../../components/format";
import { curateHash } from "../../router";
import { matrixText, scoreText } from "./network";
import { Panel } from "./Panel";
import type { NetworkOut, PathwayOut } from "./v9types";

export interface PathwayListProps {
  jobId: string;
  net: NetworkOut;
  current: PathwayOut | null;
  onSelect: (pathwayId: string) => void;
  probabilities: boolean;
  selectedEdge: string | null;
  onEdge: (edgeId: string) => void;
}

function Steps({ p, net, probabilities, selectedEdge, onEdge }: { p: PathwayOut } & Omit<PathwayListProps, "jobId" | "current" | "onSelect">) {
  const edges = new Map(net.edges.map((e) => [e.id, e] as const));
  return (
    <ol className="pv-steps">
      {p.edge_ids.map((id, k) => {
        const e = edges.get(id);
        return (
          <li key={id}>
            <button type="button" className={`legend-button pv-step${id === selectedEdge ? " selected" : ""}`} data-testid={`predict-pathway-step-${p.rank}-${k}`} data-edge={id} aria-pressed={id === selectedEdge} onClick={() => onEdge(id)}>
              <span className="legend-kind">{k + 1}</span>
              <span className="legend-main">{e ? matrixText(e.matrix) : id}</span>
              <span className="num">{e ? scoreText(e, probabilities) : ""}</span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

export function PathwayList(props: PathwayListProps) {
  const { jobId, net, current, onSelect, probabilities } = props;
  const pathways = [...net.pathways].sort((a, b) => a.rank - b.rank);
  return (
    <Panel title="Pathways" note={`${pathways.length} ranked`} className="pv-pathways-panel">
      {pathways.length === 0 && <p className="muted">No pathway yet: the root has no outgoing step.</p>}
      <ol className="pathway-list" data-testid="predict-pathways">
        {pathways.map((p) => {
          const selected = current?.id === p.id;
          return (
            <li key={p.id} data-testid={`predict-pathway-${p.rank}`} data-pathway={p.id} data-selected={selected ? "true" : "false"} className={`pathway-item${selected ? " selected" : ""}`}>
              <div className="pv-path-row">
                <button type="button" className="legend-button pv-path-button" aria-pressed={selected} onClick={() => onSelect(p.id)} title={`${p.n_steps} steps, ends ${p.ends_with}`}>
                  <span className="pv-path-rank">#{p.rank}</span>
                  <span className="pv-path-main">
                    {p.n_steps} step{p.n_steps === 1 ? "" : "s"} · {p.ends_with.toLowerCase().replace(/_/g, " ")}
                    {p.n_corrected > 0 && <span className="nv-chip nv-chip-corrected">{p.n_corrected} corrected</span>}
                    {p.closes_cycle && <span className="nv-chip nv-chip-cycle">↺</span>}
                  </span>
                  <span className="num pv-path-score" title={`log score ${p.logp}`}>
                    {probabilities ? `p ${fmt(p.probability, 3)}` : `Σ ${fmt(p.logp, 2)}`}
                  </span>
                </button>
                <a className="pv-path-curate" data-testid={`predict-curate-link-${p.rank}`} href={curateHash(jobId, p.id)} title="curate this pathway">
                  curate
                </a>
              </div>
              {selected && <Steps p={p} {...props} />}
            </li>
          );
        })}
      </ol>
    </Panel>
  );
}
