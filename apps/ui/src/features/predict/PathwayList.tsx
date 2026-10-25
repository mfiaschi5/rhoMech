// PathwayList: the ranked pathways of a network (design §4.5, §11.8). A click shows the pathway
// in the strip and thickens it in the graph; each item links to the Curate page.
//
// Props:
//   jobId      string
//   net        NetworkOut
//   current    PathwayOut | null      the pathway shown (selected, or rank 1)
//   onSelect   (pathwayId) => void
//
// DOM: <ol data-testid="pathway-list"> > <li data-testid="pathway-item-{rank}" data-pathway>,
// each with <a data-testid="curate-pathway-link" href="#/curate/<jobId>?pathway=<pid>">.
import type { NetworkOut, PathwayOut } from "../../api/types.gen";
import { curateHash } from "../../router";
import { fmt, fmtProb } from "../../components/format";

export interface PathwayListProps {
  jobId: string;
  net: NetworkOut;
  current: PathwayOut | null;
  onSelect: (pathwayId: string) => void;
}

export function PathwayList({ jobId, net, current, onSelect }: PathwayListProps) {
  const pathways = [...net.pathways].sort((a, b) => a.rank - b.rank);
  return (
    <section className="panel pathway-panel">
      <h3>Pathways</h3>
      {pathways.length === 0 && <p className="muted">No pathway yet: the root has no outgoing edge.</p>}
      <ol className="pathway-list" data-testid="pathway-list">
        {pathways.map((p) => {
          const selected = current?.id === p.id;
          return (
            <li
              key={p.id}
              data-testid={`pathway-item-${p.rank}`}
              data-pathway={p.id}
              data-ends-with={p.ends_with}
              className={`pathway-item${selected ? " selected" : ""}`}
            >
              <button type="button" className="legend-button" aria-pressed={selected} onClick={() => onSelect(p.id)}>
                <span className="legend-kind">#{p.rank}</span>
                <span className="legend-main">
                  {p.n_steps} step{p.n_steps === 1 ? "" : "s"}, ends {p.ends_with}
                  {p.n_corrected > 0 && <span className="badge badge-corrected">{p.n_corrected} corrected</span>}
                  {p.closes_cycle && <span className="badge legend-cycle">cycle</span>}
                </span>
                <span className="num" title={`log p = ${p.logp}`}>
                  p {fmtProb(p.probability)} <span className="muted">(log {fmt(p.logp, 2)})</span>
                </span>
              </button>
              <a className="btn btn-small" data-testid="curate-pathway-link" href={curateHash(jobId, p.id)}>
                curate
              </a>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
