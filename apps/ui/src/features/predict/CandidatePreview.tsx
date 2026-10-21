// CandidatePreview: one candidate tuple drawn as a curved arrow on the node's pool depiction
// (design §11.8). Clicking the arrow (hit path or handle) opens the physics drawer.
//
// Props:
//   stateId       string
//   tuple         [donor, donor_site, acceptor, acceptor_site]
//   candidate     CandidateOut | null   the row it came from (else the site maps and elements
//                                       are resolved from GET /pool/{state}/tokens)
//   onArrowClick  () => void
//   onClose       () => void
//   selected      boolean               the physics drawer shows this tuple
//
// DOM: <div data-testid="candidate-preview" data-tuple> holding mol-depiction[data-atom-xy] and
// its arrow-overlay[data-role=main] group.
import type { CandidateOut, TokensOut } from "../../api/types.gen";
import { arrowFromCandidate, type TokenArrowLike } from "../../chem/arrowGeom";
import { ErrorBanner } from "../../components/ErrorBanner";
import { PendingNote } from "../../components/PendingNote";
import { arrowText } from "./network";
import { usePoolTokens } from "./hooks";
import { StateDepiction } from "./StateDepiction";

export interface CandidatePreviewProps {
  stateId: string;
  tuple: readonly [number, number, number, number];
  candidate: CandidateOut | null;
  onArrowClick: () => void;
  onClose: () => void;
  selected: boolean;
}

/** The tuple as a drawable arrow from the atom tokens (token id = global site). */
function arrowFromTokens(tokens: TokensOut, t: readonly number[]): TokenArrowLike | null {
  const atom = (site: number) => tokens.atoms.find((a) => a.token_id === site);
  const a = atom(t[1]);
  const b = atom(t[3]);
  if (!a || !b) return null;
  return {
    donor: t[0],
    donor_site: t[1],
    acceptor: t[2],
    acceptor_site: t[3],
    donor_site_map: a.map,
    acceptor_site_map: b.map,
    donor_element: a.element,
    acceptor_element: b.element,
    source_partner_map: null,
    sink_partner_map: null,
    electrons: 2,
  };
}

const matches = (c: CandidateOut | null, t: readonly number[]) =>
  !!c && c.donor === t[0] && c.donor_site === t[1] && c.acceptor === t[2] && c.acceptor_site === t[3];

export function CandidatePreview({ stateId, tuple, candidate, onArrowClick, onClose, selected }: CandidatePreviewProps) {
  const known = matches(candidate, tuple) ? candidate : null;
  const tokens = usePoolTokens(known ? null : stateId);
  const arrow = known ? arrowFromCandidate(known) : tokens.data ? arrowFromTokens(tokens.data, tuple) : null;
  return (
    <div className="candidate-preview panel" data-testid="candidate-preview" data-tuple={tuple.join(",")}>
      <div className="row">
        <strong>Preview</strong>
        {arrow && <code className="code">{arrowText(arrow)}</code>}
        <span className="muted">
          donor #{tuple[0]} site {tuple[1]} → acceptor #{tuple[2]} site {tuple[3]}
        </span>
        <button type="button" className="btn btn-small" onClick={onClose}>
          close
        </button>
      </div>
      {!known && tokens.loading && <PendingNote label={tokens.pending?.label ?? "loading tokens"} worker={tokens.pending?.worker} />}
      <ErrorBanner error={tokens.error} title="Cannot resolve the tuple's sites" />
      {!known && tokens.data && !arrow && <div className="banner banner-error">sites {tuple[1]} / {tuple[3]} are not atoms of this pool</div>}
      {arrow && (
        <StateDepiction
          stateId={stateId}
          arrows={[arrow]}
          edgeId={`cand:${tuple.join(",")}`}
          onArrowClick={() => onArrowClick()}
          selectedArrow={selected ? 0 : null}
          width={360}
          height={250}
          caption="click the arrow for its physics"
        />
      )}
    </div>
  );
}
