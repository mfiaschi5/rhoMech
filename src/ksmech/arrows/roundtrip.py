"""Round-trip checks: arrows applied to the reactants must give the annotated products.

A step matches `exact` when the edited molecule has the annotated canonical Kekulé SMILES, and
`resonance` when only the resonance keys agree. Everything else is a failure, with a pattern
string that groups failures of the same kind: the reason code (VALENCE, SANITIZE,
UNKNOWN_ARROW, MISMATCH) and, for a mismatch, which properties of which elements differ.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ksmech.arrows.align import StepAlignment, align_step
from ksmech.arrows.align import report as alignment_report
from ksmech.arrows.apply import ApplyResult, apply_arrows
from ksmech.arrows.apply_predicted import (
    PredictedArrow,
    apply_predicted_arrows,
    labeled_arrows,
)
from ksmech.arrows.compare import MATCH_STATUSES, match_products
from ksmech.arrows.reactants import (
    CorrespondenceError,
    MissingTokens,
    ReactantTokens,
    TokenLoader,
)
from ksmech.data import chem
from ksmech.data.schema import ElementaryStep

N_EXAMPLES = 3


@dataclass(frozen=True)
class RoundTrip:
    step_id: str
    status: str  # exact | resonance | MISMATCH | an ApplyReason value
    pattern: str = ""  # failure group; empty on a match
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in MATCH_STATUSES


def arrow_signature(step: ElementaryStep) -> str:
    """Arrow shapes of a step, e.g. `lone_pair>atom,bond>atom`."""
    return ",".join(f"{a.source_kind}>{a.sink_kind}" for a in step.arrows)


def compare_products(result: ApplyResult, step: ElementaryStep) -> tuple[str, str]:
    """(status, pattern) of an applied step against its annotated products."""
    return match_products(result, chem.parse_mapped(step.products_smiles), step.products_smiles)


def round_trip_step(step: ElementaryStep) -> RoundTrip:
    """Apply the annotated arrows of `step` and compare with its products."""
    result = apply_arrows(chem.parse_mapped(step.reactants_smiles), step.arrows)
    status, pattern = compare_products(result, step)
    if status in MATCH_STATUSES:
        return RoundTrip(step.step_id, status)
    return RoundTrip(step.step_id, status, f"{pattern} | {arrow_signature(step)}", result.detail)


def summarize(results: Iterable[RoundTrip], top: int = 10) -> dict:
    """Match rate, status counts and the `top` failure patterns with example step ids."""
    results = list(results)
    status = Counter(r.status for r in results)
    examples: dict[str, list[str]] = defaultdict(list)
    for r in results:
        if not r.ok:
            examples[r.pattern].append(r.step_id)
    patterns = sorted(examples.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top]
    n_ok = sum(status[s] for s in MATCH_STATUSES)
    return {
        "n_steps": len(results),
        "n_match": n_ok,
        "match_rate": n_ok / len(results) if results else 0.0,
        "status": dict(status.most_common()),
        "top_failures": [
            {"pattern": p, "count": len(ids), "examples": ids[:N_EXAMPLES]} for p, ids in patterns
        ],
    }


def run_round_trip(
    steps: Iterable[ElementaryStep], fn: Callable[[ElementaryStep], RoundTrip] = round_trip_step
) -> tuple[list[RoundTrip], dict]:
    results = [fn(s) for s in steps]
    return results, summarize(results)


def predicted_round_trip_step(
    step: ElementaryStep, tokens: ReactantTokens, alignment: StepAlignment
) -> RoundTrip:
    """Apply the labeled (token, site) tuples of a completely aligned step from the sites
    alone (`apply_predicted_arrows`) and compare with the annotated products."""
    arrows = labeled_arrows(tokens, alignment.labels)
    result = apply_predicted_arrows(tokens.mol, arrows, tokens)
    status, pattern = compare_products(result, step)
    if status in MATCH_STATUSES:
        return RoundTrip(step.step_id, status)
    flags = ",".join(sorted({str(f) for lab in alignment.labels for f in lab.flags})) or "-"
    return RoundTrip(step.step_id, status, f"{pattern} | flags {flags}", result.detail)


def member_agreement(
    step: ElementaryStep, tokens: ReactantTokens, alignment: StepAlignment
) -> Counter:
    """Swap one non-representative member of one label set at a time into the labeled arrows
    and count ("donor" | "acceptor", reproduces the products) outcomes. Shows whether every
    member of a set, which the loss treats as correct, gives the same product."""
    base = labeled_arrows(tokens, alignment.labels)
    out: Counter = Counter()
    for k, label in enumerate(alignment.labels):
        for side, members in (("donor", label.donor_ids), ("acceptor", label.acceptor_ids)):
            for m in sorted(members):
                b = base[k]
                if m == (b.donor if side == "donor" else b.acceptor):
                    continue
                swapped = PredictedArrow(
                    m if side == "donor" else b.donor,
                    b.donor_site,
                    m if side == "acceptor" else b.acceptor,
                    b.acceptor_site,
                )
                arrows = base[:k] + [swapped] + base[k + 1 :]
                result = apply_predicted_arrows(tokens.mol, arrows, tokens)
                out[(side, compare_products(result, step)[0] in MATCH_STATUSES)] += 1
    return out


def evaluate_alignment(
    steps: Iterable[ElementaryStep], loader: TokenLoader, member_check: bool = True
) -> dict:
    """One pass over `steps`: align every step whose reactants are tokenized, apply the
    labeled tuples of every completely aligned step from their sites, and (optionally) swap
    in every other set member. Returns the alignment report with a `predicted` section."""
    alignments, predicted, skipped, members = [], [], Counter(), Counter()
    for step in steps:
        try:
            rt = ReactantTokens.from_smiles(step.reactants_smiles, loader)
        except MissingTokens:
            skipped["MISSING_TOKENS"] += 1
            continue
        except CorrespondenceError:
            skipped["CORRESPONDENCE"] += 1
            continue
        al = align_step(step, rt)
        alignments.append(al)
        if al.complete:
            predicted.append(predicted_round_trip_step(step, rt, al))
            if member_check and predicted[-1].ok:
                members.update(member_agreement(step, rt, al))
    out = alignment_report(alignments, skipped)
    out["predicted"] = summarize(predicted)
    out["predicted"]["member_agreement"] = {
        side: {
            "agree": members[(side, True)],
            "total": members[(side, True)] + members[(side, False)],
        }
        for side in ("donor", "acceptor")
    }
    return out
