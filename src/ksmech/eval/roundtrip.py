"""The round-trip check as a reranker of the forward beam (reaction-network plan, items H and I5).

A forward step is trusted more when the backward model (item H, `ksmech-train fit --direction
backward`), from the child, finds the parent again. For each stored sample the forward decoder's
completions (`eval.decode`: the beam, caps and checks of `ksmech-eval decode`) are taken in
their order, and each of the first `top` is rescored: applied on the sample's Lewis structure
(`eval.decode.lewis_product`), its child pool read as a `network.pools.PoolState` (token atom k
mapped k + 1), and -R scored there by the backward model (`network.pools.ModelExpander.step_logp`).
The first `top` are then ranked, for each lambda, by

    log P_f(R | parent) + lambda * log P_b(-R | child)

The forward term is the decoder's own score (its best anchor: the order of the beam), the
backward one the marginal over the anchors of -R (`model.decode.MatrixScore.marginal`), the
probability of the previous step whichever of its anchors the model starts from.

Only the completions with a backward score are permuted, among the places they hold; one
without keeps its forward place, with its reason counted: CHILD_INVALID (the child RDKit cannot
read, e.g. a halide with two bonds and no charge, which the network drops as PRODUCT_INVALID),
MISSING_TOKENS, STALE_ARTIFACT or NO_BOND_INDEX (a child species without usable tokens:
`missing_species` lists them for a QM batch), STEP_OUTSIDE_POOL. So lambda = 0 is the forward
order, its hits the decode's, and a missing score neither helps nor hurts a completion. The
completions past `top` keep their places.

Per lambda the report holds the own and any top-k (`eval.decode`: own = a variant of the label
class, any = also a product the source records for the same pool), the same over the samples
whose first `top` completions all have a backward score (`full`), and the top-1 hits gained and
lost against lambda = 0, sample by sample. Its per-lambda rows are `eval.decode.hit_row`s, so
each lambda is written as a `decode_hits` file that `ksmech-eval compare` pairs with another
lambda's or with the decode's.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ksmech.data.chem import SanitizeError, parse_mapped
from ksmech.eval.decode import Decode, hit_row, lewis_product, tops
from ksmech.model.decode import Completion

CHILD_INVALID = "CHILD_INVALID"


class StepScorer(Protocol):
    """`network.pools.ModelExpander` with a backward checkpoint: a `MatrixScore` (or the
    failure code) of a step given on map numbers at a pool state."""

    def step_logp(self, state: Any, bonds, lone) -> Any: ...


@dataclass(frozen=True)
class Rescored:
    """One completion of the first `top`: its backward log P, or None and the reason."""

    completion: Completion
    backward: float | None
    why: str | None = None


def child_state(product: Callable, comp: Completion):
    """The completion's child as a `PoolState`, or CHILD_INVALID when the product cannot be
    built or its SMILES does not sanitize (what the token loader reads)."""
    from ksmech.network.pools import state_of_mol

    mol = product(comp.bonds, comp.lone)
    if mol is None:
        return CHILD_INVALID
    try:
        state = state_of_mol(mol)
        parse_mapped(state.smiles)
    except (SanitizeError, ValueError, RuntimeError):
        return CHILD_INVALID
    return state


def reverse_on_maps(comp: Completion) -> tuple[list, list]:
    """-R of a completion on the child's map numbers (token atom k is map k + 1)."""
    bonds = [(int(i) + 1, int(j) + 1, -int(d)) for i, j, d in comp.bonds]
    return bonds, [(int(i) + 1, -int(d)) for i, d in comp.lone]


def rescore(sample, found: Sequence[Completion], scorer: StepScorer, top: int,
            skips: Counter) -> list[Rescored]:  # fmt: skip
    """The backward score of each of the first `top` completions (module docstring)."""
    product, out = lewis_product(sample), []
    for comp in found[:top]:
        child = child_state(product, comp)
        score = child if isinstance(child, str) else scorer.step_logp(child, *reverse_on_maps(comp))
        if isinstance(score, str):
            skips[f"ROUNDTRIP_UNSCORED:{score}"] += 1
            out.append(Rescored(comp, None, score))
        else:
            out.append(Rescored(comp, float(score.marginal)))
    return out


def combined(r: Rescored, lam: float) -> float:
    """log P_f + lambda log P_b; the forward score alone at lambda = 0 (a backward -inf, a class
    the head has not, would make it nan)."""
    return r.completion.logp if lam == 0 else r.completion.logp + lam * r.backward


def reorder(found: Sequence[Completion], rescored: Sequence[Rescored], lam: float) -> list:
    """The completions with the scored ones of the first `top` ranked by `combined` (ties in
    forward order) within the places they hold; the others where they were."""
    places = [k for k, r in enumerate(rescored) if r.backward is not None]
    ranked = sorted(places, key=lambda k: (-combined(rescored[k], lam), k))
    out = list(found)
    for place, k in zip(places, ranked, strict=True):
        out[place] = rescored[k].completion
    return out


def roundtrip_rows(samples: Iterable, decode: Decode, scorer: StepScorer, lambdas: Sequence[float],
                   top: int, outcomes: Mapping, skips: Counter) -> list[dict]:  # fmt: skip
    """One row per sample with a label: the first `top` completions' forward and backward
    scores (or reasons), and per lambda the `hit_row` of the reordered beam."""
    rows = []
    for sample in samples:
        if sample.edit_label is None:
            skips["NO_EDIT_LABEL"] += 1
            continue
        found = decode(sample, skips)
        rescored = rescore(sample, found, scorer, top, skips)
        rows.append({
            "step_id": getattr(sample, "step_id", None),
            "forward": [r.completion.logp for r in rescored],
            "backward": [r.backward for r in rescored],
            "why": [r.why for r in rescored],
            "full": all(r.backward is not None for r in rescored),
            "by_lambda": {
                f"{lam:g}": hit_row(sample, reorder(found, rescored, lam), outcomes, Counter())
                for lam in lambdas
            },
        })  # fmt: skip
    return rows


def _flips(rows: list[dict], lam: str) -> dict:
    """Top-1 own hits gained and lost against lambda 0, and the top-1s that changed."""

    def hit(r, key):
        return r["by_lambda"][key]["rank"] == 0

    gained = sum(hit(r, lam) and not hit(r, "0") for r in rows)
    lost = sum(hit(r, "0") and not hit(r, lam) for r in rows)
    return {"gained_top1": gained, "lost_top1": lost}


def roundtrip_report(rows: list[dict], lambdas: Sequence[float], ks=(1, 3, 5)) -> dict:
    """Per lambda the own and any top-k over every row and over the `full` rows, the top-1
    flips against lambda 0; the coverage of the backward scores by reason."""
    full = [r for r in rows if r["full"]]
    out: dict[str, Any] = {"n": len(rows), "n_full": len(full), "by_lambda": {}}
    for lam in (f"{x:g}" for x in lambdas):
        hits = [r["by_lambda"][lam] for r in rows]
        sub = [r["by_lambda"][lam] for r in full]
        out["by_lambda"][lam] = (
            tops(hits, "rank", ks) | tops(hits, "rank_any", ks, "_any") | _flips(rows, lam)
            | {"full": tops(sub, "rank", ks) | tops(sub, "rank_any", ks, "_any")}
        )  # fmt: skip
    whys = Counter(w for r in rows for w in r["why"] if w is not None)
    scored = sum(b is not None for r in rows for b in r["backward"])
    out["coverage"] = {"completions": scored + sum(whys.values()), "scored": scored,
                       "unscored": dict(sorted(whys.items()))}  # fmt: skip
    return out


def missing_species(samples: Iterable, decode: Decode, loader, top: int,
                    skips: Counter) -> dict[str, str]:  # fmt: skip
    """{species key: reason} of the child species of the first `top` completions that have no
    usable tokens, for `ksmech-qm run --species-from` and `ksmech-orbitals build`; children
    RDKit cannot read are counted as CHILD_INVALID."""
    from ksmech.arrows.reactants import MissingTokens, ReactantTokens

    out: dict[str, str] = {}
    for sample in samples:
        if sample.edit_label is None:
            continue
        product = lewis_product(sample)
        for comp in decode(sample, skips)[:top]:
            child = child_state(product, comp)
            if isinstance(child, str):
                skips[child] += 1
                continue
            try:
                ReactantTokens.from_smiles(child.smiles, loader)
            except MissingTokens as exc:
                out.update(exc.reasons)
    return dict(sorted(out.items()))
