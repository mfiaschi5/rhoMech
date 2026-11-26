"""Rerankers of a pool's children: a second score of each child, added with a weight lambda to
the forward log P (the cheap tests of 2026-11-26, docs/decisions.md).

    score(child) = log P_f(step | parent) + lambda * s(child)

Two scores:

- `backward`: log P_b(-R | child), the backward model's marginal over the anchors of -R at the
  child (`network.pools.ModelExpander.reverse_logp`), as `eval.roundtrip` reranks the decoder's
  beam on stored samples;
- `energy`: -E(child) / RT, E the sum of the GFN2 total energies (kcal/mol) of the child's
  fragments, read from their QM records (`EnergyScorer`). The children of one pool hold the
  same atoms, so ranking by -E / RT is ranking by -dE / RT of the step. It is an electronic
  energy with the records' ALPB solvation, no thermal or standard-state term;
- `target`: -D(child), D the one-sided deficit of the case's final product in the child pool
  (`target_deficit`): what the product's bond and charged-atom inventory needs that the pool
  lacks. Map-free, since a benchmark product carries no atom maps; one-sided, since the
  by-products are not known. A score of the case (`Rerank.case_score`).

lambda = inf ranks lexicographically: the score first, the forward log P on ties.

Only the first `top` children (best forward first) are rescored, and only those with a score
are permuted among the places they hold; a child without one keeps its forward place, its
reason counted (`RERANK_UNSCORED:<code>`). So lambda 0 is the forward order.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

NO_RECORD, RECORD_STALE, UNCANONICAL = "NO_RECORD", "RECORD_STALE", "UNCANONICAL"
UNPARSED = "UNPARSED"


@dataclass(frozen=True)
class Score:
    """A child's second score, in the slot `eval.roundtrip.rescore` reads (`marginal`)."""

    marginal: float


class EnergyScorer:
    """-E / RT of a pool state from its fragments' QM records, memoized per fragment. Its
    `step_logp(state, bonds, lone)` ignores the step (the child alone sets the score), so it
    stands where `eval.roundtrip` takes a backward scorer."""

    def __init__(self, cache_dir: Path | None = None, method: str | None = None,
                 conformer_id: int = 0, temperature_k: float = 298.15) -> None:  # fmt: skip
        from ksmech.network.free_energy import rt
        from ksmech.qm.records import DEFAULT_CACHE_DIR
        from ksmech.qm.xtb_calc import QMConfig

        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.method = method or QMConfig().method
        self.conformer_id = conformer_id
        self.rt = rt(temperature_k)
        self.memo: dict[str, float | str] = {}

    def fragment_energy(self, key: str) -> float | str:
        """The record's total energy in kcal/mol, or NO_RECORD / RECORD_STALE."""
        if key not in self.memo:
            self.memo[key] = self._load(key)
        return self.memo[key]

    def _load(self, key: str) -> float | str:
        from ksmech.data.provenance import StaleArtifact
        from ksmech.qm.records import cache_path, load_record
        from ksmech.qm.spin import method_for
        from ksmech.qm.thermo import HARTREE_KCAL

        method = method_for(key, self.method)
        path = cache_path(self.cache_dir, key, self.conformer_id, method)
        if not path.exists():
            return NO_RECORD
        try:
            record = load_record(path, (key, self.conformer_id, method))
        except StaleArtifact:
            return RECORD_STALE
        return float(record.energy_ha) * HARTREE_KCAL

    def energy(self, state) -> float | str:
        """E of a pool state (kcal/mol), or the first fragment's failure code."""
        from ksmech.network.thermo_provider import fragment_records

        try:
            keys = [record for _, record in fragment_records(state.smiles)]
        except Exception:  # noqa: BLE001 - counted by the caller
            return UNCANONICAL
        total = 0.0
        for key in keys:
            e = self.fragment_energy(key)
            if isinstance(e, str):
                return e
            total += e
        return total

    def step_logp(self, state, bonds=(), lone=()) -> Score | str:
        e = self.energy(state)
        return e if isinstance(e, str) else Score(-e / self.rt)


def backward_score(expander) -> Callable:
    """The `backward` score of a proposal (module docstring) with a backward expander."""

    def score(proposal) -> float | str:
        s = expander.reverse_logp(proposal)
        return s if isinstance(s, str) else float(s.marginal)

    return score


def energy_score(scorer: EnergyScorer) -> Callable:
    """The `energy` score of a proposal: -E(child) / RT."""

    def score(proposal) -> float | str:
        s = scorer.step_logp(proposal.child)
        return s if isinstance(s, str) else s.marginal

    return score


def inventory(smiles: str) -> Counter | str:
    """The bond and charged-atom inventory of a pool: per (elements, bond type) the bonds with
    explicit H and aromaticity perceived, per (element, charge, radical electrons) the atoms
    with a charge or an unpaired electron; UNPARSED when RDKit cannot sanitize it."""
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return UNPARSED
    mol = Chem.AddHs(mol)
    inv: Counter = Counter()
    for b in mol.GetBonds():
        pair = sorted((b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()))
        inv[("bond", *pair, str(b.GetBondType()))] += 1
    for a in mol.GetAtoms():
        if a.GetFormalCharge() or a.GetNumRadicalElectrons():
            inv[("atom", a.GetSymbol(), a.GetFormalCharge(), a.GetNumRadicalElectrons())] += 1
    return inv


def target_deficit(target: Counter, pool: Counter) -> int:
    """D = sum over the target's inventory entries of what the pool lacks of them."""
    return sum(max(n - pool.get(k, 0), 0) for k, n in target.items())


def target_score(case) -> Callable:
    """The `target` score of a proposal for one case: -D(child) against `case.target`."""
    wanted = inventory(case.target)
    memo: dict[str, float | str] = {}

    def score(proposal) -> float | str:
        if isinstance(wanted, str):
            return wanted
        smiles = proposal.child.smiles
        if smiles not in memo:
            have = inventory(smiles)
            memo[smiles] = have if isinstance(have, str) else -float(target_deficit(wanted, have))
        return memo[smiles]

    return score


def reorder_places(logps: Sequence[float], scores: Sequence[float | None], lam: float) -> list[int]:
    """The order of the first len(scores) places after the rerank: the scored ones ranked by
    log P + lambda s (ties in forward order) among the places they hold, the others where
    they were. Returns, per place, the index of the child it now holds."""
    places = [k for k, s in enumerate(scores) if s is not None]
    if math.isinf(lam):
        key = lambda k: (scores[k], logps[k])  # noqa: E731 - lexicographic (module docstring)
    elif lam == 0:
        key = lambda k: logps[k]  # noqa: E731
    else:
        key = lambda k: logps[k] + lam * scores[k]  # noqa: E731
    ranked = sorted(places, key=lambda k: (_neg(key(k)), k))
    order = list(range(len(scores)))
    for place, k in zip(places, ranked, strict=True):
        order[place] = k
    return order


def _neg(key):
    return tuple(-x for x in key) if isinstance(key, tuple) else -key


@dataclass
class Rerank:
    """A score of proposals, the lambdas it is weighted by (0 always included) and the number
    of children rescored. A score that needs the case (`target`) is given as `case_score`,
    case -> score, and bound per case by `for_case`."""

    score: Callable | None
    lambdas: tuple[float, ...]
    top: int = 5
    case_score: Callable | None = None

    def __post_init__(self) -> None:
        self.lambdas = tuple(sorted({0.0, *self.lambdas}))

    def for_case(self, case) -> Rerank:
        """This rerank with its score bound to `case` (itself when the score needs no case)."""
        if self.case_score is None:
            return self
        return replace(self, score=self.case_score(case), case_score=None)

    def ranks(self, proposals: Sequence, hit: int | None, skips: Counter) -> dict:
        """Per lambda the 1-based rank of child `hit` (None: no child holds the target), and
        `full`: every one of the first `top` children was scored."""
        head = list(proposals[: self.top])
        scores: list[float | None] = []
        for p in head:
            s = self.score(p)
            if isinstance(s, str):
                skips[f"RERANK_UNSCORED:{s}"] += 1
                scores.append(None)
            else:
                scores.append(s)
        scored = [s for s in scores if s is not None]
        out: dict = {"full": all(s is not None for s in scores), "by_lambda": {},
                     "distinct": len(set(scored))}  # fmt: skip
        logps = [p.logp for p in head]
        for lam in self.lambdas:
            if hit is None:
                rank = None
            elif hit < len(head):
                rank = reorder_places(logps, scores, lam).index(hit) + 1
            else:
                rank = hit + 1
            out["by_lambda"][f"{lam:g}"] = rank
        return out
