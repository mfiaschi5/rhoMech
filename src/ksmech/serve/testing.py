"""A toy chemistry for network and job tests (design 2.3): no RDKit, no tokens, no scheduler.

Species are opaque strings and pools are multisets of them (`Counter`). A `ToyRule` consumes
and produces species with a score; `StubChemistry.expand` turns the scores of the rules that
apply into proposal probabilities (`contracts.log_softmax_valid`) and groups and stops the scan
with `contracts.ChildScan`, exactly as the production expansion does, so the network tests
exercise the production rule. Toy arrows are complete `ArrowOut`-shaped dicts and toy states
carry the pool key as their "mapped SMILES", so `job_service.network_out` validates stub
networks too. Nothing is persisted (stub states are never served by `/pool`).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ksmech.serve.contracts import (
    ArrowSpec,
    ChildScan,
    ExpandRequest,
    Expansion,
    JobContext,
    PoolState,
    PredictorRef,
    ProposedChild,
    SpeciesCount,
    StepOutcome,
    log_softmax_valid,
)
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.keys import pool_id, pool_key_of, split_pool_key

TOY_NOT_APPLICABLE = "TOY_RULE_NOT_APPLICABLE"
MAX_ALTERNATIVES = 5
MAX_FAILED_EXAMPLES = 10

ToyPredictorRef = PredictorRef(name="stub", version="toy/1", checkpoint_hash=None, temperature=1.0)
"""The `PredictorRef` the network tests build toy networks with."""


@dataclass(frozen=True)
class ToyRule:
    """`consumes` -> `produces` with a predictor score. `form` gives the product another Lewis
    form (another state id for the same pool key), to exercise alias handling."""

    name: str
    consumes: Counter
    produces: Counter
    score: float
    form: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "consumes", Counter(dict(self.consumes)))
        object.__setattr__(self, "produces", Counter(dict(self.produces)))
        object.__setattr__(self, "score", float(self.score))

    def applies(self, pool: Counter) -> bool:
        return all(pool[k] >= n for k, n in self.consumes.items())

    def product(self, pool: Counter) -> Counter:
        return (pool - self.consumes) + self.produces


def toy_state(pool: Counter | Mapping[str, int], form: str = "") -> PoolState:
    """The toy state of a pool: mapped SMILES = the pool key (plus "|form" for another Lewis
    form), state id "s" + sha256(that)[:20], no fragments and no atoms."""
    counts = Counter({k: int(n) for k, n in dict(pool).items() if n > 0})
    key = pool_key_of(k for k, n in counts.items() for _ in range(n))
    mapped = key if not form else f"{key}|{form}"
    species = tuple(SpeciesCount(k, (k,), counts[k], 0, "OK") for k in sorted(counts))
    return PoolState(
        state_id="s" + hashlib.sha256(mapped.encode()).hexdigest()[:20],
        mapped_smiles=mapped,
        pool_key=key,
        pool_id=pool_id(key),
        species=species,
        fragment_token_keys=(),
        n_atoms=0,
    )


def toy_arrow(rule_index: int, rule: ToyRule, site_source: str = "rule") -> dict:
    """A complete `ArrowOut`-shaped dict for rule `rule_index` (donor = the rule index)."""
    consumed = sorted(rule.consumes) or ["?"]
    produced = sorted(rule.produces) or consumed
    return {
        "donor": int(rule_index),
        "donor_site": 0,
        "acceptor": 0,
        "acceptor_site": 1,
        "donor_site_map": 1,
        "acceptor_site_map": 2,
        "donor_element": "X",
        "acceptor_element": "X",
        "donor_species": consumed[0],
        "acceptor_species": consumed[-1] if len(consumed) > 1 else produced[0],
        "electrons": 2,
        "source_kind": "unknown",
        "source_partner": None,
        "source_partner_map": None,
        "sink_partner": None,
        "sink_partner_map": None,
        "site_source": site_source,
        "row": 0,
        "col": 0,
        "donor_centres_map": {},
        "acceptor_centres_map": {},
        "diagnostic": {},
        "physics": None,
    }


@dataclass
class _Proposal:
    index: int
    rule: ToyRule
    rank: int
    logp: float
    child: PoolState | None
    failure: str | None


def _proposals(rules: Sequence[ToyRule], pool: Counter, parent_key: str, t: float) -> list:
    """Every applicable rule in rank order (score desc, then rule order), with its p."""
    hits = [(i, r) for i, r in enumerate(rules) if r.applies(pool)]
    if not hits:
        return []
    scores = np.array([r.score for _, r in hits], dtype=float)
    logp = log_softmax_valid(scores, np.ones(len(hits), dtype=bool), t)
    order = sorted(range(len(hits)), key=lambda k: (-scores[k], hits[k][0]))
    out = []
    for rank, k in enumerate(order, start=1):
        i, rule = hits[k]
        child = toy_state(rule.product(pool), rule.form)
        failure = str(Code.RESONANCE_ONLY_STEP) if child.pool_key == parent_key else None
        out.append(_Proposal(i, rule, rank, float(logp[k]), None if failure else child, failure))
    return out


def _child(members: list[_Proposal], mass: float) -> ProposedChild:
    rep = members[0]
    alternatives = [
        {
            "arrows": [toy_arrow(m.index, m.rule)],
            "score": m.rule.score,
            "probability": math.exp(m.logp),
            "rank": m.rank,
        }
        for m in members[1 : 1 + MAX_ALTERNATIVES]
    ]
    return ProposedChild(
        state=rep.child,
        arrows=[toy_arrow(rep.index, rep.rule)],
        encoded=None,
        encode_reason=None,
        touched_maps=[],
        score=rep.rule.score,
        probability=mass,
        logp=math.log(mass) if mass > 0 else float("-inf"),
        rank=rep.rank,
        n_alternatives=len(members) - 1,
        alternatives=alternatives,
    )


def _failed_example(p: _Proposal) -> dict:
    return {
        "rank": p.rank,
        "arrows": [toy_arrow(p.index, p.rule)],
        "probability": math.exp(p.logp),
        "code": str(p.failure),
        "detail": f"rule {p.rule.name} gives the parent pool again",
    }


@dataclass
class StubChemistry:
    """The table-driven chemistry of the network and job tests."""

    rules: list[ToyRule]
    failures: dict[str, str] = field(default_factory=dict)  # pool key -> expand failure code
    mode: str = "stub"
    calls: list[tuple[str, str]] = field(default_factory=list)  # (method, pool key)

    def root(self, inputs: Sequence[Any]) -> PoolState:
        """Each input's SMILES is a dot-joined list of opaque species, times its count."""
        pool: Counter = Counter()
        for inp in inputs:
            parts = [p for p in str(inp.smiles).split(".") if p]
            if not parts:
                raise ServeError(Code.EMPTY_INPUT, "an input field is empty", status=422)
            for p in parts:
                pool[p] += int(inp.count)
        if not pool:
            raise ServeError(Code.EMPTY_INPUT, "no reactant given", status=422)
        return toy_state(pool)

    def expand(self, state: PoolState, req: ExpandRequest, ctx: JobContext) -> Expansion:
        self.calls.append(("expand", state.pool_key))
        if state.pool_key in self.failures:
            code = self.failures[state.pool_key]
            return Expansion.failed(code, f"toy failure for {state.pool_key}")
        t = 1.0 if req.temperature is None else float(req.temperature)
        props = _proposals(self.rules, split_pool_key(state.pool_key), state.pool_key, t)
        scan = ChildScan(req.beam, req.max_scan)
        groups: dict[str, list[_Proposal]] = {}
        failed: list[_Proposal] = []
        for k, p in enumerate(props):
            if scan.should_stop(len(props) - k):
                break
            if p.failure is not None:
                scan.add(math.exp(p.logp), None, p.failure)
                failed.append(p)
                continue
            scan.add(math.exp(p.logp), p.child.pool_key)
            groups.setdefault(p.child.pool_key, []).append(p)
        children = [_child(groups[key], scan.mass[key]) for key in scan.kept()]
        return Expansion(
            ok=True,
            code=None,
            upstream=None,
            detail="",
            children=children,
            n_valid=len(props),
            n_scanned=scan.n_scanned,
            scanned_mass=scan.scanned_mass,
            unscanned_mass=scan.unscanned_mass,
            failures=dict(sorted(scan.failures.items())),
            failed_examples=[_failed_example(p) for p in failed[:MAX_FAILED_EXAMPLES]],
            fermi_smeared=False,
            warnings=[],
        )

    def _view(self, pool: Counter, index: int, t: float) -> dict:
        props = _proposals(self.rules, pool, "", t)
        hit = next((p for p in props if p.index == index), None)
        if hit is None:
            score = self.rules[index].score if 0 <= index < len(self.rules) else None
            return {
                "score": score,
                "probability": None,
                "rank": None,
                "pair_valid": False,
                "reason": TOY_NOT_APPLICABLE,
            }
        p = math.exp(hit.logp)
        return {"score": hit.rule.score, "probability": p, "rank": hit.rank, "pair_valid": True}

    def apply(
        self,
        state: PoolState,
        arrows: Sequence[ArrowSpec],
        allow_invalid: bool,
        predictor: str,
        temperature: float | None,
        ctx: JobContext,
    ) -> StepOutcome:
        """Apply rule `arrow.donor` for each arrow, in order."""
        self.calls.append(("apply", state.pool_key))
        t = 1.0 if temperature is None else float(temperature)
        pool = split_pool_key(state.pool_key)
        out_arrows, views, form = [], [], ""
        for a in arrows:
            i = int(a.donor)
            if not 0 <= i < len(self.rules) or not self.rules[i].applies(pool):
                msg = f"toy rule {i} does not apply to {pool_key_of(pool.elements())}"
                raise ServeError(Code.APPLY_FAILED, msg, status=422, upstream=TOY_NOT_APPLICABLE)
            views.append(self._view(pool, i, t) | {"reason": None})
            arrow = toy_arrow(i, self.rules[i], site_source="user")
            arrow.update(
                donor_site=a.donor_site, acceptor=a.acceptor, acceptor_site=a.acceptor_site
            )
            out_arrows.append(arrow)
            pool, form = self.rules[i].product(pool), self.rules[i].form
        child = toy_state(pool, form)
        if child.pool_key == state.pool_key:
            msg = f"the arrows give the parent pool {state.pool_key} again"
            raise ServeError(Code.RESONANCE_ONLY_STEP, msg, status=422)
        return StepOutcome(child, out_arrows, None, None, [], views, [])


def toy_catalytic_cycle() -> StubChemistry:
    """The worked example of design 4.4: Cat + S -> CS -> CP -> Cat + P (root Cat x1, S x2)."""
    return StubChemistry(
        [
            ToyRule("bind", Counter({"Cat": 1, "S": 1}), Counter({"CS": 1}), 2.0),
            ToyRule("convert", Counter({"CS": 1}), Counter({"CP": 1}), 1.0),
            ToyRule("release", Counter({"CP": 1}), Counter({"Cat": 1, "P": 1}), 1.5),
        ]
    )


def toy_keto_enol() -> StubChemistry:
    """K <-> E: a reversible return (a recurrence cycle K -> E -> K)."""
    return StubChemistry(
        [
            ToyRule("enolize", Counter({"K": 1}), Counter({"E": 1}), 1.0),
            ToyRule("ketonize", Counter({"E": 1}), Counter({"K": 1}), 0.5),
        ]
    )
