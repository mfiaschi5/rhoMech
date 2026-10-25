"""A toy chemistry for network and job tests (design 2.3): no RDKit, no tokens, no scheduler.

Species are opaque strings and pools are multisets of them (`Counter`). A `ToyRule` consumes
and produces species with a score; `StubChemistry.expand` turns the scores of the rules that
apply into proposal log-probabilities (`contracts.log_softmax_valid`), groups them by child pool
key and keeps the beam, as the production expansion does. Each toy step is a complete
`edgestep.step_record`-shaped dict (the rule index is its matrix: bond (0, 1) changed by the
index + 1), and toy states carry the pool key as their "mapped SMILES", so
`job_service.network_out` validates stub networks too. Nothing is persisted (stub states are
never served by `/pool`).
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
    ExpandRequest,
    Expansion,
    JobContext,
    PoolState,
    PredictorRef,
    ProposedChild,
    SpeciesCount,
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


def toy_step(rule_index: int, rule: ToyRule) -> dict:
    """A complete step-record dict for rule `rule_index` (its matrix encodes the index)."""
    return {
        "matrix": {"bonds": [[0, 1, int(rule_index) + 1]], "lone": []},
        "anchor": {"i": 0, "j": 1, "sign": "form"},
        "arrows": [],
        "readings": [],
        "product_smiles": sorted(rule.produces),
        "sigma_keys": sorted(rule.produces),
        "formal_charges": [],
        "touched_maps": [],
        "channel": {"rule": rule.name},
        "predictor": "stub",
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


def _child(members: list[_Proposal]) -> ProposedChild:
    rep = members[0]
    alternatives = [
        {
            "matrix": toy_step(m.index, m.rule)["matrix"],
            "anchor": {"i": 0, "j": 1, "sign": "form"},
            "score": m.rule.score,
            "logp": m.logp,
            "rank": m.rank,
        }
        for m in members[1 : 1 + MAX_ALTERNATIVES]
    ]
    return ProposedChild(
        state=rep.child,
        step=toy_step(rep.index, rep.rule),
        score=rep.rule.score,
        logp=rep.logp,
        weight=math.log(sum(math.exp(m.logp) for m in members)),
        rank=rep.rank,
        n_alternatives=len(members) - 1,
        alternatives=alternatives,
    )


def _failed_example(p: _Proposal) -> dict:
    return {
        "rank": p.rank,
        "matrix": toy_step(p.index, p.rule)["matrix"],
        "score": p.rule.score,
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
        groups: dict[str, list[_Proposal]] = {}
        failed = [p for p in props if p.failure is not None]
        for p in props:
            if p.failure is None:
                groups.setdefault(p.child.pool_key, []).append(p)
        mass = {k: sum(math.exp(m.logp) for m in v) for k, v in groups.items()}
        kept = sorted(groups, key=lambda k: (-mass[k], k))[: req.beam]
        scanned = sum(math.exp(p.logp) for p in props)
        return Expansion(
            ok=True,
            code=None,
            upstream=None,
            detail="",
            children=[_child(groups[key]) for key in kept],
            n_valid=len(props),
            n_scanned=len(props),
            scanned_mass=scanned,
            unscanned_mass=max(0.0, 1.0 - scanned),
            failures=dict(sorted(Counter(str(p.failure) for p in failed).items())),
            failed_examples=[_failed_example(p) for p in failed[:MAX_FAILED_EXAMPLES]],
            fermi_smeared=False,
            warnings=[],
        )


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


def requires(capability: str):
    """`pytest.mark.skipif` for tests of a feature that `capabilities.CAPABILITIES` turns off;
    the skip reason carries the capability's own reason."""
    import pytest

    from ksmech.serve.capabilities import CAPABILITIES

    on, reason = CAPABILITIES[capability]
    return pytest.mark.skipif(not on, reason=f"capability {capability} is off: {reason}")
