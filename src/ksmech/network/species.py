"""The species graph and its expansion without a product (reaction-network plan, item J).

In the pool graph (`network.search`) two branches made independently (A -> B, C -> D, then
B + D -> E) are one path per interleaving. The species graph keeps them apart:

- an OR node is a species (`Species`), keyed by its identity key v2
  (`data.resonance.identity_key`, the key of `network.keys`), with a drawing whose map numbers
  are its canonical local atom indices + 1 (`network.lift.local_ranks`), so a species' atoms are
  named the same way wherever it occurs;
- an AND node is an elementary step (`Step`): the sorted multiset of its reactant species (the
  fragments it touches; the pool's other fragments are spectators) to the sorted multiset of its
  product species, its change on (reactant slot, local atom), its atom map (each product atom is
  a reactant atom) and the best log P of the pools that proposed it, so its cost is -log p.
  Steps are merged by (reactants, products) (DUPLICATE_STEP);
- a pathway is a hyperpath (`network.hyperpaths`): the set of steps that makes the target from
  the starting species, a derivation DAG whose joins are AND nodes with several inputs.

`expand_species` grows the graph species by species, as RMG grows its core: the species of
largest mass on the frontier joins the core while that mass is at least `tolerance`, and then
reacts alone, with itself and with each species already in the core, the reservoir (species
always present, never consumed: solvent, reagents in excess) added to every pool; each pool is
one call of a `PoolReactor` (`network.lift.ModelPoolReactor` over the pool `Expander` of
`network.pools`). The species that joins is the carrier of its pools and its partners stay at
fixed concentration (pseudo-first order): a step of one of its pools is a channel of the carrier
with weight w x p, p its probability on the pool and w the pool's share of the carrier's pools
(the species alone 1, each partner pool `partner_weight`, normalized). Each pair of species is
one pool, expanded when the later of the two joins. The mass reaching each species follows
(`network.flux.masses`); `branching[x]` holds the carrier's channels and `leftover[x]` the
share no channel took (the model's mass the decoder did not propose, failed pools, pools cut by
the budget): the unexplored mass of the expansion, reported.

Every cut is counted: PARTNER_ONLY (a step that touches only the partners, or only them and the
reservoir; they react with the carrier at fixed concentration, so their own steps are theirs),
RESERVOIR_ONLY, SELF_LOOP (reactants and products the same multiset, a degenerate exchange: the
carrier comes back, so its other channels are renormalized), DUPLICATE_STEP, POOL_BUDGET (a pool
not expanded), and at the end one BELOW_TOLERANCE or LEFT_ABOVE_TOLERANCE per frontier species;
a pool the reactor could not expand at all is `EXPAND_FAILED:<code>` among the skips, beside the
reactor's own skip codes, MASS_DIVERGED counts the mass solves whose gain reached 1, and
POOL_MASS_ABOVE_ONE the pools whose proposals summed above probability 1 and were renormalized
(the decoder's distinct completions never do).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

StepKey = tuple[tuple[str, ...], tuple[str, ...]]

PARTNER_ONLY, RESERVOIR_ONLY, SELF_LOOP = "PARTNER_ONLY", "RESERVOIR_ONLY", "SELF_LOOP"
DUPLICATE_STEP, POOL_BUDGET, MASS_DIVERGED = "DUPLICATE_STEP", "POOL_BUDGET", "MASS_DIVERGED"
BELOW_TOLERANCE, LEFT_ABOVE_TOLERANCE = "BELOW_TOLERANCE", "LEFT_ABOVE_TOLERANCE"
TOLERANCE, SPECIES_BUDGET = "TOLERANCE", "SPECIES_BUDGET"  # the reasons an expansion stops


@dataclass(frozen=True)
class Species:
    """An OR node: the identity key, the number of atoms (H included), and, for a real species,
    a drawing (explicit H, Kekule, map number = local atom index + 1), its heavy-atom skeleton
    key (`network.target.skeleton_key`) and the atomic number of each local atom."""

    key: str
    n_atoms: int
    smiles: str = ""
    skeleton: str = ""
    elements: tuple[int, ...] = ()


@dataclass(frozen=True)
class Step:
    """An AND node (module docstring). `atom_map[j][b]` is the (reactant slot, local atom) that
    local atom b of product slot j is; `bonds` holds (slot, atom, slot, atom, d) and `lone`
    (slot, atom, dN) on the reactant slots."""

    reactants: tuple[str, ...]
    products: tuple[str, ...]
    logp: float
    atom_map: tuple[tuple[tuple[int, int], ...], ...] = ()
    bonds: tuple[tuple[int, int, int, int, int], ...] = ()
    lone: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if list(self.reactants) != sorted(self.reactants) or list(self.products) != sorted(
            self.products
        ):
            raise ValueError("a step's reactant and product species are sorted")

    @property
    def key(self) -> StepKey:
        return (self.reactants, self.products)

    @property
    def cost(self) -> float:
        return max(-self.logp, 0.0)

    @property
    def text(self) -> str:
        return f"{' + '.join(self.reactants)} >> {' + '.join(self.products)}"


@dataclass
class PoolResult:
    """What a reactor proposes on one pool: the steps, the species records of their products,
    the skip counts, and a reason code when the pool could not be expanded at all."""

    steps: list[Step]
    species: dict[str, Species] = field(default_factory=dict)
    skips: Counter = field(default_factory=Counter)
    failure: str | None = None


class PoolReactor(Protocol):
    def react(self, pool: Sequence[Species]) -> PoolResult: ...


@dataclass(frozen=True)
class ExpansionParams:
    tolerance: float = 0.05  # the mass a frontier species needs to join the core
    max_species: int = 10  # species in the core (the k-th joins with k + 1 pools)
    max_pools: int = 100  # pools expanded (model calls)
    partner_weight: float = 1.0  # a partner pool's weight against the species alone


DEFAULT_EXPANSION = ExpansionParams()


@dataclass
class SpeciesGraph:
    species: dict[str, Species]
    sources: dict[str, float]  # starting species -> amount (molecules in the start pool)
    reservoir: tuple[str, ...] = ()
    steps: dict[StepKey, Step] = field(default_factory=dict)
    core: list[str] = field(default_factory=list)  # in the order the species joined
    branching: dict[str, dict[StepKey, float]] = field(default_factory=dict)  # carrier's channels
    leftover: dict[str, float] = field(default_factory=dict)  # carrier's unexplored share
    mass: dict[str, float] = field(default_factory=dict)
    pools: int = 0  # pools expanded
    cuts: Counter = field(default_factory=Counter)
    skips: Counter = field(default_factory=Counter)
    reason: str = ""

    def producers(self, key: str) -> list[Step]:
        return [s for s in self.steps.values() if key in s.products]

    def consumers(self, key: str) -> list[Step]:
        return [s for s in self.steps.values() if key in s.reactants]

    def frontier(self) -> list[str]:
        """The species neither in the core nor in the reservoir, sorted."""
        out = set(self.species) - set(self.core) - set(self.reservoir)
        return sorted(out)

    def add_step(self, step: Step) -> None:
        """Merge a step by (reactants, products), the best log P kept (DUPLICATE_STEP)."""
        old = self.steps.get(step.key)
        if old is None:
            self.steps[step.key] = step
            return
        self.cuts[DUPLICATE_STEP] += 1
        if step.logp > old.logp:
            self.steps[step.key] = replace(old, logp=step.logp)


def _excluded(step: Step, carrier: str, reservoir: Sequence[str]) -> str | None:
    if carrier not in step.reactants:
        return RESERVOIR_ONLY if set(step.reactants) <= set(reservoir) else PARTNER_ONLY
    if Counter(step.reactants) == Counter(step.products):
        return SELF_LOOP
    return None


def _absorb(g: SpeciesGraph, res: PoolResult, carrier: str, w: float, beta: Counter) -> float:
    """Add one pool's steps to the graph and the carrier's channels; the weight that loops back
    to the carrier (SELF_LOOP) is returned."""
    for key, sp in res.species.items():
        g.species.setdefault(key, sp)
    g.skips.update(res.skips)
    if res.failure is not None:
        g.skips[f"EXPAND_FAILED:{res.failure}"] += 1
    total = sum(math.exp(s.logp) for s in res.steps)
    scale = 1.0 / total if total > 1.0 + 1e-9 else 1.0  # never above 1: counted if it were
    if scale < 1.0:
        g.cuts["POOL_MASS_ABOVE_ONE"] += 1
    loop = 0.0
    for step in res.steps:
        p = w * scale * math.exp(step.logp)
        why = _excluded(step, carrier, g.reservoir)
        if why is not None:
            g.cuts[why] += 1
            loop += p if why == SELF_LOOP else 0.0
            continue
        g.add_step(step)
        beta[step.key] += p
    return loop


def _join_core(g: SpeciesGraph, reactor: PoolReactor, x: str, params: ExpansionParams) -> None:
    """`x` joins the core: its pools alone, with itself and with each earlier core species."""
    partners = [(), (x,), *((c,) for c in g.core)]
    weights = [1.0] + [params.partner_weight] * (len(partners) - 1)
    total = sum(weights)
    g.core.append(x)
    beta: Counter = Counter()
    loop = 0.0
    for partner, w in zip(partners, weights, strict=True):
        if g.pools >= params.max_pools:
            g.cuts[POOL_BUDGET] += 1
            continue
        pool = [g.species[k] for k in (x, *partner, *g.reservoir)]
        g.pools += 1
        loop += _absorb(g, reactor.react(pool), x, w / total, beta)
    back = 1.0 / (1.0 - loop) if loop < 1.0 - 1e-12 else 0.0
    g.branching[x] = {k: b * back for k, b in sorted(beta.items())}
    g.leftover[x] = max(0.0, 1.0 - sum(g.branching[x].values()))


def _next_species(g: SpeciesGraph, tolerance: float) -> str | None:
    """The frontier species of largest mass (ties by key) when it reaches the tolerance."""
    best = min(g.frontier(), key=lambda k: (-g.mass.get(k, 0.0), k), default=None)
    if best is None or g.mass.get(best, 0.0) < tolerance:
        return None
    return best


def expand_species(
    reactor: PoolReactor, species: Mapping[str, Species], start: Mapping[str, float],
    reservoir: Sequence[str] = (), params: ExpansionParams = DEFAULT_EXPANSION,
) -> SpeciesGraph:  # fmt: skip
    """The species graph grown from the starting species (module docstring); `species` holds
    the records of the starting and reservoir species, `start` their amounts."""
    from ksmech.network.flux import masses

    missing = sorted((set(start) | set(reservoir)) - set(species))
    if missing:
        raise ValueError(f"no species record for {missing}")
    g = SpeciesGraph(dict(species), dict(start), tuple(sorted(set(reservoir))))
    while True:
        g.mass, diverged = masses(g)
        g.cuts[MASS_DIVERGED] += int(diverged)
        nxt = _next_species(g, params.tolerance)
        if nxt is None:
            g.reason = TOLERANCE
            break
        if len(g.core) >= params.max_species:
            g.reason = SPECIES_BUDGET
            break
        if g.pools >= params.max_pools:
            g.reason = POOL_BUDGET
            break
        _join_core(g, reactor, nxt, params)
    for k in g.frontier():
        below = g.mass.get(k, 0.0) < params.tolerance
        g.cuts[BELOW_TOLERANCE if below else LEFT_ABOVE_TOLERANCE] += 1
    g.cuts = Counter({k: n for k, n in g.cuts.items() if n})
    return g
