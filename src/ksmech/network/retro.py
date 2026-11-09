"""The previous steps from a product alone: mechanistic retrosynthesis (reaction-network plan,
item K).

The start is the target molecule only; the reactants and the by-products are unknown.

- **By-products from a reservoir counted from the data** (`byproduct_reservoir`): the small
  species (at most `max_heavy` heavy atoms) that a recorded step makes and its reactants do not
  hold (leaving groups, ions, water), ranked by the number of steps that release them. Never a
  hand list.
- **Backward expansion** (`RetroExpander`): an OR node is a species to make; one backward step
  from it (a `RetroStep`) names the species it comes from, an AND node whose every parent must
  be made in turn. `ModelRetroExpander` runs the backward model (item H, `ksmech-train fit
  --direction backward`) on the species alone and on the species with each of the `n_byproducts`
  most frequent reservoir species x: a step on the enlarged pool counts only if it touches x
  (else NOT_USING_BYPRODUCT), since going back it is the by-product that joins.
- **Where a branch stops** (`Leaves`): a species of the building-block catalogue given (a new data
  dependency: none ships with the repo), a reservoir species, or one `stable(species)` declares
  stable on its own at T (item F's sink test, a callback). A START head trained on the first pools
  of pathways comes later (plan section 6).
- **AND-OR search** (`retro_search`): best-first on the AND-OR tree, AO* with an admissible value
  (0 by default, so the cost of a route is exact: the sum of -log p of its steps): from the root,
  at each OR node the AND child of lowest lower bound is followed, at each AND node its first
  unsolved parent, until an unexpanded OR node is reached and expanded; the bounds are revised up
  the tree. A species already on the path from the root is not expanded again (CYCLE). The search
  stops when the root's best route is solved and no lower bound is below it (the optimum), or at
  `max_expansions` / `max_depth` (counted).
- **Forward verification** (`verify_forward`): from the route's leaves taken as one pool, the
  forward search (`network.towards`) must reach the target; only verified routes count as found.

Budgets are flags and every cut is counted.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

INF = math.inf
CYCLE, DEPTH_CUT, EXPANSION_BUDGET = "CYCLE", "DEPTH_CUT", "EXPANSION_BUDGET"
NOT_USING_BYPRODUCT = "NOT_USING_BYPRODUCT"


@dataclass(frozen=True)
class RetroStep:
    """One backward step from a species: the species it comes from (with repeats), its log P,
    and anything else (the by-product it used, the pool steps)."""

    parents: tuple[str, ...]
    logp: float
    info: Mapping = field(default_factory=dict)

    @property
    def cost(self) -> float:
        return -self.logp


class RetroExpander(Protocol):
    def expand(self, species: str) -> list[RetroStep]: ...


@dataclass(frozen=True)
class Leaves:
    """What ends a branch: the catalogue, the reservoir, and a stability test."""

    catalogue: frozenset[str] = frozenset()
    reservoir: frozenset[str] = frozenset()
    stable: Callable[[str], bool] | None = None

    def reason(self, species: str) -> str | None:
        if species in self.catalogue:
            return "BUILDING_BLOCK"
        if species in self.reservoir:
            return "RESERVOIR"
        if self.stable is not None and self.stable(species):
            return "STABLE_AT_T"
        return None


@dataclass
class OrNode:
    species: str
    depth: int
    leaf: str | None = None  # the reason it ends a branch
    children: list[AndNode] = field(default_factory=list)
    expanded: bool = False
    bound: float = 0.0  # lower bound on the cost to make it; exact when solved
    solved: bool = False


@dataclass
class AndNode:
    step: RetroStep
    parents: list[OrNode]
    bound: float = 0.0
    solved: bool = False


@dataclass(frozen=True)
class RetroParams:
    max_expansions: int = 100  # species expanded (each one a model call per pool)
    max_depth: int = 7  # backward steps on a branch (the human benchmark goes to 7)


DEFAULT_RETRO = RetroParams()


@dataclass
class Route:
    """A solved AND-OR tree: its steps root first (species, step), its leaves, its cost."""

    steps: list[tuple[str, RetroStep]]
    leaves: Counter
    cost: float


@dataclass
class RetroResult:
    root: OrNode
    route: Route | None
    expanded: int
    reason: str  # SOLVED | EXHAUSTED | EXPANSION_BUDGET | ROOT_IS_LEAF
    cuts: Counter


def _revise_and(a: AndNode) -> None:
    a.bound = a.step.cost + sum(p.bound for p in a.parents)
    a.solved = all(p.solved for p in a.parents)


def _revise_or(o: OrNode) -> None:
    """An expanded OR node: the bound of its best AND child (inf with none)."""
    if o.leaf is not None:
        o.bound, o.solved = 0.0, True
        return
    if not o.expanded:
        return
    for a in o.children:
        _revise_and(a)
    best = min(o.children, key=lambda a: (a.bound, not a.solved), default=None)
    o.bound = INF if best is None else best.bound
    o.solved = best is not None and best.solved


def _select(root: OrNode) -> list[OrNode]:
    """The path from the root to the unexpanded OR node of the best partial route."""
    path, node = [root], root
    while node.expanded and node.leaf is None:
        best = min(node.children, key=lambda a: (a.bound, not a.solved), default=None)
        if best is None or not math.isfinite(best.bound):
            return path
        open_parents = [p for p in best.parents if not p.solved]
        if not open_parents:
            return path
        node = open_parents[0]
        path.append(node)
    return path


def _expand(node: OrNode, path: list[OrNode], expander, leaves, params, cuts, value) -> None:
    node.expanded = True
    if node.depth >= params.max_depth:
        cuts[DEPTH_CUT] += 1
        return
    on_path = {o.species for o in path}
    for step in expander.expand(node.species):
        if any(s in on_path for s in step.parents):
            cuts[CYCLE] += 1
            continue
        parents = [OrNode(s, node.depth + 1, leaves.reason(s)) for s in step.parents]
        for p in parents:
            p.bound = value(p.species) if value is not None and p.leaf is None else 0.0
            _revise_or(p)
        node.children.append(AndNode(step, parents))


def retro_search(
    expander: RetroExpander, target: str, leaves: Leaves, params: RetroParams = DEFAULT_RETRO,
    value: Callable[[str], float] | None = None,
) -> RetroResult:  # fmt: skip
    """The cheapest route from building blocks to `target` (module docstring); `value` an
    admissible lower bound on a species' cost (0 without one)."""
    root, cuts, expanded = OrNode(target, 0, leaves.reason(target)), Counter(), 0
    if root.leaf is not None:
        return RetroResult(root, Route([], Counter({target: 1}), 0.0), 0, "ROOT_IS_LEAF", cuts)
    reason = "EXHAUSTED"
    while not root.solved and math.isfinite(root.bound):
        path = _select(root)
        node = path[-1]
        if node.expanded:  # the best route cannot go on: its bounds are final
            break
        if expanded >= params.max_expansions:
            cuts[EXPANSION_BUDGET] += 1
            reason = "EXPANSION_BUDGET"
            break
        _expand(node, path, expander, leaves, params, cuts, value)
        expanded += 1
        for o in reversed(path):
            _revise_or(o)
    if root.solved:
        reason = "SOLVED"
    return RetroResult(root, best_route(root), expanded, reason, cuts)


def best_route(root: OrNode) -> Route | None:
    """The solved route of lowest cost under `root`, None when the root is unsolved."""
    if not root.solved:
        return None
    steps: list[tuple[str, RetroStep]] = []
    leaves: Counter = Counter()

    def walk(o: OrNode) -> None:
        if o.leaf is not None:
            leaves[o.species] += 1
            return
        a = min((a for a in o.children if a.solved), key=lambda a: a.bound)
        steps.append((o.species, a.step))
        for p in a.parents:
            walk(p)

    walk(root)
    return Route(steps, leaves, root.bound)


def byproduct_reservoir(
    steps: Iterable, max_heavy: int = 4, top: int = 10, skips: Counter | None = None
) -> list[tuple[str, str, int]]:
    """(identity key, SMILES, steps that release it) of the `top` small species the recorded
    steps make that their reactants do not hold, most frequent first (module docstring)."""
    from rdkit import Chem, rdBase

    from ksmech.data.chem import parse_mapped, unmapped_canonical
    from ksmech.data.resonance import identity_key

    skips = Counter() if skips is None else skips
    counts: Counter = Counter()
    smiles: dict[str, str] = {}
    for step in steps:
        try:
            with rdBase.BlockLogs():
                sides = [Chem.GetMolFrags(parse_mapped(x), asMols=True, sanitizeFrags=False)
                         for x in (step.reactants_smiles, step.products_smiles)]  # fmt: skip
                before = {identity_key(f) for f in sides[0]}
                for f in sides[1]:
                    key = identity_key(f)
                    heavy = sum(a.GetAtomicNum() > 1 for a in f.GetAtoms())
                    if key not in before and heavy <= max_heavy:
                        counts[key] += 1
                        smiles.setdefault(key, unmapped_canonical(f))
        except Exception:  # noqa: BLE001 - counted
            skips["BYPRODUCT_UNREADABLE"] += 1
    return [(k, smiles[k], n) for k, n in counts.most_common(top)]


def verify_forward(forward, route: Route, target_smiles: str, smiles_of: Mapping[str, str],
                   params=None) -> tuple[bool, object]:  # fmt: skip
    """Whether the forward search from the route's leaves (one pool, with their counts) reaches
    the target, and the search result. `smiles_of` maps a species key to a SMILES."""
    from ksmech.network.pools import pool_state
    from ksmech.network.search import DEFAULT_PARAMS
    from ksmech.network.target import target_from_smiles
    from ksmech.network.towards import search_towards

    pool = ".".join(smiles_of[k] for k, n in sorted(route.leaves.items()) for _ in range(n))
    res = search_towards(forward, pool_state(pool), target_from_smiles(target_smiles),
                         params or DEFAULT_PARAMS)  # fmt: skip
    return bool(res.search.pathways), res


def route_rows(route: Route | None) -> list[dict]:
    """A route's steps as JSON-able rows (species made, the parents it comes from, log P)."""
    if route is None:
        return []
    return [{"species": s, "from": list(step.parents), "logp": step.logp,
             **{k: v for k, v in step.info.items() if isinstance(v, str | int | float)}}
            for s, step in route.steps]  # fmt: skip


def leaves_of(route: Route | None) -> dict[str, int]:
    return {} if route is None else dict(sorted(route.leaves.items()))


class ModelRetroExpander:
    """`RetroExpander` over the backward model (module docstring): `backward` an expander of
    pool states (`network.pools.ModelExpander` with a backward checkpoint), `byproducts` the
    reservoir's (key, SMILES, count) rows, `smiles` species key -> a SMILES of it (the target's
    is given, the others are learnt from the parent pools); its skips are counted."""

    def __init__(self, backward, target_smiles: str, byproducts=(), n_byproducts: int = 3):
        from ksmech.network.pools import pool_state

        self.backward, self.byproducts = backward, list(byproducts)[: max(n_byproducts, 0)]
        self.smiles: dict[str, str] = {}
        self.skips: Counter = Counter()
        self.target = self.remember(pool_state(target_smiles))[0]

    def remember(self, state) -> list[str]:
        """The species keys of a pool state, each with a SMILES kept."""
        from ksmech.network.thermo_provider import fragment_records

        keys = []
        for key, record in fragment_records(state.smiles):
            self.smiles.setdefault(key, record)
            keys.append(key)
        return keys

    def _pools(self, species: str):
        """(pool state, map numbers of the by-product or None) of the species alone and with
        each by-product."""
        from ksmech.network.pools import pool_state

        yield pool_state(self.smiles[species]), None
        for key, smi, _ in self.byproducts:
            if key == species:
                self.skips["BYPRODUCT_IS_SPECIES"] += 1
                continue
            pool = pool_state(f"{self.smiles[species]}.{smi}")
            yield pool, byproduct_maps(pool, key)

    def expand(self, species: str) -> list[RetroStep]:
        out = []
        for pool, x_maps in self._pools(species):
            ex = self.backward.expand(pool)
            self.skips.update(ex.skips)
            if ex.failure is not None:
                self.skips[f"EXPAND_FAILED:{ex.failure}"] += 1
            for p in ex.proposals:
                if x_maps is not None and not p.touched & x_maps:
                    self.skips[NOT_USING_BYPRODUCT] += 1
                    continue
                info = {"pool": pool.smiles, "parent_pool": p.child.smiles}
                out.append(RetroStep(tuple(sorted(self.remember(p.child))), p.logp, info))
        return out


def byproduct_maps(pool, key: str) -> frozenset[int]:
    """The map numbers of the (first) fragment of species `key` in a pool state."""
    from rdkit import Chem

    from ksmech.data.chem import parse_mapped
    from ksmech.data.resonance import identity_key

    for frag in Chem.GetMolFrags(parse_mapped(pool.smiles), asMols=True, sanitizeFrags=False):
        if identity_key(frag) == key:
            return frozenset(a.GetAtomMapNum() for a in frag.GetAtoms())
    return frozenset()
