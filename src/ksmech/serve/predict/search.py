"""Search towards a product in the web app (reaction-network plan, item D: what `ksmech-network
search` does in the engine): a predict request with `target_smiles`.

`search_network(chem, root, predictor, params, search, hooks)` runs the engine's best-first
search over simple paths (`ksmech.network.search.k_best_pathways`) with the web app's chemistry
as its expander. The cost of a path is its summed -log beam weight, the weights by which
`network.rank_pathways` ranks every network (log P(R) for a predictor with probabilities, so
each cost is >= 0); there is no heuristic, since a target drawn in the web app carries no atom
map on the reactants (uniform-cost search, still exact). Each node the search pops is expanded
once by `network.expand_node`, its children recorded as the network's edges with no beam cut
beyond the predictor's own `beam` proposals (`max_nodes` still caps the nodes, NODE_BUDGET), so
the job's network is the part of the space the search looked at, in the format every view of
the web app reads. The search stops at `n_pathways` pathways to the target, at
`max_expansions` model expansions (EXPANSION_BUDGET cuts), when nothing is left (EXHAUSTED), or
when the job is cancelled (CANCELLED). Paths are at most `max_steps` steps (DEPTH_CUT).

Arrival (`arrival_level`) is `ksmech.network.target.arrival` on the web app's keys: EXACT when
every target species (the species key, identity key v2) is in the pool with its multiplicity,
by-products allowed; SKELETON when every target species' heavy-atom skeleton
(`target.skeleton_key`: bond orders kept, no H, no charges) is, so a product drawn after
work-up is reached up to proton transfers that move no bond. A pool whose skeletons cannot be
read (SKELETON_UNREADABLE, counted in the network's skips) is tested at EXACT only; a target
whose skeletons cannot be read drops SKELETON. After every change of the network
(`mark_arrivals`, run again by `refresh_pathways` after an expansion or a user step), each node
not yet tested is tested; an arrived node that was not expanded becomes TERMINAL with
terminal_reason TARGET_<level>, and the network's pathways are the paths to arrived nodes
(`network.target_pathways`). Nodes the search left unexpanded become TERMINAL NODE_BUDGET, so
they can be expanded by hand.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ksmech.network.search import Expansion, ImplicitGraph, Proposal, SearchParams, k_best_pathways
from ksmech.serve.chem.pools import SpeciesInput
from ksmech.serve.predict.network import (
    NetHooks,
    NetParams,
    Network,
    NodeStatus,
    Pathway,
    Search,
    add_edge,
    expand_node,
    rank_pathways,
)

if TYPE_CHECKING:
    from ksmech.serve.chem.chemistry import Chemistry
    from ksmech.serve.core.contracts import PoolState, PredictorRef

EXACT, SKELETON = "EXACT", "SKELETON"
LEVELS = (EXACT, SKELETON)
CANCELLED = "CANCELLED"
SKELETON_UNREADABLE = "SKELETON_UNREADABLE"
DEFAULT_MAX_EXPANSIONS = 50
UNEXPANDED = (NodeStatus.FRONTIER, NodeStatus.TERMINAL)  # an arrival may relabel these


def skeletons_of(mapped_smiles: str) -> Counter | None:
    """The heavy-atom skeleton keys of a pool's fragments, None when RDKit cannot read it."""
    from rdkit import rdBase

    from ksmech.data.chem import parse_mapped
    from ksmech.network.target import fragments, skeleton_key

    try:
        with rdBase.BlockLogs():
            return Counter(skeleton_key(f) for f in fragments(parse_mapped(mapped_smiles)))
    except (ValueError, RuntimeError):  # counted by the caller (SKELETON_UNREADABLE)
        return None


def new_search(chem: Chemistry, request: Mapping) -> Search:
    """The `Search` of a predict request: the target pool through the chemistry's own input
    checks (`chem.root`: an invalid target is the same 422 as an invalid reactant) and its
    keys; the levels and the budget of the request."""
    smiles = str(request["target_smiles"]).strip()
    target = chem.root([SpeciesInput(smiles, 1)])
    skeletons = skeletons_of(target.mapped_smiles)
    return Search(
        target_smiles=smiles,
        target_species={sp.species_key: int(sp.count) for sp in target.species},
        target_skeletons=None if skeletons is None else dict(sorted(skeletons.items())),
        levels=[str(lv) for lv in request.get("target_levels") or LEVELS],
        max_expansions=int(request.get("max_expansions") or DEFAULT_MAX_EXPANSIONS),
    )


def _wants_skeleton(search: Search) -> bool:
    return SKELETON in search.levels and search.target_skeletons is not None


def arrival_level(search: Search, state: PoolState, skeletons: Counter | None) -> str | None:
    """The first of the search's levels at which `state` holds the target, None when none."""
    from ksmech.network.target import Target, arrival

    levels = [lv for lv in search.levels if lv != SKELETON or _wants_skeleton(search)]
    target = Target(Counter(search.target_species), Counter(search.target_skeletons or {}),
                    None, None)  # fmt: skip
    species = [sp.species_key for sp in state.species for _ in range(int(sp.count))]
    return arrival(target, species, list((skeletons or Counter()).elements()), levels)


def mark_arrivals(net: Network) -> None:
    """Test every node not tested yet; relabel an arrived node that was not expanded."""
    search = net.search
    assert search is not None
    for node in net.nodes.values():
        if node.id not in search.arrivals:
            skeletons = None
            if _wants_skeleton(search):
                skeletons = skeletons_of(node.state.mapped_smiles)
                if skeletons is None:
                    net.skipped[SKELETON_UNREADABLE] += 1
            search.arrivals[node.id] = arrival_level(search, node.state, skeletons)
        level = search.arrivals[node.id]
        if level and node.status in UNEXPANDED and node.terminal_reason != CANCELLED:
            node.status, node.terminal_reason = NodeStatus.TERMINAL, f"TARGET_{level}"


def refresh_pathways(net: Network) -> list[Pathway]:
    """The network's ranked pathways after a change: the arrivals first when it has a search."""
    if net.search is not None:
        mark_arrivals(net)
    return rank_pathways(net, net.params.n_pathways)


class _Stop(Exception):  # noqa: N818 - unwinds k_best_pathways when the job is cancelled
    pass


@dataclass
class _NetExpander:
    """The engine's `Expander` over the web app's network: key = pool id; one expansion is
    `network.expand_node` on the node, its children recorded as edges (module docstring)."""

    chem: Chemistry
    net: Network
    params: NetParams
    hooks: NetHooks
    budget: int
    done: int = 0

    def key(self, state: PoolState) -> str:
        return state.pool_id

    def expand(self, state: PoolState) -> Expansion:
        net, hooks = self.net, self.hooks
        if hooks.cancelled():
            raise _Stop
        node = net.nodes[state.pool_id]
        if node.status != NodeStatus.FRONTIER:  # cut by max_nodes: never expanded
            return Expansion([], failure=node.terminal_reason or node.status.value.upper())
        hooks.progress(phase="search", done=self.done, total=self.budget, depth=node.depth,
                       max_steps=self.params.max_steps, n_nodes=len(net.nodes),
                       n_edges=len(net.edges), message=f"expanding {node.pool_key}")  # fmt: skip
        kids = expand_node(self.chem, net, node.id, self.params, hooks)
        self.done += 1
        best: dict[str, float] = {}
        for _, ch in kids:
            pid = ch.state.pool_id
            if pid not in net.nodes:
                best[pid] = max(best.get(pid, float("-inf")), float(ch.weight))
        order = sorted(best, key=lambda p: (-best[p], p))
        keep = set(order[: max(0, self.params.max_nodes - len(net.nodes))])
        proposals = []
        for parent, ch in kids:
            add_edge(net, parent, ch, node.best_logp + ch.weight, keep, (), self.params)
            proposals.append(Proposal(ch.state, float(ch.weight)))
        net.pathways = refresh_pathways(net)
        net.touch()
        hooks.snapshot(net)
        return Expansion(proposals)


def _close(net: Network, cancelled: bool) -> None:
    """Every node the search left FRONTIER becomes TERMINAL (NODE_BUDGET, or CANCELLED)."""
    for node in net.nodes.values():
        if node.status == NodeStatus.FRONTIER:
            node.status = NodeStatus.TERMINAL
            node.terminal_reason = CANCELLED if cancelled else "NODE_BUDGET"


def search_network(
    chem: Chemistry, root: PoolState, predictor: PredictorRef, params: NetParams,
    search: Search, hooks: NetHooks,
) -> Network:  # fmt: skip
    """The network of a search from `root` to the search's target (module docstring)."""
    net = Network.new(root, predictor, params, hooks.owner, getattr(chem, "mode", ""))
    net.search = search
    mark_arrivals(net)
    expander = _NetExpander(chem, net, params, hooks, search.max_expansions)
    sp = SearchParams(k=params.n_pathways, max_expansions=search.max_expansions,
                      max_depth=params.max_steps)  # fmt: skip
    try:
        res = k_best_pathways(ImplicitGraph(expander, root),
                              lambda s: search.arrivals.get(s.pool_id), None, sp)  # fmt: skip
        search.reason, search.pops = res.reason, int(res.pops)
        search.cuts = {k: int(v) for k, v in sorted(res.cuts.items())}
    except _Stop:
        search.reason = CANCELLED
    search.expanded = expander.done
    cancelled = hooks.cancelled() or search.reason == CANCELLED
    _close(net, cancelled)
    hooks.progress(phase="pathways", done=expander.done, total=search.max_expansions,
                   n_nodes=len(net.nodes), n_edges=len(net.edges))  # fmt: skip
    net.pathways = refresh_pathways(net)
    net.complete = not cancelled
    net.touch()
    return net
