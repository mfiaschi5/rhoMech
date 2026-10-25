"""The reaction network (design 4.1-4.6): beam expansion over pool states, merging by pool key,
the cycle rule, ranked pathways, overrides, and the byte-deterministic JSON form.

Pure: no file I/O, no RDKit calls, no clock. The only view of the outside is a `Chemistry`
(`expand` / `apply`) and the `NetHooks` (cancel flag, progress, reason-coded skips, snapshots).

- Nodes are keyed by pool id; a node keeps the representative state first reached in this
  network (ties by lower state id in the deterministic BFS order), and other states that reach
  the same pool key are listed as aliases.
- One edge per (source node, child pool key) and kind; predicted edge probabilities are the
  lower-bound marginals of the expansion, `logp = log(probability)`, and a corrected edge
  contributes 0 to a path's log-probability.
- `revision` grows with every mutation (each node expansion, each layer, each override and the
  end of a build), so a client polling with `since_revision` sees every change.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ksmech.serve.contracts import (
    ArrowSpec,
    ExpandRequest,
    Expansion,
    JobContext,
    PoolState,
    PredictorRef,
    ProposedChild,
    to_jsonable,
)
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.keys import edge_id, split_pool_key

if TYPE_CHECKING:
    from ksmech.serve.chemistry import Chemistry

NETWORK_FORMAT = "ksmech-serve/network-v1"
MAX_PATHWAY_POPS = 10_000
CONTINUABLE_TERMINAL = ("MAX_STEPS", "NODE_BUDGET")
MINUS = "−"


class NodeStatus(StrEnum):
    FRONTIER = "frontier"
    EXPANDED = "expanded"
    PRUNED = "pruned"
    TERMINAL = "terminal"
    FAILED = "failed"


def _logp_in(x: Any) -> float:
    return float("-inf") if x is None else float(x)


def _opt_float(x: Any) -> float | None:
    return None if x is None else float(x)


# ---- data model


@dataclass
class Node:
    id: str  # pool_id
    pool_key: str
    state: PoolState  # representative
    alias_state_ids: list[str]
    depth: int  # minimum over arrivals
    best_logp: float  # Viterbi max over arrivals (root 0.0)
    best_parent_edge: str | None
    status: NodeStatus
    terminal_reason: str | None = None
    failure: dict | None = None  # {code, upstream, detail}
    is_root: bool = False
    n_valid: int | None = None
    scanned_mass: float | None = None
    unscanned_mass: float | None = None
    apply_failures: dict[str, int] = field(default_factory=dict)
    fermi_smeared: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "state"}
        d["state"] = self.state.to_dict()
        d["status"] = self.status.value
        return to_jsonable(d)

    @classmethod
    def from_dict(cls, d: Mapping) -> Node:
        return cls(
            id=str(d["id"]),
            pool_key=str(d["pool_key"]),
            state=PoolState.from_dict(d["state"]),
            alias_state_ids=[str(s) for s in d["alias_state_ids"]],
            depth=int(d["depth"]),
            best_logp=_logp_in(d["best_logp"]),
            best_parent_edge=d.get("best_parent_edge"),
            status=NodeStatus(d["status"]),
            terminal_reason=d.get("terminal_reason"),
            failure=d.get("failure"),
            is_root=bool(d.get("is_root", False)),
            n_valid=d.get("n_valid"),
            scanned_mass=_opt_float(d.get("scanned_mass")),
            unscanned_mass=_opt_float(d.get("unscanned_mass")),
            apply_failures={str(k): int(v) for k, v in (d.get("apply_failures") or {}).items()},
            fermi_smeared=bool(d.get("fermi_smeared", False)),
            warnings=[str(w) for w in d.get("warnings") or []],
        )


@dataclass
class Edge:
    id: str
    source: str
    target: str
    kind: str  # "predicted" | "corrected"
    status: str  # "active" | "superseded"
    source_state_id: str
    product_state_id: str
    source_mapped_smiles: str
    product_mapped_smiles: str
    product_smiles: str  # display form: the product's token keys (unmapped)
    arrows: list[dict]
    encoded: dict | None
    encode_reason: str | None
    touched_maps: list[int]
    score: float | None
    probability: float | None
    logp: float
    rank: int | None
    n_alternatives: int = 0
    alternatives: list[dict] = field(default_factory=list)
    predictor: str | None = None
    predictor_version: str | None = None
    original: dict | None = None  # corrected edge: EdgeOriginalOut of the edge it replaced
    superseded_by: str | None = None
    predictor_view: list[dict] | None = None
    recurrence: bool = False  # the target lay on the Viterbi path of the source
    closes_cycle: bool = False
    cycle_ids: list[str] = field(default_factory=list)
    revision: int = 0

    @property
    def path_logp(self) -> float:
        """The edge's term in a path log-probability: a corrected edge contributes 0."""
        return 0.0 if self.kind == "corrected" else self.logp

    def specs(self) -> list[list[int]]:
        return [ArrowSpec.from_dict(a).as_list() for a in self.arrows]

    def to_dict(self) -> dict:
        return to_jsonable({k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: Mapping) -> Edge:
        kw = {k: d.get(k) for k in cls.__dataclass_fields__ if k in d}
        kw["logp"] = _logp_in(d.get("logp"))
        kw["score"] = _opt_float(d.get("score"))
        kw["probability"] = _opt_float(d.get("probability"))
        kw["touched_maps"] = [int(m) for m in d.get("touched_maps") or []]
        kw["cycle_ids"] = [str(c) for c in d.get("cycle_ids") or []]
        return cls(**kw)


@dataclass
class Cycle:
    id: str
    kind: str  # "recurrence" | "turnover"
    anchor: str
    closing_edge: str
    node_ids: list[str]
    edge_ids: list[str]
    catalyst: dict[str, int]
    consumed: dict[str, int]
    produced: dict[str, int]
    virtual_edge: dict | None  # turnover: {"source", "target", "label"}

    def to_dict(self) -> dict:
        return to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, d: Mapping) -> Cycle:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


@dataclass
class Pathway:
    id: str
    rank: int
    node_ids: list[str]
    edge_ids: list[str]
    logp: float
    probability: float
    n_steps: int
    n_corrected: int
    ends_with: str
    closes_cycle: bool
    cycle_ids: list[str]

    def to_dict(self) -> dict:
        return to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, d: Mapping) -> Pathway:
        kw = {k: d[k] for k in cls.__dataclass_fields__}
        kw["logp"] = _logp_in(kw["logp"])
        kw["probability"] = float(kw["probability"] or 0.0)
        return cls(**kw)


@dataclass(frozen=True)
class NetParams:
    beam: int = 5
    max_steps: int = 8
    temperature: float | None = None
    stop_at_turnover: bool = True
    candidate_scope: str = "all"
    max_nodes: int = 200
    n_pathways: int = 20

    @property
    def max_scan(self) -> int:
        return max(64, 8 * self.beam)

    def to_dict(self) -> dict:
        return to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, d: Mapping) -> NetParams:
        kw = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        kw["temperature"] = _opt_float(kw.get("temperature"))
        return cls(**kw)


@dataclass
class Network:
    network_id: str
    revision: int
    root: str
    nodes: dict[str, Node]
    edges: dict[str, Edge]
    cycles: dict[str, Cycle]
    pathways: list[Pathway]
    failed_proposals: list[dict]
    skipped: Counter
    params: NetParams
    predictor: PredictorRef
    chemistry: str
    complete: bool

    @classmethod
    def new(
        cls,
        root: PoolState,
        predictor: PredictorRef,
        params: NetParams,
        network_id: str = "",
        chemistry: str = "",
    ) -> Network:
        node = Node(
            root.pool_id, root.pool_key, root, [], 0, 0.0, None, NodeStatus.FRONTIER, is_root=True
        )
        return cls(
            network_id,
            0,
            node.id,
            {node.id: node},
            {},
            {},
            [],
            [],
            Counter(),
            params,
            predictor,
            chemistry,
            False,
        )

    def touch(self) -> int:
        self.revision += 1
        return self.revision


@dataclass
class NetHooks:
    """The network's only view of the outside: `ctx` is passed on to `chem.expand` /
    `chem.apply`; `snapshot(net)` runs after every node expansion and layer (the job layer
    throttles the writes)."""

    ctx: JobContext
    snapshot: Callable[[Network], None] = field(default=lambda net: None)

    @property
    def owner(self) -> str:
        return self.ctx.job_id

    def cancelled(self) -> bool:
        return bool(self.ctx.cancelled())

    def progress(self, **fields: Any) -> None:
        self.ctx.progress(**fields)

    def skip(self, code: str, detail: str = "", node: str | None = None, **where: Any) -> None:
        self.ctx.skip(str(code), detail, node=node, **where)


@dataclass
class _Run:
    """Book-keeping of one build or continuation (never serialized)."""

    total: int
    seen: set[str] = field(default_factory=set)  # nodes that were frontier in this run
    done: int = 0


def _skip(net: Network, hooks: NetHooks, code: str, detail: str, node: str | None, n: int = 1):
    net.skipped[str(code)] += int(n)
    if n == 1:
        hooks.skip(code, detail, node)
    else:
        hooks.skip(code, detail, node, count=int(n))


def display_smiles(state: PoolState) -> str:
    """The display SMILES of a pool: every species' first token key (unmapped, aromatic),
    repeated by its count. Species keys are not used: they may be charge-separated forms."""
    parts = []
    for sp in state.species:
        token = sp.token_keys[0] if sp.token_keys else sp.species_key
        parts += [token] * int(sp.count)
    return ".".join(parts)


# ---- beam expansion (4.2)


def build_network(
    chem: Chemistry, root: PoolState, predictor: PredictorRef, params: NetParams, hooks: NetHooks
) -> Network:
    """Expand from `root`, one beam-pruned layer per step, up to `params.max_steps` layers."""
    net = Network.new(root, predictor, params, hooks.owner, getattr(chem, "mode", ""))
    run = _Run(total=1 + params.beam * max(0, params.max_steps - 1))
    _run_layers(chem, net, [net.root], params.max_steps, params, hooks, run)
    net.complete = not hooks.cancelled()
    net.touch()
    return net


def continue_from(
    chem: Chemistry, net: Network, node_id: str, steps: int, params: NetParams, hooks: NetHooks
) -> None:
    """Run the build loop from one node (an override target) for `steps` layers."""
    node = net.nodes[node_id]
    node.status, node.terminal_reason = NodeStatus.FRONTIER, None
    run = _Run(total=1 + params.beam * max(0, steps - 1))
    _run_layers(chem, net, [node_id], steps, params, hooks, run)
    net.touch()


def _run_layers(chem, net, frontier, steps, params, hooks, run: _Run) -> None:
    run.seen.update(frontier)
    for depth in range(int(steps)):
        if not frontier or hooks.cancelled():
            break
        frontier = expand_layer(chem, net, frontier, params, hooks, run, depth)
        run.seen.update(frontier)
    finish(net, run.seen, hooks)
    hooks.progress(phase="pathways", n_nodes=len(net.nodes), n_edges=len(net.edges))
    net.pathways = rank_pathways(net, params.n_pathways)


def expand_layer(chem, net, frontier, params, hooks, run: _Run, depth: int = 0) -> list[str]:
    proposals: list[tuple[str, ProposedChild]] = []
    order = sorted(frontier, key=lambda n: (-net.nodes[n].best_logp, n))
    for node_id in order:
        if hooks.cancelled():
            break
        hooks.progress(
            phase="expand",
            done=run.done,
            total=max(run.total, run.done + 1),
            depth=depth,
            max_steps=params.max_steps,
            n_nodes=len(net.nodes),
            n_edges=len(net.edges),
            frontier=len(order),
            message=f"expanding {net.nodes[node_id].pool_key}",
        )
        proposals += expand_node(chem, net, node_id, params, hooks)
        run.done += 1
        net.touch()
        hooks.snapshot(net)
    nxt = select_beam(net, proposals, params, hooks)
    net.touch()
    hooks.snapshot(net)
    return nxt


def _record_stats(net: Network, node: Node, ex: Expansion) -> None:
    node.n_valid = int(ex.n_valid)
    node.scanned_mass = float(ex.scanned_mass)
    node.unscanned_mass = float(ex.unscanned_mass)
    node.apply_failures = {str(k): int(v) for k, v in sorted(ex.failures.items())}
    node.fermi_smeared = bool(ex.fermi_smeared)
    node.warnings = [str(w) for w in ex.warnings]
    for example in ex.failed_examples:
        net.failed_proposals.append({"node": node.id, **example})


def expand_node(chem, net, node_id, params, hooks) -> list[tuple[str, ProposedChild]]:
    node = net.nodes[node_id]
    req = ExpandRequest(
        net.predictor.name,
        params.beam,
        net.predictor.temperature,
        params.candidate_scope,
        params.max_scan,
    )
    ex = chem.expand(node.state, req, hooks.ctx)
    if not ex.ok:
        node.failure = {"code": ex.code, "upstream": ex.upstream, "detail": ex.detail}
        if ex.code == Code.CANCELLED or hooks.cancelled():
            node.status, node.terminal_reason = NodeStatus.TERMINAL, str(Code.CANCELLED)
        else:
            node.status = NodeStatus.FAILED
        _skip(net, hooks, str(ex.code or Code.UNEXPECTED), ex.detail, node_id)
        return []
    _record_stats(net, node, ex)
    for code, n in sorted(ex.failures.items()):
        _skip(net, hooks, code, "proposals that gave no child", node_id, n)
    if not ex.children:
        node.status = NodeStatus.TERMINAL
        node.terminal_reason = "NO_CANDIDATES" if ex.n_valid == 0 else "NO_APPLICABLE_PROPOSAL"
        return []
    node.status = NodeStatus.EXPANDED
    out = []
    for ch in ex.children:
        if ch.state.pool_id == node_id:  # the expansion should have dropped it
            _skip(
                net, hooks, str(Code.RESONANCE_ONLY_STEP), "child pool equals its parent", node_id
            )
            continue
        out.append((node_id, ch))
    return out


def select_beam(net, proposals, params, hooks) -> list[str]:
    """Keep the `beam` best new pools of the layer (by parent best_logp + child logp); record
    every edge, merges included. Returns the next frontier."""
    scored = sorted(
        ((net.nodes[p].best_logp + ch.logp, p, ch) for p, ch in proposals),
        key=lambda x: (-x[0], x[1], x[2].state.pool_id),
    )
    new_best: dict[str, float] = {}
    for logp, _, ch in scored:
        pid = ch.state.pool_id
        if pid not in net.nodes and pid not in new_best:
            new_best[pid] = logp
    order = sorted(new_best, key=lambda c: (-new_best[c], c))
    budget = max(0, params.max_nodes - len(net.nodes))
    keep = set(order[: min(params.beam, budget)])
    beam_cut = set(order[params.beam :])  # outside the beam: PRUNED; else the budget cut it
    nxt = []
    for logp, parent, ch in scored:
        created = add_edge(net, parent, ch, logp, keep, beam_cut, params)
        pid = ch.state.pool_id
        if created and pid in keep and net.nodes[pid].status == NodeStatus.FRONTIER:
            nxt.append(pid)
    return nxt


def _predicted_edge(net: Network, parent: Node, ch: ProposedChild) -> Edge:
    return Edge(
        id=edge_id("predicted", parent.state.state_id, [ArrowSpec.from_dict(a) for a in ch.arrows]),
        source=parent.id,
        target=ch.state.pool_id,
        kind="predicted",
        status="active",
        source_state_id=parent.state.state_id,
        product_state_id=ch.state.state_id,
        source_mapped_smiles=parent.state.mapped_smiles,
        product_mapped_smiles=ch.state.mapped_smiles,
        product_smiles=display_smiles(ch.state),
        arrows=list(ch.arrows),
        encoded=ch.encoded,
        encode_reason=ch.encode_reason,
        touched_maps=[int(m) for m in ch.touched_maps],
        score=float(ch.score),
        probability=float(ch.probability),
        logp=float(ch.logp),
        rank=int(ch.rank),
        n_alternatives=int(ch.n_alternatives),
        alternatives=list(ch.alternatives),
        predictor=net.predictor.name,
        predictor_version=net.predictor.version,
        revision=net.revision,
    )


def add_edge(net, parent_id, ch, logp, keep, beam_cut: Collection[str] = (), params=None) -> bool:
    """Record the predicted edge parent -> child (idempotent by edge id). Returns True when
    it created the child node."""
    parent = net.nodes[parent_id]
    e = _predicted_edge(net, parent, ch)
    if e.id in net.edges:
        return False
    net.edges[e.id] = e
    created = _attach(net, parent, e, ch.state, logp)
    if created:
        child = net.nodes[e.target]
        if e.target not in keep:
            if e.target in beam_cut:
                child.status = NodeStatus.PRUNED
            else:
                child.status, child.terminal_reason = NodeStatus.TERMINAL, "NODE_BUDGET"
    stop = True if params is None else params.stop_at_turnover
    mark_cycles(net, e, created, stop)
    return created


def _attach(net: Network, parent: Node, e: Edge, state: PoolState, logp: float) -> bool:
    """Merge the edge's target by pool key: a new node (FRONTIER), or depth / alias /
    Viterbi updates of the existing one (never re-expanded). Returns True when created."""
    node = net.nodes.get(e.target)
    if node is None:
        net.nodes[e.target] = Node(
            e.target,
            state.pool_key,
            state,
            [],
            parent.depth + 1,
            logp,
            e.id,
            NodeStatus.FRONTIER,
        )
        return True
    node.depth = min(node.depth, parent.depth + 1)
    if state.state_id != node.state.state_id and state.state_id not in node.alias_state_ids:
        node.alias_state_ids.append(state.state_id)
    if logp > node.best_logp:
        node.best_logp, node.best_parent_edge = logp, e.id
    return False


def finish(net: Network, seen: set[str] | Sequence[str], hooks: NetHooks) -> None:
    """Every node of this run still FRONTIER becomes TERMINAL(MAX_STEPS), or CANCELLED."""
    reason = str(Code.CANCELLED) if hooks.cancelled() else str(Code.MAX_STEPS)
    for node_id in sorted(seen):
        node = net.nodes.get(node_id)
        if node is not None and node.status == NodeStatus.FRONTIER:
            node.status, node.terminal_reason = NodeStatus.TERMINAL, reason


# ---- cycle rule (4.4)


def viterbi_path(net: Network, node_id: str) -> list[str]:
    """Root ... node along `best_parent_edge` pointers."""
    out, seen = [node_id], {node_id}
    cur = net.nodes[node_id]
    while cur.best_parent_edge is not None:
        src = net.edges[cur.best_parent_edge].source
        if src in seen:  # cannot happen (pointers move only on strict increases)
            raise ServeError(Code.CONSISTENCY_FAILED, f"best-parent loop at {src}", status=500)
        out.append(src)
        seen.add(src)
        cur = net.nodes[src]
    return out[::-1]


def _path_edges(net: Network, pi: list[str], start: int, closing: Edge) -> list[str]:
    """Pointer edges of pi[start+1 .. -2], then the closing edge."""
    return [net.nodes[n].best_parent_edge for n in pi[start + 1 : -1]] + [closing.id]


def _cycle_id(kind: str, anchor: str, closing: str) -> str:
    return "y" + hashlib.sha256(f"{kind}|{anchor}|{closing}".encode()).hexdigest()[:12]


def _signed(c: Counter, sign: str) -> str:
    return " ".join(f"{sign}{n} {x}" if n > 1 else f"{sign}{x}" for x, n in sorted(c.items()))


def _catalyst(pools: list[Counter], j: int, inter: Counter) -> dict[str, int]:
    between = pools[j + 1 : -1]
    return {x: n for x, n in sorted(inter.items()) if min(p[x] for p in between) < n}


def detect_closure(net: Network, e: Edge) -> Cycle | None:
    """The cycle edge e = (u -> v) closes, if any: a recurrence (v on the Viterbi path of u) or
    the latest turnover anchor w (design 4.4). Pure: nothing is mutated."""
    pi = viterbi_path(net, e.source) + [e.target]
    if e.target in pi[:-1]:
        k = pi.index(e.target)
        return Cycle(
            _cycle_id("recurrence", e.target, e.id),
            "recurrence",
            e.target,
            e.id,
            pi[k:-1],
            _path_edges(net, pi, k, e),
            {},
            {},
            {},
            None,
        )
    pools = [split_pool_key(net.nodes[n].pool_key) for n in pi]
    root_support = set(split_pool_key(net.nodes[net.root].pool_key))
    p_v = pools[-1]
    for j in range(len(pi) - 3, -1, -1):
        p_w = pools[j]
        consumed, produced = p_w - p_v, p_v - p_w
        if not consumed or not produced or not set(consumed) <= root_support:
            continue
        catalyst = _catalyst(pools, j, p_w & p_v)
        if not catalyst:
            continue
        label = f"{_signed(produced, MINUS)} {_signed(consumed, '+')}"
        return Cycle(
            _cycle_id("turnover", pi[j], e.id),
            "turnover",
            pi[j],
            e.id,
            pi[j:],
            _path_edges(net, pi, j, e),
            catalyst,
            dict(sorted(consumed.items())),
            dict(sorted(produced.items())),
            {"source": e.target, "target": pi[j], "label": label},
        )
    return None


def mark_cycles(net: Network, e: Edge, created: bool, stop_at_turnover: bool = True) -> None:
    cyc = detect_closure(net, e)
    if cyc is None:
        return
    net.cycles[cyc.id] = cyc
    if cyc.id not in e.cycle_ids:
        e.cycle_ids.append(cyc.id)
    if cyc.kind == "recurrence":
        e.recurrence = True
        return
    e.closes_cycle = True
    if stop_at_turnover and created:
        node = net.nodes[e.target]
        node.status, node.terminal_reason = NodeStatus.TERMINAL, str(Code.TURNOVER)


# ---- ranked pathways (4.5)


def _active_out(net: Network, include_pruned: bool) -> dict[str, list[Edge]]:
    out: dict[str, list[Edge]] = {n: [] for n in net.nodes}
    for eid in sorted(net.edges):
        e = net.edges[eid]
        if e.status != "active":  # a superseded edge of either kind is withdrawn
            continue
        if net.nodes[e.target].status == NodeStatus.PRUNED and not include_pruned:
            continue
        out[e.source].append(e)
    return out


def _admissible(out: dict[str, list[Edge]], x: str, on_path: Sequence[str]) -> list[Edge]:
    return [e for e in out[x] if e.recurrence or e.target not in on_path]


def _ends_with(net: Network, out, e: Edge, nodes: Sequence[str]) -> str | None:
    """How a path whose last edge is e ends, or None when it continues."""
    if e.recurrence:
        return "RECURRENCE"
    if e.closes_cycle:
        return "TURNOVER"
    x = net.nodes[e.target]
    if x.status == NodeStatus.TERMINAL:
        return x.terminal_reason or "TERMINAL"
    if x.status == NodeStatus.FAILED:
        return f"FAILED:{(x.failure or {}).get('code') or Code.UNEXPECTED}"
    if x.status in (NodeStatus.FRONTIER, NodeStatus.PRUNED):
        return x.status.value.upper()
    if not _admissible(out, x.id, nodes):
        return x.terminal_reason or x.status.value.upper()
    return None


def _pathway(net: Network, rank: int, nodes, eids, logp: float, ends: str) -> Pathway:
    last = net.edges[eids[-1]]
    n_corrected = sum(net.edges[i].kind == "corrected" for i in eids)
    return Pathway(
        "w" + hashlib.sha256("|".join(eids).encode()).hexdigest()[:12],
        rank,
        list(nodes),
        list(eids),
        logp,
        math.exp(logp) if logp > float("-inf") else 0.0,
        len(eids),
        n_corrected,
        ends,
        ends in ("TURNOVER", "RECURRENCE"),
        list(last.cycle_ids),
    )


def rank_pathways(net: Network, limit: int = 20, include_pruned: bool = False) -> list[Pathway]:
    """Best-first simple paths from the root over active edges (a final recurrence edge may
    revisit one node), by log-probability desc, then fewer edges, then edge ids."""
    out = _active_out(net, include_pruned)
    heap: list[tuple] = []

    def push(nodes: tuple[str, ...], eids: tuple[str, ...], logp: float, e: Edge) -> None:
        nodes2, eids2, logp2 = nodes + (e.target,), eids + (e.id,), logp + e.path_logp
        heapq.heappush(heap, (-logp2, len(eids2), eids2, nodes2, _ends_with(net, out, e, nodes2)))

    for e in out[net.root]:
        push((net.root,), (), 0.0, e)
    result: list[Pathway] = []
    pops = 0
    while heap and len(result) < limit and pops < MAX_PATHWAY_POPS:
        neg, _, eids, nodes, ends = heapq.heappop(heap)
        pops += 1
        if ends is not None:
            result.append(_pathway(net, len(result) + 1, nodes, eids, -neg, ends))
            continue
        for e in _admissible(out, nodes[-1], nodes):
            push(nodes, eids, -neg, e)
    return result


# ---- overrides (4.6)


def _source_of(net: Network, req: Mapping) -> tuple[Node, Edge | None]:
    eid, nid = req.get("edge_id"), req.get("node_id")
    if (eid is None) == (nid is None):
        raise ServeError(
            Code.REQUEST_INVALID, "give exactly one of edge_id and node_id", status=422
        )
    if eid is not None:
        e = net.edges.get(str(eid))
        if e is None:
            raise ServeError(Code.EDGE_NOT_FOUND, f"no edge {eid} in this network", status=404)
        node = net.nodes[e.source]
        if node.state.state_id != e.source_state_id:
            msg = f"edge {eid} leaves {e.source_state_id}, not the representative of {node.id}"
            raise ServeError(Code.CONSISTENCY_FAILED, msg, status=500)
        return node, e
    node = net.nodes.get(str(nid))
    if node is None:
        raise ServeError(Code.NODE_NOT_FOUND, f"no node {nid} in this network", status=404)
    return node, None


def edge_original(e: Edge, net: Network) -> dict:
    """`EdgeOriginalOut` fields of an edge."""
    return {
        "edge_id": e.id,
        "target": e.target,
        "target_pool_key": net.nodes[e.target].pool_key,
        "arrows": list(e.arrows),
        "score": e.score,
        "probability": e.probability,
        "rank": e.rank,
        "predictor": e.predictor,
        "predictor_version": e.predictor_version,
    }


def _original_of(old: Edge | None, net: Network) -> dict | None:
    """The comparison baseline of a correction of `old`: the prediction `old` replaced when
    `old` is itself a correction of one (a correction of a correction keeps the predictor's
    step), else `old` itself."""
    if old is None:
        return None
    if old.kind == "corrected" and old.original is not None:
        return dict(old.original)
    return edge_original(old, net)


def _corrected_edge(net: Network, p: Node, out, arrows, old: Edge | None) -> Edge:
    return Edge(
        id=edge_id("corrected", p.state.state_id, arrows),
        source=p.id,
        target=out.state.pool_id,
        kind="corrected",
        status="active",
        source_state_id=p.state.state_id,
        product_state_id=out.state.state_id,
        source_mapped_smiles=p.state.mapped_smiles,
        product_mapped_smiles=out.state.mapped_smiles,
        product_smiles=display_smiles(out.state),
        arrows=list(out.arrows),
        encoded=out.encoded,
        encode_reason=out.encode_reason,
        touched_maps=[int(m) for m in out.touched_maps],
        score=None,
        probability=None,
        logp=0.0,
        rank=None,
        predictor=net.predictor.name,
        predictor_version=net.predictor.version,
        original=_original_of(old, net),
        predictor_view=list(out.predictor_view),
        revision=net.revision + 1,
    )


def _record_correction(net: Network, p: Node, out, arrows, old: Edge | None) -> tuple[str, bool]:
    """Add the corrected edge (or reuse the identical one), supersede `old`; returns
    (corrected edge id, created the target)."""
    ce = _corrected_edge(net, p, out, arrows, old)
    existing = net.edges.get(ce.id)
    if existing is not None and (old is None or old.superseded_by == ce.id or old.id == ce.id):
        raise ServeError(Code.OVERRIDE_NOOP, f"this correction already exists: {ce.id}")
    if old is not None:
        old.status, old.superseded_by = "superseded", ce.id
    if existing is not None:
        return ce.id, False
    net.edges[ce.id] = ce
    created = _attach(net, p, ce, out.state, p.best_logp)
    mark_cycles(net, ce, created, net.params.stop_at_turnover)
    return ce.id, created


def _continuable(node: Node, created: bool) -> bool:
    """Whether an override target may be expanded further. A target the correction just
    marked TERMINAL(TURNOVER) (it closes a catalytic cycle, stop_at_turnover) never is."""
    if node.status == NodeStatus.TERMINAL and node.terminal_reason == str(Code.TURNOVER):
        return False
    if created or node.status in (NodeStatus.FRONTIER, NodeStatus.PRUNED):
        return True
    return node.status == NodeStatus.TERMINAL and node.terminal_reason in CONTINUABLE_TERMINAL


def apply_override(
    chem: Chemistry, net: Network, req: Mapping, params: NetParams, hooks: NetHooks
) -> dict:
    """Apply the user's arrows at an edge's source (or a node), record the corrected edge
    (the original is kept, superseded), merge its target by pool key and continue prediction
    from it if it has not been expanded. Returns the `OverrideResultOut` fields."""
    p, old = _source_of(net, req)
    arrows = [ArrowSpec.from_dict(a) for a in req["arrows"]]
    hooks.progress(phase="apply", message=f"applying {len(arrows)} arrow(s) at {p.pool_key}")
    out = chem.apply(
        p.state,
        arrows,
        bool(req.get("allow_invalid", False)),
        net.predictor.name,
        net.params.temperature,
        hooks.ctx,
    )
    specs = [a.as_list() for a in arrows]
    if old is not None and old.target == out.state.pool_id and old.specs() == specs:
        raise ServeError(Code.OVERRIDE_NOOP, f"edge {old.id} already applies these arrows")
    cid, created = _record_correction(net, p, out, arrows, old)
    net.touch()
    hooks.snapshot(net)
    target = net.nodes[net.edges[cid].target]
    continued = None
    if req.get("continue_prediction", True) and _continuable(target, created):
        given = req.get("continue_steps")
        steps = int(given) if given is not None else max(1, params.max_steps - target.depth)
        if steps > 0:
            continue_from(chem, net, target.id, steps, params, hooks)
            continued = target.id
    net.touch()
    net.pathways = rank_pathways(net, params.n_pathways)
    return {
        "parent_job_id": net.network_id,
        "network_revision": net.revision,
        "corrected_edge_id": cid,
        "target_node": target.id,
        "continued_from": continued,
        "issues": list(out.issues),
    }


# ---- JSON form


def network_to_dict(net: Network) -> dict:
    """The network as JSON-able data (no timestamps; nodes and edges in insertion order), so
    `json.dumps(network_to_dict(net), sort_keys=True)` is byte-deterministic."""
    return {
        "format": NETWORK_FORMAT,
        "network_id": net.network_id,
        "revision": int(net.revision),
        "root": net.root,
        "complete": bool(net.complete),
        "chemistry": net.chemistry,
        "params": net.params.to_dict(),
        "predictor": net.predictor.to_dict(),
        "nodes": [n.to_dict() for n in net.nodes.values()],
        "edges": [e.to_dict() for e in net.edges.values()],
        "cycles": [c.to_dict() for c in net.cycles.values()],
        "pathways": [p.to_dict() for p in net.pathways],
        "failed_proposals": to_jsonable(net.failed_proposals),
        "skipped": {k: int(v) for k, v in sorted(net.skipped.items())},
    }


def network_from_dict(d: Mapping) -> Network:
    if d.get("format", NETWORK_FORMAT) != NETWORK_FORMAT:
        raise ValueError(f"unknown network format {d.get('format')!r}")
    nodes = [Node.from_dict(n) for n in d["nodes"]]
    edges = [Edge.from_dict(e) for e in d["edges"]]
    cycles = [Cycle.from_dict(c) for c in d["cycles"]]
    return Network(
        network_id=str(d["network_id"]),
        revision=int(d["revision"]),
        root=str(d["root"]),
        nodes={n.id: n for n in nodes},
        edges={e.id: e for e in edges},
        cycles={c.id: c for c in cycles},
        pathways=[Pathway.from_dict(p) for p in d["pathways"]],
        failed_proposals=list(d.get("failed_proposals") or []),
        skipped=Counter({str(k): int(v) for k, v in (d.get("skipped") or {}).items()}),
        params=NetParams.from_dict(d["params"]),
        predictor=PredictorRef.from_dict(d["predictor"]),
        chemistry=str(d.get("chemistry") or ""),
        complete=bool(d.get("complete", False)),
    )
