"""Rebuild multi-step pathways from steps that carry no pathway id (PMechDB via PMechRP).

Step j follows step i when the non-trivial products of i equal the non-trivial reactants of j,
both taken as multisets of map-free canonical SMILES with stereo kept. A species is trivial when
it has at most 3 heavy atoms (water, H3O+, OH-, halides, metal ions; see
`chem.nontrivial_species`). Weakly connected components of these links are pathways. A component
with more than `MAX_PATHWAY_STEPS` steps is cut at its shared intermediates, smallest first,
because common substrates glue unrelated reactions together. A plain chain, with no shared
intermediate, is kept whole however long it is. Steps with no link keep `pathway_id=None`.

The measured coverage on PMechDB is in docs/decisions.md (2026-10-19, "Rebuilding PMechDB
pathways").
"""

from __future__ import annotations

import hashlib
import logging
import statistics
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from functools import cache

from rdkit import Chem

from ksmech.data import chem
from ksmech.data.schema import ElementaryStep

log = logging.getLogger(__name__)

MAX_PATHWAY_STEPS = 15

SideKey = tuple[str, ...]
Keys = dict[str, tuple[SideKey, SideKey]]  # step_id -> (reactant key, product key)
Links = dict[str, list[str]]  # step_id -> step_ids that consume its products


@dataclass
class PathwayReport:
    n_steps: int = 0
    n_empty_reactant_key: int = 0  # only trivial species: nothing can lead to the step
    n_empty_product_key: int = 0  # only trivial species: the step leads nowhere
    n_links: int = 0
    n_reverse_pairs: int = 0  # i -> j and j -> i: the release curates both directions
    n_branching_steps: int = 0  # >1 successor once reverse links are ignored (before cuts)
    n_merging_steps: int = 0  # >1 predecessor once reverse links are ignored (before cuts)
    n_components_cut: int = 0  # components over the size cap that had to be cut
    n_intermediates_cut: int = 0
    n_long_chains_kept: int = 0  # pathways over the cap with no shared intermediate
    pathway_sizes: list[int] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)  # reason code -> steps not keyed

    def as_dict(self) -> dict:
        sizes = self.pathway_sizes
        counts = {k: v for k, v in vars(self).items() if k.startswith("n_")}
        return {
            **counts,
            "n_pathways": len(sizes),
            "n_steps_in_pathways": sum(sizes),
            "fraction_in_pathways": sum(sizes) / self.n_steps if self.n_steps else 0.0,
            "median_pathway_size": statistics.median(sizes) if sizes else 0,
            "max_pathway_size": max(sizes, default=0),
            "pathway_sizes": {str(k): v for k, v in sorted(Counter(sizes).items())},
            "skipped": dict(self.skipped.most_common()),
        }


def side_key(smiles: str) -> SideKey:
    return tuple(chem.nontrivial_species(smiles))


def step_keys(steps: Iterable[ElementaryStep], report: PathwayReport) -> Keys:
    """step_id -> (reactant key, product key). Unkeyable steps are logged, counted, left out."""
    out = {}
    for s in steps:
        try:
            out[s.step_id] = (side_key(s.reactants_smiles), side_key(s.products_smiles))
        except (chem.SmilesParseError, chem.SanitizeError) as exc:
            log.warning("pathway key failed for %s: %s", s.step_id, exc)
            report.skipped["KEY_FAILED"] += 1
    report.n_empty_reactant_key = sum(not r for r, _ in out.values())
    report.n_empty_product_key = sum(not p for _, p in out.values())
    return out


def find_links(keys: Keys) -> Links:
    """step_id i -> sorted step_ids j != i whose reactant key equals the product key of i.
    Empty keys never link: they would join every step made only of trivial species."""
    by_reactants: dict[SideKey, list[str]] = defaultdict(list)
    for sid, (reactants, _) in keys.items():
        if reactants:
            by_reactants[reactants].append(sid)
    links = {}
    for sid, (_, products) in keys.items():
        nxt = sorted(j for j in by_reactants.get(products, ()) if j != sid)
        if nxt:
            links[sid] = nxt
    return links


def predecessors(links: Links) -> Links:
    pred: Links = defaultdict(list)
    for i, js in links.items():
        for j in js:
            pred[j].append(i)
    return {j: sorted(v) for j, v in pred.items()}


def forward_links(links: Links) -> Links:
    """Drop both directions of every reverse pair (j undoes i)."""
    fwd = {i: [j for j in js if i not in links.get(j, ())] for i, js in links.items()}
    return {i: js for i, js in fwd.items() if js}


def components(step_ids: Iterable[str], links: Links) -> list[list[str]]:
    """Weakly connected components, each sorted, ordered by their smallest step_id. Every link
    must stay inside `step_ids`."""
    parent = {s: s for s in step_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, js in links.items():
        for j in js:
            a, b = find(i), find(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups: dict[str, list[str]] = defaultdict(list)
    for s in parent:
        groups[find(s)].append(s)
    return sorted(sorted(g) for g in groups.values())


@cache
def _heavy_atoms(key: SideKey) -> int:
    return sum(Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in key)


def _shared_intermediates(piece: Sequence[str], links: Links, keys: Keys) -> dict[SideKey, int]:
    """Product keys that more than one step of `piece` makes, or more than one consumes, with
    their number of producer x consumer pairs."""
    makers: dict[SideKey, set[str]] = defaultdict(set)
    takers: dict[SideKey, set[str]] = defaultdict(set)
    for i in piece:
        for j in links.get(i, ()):
            makers[keys[i][1]].add(i)
            takers[keys[i][1]].add(j)
    return {
        k: len(m) * len(takers[k]) for k, m in makers.items() if len(m) > 1 or len(takers[k]) > 1
    }


def split_oversize(
    members: Sequence[str], links: Links, keys: Keys, max_steps: int = MAX_PATHWAY_STEPS
) -> tuple[list[list[str]], Links, int]:
    """Cut a component into pieces of at most `max_steps` steps. Each cut removes every link
    through one shared intermediate of an oversize piece: the one with the fewest heavy atoms,
    then the most producer x consumer pairs. A piece with no shared intermediate is a plain
    chain and is kept whole. Returns (pieces, links left inside the component, cuts made)."""
    links = {i: links[i] for i in members if i in links}
    todo, done, cuts = [sorted(members)], [], 0
    while todo:
        piece = todo.pop()
        shared = _shared_intermediates(piece, links, keys) if len(piece) > max_steps else {}
        if not shared:
            done.append(piece)
            continue
        cut = min(shared, key=lambda k: (_heavy_atoms(k), -shared[k], k))
        inside = set(piece)
        links = {i: js for i, js in links.items() if i not in inside or keys[i][1] != cut}
        cuts += 1
        todo += components(piece, {i: js for i, js in links.items() if i in inside})
    return sorted(done), links, cuts


def _roots(nodes: set[str], links: Links, pred: Links, fwd: Links, fwd_pred: Links) -> list[str]:
    """Steps nothing in `nodes` leads to. Failing that (a cycle), steps that start a forward
    chain once reverse links are ignored. Failing that, the smallest step_id."""
    roots = [s for s in nodes if not set(pred.get(s, ())) & nodes]
    if not roots:
        roots = [s for s in nodes if s in fwd and not set(fwd_pred.get(s, ())) & nodes]
    return sorted(roots) or [min(nodes)]


def order_pathway(members: Sequence[str], links: Links) -> dict[str, int]:
    """step_id -> BFS depth from the pathway's start steps. Branches can share a depth.

    `members` must be a whole component of `links`, so no link leaves it."""
    links = {i: links[i] for i in members if i in links}
    pred, fwd = predecessors(links), forward_links(links)
    fwd_pred = predecessors(fwd)
    depth: dict[str, int] = {}
    remaining = set(members)
    while remaining:
        queue = deque(_roots(remaining, links, pred, fwd, fwd_pred))
        for r in queue:
            depth[r] = 0
        while queue:
            cur = queue.popleft()
            for nxt in links.get(cur, ()):
                if nxt in remaining and nxt not in depth:
                    depth[nxt] = depth[cur] + 1
                    queue.append(nxt)
        remaining -= depth.keys()
    return depth


def pathway_id(source: str, first_step_id: str) -> str:
    return f"{source}-path-{hashlib.sha1(first_step_id.encode()).hexdigest()[:16]}"


def _count_links(report: PathwayReport, links: Links) -> None:
    fwd = forward_links(links)
    report.n_links = sum(map(len, links.values()))
    report.n_reverse_pairs = (report.n_links - sum(map(len, fwd.values()))) // 2
    report.n_branching_steps = sum(len(js) > 1 for js in fwd.values())
    report.n_merging_steps = sum(len(v) > 1 for v in predecessors(fwd).values())


def build_pathways(
    keys: Keys, report: PathwayReport, max_steps: int = MAX_PATHWAY_STEPS
) -> list[tuple[list[str], Links]]:
    """Link, group and cut. Returns each pathway (2+ steps) with the links inside it."""
    links = find_links(keys)
    _count_links(report, links)
    out = []
    for comp in components(keys, links):
        if len(comp) == 1:
            continue
        pieces, inner, cuts = split_oversize(comp, links, keys, max_steps)
        report.n_components_cut += cuts > 0
        report.n_intermediates_cut += cuts
        report.n_long_chains_kept += sum(len(p) > max_steps for p in pieces)
        out += [(p, inner) for p in pieces if len(p) > 1]
    report.pathway_sizes = sorted(len(p) for p, _ in out)
    return out


def assign_pathways(
    steps: Sequence[ElementaryStep], max_steps: int = MAX_PATHWAY_STEPS
) -> tuple[list[ElementaryStep], PathwayReport]:
    """Copies of `steps`, in input order, with `pathway_id` / `step_index` filled for every step
    that ends up in a pathway. Steps that already carry a pathway id are left alone."""
    report = PathwayReport(n_steps=len(steps))
    todo = [s for s in steps if s.pathway_id is None]
    if len(todo) < len(steps):
        report.skipped["HAS_PATHWAY"] = len(steps) - len(todo)
    source = {s.step_id: str(s.source) for s in steps}
    assigned: dict[str, tuple[str, int]] = {}
    for members, links in build_pathways(step_keys(todo, report), report, max_steps):
        depth = order_pathway(members, links)
        first = min(members, key=lambda s: (depth[s], s))
        pid = pathway_id(source[first], first)
        assigned.update({s: (pid, depth[s]) for s in members})
    out = [
        replace(s, pathway_id=assigned[s.step_id][0], step_index=assigned[s.step_id][1])
        if s.step_id in assigned
        else s
        for s in steps
    ]
    return out, report
