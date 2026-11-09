"""Between pool states and species (reaction-network plan, item J): the species graph's pools
are built from species drawings, the pool expander's steps are split into species steps, and a
pool-graph pathway is lifted onto the species graph to be compared with a hyperpath.

A species' atoms are numbered by `local_ranks`: RDKit's canonical ranks, ties broken, of its
sigma skeleton with every H an atom (every bond single, no charge, radical, aromatic flag,
stereo or map number). Two drawings with one identity key v2 have one sigma skeleton with the
same H, so they get the same labelled graph up to its automorphisms, whatever the resonance form
or the atom order; a species' drawing (`Species.smiles`) carries these indices + 1 as its map
numbers, and every step's change and atom map are written on them.

- `pool_of`: one molecule of each species given, in key order (the model's output depends on
  the order of a pool's fragments: the same pool drawn in another order gets other log P), map
  numbers offset species by species, as a `network.pools.PoolState`;
- `split_proposal`: a proposal on a pool as a species step: its reactants are the fragments it
  touches, its products the fragments their atoms end in (the others are spectators), its change
  and atom map on (slot, local atom); POOL_INVALID, NO_STEP_INFO, ATOMS_NOT_CONSERVED and
  STEP_OUTSIDE_POOL when it cannot be split;
- `ModelPoolReactor`: `species.PoolReactor` over a pool `Expander` whose proposals carry their
  `bonds` and `lone` on map numbers (`network.pools.ModelExpander`);
- `start_of`, `start_numbering`, `target_slots`: the starting species of a pool, the map numbers
  of their molecules by local atom (for `hyperpaths.linearize` on a start pool's numbers), and a
  target's slots for `hyperpaths.hypergraph_of`;
- `lift_pathway` and `compare`: a pool-graph pathway (`search.Pathway.edges`) as species steps,
  and the comparison of a hyperpath with it: the same species class (the multiset of steps by
  reactants and products), and the same `search.mechanism_class` once the hyperpath is
  linearized on the start pool's map numbers. The second can differ where the two graphs chose
  different atoms among symmetry-equivalent ones (two H of one methyl), which neither graph
  canonicalizes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from rdkit import Chem, rdBase

from ksmech.data.chem import parse_mapped
from ksmech.data.resonance import identity_key
from ksmech.network.hyperpaths import Hyperpath, linearize, species_class, unfold
from ksmech.network.pools import PoolState, pool_state
from ksmech.network.search import Edge, Proposal, mechanism_class
from ksmech.network.species import PoolResult, Species, Step
from ksmech.network.target import skeleton_key

POOL_INVALID, NO_STEP_INFO = "POOL_INVALID", "NO_STEP_INFO"
ATOMS_NOT_CONSERVED, STEP_OUTSIDE_POOL = "ATOMS_NOT_CONSERVED", "STEP_OUTSIDE_POOL"


def local_ranks(frag: Chem.Mol) -> list[int]:
    """The canonical local index of every atom of a fragment (module docstring)."""
    m = Chem.RWMol(frag)
    for atom in m.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetIsAromatic(False)
        atom.SetAtomMapNum(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in m.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    m.UpdatePropertyCache(strict=False)
    return list(Chem.CanonicalRankAtoms(m, breakTies=True, includeChirality=False))


def species_of(frag: Chem.Mol) -> tuple[Species, list[int]]:
    """(the species record of an explicit-H fragment, the local index of each of its atoms)."""
    ranks = local_ranks(frag)
    drawing = Chem.RWMol(frag)
    elements = [0] * len(ranks)
    for atom, r in zip(drawing.GetAtoms(), ranks, strict=True):
        atom.SetAtomMapNum(r + 1)
        elements[r] = atom.GetAtomicNum()
    with rdBase.BlockLogs():
        smiles = Chem.MolToSmiles(drawing, canonical=True)
        key = identity_key(frag)
    sp = Species(key, len(ranks), smiles, skeleton_key(frag), tuple(elements))
    return sp, ranks


def mapped_fragments(mol: Chem.Mol) -> list[tuple[tuple[int, ...], Chem.Mol]]:
    """(map numbers of its atoms in order, fragment) of every fragment of a mapped molecule."""
    with rdBase.BlockLogs():
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return [(tuple(a.GetAtomMapNum() for a in f.GetAtoms()), f) for f in frags]


def pool_of(species: Sequence[Species]) -> PoolState:
    """One molecule of each species given, in key order (the model's output depends on the
    order of a pool's fragments, so one multiset is always the same pool), map numbers offset
    species by species."""
    parts, offset = [], 0
    for sp in sorted(species, key=lambda s: s.key):
        mol = parse_mapped(sp.smiles)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetAtomMapNum() + offset)
        offset += mol.GetNumAtoms()
        with rdBase.BlockLogs():
            parts.append(Chem.MolToSmiles(mol, canonical=True))
    return pool_state(".".join(parts))


def _side(fragments) -> tuple[list[Species], list[dict[int, int]]]:
    """The species of the fragments sorted by (key, smallest map number), with each one's
    map number -> local index."""
    rows = []
    for maps, frag in fragments:
        sp, ranks = species_of(frag)
        rows.append((sp.key, min(maps), sp, dict(zip(maps, ranks, strict=True))))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [r[2] for r in rows], [r[3] for r in rows]


def _split_parts(parent: PoolState, proposal: Proposal):
    pmol, cmol = parse_mapped(parent.smiles), parse_mapped(proposal.child.smiles)
    touched = set(proposal.touched)
    reac = [(m, f) for m, f in mapped_fragments(pmol) if touched & set(m)]
    atoms = {x for m, _ in reac for x in m}
    prod = [(m, f) for m, f in mapped_fragments(cmol) if atoms & set(m)]
    if {x for m, _ in prod for x in m} != atoms:
        return None
    return _side(reac), _side(prod)


def split_proposal(
    parent: PoolState, proposal: Proposal
) -> tuple[Step | None, dict[str, Species], str | None]:
    """(the species step, its product species' records, None) or (None, {}, the reason)."""
    info = proposal.info
    if "bonds" not in info or "lone" not in info:
        return None, {}, NO_STEP_INFO
    parts = _split_parts(parent, proposal)
    if parts is None:
        return None, {}, ATOMS_NOT_CONSERVED
    (rs, rmaps), (ps, pmaps) = parts
    where = {m: (i, a) for i, local in enumerate(rmaps) for m, a in local.items()}
    if any(m not in where for b in info["bonds"] for m in b[:2]) or any(
        m not in where for m, _ in info["lone"]
    ):
        return None, {}, STEP_OUTSIDE_POOL
    atom_map = tuple(
        tuple(where[m] for m, _ in sorted(local.items(), key=lambda x: x[1])) for local in pmaps
    )
    bonds = tuple(sorted((*min(where[m], where[n]), *max(where[m], where[n]), int(d))
                         for m, n, d in info["bonds"]))  # fmt: skip
    lone = tuple(sorted((*where[m], int(d)) for m, d in info["lone"]))
    step = Step(tuple(s.key for s in rs), tuple(s.key for s in ps), proposal.logp, atom_map,
                bonds, lone)  # fmt: skip
    return step, {s.key: s for s in ps}, None


class ModelPoolReactor:
    """`species.PoolReactor` over a pool `Expander` (module docstring)."""

    def __init__(self, expander) -> None:
        self.expander = expander

    def react(self, pool: Sequence[Species]) -> PoolResult:
        from ksmech.data.chem import SanitizeError, SmilesParseError

        try:
            state = pool_of(pool)
        except (SmilesParseError, SanitizeError):
            return PoolResult([], failure=POOL_INVALID)
        ex = self.expander.expand(state)
        out = PoolResult([], {}, Counter(ex.skips), ex.failure)
        for p in ex.proposals:
            step, species, why = split_proposal(state, p)
            if why is not None:
                out.skips[f"SPLIT:{why}"] += 1
                continue
            out.steps.append(step)
            out.species.update(species)
        return out


def start_of(state: PoolState) -> tuple[dict[str, Species], Counter]:
    """(the species records of a pool's fragments, how many molecules of each)."""
    records: dict[str, Species] = {}
    amounts: Counter = Counter()
    for _, frag in mapped_fragments(parse_mapped(state.smiles)):
        sp, _ = species_of(frag)
        records.setdefault(sp.key, sp)
        amounts[sp.key] += 1
    return records, amounts


def start_numbering(state: PoolState) -> dict[str, list[tuple[int, ...]]]:
    """species key -> the map numbers of each of its molecules in the pool by local atom, the
    molecules by their smallest map number."""
    rows = []
    for maps, frag in mapped_fragments(parse_mapped(state.smiles)):
        sp, ranks = species_of(frag)
        by_rank = sorted(zip(ranks, maps, strict=True))
        rows.append((sp.key, min(maps), tuple(m for _, m in by_rank)))
    out: dict[str, list[tuple[int, ...]]] = {}
    for key, _, nums in sorted(rows):
        out.setdefault(key, []).append(nums)
    return out


def target_slots(smiles: str) -> tuple[tuple[str, str], ...]:
    """(identity key, skeleton key) of every fragment of a target SMILES, as
    `network.target.target_from_smiles` reads it."""
    from ksmech.network.target import fragments

    with rdBase.BlockLogs():
        frags = fragments(parse_mapped(smiles))
        return tuple(sorted((identity_key(f), skeleton_key(f)) for f in frags))


def lift_pathway(start: PoolState, edges: Sequence[Edge]) -> tuple[list[Step], str | None]:
    """The species steps of a pool-graph pathway from `start`, or those before the first edge
    that cannot be split and its reason."""
    steps, parent = [], start
    for e in edges:
        step, _, why = split_proposal(parent, e.proposal)
        if why is not None:
            return steps, why
        steps.append(step)
        parent = e.proposal.child
    return steps, None


@dataclass(frozen=True)
class ClassComparison:
    same_species_class: bool
    same_mechanism_class: bool
    hyperpath_class: tuple
    pathway_class: tuple
    failure: str | None = None  # the pathway could not be lifted


def compare(hp: Hyperpath, edges: Sequence[Edge], start: PoolState) -> ClassComparison:
    """A hyperpath against a pool-graph pathway from `start` (module docstring)."""
    steps, why = lift_pathway(start, edges)
    u = unfold(hp)
    if why is not None:
        return ClassComparison(False, False, u.species_class, (), why)
    mapped = mechanism_class(linearize(u, numbering=start_numbering(start)))
    return ClassComparison(u.species_class == species_class(steps),
                           mapped == mechanism_class(edges), u.species_class,
                           species_class(steps))  # fmt: skip
