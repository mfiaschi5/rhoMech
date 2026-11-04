"""Resonance identity, resonance forms, and the training-target filter.

`sigma_key(mol)` is the resonance identity (docs/decisions.md, 2026-10-25): the map-free
canonical SMILES of the sigma skeleton (every bond single, no aromatic flag, no stereo) with
every atom's H count frozen and every charge and radical zeroed, followed by `|` and the total
charge. Two closed-shell Lewis structures with one sigma key have the same atoms, sigma bonds,
H and total charge, so the same electron count: they differ only in where the pi bonds, lone
pairs and charges sit, which is what resonance forms are. No enumeration is needed, so the key
is exact and never truncated. A step whose two sides have one sigma key with the atom maps
kept (`is_resonance_step`) is a resonance step and leaves the training target; a degenerate
exchange, whose sides are the same molecules but whose atoms change partners, is a real step.

`resonance_forms` enumerates the forms themselves, for the edit-label class (`arrows.delta_be`)
and for input augmentation. The product it gets for a label is the one the step's edit gives:
for a step without arrows, the product in the Kekule structure closest to the reactant
(`delta_be.mapped_closest`), not the dataset's own kekulization. The enumeration does not take
every Kekule structure (KEKULE_ALL): that would multiply the label class, and RESONANCE_CAP
already cuts it short for many ReactMech products. It runs RDKit's `ResonanceMolSupplier` on a
copy whose H counts are frozen (`frozen_hydrogens`) and keeps only the structures with the
input's H count on every atom, its total charge and its radical electrons. RDKit re-derives the
implicit H of an unbracketed atom in each structure (a nitro O came out as [OH2+]) and also
returns structures with unpaired electrons (CCN([O+])[O-], an azide as N=N[N]); neither is a
resonance form.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import replace
from enum import StrEnum

from rdkit import Chem, rdBase
from rdkit.Chem import rdchem

from ksmech.data import spin
from ksmech.data.domain import admitted
from ksmech.data.schema import ElementaryStep

log = logging.getLogger(__name__)

_FLAGS = (
    rdchem.ResonanceFlags.UNCONSTRAINED_CATIONS
    | rdchem.ResonanceFlags.UNCONSTRAINED_ANIONS
    | rdchem.ResonanceFlags.ALLOW_CHARGE_SEPARATION
)
_KEKULE_OPS = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY


def sigma_key(mol: Chem.Mol, keep_maps: bool = False) -> str:
    """The sigma skeleton of `mol` with its H frozen and its charges and radicals zeroed, as
    canonical SMILES without stereo, then `|<total charge>`; map-free unless `keep_maps`. An H
    drawn as an atom counts on its neighbour, as an implicit H does, and H2 drawn [H][H] is
    [HH]."""
    m = Chem.RWMol(mol)
    m.UpdatePropertyCache(strict=False)
    charge = sum(a.GetFormalCharge() for a in m.GetAtoms())
    for atom in m.GetAtoms():
        atom.SetNumExplicitHs(atom.GetTotalNumHs())
        atom.SetNoImplicit(True)
        atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetIsAromatic(False)
        if not keep_maps:
            atom.SetAtomMapNum(0)
    for bond in m.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    m.UpdatePropertyCache(strict=False)
    with rdBase.BlockLogs():  # removing an H atom adds it to its frozen neighbour's count
        m = _dihydrogen_as_one_atom(Chem.RWMol(Chem.RemoveHs(m, sanitize=False)))
    m.UpdatePropertyCache(strict=False)
    # every atom with its H count written out: the writer's own choice of brackets does not
    # tell an O with no H (a zeroed alkoxide) from an O-H
    return f"{Chem.MolToSmiles(m, isomericSmiles=False, allHsExplicit=True)}|{charge}"


def _dihydrogen_as_one_atom(m: Chem.RWMol) -> Chem.RWMol:
    """H2 drawn as two atoms becomes one atom carrying an H (`RemoveHs` removes an H only
    from a heavier atom)."""
    pairs = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in m.GetBonds()
        if b.GetBeginAtom().GetAtomicNum() == b.GetEndAtom().GetAtomicNum() == 1
        and b.GetBeginAtom().GetDegree() == b.GetEndAtom().GetDegree() == 1
    ]
    for keep, drop in sorted(pairs, key=lambda p: -max(p)):
        m.GetAtomWithIdx(keep).SetNumExplicitHs(m.GetAtomWithIdx(keep).GetNumExplicitHs() + 1)
        m.RemoveAtom(drop)
    return m


def _fragments(mol: Chem.Mol) -> list[tuple[tuple[int, ...], Chem.Mol]]:
    mapping: list = []
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping)
    return [(tuple(a), f) for a, f in zip(mapping, frags, strict=True)]


def fragment_spins(
    mol: Chem.Mol, nonbonding: Sequence[int] | None = None
) -> list[tuple[tuple[int, ...], Chem.Mol | None, int]]:
    """(atom indices, fragment, unpaired count) of every fragment of `mol`, the count from the
    drawing's non-bonding electrons (`spin.spin_of_drawing`; `nonbonding` is N per atom of
    `mol`, read from the drawing when not given). When no species-table row can raise any
    fragment (`spin.may_raise`), the counts are the odd-N atoms and no fragment is built (its
    place holds None): the identity key of a closed shell costs about one sigma key."""
    from ksmech.arrows.apply import nonbonding_electrons

    m, n = mol, nonbonding
    if n is None:
        m = spin.kekule_copy(mol) if any(b.GetIsAromatic() for b in mol.GetBonds()) else mol
        m.UpdatePropertyCache(strict=False)
        n = [nonbonding_electrons(a) for a in m.GetAtoms()]
    else:
        m.UpdatePropertyCache(strict=False)
    groups = [tuple(g) for g in Chem.GetMolFrags(m, sanitizeFrags=False)]
    drawn = [sum(int(n[i]) % 2 for i in g) for g in groups]
    if not any(spin.may_raise(m, g, d) for g, d in zip(groups, drawn, strict=True)):
        return [(g, None, d) for g, d in zip(groups, drawn, strict=True)]
    out = []
    for atoms, frag in _fragments(mol):
        n = None if nonbonding is None else [int(nonbonding[i]) for i in atoms]
        out.append((atoms, frag, spin.spin_of_drawing(frag, n)[0]))
    return out


def identity_key(
    mol: Chem.Mol, keep_maps: bool = False, nonbonding: Sequence[int] | None = None
) -> str:
    """The identity key v2 (open-shell plan U12): the v1 `sigma_key` when the drawing has no
    unpaired electron (every closed-shell key unchanged), else the v1 key, `|`, and the sorted
    multiset of (fragment sigma SMILES, fragment charge, fragment unpaired count), fields
    joined by ',' and fragments by ';' (never '.': serve splits pool keys on it). The count
    comes from the drawing's N (`fragment_spins`), so a radical cation and its closed-shell
    parent, or D. + A and D+ + A.-, differ, while O=O and [O][O] agree where the species table
    holds O2 as a Hund case."""
    v1 = sigma_key(mol, keep_maps)
    parts = fragment_spins(mol, nonbonding)
    if not any(u for *_, u in parts):
        return v1
    frags = [f for _, f in _fragments(mol)]
    rows = []
    for frag, (*_, u) in zip(frags, parts, strict=True):
        smiles, charge = sigma_key(frag, keep_maps).rsplit("|", 1)
        rows.append(f"{smiles},{charge},{u}")
    return f"{v1}|{';'.join(sorted(rows))}"


def mapped_changes(
    reactants: Chem.Mol, products: Chem.Mol
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
    """(bond-order changes (m, n, d) with m < n, changes of non-bonding electrons (m, dN)) between
    two mapped sides, on map numbers; unmapped atoms are left out."""
    from ksmech.arrows.apply import TYPE_TO_ORDER, nonbonding_electrons  # arrows reads data

    def state(mol: Chem.Mol) -> tuple[dict, dict]:
        m = Chem.Mol(mol)
        m.UpdatePropertyCache(strict=False)
        orders = {}
        for b in m.GetBonds():
            i, j = b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
            if i and j:
                orders[(min(i, j), max(i, j))] = TYPE_TO_ORDER.get(b.GetBondType(), 1)
        n = {a.GetAtomMapNum(): nonbonding_electrons(a) for a in m.GetAtoms() if a.GetAtomMapNum()}
        return orders, n

    (ob, nb), (oa, na) = state(reactants), state(products)
    bonds = [
        (i, j, oa.get((i, j), 0) - ob.get((i, j), 0)) for i, j in sorted(ob.keys() | oa.keys())
    ]
    lone = [(m, na[m] - nb[m]) for m in sorted(nb.keys() & na.keys())]
    return [b for b in bonds if b[2]], [x for x in lone if x[1]]


def balanced_flow(bonds: Iterable[tuple[int, int, int]], lone: Iterable[tuple[int, int]]) -> bool:
    """The electron-flow law (open-shell plan P6): electrons move only between bonded atoms.
    Join the atoms whose bond order changes into components; the flow is BALANCED when every
    atom whose non-bonding electrons change lies on a changed bond and every component has
    sum dN + 2 sum dB = 0 (its electrons, so its charge, are conserved). The closure of arrow
    moves on the drawn skeleton, with no enumeration cap."""
    parent: dict[int, int] = {}

    def root(a: int) -> int:
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    bonds = list(bonds)
    for i, j, _ in bonds:
        parent[root(i)] = root(j)
    total: dict[int, int] = {}
    for i, _, d in bonds:
        total[root(i)] = total.get(root(i), 0) + 2 * d
    for a, d in lone:
        if a not in parent:
            return False
        total[root(a)] = total.get(root(a), 0) + d
    return not any(total.values())


FLOW_FORMS = 32  # resonance forms per side tried when the drawn flow is unbalanced


def _side_forms(mol: Chem.Mol) -> list[Chem.Mol]:
    frozen = frozen_hydrogens(mol)
    if frozen is None:
        return []
    return resonance_forms(frozen, FLOW_FORMS)[0] + radical_forms(frozen, FLOW_FORMS)


def flow_balanced(reactants: Chem.Mol, products: Chem.Mol) -> bool:
    """The flow between the two sides is balanced as drawn, or between some resonance
    drawing of the reactants and some form of the products (`_side_forms`, the closed-shell
    enumeration and the radical anion proxy)."""
    if balanced_flow(*mapped_changes(reactants, products)):
        return True
    return any(
        balanced_flow(*mapped_changes(r, p))
        for r in _side_forms(reactants)
        for p in _side_forms(products)
    )


class StepKind(StrEnum):
    """What one mapped step is by the bookkeeping law (`classify_step`)."""

    NULL_STEP = "NULL_STEP"  # identical sides: R = 0
    RESONANCE = "RESONANCE"  # one identity key with the maps kept, and a balanced flow
    NONCONJUGATED_SHIFT = "NONCONJUGATED_SHIFT"  # one key, every flow unbalanced: an electron
    #   jump across saturated atoms, not resonance
    STEP = "STEP"  # a real step: the keys differ (sigma bonds, H, fragment charges or spins)


def classify_step(reactants: Chem.Mol, products: Chem.Mol) -> StepKind:
    """The step's kind (StepKind): NULL_STEP, else RESONANCE or NONCONJUGATED_SHIFT when the
    two sides have one identity key with the atom maps kept (no atom changes its sigma bonds
    or H; a degenerate exchange, whose atoms change partners, is a real step), else STEP."""
    from ksmech.data import chem

    if chem.canonical_mapped_smiles(reactants) == chem.canonical_mapped_smiles(products):
        return StepKind.NULL_STEP
    if identity_key(reactants, keep_maps=True) != identity_key(products, keep_maps=True):
        return StepKind.STEP
    if flow_balanced(reactants, products):
        return StepKind.RESONANCE
    return StepKind.NONCONJUGATED_SHIFT


def is_resonance_step(reactants: Chem.Mol, products: Chem.Mol) -> bool:
    """The two sides are resonance forms atom by atom (`classify_step` is RESONANCE)."""
    return classify_step(reactants, products) is StepKind.RESONANCE


def frozen_hydrogens(mol: Chem.Mol) -> Chem.Mol | None:
    """A copy in which every atom keeps the H count it has (explicit H = total H, no implicit
    H), sanitized again, since the enumeration needs the conjugation that sanitization sets;
    None when it does not sanitize. Aromaticity is not perceived: on a Kekule structure it
    changes which structures RDKit returns, and the edit labels read bond orders."""
    m = Chem.Mol(mol)
    m.UpdatePropertyCache(strict=False)
    for atom in m.GetAtoms():
        atom.SetNumExplicitHs(atom.GetTotalNumHs())
        atom.SetNoImplicit(True)
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(m, sanitizeOps=_KEKULE_OPS)
    except Chem.rdchem.MolSanitizeException:
        return None
    return m


def _electron_state(mol: Chem.Mol) -> tuple[tuple[int, ...], int, int]:
    """(H count of every atom, total charge, radical electrons): what a resonance form keeps."""
    m = Chem.Mol(mol)
    m.UpdatePropertyCache(strict=False)
    atoms = list(m.GetAtoms())
    return (
        tuple(a.GetTotalNumHs() for a in atoms),
        sum(a.GetFormalCharge() for a in atoms),
        sum(a.GetNumRadicalElectrons() for a in atoms),
    )


def resonance_forms(mol: Chem.Mol, max_structures: int) -> tuple[list[Chem.Mol], int]:
    """The resonance forms of `mol` (its H frozen by the caller: `frozen_hydrogens`) among the
    first `max_structures` structures RDKit enumerates, keeping only those with the electron
    state of `mol`; and the number of structures RDKit counted (truncated when it reaches
    `max_structures`)."""
    ref = _electron_state(mol)
    with rdBase.BlockLogs():
        supplier = rdchem.ResonanceMolSupplier(mol, int(_FLAGS), max_structures)
        n = len(supplier)
        raw = [supplier[k] for k in range(min(n, max_structures))]
    return [f for f in raw if f is not None and _electron_state(f) == ref], n


def _signature(mol: Chem.Mol) -> tuple:
    bonds = sorted(
        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        + (str(b.GetBondType()),)
        for b in mol.GetBonds()
    )
    return tuple(bonds), tuple(a.GetFormalCharge() for a in mol.GetAtoms())


def with_radicals(mol: Chem.Mol) -> Chem.Mol:
    """`mol` with every atom's radical electrons set to the parity of its N (RDKit's flags
    follow the drawing's electrons: a moved radical site, a paired [Br+])."""
    from ksmech.arrows.apply import nonbonding_electrons

    m = Chem.RWMol(mol)
    m.UpdatePropertyCache(strict=False)
    for a in m.GetAtoms():
        a.SetNumRadicalElectrons(nonbonding_electrons(a) % 2)
    out = m.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def _anion_forms(base: Chem.Mol, centre: int, max_structures: int):
    """(the anion proxy of `base` at the radical `centre`, H frozen; its resonance forms; the
    atoms of the centre's conjugated group), or None when the anion does not sanitize."""
    anion = Chem.RWMol(base)
    atom = anion.GetAtomWithIdx(centre)
    atom.SetFormalCharge(atom.GetFormalCharge() - 1)
    atom.SetNumRadicalElectrons(0)
    anion = frozen_hydrogens(anion.GetMol())
    if anion is None:
        return None
    with rdBase.BlockLogs():
        supplier = rdchem.ResonanceMolSupplier(anion, int(_FLAGS), max_structures)
        group = supplier.GetAtomConjGrpIdx(centre)
        atoms = {
            a.GetIdx() for a in anion.GetAtoms() if supplier.GetAtomConjGrpIdx(a.GetIdx()) == group
        }
    if group < 0:
        atoms = {centre}
    return anion, resonance_forms(anion, max_structures)[0], atoms


def _bond_changes(a: Chem.Mol, b: Chem.Mol) -> int:
    return sum(x != y for x, y in zip(_signature(a)[0], _signature(b)[0], strict=True))


def radical_forms(mol: Chem.Mol, max_structures: int, per_site: bool = False) -> list[Chem.Mol]:
    """The other resonance forms of a radical drawing (H frozen by the caller), through an
    anion proxy (open-shell plan P6 step 6): each radical centre (an odd-N atom) in turn takes
    one more electron, RDKit enumerates the anion's forms (`resonance_forms`), and in every
    form whose charges outside the centre's conjugated group are the anion's, each negative
    atom of that group gives the electron back: it carries the radical. The new structures, on
    the atom indices of `mol`, each once; none for a closed shell. `per_site` keeps one
    structure per radical site, the one with the fewest formal charges, then the fewest bond
    changes from `mol` (the input drawings: NO3. has three, one per O). Known gap: 3-electron
    N-O forms are missed."""
    from ksmech.arrows.apply import nonbonding_electrons

    base = Chem.Mol(mol)
    base.UpdatePropertyCache(strict=False)
    centres = [a.GetIdx() for a in base.GetAtoms() if nonbonding_electrons(a) % 2]
    found: dict[tuple, Chem.Mol] = {}
    for c in centres:
        proxy = _anion_forms(base, c, max_structures)
        if proxy is None:
            continue
        anion, forms, group = proxy
        ref = [a.GetFormalCharge() for a in anion.GetAtoms()]
        for form in forms:
            q = [a.GetFormalCharge() for a in form.GetAtoms()]
            if any(b.GetIsAromatic() for b in form.GetBonds()) or any(
                q[k] != ref[k] for k in range(len(q)) if k not in group
            ):
                continue
            for k in sorted(i for i in group if q[i] < 0):
                back = Chem.RWMol(form)
                back.GetAtomWithIdx(k).SetFormalCharge(q[k] + 1)
                back = with_radicals(back.GetMol())
                found.setdefault(_signature(back), back)
    found.pop(_signature(base), None)
    out = list(found.values())
    if per_site:
        out = _one_per_site(base, out)
    return out


def _one_per_site(base: Chem.Mol, forms: list[Chem.Mol]) -> list[Chem.Mol]:
    """One form per set of radical atoms (other than the drawn set): fewest formal charges,
    then fewest bond changes from `base`, then the first."""

    def site(m: Chem.Mol) -> tuple:
        return tuple(a.GetIdx() for a in m.GetAtoms() if a.GetNumRadicalElectrons())

    def cost(m: Chem.Mol) -> tuple:
        return sum(abs(a.GetFormalCharge()) for a in m.GetAtoms()), _bond_changes(base, m)

    best: dict[tuple, Chem.Mol] = {}
    for m in forms:
        if site(m) != site(with_radicals(base)) and (
            site(m) not in best or cost(m) < cost(best[site(m)])
        ):
            best[site(m)] = m
    return list(best.values())


def training_steps(
    steps: list[ElementaryStep], admit: Mapping[str, Collection[str]] | None = None
) -> list[ElementaryStep]:
    """The training target: drop resonance steps and renumber `step_index` 0.. within each
    pathway (input order), then drop the steps outside the closed-shell domain
    (`domain_exclusion`, `data.domain`) and those whose `domain_class` holds a class their
    source's admit set lacks (`admit`, source -> classes; default `data.domain.ADMIT`), without
    renumbering, so a pathway that lost one keeps the gap in its indices. The steps left out
    by the admit set are logged by class (`not_admitted` counts them)."""
    kept = [s for s in _renumbered(steps) if s.domain_exclusion is None]
    if counts := not_admitted(kept, admit):
        log.info("not admitted by domain_class: %s", dict(counts))
    return [s for s in kept if admitted(s, admit)]


def not_admitted(
    steps: list[ElementaryStep], admit: Mapping[str, Collection[str]] | None = None
) -> Counter[str]:
    """`domain_class` values of the steps their source's admit set leaves out."""
    return Counter(s.domain_class for s in steps if not admitted(s, admit))


def _renumbered(steps: list[ElementaryStep]) -> list[ElementaryStep]:
    kept = [s for s in steps if not s.resonance_step]
    counters: dict[str, int] = {}
    out = []
    for s in kept:
        if s.pathway_id is None or s.step_index is None:
            out.append(s)
            continue
        idx = counters.get(s.pathway_id, 0)
        counters[s.pathway_id] = idx + 1
        out.append(s if s.step_index == idx else replace(s, step_index=idx))
    return out
