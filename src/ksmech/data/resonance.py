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
and for input augmentation. It runs RDKit's `ResonanceMolSupplier` on a copy whose H counts are
frozen (`frozen_hydrogens`) and keeps only the structures with the input's H count on every
atom, its total charge and its radical electrons. RDKit re-derives the implicit H of an
unbracketed atom in each structure (a nitro O came out as [OH2+]) and also returns structures
with unpaired electrons (CCN([O+])[O-], an azide as N=N[N]); neither is a resonance form.
"""

from __future__ import annotations

from dataclasses import replace

from rdkit import Chem, rdBase
from rdkit.Chem import rdchem

from ksmech.data.schema import ElementaryStep

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


def is_resonance_step(reactants: Chem.Mol, products: Chem.Mol) -> bool:
    """The two sides are resonance forms atom by atom: with the atom maps kept they have one
    sigma key, so no atom changes its sigma bonds or its H and the step moves only pi bonds,
    lone pairs and charges. A degenerate exchange (the same molecules on both sides, but atoms
    that change partners: a thiol-disulfide exchange) has one map-free key and is a real step."""
    return sigma_key(reactants, keep_maps=True) == sigma_key(products, keep_maps=True)


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


def training_steps(steps: list[ElementaryStep]) -> list[ElementaryStep]:
    """The training target: drop resonance steps and renumber `step_index` 0.. within each
    pathway (input order), then drop the steps outside the closed-shell domain
    (`domain_exclusion`, `data.domain`) without renumbering, so a pathway that lost one keeps
    the gap in its indices."""
    return [s for s in _renumbered(steps) if s.domain_exclusion is None]


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
