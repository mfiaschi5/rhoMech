"""Steps without arrows (ReactMech): the hydrogens that move, written as mapped atoms.

A source without arrows gives R as the difference of its two mapped sides. ReactMech writes
every hydrogen as a count on its heavy atom (`[OH+:5]`, `[ClH:1]`), so a proton or a hydride
that moves is not an atom of either side: the difference would show the lone pairs that change
and miss the bond that breaks and the bond that forms. `explicit_moving_hydrogens` makes each
hydrogen that moves an atom of both sides, as PMechDB draws it (`[H:20]`): the donor (a mapped
atom that loses hydrogens) gives one to the acceptor (a mapped atom that gains them), the H atom
taking a fresh map number. The pairing is unique when the step has one donor or one acceptor;
which of a donor's hydrogens moves is not, and the label's symmetric closure
(`arrows.symmetry`) makes its equivalent hydrogens one. A step with several donors and several
acceptors has no unique pairing (`delta_be.EditSkip.H_TRANSFER_AMBIGUOUS`); none occurs in the
ReactMech pilot (docs/decisions.md, 2026-10-27).
"""

from __future__ import annotations

from dataclasses import replace

from rdkit import Chem

from ksmech.data import chem
from ksmech.data.schema import ElementaryStep


def _hydrogen_counts(mol: Chem.Mol) -> dict[int, int]:
    return {a.GetAtomMapNum(): a.GetTotalNumHs() for a in mol.GetAtoms() if a.GetAtomMapNum()}


def hydrogen_moves(reac: Chem.Mol, prod: Chem.Mol) -> list[tuple[int, int]] | None:
    """(donor map, acceptor map) of every hydrogen that moves between mapped atoms; None when
    the pairing is not unique (several donors and several acceptors). Hydrogens that leave or
    reach unmapped atoms are not moves: the R check rejects such a step."""
    hr, hp = _hydrogen_counts(reac), _hydrogen_counts(prod)
    gives = [m for m in sorted(hr) if m in hp for _ in range(max(hr[m] - hp[m], 0))]
    takes = [m for m in sorted(hp) if m in hr for _ in range(max(hp[m] - hr[m], 0))]
    if len(set(gives)) > 1 and len(set(takes)) > 1:
        return None
    return list(zip(gives, takes, strict=False))


def _with_hydrogen(mol: Chem.Mol, owner: int, h_map: int) -> Chem.Mol:
    """`mol` with one hydrogen of the atom mapped `owner` made an atom mapped `h_map`."""
    rw = Chem.RWMol(mol)
    atom = next(a for a in rw.GetAtoms() if a.GetAtomMapNum() == owner)
    n = atom.GetTotalNumHs()
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(n - 1)
    h = Chem.Atom(1)
    h.SetAtomMapNum(h_map)
    rw.AddBond(atom.GetIdx(), rw.AddAtom(h), Chem.BondType.SINGLE)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def explicit_moving_hydrogens(step: ElementaryStep) -> ElementaryStep | None:
    """`step` with every hydrogen that moves between mapped atoms written as a mapped atom on
    both sides, the donor's in the reactants and the acceptor's in the products; `step` itself
    when it has arrows (they name their hydrogens) or no hydrogen moves; None when the pairing
    is not unique."""
    if step.arrows:
        return step
    reac = chem.parse_mapped(step.reactants_smiles)
    prod = chem.parse_mapped(step.products_smiles)
    moves = hydrogen_moves(reac, prod)
    if moves is None:
        return None
    if not moves:
        return step
    top = max(a.GetAtomMapNum() for m in (reac, prod) for a in m.GetAtoms())
    for k, (donor, acceptor) in enumerate(moves, top + 1):
        reac, prod = _with_hydrogen(reac, donor, k), _with_hydrogen(prod, acceptor, k)
    return replace(
        step,
        reactants_smiles=chem.canonical_mapped_smiles(reac),
        products_smiles=chem.canonical_mapped_smiles(prod),
    )
