"""Closed-shell domain screen at ingest: odd-electron fragments, charge repairs, exclusions.

The QM layer runs closed-shell GFN2 on every fragment. A fragment whose electron count (sum of
Z minus the total charge) is odd while RDKit marks no radical electron cannot be computed as
written (`qm.xtb_calc.resolve_charge_spin` raises SPIN_PARITY). In PMechDB all such fragments
but one carry a metal. Each odd fragment, on either side of a step, is handled by explicit
rules, in this order (user decisions of 2026-10-21; docs/decisions.md):

1. **METAL_PLACEHOLDER**: a metal (not an alkali or alkaline-earth metal) with at most one
   neighbour, counting H. These are Lewis-acid catalysts drawn without their ligands or
   counterions (`C=[O+][Ti]`), not molecules. The step is excluded.
2. **Charge repair** (`CHARGE_REPAIRED`, the step stays), one change per fragment, kept only if
   the electron count becomes even and the whole step still validates (charge balance, maps,
   arrow replay):
   - R1: a -1 on a four-bonded group-14 atom (C, Si, Ge, Sn, Pb) is removed;
   - R2: an onium (positive non-metal) bonded to a d0 or d10 acceptor (Ti, Zr, Hf, Sc, Al, B,
     Zn, Cd) whose charge is not negative is read as a dative bond: the acceptor takes the
     compensating -1. Never for Cu, Fe, Co, Ni or any other open-shell-capable metal.
3. **OPEN_SHELL_METAL**: any other odd fragment with a metal outside the d0 / d10 set (Cu(II),
   Fe(III) adducts): outside the closed-shell domain until the radical extension; the step is
   excluded.
4. **CHARGE_UNREPAIRED**: an odd fragment whose only metals are d0 / d10 and that no rule
   repairs consistently (a missing ate charge on a Ti alkoxide, or a repair that would leave
   the step unbalanced). It is a charge error, not an open shell; the step is excluded.

An odd fragment without any metal gets no code here; the QM layer reports it (SPIN_PARITY).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

from rdkit import Chem, rdBase

from ksmech.data import chem


class DomainCode(StrEnum):
    METAL_PLACEHOLDER = "METAL_PLACEHOLDER"
    OPEN_SHELL_METAL = "OPEN_SHELL_METAL"
    CHARGE_UNREPAIRED = "CHARGE_UNREPAIRED"


PRECEDENCE = (
    DomainCode.METAL_PLACEHOLDER,
    DomainCode.OPEN_SHELL_METAL,
    DomainCode.CHARGE_UNREPAIRED,
)
CHARGE_REPAIRED = "CHARGE_REPAIRED"
_S_BLOCK = frozenset([3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88])
METALS = _S_BLOCK | frozenset([13, 31, 49, 50, 81, 82, 83, *range(21, 31), *range(39, 49)])
METALS = METALS | frozenset(range(57, 81)) | frozenset(range(89, 104))
D0_D10 = frozenset(["Ti", "Zr", "Hf", "Sc", "Al", "B", "Zn", "Cd"])
GROUP_14 = frozenset(["C", "Si", "Ge", "Sn", "Pb"])


def electron_count(mol: Chem.Mol) -> int:
    """All electrons of a (possibly mapped, H-explicit or implicit) molecule: sum Z - charge."""
    z = sum(a.GetAtomicNum() + a.GetTotalNumHs() for a in mol.GetAtoms())
    return z - chem.formal_charge(mol)


def is_odd(mol: Chem.Mol) -> bool:
    """Odd electron count that RDKit's radical electrons do not account for."""
    unpaired = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    n = electron_count(mol)
    return n < unpaired or (n - unpaired) % 2 == 1


def _is_metal(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() in METALS


def is_placeholder(frag: Chem.Mol) -> bool:
    """A non-s-block metal with at most one neighbour (H included): ligands dropped."""
    return any(
        _is_metal(a) and a.GetAtomicNum() not in _S_BLOCK and a.GetDegree() + a.GetTotalNumHs() <= 1
        for a in frag.GetAtoms()
    )


def repair_candidate(frag: Chem.Mol) -> tuple[int, int, str] | None:
    """(atom index in `frag`, new formal charge, rule) of the first applicable rule, or None."""
    for a in frag.GetAtoms():
        if a.GetSymbol() in GROUP_14 and a.GetFormalCharge() == -1 and a.GetTotalValence() == 4:
            return a.GetIdx(), 0, "R1 group-14 -1 removed"
    for a in frag.GetAtoms():
        if a.GetFormalCharge() <= 0 or _is_metal(a):
            continue
        for n in a.GetNeighbors():
            if n.GetSymbol() in D0_D10 and n.GetFormalCharge() >= 0:
                return n.GetIdx(), n.GetFormalCharge() - 1, f"R2 dative onium->{n.GetSymbol()}"
    return None


def fallthrough_code(frag: Chem.Mol) -> DomainCode | None:
    """The code of an odd fragment that is neither a placeholder nor repaired."""
    metals = [a for a in frag.GetAtoms() if _is_metal(a) or a.GetSymbol() == "B"]
    if not metals:
        return None
    open_capable = [
        a for a in metals if a.GetSymbol() not in D0_D10 and a.GetAtomicNum() not in _S_BLOCK
    ]
    return DomainCode.OPEN_SHELL_METAL if open_capable else DomainCode.CHARGE_UNREPAIRED


@dataclass
class Screen:
    """The outcome of screening one step: the sides to keep, the exclusion code (None when the
    step stays) and the charge repairs applied, as (side, rule, fragment before, after)."""

    reactants: Chem.Mol
    products: Chem.Mol
    code: DomainCode | None = None
    repairs: list[tuple[str, str, str, str]] = field(default_factory=list)

    def repair_note(self, reactants_before: str, products_before: str) -> str | None:
        """JSON record of the repairs, with both sides as drawn, for `charge_repairs.csv`."""
        if not self.repairs:
            return None
        rows = [{"side": s, "rule": r, "before": b, "after": a} for s, r, b, a in self.repairs]
        return json.dumps(
            {
                "repairs": rows,
                "reactants_before": reactants_before,
                "products_before": products_before,
            }
        )


def _fragments(mol: Chem.Mol) -> list[tuple[tuple[int, ...], Chem.Mol]]:
    ids = Chem.GetMolFrags(mol)
    mols = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return list(zip(ids, mols, strict=True))


def _recharged(mol: Chem.Mol, atom: int, charge: int) -> Chem.Mol:
    out = Chem.RWMol(mol)
    out.GetAtomWithIdx(atom).SetFormalCharge(charge)
    out.GetAtomWithIdx(atom).SetNoImplicit(True)  # the H count stays as drawn
    with rdBase.BlockLogs():
        out.UpdatePropertyCache(strict=False)
    return out.GetMol()


def screen_side(mol: Chem.Mol, side: str, repair: bool = True):
    """(side with the repairs applied, codes of its odd fragments, repairs) for one side."""
    out, codes, repairs = mol, [], []
    for atoms, frag in _fragments(mol):
        if not is_odd(frag):
            continue
        if is_placeholder(frag):
            codes.append(DomainCode.METAL_PLACEHOLDER)
            continue
        fix = repair_candidate(frag) if repair else None
        if fix is not None:
            fixed = _recharged(frag, fix[0], fix[1])
            if not is_odd(fixed):
                out = _recharged(out, atoms[fix[0]], fix[1])
                before, after = chem.unmapped_canonical(frag), chem.unmapped_canonical(fixed)
                repairs.append((side, fix[2], before, after))
                continue
        code = fallthrough_code(frag)
        if code is not None:
            codes.append(code)
    return out, codes, repairs


def screen_step(reactants: Chem.Mol, products: Chem.Mol, repair: bool = True) -> Screen:
    """Screen both sides; the step's code is the first of `PRECEDENCE` that any fragment has.
    Repaired sides are re-parsed from their canonical mapped SMILES (sanitized as at ingest)."""
    r, r_codes, r_rep = screen_side(reactants, "reactants", repair)
    p, p_codes, p_rep = screen_side(products, "products", repair)
    codes = set(r_codes + p_codes)
    code = next((c for c in PRECEDENCE if c in codes), None)
    if r_rep:
        r = chem.parse_mapped(chem.canonical_mapped_smiles(r))
    if p_rep:
        p = chem.parse_mapped(chem.canonical_mapped_smiles(p))
    return Screen(r, p, code, r_rep + p_rep)
