"""Closed-shell domain laws at ingest: drawing repairs, exclusions and flags (spec v9.1,
'Domain coverage'; open-shell plan P0a step 6).

The QM layer runs closed-shell GFN2 on every fragment, so a fragment whose electron count (sum
of Z minus the total charge) is odd while RDKit marks no radical electron cannot be computed as
written (`qm.xtb_calc.resolve_charge_spin` raises SPIN_PARITY). Every step is screened by laws
that hold for every element and every parity, in this order:

1. **Repairs** (`CHARGE_REPAIRED`, logged in `charge_repair` and `charge_repairs.csv`), kept only
   if the repaired step still validates (charge balance, maps, arrow replay where arrows exist;
   `validation.screened`), otherwise the step is screened as drawn:
   - R1: a -1 on a four-bonded group-14 atom (C, Si, Ge, Sn, Pb) of an odd fragment is removed;
   - D, the dative law: an onium donor (a positively charged non-metal other than C) bonded to
     an acceptor (a metal, or B) gives it a dative pair. Each acceptor's u is its onium donors
     minus the negative charge it already carries (never below 0). In each fragment the
     acceptors' charges are lowered by the largest j <= sum(u) that leaves the fragment's
     electron count even and keeps the side's net charge >= 0 (fragments that together would
     take it below zero leave the side as drawn), and only when both sides of the step are
     lowered by the same total; otherwise the drawing stays. This is closed-shell
     electron bookkeeping: on an odd fragment it completes the dative bond the drawing left
     uncompensated ([Pd][P+]R3 -> [Pd-][P+]R3, Pd(0)), on an even one it re-charges a
     two-onium complex drawn +2 too high, and it leaves a correctly charged cation adduct alone
     (C=[O+][Li] in a +1 pool is even, so j = 0).
2. **METAL_PLACEHOLDER**, on every fragment of either side, odd or even, before any other code:
   a non-s-block metal with exactly one neighbour (H counted) and formal charge <= 0, unless that
   neighbour is an onium donor the metal compensates (an ML1 complex: the monoligated Pd(0)
   drawing [Pd-][P+]R3). These are Lewis acids or catalysts drawn without their ligands
   (`C=[O+][Ti]`, `C[O][Zn-]`, an Sn with radical electrons). A free atom or ion (no neighbour)
   and MX+ drawn with the charge on the metal ([Cl][Zn+]) are defined species. The step is
   excluded.
3. **OPEN_SHELL_METAL**: an odd fragment left after the repairs with a metal outside the d0 / d10
   set (Cu(II), Fe(III) adducts), outside the closed-shell domain until the radical extension;
   **CHARGE_UNREPAIRED**: one whose only metals are d0 / d10 (a charge error, not an open shell).
   An odd fragment without any metal gets no code here; the QM layer reports it (SPIN_PARITY).
4. **RADICAL_DRAWN**: an even fragment with an atom RDKit marks with unpaired electrons. In a
   polar dataset these are drawing errors (a methyl written `[C:20]`, K without its charge);
   read literally they are open shells (user decision of 2026-10-24); the step is excluded.
5. **METAL_SUSPECT**, a flag, never an exclusion: a kept step with an even fragment holding a
   first-row transition metal (Sc to Cu) whose oxidation-state bookkeeping (`oxidation_state`)
   leaves an open d shell (d1-d9). 3d ligand-field splittings are comparable to the pairing
   energy, so the closed-shell singlet the QM computes is an assumption.

A step's `domain_class` records the repairs and flags it carries (`DomainClass`, several joined
by '+' in enum order); `training_steps` admits a class into the training target only when it is
in the source's admit set (`ADMIT`).
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from rdkit import Chem, rdBase

from ksmech.data import chem
from ksmech.data.schema import ElementaryStep, Source


class DomainCode(StrEnum):
    METAL_PLACEHOLDER = "METAL_PLACEHOLDER"
    OPEN_SHELL_METAL = "OPEN_SHELL_METAL"
    CHARGE_UNREPAIRED = "CHARGE_UNREPAIRED"
    RADICAL_DRAWN = "RADICAL_DRAWN"


class DomainClass(StrEnum):
    """What a step's `domain_class` can hold (later phases add values): repairs first, flags
    after."""

    REPAIRED_DATIVE = "REPAIRED_DATIVE"  # the dative law re-charged an acceptor
    METAL_SUSPECT = "METAL_SUSPECT"  # an open-d-shell first-row metal computed as a singlet


PRECEDENCE = (
    DomainCode.METAL_PLACEHOLDER,
    DomainCode.OPEN_SHELL_METAL,
    DomainCode.CHARGE_UNREPAIRED,
    DomainCode.RADICAL_DRAWN,
)
CHARGE_REPAIRED = "CHARGE_REPAIRED"
R1_RULE = "R1 group-14 -1 removed"
DATIVE_RULE = "D dative onium->M"
CLASS_SEP = "+"
# the classes each source admits into the training target (P0a: every class, both are flags)
ADMIT: dict[str, frozenset[str]] = {str(s): frozenset(map(str, DomainClass)) for s in Source}

_S_BLOCK = frozenset([3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88])
METALS = _S_BLOCK | frozenset([13, 31, 49, 50, 81, 82, 83, *range(21, 31), *range(39, 49)])
METALS = METALS | frozenset(range(57, 81)) | frozenset(range(89, 104))
D0_D10 = frozenset(["Ti", "Zr", "Hf", "Sc", "Al", "B", "Zn", "Cd"])
GROUP_14 = frozenset(["C", "Si", "Ge", "Sn", "Pb"])
FIRST_ROW_TM = frozenset(range(21, 30))  # Sc to Cu
GROUP = {**{z: z - 18 for z in range(21, 31)}, **{z: z - 36 for z in range(39, 49)}}
GROUP |= {z: z - 68 for z in range(71, 81)}
# Pauling electronegativities of the metals, for metal-metal bonds (CRC values; a metal without
# a tabulated value splits its metal-metal bonds evenly)
PAULING = {
    3: 0.98, 4: 1.57, 11: 0.93, 12: 1.31, 13: 1.61, 19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54,
    23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81,
    37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.6, 42: 2.16, 43: 1.9, 44: 2.2, 45: 2.28,
    46: 2.20, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96, 55: 0.79, 56: 0.89, 57: 1.10, 58: 1.12,
    59: 1.13, 60: 1.14, 62: 1.17, 64: 1.20, 66: 1.22, 67: 1.23, 68: 1.24, 69: 1.25, 71: 1.27,
    72: 1.3, 73: 1.5, 74: 2.36, 75: 1.9, 76: 2.2, 77: 2.20, 78: 2.28, 79: 2.54, 80: 2.00,
    81: 1.62, 82: 2.33, 83: 2.02, 90: 1.3, 91: 1.5, 92: 1.38, 93: 1.36, 94: 1.28,
}  # fmt: skip


def electron_count(mol: Chem.Mol) -> int:
    """All electrons of a (possibly mapped, H-explicit or implicit) molecule: sum Z - charge."""
    z = sum(a.GetAtomicNum() + a.GetTotalNumHs() for a in mol.GetAtoms())
    return z - chem.formal_charge(mol)


def is_odd(mol: Chem.Mol) -> bool:
    """Odd electron count that RDKit's radical electrons do not account for."""
    unpaired = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    n = electron_count(mol)
    return n < unpaired or (n - unpaired) % 2 == 1


def has_drawn_radical(mol: Chem.Mol) -> bool:
    """Some atom carries unpaired electrons as drawn (RDKit's radical electrons)."""
    return any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())


def _is_metal(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() in METALS


def is_acceptor(atom: Chem.Atom) -> bool:
    """A dative acceptor: a metal, or B."""
    return _is_metal(atom) or atom.GetAtomicNum() == 5


def is_onium_donor(atom: Chem.Atom) -> bool:
    """A positively charged non-metal other than C (its + is a lone pair it gave away)."""
    return atom.GetFormalCharge() > 0 and not _is_metal(atom) and atom.GetAtomicNum() != 6


def uncompensated(atom: Chem.Atom) -> int:
    """u of an acceptor: its onium donors minus the negative charge it already carries, >= 0."""
    donors = sum(is_onium_donor(n) for n in atom.GetNeighbors())
    return max(0, donors - max(0, -atom.GetFormalCharge()))


def is_placeholder(frag: Chem.Mol) -> bool:
    """A non-s-block metal with exactly one neighbour (H included) and charge <= 0, unless the
    neighbour is an onium donor the metal compensates (ML1, [Pd-][P+]R3): ligands dropped."""
    for a in frag.GetAtoms():
        if not _is_metal(a) or a.GetAtomicNum() in _S_BLOCK or a.GetFormalCharge() > 0:
            continue
        if a.GetDegree() + a.GetTotalNumHs() != 1:
            continue
        donor = a.GetDegree() == 1 and is_onium_donor(a.GetNeighbors()[0])
        if not (donor and a.GetFormalCharge() < 0):
            return True
    return False


def _bond_share(atom: Chem.Atom, bond: Chem.Bond) -> int:
    """What one bond adds to `atom`'s oxidation state: its order when the pair goes to the
    partner (a non-metal, H included, or a more electronegative metal), minus it when it comes
    to `atom`, 0 for an even split or a dative bond (an L ligand)."""
    if bond.GetBondType() == Chem.BondType.DATIVE:
        return 0
    other = bond.GetOtherAtom(atom)
    order = int(bond.GetBondTypeAsDouble())
    if not _is_metal(other):
        return order  # an onium donor too: its +1 and the metal's compensating -1 cancel (L)
    mine, theirs = PAULING.get(atom.GetAtomicNum()), PAULING.get(other.GetAtomicNum())
    if mine is None or theirs is None or mine == theirs:
        return 0
    return order if theirs > mine else -order


def oxidation_state(atom: Chem.Atom) -> int:
    """The formal oxidation state of a metal in its drawing: the formal charge, +1 per H
    (hydride), and every bond's pair given to its more electronegative end (`_bond_share`).
    [Pd-][P+]R3 reads Pd(0); a drawing that leaves its onium donors uncompensated reads one
    unit higher per donor (Ni(PMe3)2 drawn +2 reads Ni(II))."""
    return (
        atom.GetFormalCharge()
        + atom.GetTotalNumHs()
        + sum(_bond_share(atom, b) for b in atom.GetBonds())
    )


def d_count(atom: Chem.Atom) -> int | None:
    """d electrons of a transition metal at its oxidation state, None for the others."""
    group = GROUP.get(atom.GetAtomicNum())
    return None if group is None else group - oxidation_state(atom)


def is_metal_suspect(frag: Chem.Mol) -> bool:
    """An even fragment with a first-row transition metal left with an open d shell (d1-d9)."""
    if is_odd(frag):
        return False
    return any(a.GetAtomicNum() in FIRST_ROW_TM and 1 <= d_count(a) <= 9 for a in frag.GetAtoms())


def repair_candidate(frag: Chem.Mol) -> tuple[int, int, str] | None:
    """(atom index in `frag`, new formal charge, rule) of R1 on an odd fragment, or None."""
    for a in frag.GetAtoms():
        if a.GetSymbol() in GROUP_14 and a.GetFormalCharge() == -1 and a.GetTotalValence() == 4:
            return a.GetIdx(), 0, R1_RULE
    return None


def dative_lowering(frag: Chem.Mol, budget: int) -> list[tuple[int, int]]:
    """(atom index in `frag`, new charge) of the acceptors the dative law re-charges in one
    fragment, lowering at most `budget` charges in all (the side's net charge stays >= 0); the
    largest such j that makes the electron count even, shared out in atom order."""
    acceptors = [(a, uncompensated(a)) for a in frag.GetAtoms() if is_acceptor(a)]
    top = min(sum(u for _, u in acceptors), budget)
    j = top - (electron_count(frag) + top) % 2
    out = []
    for a, u in acceptors:
        take = min(u, j)
        if take > 0:
            out.append((a.GetIdx(), a.GetFormalCharge() - take))
            j -= take
    return out


def fallthrough_code(frag: Chem.Mol) -> DomainCode | None:
    """The code of an odd fragment that is neither a placeholder nor repaired."""
    metals = [a for a in frag.GetAtoms() if _is_metal(a) or a.GetSymbol() == "B"]
    if not metals:
        return None
    open_capable = [
        a for a in metals if a.GetSymbol() not in D0_D10 and a.GetAtomicNum() not in _S_BLOCK
    ]
    return DomainCode.OPEN_SHELL_METAL if open_capable else DomainCode.CHARGE_UNREPAIRED


def join_classes(classes: Iterable[str]) -> str | None:
    """The `domain_class` value of a set of classes: None, or the classes joined by '+', the
    `DomainClass` ones in enum order, any other after them sorted."""
    have = set(classes)
    order = [str(c) for c in DomainClass if c in have]
    return CLASS_SEP.join(order + sorted(have - set(order))) or None


def domain_classes(value: str | None) -> frozenset[str]:
    """The classes a `domain_class` value holds."""
    return frozenset(value.split(CLASS_SEP)) if value else frozenset()


def admitted(step: ElementaryStep, admit: Mapping[str, Collection[str]] | None = None) -> bool:
    """Every class of the step's `domain_class` is in its source's admit set (`ADMIT`)."""
    allowed = (ADMIT if admit is None else admit).get(str(step.source), frozenset())
    return domain_classes(step.domain_class) <= set(allowed)


def admit_config(source: str, admit: Mapping[str, Collection[str]] | None = None) -> list[str]:
    """The source's admit set as recorded in a provenance config."""
    return sorted((ADMIT if admit is None else admit).get(str(source), ()))


@dataclass
class Screen:
    """The outcome of screening one step: the sides to keep, the exclusion code (None when the
    step stays), the repairs applied, as (side, rule, fragment before, after), and the
    `DomainClass` values."""

    reactants: Chem.Mol
    products: Chem.Mol
    code: DomainCode | None = None
    repairs: list[tuple[str, str, str, str]] = field(default_factory=list)
    classes: set[str] = field(default_factory=set)

    @property
    def domain_class(self) -> str | None:
        return join_classes(self.classes)

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


def _recharged(mol: Chem.Mol, charges: Iterable[tuple[int, int]]) -> Chem.Mol:
    """`mol` with the (atom index, formal charge) changes applied; each atom keeps its H."""
    out = Chem.RWMol(mol)
    for idx, charge in charges:
        atom = out.GetAtomWithIdx(idx)
        atom.SetNumExplicitHs(atom.GetTotalNumHs())
        atom.SetNoImplicit(True)  # the H count stays as drawn, implicit H included
        atom.SetFormalCharge(charge)
    with rdBase.BlockLogs():
        out.UpdatePropertyCache(strict=False)
    return out.GetMol()


def _r1_side(mol: Chem.Mol, side: str) -> tuple[Chem.Mol, list[tuple[str, str, str, str]]]:
    """R1 on every odd fragment of one side it makes even."""
    out, repairs = mol, []
    for atoms, frag in _fragments(mol):
        fix = repair_candidate(frag) if is_odd(frag) else None
        if fix is None:
            continue
        fixed = _recharged(frag, [fix[:2]])
        if not is_odd(fixed):
            out = _recharged(out, [(atoms[fix[0]], fix[1])])
            repairs.append((side, fix[2], *_before_after(frag, fixed)))
    return out, repairs


def _before_after(frag: Chem.Mol, fixed: Chem.Mol) -> tuple[str, str]:
    return chem.unmapped_canonical(frag), chem.unmapped_canonical(fixed)


def _dative_side(mol: Chem.Mol) -> list[tuple[tuple[int, ...], Chem.Mol, list[tuple[int, int]]]]:
    """(fragment atoms, fragment, its re-charges) of every fragment of one side the dative law
    changes, each within the side's net charge; [] when together they would take the side below
    zero (the side stays as drawn: no fragment order decides which one is lowered)."""
    budget = max(0, chem.formal_charge(mol))
    plan = [(atoms, frag, dative_lowering(frag, budget)) for atoms, frag in _fragments(mol)]
    plan = [entry for entry in plan if entry[2]]
    return plan if _lowered(plan) <= budget else []


def _lowered(plan) -> int:
    return sum(f.GetAtomWithIdx(i).GetFormalCharge() - q for _, f, ch in plan for i, q in ch)


def dative_step(
    reactants: Chem.Mol, products: Chem.Mol
) -> tuple[Chem.Mol, Chem.Mol, list[tuple[str, str, str, str]]]:
    """Both sides with the dative law applied, and its repairs; unchanged unless both sides are
    lowered by the same total charge."""
    plans = {"reactants": _dative_side(reactants), "products": _dative_side(products)}
    if _lowered(plans["reactants"]) != _lowered(plans["products"]):
        return reactants, products, []
    sides, repairs = {"reactants": reactants, "products": products}, []
    for side, plan in plans.items():
        for atoms, frag, changes in plan:
            sides[side] = _recharged(sides[side], [(atoms[i], q) for i, q in changes])
            repairs.append((side, DATIVE_RULE, *_before_after(frag, _recharged(frag, changes))))
    return sides["reactants"], sides["products"], repairs


def side_codes(mol: Chem.Mol) -> tuple[list[DomainCode], bool]:
    """(codes of the fragments of one side, whether an even fragment is METAL_SUSPECT)."""
    codes, suspect = [], False
    for _, frag in _fragments(mol):
        if is_placeholder(frag):
            codes.append(DomainCode.METAL_PLACEHOLDER)
        elif is_odd(frag):
            if (code := fallthrough_code(frag)) is not None:
                codes.append(code)
        elif has_drawn_radical(frag):
            codes.append(DomainCode.RADICAL_DRAWN)
        else:
            suspect = suspect or is_metal_suspect(frag)
    return codes, suspect


def screen_step(reactants: Chem.Mol, products: Chem.Mol, repair: bool = True) -> Screen:
    """Repair (R1, then the dative law), then code both sides: the step's code is the first of
    `PRECEDENCE` that any fragment has. Repaired sides are re-parsed from their canonical mapped
    SMILES (sanitized as at ingest; `chem.SanitizeError` propagates) before they are coded."""
    r, p, repairs = reactants, products, []
    if repair:
        r, r_rep = _r1_side(r, "reactants")
        p, p_rep = _r1_side(p, "products")
        r, p, d_rep = dative_step(r, p)
        repairs = r_rep + p_rep + d_rep
    if any(s == "reactants" for s, *_ in repairs):
        r = chem.parse_mapped(chem.canonical_mapped_smiles(r))
    if any(s == "products" for s, *_ in repairs):
        p = chem.parse_mapped(chem.canonical_mapped_smiles(p))
    (r_codes, r_sus), (p_codes, p_sus) = side_codes(r), side_codes(p)
    codes = set(r_codes + p_codes)
    code = next((c for c in PRECEDENCE if c in codes), None)
    classes = set()
    if any(rule == DATIVE_RULE for _, rule, _, _ in repairs):
        classes.add(str(DomainClass.REPAIRED_DATIVE))
    if code is None and (r_sus or p_sus):  # a flag on the training target only
        classes.add(str(DomainClass.METAL_SUSPECT))
    return Screen(r, p, code, repairs, classes)
