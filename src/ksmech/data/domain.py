"""Closed-shell domain laws at ingest: drawing repairs, exclusions and flags (spec v9.1,
'Domain coverage'; open-shell plan P0a step 6).

The QM layer runs closed-shell GFN2 on every fragment, so a fragment whose electron count (sum
of Z minus the total charge) is odd while RDKit marks no radical electron cannot be computed as
written (`qm.xtb_calc.resolve_charge_spin` raises SPIN_PARITY). Every step is screened by laws
that hold for every element and every parity, in this order:

1. **Repairs** (`CHARGE_REPAIRED`, logged in `charge_repair` and `charge_repairs.csv`), kept only
   if the repaired step still validates (charge balance, maps, arrow replay where arrows exist;
   `validation.screened`), otherwise the step is screened as drawn. First the drawing repairs of
   open-shell plan P1, each only on an atom drawn the same way on both sides (the same map
   number, or unmapped atoms of the same shape equally many on both sides), so an atom that
   reacts or a radical drawn on one side only is never touched:
   - V1, H completion of two signatures no radical reading fits: a C with 3 radical electrons
     and one neighbour (a methyl drawn `[C]`) gets 3 H, and the imidazol-2-yl C of CDI (a C in
     a 5-ring with two N neighbours, degree 2, 1 radical electron; RDKit does not flag it
     aromatic) gets 1 H. A C with 1 radical electron elsewhere (a ketyl, the Birch [CH]) is never
     completed;
   - A1: a bare neutral alkali atom (no neighbour, no H) gets charge +1;
   - the named per-step repairs of `repairs_v1.json` (the beta-lactam N, rule BL_NAMED), each
     its own decision in docs/decisions.md.

   Then, on the completed sides:
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
   set (Cu(II), Fe(III) adducts); **CHARGE_UNREPAIRED**: one whose only metals are d0 / d10 (a
   charge error, not an open shell). An odd fragment without any metal gets no code here; the
   QM layer reports it (SPIN_PARITY). A source whose `DomainPolicy` admits open-shell metals
   by rule (PMechDB, open-shell plan P4) re-reads an OPEN_SHELL_METAL step with the laws of 4:
   `data.spin` gives each odd metal fragment its count from the metal table (Cu(II) 1,
   Fe(III) high spin 5, each confirmed by the species table), or MULTIPLICITY_AMBIGUOUS /
   MULTIPLICITY_UNTABULATED, which then replace OPEN_SHELL_METAL as the exclusion; a step that
   passes is admitted as OSM_CU2 or OSM_FE3 (the open_shell_metal cohort). Its metal's d
   open-shell tokens stay masked: the metal is a Lewis acid or a spectator of a polar step.
4. **RADICAL_DRAWN**: an even fragment with an atom RDKit marks with unpaired electrons. In a
   polar dataset these are drawing errors (a methyl written `[C:20]`, K without its charge);
   read literally they are open shells (user decision of 2026-10-24); the step is excluded.
   A source whose `DomainPolicy` admits open shells by rule (PMechDB, open-shell plan P3)
   re-reads such a step with three counted laws (`validation.admit_open_shell`):
   - (u) every species' unpaired count under `data.spin.assign` (RDKit for u <= 1, the species
     table for u >= 2, the metal table for an odd metal fragment) is defined (else
     MULTIPLICITY_UNTABULATED / MULTIPLICITY_AMBIGUOUS) and the total is conserved across the
     step (else SPIN_NOT_CONSERVED). A count other than RDKit's stands since P4, whose QM
     records are computed at it under a tagged method (`qm.spin`); P3 held such steps as
     SPIN_RULE_PENDING, a code old artifacts still carry;
   - (R) every atom entry of R read from the dataset arrows is even (else NOT_ELEMENTARY): a
     polar step neither creates, destroys nor moves an unpaired electron;
   - (S) every open-shell reactant has an integral SOMO, checked by the sample builder on the
     tokens (SOMO_NOT_INTEGRAL).
   A step that passes is admitted with the classes of its species (`open_shell_classes`):
   RADICAL_SPECTATOR, or RADICAL_REACTIVE when an R entry touches a radical atom (one drawn
   with radical electrons in a species that stays an open shell under `data.spin`);
   CARBENE_SINGLET when the species table pairs a species drawn with radical electrons
   (Cl[C]Cl, the vinylidenes, CO drawn [C]=O read 0); OSM_CU2 / OSM_FE3 for a metal fragment.
   The source's curation file (`curation/<source>_open_shell.json`) can still exclude it: it
   then keeps RADICAL_DRAWN with the curated reason as its class (RADICAL_AMBIGUOUS,
   DRAWING_ERROR_UNREPAIRED).
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
import logging
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache
from pathlib import Path

from rdkit import Chem, rdBase

from ksmech.data import chem, spin
from ksmech.data.schema import ElementaryStep, Source


class DomainCode(StrEnum):
    METAL_PLACEHOLDER = "METAL_PLACEHOLDER"
    OPEN_SHELL_METAL = "OPEN_SHELL_METAL"
    CHARGE_UNREPAIRED = "CHARGE_UNREPAIRED"
    RADICAL_DRAWN = "RADICAL_DRAWN"
    # the open-shell laws of a source that admits by rule (P3), a reason each
    MULTIPLICITY_UNTABULATED = "MULTIPLICITY_UNTABULATED"  # a u >= 2 species the table lacks
    MULTIPLICITY_AMBIGUOUS = "MULTIPLICITY_AMBIGUOUS"  # the table assigns it no multiplicity
    SPIN_NOT_CONSERVED = "SPIN_NOT_CONSERVED"  # the total unpaired count changes
    SPIN_RULE_PENDING = "SPIN_RULE_PENDING"  # P3: the table re-assigned a species (old artifacts)
    NOT_ELEMENTARY = "NOT_ELEMENTARY"  # an odd atom entry of R, or no R from the arrows
    # the electron-flow law's ingest codes (open-shell plan P6, `resonance.classify_step`)
    NULL_STEP = "NULL_STEP"  # identical sides: R = 0 (never a resonance pair)
    NONCONJUGATED_SHIFT = "NONCONJUGATED_SHIFT"  # one identity key, every flow unbalanced
    # P8: a non-metal species with u >= 2 that is not a Hund case, or one the table decides
    # singlet that no closed-shell drawing reaches (`pairing.redraw`); exempt steps aside
    MULTI_RADICAL_SPECIES = "MULTI_RADICAL_SPECIES"


class DomainClass(StrEnum):
    """What a step's `domain_class` can hold (later phases add values): repairs first, flags
    after."""

    REPAIRED_DATIVE = "REPAIRED_DATIVE"  # the dative law re-charged an acceptor
    REPAIRED_V1 = "REPAIRED_V1"  # V1 completed a methyl or an imidazol-2-yl C with its H
    REPAIRED_A1 = "REPAIRED_A1"  # A1 gave a bare alkali atom its +1
    REPAIRED_BL = "REPAIRED_BL"  # the named beta-lactam N repair (repairs_v1.json)
    METAL_SUSPECT = "METAL_SUSPECT"  # an open-d-shell first-row metal computed as a singlet
    RADICAL_SPECTATOR = "RADICAL_SPECTATOR"  # admitted open shell, no R entry on a radical atom
    RADICAL_REACTIVE = "RADICAL_REACTIVE"  # admitted open shell, an R entry on a radical atom
    RADICAL_AMBIGUOUS = "RADICAL_AMBIGUOUS"  # curated out: the drawing has two readings
    DRAWING_ERROR_UNREPAIRED = "DRAWING_ERROR_UNREPAIRED"  # curated out: no repair law fits
    CARBENE_SINGLET = "CARBENE_SINGLET"  # the species table pairs a drawn carbene or CO (P4)
    OSM_CU2 = "OSM_CU2"  # an admitted Cu(II) fragment, u = 1 (the open_shell_metal cohort, P4)
    OSM_FE3 = "OSM_FE3"  # an admitted high-spin Fe(III) fragment, u = 5 (idem)
    PAIRED_DRAWING = "PAIRED_DRAWING"  # a singlet species redrawn closed-shell (P8, `pairing`)
    HIGH_SPIN_DRAWING = "HIGH_SPIN_DRAWING"  # a Hund-case species redrawn unpaired (P8)
    RADICAL_STEP = "RADICAL_STEP"  # a step of a one-electron source (rmechdb, P8)


PRECEDENCE = (
    DomainCode.METAL_PLACEHOLDER,
    DomainCode.OPEN_SHELL_METAL,
    DomainCode.CHARGE_UNREPAIRED,
    DomainCode.RADICAL_DRAWN,
    DomainCode.MULTIPLICITY_UNTABULATED,  # the open-shell law codes: never a fragment's code,
    DomainCode.MULTIPLICITY_AMBIGUOUS,  # listed so that every count report shows them
    DomainCode.SPIN_NOT_CONSERVED,
    DomainCode.SPIN_RULE_PENDING,
    DomainCode.MULTI_RADICAL_SPECIES,
    DomainCode.NOT_ELEMENTARY,
    DomainCode.NULL_STEP,  # step codes of the flow law: after every fragment and law code
    DomainCode.NONCONJUGATED_SHIFT,
)
CHARGE_REPAIRED = "CHARGE_REPAIRED"
R1_RULE = "R1 group-14 -1 removed"
DATIVE_RULE = "D dative onium->M"
V1_METHYL = "V1 methyl [C] +3H"
V1_IMIDAZOLYL = "V1 imidazol-2-yl C +H"
A1_RULE = "A1 bare alkali +1"
BL_RULE = "BL_NAMED"
PAIRED_RULE = "P paired drawing"  # P8: the drawings the spin decision asks for
HIGH_SPIN_RULE = "H high-spin drawing"
# the class each repair rule gives a step (R1 gives none: a charge typo, not a drawing choice)
RULE_CLASS = {
    DATIVE_RULE: DomainClass.REPAIRED_DATIVE,
    V1_METHYL: DomainClass.REPAIRED_V1,
    V1_IMIDAZOLYL: DomainClass.REPAIRED_V1,
    A1_RULE: DomainClass.REPAIRED_A1,
    BL_RULE: DomainClass.REPAIRED_BL,
    PAIRED_RULE: DomainClass.PAIRED_DRAWING,
    HIGH_SPIN_RULE: DomainClass.HIGH_SPIN_DRAWING,
}
REPAIRS_V1 = Path(__file__).with_name("repairs_v1.json")
CLASS_SEP = "+"
log = logging.getLogger(__name__)
# the classes each source admits into the training target (P0a: every class, both are flags)
CLOSED_SHELL_CLASSES = (  # the P0a and P1 classes, every source admits them
    DomainClass.REPAIRED_DATIVE,
    DomainClass.REPAIRED_V1,
    DomainClass.REPAIRED_A1,
    DomainClass.REPAIRED_BL,
    DomainClass.METAL_SUSPECT,
    DomainClass.PAIRED_DRAWING,  # P8: a closed-shell drawing, admitted wherever it occurs
)
ADMIT: dict[str, frozenset[str]] = {
    str(s): frozenset(map(str, CLOSED_SHELL_CLASSES)) for s in Source
}


class OpenShellPolicy(StrEnum):
    EXCLUDE = "exclude"  # a step with a drawn radical stays RADICAL_DRAWN
    ADMIT_BY_RULE = "admit_by_rule"  # re-read by the open-shell laws (module docstring, 4)


@dataclass(frozen=True)
class DomainPolicy:
    """What a source's ingest does with a domain phenomenon: drawn radicals (P3) and open-shell
    metals (P4). PMechDB admits both by rule; ReactMech, re-ingested under `data.spin` in P8,
    admits open-shell metals by rule (an odd metal without a row is MULTIPLICITY_AMBIGUOUS
    instead of OPEN_SHELL_METAL) and keeps drawn radicals out; rmechdb admits both.
    `open_shell_labels` is the source's elementary rule (P6), read by the labels and by the
    open-shell laws (P8): a source of one-electron steps (rmechdb) accepts atom entries of R of
    +-1 and +-2, and its spin is conserved only by a step whose R moves no single electron; the
    polar sources keep +-2. The source never
    becomes a sample input, a model option or a decoder argument."""

    open_shell: OpenShellPolicy = OpenShellPolicy.EXCLUDE
    open_shell_metal: OpenShellPolicy = OpenShellPolicy.EXCLUDE
    open_shell_labels: bool = False


POLICY: dict[str, DomainPolicy] = {str(s): DomainPolicy() for s in Source}
POLICY[str(Source.PMECHDB)] = DomainPolicy(
    OpenShellPolicy.ADMIT_BY_RULE, OpenShellPolicy.ADMIT_BY_RULE
)
POLICY[str(Source.REACTMECH)] = DomainPolicy(  # its re-ingest under data.spin (P8)
    OpenShellPolicy.EXCLUDE, OpenShellPolicy.ADMIT_BY_RULE
)
POLICY[str(Source.RMECHDB)] = DomainPolicy(  # one-electron steps (P8)
    OpenShellPolicy.ADMIT_BY_RULE, OpenShellPolicy.ADMIT_BY_RULE, open_shell_labels=True
)
CURATION_DIR = Path(__file__).with_name("curation")


def open_shell_labels(source: str | None) -> bool:
    """The source's elementary rule (`DomainPolicy.open_shell_labels`); False when unknown."""
    return POLICY.get(str(source), DomainPolicy()).open_shell_labels


OPEN_SHELL_CODES = frozenset(  # what the open-shell laws put in place of RADICAL_DRAWN
    [
        DomainCode.MULTIPLICITY_UNTABULATED,
        DomainCode.MULTIPLICITY_AMBIGUOUS,
        DomainCode.SPIN_NOT_CONSERVED,
        DomainCode.SPIN_RULE_PENDING,
        DomainCode.MULTI_RADICAL_SPECIES,
        DomainCode.NOT_ELEMENTARY,
    ]
)
ADMITTED_OPEN_SHELL = frozenset(
    [
        DomainClass.RADICAL_SPECTATOR,
        DomainClass.RADICAL_REACTIVE,
        DomainClass.CARBENE_SINGLET,
        DomainClass.HIGH_SPIN_DRAWING,
        DomainClass.RADICAL_STEP,
    ]
)
ADMITTED_METAL = frozenset([DomainClass.OSM_CU2, DomainClass.OSM_FE3])
for _source, _policy in POLICY.items():  # a source that admits open shells admits their classes
    if _policy.open_shell is OpenShellPolicy.ADMIT_BY_RULE:
        ADMIT[_source] = ADMIT[_source] | frozenset(map(str, ADMITTED_OPEN_SHELL))
    if _policy.open_shell_metal is OpenShellPolicy.ADMIT_BY_RULE:
        ADMIT[_source] = ADMIT[_source] | frozenset(map(str, ADMITTED_METAL))
CURATED_CLASSES = frozenset([DomainClass.RADICAL_AMBIGUOUS, DomainClass.DRAWING_ERROR_UNREPAIRED])

_S_BLOCK = frozenset([3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88])
ALKALI = frozenset([3, 11, 19, 37, 55, 87])  # group 1 without H
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


# ---- the P1 drawing repairs: V1, A1 and the named per-step repairs

# an atom edit: (rule, H added, formal charge added)
Edit = tuple[str, int, int]


def v1_edit(atom: Chem.Atom) -> Edit | None:
    """V1 on one atom: a neutral C with 3 radical electrons and one neighbour (a methyl drawn
    without its H) gets 3 H; a neutral C in a 5-ring with two N neighbours, degree 2 and 1
    radical electron (the imidazol-2-yl C of CDI) gets 1 H. Any other C is left alone."""
    if atom.GetAtomicNum() != 6 or atom.GetFormalCharge() != 0:
        return None
    radicals, degree = atom.GetNumRadicalElectrons(), atom.GetDegree()
    if radicals == 3 and degree == 1:
        return V1_METHYL, 3, 0
    nitrogens = all(n.GetAtomicNum() == 7 for n in atom.GetNeighbors())
    if radicals == 1 and degree == 2 and nitrogens and atom.IsInRingSize(5):
        return V1_IMIDAZOLYL, 1, 0
    return None


def a1_edit(atom: Chem.Atom) -> Edit | None:
    """A1 on one atom: a bare neutral alkali atom (no neighbour, no H) gets charge +1."""
    bare = atom.GetDegree() == 0 and atom.GetTotalNumHs() == 0
    if atom.GetAtomicNum() in ALKALI and atom.GetFormalCharge() == 0 and bare:
        return A1_RULE, 0, 1
    return None


def matched(
    reactants: Chem.Mol, products: Chem.Mol, rule: Callable[[Chem.Atom], Edit | None]
) -> tuple[dict[int, Edit], dict[int, Edit]]:
    """The edits `rule` makes on each side (atom index -> edit), kept only where the atom is
    drawn the same way on both sides: a mapped atom whose map number gets the same edit on the
    other side, and unmapped atoms when both sides hold equally many of the same element with
    the same edit."""
    sides = (reactants, products)
    edits = [{a.GetIdx(): e for a in m.GetAtoms() if (e := rule(a)) is not None} for m in sides]
    by_map, unmapped = [], []
    for mol, side in zip(sides, edits, strict=True):
        atoms = {i: mol.GetAtomWithIdx(i) for i in side}
        by_map.append({a.GetAtomMapNum(): side[i] for i, a in atoms.items() if a.GetAtomMapNum()})
        unmapped.append(
            Counter((a.GetSymbol(), side[i]) for i, a in atoms.items() if not a.GetAtomMapNum())
        )
    out = []
    for k, (mol, side) in enumerate(zip(sides, edits, strict=True)):
        keep = {}
        for i, edit in side.items():
            atom = mol.GetAtomWithIdx(i)
            if n := atom.GetAtomMapNum():
                same = by_map[1 - k].get(n) == edit
            else:
                key = (atom.GetSymbol(), edit)
                same = unmapped[0][key] == unmapped[1][key]
            if same:
                keep[i] = edit
        out.append(keep)
    return out[0], out[1]


@cache
def named_repairs() -> dict[str, list[dict]]:
    """The named per-step repairs of `repairs_v1.json`, by step id."""
    rows = json.loads(REPAIRS_V1.read_text())["repairs"]
    out: dict[str, list[dict]] = {}
    for row in rows:
        if row["rule"] not in RULE_CLASS:
            raise ValueError(f"repairs_v1.json: unknown rule {row['rule']!r}")
        out.setdefault(row["step_id"], []).append(row)
    return out


def named_edits(
    reactants: Chem.Mol, products: Chem.Mol, step_id: str | None
) -> tuple[dict[int, Edit], dict[int, Edit]]:
    """The named repairs of one step as edits on each side; a row whose map number is not on
    both sides is not applied (logged)."""
    out: tuple[dict[int, Edit], dict[int, Edit]] = ({}, {})
    for row in named_repairs().get(step_id or "", []):
        where = [chem.map_to_index(m).get(row["map"]) for m in (reactants, products)]
        if None in where:
            log.warning("named repair %s of %s: map %s not on both sides", row["rule"], step_id,
                        row["map"])  # fmt: skip
            continue
        edit = (row["rule"], int(row["change"].get("h", 0)), int(row["change"].get("charge", 0)))
        for side, idx in zip(out, where, strict=True):
            side[idx] = edit
    return out


def _edited(mol: Chem.Mol, edits: Mapping[int, Edit]) -> Chem.Mol:
    """`mol` with the edits applied: H added as an explicit count (NoImplicit, so the count
    stays as written), the charge shifted, the atom's radical electrons removed."""
    out = Chem.RWMol(mol)
    for idx, (_, h, q) in edits.items():
        atom = out.GetAtomWithIdx(idx)
        atom.SetNumExplicitHs(atom.GetTotalNumHs() + h)
        atom.SetNoImplicit(True)
        atom.SetFormalCharge(atom.GetFormalCharge() + q)
        atom.SetNumRadicalElectrons(0)
    with rdBase.BlockLogs():
        out.UpdatePropertyCache(strict=False)
    return out.GetMol()


def _edited_side(
    mol: Chem.Mol, side: str, edits: Mapping[int, Edit]
) -> tuple[Chem.Mol, list[tuple[str, str, str, str]]]:
    """One side with its edits, and a repair row per fragment and rule."""
    repairs = []
    for atoms, frag in _fragments(mol):
        local = {atoms.index(i): e for i, e in edits.items() if i in atoms}
        for rule in sorted({e[0] for e in local.values()}):
            repairs.append((side, rule, *_before_after(frag, _edited(frag, local))))
    return (_edited(mol, edits) if edits else mol), repairs


def completed_step(
    reactants: Chem.Mol, products: Chem.Mol, step_id: str | None = None
) -> tuple[Chem.Mol, Chem.Mol, list[tuple[str, str, str, str]]]:
    """Both sides with V1, A1 and the step's named repairs applied, and the repairs."""
    edits: tuple[dict[int, Edit], dict[int, Edit]] = ({}, {})
    for found in (
        matched(reactants, products, v1_edit),
        matched(reactants, products, a1_edit),
        named_edits(reactants, products, step_id),
    ):
        for side, new in zip(edits, found, strict=True):
            side.update(new)
    r, r_rep = _edited_side(reactants, "reactants", edits[0])
    p, p_rep = _edited_side(products, "products", edits[1])
    return r, p, r_rep + p_rep


def repair_rule_counts(steps: Iterable[ElementaryStep]) -> Counter[str]:
    """Steps per repair rule (a step repaired on both sides counts once)."""
    out: Counter[str] = Counter()
    for s in steps:
        if s.charge_repair:
            out.update({r["rule"] for r in json.loads(s.charge_repair)["repairs"]})
    return out


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


def open_shell_curation(source: str) -> dict[str, str]:
    """step id -> curated reason of `curation/<source>_open_shell.json` (empty without one)."""
    return dict(_curation(str(source)))


@cache
def _curation(source: str) -> tuple[tuple[str, str], ...]:
    path = CURATION_DIR / f"{source}_open_shell.json"
    if not path.exists():
        return ()
    rows = json.loads(path.read_text())["entries"]
    bad = [r["step_id"] for r in rows if r["reason"] not in CURATED_CLASSES]
    if bad:
        raise ValueError(f"{path}: reasons outside {sorted(CURATED_CLASSES)} for {bad}")
    return tuple((r["step_id"], r["reason"]) for r in rows)


def _odd_count(m: Chem.Mol) -> int:
    from ksmech.arrows.apply import nonbonding_electrons  # arrows reads data: lazy

    return sum(nonbonding_electrons(a) % 2 for a in m.GetAtoms())


def drawing_decision(frag: Chem.Mol) -> tuple[str | None, int]:
    """The drawing one fragment's spin asks for (open-shell plan P8, step 3): (PAIRED_RULE, 0)
    when the species table decides a species drawn with odd-N atoms singlet; (HIGH_SPIN_RULE,
    u) when a Hund row raises a drawing with fewer odd-N atoms to u (O=O -> [O][O]); (None, 0)
    otherwise. A metal fragment's spin is the metal table's (P4) and is never redrawn."""
    m = spin.kekule_copy(frag)
    if any(_is_metal(a) for a in m.GetAtoms()):
        return None, 0
    odd = _odd_count(m)
    if odd and spin.rdkit_unpaired(m) >= 2 and spin.assign(m).unpaired == 0:
        return PAIRED_RULE, 0
    raised, _ = spin.spin_of_drawing(m)
    if raised > spin.rdkit_unpaired(m) and raised >= 2 and hund_case(m):
        return HIGH_SPIN_RULE, raised  # not [O]: its own row is the drawing
    return None, 0


def hund_case(frag: Chem.Mol) -> bool:
    """The species table holds this drawing's species as a Hund case (one degenerate open
    group, `spin.is_hund_row`): its own row when it has one, else a Hund row with its v1 key
    (a Hund verdict holds for every drawing of the species, O=O and [O][O])."""
    from ksmech.data.resonance import sigma_key  # resonance reads domain: lazy

    entries = spin.species_table()["entries"]
    own = entries.get(chem.unmapped_canonical(frag))
    if own is not None:
        return spin.is_hund_row(own)
    key = sigma_key(frag)
    return any(spin.is_hund_row(e) and spin._v1_key(s) == key for s, e in entries.items())


def _apply_fragment(mol: Chem.Mol, atoms: Sequence[int], frag: Chem.Mol) -> Chem.Mol:
    """`mol` (Kekule) with the bonds, charges and radical electrons of `frag` (a redrawing of
    its fragment on `atoms`, Kekule) copied onto those atoms; H counts stay as drawn."""
    rw = Chem.RWMol(mol)
    for a in frag.GetAtoms():
        atom = rw.GetAtomWithIdx(atoms[a.GetIdx()])
        atom.SetNumExplicitHs(atom.GetTotalNumHs())
        atom.SetNoImplicit(True)
        atom.SetFormalCharge(a.GetFormalCharge())
        atom.SetNumRadicalElectrons(a.GetNumRadicalElectrons())
    for b in frag.GetBonds():
        i, j = atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()]
        rw.GetBondBetweenAtoms(i, j).SetBondType(b.GetBondType())
    out = rw.GetMol()
    with rdBase.BlockLogs():
        out.UpdatePropertyCache(strict=False)
    return out


def redrawn_side(mol: Chem.Mol, side: str) -> tuple[Chem.Mol, list[tuple[str, str, str, str]]]:
    """Every fragment of one side in the drawing its spin asks for (`drawing_decision`,
    `pairing.redraw`), with the repairs as (side, rule, before, after). A fragment the search
    cannot redraw stays as drawn (the spin law then names it). The side is returned Kekule
    when a fragment was redrawn (it is re-parsed before it is coded)."""
    from ksmech.data.pairing import redraw

    out, repairs = None, []
    for atoms, frag in _fragments(mol):
        rule, target = drawing_decision(frag)
        new = redraw(spin.kekule_copy(frag), target) if rule else None
        if new is None:
            continue
        Chem.Kekulize(new, clearAromaticFlags=True)
        out = _apply_fragment(spin.kekule_copy(mol) if out is None else out, atoms, new)
        repairs.append((side, rule, *_before_after(frag, new)))
    return (mol if out is None else out), repairs


def multi_radical_species(frag: Chem.Mol) -> bool:
    """MULTI_RADICAL_SPECIES (open-shell plan P8, step 3): a non-metal species with u >= 2
    after the decision (`spin.assign`, raised by `spin.spin_of_drawing`) that is not a Hund
    case (`hund_case`, the only u >= 2 form the tokens represent), or one the table decides
    singlet that is still drawn with odd-N atoms (no closed-shell drawing reached it: an
    open-shell singlet biradical). A species without a count is the spin law's
    MULTIPLICITY_UNTABULATED or MULTIPLICITY_AMBIGUOUS, not this."""
    m = spin.kekule_copy(frag)
    if any(_is_metal(a) for a in m.GetAtoms()):
        return False
    a = spin.assign(m)
    if not a.assigned:
        return False
    if a.unpaired == 0:  # the table's singlet (not RDKit's count of a hypervalent drawing)
        return a.source == spin.SPECIES and _odd_count(m) > 0
    u = max(a.unpaired, spin.spin_of_drawing(m)[0])
    return u >= 2 and not hund_case(m)


@cache
def multi_radical_exemptions() -> frozenset[tuple[str, str]]:
    """(source, step id) of every counted exemption of the MULTI_RADICAL_SPECIES policy
    (`curation/multi_radical_exemptions.json`; written in P3, read by P8's policy). An exempt
    step's species are kept as drawn: neither redrawn nor named multi-radical."""
    rows = json.loads((CURATION_DIR / "multi_radical_exemptions.json").read_text())["entries"]
    return frozenset((r["source"], r["step_id"]) for r in rows)


def is_exempt(step_id: str | None) -> bool:
    return step_id is not None and any(step_id == i for _, i in multi_radical_exemptions())


def side_spins(mol: Chem.Mol) -> tuple[list[int | str], list[int]]:
    """(`spin.species_u`, RDKit's count) of every fragment of one side."""
    frags = [f for _, f in _fragments(mol)]
    return [spin.species_u(f) for f in frags], [spin.rdkit_unpaired(f) for f in frags]


def spin_law(
    reactants: Chem.Mol, products: Chem.Mol, conserved: bool = True, exempt: bool = False
) -> DomainCode | None:
    """Law (u) of the open-shell admission, None when it holds: every species has a count, no
    species is a multi-radical one (`multi_radical_species`, unless the step is `exempt`), and,
    for a step whose R moves no single electron (`conserved`), the total is conserved."""
    (r, _), (p, _) = side_spins(reactants), side_spins(products)
    every = r + p
    for code in (DomainCode.MULTIPLICITY_UNTABULATED, DomainCode.MULTIPLICITY_AMBIGUOUS):
        if str(code) in every:
            return code
    frags = [f for mol in (reactants, products) for _, f in _fragments(mol)]
    if not exempt and any(multi_radical_species(f) for f in frags):
        return DomainCode.MULTI_RADICAL_SPECIES
    if conserved and sum(r) != sum(p):
        return DomainCode.SPIN_NOT_CONSERVED
    return None


def parity_law(lone: Iterable[tuple[int, int]]) -> bool:
    """Law (R): every atom entry (change of non-bonding electrons) of R is even."""
    return all(d % 2 == 0 for _, d in lone)


def _metal_class(row_id: str) -> str:
    """The domain class of an admitted metal row (`spin_tables/metal_v1.json`): OSM_<EL><OS>."""
    row = next(r for r in spin.metal_table()["rows"] if r["id"] == row_id)
    return str(DomainClass(f"OSM_{row['element'].upper()}{row['oxidation_state']}"))


def _side_spin(mol: Chem.Mol) -> tuple[set[int], set[str]]:
    """(atom indices of `mol` that are radical atoms, the P4 classes) of one side: a radical atom
    is drawn with radical electrons in a species `data.spin` leaves an open shell; a species
    the species table pairs gives CARBENE_SINGLET, a metal row its OSM class."""
    atoms, classes = set(), set()
    for idx, frag in _fragments(mol):
        a, drawn = spin.assign(frag), spin.rdkit_unpaired(frag)
        if a.source == spin.METAL and a.assigned:
            classes.add(_metal_class(a.row))
        elif a.assigned and a.unpaired == 0 and drawn:
            classes.add(str(DomainClass.CARBENE_SINGLET))
        elif a.assigned and a.unpaired:
            radical = [i for i, f in enumerate(frag.GetAtoms()) if f.GetNumRadicalElectrons()]
            atoms |= {idx[i] for i in radical}
    return atoms, classes


def open_shell_classes(
    reactants: Chem.Mol,
    products: Chem.Mol,
    bonds: Iterable[tuple[int, int, int]],
    lone: Iterable[tuple[int, int]],
) -> set[str]:
    """The classes of a step that passed laws (u) and (R) (module docstring, 4): the P4 classes
    of its species, and RADICAL_REACTIVE when an entry of R (reactant indices) touches a radical
    atom, RADICAL_SPECTATOR when it has radical atoms R does not touch."""
    r_atoms, r_classes = _side_spin(reactants)
    p_atoms, p_classes = _side_spin(products)
    index = chem.map_to_index(reactants)
    maps = {products.GetAtomWithIdx(i).GetAtomMapNum() for i in p_atoms}
    atoms = r_atoms | {index[m] for m in maps if m in index}
    classes = r_classes | p_classes
    if atoms:
        touched = {i for i, j, _ in bonds} | {j for i, j, _ in bonds} | {i for i, _ in lone}
        classes.add(
            str(DomainClass.RADICAL_REACTIVE if touched & atoms else DomainClass.RADICAL_SPECTATOR)
        )
    return classes


def max_species_unpaired(step: ElementaryStep) -> int:
    """The largest RDKit unpaired count of one species on either side of a step."""
    sides = (step.reactants_smiles, step.products_smiles)
    frags = [f for smi in sides for _, f in _fragments(chem.parse_mapped(smi))]
    return max((spin.rdkit_unpaired(f) for f in frags), default=0)


def _reparsed(mol: Chem.Mol, side: str, repairs: Sequence[tuple[str, ...]]) -> Chem.Mol:
    """A repaired side re-parsed from its canonical mapped SMILES (sanitized as at ingest;
    `chem.SanitizeError` propagates), so its radical electrons and rings are recomputed."""
    if any(s == side for s, *_ in repairs):
        return chem.parse_mapped(chem.canonical_mapped_smiles(mol))
    return mol


def screen_step(
    reactants: Chem.Mol,
    products: Chem.Mol,
    repair: bool = True,
    step_id: str | None = None,
    redraw: bool = True,
) -> Screen:
    """Repair (V1, A1 and the named repairs, then R1, then the dative law, then, with
    `redraw`, the spin redrawings of `redrawn_side` unless the step is an exemption), then
    code both sides: the step's code is the first of `PRECEDENCE` that any fragment has.
    Repaired sides are re-parsed after the drawing repairs and again before they are coded."""
    r, p, repairs = reactants, products, []
    if repair:
        r, p, v_rep = completed_step(r, p, step_id)
        r, p = _reparsed(r, "reactants", v_rep), _reparsed(p, "products", v_rep)
        r, r_rep = _r1_side(r, "reactants")
        p, p_rep = _r1_side(p, "products")
        r, p, d_rep = dative_step(r, p)
        s_rep = []
        if redraw and not is_exempt(step_id):
            r, r_s = redrawn_side(r, "reactants")
            p, p_s = redrawn_side(p, "products")
            s_rep = r_s + p_s
        later = r_rep + p_rep + d_rep + s_rep
        r, p = _reparsed(r, "reactants", later), _reparsed(p, "products", later)
        repairs = v_rep + later
    (r_codes, r_sus), (p_codes, p_sus) = side_codes(r), side_codes(p)
    codes = set(r_codes + p_codes)
    code = next((c for c in PRECEDENCE if c in codes), None)
    classes = {str(RULE_CLASS[rule]) for _, rule, _, _ in repairs if rule in RULE_CLASS}
    if code is None and (r_sus or p_sus):  # a flag on the training target only
        classes.add(str(DomainClass.METAL_SUSPECT))
    return Screen(r, p, code, repairs, classes)
