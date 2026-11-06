"""The closed-shell domain laws at ingest (`ksmech.data.domain`): the R1 repair, the dative law,
the placeholder law and the METAL_SUSPECT flag (open-shell plan P0a step 6), and the drawing
repairs V1, A1 and the named beta-lactam repair (P1), on two PMechDB steps the repairs once
touched and on hand-written fixtures, never other dataset SMILES."""

import json
from dataclasses import replace

import pytest
from rdkit import Chem

from ksmech.data import chem, domain
from ksmech.data.domain import (
    A1_RULE,
    BL_RULE,
    DATIVE_RULE,
    V1_IMIDAZOLYL,
    V1_METHYL,
    DomainClass,
    DomainCode,
    admit_config,
    d_count,
    dative_lowering,
    dative_step,
    domain_classes,
    fallthrough_code,
    is_odd,
    is_placeholder,
    join_classes,
    named_repairs,
    open_shell_classes,
    oxidation_state,
    parity_law,
    repair_candidate,
    screen_step,
    side_codes,
    spin_law,
    uncompensated,
)
from ksmech.data.ingest import IngestResult
from ksmech.data.resonance import training_steps
from ksmech.data.schema import Arrow, ElementaryStep, SinkKind, Source, SourceKind
from ksmech.data.stats import class_counts, compute_stats, domain_counts
from ksmech.data.validation import build_step
from tests.fixtures.molecules import MOLECULES, OPEN_SHELL

# the two steps the explicit repair rules touch (PMechDB, as ingested on 2026-10-21)
SILYL_SHIFT = (  # pmechdb-c7de295e11f86278: a spurious -1 on a four-bonded Si, both sides
    "C[Si-:10](C)(C)[C:11](C)=[C+:20]CC(CCC1=CC=CC=C1)[O][Ti-]([Cl])([Cl])([Cl])[Cl]>>"
    "C[Si-:10](C)(C)[C:20](CC(CCC1=CC=CC=C1)[O][Ti-]([Cl])([Cl])([Cl])[Cl])=[C+:11]C",
    [Arrow(SourceKind.BOND, (10, 11), SinkKind.BOND, (10, 20))],
)
SILYL_TRANSFER = (  # pmechdb-bec502e1804f7d61: O+ -> Ti in the products, Ti alkoxide reactant
    "C=[C+]C[Si:20](C)(C)C.C[O:10][Ti]([Cl])([Cl])([Cl])[Cl]>>"
    "C=[C+]C[Si-:20](C)(C)(C)[O+:10](C)[Ti]([Cl])([Cl])([Cl])[Cl]",
    [Arrow(SourceKind.LONE_PAIR, (10,), SinkKind.ATOM, (20,))],
)
PME3 = "[P+](C)(C)C"
# hand-written steps without arrows (ReactMech-like): repaired by the dative law
L1PD_OXIDATIVE_ADDITION = (  # [Pd][P+]R3 + PhBr -> Pd(II), both sides drawn +1 too high
    "[Pd:1][P+:2](C)(C)C.[Br:3][c:4]1ccccc1>>[Br:3][Pd:1]([c:4]1ccccc1)[P+:2](C)(C)C"
)
NI_OXIDATIVE_ADDITION = (  # Ni(PMe3)2 + MeCl -> Ni(II), both sides +2 with two P+
    "[Ni:1]([P+:2](C)(C)C)[P+:3](C)(C)C.[Cl:4][CH3:5]>>"
    "[Cl:4][Ni:1]([CH3:5])([P+:2](C)(C)C)[P+:3](C)(C)C"
)
ONE_SIDE_ONLY = (  # the onium -> Cu bond only in the reactants: the two sides would differ
    "[CH3:1][C:2]([CH3:3])=[O+:4][Cu:5][Cl:6].[CH3:7][Mg:8][Br:9]>>"
    "[CH3:1][C:2]([CH3:3])([CH3:7])[O:4][Cu:5][Cl:6].[Mg+:8][Br:9]"
)
LIGANDLESS_TI = "[CH2:1]=[O+:2][Ti-:3].[H-:4]>>[H:4][CH2:1][O:2][Ti-:3]"


def frag(smiles: str) -> Chem.Mol:
    return chem.parse_mapped(smiles)


def metal(smiles: str, symbol: str) -> Chem.Atom:
    return next(a for a in frag(smiles).GetAtoms() if a.GetSymbol() == symbol)


def unmapped_step(rxn: str) -> ElementaryStep:
    return build_step(
        step_id="x", source=Source.REACTMECH, rxn_smiles=rxn, arrows=[], require_arrows=False
    )


def same_on_both_sides(smiles: str):
    mol = frag(smiles)
    return screen_step(mol, Chem.Mol(mol))


@pytest.mark.parametrize(
    ("smiles", "odd"),
    [
        ("CC(C)=O", False),
        ("[Br][Cu][Br]", True),  # Cu(II), d9
        ("C=[O+][Ti]", True),
        ("[Na+]", False),
        ("[CH3]", False),  # a radical RDKit marks
    ],
)
def test_odd_electron_count(smiles, odd):
    assert is_odd(frag(smiles)) is odd


# ---- the dative law


def test_dative_law_completes_the_monoligated_pd0_on_both_sides():
    step = unmapped_step(L1PD_OXIDATIVE_ADDITION)
    assert step.domain_exclusion is None and step.domain_class == DomainClass.REPAIRED_DATIVE
    rows = json.loads(step.charge_repair)["repairs"]
    assert [(r["side"], r["rule"]) for r in rows] == [
        ("reactants", DATIVE_RULE),
        ("products", DATIVE_RULE),
    ]
    assert "[Pd-:1]" in step.reactants_smiles and "[Pd-:1]" in step.products_smiles
    pd0 = metal(step.reactants_smiles, "Pd")
    pd2 = metal(step.products_smiles, "Pd")
    assert (oxidation_state(pd0), oxidation_state(pd2)) == (0, 2)
    for side in (step.reactants_smiles, step.products_smiles):
        assert chem.formal_charge(frag(side)) == 0  # the +1 the drawing carried is gone


def test_dative_law_recharges_an_even_two_onium_nickel_by_two():
    step = unmapped_step(NI_OXIDATIVE_ADDITION)
    assert "[Ni-2:1]" in step.reactants_smiles and "[Ni-2:1]" in step.products_smiles
    ni0, ni2 = metal(step.reactants_smiles, "Ni"), metal(step.products_smiles, "Ni")
    assert (oxidation_state(ni0), oxidation_state(ni2), d_count(ni2)) == (0, 2, 8)
    # the Ni(II) d8 product is flagged; the repair and the flag both stay on the step
    assert step.domain_exclusion is None
    assert domain_classes(step.domain_class) == {"REPAIRED_DATIVE", "METAL_SUSPECT"}
    assert step.domain_class == "REPAIRED_DATIVE+METAL_SUSPECT"


@pytest.mark.parametrize(
    "smiles",
    [
        "C=[O+][Li]",  # Li+ on a carbonyl in a +1 pool: even with u = 1, so j = 0
        "[OH2+][Mg]OC",  # the same: one unit would leave it odd
        "C[O+](C)B(Br)Br.[Br-]",  # BBr2+ adduct with its bromide: no positive charge to lower
        "Br[Pd-](Br)(C)" + PME3,  # the Pd already carries the -1: u = 0
    ],
)
def test_dative_law_leaves_correctly_charged_adducts_alone(smiles):
    screen = same_on_both_sides(smiles)
    assert not screen.repairs and DomainClass.REPAIRED_DATIVE not in screen.classes


def test_dative_law_needs_the_same_total_on_both_sides():
    step = unmapped_step(ONE_SIDE_ONLY)
    assert step.charge_repair is None and step.domain_class is None
    # the odd Cu(II) adduct stays out: since P8 ReactMech re-reads odd metals under data.spin,
    # and this unscanned Cu(II) species has no confirmed row
    assert step.domain_exclusion == DomainCode.MULTIPLICITY_UNTABULATED


def test_fragments_that_would_overdraw_the_side_leave_it_as_drawn():
    two = frag("[Pd]" + PME3 + ".[Pd]" + PME3 + ".[Cl-]")  # +1: room for one of the two
    reactants, products, repairs = dative_step(two, Chem.Mol(two))
    assert not repairs and Chem.MolToSmiles(reactants) == Chem.MolToSmiles(two)
    room = frag("[Pd]" + PME3 + ".[Pd]" + PME3)  # +2: both
    assert len(dative_step(room, Chem.Mol(room))[2]) == 4


def test_dative_lowering_takes_the_largest_even_making_j_within_the_budget():
    onium_ti = frag("C=[C+]C[Si-](C)(C)(C)[O+](C)[Ti]([Cl])([Cl])([Cl])[Cl]")  # odd, u = 1
    [(idx, charge)] = dative_lowering(onium_ti, budget=1)
    assert onium_ti.GetAtomWithIdx(idx).GetSymbol() == "Ti" and charge == -1
    assert dative_lowering(onium_ti, budget=0) == []  # the side's net charge stays >= 0
    two = frag("Cl[Ni](C)(" + PME3 + ")" + PME3)  # even, u = 2
    assert [q for _, q in dative_lowering(two, budget=2)] == [-2]
    assert dative_lowering(two, budget=1) == []  # j = 1 would leave it odd
    odd_two = frag("[Pd](c1ccccc1)(" + PME3 + ")[NH+](C)C")  # odd, u = 2: j = 1, Pd(II)+
    assert [q for _, q in dative_lowering(odd_two, budget=5)] == [-1]


def test_r1_removes_a_spurious_charge_on_a_four_bonded_group_14_atom():
    silyl = "CC(=[C+]CC(CCc1ccccc1)[O][Ti-]([Cl])([Cl])([Cl])[Cl])[Si-](C)(C)C"
    atom, charge, rule = repair_candidate(frag(silyl))
    assert rule.startswith("R1") and frag(silyl).GetAtomWithIdx(atom).GetSymbol() == "Si"
    assert charge == 0
    assert repair_candidate(frag("C[O][Ti]([Cl])([Cl])([Cl])[Cl]")) is None
    assert fallthrough_code(frag("C[O][Ti]([Cl])([Cl])([Cl])[Cl]")) is DomainCode.CHARGE_UNREPAIRED
    assert fallthrough_code(frag("O=[P-](Cl)(Cl)Cl")) is None  # no metal: left to the QM layer


def test_step_repaired_on_both_sides():
    rxn, arrows = SILYL_SHIFT
    step = build_step(step_id="x", source=Source.PMECHDB, rxn_smiles=rxn, arrows=arrows)
    assert step.domain_exclusion is None  # Ti(IV) ate complex once the Si charge is gone
    note = json.loads(step.charge_repair)
    assert [r["side"] for r in note["repairs"]] == ["reactants", "products"]
    assert all(r["rule"].startswith("R1") for r in note["repairs"])
    assert "[Si-:10]" in note["reactants_before"] and "[Si-" not in step.reactants_smiles
    assert step.domain_class is None  # R1 is a repair, not a domain class
    for side in (step.reactants_smiles, step.products_smiles):
        assert not any(is_odd(f) for f in chem.fragments(chem.parse_mapped(side)))


def test_an_onium_on_one_side_only_leaves_the_step_as_drawn():
    rxn, arrows = SILYL_TRANSFER
    step = build_step(step_id="y", source=Source.PMECHDB, rxn_smiles=rxn, arrows=arrows)
    assert step.charge_repair is None  # only the products have an onium -> Ti bond
    assert step.domain_exclusion == DomainCode.CHARGE_UNREPAIRED
    assert step.products_smiles == chem.canonicalize_mapped(rxn.split(">>")[1])


# ---- the placeholder law


def test_placeholders_are_one_ligand_metals_whatever_the_parity():
    assert is_placeholder(frag("C=[O+][Ti]"))  # odd; the onium is not compensated
    assert is_placeholder(frag("C[O][Ti-]"))  # even: Ti(0) with one X ligand
    assert is_placeholder(frag("C=[Cu]"))  # a carbenoid, ligands dropped
    assert is_placeholder(frag("COC(=O)[C](=[Cu])c1ccccc1"))
    assert not is_placeholder(frag("[Pd-]" + PME3))  # ML1: monoligated Pd(0)
    assert not is_placeholder(frag("[Cl][Zn+]"))  # MX+ with the charge on the metal
    assert not is_placeholder(frag("[Ag+]")) and not is_placeholder(frag("[Pd]"))  # free
    assert not is_placeholder(frag("C=[NH+][CuH2-]"))  # three neighbours with the two H
    assert not is_placeholder(frag("C[Mg]"))  # s-block
    na_cu = "O=C1C[N+]2(CP(=O)([O][Na])[O][Na])CC(=O)[O][Cu-]2[O]1"
    assert not is_placeholder(frag(na_cu))  # O-Na drawn covalently is not a bare metal
    assert fallthrough_code(frag(na_cu)) is DomainCode.OPEN_SHELL_METAL


def test_a_ligandless_metal_on_either_side_excludes_the_step():
    step = unmapped_step(LIGANDLESS_TI)
    assert step.domain_exclusion == DomainCode.METAL_PLACEHOLDER and step.charge_repair is None
    assert step.domain_class is None  # flags mark the training target only


@pytest.mark.parametrize(
    ("smiles", "code"),
    [
        ("[Cl][Zn+].CC(C)=O", None),
        ("[Ag+].CC(=O)[O-]", None),
        ("C=[Cu]", DomainCode.METAL_PLACEHOLDER),
        ("C[Sn]", DomainCode.METAL_PLACEHOLDER),  # radical electrons: the placeholder wins
    ],
)
def test_placeholder_codes(smiles, code):
    assert same_on_both_sides(smiles).code == code


def test_side_codes_and_untouched_even_fragments():
    side = frag("CC(C)=O.[Br][Fe-]([Br])([Br])[Br].C=[O+][Zn]")
    codes, suspect = side_codes(side)
    assert sorted(codes) == [DomainCode.METAL_PLACEHOLDER, DomainCode.OPEN_SHELL_METAL]
    assert not suspect
    screen = screen_step(side, Chem.Mol(side))  # net charge 0: no dative re-charge
    assert not screen.repairs and Chem.MolToSmiles(screen.reactants) == Chem.MolToSmiles(side)


@pytest.mark.parametrize(
    "smiles",
    [
        "C[O+](B(Br)Br)[C:20]",  # a methyl drawn without its hydrogens
        "[Sn-][O+:21]=[CH2:20]",  # a compensated one-ligand tin, drawn with radical electrons
        "CC(C)(C)[O-].[K]",  # potassium without its charge
    ],
)
def test_atoms_drawn_as_radicals_are_outside_the_closed_shell_domain(smiles):
    codes, _ = side_codes(frag(smiles))
    assert codes == [DomainCode.RADICAL_DRAWN]


def test_closed_shell_and_odd_fragments_keep_their_codes():
    assert side_codes(frag("CC(C)=O.C[O-].[K+]"))[0] == []
    codes, _ = side_codes(frag("[Br][Fe-]([Br])([Br])[Br]"))
    assert codes == [DomainCode.OPEN_SHELL_METAL]  # odd: the earlier rule, not RADICAL_DRAWN


# ---- METAL_SUSPECT and the oxidation-state bookkeeping


@pytest.mark.parametrize(
    ("smiles", "flagged"),
    [
        ("[O]=[Cr]([OH])[OH]", True),  # Cr(IV), d2
        ("Cl[Ni-2](Cl)(" + PME3 + ")" + PME3, True),  # NiCl2(PMe3)2: Ni(II), d8
        ("Cl[Pd-2](Cl)(" + PME3 + ")" + PME3, False),  # second row
        ("Cl[Ti](Cl)(Cl)Cl", False),  # Ti(IV), d0
        ("[Li][Cu](C)C", False),  # Gilman cuprate: Cu(I), d10 (Li-Cu by electronegativity)
    ],
)
def test_metal_suspect_flags_an_open_first_row_d_shell(smiles, flagged):
    screen = same_on_both_sides(smiles)
    assert screen.code is None and not screen.repairs
    assert (DomainClass.METAL_SUSPECT in screen.classes) is flagged


@pytest.mark.parametrize(
    ("smiles", "symbol", "state"),
    [
        ("[Ni-2](" + PME3 + ")" + PME3, "Ni", 0),  # a neutral NiL2
        ("Cl[Pd-2](Cc1ccccc1)(" + PME3 + ")" + PME3, "Pd", 2),  # Pd(II)L2(Cl)(Bn)
        ("[Pd-]" + PME3, "Pd", 0),  # monoligated Pd(0)
        ("Cl[Fe-](Cl)(Cl)Cl", "Fe", 3),
        ("[Li][Cu](C)C", "Cu", 1),
        ("C[P](C)(C)->[Pd]Cl", "Pd", 1),  # a dative bond is an L ligand
    ],
)
def test_oxidation_states(smiles, symbol, state):
    assert oxidation_state(metal(smiles, symbol)) == state


def test_oxidation_state_shows_a_fragment_mis_charged_by_two():
    drawn = "Cl[Ni](C)(" + PME3 + ")" + PME3  # Ni(II)L2XR drawn without the two -1
    ni = metal(drawn, "Ni")
    assert (oxidation_state(ni), uncompensated(ni)) == (4, 2)  # reads Ni(IV): two units high
    [(idx, charge)] = dative_lowering(frag(drawn), budget=2)
    fixed = Chem.RWMol(frag(drawn))
    fixed.GetAtomWithIdx(idx).SetFormalCharge(charge)
    assert oxidation_state(fixed.GetAtomWithIdx(idx)) == 2


# ---- the P1 drawing repairs: V1, A1 and the named beta-lactam repair

METHYL_TRANSFER = (  # BBr3 demethylation: the methyl drawn [C:20] on both sides
    "c1ccccc1[O+:21](B(Br)Br)[C:20].[Br-:10]>>c1ccccc1[O:21]B(Br)Br.[Br:10][C:20]",
    [
        Arrow(SourceKind.LONE_PAIR, (10,), SinkKind.ATOM, (20,)),
        Arrow(SourceKind.BOND, (20, 21), SinkKind.ATOM, (21,)),
    ],
)
POTASSIUM_SPECTATOR = (  # an alkoxide's counter-ion drawn as a neutral atom on both sides
    "CC(C)(C)[O-:10].C[O:21][H:20].[K]>>CC(C)(C)[O:10][H:20].C[O-:21].[K]",
    [
        Arrow(SourceKind.LONE_PAIR, (10,), SinkKind.ATOM, (20,)),
        Arrow(SourceKind.BOND, (20, 21), SinkKind.ATOM, (21,)),
    ],
)
BETA_LACTAM_STEP = "pmechdb-65a6c92b04e820b5"
BETA_LACTAM = (  # an N-acyliminium adds an alkyne; the ring N drawn without its H on both sides
    "CC1C(=O)[N+:21]=[CH:20]1.C[C:11]#[CH:10]>>CC1C(=O)[N:21][CH:20]1[CH:10]=[C+:11]C",
    [
        Arrow(SourceKind.BOND, (10, 11), SinkKind.BOND, (10, 20)),
        Arrow(SourceKind.BOND, (20, 21), SinkKind.ATOM, (21,)),
    ],
)
CDI = "O=C(n1[c]ncc1)n1[c]ncc1"


def pmechdb_step(rxn_arrows, step_id: str = "x") -> ElementaryStep:
    rxn, arrows = rxn_arrows
    return build_step(step_id=step_id, source=Source.PMECHDB, rxn_smiles=rxn, arrows=arrows)


def no_radicals(smiles: str) -> bool:
    return not any(a.GetNumRadicalElectrons() for a in frag(smiles).GetAtoms())


def test_v1_completes_a_methyl_drawn_without_its_hydrogens():
    step = pmechdb_step(METHYL_TRANSFER)  # validated with the arrows replayed
    assert step.domain_exclusion is None and step.domain_class == DomainClass.REPAIRED_V1
    assert "[CH3:20]" in step.reactants_smiles and "[CH3:20]" in step.products_smiles
    rows = json.loads(step.charge_repair)["repairs"]
    assert [(r["side"], r["rule"]) for r in rows] == [
        ("reactants", V1_METHYL),
        ("products", V1_METHYL),
    ]
    assert step.step_id == "x"  # a repair never changes the step id
    assert no_radicals(step.reactants_smiles) and no_radicals(step.products_smiles)


def test_v1_completes_the_imidazolyl_carbons_of_cdi():
    screen = same_on_both_sides(CDI)
    assert screen.code is None and screen.classes == {DomainClass.REPAIRED_V1}
    assert {r[1] for r in screen.repairs} == {V1_IMIDAZOLYL}
    ring_c = [a for a in screen.reactants.GetAtoms() if a.GetSymbol() == "C" and a.IsInRing()]
    assert sorted(a.GetTotalNumHs() for a in ring_c) == [1, 1, 1, 1, 1, 1]
    assert no_radicals(chem.canonical_mapped_smiles(screen.reactants))


@pytest.mark.parametrize(
    "smiles",
    [
        "CC(O[O])O",  # a peroxyl
        "C[C](C)[O][Na]",  # a ketyl, possibly sodium acetone ketyl: structure cannot tell
        "Cl[C]Cl",  # a carbene
        "[CH]1C=CC=C[CH]1[Na]",  # the Birch radical
        "C[CH2]",  # a C with one radical electron and one neighbour
        "c1cc[c]cc1",  # a phenyl radical: a 6-ring C
    ],
)
def test_v1_and_a1_leave_genuine_radicals_alone(smiles):
    screen = same_on_both_sides(smiles)
    assert not screen.repairs and screen.code == DomainCode.RADICAL_DRAWN


def test_the_birch_step_is_not_repaired():
    rxn = "[CH]1C=CC=C[CH:11]1[Na:10]>>[CH]1C=CC=C[CH-:11]1.[Na+:10]"
    step = unmapped_step(rxn)
    assert step.charge_repair is None and step.domain_exclusion == DomainCode.RADICAL_DRAWN


@pytest.mark.parametrize(
    ("reactants", "products"),
    [
        ("C[O+:21](C)[C:20].[Br-:10]", "C[O:21]C.[Br:10][CH3:20]"),  # mapped, one side only
        ("C[O+](C)[C].[Br-]", "COC.[Br]C"),  # unmapped, one side only
        ("[Na:10].[Cl:11]C", "[Na:10][Cl:11].[CH3]"),  # an alkali atom that reacts
    ],
)
def test_an_atom_drawn_differently_on_the_two_sides_is_not_repaired(reactants, products):
    screen = screen_step(frag(reactants), frag(products))
    assert not [r for r in screen.repairs if r[1] in (V1_METHYL, A1_RULE)]
    assert screen.code == DomainCode.RADICAL_DRAWN


def test_a1_charges_a_bare_alkali_spectator():
    step = pmechdb_step(POTASSIUM_SPECTATOR)
    assert step.domain_exclusion is None and step.domain_class == DomainClass.REPAIRED_A1
    assert "[K+]" in step.reactants_smiles and "[K+]" in step.products_smiles
    for side in (step.reactants_smiles, step.products_smiles):
        assert chem.formal_charge(frag(side)) == 0
    na = same_on_both_sides("CC[O-].[Na]")
    assert {r[1] for r in na.repairs} == {A1_RULE} and na.code is None
    assert not same_on_both_sides("CC[O-].[Na+]").repairs  # drawn right: nothing to do


def test_the_named_beta_lactam_repair_applies_to_its_step_only():
    assert [r["map"] for r in named_repairs()[BETA_LACTAM_STEP]] == [21]
    step = pmechdb_step(BETA_LACTAM, step_id=BETA_LACTAM_STEP)  # arrows replayed
    assert step.domain_exclusion is None and step.domain_class == DomainClass.REPAIRED_BL
    assert "[NH+:21]" in step.reactants_smiles and "[NH:21]" in step.products_smiles
    assert {r["rule"] for r in json.loads(step.charge_repair)["repairs"]} == {BL_RULE}
    other = pmechdb_step(BETA_LACTAM)
    assert other.charge_repair is None and other.domain_exclusion is None
    # as drawn, P3's laws read a conserved N radical the arrows touch: the repair is named
    assert other.domain_class == DomainClass.RADICAL_REACTIVE


# ---- the fixed molecules, the open-shell fixtures


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_the_fixed_molecules_pass_the_screen_unchanged(name):
    screen = same_on_both_sides(MOLECULES[name])
    assert screen.code is None and not screen.repairs and not screen.classes
    assert Chem.MolToSmiles(screen.reactants) == Chem.MolToSmiles(frag(MOLECULES[name]))


PAIRED_FIXTURES = {"ozone_drawn", "ozone_charge_separated", "n2_drawn"}  # P8: singlet species


@pytest.mark.parametrize("name", sorted(set(OPEN_SHELL) - PAIRED_FIXTURES))
def test_open_shell_fixtures_are_excluded_never_repaired_or_flagged(name):
    screen = same_on_both_sides(OPEN_SHELL[name][0])
    assert screen.code in (DomainCode.RADICAL_DRAWN, DomainCode.OPEN_SHELL_METAL)
    assert not screen.repairs and not screen.classes


@pytest.mark.parametrize("name", sorted(PAIRED_FIXTURES))
def test_singlet_open_shell_fixtures_are_redrawn_paired(name):
    """P8's pairing rule: ozone and N2 drawn with odd-N atoms are decided singlet by the
    species table and redrawn closed shell on both sides, counted as PAIRED_DRAWING."""
    screen = same_on_both_sides(OPEN_SHELL[name][0])
    assert screen.code is None and screen.classes == {str(DomainClass.PAIRED_DRAWING)}
    assert {rule for _, rule, _, _ in screen.repairs} == {domain.PAIRED_RULE}
    assert not any(a.GetNumRadicalElectrons() for a in screen.reactants.GetAtoms())


# ---- P3: open shells admitted by counted laws (PMechDB's admit_by_rule)

SPECTATOR_RADICAL = (  # an alcohol deprotonated beside its own C radical, drawn alike on both sides
    "C[CH:5]C[O:21][H:20].[OH-:10]>>C[CH:5]C[O-:21].[OH:10][H:20]",
    [
        Arrow(SourceKind.LONE_PAIR, (10,), SinkKind.ATOM, (20,)),
        Arrow(SourceKind.BOND, (20, 21), SinkKind.ATOM, (21,)),
    ],
)


def test_a_doublet_spectator_drawn_alike_on_both_sides_is_admitted():
    step = pmechdb_step(SPECTATOR_RADICAL)
    assert step.domain_exclusion is None
    assert step.domain_class == DomainClass.RADICAL_SPECTATOR
    assert training_steps([step]) == [step]


def test_a_source_that_excludes_open_shells_keeps_radical_drawn():
    rxn, arrows = SPECTATOR_RADICAL
    step = build_step(step_id="x", source=Source.REACTMECH, rxn_smiles=rxn, arrows=arrows)
    assert step.domain_exclusion == DomainCode.RADICAL_DRAWN and step.domain_class is None


@pytest.mark.parametrize(
    ("reactants", "products", "code"),
    [
        ("[CH3:1][Cl:2]", "[CH3:1].[Cl:2]", DomainCode.SPIN_NOT_CONSERVED),  # homolysis
        ("[CH2:1]=[O:2].[H-:3]", "[CH2:1]([H:3])[O:2]", DomainCode.SPIN_NOT_CONSERVED),
        ("[C]=O.C[O:1][H:2]", "[C]=O.C[O-:1].[H+:2]", None),  # 0 from the table (P4: stands)
        ("Cl[C]Cl.C[O:1][H:2]", "Cl[C]Cl.C[O-:1].[H+:2]", None),
        ("[Cl][Cu][Cl].C[O:1][H:2]", "[Cl][Cu][Cl].C[O-:1].[H+:2]", None),  # metal_v1: 1, 1
        ("[I][Cu][I].C[O:1][H:2]", "[I][Cu][I].C[O-:1].[H+:2]", DomainCode.MULTIPLICITY_AMBIGUOUS),
        (
            "[Cl][Fe]([Cl])[Cl].C[O:1][H:2]",
            "[Cl][Fe]([Cl])[Cl].C[O-:1].[H+:2]",
            DomainCode.MULTIPLICITY_AMBIGUOUS,  # FeCl3: the species table does not confirm 5
        ),  # fmt: skip
        (
            "[O][N+]([O])=O.C[O:1][H:2]",
            "[O][N+]([O])=O.C[O-:1].[H+:2]",
            DomainCode.MULTIPLICITY_AMBIGUOUS,
        ),  # fmt: skip
        ("[CH2]CCC[CH2].[OH-]", "[CH2]CCC[CH2].[OH-]", DomainCode.MULTIPLICITY_UNTABULATED),
        ("[O][O-].[H+]", "[O]O", None),
        ("[O][O].C", "[O][O].C", None),  # a triplet spectator: the table's 2 is RDKit's 2
    ],
)
def test_the_spin_law(reactants, products, code):
    assert spin_law(frag(reactants), frag(products)) == code


def test_the_parity_law_refuses_an_odd_atom_entry():
    assert parity_law([(0, 2), (3, -2)]) and not parity_law([(0, 2), (3, -1)])


def test_an_odd_atom_entry_of_the_arrows_edit_is_not_elementary(monkeypatch):
    from ksmech.arrows.delta_be import ReactionMatrix
    from ksmech.data import validation

    real = validation.dataset_matrix

    def odd(step):
        mol, prod, r, removed = real(step)
        return mol, prod, ReactionMatrix(r.bonds, ((0, 1),)), removed

    monkeypatch.setattr(validation, "dataset_matrix", odd)
    assert pmechdb_step(SPECTATOR_RADICAL).domain_exclusion == DomainCode.NOT_ELEMENTARY


def test_an_r_entry_on_a_radical_atom_makes_the_step_reactive():
    reactants, products = frag("C[CH:5]C[O:21][H:20]"), frag("C[CH:5]C[O-:21].[H+:20]")
    index = chem.map_to_index(reactants)
    on_o = [(index[20], index[21], -1)]
    on_radical = [(min(index[5], index[21]), max(index[5], index[21]), 1)]
    spectator, reactive = {DomainClass.RADICAL_SPECTATOR}, {DomainClass.RADICAL_REACTIVE}
    assert open_shell_classes(reactants, products, on_o, []) == spectator
    assert open_shell_classes(reactants, products, on_radical, []) == reactive
    assert open_shell_classes(reactants, products, [], [(index[5], 2)]) == reactive


def _beside(spectator: str) -> tuple[str, list[Arrow]]:
    """An alcohol deprotonated by OH- beside an unmapped `spectator`, drawn alike on both
    sides."""
    rxn = f"{spectator}.C[O:21][H:20].[OH-:10]>>{spectator}.C[O-:21].[OH:10][H:20]"
    return rxn, SPECTATOR_RADICAL[1]


@pytest.mark.parametrize(
    ("spectator", "exclusion", "classes"),
    [
        ("[Cl][Cu][Cl]", None, {DomainClass.OSM_CU2}),
        ("[Br][Fe-]([Br])([Br])[Br]", None, {DomainClass.OSM_FE3}),
        ("[I][Cu][I]", DomainCode.MULTIPLICITY_AMBIGUOUS, set()),  # REDOX_INCOMPATIBLE
        ("C[NH+](C)[CuH2-]", DomainCode.MULTIPLICITY_AMBIGUOUS, set()),
        ("[Cl][Fe]([Cl])[Cl]", DomainCode.MULTIPLICITY_AMBIGUOUS, set()),  # not confirmed
        ("Cl[C]Cl", None, {DomainClass.CARBENE_SINGLET}),
        ("[C]=O", None, {DomainClass.CARBENE_SINGLET}),
        ("[O][O]", None, {DomainClass.RADICAL_SPECTATOR}),  # a triplet: stays an open shell
    ],
)
def test_pmechdb_admits_metals_and_paired_carbenes_by_the_spin_laws(spectator, exclusion, classes):
    step = pmechdb_step(_beside(spectator))
    assert step.domain_exclusion == exclusion
    assert domain_classes(step.domain_class) == {str(c) for c in classes}


def test_a_source_that_excludes_open_shell_metals_keeps_open_shell_metal():
    rxn, arrows = _beside("[Cl][Cu][Cl]")
    step = build_step(step_id="x", source=Source.MECH_USPTO, rxn_smiles=rxn, arrows=arrows)
    assert step.domain_exclusion == DomainCode.OPEN_SHELL_METAL and step.domain_class is None


def test_a_fragment_code_the_laws_do_not_read_keeps_the_step_out():
    """CuCl2 is re-read, but an odd fragment whose only metal is d10 (ZnCl3 drawn neutral,
    CHARGE_UNREPAIRED) keeps the step out under its own code."""
    step = pmechdb_step(_beside("[Cl][Cu][Cl].[Cl][Zn]([Cl])[Cl]"))
    assert step.domain_exclusion == DomainCode.CHARGE_UNREPAIRED


def test_a_curated_step_keeps_radical_drawn_with_its_reason(monkeypatch):
    monkeypatch.setattr(
        domain, "open_shell_curation", lambda source: {"cur": str(DomainClass.RADICAL_AMBIGUOUS)}
    )
    step = pmechdb_step(SPECTATOR_RADICAL, step_id="cur")
    assert step.domain_exclusion == DomainCode.RADICAL_DRAWN
    assert step.domain_class == DomainClass.RADICAL_AMBIGUOUS


def test_the_curation_files_hold_step_ids_and_reasons_only():
    rows = json.loads((domain.CURATION_DIR / "pmechdb_open_shell.json").read_text())["entries"]
    assert {r["reason"] for r in rows} <= {str(c) for c in domain.CURATED_CLASSES}
    assert all(set(r) == {"step_id", "reason", "decided", "why"} for r in rows)
    assert all(r["step_id"].startswith("pmechdb-") for r in rows)
    assert domain.open_shell_curation("pmechdb") == {r["step_id"]: r["reason"] for r in rows}
    assert domain.open_shell_curation("reactmech") == {}
    assert ("pmechdb", "pmechdb-5d007b38296827b6") in domain.multi_radical_exemptions()


def test_the_ingest_report_counts_admitted_open_shells():
    result = IngestResult(Source.PMECHDB)
    result.add("s0", SPECTATOR_RADICAL[0], lambda: pmechdb_step(SPECTATOR_RADICAL, "s0"))
    report = result.report()
    assert report["open_shell"] == {
        "admitted": {"RADICAL_SPECTATOR": 1},
        "admitted_with_u_ge_2": [],
        "admitted_with_hund_species": 0,
        "curated_out": {},
    }
    assert report["spin"]["spin_table"] == "spin_table_v2"


# ---- domain_class, the admit set, the counts


def test_domain_class_values_join_and_split():
    assert join_classes([]) is None
    assert join_classes(["METAL_SUSPECT", "REPAIRED_DATIVE"]) == "REPAIRED_DATIVE+METAL_SUSPECT"
    assert join_classes(["Z_LATER", "METAL_SUSPECT"]) == "METAL_SUSPECT+Z_LATER"
    assert domain_classes(None) == frozenset()
    assert admit_config("reactmech") == [
        "METAL_SUSPECT", "OSM_CU2", "OSM_FE3", "PAIRED_DRAWING", "REPAIRED_A1", "REPAIRED_BL",
        "REPAIRED_DATIVE", "REPAIRED_V1"
    ]  # fmt: skip


def test_training_steps_drop_excluded_and_unadmitted_steps_and_keep_the_gap():
    base = ElementaryStep("s", Source.PMECHDB, "p", 0, "[CH4:1]", "[CH4:1]")
    steps = [
        replace(base, step_id="a", step_index=0),
        replace(base, step_id="b", step_index=1, resonance_step=True),
        replace(base, step_id="c", step_index=2, domain_exclusion="OPEN_SHELL_METAL"),
        replace(base, step_id="d", step_index=3, domain_class="REPAIRED_DATIVE+METAL_SUSPECT"),
        replace(base, step_id="e", step_index=4),
    ]
    kept = training_steps(steps)  # the default admit set takes every P0a class
    assert [(s.step_id, s.step_index) for s in kept] == [("a", 0), ("d", 2), ("e", 3)]
    strict = {"pmechdb": {"REPAIRED_DATIVE"}}
    assert [s.step_id for s in training_steps(steps, strict)] == ["a", "e"]
    counts = domain_counts(steps)
    assert counts["RESONANCE_STEP"] == 1 and counts["OPEN_SHELL_METAL"] == 1
    assert counts["METAL_PLACEHOLDER"] == 0
    assert class_counts(steps) == {"all": {"REPAIRED_DATIVE+METAL_SUSPECT": 1}, "not_admitted": {}}
    back = ElementaryStep.from_dict(steps[3].to_dict())
    assert back.domain_class == "REPAIRED_DATIVE+METAL_SUSPECT" and back.charge_repair is None


def test_domain_counts_list_every_distinct_code():
    base = ElementaryStep("s", Source.RMECHDB, None, None, "[CH4:1]", "[CH4:1]")
    steps = [replace(base, domain_exclusion="SPIN_RULE_PENDING"), base]
    counts = domain_counts(steps)
    assert counts["SPIN_RULE_PENDING"] == 1 and counts["METAL_PLACEHOLDER"] == 0


def test_ingest_report_counts_repairs_per_rule_and_domain_classes():
    result = IngestResult(Source.REACTMECH)
    cdi_hydrolysis = "[OH2:1].O=C(n1[c]ncc1)n1[c]ncc1>>[OH2:1].O=C(n1[c]ncc1)n1[c]ncc1"
    rxns = (L1PD_OXIDATIVE_ADDITION, NI_OXIDATIVE_ADDITION, LIGANDLESS_TI, cdi_hydrolysis)
    for k, rxn in enumerate(rxns):
        result.add(f"s{k}", rxn, lambda r=rxn: unmapped_step(r))
    report = result.report()
    assert report["charge_repairs_by_rule"] == {DATIVE_RULE: 2, V1_IMIDAZOLYL: 1}
    assert report["domain_class"] == {
        "REPAIRED_DATIVE": 1,
        "REPAIRED_DATIVE+METAL_SUSPECT": 1,
        "REPAIRED_V1": 1,
    }
    # the CDI line carries V1 as an identity line: NULL_STEP under the flow law (P6)
    assert report["domain_exclusions"] == {"METAL_PLACEHOLDER": 1, "NULL_STEP": 1}
    stats = compute_stats([result.steps[k] for k in (0, 3)])
    assert stats["charge_repairs_by_rule"] == {DATIVE_RULE: 1, V1_IMIDAZOLYL: 1}


# ---- real data: the laws on every stored step, from its sides as drawn


def _rescreened(step: ElementaryStep) -> ElementaryStep:
    """The step built again from the sides the source drew (before any stored repair), so the
    counts are the same on the P0 store and after the P0a re-ingest."""
    if step.charge_repair:
        note = json.loads(step.charge_repair)
        sides = note["reactants_before"], note["products_before"]
    else:
        sides = step.reactants_smiles, step.products_smiles
    return build_step(
        step_id=step.step_id,
        source=step.source,
        rxn_smiles=">>".join(sides),
        arrows=step.arrows,
        require_arrows=bool(step.arrows),
    )


# open-shell plans P0a step 6, P1, P3 and P4: (codes, repairs per rule, domain classes) over
# every stored step. P1's rules equal the P0 survey: C3_METHYL 9, BARE_ALKALI 4, BETA_LACTAM_N 1
# and, in the ReactMech pilot, CDI_IMIDAZOLYL 62. P3's laws split PMechDB's 30 RADICAL_DRAWN
# steps: 11 admitted, 12 + 2 by a law, 5 curated (they keep RADICAL_DRAWN). P4: the 12 held
# SPIN_RULE_PENDING are CARBENE_SINGLET (2 of them the CO resonance pairs); the 78
# OPEN_SHELL_METAL steps are OSM_CU2 23, OSM_FE3 19 (2 resonance pairs each) and 36
# MULTIPLICITY_AMBIGUOUS (12 Cu, REDOX_INCOMPATIBLE or a failed scan; 24 Fe the species table
# does not confirm). ReactMech kept OPEN_SHELL_METAL until P8, which re-reads its 2 odd-Pd steps.
P4_COUNTS = {
    "pmechdb": (
        {None: 12990, "MULTIPLICITY_AMBIGUOUS": 36, "METAL_PLACEHOLDER": 56, "RADICAL_DRAWN": 5,
         "CHARGE_UNREPAIRED": 1, "SPIN_NOT_CONSERVED": 2},
        {"R1 group-14 -1 removed": 1, V1_METHYL: 9, A1_RULE: 4, BL_RULE: 1},
        {"REPAIRED_V1": 9, "REPAIRED_A1": 4, "REPAIRED_BL": 1, "RADICAL_SPECTATOR": 11,
         "RADICAL_AMBIGUOUS": 3, "DRAWING_ERROR_UNREPAIRED": 2, "CARBENE_SINGLET": 12,
         "OSM_CU2": 23, "OSM_FE3": 19},
    ),
    "reactmech": (  # the pilot
        {None: 9998, "MULTIPLICITY_AMBIGUOUS": 2},  # P8: odd Pd re-read under data.spin
        {DATIVE_RULE: 254, V1_IMIDAZOLYL: 62},  # dative: 49 kept steps re-charged, 205 returned
        {"REPAIRED_DATIVE": 229, "REPAIRED_DATIVE+METAL_SUSPECT": 25, "METAL_SUSPECT": 10,
         "REPAIRED_V1": 62},
    ),
}  # fmt: skip
# P1 step 4: the steps that stay as P0a left them (the Sn ones by the placeholder law); P3's
# laws pass the Al and nitrone drawings, which the curation file keeps out, and hold back the P
SN_PLACEHOLDERS = 8
LEFT_AS_DRAWN = {
    "pmechdb-29a4a03883007779": DomainCode.RADICAL_DRAWN,  # Al of degree 3 and 2
    "pmechdb-ef20cdaad5325eca": DomainCode.RADICAL_DRAWN,  # the nitrone
    "pmechdb-e57aaee9d354b63c": DomainCode.SPIN_NOT_CONSERVED,  # a four-bonded P, u 1 -> 0
}


def _rescreened_store(source: str) -> list[ElementaryStep]:
    from rdkit import RDLogger

    from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_steps, steps_path

    RDLogger.DisableLog("rdApp.*")
    return [_rescreened(s) for s in read_steps(steps_path(source, DEFAULT_PROCESSED_DIR))]


@pytest.mark.realdata
@pytest.mark.parametrize("source", sorted(P4_COUNTS))
def test_the_domain_laws_give_the_p4_counts(source):
    from collections import Counter

    from ksmech.data.ingest import repair_rule_counts

    steps = _rescreened_store(source)
    codes, rules, classes = P4_COUNTS[source]
    assert dict(Counter(s.domain_exclusion for s in steps)) == codes
    assert dict(repair_rule_counts(steps)) == rules
    assert dict(Counter(s.domain_class for s in steps if s.domain_class)) == classes


@pytest.mark.realdata
def test_p1_leaves_the_tin_placeholders_and_the_other_drawings_as_they_were():
    steps = {s.step_id: s for s in _rescreened_store("pmechdb")}
    sn = [
        s for s in steps.values()
        if s.domain_exclusion == DomainCode.METAL_PLACEHOLDER
        and any(a.GetSymbol() == "Sn" for a in frag(s.reactants_smiles).GetAtoms())
    ]  # fmt: skip
    assert len(sn) == SN_PLACEHOLDERS
    for step_id, code in LEFT_AS_DRAWN.items():
        assert steps[step_id].domain_exclusion == code
