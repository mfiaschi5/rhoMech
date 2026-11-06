"""Open-shell plan P8 step 3: one spin and domain law for every source.

The species table decides a drawn species' multiplicity; the pairing rule redraws a singlet
drawn with odd-N atoms closed shell (PAIRED_DRAWING) and a Hund-case species drawn paired
unpaired (HIGH_SPIN_DRAWING); a species left with u >= 2 that is not a Hund case, or a singlet
no closed-shell drawing reaches, is MULTI_RADICAL_SPECIES in every source, with the counted
exemption table.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from ksmech.data import chem, domain, spin
from ksmech.data.domain import DomainClass, DomainCode, screen_step
from ksmech.data.ingest import IngestResult
from ksmech.data.pairing import redraw
from ksmech.data.resonance import identity_key, sigma_key
from ksmech.data.schema import Source
from ksmech.data.validation import build_step

EXEMPT = "pmechdb-5d007b38296827b6"
DISTANT = "CC([O])(C=CO)C[O]"  # a distant alkoxyl biradical the table decides singlet
P_BENZYNE = "[c]1cc[c]cc1"


def mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    return Chem.MolFromSmiles(smiles, params)


def mapped(smiles: str) -> str:
    """`smiles` with every atom mapped 100, 101, ... (explicit H stay unmapped)."""
    m = mol(smiles)
    for i, a in enumerate(m.GetAtoms()):
        a.SetAtomMapNum(100 + i)
    return Chem.MolToSmiles(m)


def spectator_step(species: str, source: Source, step_id: str = "x"):
    """A closed-shell proton transfer beside `species`, drawn alike on both sides."""
    s = mapped(species)
    rxn = f"{s}.[CH3:1][O:2][H:3].[OH-:4]>>{s}.[CH3:1][O-:2].[OH:4][H:3]"
    return build_step(
        step_id=step_id, source=source, rxn_smiles=rxn, arrows=[], require_arrows=False
    )


def same_on_both_sides(smiles: str, step_id: str | None = None):
    m = chem.parse_mapped(mapped(smiles))
    return screen_step(m, Chem.Mol(m), step_id=step_id)


# ---- the pairing rule


def test_n2_drawn_with_radicals_is_redrawn_as_n_triple_n_and_counted():
    screen = same_on_both_sides("[N]=[N]")
    assert screen.code is None and screen.classes == {str(DomainClass.PAIRED_DRAWING)}
    assert [r[1:] for r in screen.repairs] == [(domain.PAIRED_RULE, "[N]=[N]", "N#N")] * 2
    step = spectator_step("[N]=[N]", Source.RMECHDB)
    result = IngestResult(Source.RMECHDB)
    result.add("x", "", lambda: step)
    assert result.report()["charge_repairs_by_rule"] == {domain.PAIRED_RULE: 1}


def test_ozone_chain_is_redrawn_charge_separated_given_its_singlet_row():
    assert spin.species_table()["entries"]["[O]O[O]"]["unpaired"] == 0
    new = redraw(mol("[O]O[O]"), 0)
    assert Chem.MolToSmiles(new) == Chem.MolToSmiles(mol("[O-][O+]=O"))


def test_the_criegee_intermediate_keeps_its_charges_on_the_oxygens():
    assert Chem.MolToSmiles(redraw(mol("[CH2]O[O]"), 0)) == Chem.MolToSmiles(mol("C=[O+][O-]"))


def test_o2_drawn_paired_keys_and_draws_as_the_triplet_through_the_hund_row():
    assert identity_key(mol("O=O")) == identity_key(mol("[O][O]"))
    assert domain.drawing_decision(mol("O=O")) == (domain.HIGH_SPIN_RULE, 2)
    screen = same_on_both_sides("O=O")
    assert {(r[1], r[2], r[3]) for r in screen.repairs} == {
        (domain.HIGH_SPIN_RULE, "O=O", "[O][O]")
    }
    assert not any(domain.multi_radical_species(f) for f in chem.fragments(screen.reactants))


def test_the_triplet_and_atomic_oxygen_rows_are_their_own_drawings():
    for smiles in ("[O][O]", "[O]"):
        assert domain.drawing_decision(mol(smiles)) == (None, 0)
        assert not domain.multi_radical_species(mol(smiles))


def test_c_double_c_does_not_take_a_ch2_ch2_row(monkeypatch):
    ethene = spin._formula(mol("C=C"))
    rows = spin._raising_rows() | {ethene: [("[CH2][CH2]", 2, False)]}
    monkeypatch.setattr(spin, "_raising_rows", lambda: rows)
    assert spin.spin_of_drawing(mol("C=C"))[0] == 0
    assert domain.drawing_decision(mol("C=C")) == (None, 0)
    assert identity_key(mol("C=C")) == sigma_key(mol("C=C"))


def test_carbon_monoxide_is_admitted_as_drawn_with_the_v1_key():
    assert identity_key(mol("[C]=O")) == sigma_key(mol("[C]=O"))
    assert domain.drawing_decision(mol("[C]=O")) == (None, 0)
    for source in (Source.PMECHDB, Source.RMECHDB):
        step = spectator_step("[C]=O", source)
        assert step.domain_exclusion is None and step.charge_repair is None


def test_a_metal_fragment_is_never_redrawn():
    assert domain.drawing_decision(mol("C[C](C)[O][Mg]")) == (None, 0)
    assert redraw(mol("C[C](C)[O][Mg]"), 0) is None


# ---- MULTI_RADICAL_SPECIES


@pytest.mark.parametrize("species", [DISTANT, P_BENZYNE])
@pytest.mark.parametrize("source", [Source.PMECHDB, Source.RMECHDB, Source.REACTMECH])
def test_a_distant_centre_biradical_is_multi_radical_in_every_source(species, source):
    assert spin.species_table()["entries"][chem.unmapped_canonical(mol(species))]["unpaired"] == 0
    assert redraw(spin.kekule_copy(mol(species)), 0) is None  # no move pairs distant centres
    assert domain.multi_radical_species(mol(species))
    assert spectator_step(species, source).domain_exclusion == DomainCode.MULTI_RADICAL_SPECIES


def test_a_triplet_that_is_not_a_hund_case_is_multi_radical():
    carbene = "COC(=O)[C]C(=O)OC"  # the table's triplet carbene: two non-degenerate SOMOs
    assert spin.species_table()["entries"][carbene]["unpaired"] == 2
    assert domain.multi_radical_species(mol(carbene))
    step = spectator_step(carbene, Source.RMECHDB)
    assert step.domain_exclusion == DomainCode.MULTI_RADICAL_SPECIES


def test_an_untabulated_biradical_keeps_the_untabulated_code():
    step = spectator_step("[CH2]CCC[CH2]", Source.RMECHDB)
    assert step.domain_exclusion == DomainCode.MULTIPLICITY_UNTABULATED


def test_the_exempt_step_is_neither_redrawn_nor_named_multi_radical():
    """5d007b38 (Et3B + O2): its adduct, decided singlet by the table, is kept as drawn. The
    spin law still holds the step back (the triplet O2 does not conserve u against a singlet
    adduct), the difference P3 reported; whether to admit it is the user's question."""
    assert domain.is_exempt(EXEMPT) and not domain.is_exempt("pmechdb-x")
    adduct = "CC[B-](CC)(CC)[O+][O]"
    exempt = same_on_both_sides(adduct, step_id=EXEMPT)
    assert not exempt.repairs and exempt.code == DomainCode.RADICAL_DRAWN
    other = same_on_both_sides(adduct, step_id="pmechdb-x")
    assert {r[1] for r in other.repairs} == {domain.PAIRED_RULE}
    rxn = (
        "[CH3:1][CH2:2][B:3]([CH2:4][CH3:5])[CH2:6][CH3:7].[O:8][O:9]>>"
        "[CH3:1][CH2:2][B-:3]([CH2:4][CH3:5])([CH2:6][CH3:7])[O+:8][O:9]"
    )
    step = build_step(
        step_id=EXEMPT, source=Source.PMECHDB, rxn_smiles=rxn, arrows=[], require_arrows=False
    )
    assert step.charge_repair is None
    assert step.domain_exclusion == DomainCode.SPIN_NOT_CONSERVED


# ---- the policies


def test_rmechdb_admits_one_electron_steps_with_radical_step():
    rxn = "[CH3:1][O:2][O:3][CH3:4]>>[CH3:1][O:2].[O:3][CH3:4]"  # O-O homolysis
    step = build_step(
        step_id="h", source=Source.RMECHDB, rxn_smiles=rxn, arrows=[], require_arrows=False
    )
    assert step.domain_exclusion is None
    assert str(DomainClass.RADICAL_STEP) in domain.domain_classes(step.domain_class)
    polar = build_step(
        step_id="h", source=Source.PMECHDB, rxn_smiles=rxn, arrows=[], require_arrows=False
    )
    # a polar source conserves u whatever R is (law u before law R): 0 -> 2 is held back
    assert polar.domain_exclusion == DomainCode.SPIN_NOT_CONSERVED


def test_a_closed_shell_rmechdb_step_carries_radical_step_too():
    step = spectator_step("C", Source.RMECHDB)
    assert step.domain_exclusion is None
    assert step.domain_class == str(DomainClass.RADICAL_STEP)


def test_a_spectator_carbon_radical_is_not_h_completed():
    """V1 completes only a C with 3 radical electrons and one neighbour."""
    step = spectator_step("C[CH]C", Source.RMECHDB)
    assert step.charge_repair is None and step.domain_exclusion is None


def test_reactmech_reads_odd_metals_by_the_spin_law_and_keeps_radicals_out():
    policy = domain.POLICY[str(Source.REACTMECH)]
    assert policy.open_shell is domain.OpenShellPolicy.EXCLUDE
    assert policy.open_shell_metal is domain.OpenShellPolicy.ADMIT_BY_RULE
    assert spectator_step("C[CH]C", Source.REACTMECH).domain_exclusion == DomainCode.RADICAL_DRAWN


def test_every_new_code_and_class_is_ordered_and_admitted():
    assert DomainCode.MULTI_RADICAL_SPECIES in domain.PRECEDENCE
    assert DomainCode.MULTI_RADICAL_SPECIES in domain.OPEN_SHELL_CODES
    for source in Source:
        assert str(DomainClass.PAIRED_DRAWING) in domain.ADMIT[str(source)]
    for cls in (DomainClass.RADICAL_STEP, DomainClass.HIGH_SPIN_DRAWING):
        assert str(cls) in domain.ADMIT[str(Source.RMECHDB)]
        assert str(cls) not in domain.ADMIT[str(Source.REACTMECH)]


# ---- the spin census before QM (step 5)


@pytest.mark.parametrize(
    "smiles, allowed",
    [
        ("C", True),
        ("C[CH2]", True),
        ("[O][O]", True),  # Hund
        ("[O]", True),  # Hund
        ("[C]=O", True),  # the table's singlet, paired in the BE drawing
        ("[Cl][Fe-]([Cl])([Cl])[Cl]", True),  # a metal table's count
        (P_BENZYNE, False),  # a singlet no closed-shell drawing reaches
        ("COC(=O)[C]C(=O)OC", False),  # a triplet that is not a Hund case
        ("[CH2]CCC[CH2]", False),  # RDKit u = 2 without a count
    ],
)
def test_the_census_refuses_u2_species_without_a_hund_verdict(smiles, allowed):
    from ksmech.qm.batch import census_allows

    assert census_allows(chem.unmapped_canonical(mol(smiles))) is allowed


def test_the_census_lets_an_exemptions_species_through():
    from ksmech.qm.batch import _census

    adduct = "CC[B-](CC)(CC)[O+][O]"
    kept, refused = _census({adduct, "C"}, exempt=set())
    assert kept == ["C"] and [r["reason"] for r in refused] == ["SPIN_CENSUS_REFUSED"]
    assert _census({adduct, "C"}, exempt={adduct}) == (["C", adduct], [])
