"""The closed-shell domain screen at ingest (`ksmech.data.domain`), on PMechDB molecules."""

import json
from dataclasses import replace

import pytest
from rdkit import Chem

from ksmech.data import chem
from ksmech.data.domain import (
    DomainCode,
    fallthrough_code,
    is_odd,
    is_placeholder,
    repair_candidate,
    screen_side,
)
from ksmech.data.resonance import training_steps
from ksmech.data.schema import Arrow, ElementaryStep, SinkKind, Source, SourceKind
from ksmech.data.stats import domain_counts
from ksmech.data.validation import build_step

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


def frag(smiles: str) -> Chem.Mol:
    return chem.parse_mapped(smiles)


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


def test_placeholders_are_bare_non_s_block_metals():
    assert is_placeholder(frag("C=[O+][Ti]"))
    assert is_placeholder(frag("COC(=O)[C](=[Cu])c1ccccc1"))  # a carbenoid, ligands dropped
    assert not is_placeholder(frag("C=[NH+][CuH2-]"))  # three neighbours with the two H
    na_cu = "O=C1C[N+]2(CP(=O)([O][Na])[O][Na])CC(=O)[O][Cu-]2[O]1"
    assert not is_placeholder(frag(na_cu))  # O-Na drawn covalently is not a bare metal
    assert fallthrough_code(frag(na_cu)) is DomainCode.OPEN_SHELL_METAL


def test_repair_rules():
    silyl = "CC(=[C+]CC(CCc1ccccc1)[O][Ti-]([Cl])([Cl])([Cl])[Cl])[Si-](C)(C)C"
    atom, charge, rule = repair_candidate(frag(silyl))
    assert rule.startswith("R1") and frag(silyl).GetAtomWithIdx(atom).GetSymbol() == "Si"
    assert charge == 0
    onium_ti = "C=[C+]C[Si-](C)(C)(C)[O+](C)[Ti]([Cl])([Cl])([Cl])[Cl]"
    atom, charge, rule = repair_candidate(frag(onium_ti))
    assert rule == "R2 dative onium->Ti" and charge == -1
    for cu in ("C=[O+][Cu][O]S(=O)(=O)C(F)(F)F", "CC#[N+][Cu-]([Br])[Br]"):
        assert repair_candidate(frag(cu)) is None  # never for Cu, Fe, Co, Ni
        assert fallthrough_code(frag(cu)) is DomainCode.OPEN_SHELL_METAL
    assert repair_candidate(frag("C[O][Ti]([Cl])([Cl])([Cl])[Cl]")) is None  # no onium
    assert fallthrough_code(frag("C[O][Ti]([Cl])([Cl])([Cl])[Cl]")) is DomainCode.CHARGE_UNREPAIRED
    assert fallthrough_code(frag("O=[P-](Cl)(Cl)Cl")) is None  # no metal: left to the QM layer


def test_side_screen_codes_and_untouched_even_fragments():
    side = frag("CC(C)=O.[Br][Fe-]([Br])([Br])[Br].C=[O+][Zn]")
    out, codes, repairs = screen_side(side, "reactants")
    assert sorted(codes) == [DomainCode.METAL_PLACEHOLDER, DomainCode.OPEN_SHELL_METAL]
    assert not repairs and Chem.MolToSmiles(out) == Chem.MolToSmiles(side)


def test_step_repaired_on_both_sides():
    rxn, arrows = SILYL_SHIFT
    step = build_step(step_id="x", source=Source.PMECHDB, rxn_smiles=rxn, arrows=arrows)
    assert step.domain_exclusion is None  # Ti(IV) ate complex once the Si charge is gone
    note = json.loads(step.charge_repair)
    assert [r["side"] for r in note["repairs"]] == ["reactants", "products"]
    assert all(r["rule"].startswith("R1") for r in note["repairs"])
    assert "[Si-:10]" in note["reactants_before"] and "[Si-" not in step.reactants_smiles
    for side in (step.reactants_smiles, step.products_smiles):
        assert not any(is_odd(f) for f in chem.fragments(chem.parse_mapped(side)))


def test_repair_that_unbalances_the_step_is_dropped():
    rxn, arrows = SILYL_TRANSFER
    step = build_step(step_id="y", source=Source.PMECHDB, rxn_smiles=rxn, arrows=arrows)
    assert step.charge_repair is None  # only the products have an onium -> Ti bond
    assert step.domain_exclusion == DomainCode.CHARGE_UNREPAIRED
    assert step.products_smiles == chem.canonicalize_mapped(rxn.split(">>")[1])


def test_training_steps_drop_excluded_steps_and_keep_the_gap():
    base = ElementaryStep("s", Source.PMECHDB, "p", 0, "[CH4:1]", "[CH4:1]")
    steps = [
        replace(base, step_id="a", step_index=0),
        replace(base, step_id="b", step_index=1, resonance_step=True),
        replace(base, step_id="c", step_index=2, domain_exclusion="OPEN_SHELL_METAL"),
        replace(base, step_id="d", step_index=3),
    ]
    kept = training_steps(steps)
    assert [(s.step_id, s.step_index) for s in kept] == [("a", 0), ("d", 2)]
    counts = domain_counts(steps)
    assert counts["RESONANCE_STEP"] == 1 and counts["OPEN_SHELL_METAL"] == 1
    assert counts["METAL_PLACEHOLDER"] == 0
    back = ElementaryStep.from_dict(steps[2].to_dict())
    assert back.domain_exclusion == "OPEN_SHELL_METAL" and back.charge_repair is None
