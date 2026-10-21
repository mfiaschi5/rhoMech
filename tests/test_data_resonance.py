"""Resonance keys, the ingestion flag and the training-target filter."""

from dataclasses import replace

import pytest
from rdkit import Chem

from ksmech.data import chem
from ksmech.data.resonance import (
    ResonanceFlag,
    canonical_resonance_key,
    is_resonance_step,
    skeleton_key,
    training_steps,
)
from ksmech.data.schema import ElementaryStep, Source
from ksmech.data.stats import compute_stats

KEKULE_A = "[CH:1]1=[CH:2][CH:3]=[CH:4][CH:5]=[CH:6]1"
KEKULE_B = "[CH:1]1[CH:2]=[CH:3][CH:4]=[CH:5][CH:6]=1"
ENOLATE = "[CH2:1]=[C:2]([CH3:4])[O-:3]"
CARBANION = "[CH2-:1][C:2]([CH3:4])=[O:3]"


def key(smiles: str, **kw):
    return canonical_resonance_key(chem.parse_mapped(smiles), **kw)


def test_kekule_structures_share_a_key():
    a, b = key(KEKULE_A), key(KEKULE_B)
    assert a.key == b.key and a.flag == b.flag == ResonanceFlag.OK and a.n_structures >= 1


def test_enolate_and_carbanion_share_a_key():
    a, b = key(ENOLATE), key(CARBANION)
    assert a.key == b.key and a.flag == ResonanceFlag.OK
    assert skeleton_key(chem.parse_mapped(ENOLATE)) == skeleton_key(chem.parse_mapped(CARBANION))


def test_atom_maps_distinguish_degenerate_exchange():
    """A proton that moved to another atom changes the mapped skeleton: not resonance."""
    water_a, water_b = "[OH2:1].[OH3+:2]", "[OH3+:1].[OH2:2]"
    assert skeleton_key(chem.parse_mapped(water_a)) != skeleton_key(chem.parse_mapped(water_b))
    flag, _ = is_resonance_step(chem.parse_mapped(water_a), chem.parse_mapped(water_b))
    assert flag is False


def test_is_resonance_step_on_sides():
    assert is_resonance_step(chem.parse_mapped(ENOLATE), chem.parse_mapped(CARBANION))[0]
    assert not is_resonance_step(
        chem.parse_mapped(ENOLATE), chem.parse_mapped("[CH3:1][C:2]([CH3:4])=[O:3]")
    )[0]


def test_truncation_falls_back_to_the_plain_smiles():
    mol = chem.parse_mapped("[CH2-:1][CH:2]=[CH:3][CH:4]=[CH:5][CH:6]=[O:7]")
    full = canonical_resonance_key(mol)
    assert full.flag == ResonanceFlag.OK and full.n_structures > 2
    cut = canonical_resonance_key(mol, max_structures=2)
    assert cut.flag == ResonanceFlag.RESONANCE_TRUNCATED and cut.n_structures == 2
    assert cut.key == "[CH2-]C=CC=CC=O"  # the input itself, maps stripped
    assert full.key == "C=CC=CC=C[O-]"  # the smallest over the enumeration
    sides = (mol, chem.parse_mapped("[CH2:1]=[CH:2][CH:3]=[CH:4][CH:5]=[CH:6][O-:7]"))
    assert is_resonance_step(*sides)[0] is True
    assert is_resonance_step(*sides, max_structures=2) == (False, ResonanceFlag.RESONANCE_TRUNCATED)


def step(i: int, pathway: str | None, index: int | None, resonance: bool = False):
    return ElementaryStep(
        step_id=f"s{i}",
        source=Source.PMECHDB,
        pathway_id=pathway,
        step_index=index,
        reactants_smiles="C",
        products_smiles="C",
        resonance_step=resonance,
    )


def test_training_steps_drops_and_renumbers_within_pathway():
    steps = [
        step(0, "p", 0),
        step(1, "p", 1, resonance=True),
        step(2, "p", 2),
        step(3, "q", 5),
        step(4, None, None, resonance=True),
        step(5, None, None),
    ]
    kept = training_steps(steps)
    assert [s.step_id for s in kept] == ["s0", "s2", "s3", "s5"]
    assert [s.step_index for s in kept] == [0, 1, 0, None]
    assert steps[2].step_index == 2  # the input is not mutated


def test_resonance_step_survives_the_dict_round_trip():
    s = step(0, "p", 0, resonance=True)
    assert ElementaryStep.from_dict(s.to_dict()).resonance_step is True
    assert ElementaryStep.from_dict({**s.to_dict(), "resonance_step": None}).resonance_step is False
    assert replace(s, resonance_step=False).resonance_step is False


def test_ingestion_flags_the_mini_fixture_resonance_step(mini_steps, mini_result):
    assert mini_steps["resonance"].resonance_step is True
    assert all(not s.resonance_step for name, s in mini_steps.items() if name != "resonance")
    assert mini_result.report()["n_resonance_steps"] == 1
    stats = compute_stats(list(mini_steps.values()))
    assert stats["n_resonance_steps"] == 1 and stats["n_training_steps"] == 4


@pytest.mark.parametrize("smiles", ["[Na+].[Cl-]", "[H+]", "C"])
def test_single_structure_molecules(smiles):
    k = key(smiles)
    assert k.flag == ResonanceFlag.OK and k.n_structures == 1


@pytest.mark.parametrize(
    "smiles",
    ["c1ccccc1[O-]", "CO[C:20]([O-:10])=[CH:21][CH:22]=[CH2:23]", "CC(=O)C=C[O-]", "C=C[CH2+]"],
)
def test_key_does_not_depend_on_atom_order(smiles):
    """RDKit's resonance enumeration depends on atom order (phenoxide gave two keys before the
    key was taken on the canonical re-read molecule)."""
    import random

    mol = Chem.MolFromSmiles(smiles)
    keys = set()
    for seed in range(30):
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        keys.add(canonical_resonance_key(Chem.RenumberAtoms(mol, order)).key)
    assert len(keys) == 1, keys


def test_key_strips_atom_maps():
    """Deprotonating either methyl of acetone gives the same product."""
    a = canonical_resonance_key(Chem.MolFromSmiles("[CH3:1][C:2](=[O:3])[CH2-:4]"))
    b = canonical_resonance_key(Chem.MolFromSmiles("[CH2-:1][C:2](=[O:3])[CH3:4]"))
    assert a.key == b.key and ":" not in a.key


def test_a_charge_separation_rdkit_reaches_from_one_side_only_is_a_resonance_step():
    """C=N-O <-> C(+)-N(-)-O: RDKit enumerates the neutral form from the charged one, not the
    reverse, so the minima differ; the step is still a resonance step (the sets intersect)."""
    neutral = chem.parse_mapped("C[CH:10]=[N:11]O")
    charged = chem.parse_mapped("C[CH+:10][N-:11]O")
    assert is_resonance_step(neutral, charged) == (True, ResonanceFlag.OK)
    assert is_resonance_step(charged, neutral) == (True, ResonanceFlag.OK)
