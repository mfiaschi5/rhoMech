"""Resonance keys, the ingestion flag and the training-target filter."""

from dataclasses import replace

import pytest
from rdkit import Chem, rdBase
from rdkit.Chem import rdchem

from ksmech.data import chem
from ksmech.data.resonance import (
    _FLAGS,
    MAX_STRUCTURES,
    ResonanceFlag,
    canonical_resonance_key,
    frozen_hydrogens,
    is_resonance_step,
    resonance_forms,
    resonance_structures,
    skeleton_key,
    training_steps,
)
from ksmech.data.schema import ElementaryStep, Source
from ksmech.data.stats import compute_stats
from tests.fixtures.molecules import MOLECULES

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


def electron_state(mol: Chem.Mol) -> tuple:
    m = Chem.Mol(mol)
    m.UpdatePropertyCache(strict=False)
    atoms = list(m.GetAtoms())
    return (
        [a.GetTotalNumHs() for a in atoms],
        sum(a.GetFormalCharge() for a in atoms),
        sum(a.GetNumRadicalElectrons() for a in atoms),
    )


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_resonance_forms_keep_every_hydrogen_the_charge_and_the_pairing(name):
    """With the H frozen, every form keeps the H count of every atom, the total charge and no
    unpaired electron, and the molecule itself is one of its forms."""
    mol = frozen_hydrogens(Chem.MolFromSmiles(MOLECULES[name]))
    forms, n = resonance_forms(mol, MAX_STRUCTURES)
    assert 1 <= len(forms) <= n < MAX_STRUCTURES
    assert all(electron_state(f) == electron_state(mol) for f in forms)
    assert Chem.MolToSmiles(mol) in {Chem.MolToSmiles(f) for f in forms}


def test_nitro_forms_never_move_a_hydrogen_or_unpair_electrons():
    """RDKit's own enumeration of nitroethane re-derives the implicit H of an unbracketed O
    ([OH2+]) and returns the radical CCN([O+])[O-]. Neither survives: the two equivalent N-O
    forms are left, and they give one key."""
    mol = Chem.MolFromSmiles("CC[N+](=O)[O-]")
    with rdBase.BlockLogs():
        raw = rdchem.ResonanceMolSupplier(mol, int(_FLAGS), 64)
        raw_smiles = {Chem.MolToSmiles(raw[k]) for k in range(len(raw))}
    assert {"CCN([O-])[OH2+]", "CCN([O+])[O-]"} <= raw_smiles  # what the filter is for
    forms, _ = resonance_forms(frozen_hydrogens(mol), 64)
    assert len(forms) == 2
    assert all(electron_state(f) == electron_state(mol) for f in forms)
    keys, _, flag = resonance_structures(mol)
    assert flag == ResonanceFlag.OK and len(keys) == 1


def test_enolate_forms_are_the_alkoxide_and_the_carbanion():
    """From either drawing: two forms, one with the charge on O and one on the CH2."""
    for smiles in (ENOLATE, CARBANION):
        forms, _ = resonance_forms(frozen_hydrogens(chem.parse_mapped(smiles)), 64)
        charged = sorted(a.GetSymbol() for f in forms for a in f.GetAtoms() if a.GetFormalCharge())
        assert charged == ["C", "O"], smiles
        assert all(electron_state(f) == electron_state(chem.parse_mapped(smiles)) for f in forms)


def test_frozen_hydrogens_keep_a_kekule_structure_unless_aromaticity_is_asked():
    """The edit labels read bond orders, so their enumeration starts from the Kekule structure
    as drawn; the resonance key works on the aromatic canonical form."""
    kekule = Chem.MolFromSmiles("C1=CC=CC=C1")
    Chem.Kekulize(kekule, clearAromaticFlags=True)
    assert not any(b.GetIsAromatic() for b in frozen_hydrogens(kekule).GetBonds())
    assert all(b.GetIsAromatic() for b in frozen_hydrogens(kekule, aromaticity=True).GetBonds())
