"""Bond-electron-matrix edits of dataset arrows (`ksmech.arrows.apply`)."""

import numpy as np
import pytest
from rdkit import Chem

from ksmech.arrows.apply import (
    ApplyReason,
    ArrowApplyError,
    Displacement,
    Move,
    apply_arrows,
    apply_edit,
    edit_from_moves,
    sanitize_product,
)
from ksmech.arrows.roundtrip import round_trip_step, run_round_trip
from ksmech.data import chem
from ksmech.data.io import read_training_steps, steps_path
from ksmech.data.schema import Arrow, SinkKind, SourceKind

LP, BOND = SourceKind.LONE_PAIR, SourceKind.BOND
ATOM, SINK_BOND = SinkKind.ATOM, SinkKind.BOND


def _apply(smiles: str, arrows: list) -> tuple:
    result = apply_arrows(chem.parse_mapped(smiles), arrows)
    return result, (result.smiles if result.ok else None)


def _same(got: str, want: str) -> bool:
    return got == chem.canonicalize_mapped(want)


def test_lone_pair_to_atom_forms_a_bond_and_shifts_charges():
    result, smi = _apply("[OH-:1].[CH3+:2]", [Arrow(LP, (1,), ATOM, (2,))])
    assert result.ok and _same(smi, "[OH:1][CH3:2]")
    o, c = (result.mol.GetAtomWithIdx(chem.map_to_index(result.mol)[m]) for m in (1, 2))
    assert (o.GetFormalCharge(), c.GetFormalCharge()) == (0, 0)  # O -1 +1, C +1 -1


def test_lone_pair_to_atom_raises_an_existing_bond_order():
    result, smi = _apply("[CH2+:2][O-:1]", [Arrow(LP, (1,), ATOM, (2,))])
    assert result.ok and _same(smi, "[CH2:2]=[O:1]")


def test_lone_pair_to_own_bond_is_the_same_as_to_the_atom():
    a, _ = _apply("[OH-:1].[CH3+:2]", [Arrow(LP, (1,), ATOM, (2,))])
    b, _ = _apply("[OH-:1].[CH3+:2]", [Arrow(LP, (1,), SINK_BOND, (1, 2))])  # mech-USPTO form
    assert a.smiles == b.smiles


def test_heterolysis_moves_the_pair_onto_the_sink_atom():
    result, smi = _apply("[CH3:1][Cl:2]", [Arrow(BOND, (1, 2), ATOM, (2,))])
    assert result.ok and _same(smi, "[CH3+:1].[Cl-:2]")


def test_bond_to_bond_forms_the_new_bond_at_the_shared_atom():
    # C1-H3 pair becomes the C1=C2 pi bond; H3 leaves as a proton (c +1), C2+ is neutralized
    result, smi = _apply("[H:3][CH2:1][CH2+:2]", [Arrow(BOND, (1, 3), SINK_BOND, (1, 2))])
    assert result.ok and _same(smi, "[H+:3].[CH2:1]=[CH2:2]")


def test_concerted_displacement_is_one_arrow():
    result, smi = _apply("[OH-:1].[CH3:2][Cl:3]", [Displacement(donor=1, attacked=2, leaving=3)])
    assert result.ok and _same(smi, "[OH:1][CH3:2].[Cl-:3]")
    two_arrows, _ = _apply(
        "[OH-:1].[CH3:2][Cl:3]", [Arrow(LP, (1,), ATOM, (2,)), Arrow(BOND, (2, 3), ATOM, (3,))]
    )
    assert two_arrows.smiles == result.smiles


def test_step_is_one_atomic_edit_in_any_arrow_order(mini_steps):
    step = mini_steps["e2"]
    reactants = chem.parse_mapped(step.reactants_smiles)
    forward = apply_arrows(reactants, step.arrows)
    backward = apply_arrows(reactants, step.arrows[::-1])
    assert forward.smiles == backward.smiles == step.products_smiles


def test_radical_arrows_are_not_implemented():
    with pytest.raises(NotImplementedError):
        apply_arrows(chem.parse_mapped("[CH3:1].[Cl:2]"), [Arrow("radical", (1,), ATOM, (2,))])


def test_unknown_arrow_codes():
    mol = chem.parse_mapped("[OH-:1].[CH3+:2]")
    assert apply_arrows(mol, ["10=20"]).reason == ApplyReason.UNKNOWN_ARROW
    missing = apply_arrows(mol, [Arrow(LP, (1,), ATOM, (9,))])
    assert missing.reason == ApplyReason.UNKNOWN_ARROW
    assert apply_arrows(mol, [Displacement(1, 2, 2)]).reason == ApplyReason.UNKNOWN_ARROW


def test_valence_codes():
    no_pair = apply_arrows(chem.parse_mapped("[CH4:1].[H+:2]"), [Arrow(LP, (1,), ATOM, (2,))])
    assert no_pair.reason == ApplyReason.VALENCE and "lone pair" in no_pair.detail
    five_bonds = apply_arrows(chem.parse_mapped("[OH-:1].[CH4:2]"), [Arrow(LP, (1,), ATOM, (2,))])
    assert five_bonds.reason == ApplyReason.VALENCE
    no_bond = apply_arrows(chem.parse_mapped("[CH4:1].[CH4:2]"), [Arrow(BOND, (1, 2), ATOM, (2,))])
    assert no_bond.reason == ApplyReason.VALENCE


def test_sanitize_code_for_a_non_kekulizable_ring():
    with pytest.raises(ArrowApplyError) as exc:
        sanitize_product(Chem.RWMol(Chem.MolFromSmiles("c1cccc1", sanitize=False)))
    assert exc.value.reason == ApplyReason.SANITIZE


def test_hydrogen_counts_are_frozen_on_unbracketed_atoms():
    mol = Chem.MolFromSmiles("CC")  # implicit hydrogens, no brackets
    edited = apply_edit(mol, edit_from_moves([Move((1, 2), (2,))], {1: 0, 2: 1}))
    out = sanitize_product(edited)  # heterolysis: CH3+ and CH3-, still three H each
    assert [a.GetTotalNumHs() for a in out.GetAtoms()] == [3, 3]
    assert sorted(a.GetFormalCharge() for a in out.GetAtoms()) == [-1, 1]


def test_mini_fixture_round_trips(mini_steps):
    for label, step in mini_steps.items():
        assert round_trip_step(step).status == "exact", label


def _mapped_with_h(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    Chem.Kekulize(mol, clearAromaticFlags=True)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)
    return chem.parse_mapped(Chem.MolToSmiles(mol))


def _random_arrow(mol: Chem.Mol, rng: np.random.Generator):
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    bonds = [
        (b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()) for b in mol.GetBonds()
    ]
    kind = rng.integers(4)
    a, b = (int(x) for x in rng.choice(maps, 2, replace=False))
    if kind == 0 or not bonds:
        return Arrow(LP, (a,), ATOM, (b,))
    p, q = (int(x) for x in bonds[rng.integers(len(bonds))])
    if kind == 1:
        return Arrow(BOND, (p, q), ATOM, (q,))
    other = int(rng.choice([m for m in maps if m not in (p, q)] or [a]))
    if kind == 2 and other not in (p, q):
        return Arrow(BOND, (p, q), SINK_BOND, (p, other))
    return Displacement(donor=other, attacked=p, leaving=q) if other not in (p, q) else None


def test_property_heavy_atoms_charge_and_hydrogens_are_conserved():
    rng = np.random.default_rng(0)
    pool = ["CC(=O)[O-]", "C[NH3+]", "CC=O", "ClCC[O-]", "C=CC=O", "[OH-].CBr", "C#N", "OCC"]
    n_ok = 0
    for _ in range(600):
        mol = _mapped_with_h(pool[rng.integers(len(pool))])
        arrows = [x for x in (_random_arrow(mol, rng) for _ in range(rng.integers(1, 4))) if x]
        result = apply_arrows(mol, arrows)
        if not result.ok:
            assert result.reason in set(ApplyReason)
            continue
        n_ok += 1
        assert chem.heavy_atom_count(result.mol) == chem.heavy_atom_count(mol)
        assert chem.formal_charge(result.mol) == chem.formal_charge(mol)
        assert chem.hydrogen_count(result.mol) == chem.hydrogen_count(mol)
    assert n_ok >= 50  # the property was exercised, not only the failure codes


@pytest.mark.realdata
def test_pmechdb_round_trip_rate():
    path = steps_path("pmechdb")
    if not path.exists():
        pytest.skip("data/processed/pmechdb/steps.parquet missing")
    _, summary = run_round_trip(read_training_steps(path))
    print(summary)
    assert summary["match_rate"] >= 0.95
