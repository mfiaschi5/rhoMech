"""Token building, the shared id sequence, atom features, and the .npz round trip."""

from dataclasses import fields

import numpy as np
import pytest
from rich.console import Console

from ksmech.orbitals.classify import OrbitalType
from ksmech.orbitals.descriptors import MODEL_INPUT
from ksmech.orbitals.fgroups import NONE_ID, default_vocab
from ksmech.orbitals.tokens import (
    ATOM_CATEGORICAL,
    VALENCE_D,
    AtomToken,
    CandidateConfig,
    OrbitalKind,
    atom_feature_names,
    build_molecule_tokens,
    is_antibonding_partner,
    is_d_type,
    load_tokens,
    save_tokens,
    token_table,
)
from ksmech.qm.fukui import HARTREE_TO_EV
from tests.token_select import leading, one_centre, two_centre


@pytest.fixture(scope="module")
def acetone(sp_record):
    return build_molecule_tokens(sp_record("CC(=O)C"))


def test_ids_form_one_sequence_atoms_first(acetone):
    n = len(acetone.atoms)
    assert [a.token_id for a in acetone.atoms] == list(range(n))
    assert [o.token_id for o in acetone.orbitals] == list(range(n, n + len(acetone.orbitals)))
    kinds = [o.kind for o in acetone.orbitals]
    assert kinds == sorted(kinds, key=lambda k: k != OrbitalKind.OCCUPIED)  # occupied first
    for block in (acetone.occupied(), acetone.virtual()):
        energies = [o.energy_ev for o in block]
        assert energies == sorted(energies)
    assert len(acetone.occupied()) == 12 and len(acetone.virtual()) == 10


@pytest.mark.diagnostic
def test_atom_token_fields(acetone):
    vocab = default_vocab()
    carbonyl_c, oxygen = 1, 3
    assert acetone.atoms[oxygen].atomic_number == 8
    assert acetone.atoms[carbonyl_c].hybridization == "SP2"
    assert acetone.atoms[0].hybridization == "SP3" and not acetone.atoms[0].aromatic
    assert acetone.atoms[oxygen].fg_id == vocab.id_of("C=O") == acetone.atoms[carbonyl_c].fg_id
    assert acetone.atoms[0].fg_id == NONE_ID and all(a.fg_id == NONE_ID for a in acetone.atoms[4:])
    assert all(a.map_number == 0 and a.formal_charge == 0 for a in acetone.atoms)
    assert acetone.atoms[oxygen].mulliken_charge < 0
    assert np.isfinite(acetone.atoms[oxygen].f_minus_fmo) and np.isnan(acetone.atoms[oxygen].f_plus)
    acetone.assign_map_numbers({oxygen: 20})
    assert acetone.atoms[oxygen].map_number == 20


def test_atom_model_input_has_no_taxonomy(sp_record):
    """Hybridization, aromaticity and functional-group ids are stored but never model input."""
    t = build_molecule_tokens(sp_record("CC(=O)C"))
    X, Z = t.atom_features(), t.atom_categorical()
    assert X.shape == (len(t.atoms), len(atom_feature_names())) and np.isfinite(X).all()
    assert Z.shape == (len(t.atoms), 2) and list(Z[:, 0]) == [a.atomic_number for a in t.atoms]
    for a in t.atoms:
        a.hybridization, a.aromatic, a.fg_id = "SP", True, 1
        a.f_plus, a.f_minus, a.position = 9.0, 9.0, a.position + 5.0
    assert np.array_equal(t.atom_features(), X) and np.array_equal(t.atom_categorical(), Z)
    tags = {f.name: f.metadata[MODEL_INPUT] for f in fields(AtomToken)}
    assert {n for n, is_input in tags.items() if not is_input} == {
        "token_id",
        "atom_index",
        "map_number",
        "f_plus",
        "f_minus",
        "hybridization",
        "aromatic",
        "fg_id",
        "position",
    }
    assert set(ATOM_CATEGORICAL) == {"atomic_number", "formal_charge"}
    assert atom_feature_names()[:4] == ["mulliken_charge", "f_plus_fmo", "f_minus_fmo", "dual"]


def test_missing_atom_values_are_zero_and_masked(sp_record):
    t = build_molecule_tokens(sp_record("[H+]"))  # no occupied levels: HOMO-1, HOMO are NaN
    names = atom_feature_names()
    missing = t.atom_missing()[0]
    assert missing[names.index("fmo_homo")] and not missing[names.index("fmo_lumo")]
    assert t.atom_features()[0, names.index("fmo_homo")] == 0.0


def test_orbital_energy_is_the_rotated_fock_diagonal(acetone):
    for o in acetone.orbitals:
        e = o.coeff_vector @ acetone.F @ o.coeff_vector * HARTREE_TO_EV
        assert abs(e - o.energy_ev) < 1e-8
    assert acetone.C_loc.shape == (acetone.S.shape[0], len(acetone.orbitals))


def test_occupied_energies_sum_to_canonical_trace(sp_record, acetone):
    rec = sp_record("CC(=O)C")
    canonical = rec.orbital_energies_ev[rec.occupations >= 0.5].sum()
    assert abs(sum(o.energy_ev for o in acetone.occupied()) - canonical) < 1e-6


def test_bond_centroids_lie_on_the_bond(acetone):
    pos = np.array([a.position for a in acetone.atoms])
    n_two = n_one = 0
    for o in acetone.orbitals:
        if two_centre(o):
            a, b = leading(o)
            mid, half = 0.5 * (pos[a] + pos[b]), 0.5 * np.linalg.norm(pos[a] - pos[b])
            assert np.linalg.norm(o.centroid - mid) < half
            n_two += 1
        if one_centre(o):
            assert np.linalg.norm(o.centroid - pos[leading(o, 1)[0]]) < 0.2
            n_one += 1
    assert n_one == 2 and n_two == len(acetone.orbitals) - 2  # two O lone pairs, rest bonds


def test_round_trip(acetone, tmp_path):
    path = save_tokens(acetone, tmp_path / "sub" / "acetone.npz")
    back = load_tokens(path)
    assert back.smiles == acetone.smiles and back.method == acetone.method
    assert back.pm_sweeps_occ == acetone.pm_sweeps_occ and back.pm_converged is True
    assert back.fg_vocab_size == len(default_vocab())
    for a, b in zip(back.atoms, acetone.atoms, strict=True):
        np.testing.assert_equal(vars(a), vars(b))  # NaN-aware, positions included
    for o, p in zip(back.orbitals, acetone.orbitals, strict=True):
        assert (o.token_id, o.kind, o.type, o.parent_atoms) == (
            p.token_id,
            p.kind,
            p.type,
            p.parent_atoms,
        )
        assert o.parent_weights == p.parent_weights and o.energy_ev == p.energy_ev
        np.testing.assert_equal(
            (o.s_share, o.p_share, o.d_share, o.overlap_population),
            (p.s_share, p.p_share, p.d_share, p.overlap_population),
        )
        assert np.array_equal(o.coeff_vector, p.coeff_vector)
        for f in fields(type(p.descriptors)):  # every descriptor, ragged centre sets included
            np.testing.assert_equal(getattr(o.descriptors, f.name), getattr(p.descriptors, f.name))
    assert np.array_equal(back.S, acetone.S) and np.array_equal(back.F, acetone.F)
    assert np.array_equal(back.C_loc, acetone.C_loc)


@pytest.mark.diagnostic
def test_round_trip_zero_electron_species(sp_record, tmp_path):
    tokens = build_molecule_tokens(sp_record("[H+]"))
    assert tokens.occupied() == [] and len(tokens.virtual()) == 1
    back = load_tokens(save_tokens(tokens, tmp_path / "h.npz"))
    assert back.virtual()[0].type == OrbitalType.EMPTY_S and back.C_loc.shape == (1, 1)


def test_mol_order_mismatch_is_rejected(sp_record):
    from rdkit import Chem

    rec = sp_record("CCO")
    wrong = Chem.AddHs(Chem.MolFromSmiles("OCC"))
    with pytest.raises(ValueError, match="atom order"):
        build_molecule_tokens(rec, mol=wrong)


@pytest.mark.diagnostic
def test_token_table_renders(acetone):
    console = Console(record=True, width=120)
    console.print(token_table(acetone, title="acetone"))
    text = console.export_text()
    assert "lone_pair" in text and "pi_star" in text and "O3:" in text


def test_acceptor_candidates_exclude_d_type_by_default(sp_record):
    chloride = build_molecule_tokens(sp_record("[Cl-]"))
    assert len(chloride.virtual()) == 5 and all(chloride.is_d_type(o) for o in chloride.virtual())
    assert chloride.acceptor_candidates() == []
    assert len(chloride.acceptor_candidates(CandidateConfig(allow_d_acceptors=True))) == 5
    assert len(chloride.donor_candidates()) == 4
    assert chloride.atoms_with_only_d_virtuals() == [0]
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    acceptors = chloromethane.acceptor_candidates()
    assert len(acceptors) == 4 and chloromethane.atoms_with_only_d_virtuals() == []
    assert all(len(leading(o)) == 2 for o in acceptors)  # C-Cl and 3 C-H sigma*
    c_cl = next(o for o in chloromethane.occupied() if set(leading(o)) == {0, 1})
    without_partner = chloromethane.acceptor_candidates(donor=c_cl)
    assert len(without_partner) == 3 and all(set(leading(o)) != {0, 1} for o in without_partner)
    assert all(is_antibonding_partner(c_cl, o) for o in acceptors if o not in without_partner)


@pytest.mark.diagnostic
@pytest.mark.parametrize("smiles", ["[Cl-]", "CCl", "CBr", "CI", "CS(C)=O", "ClP(Cl)Cl", "CC(=O)C"])
def test_d_type_matches_the_classifier_empty_d(sp_record, smiles):
    """Diagnostic: the descriptor rule picks exactly the tokens the classifier calls empty_d."""
    t = build_molecule_tokens(sp_record(smiles))
    by_rule = [t.is_d_type(o) for o in t.virtual()]
    by_label = [o.type == OrbitalType.EMPTY_D for o in t.virtual()]
    assert by_rule == by_label


def test_transition_metal_d_shells_are_valence_not_polarization(sp_record):
    """Ti(IV) accepts into its d shell; only main-group (and d10-core) d functions are masked."""
    ticl4 = build_molecule_tokens(sp_record("Cl[Ti](Cl)(Cl)Cl"))
    titanium = next(a.atom_index for a in ticl4.atoms if a.atomic_number == 22)
    on_ti = [o for o in ticl4.virtual() if int(o.descriptors.center_atoms[0]) == titanium]
    mostly_d = [o for o in on_ti if int(np.argmax(o.descriptors.center_spd[0])) == 2]
    assert mostly_d and not any(ticl4.is_d_type(o) for o in mostly_d)
    assert all(o in ticl4.acceptor_candidates() for o in mostly_d)
    assert is_d_type(mostly_d[0].descriptors, 17)  # the same shape on Cl is a polarization function
    assert not any(z in VALENCE_D for z in (12, 13, 14, 15, 16, 17, 30, 35, 48, 53, 80))


def test_metal_bound_lone_pair_round_trip(sp_record, tmp_path):
    tokens = build_molecule_tokens(sp_record("C[Mg]Br"))
    bound = [o for o in tokens.occupied() if o.metal_bound]
    assert len(bound) == 2 and all(tokens.symbol(o.metal_atom) == "Mg" for o in bound)
    carbanion = next(o for o in bound if tokens.symbol(o.parent_atoms[0]) == "C")
    assert carbanion.type == OrbitalType.LONE_PAIR and carbanion in tokens.donor_candidates()
    back = load_tokens(save_tokens(tokens, tmp_path / "grignard.npz"))
    for o, p in zip(back.orbitals, tokens.orbitals, strict=True):
        assert (o.metal_bound, o.metal_atom) == (p.metal_bound, p.metal_atom)
    text = Console(record=True, width=120)
    text.print(token_table(back))
    assert "Mg1-bound]" in text.export_text() or "-bound]" in text.export_text()
