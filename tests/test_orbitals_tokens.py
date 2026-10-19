"""Token building, the shared id sequence, atom features, and the .npz round trip."""

import numpy as np
import pytest
from rich.console import Console

from ksmech.orbitals.classify import OrbitalType
from ksmech.orbitals.fgroups import NONE_ID, default_vocab
from ksmech.orbitals.tokens import (
    CandidateConfig,
    OrbitalKind,
    build_molecule_tokens,
    load_tokens,
    save_tokens,
    token_table,
)
from ksmech.qm.fukui import HARTREE_TO_EV


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
    for o in acetone.orbitals:
        if o.type in (OrbitalType.SIGMA, OrbitalType.SIGMA_STAR, OrbitalType.PI):
            a, b = o.parent_atoms
            mid, half = 0.5 * (pos[a] + pos[b]), 0.5 * np.linalg.norm(pos[a] - pos[b])
            assert np.linalg.norm(o.centroid - mid) < half
        if o.type == OrbitalType.LONE_PAIR:
            assert np.linalg.norm(o.centroid - pos[o.parent_atoms[0]]) < 0.2


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
        assert np.array_equal(o.centroid, p.centroid)
    assert np.array_equal(back.S, acetone.S) and np.array_equal(back.F, acetone.F)
    assert np.array_equal(back.C_loc, acetone.C_loc)


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


def test_token_table_renders(acetone):
    console = Console(record=True, width=120)
    console.print(token_table(acetone, title="acetone"))
    text = console.export_text()
    assert "lone_pair" in text and "pi_star" in text and "O3:" in text


def test_acceptor_candidates_exclude_empty_d_by_default(sp_record):
    chloride = build_molecule_tokens(sp_record("[Cl-]"))
    assert [o.type for o in chloride.virtual()] == [OrbitalType.EMPTY_D] * 5
    assert chloride.acceptor_candidates() == []
    assert len(chloride.acceptor_candidates(CandidateConfig(allow_empty_d_acceptors=True))) == 5
    assert len(chloride.donor_candidates()) == 4
    assert chloride.atoms_with_only_empty_d_virtuals() == [0]
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    accepted = {o.type for o in chloromethane.acceptor_candidates()}
    assert (
        accepted == {OrbitalType.SIGMA_STAR}
        and chloromethane.atoms_with_only_empty_d_virtuals() == []
    )
    assert len(chloromethane.acceptor_candidates()) == 4  # C-Cl and 3 C-H sigma*


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
