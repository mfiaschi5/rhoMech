"""Token building, the shared id sequence, atom features, and the .npz round trip."""

from dataclasses import fields, replace

import numpy as np
import pytest
from rich.console import Console

from ksmech.orbitals.classify import OrbitalType
from ksmech.orbitals.descriptors import MODEL_INPUT
from ksmech.orbitals.fgroups import NONE_ID, default_vocab
from ksmech.orbitals.localize import aufbau_occupations, is_fermi_smeared
from ksmech.orbitals.tokens import (
    ATOM_CATEGORICAL,
    VALENCE_D,
    AtomToken,
    CandidateConfig,
    OrbitalKind,
    antibonding_partner_matrix,
    atom_feature_names,
    bonded_matrix,
    build_molecule_tokens,
    excluded_pair,
    is_antibonding_partner,
    is_d_type,
    load_tokens,
    save_tokens,
    site_excluded_matrix,
    site_pairs_excluded,
    token_table,
)
from ksmech.qm.fukui import HARTREE_TO_EV, homo_index
from tests.fixtures.molecules import MOLECULES
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
    kept = chloromethane.acceptor_candidates(donor=c_cl)  # only its own antibond is excluded
    assert len(kept) == 3 and all(set(leading(o)) != {0, 1} for o in kept)
    assert all(is_antibonding_partner(c_cl, o) for o in acceptors if o not in kept)
    lone_cl = next(o for o in chloromethane.occupied() if leading(o) == (1,) and one_centre(o))
    kept = chloromethane.acceptor_candidates(donor=lone_cl)  # not into its own C-Cl sigma*
    assert len(kept) == 3 and all(set(leading(o)) != {0, 1} for o in kept)


def smeared(rec, shift: float):
    """A copy of `rec` with `shift` electrons moved from the HOMO to the LUMO."""
    occ = rec.occupations.copy()
    h = homo_index(occ)
    occ[h] -= shift
    occ[h + 1] += shift
    return replace(rec, occupations=occ)


def test_fermi_smeared_flag(sp_record):
    assert not any(is_fermi_smeared(sp_record(s)) for s in MOLECULES.values())
    methyl = sp_record("[CH3]")
    assert list(aufbau_occupations(methyl)) == [2, 2, 2, 1, 0, 0, 0]
    assert not is_fermi_smeared(methyl)  # the SOMO at 1 is the spin state, not smearing
    assert is_fermi_smeared(smeared(methyl, 0.1))
    ethylene = sp_record("C=C")
    assert is_fermi_smeared(smeared(ethylene, 0.3)) and not is_fermi_smeared(
        smeared(ethylene, 0.04)
    )


def test_fractional_band_bounds():
    config = CandidateConfig()
    assert [config.is_fractional(n) for n in (0.2, 0.21, 1.0, 1.79, 1.8)] == [
        False,
        True,
        True,
        True,
        False,
    ]
    assert not CandidateConfig(exclude_fractional=False).is_fractional(1.0)


def test_fractional_tokens_keep_their_occupation_and_are_masked(sp_record, tmp_path):
    t = build_molecule_tokens(smeared(sp_record("C=C"), 0.3))
    assert t.fermi_smeared and not build_molecule_tokens(sp_record("C=C")).fermi_smeared
    pi = [o for o in t.occupied() if abs(o.descriptors.occupation - 2.0) > 0.05]
    pi_star = [o for o in t.virtual() if o.descriptors.occupation > 0.05]
    assert len(pi) == len(pi_star) == 1
    assert 1.6 < pi[0].descriptors.occupation < 1.8  # the continuous input stays fractional
    assert 0.2 < pi_star[0].descriptors.occupation < 0.4

    def ids(tokens):
        return {o.token_id for o in tokens}

    assert pi[0].token_id not in ids(t.donor_candidates())
    assert pi_star[0].token_id not in ids(t.acceptor_candidates())
    loose = CandidateConfig(exclude_fractional=False)
    assert pi[0].token_id in ids(t.donor_candidates(loose))
    assert pi_star[0].token_id in ids(t.acceptor_candidates(loose))
    assert load_tokens(save_tokens(t, tmp_path / "smeared.npz")).fermi_smeared


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


def _lone_pair(t, z: int):
    atom = next(a.atom_index for a in t.atoms if a.atomic_number == z)
    lone = [o for o in t.occupied() if leading(o) == (atom,) and one_centre(o)]
    return max(lone, key=lambda o: o.energy_ev), atom


def test_excluded_pairs(sp_record):
    """Within one molecule a pair is excluded when it is the donor's own antibond, or when
    every site pair (donor centre a, acceptor centre b) has a and b identical or bonded with
    the acceptor carrying >= 0.20 on a or the donor >= 0.20 on b: a bond interacting with
    itself."""
    acetone = build_molecule_tokens(sp_record("CC(C)=O"))
    n_o, oxygen = _lone_pair(acetone, 8)
    pi_star = min(acetone.virtual(), key=lambda o: o.energy_ev)
    pi = next(
        o
        for o in acetone.occupied()
        if set(leading(o)) == set(leading(pi_star))
        and o.descriptors.overlap_population > 0
        and o.descriptors.cos_character() < 0.3
    )
    assert oxygen in leading(pi_star) and leading(pi_star)[0] != oxygen
    assert excluded_pair(n_o, pi_star, acetone)  # n(O) into its own C=O pi*
    assert pi_star not in acetone.acceptor_candidates(donor=n_o)
    assert excluded_pair(pi, pi_star, acetone) and is_antibonding_partner(pi, pi_star)
    methanol = build_molecule_tokens(sp_record("CO"))
    n_o, oxygen = _lone_pair(methanol, 8)
    o_h = next(
        o for o in methanol.virtual() if {methanol.symbol(a) for a in leading(o)} == {"O", "H"}
    )
    assert excluded_pair(n_o, o_h, methanol)  # n(O) -> sigma*(O-H) of the same O-H
    c_h = [o for o in methanol.virtual() if {methanol.symbol(a) for a in leading(o)} == {"C", "H"}]
    assert c_h and not any(excluded_pair(n_o, o, methanol) for o in c_h)  # vicinal: kept
    hbr = build_molecule_tokens(sp_record("Br"))
    n_br, _ = _lone_pair(hbr, 35)
    h_br = next(o for o in hbr.virtual() if len(leading(o)) == 2)
    assert excluded_pair(n_br, h_br, hbr)  # HBr's own n(Br) -> sigma*(H-Br)
    cation = build_molecule_tokens(sp_record("C[CH+]C"))  # 1,2-hydride shift: allowed
    empty_p = min(cation.virtual(), key=lambda o: o.energy_ev)
    c2 = leading(empty_p)[0]
    sigma_ch = next(
        o
        for o in cation.occupied()
        if two_centre(o)
        and {cation.symbol(a) for a in leading(o)} == {"C", "H"}
        and c2 not in leading(o)
    )
    assert not excluded_pair(sigma_ch, empty_p, cation)
    assert empty_p in cation.acceptor_candidates(donor=sigma_ch)


def test_site_rule_threshold(sp_record):
    """The weight test is a threshold on the population at the near site (a tetrahedral
    intermediate's sigma* carries 0.20 on its O(-): excluded at 0.20 as a site pair, but its
    O -> Cl site pair is never near, so the collapse stays a candidate); sites
    neither identical nor bonded never exclude; a token pair is excluded when every one of its
    centre pairs is."""
    methanol = build_molecule_tokens(sp_record("CO"))
    n_o, oxygen = _lone_pair(methanol, 8)
    o_h = next(
        o for o in methanol.virtual() if {methanol.symbol(a) for a in leading(o)} == {"O", "H"}
    )
    hydrogen = next(a for a in leading(o_h) if a != oxygen)
    d = o_h.descriptors
    w_o = dict(zip(d.all_center_atoms, d.all_center_weights, strict=True))[oxygen]

    def excluded(d, a, s, b, w=CandidateConfig.site_weight) -> bool:
        return bool(site_pairs_excluded(methanol, [d], [a], [s], [b], w)[0])

    assert excluded(n_o, oxygen, o_h, hydrogen, w_o - 0.01)
    assert not excluded(n_o, oxygen, o_h, hydrogen, w_o + 0.01)
    carbon = next(a.atom_index for a in methanol.atoms if a.atomic_number == 6)
    near_c = bonded_matrix(methanol)[carbon]
    h_on_c = next(i for i, a in enumerate(methanol.atoms) if a.atomic_number == 1 and near_c[i])
    assert not excluded(n_o, oxygen, o_h, h_on_c, 0.0)  # O and a methyl H: not bonded
    c_o = next(
        o for o in methanol.occupied() if {methanol.symbol(a) for a in leading(o)} == {"C", "O"}
    )
    w_d = dict(
        zip(c_o.descriptors.all_center_atoms, c_o.descriptors.all_center_weights, strict=True)
    )
    # donor sigma(C-O) at C, acceptor sigma*(O-H) at O: bonded sites, no acceptor weight on C,
    # so only the donor's share on O decides
    assert excluded(c_o, carbon, o_h, oxygen, w_d[oxygen] - 0.01)
    assert not excluded(c_o, carbon, o_h, oxygen, w_d[oxygen] + 0.01)
    occ, virt = methanol.occupied(), methanol.virtual()
    matrix = site_excluded_matrix(methanol, occ, virt, CandidateConfig().site_weight)
    for i, d in enumerate(occ):
        for j, s in enumerate(virt):
            every = all(
                excluded(d, int(a), s, int(b))
                for a in d.descriptors.all_center_atoms
                for b in s.descriptors.all_center_atoms
            )
            assert matrix[i, j] == every


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_candidate_matrices_give_their_pairwise_rules(sp_record, name):
    """The closed-form site matrix and the vectorized antibonding-partner matrix agree with
    the rules they summarize (`site_pairs_excluded` over every centre pair,
    `is_antibonding_partner`), at the default site weight and at weights that cut through the
    centre populations."""
    t = build_molecule_tokens(sp_record(MOLECULES[name]))
    occ, virt = t.occupied(), t.virtual()
    own = antibonding_partner_matrix(occ, virt)
    assert own.tolist() == [[is_antibonding_partner(d, s) for s in virt] for d in occ]
    bonded = bonded_matrix(t)
    for w in (CandidateConfig().site_weight, 0.35, 0.55):
        matrix = site_excluded_matrix(t, occ, virt, w)
        every = [
            [
                all(
                    site_pairs_excluded(t, [d], [int(a)], [s], [int(b)], w, bonded)[0]
                    for a in d.descriptors.all_center_atoms
                    for b in s.descriptors.all_center_atoms
                )
                for s in virt
            ]
            for d in occ
        ]
        assert matrix.tolist() == every, (name, w)
    default = site_excluded_matrix(t, occ, virt, CandidateConfig().site_weight)
    assert default.any() and not default.all(), name  # a bond with itself, and real arrows
