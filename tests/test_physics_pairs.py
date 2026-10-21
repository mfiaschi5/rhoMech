"""The pair-track features z_rs of every (occupied, virtual) pair, and their statistics."""

import copy

import numpy as np
import pytest

from ksmech.orbitals.tokens import build_molecule_tokens
from ksmech.physics.contact_table import vdw_sum
from ksmech.physics.coulomb import coulomb
from ksmech.physics.ks_terms import gap_kernel_matrix
from ksmech.physics.pairs import (
    FEATURE_DISTANCES,
    PAIR_FEATURES,
    PRIMARY_DISTANCE,
    F,
    PairStats,
    fit_pair_stats,
    frozen_head,
    ks_energies,
    oriented_kernels,
    pair_block,
    standardize,
)
from ksmech.physics.params import COULOMB_EV_ANG, REDUCED, KSParams, to_params
from ksmech.physics.repulsion import (
    NEIGHBOURHOOD_ANG,
    atom_populations,
    repulsion_matrix,
)
from ksmech.qm.sterics import buried_volume_per_atom
from tests.token_select import highest_lone_pair, leading, sigma_star, sigma_stars


def with_sterics(rec):
    """A copy of a single-point record with %Vbur filled in (the sp fixture skips sterics)."""
    out = copy.copy(rec)
    out.buried_volume = buried_volume_per_atom(rec.atomic_numbers, rec.coords_ang)
    return out


@pytest.fixture(scope="module")
def pair(sp_record):
    methanol = build_molecule_tokens(with_sterics(sp_record("CO")))
    chloromethane = build_molecule_tokens(with_sterics(sp_record("CCl")))
    kernels = oriented_kernels()
    return methanol, chloromethane, kernels, pair_block([methanol, chloromethane], kernels)


def test_coulomb_term():
    assert coulomb(0.5, -0.5, 2.0, KSParams(eps=4.0)) == COULOMB_EV_ANG * -0.25 / 8.0
    assert coulomb(1.0, 1.0, 0.0) == 0.0


def test_repulsion_uses_electron_populations(sp_record):
    t = build_molecule_tokens(sp_record("CO"))
    pops = atom_populations(t)
    oxygen = next(a.atom_index for a in t.atoms if a.atomic_number == 8)
    assert abs(pops[oxygen] - (6 - t.atoms[oxygen].mulliken_charge)) < 1e-12
    M = repulsion_matrix(t)
    assert M.shape == (len(t.occupied()),) * 2 and np.allclose(M, M.T) and not np.diag(M).any()
    assert np.allclose(repulsion_matrix(t, KSParams(K=2.0)), 2.0 / 1.75 * M)
    assert (M >= 0).all() and M.any()


def test_repulsion_ignores_the_lmo_phase(sp_record):
    """Pipek-Mezey fixes no phase: flipping one occupied LMO must not change KS1."""
    t = build_molecule_tokens(sp_record("CO"))
    flipped = copy.deepcopy(t)
    for o in flipped.occupied()[::2]:
        o.coeff_vector = -o.coeff_vector
    assert np.allclose(repulsion_matrix(flipped), repulsion_matrix(t), rtol=1e-12, atol=1e-14)


def test_undefined_entries_are_stored_as_zero(sp_record, pair):
    """A single-atom fragment has KS1 == 0 within itself (no resonance integral inside an
    atom): log|KS1| is undefined there and stored as 0, not as the log floor."""
    methanol, _, kernels, _ = pair
    bromide = build_molecule_tokens(with_sterics(sp_record("[Br-]")))
    block = pair_block([bromide, methanol], kernels)
    n_occ, n_virt = len(bromide.occupied()), len(bromide.virtual())
    assert n_occ and n_virt
    assert not block.defined[:n_occ, :n_virt, F["log_abs_ks1"]].any()
    assert (block.z[~block.defined] == 0).all()


def _cell(methanol, chloromethane, donor, acceptor) -> tuple[int, int]:
    """(row, column) of a methanol donor and a chloromethane acceptor in the block."""
    i = next(k for k, o in enumerate(methanol.occupied()) if o is donor)
    j = len(methanol.virtual()) + next(
        k for k, o in enumerate(chloromethane.virtual()) if o is acceptor
    )
    return i, j


def test_layout_and_defined_entries(pair):
    methanol, chloromethane, _, block = pair
    n_occ = len(methanol.occupied()) + len(chloromethane.occupied())
    n_virt = len(methanol.virtual()) + len(chloromethane.virtual())
    assert len(PAIR_FEATURES) == 11 and block.z.shape == (n_occ, n_virt, 11)
    assert block.n_non_finite == 0 and (block.z[~block.defined] == 0).all()
    intra = np.zeros((n_occ, n_virt), bool)
    intra[: len(methanol.occupied()), : len(methanol.virtual())] = True
    intra[len(methanol.occupied()) :, len(methanol.virtual()) :] = True
    for name in ("log_beta2_d2.0", "log_beta2_d2.3", "log_beta2_d3.0", "log_abs_g", "inter"):
        assert block.defined[..., F[name]].all(), name
    for name in ("centroid_distance", "cross_coupling"):
        assert np.array_equal(block.defined[..., F[name]], intra), name
    assert np.array_equal(block.z[..., F["inter"]] == 1, ~intra)
    assert set(np.unique(block.z[..., F["clash"]])) <= {0.0, 1.0}
    assert set(np.unique(block.gap_sign)) <= {-1, 1} and (block.gap_sign == -1).all()
    assert (block.sites >= 0).all()  # the oriented kernel records its argmax sites


def test_values_match_the_terms(pair):
    methanol, chloromethane, kernels, block = pair
    lone = highest_lone_pair(methanol, "O")
    star = sigma_star(chloromethane, {"C", "Cl"})
    i, j = _cell(methanol, chloromethane, lone, star)
    beta = {
        d: kernels.by_distance[d].beta(lone, methanol, star, chloromethane)
        for d in FEATURE_DISTANCES
    }
    for d in FEATURE_DISTANCES:
        assert block.z[i, j, F[f"log_beta2_d{d}"]] == pytest.approx(np.log(beta[d] ** 2))
    G = gap_kernel_matrix([lone], [star])[0, 0]
    assert block.z[i, j, F["log_abs_g"]] == pytest.approx(np.log(abs(G)))
    ks3 = beta[PRIMARY_DISTANCE] ** 2 * G
    assert block.z[i, j, F["log_abs_ks3"]] == pytest.approx(np.log(abs(ks3)))
    o_atom, c_atom = int(lone.descriptors.center_atoms[0]), int(star.descriptors.center_atoms[0])
    q = methanol.atoms[o_atom].mulliken_charge * chloromethane.atoms[c_atom].mulliken_charge
    r = vdw_sum(8, chloromethane.atoms[c_atom].atomic_number)
    assert block.z[i, j, F["ks2"]] == pytest.approx(COULOMB_EV_ANG * q / (4.0 * r))
    # intramolecular KS1 of a donor-acceptor pair: repulsion with the occupied tokens around m(s)
    s = sigma_star(methanol, {"C", "O"})
    j2 = next(k for k, o in enumerate(methanol.virtual()) if o is s)
    M = repulsion_matrix(methanol)
    occ = methanol.occupied()
    pos = np.array([a.position for a in methanol.atoms])
    m_s = pos[int(s.descriptors.center_atoms[0])]
    near = [
        k
        for k, o in enumerate(occ)
        if np.linalg.norm(pos[int(o.descriptors.center_atoms[0])] - m_s) <= NEIGHBOURHOOD_ANG
    ]
    r_i = next(k for k, o in enumerate(occ) if o is lone)
    expected = sum(M[r_i, k] for k in near if k != r_i)
    assert block.z[r_i, j2, F["log_abs_ks1"]] == pytest.approx(np.log(abs(expected)))
    d = np.linalg.norm(lone.descriptors.centroid - s.descriptors.centroid)
    assert block.z[r_i, j2, F["centroid_distance"]] == pytest.approx(d)
    # intramolecular beta: the one-molecule block (site rule applied), not two copies
    b_intra = kernels.main.beta(lone, methanol, s, methanol, same=True)
    assert block.z[r_i, j2, F["log_beta2_d2.3"]] == pytest.approx(np.log(b_intra**2))
    n_v = len(methanol.virtual())
    one = kernels.main.beta_matrix(occ, methanol, methanol.virtual(), methanol, same=True)
    two = kernels.main.beta_matrix(occ, methanol, methanol.virtual(), methanol)
    stored = block.z[: len(occ), :n_v, F["log_beta2_d2.3"]]
    assert np.allclose(stored, np.log(np.maximum(one**2, 1e-12)))
    assert not np.allclose(stored, np.log(np.maximum(two**2, 1e-12)))


def test_intramolecular_ks2_is_a_difference(pair):
    """Within a fragment KS2 = q q / eps (1/R_contact - 1/R): the change on bringing the two
    main centres from their reactant distance to the van der Waals contact."""
    methanol, chloromethane, kernels, block = pair
    lone = highest_lone_pair(methanol, "O")
    occ, virt = methanol.occupied(), methanol.virtual()
    r_i = next(k for k, o in enumerate(occ) if o is lone)
    a = int(lone.descriptors.center_atoms[0])
    checked = 0
    for s in sigma_stars(methanol):
        b = int(s.descriptors.center_atoms[0])
        if b == a:
            continue
        j = next(k for k, o in enumerate(virt) if o is s)
        qq = methanol.atoms[a].mulliken_charge * methanol.atoms[b].mulliken_charge
        R = np.linalg.norm(methanol.atoms[a].position - methanol.atoms[b].position)
        Rc = vdw_sum(methanol.atoms[a].atomic_number, methanol.atoms[b].atomic_number)
        expected = COULOMB_EV_ANG * qq / 4.0 * (1.0 / Rc - 1.0 / R)
        assert block.z[r_i, j, F["ks2"]] == pytest.approx(expected)
        checked += 1
    assert checked
    geometric = pair_block([methanol, chloromethane], kernels, ks2_form="geometric")
    n_o, n_v = len(occ), len(virt)
    lone_geo = geometric.z[r_i, :n_v, F["ks2"]]
    assert not np.allclose(lone_geo, block.z[r_i, :n_v, F["ks2"]])  # the ablation differs
    assert np.array_equal(  # across fragments both forms are the contact Coulomb
        block.z[:n_o, n_v:, F["ks2"]], geometric.z[:n_o, n_v:, F["ks2"]]
    )


def test_every_entry_follows_k_and_eps(pair):
    methanol, chloromethane, kernels, _ = pair
    reduced = pair_block([methanol, chloromethane], kernels, REDUCED)
    params = KSParams(K=2.0, eps=8.0)
    full = pair_block([methanol, chloromethane], kernels, params)
    moved = to_params(reduced.z, params, PAIR_FEATURES)
    both = full.defined & reduced.defined
    assert np.allclose(np.where(both, moved, 0), np.where(both, full.z, 0), rtol=1e-9, atol=1e-9)


def test_beta_decays_with_distance(pair):
    *_, block = pair
    near, far = block.z[..., F["log_beta2_d2.0"]], block.z[..., F["log_beta2_d3.0"]]
    assert np.median(near - far) > 0


def test_standardization(pair):
    *_, block = pair
    stats = fit_pair_stats([(block.z, block.defined)])
    zhat = standardize(block.z, block.defined, stats)
    for k in range(len(PAIR_FEATURES)):
        v = zhat[..., k][block.defined[..., k]]
        assert abs(v.mean()) < 1e-9 and (stats.std[k] == 0 or abs(v.std() - 1.0) < 1e-9)
    assert (zhat[~block.defined] == 0).all()
    back = PairStats.from_json(stats.to_json())
    assert np.allclose(back.mean, stats.mean) and np.allclose(back.std, stats.std)


def test_frozen_head_is_the_floor_minus_the_repulsion(pair):
    """Logits -(KS3 + KS2 + KS1) / tau in eV, no standardization: tau only scales, KS3 is the
    floor's own beta^2 G, KS1 >= 0 lowers the logit, and without KS1 the ranking is the
    floor's -(KS3 + KS2) exactly."""
    methanol, chloromethane, kernels, block = pair
    params = KSParams()
    reduced = pair_block([methanol, chloromethane], kernels, REDUCED)  # as stored in samples
    z = np.where(reduced.defined, to_params(reduced.z, params, PAIR_FEATURES), 0.0)
    assert np.allclose(z, np.where(block.defined, block.z, 0.0))  # the fixture is physical
    head = frozen_head(z, block.gap_sign, block.defined, params)
    ks3, ks2, ks1 = ks_energies(z, block.gap_sign, block.defined)
    assert np.allclose(head, -(ks3 + ks2 + ks1)) and (ks1 >= 0).all() and (ks1 > 0).any()
    hot = frozen_head(z, block.gap_sign, block.defined, KSParams(tau=0.25))
    assert np.allclose(hot, 4 * head)
    occ, virt = methanol.occupied(), chloromethane.virtual()
    n_o, n_v = len(occ), len(methanol.virtual())
    beta = kernels.main.beta_matrix(occ, methanol, virt, chloromethane, params)
    floor_ks3 = beta**2 * gap_kernel_matrix(occ, virt)
    assert np.allclose(ks3[:n_o, n_v:], floor_ks3, rtol=1e-9)
    no_ks1 = block.defined.copy()
    no_ks1[..., F["log_abs_ks1"]] = False
    floor = -(ks3 + ks2)
    assert np.array_equal(
        np.argsort(frozen_head(z, block.gap_sign, no_ks1), axis=None, kind="stable"),
        np.argsort(floor, axis=None, kind="stable"),
    )


def test_contact_repulsion_needs_no_buried_volume(sp_record):
    """Spec v7.2: no buried-volume factor in KS1. Without %Vbur (the sp fixture skips sterics)
    the contact form is defined everywhere it was, and one cell is (N_m(r) + N_m(r')) |beta| a."""
    from ksmech.physics.repulsion import contact_repulsion_matrix

    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    kernels = oriented_kernels()
    block = pair_block([methanol, chloromethane], kernels)
    n_occ_m, n_virt_m = len(methanol.occupied()), len(methanol.virtual())
    assert block.n_non_finite == 0
    assert block.defined[:n_occ_m, n_virt_m:, F["log_abs_ks1"]].any()  # intermolecular
    r, r2 = methanol.occupied()[0], chloromethane.occupied()[0]
    C = contact_repulsion_matrix([r], methanol, [r2], chloromethane, kernels.main)
    n = atom_populations(methanol)[int(r.descriptors.center_atoms[0])]
    n += atom_populations(chloromethane)[int(r2.descriptors.center_atoms[0])]
    beta = kernels.main.beta_matrix([r], methanol, [r2], chloromethane)[0, 0]
    amp = kernels.main.amplitude_matrix([r], methanol, [r2], chloromethane)[0, 0]
    assert C[0, 0] >= 0 and C[0, 0] == pytest.approx(n * abs(beta) * amp, rel=1e-12)


def _at(seq: list, x) -> int:
    """Position of `x` in `seq` by identity (tokens of two molecules may compare unequal-ly)."""
    return next(k for k, y in enumerate(seq) if y is x)


def test_valid_pairs_filter(sp_record):
    from ksmech.physics.pairs import valid_pairs

    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    frags = [methanol, chloromethane]
    valid = valid_pairs(frags)
    occ = [o for t in frags for o in t.occupied()]
    virt = [o for t in frags for o in t.virtual()]
    assert valid.shape == (len(occ), len(virt))
    d_type = [_at(virt, o) for o in chloromethane.virtual() if chloromethane.is_d_type(o)]
    assert d_type and not valid[:, d_type].any()  # chlorine d functions are never acceptors
    star = sigma_star(chloromethane, {"C", "Cl"})
    bond = next(o for o in chloromethane.occupied() if set(leading(o)) == set(leading(star)))
    assert bond.descriptors.overlap_population > 0 > star.descriptors.overlap_population
    assert not valid[_at(occ, bond), _at(virt, star)]  # the donor's own antibond
    lone = highest_lone_pair(methanol, "O")
    assert valid[_at(occ, lone), _at(virt, star)]
    assert valid.sum() == sum(len(t.donor_candidates()) for t in frags) * sum(
        len(t.acceptor_candidates()) for t in frags
    ) - sum(
        len(t.acceptor_candidates()) - len(t.acceptor_candidates(donor=d))
        for t in frags
        for d in t.donor_candidates()
    )


def test_atom_coulomb_matrix(pair):
    from ksmech.physics.pairs import atom_coulomb_matrix

    methanol, chloromethane, _, _ = pair
    M = atom_coulomb_matrix([methanol, chloromethane])
    atoms = methanol.atoms + chloromethane.atoms
    assert M.shape == (len(atoms),) * 2 and np.allclose(M, M.T) and not np.diag(M).any()
    a, b = 0, 1  # two atoms of methanol: the geometric distance
    r = np.linalg.norm(atoms[a].position - atoms[b].position)
    q = atoms[a].mulliken_charge * atoms[b].mulliken_charge
    assert M[a, b] == pytest.approx(COULOMB_EV_ANG * q / (4.0 * r))
    c = len(methanol.atoms)  # first atom of chloromethane: the van der Waals contact
    q = atoms[a].mulliken_charge * atoms[c].mulliken_charge
    r = vdw_sum(atoms[a].atomic_number, atoms[c].atomic_number)
    assert M[a, c] == pytest.approx(COULOMB_EV_ANG * q / (4.0 * r))
    assert np.allclose(atom_coulomb_matrix([methanol, chloromethane], REDUCED), 4.0 * M)
