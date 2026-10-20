"""KS2, KS1 and the pair vector z_ij over the tokens of one or two fragments."""

import copy

import numpy as np
import pytest

from ksmech.orbitals.tokens import build_molecule_tokens
from ksmech.physics.contact_kernel import ContactKernel
from ksmech.physics.contact_table import vdw_sum
from ksmech.physics.coulomb import coulomb
from ksmech.physics.ks_terms import ks3_contact, token_gap_kernel
from ksmech.physics.pairs import PAIR_FEATURES, fit_pair_stats, pair_features, standardize
from ksmech.physics.params import COULOMB_EV_ANG, REDUCED, KSParams, scale
from ksmech.physics.repulsion import (
    NEIGHBOURHOOD_ANG,
    atom_populations,
    repulsion_matrix,
)
from ksmech.qm.sterics import buried_volume_per_atom
from tests.token_select import highest_lone_pair, sigma_star

F = {name: k for k, name in enumerate(PAIR_FEATURES)}


def with_sterics(rec):
    """A copy of a single-point record with %Vbur filled in (the sp fixture skips sterics)."""
    out = copy.copy(rec)
    out.buried_volume = buried_volume_per_atom(rec.atomic_numbers, rec.coords_ang)
    return out


@pytest.fixture(scope="module")
def pair(sp_record, contact_tables):
    methanol = build_molecule_tokens(with_sterics(sp_record("CO")))
    chloromethane = build_molecule_tokens(with_sterics(sp_record("CCl")))
    kernel = ContactKernel(contact_tables)
    return methanol, chloromethane, kernel, pair_features([methanol, chloromethane], kernel)


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


def test_layout_and_defined_entries(pair):
    methanol, chloromethane, _, pf = pair
    n_m = len(methanol.atoms) + len(methanol.orbitals)
    n = n_m + len(chloromethane.atoms) + len(chloromethane.orbitals)
    assert pf.z.shape == (n, n, 7) and pf.n_non_finite == 0
    ix = pf.index
    occ = np.concatenate([ix.occ[0], ix.occ[1]])
    virt = np.concatenate([ix.virt[0], ix.virt[1]])
    ks3 = pf.defined[..., F["ks3"]]
    assert ks3[np.ix_(occ, virt)].all() and ks3.sum() == len(occ) * len(virt)
    coupling = pf.defined[..., F["coupling_sq"]]
    assert coupling.sum() == sum(len(ix.occ[f]) * len(ix.virt[f]) for f in (0, 1))
    assert not coupling[np.ix_(ix.occ[0], ix.virt[1])].any()  # intramolecular only
    assert pf.defined[..., F["inter"]].all() and pf.defined[..., F["distance"]].all()
    assert (pf.z[:n_m, :n_m, F["inter"]] == 0).all() and (pf.z[:n_m, n_m:, F["inter"]] == 1).all()
    assert not pf.defined[np.arange(n), np.arange(n), F["ks1"]].any()
    assert (pf.z[~pf.defined] == 0).all()


def test_values_match_the_scalar_terms(pair):
    methanol, chloromethane, kernel, pf = pair
    n_m = len(methanol.atoms) + len(methanol.orbitals)
    lone = highest_lone_pair(methanol, "O")
    c_cl = sigma_star(chloromethane, {"C", "Cl"})
    i, j = lone.token_id, n_m + c_cl.token_id
    assert pf.z[i, j, F["ks3"]] == pytest.approx(
        ks3_contact(lone, methanol, c_cl, chloromethane, kernel)
    )
    assert pf.z[i, j, F["gap_kernel"]] == pytest.approx(token_gap_kernel(lone, c_cl))
    assert pf.z[i, j, F["ks3"]] < 0
    o_atom = int(lone.descriptors.center_atoms[0])
    cl = int(c_cl.descriptors.center_atoms[0])
    z_o, z_cl = 8, chloromethane.atoms[cl].atomic_number
    assert pf.z[i, j, F["distance"]] == pytest.approx(vdw_sum(z_o, z_cl))
    q = methanol.atoms[o_atom].mulliken_charge * chloromethane.atoms[cl].mulliken_charge
    assert pf.z[i, j, F["ks2"]] == pytest.approx(COULOMB_EV_ANG * q / (4.0 * vdw_sum(z_o, z_cl)))
    # intramolecular KS1 of a donor-acceptor pair: repulsion with the occupied tokens around m(s)
    s = sigma_star(methanol, {"C", "O"})
    M = repulsion_matrix(methanol)
    occ = methanol.occupied()
    pos = np.array([a.position for a in methanol.atoms])
    m_s = pos[int(s.descriptors.center_atoms[0])]
    near = [
        k
        for k, o in enumerate(occ)
        if np.linalg.norm(pos[int(o.descriptors.center_atoms[0])] - m_s) <= NEIGHBOURHOOD_ANG
    ]
    r = occ.index(lone)
    expected = sum(M[r, k] for k in near if k != r)
    assert pf.z[lone.token_id, s.token_id, F["ks1"]] == pytest.approx(expected)


def test_every_entry_scales_with_k_and_eps(pair):
    methanol, chloromethane, kernel, _ = pair
    reduced = pair_features([methanol, chloromethane], kernel, REDUCED)
    params = KSParams(K=2.0, eps=8.0)
    full = pair_features([methanol, chloromethane], kernel, params)
    for name, k in F.items():
        assert np.allclose(full.z[..., k], reduced.z[..., k] * scale(params, name)), name


def test_standardization_uses_defined_entries_only(pair):
    *_, pf = pair
    stats = fit_pair_stats([pf])
    zhat = standardize(pf, stats)
    for k in range(len(PAIR_FEATURES)):
        v = zhat[..., k][pf.defined[..., k]]
        assert abs(v.mean()) < 1e-9 and (stats.std[k] == 0 or abs(v.std() - 1.0) < 1e-9)
    assert (zhat[~pf.defined] == 0).all()


def test_non_finite_entries_are_counted_and_left_undefined(sp_record, contact_tables):
    """Without %Vbur (NaN) the contact repulsion cannot be formed: counted, never silent."""
    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    pf = pair_features([methanol, chloromethane], ContactKernel(contact_tables))
    assert pf.n_non_finite > 0 and np.isfinite(pf.z).all()
    n_m = len(methanol.atoms) + len(methanol.orbitals)
    assert not pf.defined[:n_m, n_m:, F["ks1"]].any()  # every intermolecular KS1
    assert pf.defined[:n_m, :n_m, F["ks1"]].any()  # intramolecular KS1 needs no %Vbur
