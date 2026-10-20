"""The oriented contact overlap: local tables, rotations, lobes, placement, cache, kernel."""

import numpy as np
import pytest

from ksmech.orbitals.tokens import build_molecule_tokens
from ksmech.physics import oriented as O
from ksmech.physics.contact_kernel import ContactConfig, ContactKernel
from ksmech.physics.gto import ao_values, atom_ao_slices, atom_pair_blocks, element_shells
from ksmech.physics.params import KSParams
from ksmech.qm.geometry import BOHR_ANG
from tests.token_select import highest_lone_pair, leading, sigma_star, sigma_stars


@pytest.mark.parametrize("pair", [(6, 1), (8, 6), (17, 6), (22, 17), (35, 53)])
def test_local_tables_reproduce_the_direct_blocks(pair):
    V = np.random.default_rng(0).normal(size=(40, 3)) * 2.5
    direct = atom_pair_blocks(*pair, V)
    assert np.abs(O.atom_pair_blocks_local(*pair, V) - direct).max() < 1e-9


@pytest.mark.parametrize("z", [6, 17, 22])
def test_shell_rotation_rotates_the_function(z):
    rng = np.random.default_rng(1)
    M = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    M *= np.sign(np.linalg.det(M))
    c = rng.normal(size=sum(s.n_sph for s in element_shells(z)))
    c2, k = np.empty_like(c), 0
    for sh in element_shells(z):
        c2[k : k + sh.n_sph] = O.shell_rotation(sh.l, M) @ c[k : k + sh.n_sph]
        k += sh.n_sph
    pts = rng.normal(size=(30, 3))
    f = ao_values(np.array([z]), np.zeros((1, 3)), pts) @ c
    g = ao_values(np.array([z]), np.zeros((1, 3)), pts @ M.T) @ c2  # same function, x' = M x
    assert np.abs(f - g).max() < 1e-12


def test_rotation_to():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(5, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.vstack([rng.normal(size=(4, 3)), -a[4:]])
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    R = O.rotation_to(a, b)
    assert np.allclose(np.einsum("nij,nj->ni", R, a), b)
    assert np.allclose(R @ np.swapaxes(R, 1, 2), np.eye(3)) and np.allclose(np.linalg.det(R), 1)


def _angle(t, site, lobe, atom) -> float:
    pos = np.array([a.position for a in t.atoms])
    v = pos[atom] - pos[site]
    return float(np.degrees(np.arccos(lobe @ v / np.linalg.norm(v))))


def _at(seq, x) -> int:
    return next(k for k, y in enumerate(seq) if y is x)


def test_site_lobe_rules(sp_record):
    t = build_molecule_tokens(sp_record("CCl"))
    cands = O.molecule_candidates(t)
    star = sigma_star(t, {"C", "Cl"})
    k = _at(t.orbitals, star)
    carbon, chlorine = leading(star)
    assert _angle(t, carbon, cands[k, carbon][0][0], chlorine) > 160  # backside at C
    assert _angle(t, chlorine, cands[k, chlorine][0][0], carbon) > 160  # and away from C at Cl
    for ch in sigma_stars(t, {"C", "H"}):  # H site: an s function, outward beyond H
        h = next(a for a in leading(ch) if t.symbol(a) == "H")
        c = next(a for a in leading(ch) if t.symbol(a) == "C")
        lobes, opposite = cands[_at(t.orbitals, ch), h]
        assert len(lobes) == 1 and _angle(t, h, lobes[0], c) > 170
        assert np.allclose(opposite, -lobes[0])
    acetone = build_molecule_tokens(sp_record("CC(C)=O"))
    pi_star = min(acetone.virtual(), key=lambda o: o.energy_ev)
    c1 = next(a for a in leading(pi_star) if acetone.symbol(a) == "C")
    o3 = next(a for a in leading(pi_star) if acetone.symbol(a) == "O")
    lobes, _ = O.molecule_candidates(acetone)[_at(acetone.orbitals, pi_star), c1]
    assert len(lobes) == 2  # the two faces of the pi* lobe tie
    assert all(85 < _angle(acetone, c1, n, o3) < 120 for n in lobes)
    assert lobes[0] @ lobes[1] < -0.8


def test_candidates_respect_the_hemisphere(sp_record):
    t = build_molecule_tokens(sp_record("C=CC=O"))
    pos = np.array([a.position for a in t.atoms])
    for (k, site), (lobes, _) in O.molecule_candidates(t).items():
        o = t.orbitals[k]
        others = [int(c) for c in o.descriptors.all_center_atoms if int(c) != site]
        ok = O.admissible(o, t, site)
        for n in lobes:
            if others and ok[np.argmin(np.linalg.norm(O.sphere_directions() - n, axis=1))]:
                assert all(n @ (pos[c] - pos[site]) <= 1e-6 for c in others)


def _brute(t_F, r, a, u, t_G, s, b, v, d):
    """l_r^T S l_s with G moved rigidly (site b to x_a + d u, lobe v onto -u), all direct."""
    zF = np.array([x.atomic_number for x in t_F.atoms])
    zG = np.array([x.atomic_number for x in t_G.atoms])
    xF = np.array([x.position for x in t_F.atoms]) / BOHR_ANG
    xG = np.array([x.position for x in t_G.atoms]) / BOHR_ANG
    R = O.rotation_to(v[None], -u[None])[0]
    placed = (xG - xG[b]) @ R.T + xF[a] + d / BOHR_ANG * u
    c = t_G.orbitals[s].coeff_vector.copy()
    for atom, sl in enumerate(atom_ao_slices(zG)):
        k = sl.start
        for sh in element_shells(int(zG[atom])):
            c[k : k + sh.n_sph] = O.shell_rotation(sh.l, R) @ c[k : k + sh.n_sph]
            k += sh.n_sph
    slF, slG = atom_ao_slices(zF), atom_ao_slices(zG)
    S = np.zeros((slF[-1].stop, slG[-1].stop))
    for i in range(len(zF)):
        for j in range(len(zG)):
            S[slF[i], slG[j]] = atom_pair_blocks(int(zF[i]), int(zG[j]), placed[j : j + 1] - xF[i])
    return t_F.orbitals[r].coeff_vector @ S @ c


def test_placed_overlaps_equal_the_brute_force(sp_record):
    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    lone = highest_lone_pair(methanol, "O")
    star = sigma_star(chloromethane, {"C", "Cl"})
    r, s = _at(methanol.orbitals, lone), _at(chloromethane.orbitals, star)
    a, b = int(lone.descriptors.center_atoms[0]), leading(star)[0]
    F, G = O._Side.of(methanol, [r]), O._Side.of(chloromethane, [s])
    EF, EG = O._entries(methanol, [r], [[a]]), O._entries(chloromethane, [s], [[b]])
    pl = O._placements(F, G, EF, EG, np.array([0]), O.TIER1, 2.3 / BOHR_ANG)
    values = O._overlaps(F, G, pl, np.arange(len(pl.sp)), 0.0)
    for k in range(len(pl.sp)):
        u = (pl.shift[k] - F.xyz[a]) / (2.3 / BOHR_ANG)
        v = -(pl.rot[k].T @ u)
        brute = _brute(methanol, r, a, u, chloromethane, s, b, v, 2.3)
        assert values[k] == pytest.approx(abs(brute), rel=1e-9, abs=1e-12)


def test_clash_guard_and_centre_maximum(sp_record):
    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    rows = [k for k, o in enumerate(methanol.orbitals) if o.kind == "occupied"]
    cols = [k for k, o in enumerate(chloromethane.orbitals) if o.kind == "virtual"]
    free = O.OrientedConfig(distance=2.3, clash=0.0)
    open_ = O.block_contacts(methanol, rows, chloromethane, cols, free)
    assert not open_.clash.any() and (open_.S > 0).all()
    jammed = O.OrientedConfig(distance=2.3, clash=100.0)  # every placement clashes
    closed = O.block_contacts(methanol, rows, chloromethane, cols, jammed)
    assert closed.clash.all() and (closed.S <= open_.S + 1e-12).all()  # the smallest found
    i, j = 1, 2  # the pair value is the best site pair; its sites are recorded
    r, s = rows[i], cols[j]
    per_site = {
        (int(a), int(b)): O.site_contact(methanol, r, int(a), chloromethane, s, int(b), free)[0]
        for a in methanol.orbitals[r].descriptors.all_center_atoms
        for b in chloromethane.orbitals[s].descriptors.all_center_atoms
    }
    assert open_.S[i, j] == pytest.approx(max(per_site.values()))
    assert per_site[tuple(open_.sites[i, j])] == pytest.approx(open_.S[i, j])


def test_blocks_are_cached_in_memory_and_on_disk(sp_record, tmp_path):
    t = build_molecule_tokens(sp_record("CO"))
    u = build_molecule_tokens(sp_record("CCl"))
    config = O.OrientedConfig(distance=2.6, cache_dir=tmp_path)
    first = O.oriented_block(t, u, "occ_virt", config)
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1 and first.S.shape == (len(t.occupied()), len(u.virtual()))
    O._BLOCKS.clear()
    again = O.oriented_block(t, u, "occ_virt", config)  # from disk
    assert np.array_equal(again.S, first.S) and np.array_equal(again.sites, first.sites)
    assert np.array_equal(again.clash, first.clash)
    occ_occ = O.oriented_block(t, u, "occ_occ", config)
    assert occ_occ.S.shape == (len(t.occupied()), len(u.occupied()))
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_oriented_kernel_amplitude_beta_and_details(sp_record):
    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    kernel = ContactKernel(config=ContactConfig(oriented=O.OrientedConfig(distance=2.3)))
    occ, virt = methanol.occupied(), chloromethane.virtual()
    block = O.oriented_block(methanol, chloromethane, "occ_virt", kernel.config.oriented)
    A = kernel.amplitude_matrix(occ, methanol, virt, chloromethane)
    assert np.allclose(A, block.S)
    B = kernel.beta_matrix(occ, methanol, virt, chloromethane, KSParams(K=2.0))
    f = 0.5 * (np.array([o.energy_ev for o in occ])[:, None] + [o.energy_ev for o in virt])
    assert np.allclose(B, 2.0 * A * f)
    sites, clash = kernel.oriented_details(occ, methanol, virt, chloromethane)
    assert np.array_equal(sites, block.sites) and np.array_equal(clash, block.clash)
    sub = kernel.amplitude_matrix(occ[1:3], methanol, virt[::2], chloromethane)
    assert np.allclose(sub, A[1:3][:, ::2])


@pytest.mark.parametrize("ell", [0, 1, 2])
def test_rotate_coeffs_equals_the_rotation_matrices(ell):
    rng = np.random.default_rng(4)
    M = np.linalg.qr(rng.normal(size=(6, 3, 3)))[0]
    c = rng.normal(size=(6, 2 * ell + 1))
    expected = np.einsum("tij,tj->ti", O.shell_rotation(ell, M), c)
    assert np.allclose(O.rotate_coeffs(ell, c, M), expected, atol=1e-13)
