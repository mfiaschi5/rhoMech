"""The oriented contact overlap: local tables, rotations, lobes, placement, cache, kernel."""

import numpy as np
import pytest

from ksmech.orbitals.tokens import (
    build_molecule_tokens,
    site_excluded_matrix,
    site_pairs_excluded,
)
from ksmech.physics import oriented as O
from ksmech.physics.contact_kernel import ContactConfig, ContactKernel
from ksmech.physics.gto import ao_values, atom_ao_slices, atom_pair_blocks, element_shells
from ksmech.physics.params import KSParams
from ksmech.qm.fukui import HARTREE_TO_EV
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


def _shell_means(t) -> np.ndarray:
    """Fock diagonal (eV) averaged over the AOs of each (atom, l >= 1): one p or d shell per
    atom in GFN2; s AOs kept as they are."""
    f = np.diag(t.F) * HARTREE_TO_EV
    out = f.copy()
    for atom in np.unique(t.ao_atom):
        for ell in (1, 2):
            rows = (t.ao_atom == atom) & (t.ao_l == ell)
            if rows.any():
                out[rows] = f[rows].mean()
    return out


def _brute(t_F, r, a, u, t_G, s, b, v, d, with_beta=False):
    """l_r^T S l_s with the acceptor's cap moved rigidly (site b to x_a + d u, lobe v onto -u),
    its LMO truncated to the cap atoms, every block computed directly. `with_beta`: also the
    AO-level sum l_r^T [S (F_mu,mu + F_nu,nu) / 2] l_s (eV, K = 1), each AO energy carried by
    its own molecule's AO through the rotation."""
    cap = O.cap_atoms(t_G.orbitals[s], t_G)
    zF = np.array([x.atomic_number for x in t_F.atoms])
    zG = np.array([x.atomic_number for x in t_G.atoms])
    xF = np.array([x.position for x in t_F.atoms]) / BOHR_ANG
    xG = np.array([x.position for x in t_G.atoms]) / BOHR_ANG
    R = O.rotation_to(v[None], -u[None])[0]
    placed = (xG - xG[b]) @ R.T + xF[a] + d / BOHR_ANG * u
    c = t_G.orbitals[s].coeff_vector.copy()
    f_G, f_F = _shell_means(t_G), _shell_means(t_F)
    cf = c * f_G
    for atom, sl in enumerate(atom_ao_slices(zG)):
        k = sl.start
        for sh in element_shells(int(zG[atom])):
            rot = O.shell_rotation(sh.l, R)
            c[k : k + sh.n_sph] = rot @ c[k : k + sh.n_sph]
            cf[k : k + sh.n_sph] = rot @ cf[k : k + sh.n_sph]
            k += sh.n_sph
        if atom not in cap:
            c[sl], cf[sl] = 0.0, 0.0
    slF, slG = atom_ao_slices(zF), atom_ao_slices(zG)
    S = np.zeros((slF[-1].stop, slG[-1].stop))
    for i in range(len(zF)):
        for j in cap:
            S[slF[i], slG[j]] = atom_pair_blocks(int(zF[i]), int(zG[j]), placed[j : j + 1] - xF[i])
    l_r = t_F.orbitals[r].coeff_vector
    if not with_beta:
        return l_r @ S @ c
    return l_r @ S @ c, 0.5 * ((l_r * f_F) @ S @ c + l_r @ S @ cf)


def _methyl_sigma_star(ethanol):
    """sigma*(C-H) on the methyl carbon (the carbon not bonded to O): its cap leaves out O."""
    oxygen = next(a.atom_index for a in ethanol.atoms if a.atomic_number == 8)
    methyl = next(
        a.atom_index
        for a in ethanol.atoms
        if a.atomic_number == 6 and oxygen not in O.bonded_neighbours(ethanol, a.atom_index)
    )
    return next(o for o in sigma_stars(ethanol, {"C", "H"}) if methyl in leading(o)), methyl


def _cases(sp_record):
    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    ethanol = build_molecule_tokens(sp_record("CCO"))
    star, methyl = _methyl_sigma_star(ethanol)
    return {  # (donor tokens, donor, acceptor tokens, acceptor, acceptor site)
        "inter, whole-molecule cap": (
            methanol,
            highest_lone_pair(methanol, "O"),
            chloromethane,
            sigma_star(chloromethane, {"C", "Cl"}),
            leading(sigma_star(chloromethane, {"C", "Cl"}))[0],
        ),
        "inter, partial cap": (
            methanol,
            highest_lone_pair(methanol, "O"),
            ethanol,
            sigma_star(ethanol, {"O", "H"}),
            leading(sigma_star(ethanol, {"O", "H"}))[0],
        ),
        "intra, partial cap": (ethanol, highest_lone_pair(ethanol, "O"), ethanol, star, methyl),
    }


@pytest.mark.parametrize(
    "case", ["inter, whole-molecule cap", "inter, partial cap", "intra, partial cap"]
)
def test_placed_overlaps_equal_the_brute_force(sp_record, case):
    t_F, lone, t_G, star, b = _cases(sp_record)[case]
    if "partial" in case:  # the truncation and the cap-local AO offsets are exercised
        assert len(O.cap_atoms(star, t_G)) < len(t_G.atoms)
    r, s = _at(t_F.orbitals, lone), _at(t_G.orbitals, star)
    a = int(lone.descriptors.center_atoms[0])
    F, G = O._Side.of(t_F, [r]), O._Side.of(t_G, [s])
    caps = {0: O._Side.cap(t_G, s)}
    EF, EG = O._entries(t_F, [r], [[a]]), O._entries(t_G, [s], [[b]])
    pl = O._placements(F, G, EF, EG, np.array([0]), O.TIER1, 2.3 / BOHR_ANG)
    values, betas = O._overlaps(F, caps, pl, np.arange(len(pl.sp)), 0.0)
    assert len(pl.sp) and np.abs(values).max() > 1e-4
    for k in range(len(pl.sp)):
        u = (pl.shift[k] - F.xyz[a]) / (2.3 / BOHR_ANG)
        v = -(pl.rot[k].T @ u)
        brute, beta = _brute(t_F, r, a, u, t_G, s, b, v, 2.3, with_beta=True)
        assert values[k] == pytest.approx(abs(brute), rel=1e-9, abs=1e-12)
        assert betas[k] == pytest.approx(beta * np.sign(brute), rel=1e-9, abs=1e-11)


@pytest.mark.parametrize("case", ["inter, partial cap", "intra, partial cap"])
def test_site_contact_places_the_cap(sp_record, case):
    """The production path (`site_contact` -> `block_contacts`) gives the cap-truncated brute
    force, maximized over the tied-lobe placements (clash guard off, no screening)."""
    t_F, lone, t_G, star, b = _cases(sp_record)[case]
    r, s = _at(t_F.orbitals, lone), _at(t_G.orbitals, star)
    a = int(lone.descriptors.center_atoms[0])
    F, G = O._Side.of(t_F, [r]), O._Side.of(t_G, [s])
    EF, EG = O._entries(t_F, [r], [[a]]), O._entries(t_G, [s], [[b]])
    pl = O._placements(F, G, EF, EG, np.array([0]), O.TIER1, 2.3 / BOHR_ANG)
    brute = []
    for k in range(len(pl.sp)):
        u = (pl.shift[k] - F.xyz[a]) / (2.3 / BOHR_ANG)
        brute.append(abs(_brute(t_F, r, a, u, t_G, s, b, -(pl.rot[k].T @ u), 2.3)))
    config = O.OrientedConfig(distance=2.3, clash=0.0, tol=0.0)
    value, clash = O.site_contact(t_F, r, a, t_G, s, b, config)
    assert not clash and value == pytest.approx(max(brute), rel=1e-9, abs=1e-12)


def test_cap_frees_intramolecular_placements(sp_record):
    """Placing a whole copy of the molecule against itself clashes where the local cap does
    not; a cap clash is always a whole-molecule clash (the cap is a subset)."""
    ethanol = build_molecule_tokens(sp_record("CCO"))
    lone = highest_lone_pair(ethanol, "O")
    r, a = _at(ethanol.orbitals, lone), int(lone.descriptors.center_atoms[0])
    F = O._Side.of(ethanol, [r])
    limit = 1.5 / BOHR_ANG
    freed = 0
    for star in sigma_stars(ethanol):
        s = _at(ethanol.orbitals, star)
        whole, cap = O._Side.of(ethanol, [s]), O._Side.cap(ethanol, s)
        for b in leading(star):
            EF, EG = O._entries(ethanol, [r], [[a]]), O._entries(ethanol, [s], [[b]])
            pl = O._placements(F, whole, EF, EG, np.array([0]), O.TIER1, 2.3 / BOHR_ANG)
            by_cap = O._clashes(F, {0: cap}, pl, limit)
            by_whole = O._clashes(F, {0: whole}, pl, limit)
            assert not (by_cap & ~by_whole).any()
            freed += int((by_whole & ~by_cap).sum())
    assert freed > 0


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
    assert np.allclose(B, 2.0 * block.beta) and (B < 0).mean() > 0.9  # bound AOs: beta < 0
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


def test_cap_is_the_centres_and_their_neighbours(sp_record):
    t = build_molecule_tokens(sp_record("CCO"))
    pos = np.array([a.position for a in t.atoms])
    oxygen = next(a.atom_index for a in t.atoms if a.atomic_number == 8)
    o_h = sigma_star(t, {"O", "H"})  # sigma*(O-H): exactly O, its H and the carbon on O
    expected = {*leading(o_h), *O.bonded_neighbours(t, oxygen).tolist()}
    assert set(O.cap_atoms(o_h, t).tolist()) == expected and len(expected) == 3
    for o in t.orbitals:
        cap = set(O.cap_atoms(o, t).tolist())
        centres = {int(a) for a in o.descriptors.all_center_atoms}
        assert centres <= cap
        for c in centres:
            assert set(O.bonded_neighbours(t, c).tolist()) <= cap
        far = [a for a in range(len(t.atoms)) if a not in cap]
        for a in far:  # nothing outside the cap is bonded to a centre
            assert all(np.linalg.norm(pos[a] - pos[c]) > 1.6 for c in centres)


def test_rolled_placements_keep_the_approach(sp_record):
    """A roll about the approach axis keeps the acceptor lobe antiparallel and its site fixed."""
    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    lone = highest_lone_pair(methanol, "O")
    star = sigma_star(chloromethane, {"C", "Cl"})
    r, s = _at(methanol.orbitals, lone), _at(chloromethane.orbitals, star)
    a, b = int(lone.descriptors.center_atoms[0]), leading(star)[0]
    F, G = O._Side.of(methanol, [r]), O._Side.of(chloromethane, [s])
    EF, EG = O._entries(methanol, [r], [[a]]), O._entries(chloromethane, [s], [[b]])
    base = O._placements(F, G, EF, EG, np.array([0]), O.TIER1[:1], 2.3 / BOHR_ANG)
    rolled = O._rolled(base, np.array([0]))
    assert len(rolled.sp) == O.N_ROLLS - 1 and list(rolled.roll) == list(range(1, O.N_ROLLS))
    v = EG.dirs[0, 0]
    u = base.axis[0]
    cap = O._Side.cap(chloromethane, s)
    at_base = O._placed(cap, base, slice(0, 1))[0] - base.shift[0]  # relative to the site
    for k in range(len(rolled.sp)):
        assert np.allclose(rolled.rot[k] @ v, -u)  # lobe still antiparallel
        assert np.allclose(rolled.rot[k] @ rolled.rot[k].T, np.eye(3))
        at_k = O._placed(cap, rolled, slice(k, k + 1))[0] - rolled.shift[k]
        # a roll about u through the site: each cap atom keeps its height along u and its
        # distance from the axis, and turns by k x 360 / N_ROLLS degrees about it
        assert np.allclose(at_k @ u, at_base @ u)
        off_base, off_k = at_base - np.outer(at_base @ u, u), at_k - np.outer(at_k @ u, u)
        assert np.allclose(np.linalg.norm(off_k, axis=1), np.linalg.norm(off_base, axis=1))
        far = np.argmax(np.linalg.norm(off_base, axis=1))
        cos = off_k[far] @ off_base[far] / (off_base[far] @ off_base[far])
        assert cos == pytest.approx(np.cos(2 * np.pi * rolled.roll[k] / O.N_ROLLS), abs=1e-9)


def _hbr_betas(qm_record):
    """AO-level and orbital-level beta (eV, d = 2.3) of HBr's own n(Br) -> sigma*(H-Br) on the
    production-method record (ALPB water, optimized)."""
    hbr = build_molecule_tokens(qm_record("Br"))
    bromine = next(a.atom_index for a in hbr.atoms if a.atomic_number == 35)
    lone = max((o for o in hbr.occupied() if leading(o) == (bromine,)), key=lambda o: o.energy_ev)
    star = next(o for o in hbr.virtual() if len(leading(o)) == 2)
    kernel = ContactKernel(config=ContactConfig(oriented=O.OrientedConfig(distance=2.3)))
    params = KSParams()
    ao = kernel.beta(lone, hbr, star, hbr, params, same=True)
    amp = kernel.amplitude(lone, hbr, star, hbr, same=True)
    orbital = params.K * amp * 0.5 * (lone.energy_ev + star.energy_ev)
    return ao, orbital, amp, lone, star


def test_ao_level_beta_removes_the_lmo_energy_cancellation(qm_record):
    """beta from the AO energies, not the LMO energies: sigma*(H-Br) lies far above 0 eV, so
    K S (E_r + E_s) / 2 nearly cancels for n(Br) -> sigma*(H-Br); the AO-level sum keeps the
    scale of bound valence AOs."""
    ao, orbital, amp, lone, star = _hbr_betas(qm_record)
    assert star.energy_ev > 5.0 and amp > 0.05
    assert abs(0.5 * (lone.energy_ev + star.energy_ev)) < 3.0  # the cancellation
    assert ao < 0 and abs(ao) / (KSParams().K * amp) > 8.0  # per unit overlap: AO scale
    assert abs(ao) >= 5 * abs(orbital), (ao, orbital)


def test_intramolecular_block_skips_self_site_pairs(sp_record):
    """Within one molecule the pair-track value comes from a site pair that is not a bond
    interacting with itself whenever the token pair has one; two copies of a molecule (two
    fragments) are an intermolecular block with its own cache entry."""
    methanol = build_molecule_tokens(sp_record("CO"))
    config = O.OrientedConfig(distance=2.3)
    occ = [k for k, o in enumerate(methanol.orbitals) if o.kind == "occupied"]
    virt = [k for k, o in enumerate(methanol.orbitals) if o.kind == "virtual"]
    intra = O.oriented_block(methanol, methanol, "occ_virt", config, intra=True)
    inter = O.oriented_block(methanol, methanol, "occ_virt", config)
    assert O._block_key(methanol, methanol, "occ_virt", config, True) != O._block_key(
        methanol, methanol, "occ_virt", config
    )
    tok_o, tok_v = methanol.occupied(), methanol.virtual()
    excluded = site_excluded_matrix(methanol, tok_o, tok_v, config.site_weight)
    for i, j in np.argwhere(~excluded):
        a, b = intra.sites[i, j]
        own = site_pairs_excluded(methanol, [tok_o[i]], [a], [tok_v[j]], [b], config.site_weight)
        assert not own[0]
    assert (intra.S <= inter.S + 1e-12).all()  # a restricted maximum
    assert (intra.S[~excluded] < inter.S[~excluded] - 1e-6).any()  # and it does restrict
    assert len(occ) == len(tok_o) and len(virt) == len(tok_v)
    kernel = ContactKernel(config=ContactConfig(oriented=config))  # `same` selects the block
    one = kernel.amplitude_matrix(tok_o, methanol, tok_v, methanol, same=True)
    two = kernel.amplitude_matrix(tok_o, methanol, tok_v, methanol)
    assert np.array_equal(one, intra.S) and np.array_equal(two, inter.S)
    beta = kernel.beta_matrix(tok_o, methanol, tok_v, methanol, KSParams(K=1.0), same=True)
    assert np.array_equal(beta, intra.beta)
    sites, clash = kernel.oriented_details(tok_o, methanol, tok_v, methanol, same=True)
    assert np.array_equal(sites, intra.sites) and np.array_equal(clash, intra.clash)


def test_ao_energies_are_rotation_invariant(sp_record):
    """The resonance integral's AO energies do not depend on the record's orientation: the
    lab-frame Fock diagonal of a p shell does (GFN2's anisotropic terms), its shell average
    does not."""
    from ksmech.qm.geometry import new_calculator
    from ksmech.qm.xtb_calc import GFN2Calculator

    rec = sp_record("CC(C)=O")
    M = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))[0]
    xyz = rec.coords_ang @ M.T
    calc = new_calculator(
        rec.atomic_numbers, xyz / BOHR_ANG, rec.charge, rec.unpaired, None, save_integrals=True
    )
    turned = GFN2Calculator()._assemble(
        calc.singlepoint(), rec.atomic_numbers, xyz, rec.smiles, 0, rec.charge, rec.unpaired
    )
    t, u = build_molecule_tokens(rec), build_molecule_tokens(turned)
    raw_t, raw_u = np.diag(t.F) * HARTREE_TO_EV, np.diag(u.F) * HARTREE_TO_EV
    assert np.abs(raw_t - raw_u).max() > 1e-3  # the lab-frame diagonal moves
    assert np.allclose(O.ao_energies(t), O.ao_energies(u), atol=1e-6)
