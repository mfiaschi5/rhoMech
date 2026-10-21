"""Pipek-Mezey localization: rotation properties, convergence, valence-only guard, AO order."""

from dataclasses import replace

import numpy as np
import pytest
from tblite.interface import Calculator

from ksmech.orbitals import localize as L
from ksmech.qm.ao import ao_layout


def test_valence_electron_table_matches_gfn2():
    z = (1, 6, 8, 17, 26, 29, 30, 35, 48, 57, 80)
    assert [L.valence_electrons(x) for x in z] == [1, 4, 6, 7, 8, 11, 2, 7, 2, 3, 2]
    with pytest.raises(L.LocalizationError):
        L.valence_electrons(87)


def test_pm_is_an_orthogonal_rotation_of_the_block(sp_record):
    rec = sp_record("CCO")
    occ = rec.C[:, rec.occupations >= 0.5]
    C_loc, U = L.pipek_mezey(occ, rec.S, rec.ao_atom)
    assert np.allclose(U.T @ U, np.eye(U.shape[0]), atol=1e-10)
    assert np.allclose(C_loc, occ @ U)
    assert np.allclose(C_loc.T @ rec.S @ C_loc, np.eye(C_loc.shape[1]), atol=1e-8)
    assert np.allclose(occ @ occ.T, C_loc @ C_loc.T, atol=1e-10)  # same subspace


def test_pm_raises_the_functional_and_converges(sp_record):
    rec = sp_record("CCO")
    occ = rec.C[:, rec.occupations >= 0.5]
    C_loc, _, st = L.pipek_mezey_stats(occ, rec.S, rec.ao_atom)
    assert st.converged and 1 <= st.n_sweeps < 50 and st.n_orbitals == occ.shape[1]
    assert st.functional_end > st.functional_start + 0.5
    # a second pass from the localized result has nothing left to gain
    _, U2, st2 = L.pipek_mezey_stats(C_loc, rec.S, rec.ao_atom)
    assert st2.n_sweeps == 1 and abs(st2.functional_end - st.functional_end) < 1e-6
    assert np.allclose(np.abs(U2), np.eye(U2.shape[0]), atol=1e-3)


def test_pm_reports_non_convergence_when_sweeps_run_out(sp_record):
    rec = sp_record("CCO")
    occ = rec.C[:, rec.occupations >= 0.5]
    _, _, st = L.pipek_mezey_stats(occ, rec.S, rec.ao_atom, L.PMConfig(tol=0.0, max_sweeps=2))
    assert not st.converged and st.n_sweeps == 2


def test_pm_trivial_blocks(sp_record):
    rec = sp_record("O")
    one = rec.C[:, :1]
    C_loc, U = L.pipek_mezey(one, rec.S, rec.ao_atom)
    assert np.array_equal(C_loc, one) and U.shape == (1, 1)
    empty = rec.C[:, :0]
    C_loc, U = L.pipek_mezey(empty, rec.S, rec.ao_atom)
    assert C_loc.shape == (rec.S.shape[0], 0) and U.shape == (0, 0)
    with pytest.raises(L.LocalizationError):
        L.pipek_mezey(rec.C[:3], rec.S, rec.ao_atom)


def test_localize_record_splits_occupied_and_virtual(sp_record):
    rec = sp_record("O")
    loc = L.localize_record(rec)
    assert loc.C_occ.shape == (6, 4) and loc.C_virt.shape == (6, 2)
    assert loc.converged and list(loc.occ_index) == [0, 1, 2, 3]


def test_somos_are_never_mixed_with_doubly_occupied_mos(sp_record):
    rec = sp_record("[CH2]")  # triplet methylene: two C-H bonds, two SOMOs
    loc = L.localize_record(rec)
    single = rec.occupations[loc.occ_index] < L.DOUBLE_THRESHOLD
    assert single.sum() == 2 and loc.stats_occ.n_orbitals == 4
    assert not loc.U_occ[np.ix_(single, ~single)].any()
    assert not loc.U_occ[np.ix_(~single, single)].any()
    assert np.allclose(loc.U_occ.T @ loc.U_occ, np.eye(4))
    assert np.allclose(loc.C_occ, rec.C[:, loc.occ_index] @ loc.U_occ)


def test_valence_only_guard_rejects_core_electrons(sp_record):
    rec = sp_record("O")
    L.assert_valence_only(rec)
    with_core = replace(rec, occupations=rec.occupations + 0.5)
    with pytest.raises(L.LocalizationError, match="not valence-only"):
        L.assert_valence_only(with_core)
    bad_map = replace(rec, ao_atom=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32))
    with pytest.raises(L.LocalizationError, match="non-minimal AO count"):
        L.assert_valence_only(bad_map)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_p_axis_order_matches_tblite(axis):
    """HF along one Cartesian axis: the sigma MO's p coefficient on F must sit on that axis."""
    from ksmech.orbitals.classify import p_vector

    z = np.array([1, 9], dtype=np.int32)
    xyz = np.zeros((2, 3))
    xyz[1, axis] = 1.75
    calc = Calculator("GFN2-xTB", z, xyz)
    calc.set("verbosity", 0)
    calc.set("save-integrals", 1)
    res = calc.singlepoint()
    C = np.asarray(res.get("orbital-coefficients"))
    occ = np.asarray(res.get("orbital-occupations")) > 1
    sigma = int(np.argmax(np.abs(C[0, occ])))  # the occupied MO with the largest H 1s weight
    ao_atom, ao_l = ao_layout(z)
    vec = p_vector(C[:, occ][:, sigma], ao_atom, ao_l, atom=1)
    assert abs(vec[axis]) > 0.5
    assert all(abs(vec[k]) < 1e-6 for k in range(3) if k != axis)


def test_mulliken_populations_sum_to_one(sp_record):
    rec = sp_record("CCO")
    loc = L.localize_record(rec)
    Q = L.mulliken_populations(loc.C_occ, rec.S, rec.ao_atom, rec.n_atoms)
    assert Q.shape == (rec.n_atoms, loc.C_occ.shape[1])
    assert np.allclose(Q.sum(axis=0), 1.0, atol=1e-8)


# ---- intrinsic bond orbitals (default since spec v7.2)


def test_ibo_is_an_orthogonal_rotation_that_raises_the_functional(sp_record):
    rec = sp_record("CCO")
    occ = rec.C[:, rec.occupations >= 0.5]
    C_loc, U, st = L.localize_block(occ, rec.S, rec.ao_atom)
    assert np.allclose(U.T @ U, np.eye(U.shape[0]), atol=1e-10)
    assert np.allclose(C_loc, occ @ U) and np.allclose(occ @ occ.T, C_loc @ C_loc.T, atol=1e-10)
    assert st.converged and st.functional_end >= st.functional_start - 1e-12


def test_ibo_angle_never_loses_and_finds_the_exact_maximum():
    rng = np.random.default_rng(0)
    m, d, q = rng.random((5, 40)), rng.normal(size=(5, 40)) * 0.3, rng.normal(size=(5, 40)) * 0.3
    gamma = L.ibo_angles(m, d, q)
    dense = np.linspace(-np.pi / 4, np.pi / 4, 20001)
    best = L._ibo_gain(m, d, q, np.repeat(dense[:, None], 40, axis=1)).max(axis=0)
    got = L._ibo_gain(m, d, q, gamma[None, :])[0]
    assert (got >= -1e-15).all() and np.allclose(got, best, atol=1e-9)


def _ring_pi(tokens, kind: str) -> list:
    toks = tokens.occupied() if kind == "occ" else tokens.virtual()
    return [o for o in toks if o.descriptors.n_centers >= 2 and o.descriptors.cos_character() < 0.2]


def test_benzene_ibo_converges_with_three_pi_tokens_per_block(sp_record):
    """IBO converges in a few sweeps (no cap) and the e1g weight stays in the ring tokens:
    whatever local form the pi space takes, energies are invariant to it."""
    from ksmech.orbitals.frontier import frontier_blocks
    from ksmech.orbitals.tokens import build_molecule_tokens

    rec = sp_record("c1ccccc1")
    lo = L.localize_record(rec)
    assert lo.converged and max(lo.stats_occ.n_sweeps, lo.stats_virt.n_sweeps) < 60
    t = build_molecule_tokens(rec)
    pi_occ, pi_virt = _ring_pi(t, "occ"), _ring_pi(t, "virt")
    assert len(pi_occ) == 3 and len(pi_virt) == 3
    e1g = frontier_blocks(rec.orbital_energies_ev, rec.occupations)["homo"]
    rows = [list(lo.occ_index).index(k) for k in e1g]
    weight = sum(o.descriptors.full_weights[rows].sum() for o in pi_occ)
    assert abs(weight - 2.0) < 1e-3  # the local form moves no e1g weight out of the ring


@pytest.mark.xfail(
    strict=True,
    reason="In GFN2's minimal basis with Lowdin populations the exponent-4 maximum of benzene's "
    "pi space is three 3-centre LMOs (0.50 / 0.22 / 0.22; pi P4 0.202 vs 0.177 for Kekule; "
    "random starts reach the same P4), not three 2-centre bonds as spec v7.2 states. "
    "docs/decisions.md 2026-10-22, fix item 4b.",
)
def test_benzene_ibo_gives_two_centre_pi_bonds(sp_record):
    from ksmech.orbitals.tokens import build_molecule_tokens

    t = build_molecule_tokens(sp_record("c1ccccc1"))
    assert all(o.descriptors.n_centers == 2 for o in _ring_pi(t, "occ"))


def test_pm_mulliken_and_the_canonical_start_stay_available(sp_record):
    rec = sp_record("CC=O")
    occ = rec.C[:, rec.occupations >= 0.5]
    for cfg in (L.PMConfig(method="pm_mulliken", start="canonical"), L.PMConfig(start="canonical")):
        C_loc, _, st = L.localize_block(occ, rec.S, rec.ao_atom, cfg)
        assert st.converged and np.allclose(occ @ occ.T, C_loc @ C_loc.T, atol=1e-10)
    with pytest.raises(L.LocalizationError):
        L.PMConfig(method="boys")


def test_token_files_are_byte_identical_across_two_builds(sp_record, tmp_path):
    """Every array member of two independent builds is byte-identical; the provenance header
    differs only in `built_at`."""
    import json
    import zipfile

    from ksmech.orbitals.tokens import build_molecule_tokens, save_tokens

    rec = sp_record("c1ccccc1O")
    paths = [save_tokens(build_molecule_tokens(rec), tmp_path / f"{k}.npz") for k in "ab"]
    members = [dict((n, z.read(n)) for n in z.namelist()) for z in map(zipfile.ZipFile, paths)]
    header = "__provenance__.npy"
    assert {k: v for k, v in members[0].items() if k != header} == {
        k: v for k, v in members[1].items() if k != header
    }
    heads = [json.loads(np.load(p)["__provenance__"].item()) for p in paths]
    for h in heads:
        h.pop("built_at")
    assert heads[0] == heads[1]


@pytest.mark.slow
@pytest.mark.realdata
def test_molecules_that_hit_the_pm_cap_converge_under_ibo():
    """The 35 PMechDB molecules whose occupied or virtual block reached 200 PM sweeps in the
    2026-10-20 token build (tests/fixtures/pm_capped_smiles.json)."""
    import json
    from pathlib import Path

    from ksmech.qm.records import DEFAULT_CACHE_DIR, cache_path, is_valid_cache_file, load_record
    from ksmech.qm.xtb_calc import QMConfig

    method = QMConfig().method
    smiles = json.loads(Path("tests/fixtures/pm_capped_smiles.json").read_text())
    paths = {s: cache_path(DEFAULT_CACHE_DIR, s, 0, method) for s in smiles}
    if not all(is_valid_cache_file(p) for p in paths.values()):
        pytest.skip("PMechDB QM records missing")
    capped = []
    for s, path in paths.items():
        lo = L.localize_record(load_record(path, (s, 0, method)))
        if not lo.converged:
            capped.append((s, lo.stats_occ.n_sweeps, lo.stats_virt.n_sweeps))
    assert not capped, capped
