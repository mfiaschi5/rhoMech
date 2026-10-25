"""placement.py: the reconstructed oriented placement equals what the kernel evaluated
(`block_contacts` per site pair and `oriented_block` at the contact sites, rel 1e-10), the
geometry rules, the soft analytic check, the contact payload and the version pin."""

import numpy as np
import pytest

from ksmech.serve import placement
from ksmech.serve.analysis import pool_work
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.testing import requires
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    sn2_state,
    synthetic_loader,
    synthetic_root,
    synthetic_runtime,
    synthetic_settings,
)

pytestmark = [pytest.mark.serve, requires("contact")]
REL = 1e-10


@pytest.fixture(scope="module")
def work(synthetic_runtime, sn2_state):  # noqa: F811
    return pool_work(synthetic_runtime, sn2_state)


@pytest.fixture(scope="module")
def cfg(synthetic_runtime):  # noqa: F811
    return synthetic_runtime.kernels.main.config.oriented


def _symbols(t, atoms) -> set[str]:
    return {t.symbol(int(a)) for a in atoms}


def _token(t, kind: str, centres: set[str], n_centres: int | None = None) -> int:
    """Position in t.orbitals of the first token of `kind` whose centre elements are
    `centres` (and that has `n_centres` centres when given)."""
    for k, o in enumerate(t.orbitals):
        atoms = list(o.descriptors.all_center_atoms)
        if o.kind.value != kind or _symbols(t, atoms) != centres:
            continue
        if n_centres is None or len(atoms) == n_centres:
            return k
    raise AssertionError(f"no {kind} token with centres {centres} in {t.smiles}")


def _global(rt, f: int, k: int) -> int:
    return rt.n_atoms + int(rt.orbital_offsets[f]) + k


def _sn2_tokens(work):
    rt = work.rt
    f = [fr.key for fr in rt.fragments].index("C[O-]")
    g = [fr.key for fr in rt.fragments].index("CCl")
    t_F, t_G = rt.fragments[f].tokens, rt.fragments[g].tokens
    r = _token(t_F, "occupied", {"O"}, 1)
    s = _token(t_G, "virtual", {"C", "Cl"})
    return f, r, g, s


def _site_pairs(t_F, r, t_G, s):
    for a in t_F.orbitals[r].descriptors.all_center_atoms:
        for b in t_G.orbitals[s].descriptors.all_center_atoms:
            yield int(a), int(b)


def _assert_production(pl, t_F, r, a, t_G, s, b, cfg, intra):
    from ksmech.serve.upstream import oriented_module

    block_contacts = oriented_module().block_contacts

    bc = block_contacts(t_F, [r], t_G, [s], cfg, sites_F=[[a]], sites_G=[[b]], intra=intra)
    assert abs(pl.S - bc.S[0, 0]) <= REL * max(1.0, pl.S)
    assert abs(pl.beta1 - bc.beta[0, 0]) <= REL * max(1.0, abs(pl.beta1))
    assert pl.clash == bool(bc.clash[0, 0])


def _assert_geometry(pl, t_F, a, b, cfg):
    x_a = np.asarray(t_F.atoms[a].position)
    placed_b = pl.cap_xyz_ang[list(pl.cap_atoms).index(b)]
    assert np.linalg.norm(placed_b - x_a) == pytest.approx(cfg.distance, abs=1e-9)
    v = pl.acceptor_dirs[pl.slots[1]]
    np.testing.assert_allclose(pl.rot @ v, -pl.u, atol=1e-9)
    np.testing.assert_allclose(pl.donor_dirs[pl.slots[0]], pl.u, atol=1e-12)
    np.testing.assert_allclose(pl.rot @ pl.rot.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(pl.rot) == pytest.approx(1.0)
    assert pl.tier in (1, 2) and 0 <= pl.roll < 6


def test_inter_pair_matches_block_contacts_and_oriented_block(work, cfg):
    from ksmech.serve.upstream import oriented_module

    oriented_block = oriented_module().oriented_block
    from ksmech.physics.params import KSParams

    f, r, g, s = _sn2_tokens(work)
    rt, an = work.rt, work.an
    t_F, t_G = rt.fragments[f].tokens, rt.fragments[g].tokens
    i, j = an.cell(_global(rt, f, r), _global(rt, g, s))
    sites = zip(an.contact_sites[i, j], (f, g), strict=True)
    a, b = (int(x) - int(rt.atom_offsets[h]) for x, h in sites)
    pl = placement.reconstruct_contact(t_F, r, a, t_G, s, b, cfg)
    _assert_production(pl, t_F, r, a, t_G, s, b, cfg, intra=False)
    ob = oriented_block(t_F, t_G, "occ_virt", cfg)
    s_virt = s - len(t_G.occupied())
    assert tuple(ob.sites[r, s_virt]) == (a, b)
    assert abs(pl.S - ob.S[r, s_virt]) <= REL
    assert abs(KSParams().K * pl.beta1 - an.beta[i, j]) <= 1e-9
    assert abs(pl.S - an.amp[i, j]) <= 1e-9
    _assert_geometry(pl, t_F, a, b, cfg)
    checks = placement.verify_placement(pl, t_F, r, a, t_G, s, b, cfg, False, an.beta[i, j], True)
    assert checks["consistent"] and checks["production_ok"] and checks["geometry_ok"]
    assert checks["dS_block"] <= REL and checks["dbeta_analysis"] <= 1e-9
    assert checks["analytic_ok"], checks  # soft: screening at tol 1e-6 drops little here
    assert checks["analytic_drift"] <= 2e-5 + 1e-4 * pl.S


def test_every_site_pair_of_the_inter_pair(work, cfg):
    f, r, g, s = _sn2_tokens(work)
    t_F, t_G = work.rt.fragments[f].tokens, work.rt.fragments[g].tokens
    pairs = list(_site_pairs(t_F, r, t_G, s))
    assert len(pairs) == 2  # O x {C, Cl}
    for a, b in pairs:
        pl = placement.reconstruct_contact(t_F, r, a, t_G, s, b, cfg)
        _assert_production(pl, t_F, r, a, t_G, s, b, cfg, intra=False)
        _assert_geometry(pl, t_F, a, b, cfg)
        checks = placement.verify_placement(pl, t_F, r, a, t_G, s, b, cfg, False, None, False)
        assert checks["consistent"] and "dS_block" not in checks


def test_intra_pair_ethanol_partial_cap(synthetic_loader, cfg):  # noqa: F811
    from ksmech.serve.upstream import oriented_module

    oriented_block = oriented_module().oriented_block

    t = synthetic_loader("CCO")
    r = _token(t, "occupied", {"O"}, 1)
    s = next(
        k
        for k, o in enumerate(t.orbitals)
        if o.kind.value == "virtual"
        and _symbols(t, o.descriptors.all_center_atoms) == {"C", "H"}
        and t.atoms[int(o.descriptors.center_atoms[0])].atomic_number in (1, 6)
        and all(int(n) != 2 for n in o.descriptors.all_center_atoms)
    )
    ob = oriented_block(t, t, "occ_virt", cfg, intra=True)
    s_virt = s - len(t.occupied())
    contact = tuple(int(x) for x in ob.sites[r, s_virt])
    seen_contact = False
    for a, b in _site_pairs(t, r, t, s):
        pl = placement.reconstruct_contact(t, r, a, t, s, b, cfg)
        assert len(pl.cap_atoms) < len(t.atoms)  # a partial cap: O and its H stay out
        assert 2 not in set(pl.cap_atoms.tolist())
        _assert_production(pl, t, r, a, t, s, b, cfg, intra=True)
        _assert_geometry(pl, t, a, b, cfg)
        is_contact = (a, b) == contact
        checks = placement.verify_placement(pl, t, r, a, t, s, b, cfg, True, None, is_contact)
        assert checks["consistent"], checks
        if is_contact:
            seen_contact = True
            assert abs(pl.S - ob.S[r, s_virt]) <= REL
    assert seen_contact


def test_inter_partial_cap_acetone(synthetic_loader, cfg):  # noqa: F811
    t_F, t_G = synthetic_loader("C[O-]"), synthetic_loader("CC(C)=O")
    r = _token(t_F, "occupied", {"O"}, 1)
    s = _token(t_G, "virtual", {"C", "H"})
    for a, b in _site_pairs(t_F, r, t_G, s):
        pl = placement.reconstruct_contact(t_F, r, a, t_G, s, b, cfg)
        assert len(pl.cap_atoms) < len(t_G.atoms)
        _assert_production(pl, t_F, r, a, t_G, s, b, cfg, intra=False)
        _assert_geometry(pl, t_F, a, b, cfg)


def test_placed_cap_coefficients_reproduce_the_kernel_overlap(work, cfg):
    """The soft check's analytic S, recomputed independently with gto.overlap_matrix."""
    from ksmech.physics.gto import overlap_matrix

    f, r, g, s = _sn2_tokens(work)
    t_F, t_G = work.rt.fragments[f].tokens, work.rt.fragments[g].tokens
    a, b = next(_site_pairs(t_F, r, t_G, s))
    pl = placement.reconstruct_contact(t_F, r, a, t_G, s, b, cfg)
    c = placement.placed_cap_coefficients(t_G, s, pl.rot)
    n_cap_ao = sum(int((t_G.ao_atom == x).sum()) for x in pl.cap_atoms)
    assert c.shape == (n_cap_ao,)
    z = [x.atomic_number for x in t_F.atoms] + [
        t_G.atoms[int(x)].atomic_number for x in pl.cap_atoms
    ]
    xyz = np.vstack([[x.position for x in t_F.atoms], pl.cap_xyz_ang])
    S = overlap_matrix(np.array(z), xyz)
    l_r = t_F.orbitals[r].coeff_vector
    S_an = abs(float(l_r @ S[: len(l_r), len(l_r) :] @ c))
    assert S_an == pytest.approx(pl.S, abs=2e-5 + 1e-4 * pl.S)
    # an unrotated cap of the identity placement keeps its own coefficients
    same = placement.placed_cap_coefficients(t_G, s, np.eye(3))
    from ksmech.serve.cubes import cap_coefficients

    np.testing.assert_allclose(same, cap_coefficients(t_G, s, pl.cap_atoms), atol=1e-14)


def test_contact_payload_ks3_site_equals_the_panel(synthetic_runtime, sn2_state, work):  # noqa: F811
    f, r, g, s = _sn2_tokens(work)
    rt, an = work.rt, work.an
    donor, acceptor = _global(rt, f, r), _global(rt, g, s)
    i, j = an.cell(donor, acceptor)
    ds, as_ = (int(x) for x in an.contact_sites[i, j])
    body = placement.contact_payload(synthetic_runtime, sn2_state, an, donor, ds, acceptor, as_)
    assert body["is_contact_argmax"] and body["consistent"] and not body["same_fragment"]
    assert body["ks3_site"] == pytest.approx(float(an.ks3[i, j]), rel=1e-5)
    assert body["S"] == pytest.approx(float(an.amp[i, j]), abs=1e-9)
    assert body["beta"] == pytest.approx(float(an.beta[i, j]), abs=1e-9)
    assert body["distance_ang"] == 2.3 and body["clash_limit_ang"] == 1.5
    assert body["roll_deg"] == 60.0 * body["roll_step"]
    assert body["donor_lobe"]["site"] == ds and body["acceptor_lobe_placed"]["site"] == as_
    np.testing.assert_allclose(
        body["acceptor_lobe_placed"]["direction"], -np.array(body["donor_lobe"]["direction"])
    )
    cap = body["cap"]
    assert len(cap["atoms"]) == len(cap["local_ids"]) and cap["molblock"].strip()
    assert {a["global_site"] for a in cap["atoms"]} <= set(range(rt.atom_offsets[g], rt.n_atoms))
    context = body["acceptor_context"]
    assert context is not None and len(context["atoms"]) == len(rt.fragments[g].tokens.atoms)
    by_id = {a["record_index"]: (a["x"], a["y"], a["z"]) for a in context["atoms"]}
    for a in cap["atoms"]:
        np.testing.assert_allclose(by_id[a["record_index"]], (a["x"], a["y"], a["z"]), atol=1e-9)
    other = [int(x) for x in rt.centres(acceptor) if int(x) != as_][0]
    off = placement.contact_payload(synthetic_runtime, sn2_state, an, donor, ds, acceptor, other)
    assert not off["is_contact_argmax"] and off["consistent"]


def test_contact_payload_rejects_bad_ids(synthetic_runtime, sn2_state, work):  # noqa: F811
    f, r, g, s = _sn2_tokens(work)
    rt, an = work.rt, work.an
    donor, acceptor = _global(rt, f, r), _global(rt, g, s)
    off = int(rt.atom_offsets[f])
    t_F = rt.fragments[f].tokens
    hydrogen = next(off + i for i, a in enumerate(t_F.atoms) if a.atomic_number == 1)
    acc_site = min(rt.centres(acceptor))
    with pytest.raises(ServeError) as err:
        placement.contact_payload(
            synthetic_runtime, sn2_state, an, donor, hydrogen, acceptor, acc_site
        )
    assert err.value.code == Code.SITE_NOT_A_CENTRE and err.value.status == 422
    with pytest.raises(ServeError) as err:
        placement.contact_payload(synthetic_runtime, sn2_state, an, 0, 0, acceptor, acc_site)
    assert err.value.code == Code.NOT_AN_ORBITAL_TOKEN


def test_oriented_version_mismatch(monkeypatch, synthetic_runtime, sn2_state, work, cfg):  # noqa: F811
    from ksmech.serve.upstream import oriented_module

    oriented = oriented_module()

    f, r, g, s = _sn2_tokens(work)
    t_F, t_G = work.rt.fragments[f].tokens, work.rt.fragments[g].tokens
    monkeypatch.setattr(oriented, "ORIENTED_VERSION", 6)
    with pytest.raises(ServeError) as err:
        placement.reconstruct_contact(t_F, r, 1, t_G, s, 0, cfg)
    assert err.value.code == Code.PHYSICS_VERSION_MISMATCH and err.value.status == 409
    payload = {"state": sn2_state.to_dict(), "donor": 0, "donor_site": 0}
    payload |= {"acceptor": 0, "acceptor_site": 0}
    with pytest.raises(ServeError) as err:
        placement.t_contact(synthetic_runtime, payload)
    assert err.value.code == Code.PHYSICS_VERSION_MISMATCH
    monkeypatch.undo()
    monkeypatch.delattr(oriented, "_tier")
    with pytest.raises(ServeError) as err:
        placement.oriented_module()
    assert err.value.code == Code.PHYSICS_VERSION_MISMATCH


def test_t_contact_writes_json_and_cubes(synthetic_runtime, sn2_state, work):  # noqa: F811
    import json
    from pathlib import Path

    f, r, g, s = _sn2_tokens(work)
    rt, an = work.rt, work.an
    donor, acceptor = _global(rt, f, r), _global(rt, g, s)
    i, j = an.cell(donor, acceptor)
    ds, as_ = (int(x) for x in an.contact_sites[i, j])
    payload = {"state": sn2_state.to_dict(), "donor": donor, "donor_site": ds}
    payload |= {"acceptor": acceptor, "acceptor_site": as_}
    out = placement.t_contact(synthetic_runtime, payload)
    path = Path(out["path"])
    assert path.parent == synthetic_runtime.settings.contact
    body = json.loads(path.read_text())
    assert body["consistent"] and body["placement_version"] == placement.PLACEMENT_VERSION
    for which in ("donor", "acceptor"):
        cube = Path(body["cube_paths"][which])
        assert cube.parent == synthetic_runtime.settings.cubes and cube.is_file()
        assert f"which={which}" in cube.read_text().splitlines()[0]
    mtime = path.stat().st_mtime_ns
    assert placement.t_contact(synthetic_runtime, payload) == out  # cached
    assert path.stat().st_mtime_ns == mtime
