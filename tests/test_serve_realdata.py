"""The design 7 equalities on real GFN2 tokens of MeO- + MeCl, read-only from the shared caches.

A `cached`-chemistry runtime reads `data/cache/tokens` (and the QM occupations in
`data/cache/qm`) through a `LayeredTokenLoader`; everything it writes (the state, the oriented
blocks, the analysis) lands in a tmp data root. It never computes QM (the no-xTB guard stays
on) and skips with a reason when either token file is missing or stale (TokenSchemaError).
"""

import math
from pathlib import Path

import numpy as np
import pytest

import ksmech
from ksmech.serve import physics_view as pv
from ksmech.serve import stepapply as sa
from tests.test_serve_fixtures import no_xtb  # noqa: F401 (fixture)

pytestmark = [pytest.mark.serve, pytest.mark.realdata]

REPO = Path(ksmech.__file__).resolve().parents[2]
KEYS = ("C[O-]", "CCl")


def _skip_reason(loader, key: str) -> str:
    if key in loader.stale:
        return f"real tokens for {key} are stale (TokenSchemaError): {loader.stale[key]}"
    return f"no real tokens for {key} under the shared cache (run ksmech-orbitals build)"


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    from ksmech.serve.settings import TaskEnv, load_settings
    from ksmech.serve.tasks import runtime

    root = tmp_path_factory.mktemp("serve-realdata")
    s = load_settings({}, data_root=root, chemistry="cached", shared_cache_root=REPO)
    assert s.shared_tokens.resolve() == (REPO / "data" / "cache" / "tokens").resolve()
    rt_ctx = runtime(TaskEnv(s))
    for key in KEYS:
        if rt_ctx.loader(key) is None:
            pytest.skip(_skip_reason(rt_ctx.loader, key))
    return s, rt_ctx


@pytest.fixture(scope="module")
def work(real):
    from ksmech.serve.analysis import pool_work
    from ksmech.serve.pools import SpeciesInput, initial_state, save_state

    s, rt_ctx = real
    state = initial_state([SpeciesInput(k, 1) for k in KEYS], s.limits)
    save_state(state, s)
    return pool_work(rt_ctx, state)


def _symbol(rt, site):
    f = int(rt.atom_fragment[site])
    return rt.fragments[f].tokens.symbol(site - int(rt.atom_offsets[f]))


def _sn2_cell(work):
    """The one-centre O lone pair (highest LMO energy) -> sigma*(C-Cl), by descriptors."""
    an, rt = work.an, work.rt
    donors = [
        t
        for t in an.rows
        if rt.orbital(int(t)).descriptors.n_centers == 1
        and {_symbol(rt, a) for a in rt.centres(int(t))} == {"O"}
    ]
    acceptors = [
        t
        for t in an.cols
        if {_symbol(rt, a) for a in rt.centres(int(t))} == {"C", "Cl"}
        and rt.orbital(int(t)).descriptors.overlap_population < 0
    ]
    assert donors and len(acceptors) == 1
    donor = max(donors, key=lambda t: rt.orbital(int(t)).energy_ev)
    return an.cell(int(donor), int(acceptors[0]))


def test_writes_stay_in_the_tmp_root(real, work):
    s, rt_ctx = real
    assert s.oriented_dir.is_relative_to(s.data_root)
    assert not s.oriented_dir.resolve().is_relative_to((REPO / "data" / "cache").resolve())
    assert work.npz.is_relative_to(s.data_root)
    assert work.an.meta["chemistry"] == "cached"


def test_identities(work):
    an = work.an
    assert np.array_equal(an.floor, -(an.ks3 + an.ks2))
    assert np.array_equal(an.reasons == 0, an.valid) and an.n_valid > 0
    stored = an.beta2("2.3")
    defined = ~np.isnan(stored)
    assert np.allclose(an.beta[defined] ** 2, stored[defined], rtol=1e-5, atol=1e-11)


def test_dominant_pair_and_g_on_every_valid_pair(work, real):
    from ksmech.physics.ks_terms import token_gap_kernel

    _, rt_ctx = real
    an, rt = work.an, work.rt
    for i, j in np.argwhere(an.valid):
        p = pv._pair(an, rt, int(i), int(j))
        ix_f = pv.block_indices(p.t_f, rt_ctx.occupations(p.key_f))
        ix_g = pv.block_indices(p.t_g, rt_ctx.occupations(p.key_g))
        d = pv.dominant_gap_pair(p.r, p.s, ix_f.occ, ix_g.virt, ix_f.homo, ix_g.lumo, 1.0)
        exact = token_gap_kernel(p.r, p.s)
        assert d["term"] == pytest.approx(d["share"] * exact, rel=1e-12)
        assert abs(float(an.gap_kernel[i, j]) - exact) <= 1e-5 * abs(exact) + 1e-9
        assert pv.spectral_consistent(p.r, ix_f.occ) and pv.spectral_consistent(p.s, ix_g.virt)


def test_brief_and_panel_on_the_sn2_cell(work, real):
    from ksmech.serve.schemas import ArrowPhysicsBrief

    _, rt_ctx = real
    an, rt = work.an, work.rt
    i, j = _sn2_cell(work)
    brief = ArrowPhysicsBrief.model_validate(pv.brief_physics(an, rt, i, j, rt_ctx.occupations))
    assert brief.pair_valid and brief.ks2_form == "contact"
    assert brief.contact_distance_ang == rt_ctx.kernels.main.config.oriented.distance == 2.3
    assert brief.dominant is not None and brief.dominant.index_source.startswith(
        ("qm_occupations", "aufbau")
    )
    assert "G_PRECISION" not in brief.flags and "BETA2_MISMATCH" not in brief.flags
    a, b = (int(x) for x in an.contact_sites[i, j])
    panel = pv.arrow_physics(an, rt, i, j, a, b, an.score_floor, 1.0, rt_ctx.occupations, None)
    assert panel["requested_sites_are_contact"]
    assert math.isclose(panel["ks3"], panel["beta2"] * panel["G"], rel_tol=1e-12)


def test_site_pairs_at_the_contact_reproduce_beta(work, real):
    _, rt_ctx = real
    an = work.an
    i, j = _sn2_cell(work)
    out = pv.site_pairs(work, rt_ctx.kernels, int(an.rows[i]), int(an.cols[j]))
    contact = [x for x in out["site_pairs"] if x["is_contact"]]
    assert len(contact) == 1
    assert contact[0]["beta"] == pytest.approx(float(an.beta[i, j]), rel=1e-10)


def test_sn2_step_and_encoding(work):
    from ksmech.serve.contracts import ArrowSpec

    an, rt = work.an, work.rt
    i, j = _sn2_cell(work)
    donor, acceptor = int(an.rows[i]), int(an.cols[j])
    o_site = next(iter(rt.centres(donor)))
    c_site = next(a for a in rt.centres(acceptor) if _symbol(rt, a) == "C")
    arrow = ArrowSpec(donor, o_site, acceptor, c_site)
    out = sa.apply_step(rt, [arrow])
    assert out.ok and out.state.pool_key == "COC.[Cl-]"
    enc = sa.encode_dataset_arrows(rt, out.product, [arrow])
    lone, sink = enc.code.split(";")
    o_map, c_map = (rt.mol.GetAtomWithIdx(rt.mol_atom[x]).GetAtomMapNum() for x in (o_site, c_site))
    assert lone == f"{o_map}={c_map}" and sink.startswith(f"{c_map},")
