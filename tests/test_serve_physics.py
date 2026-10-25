"""physics_view.py on MeO- + MeCl (synthetic chemistry): the per-arrow brief and the full panel
against the stored pair features, the dominant canonical MO pair against token_gap_kernel, the
canonical index mapping, the floor rank against label_rank, the frozen-head weights, and the
site-pair block against block_contacts."""

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from ksmech.serve import physics_view as pv
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.schemas import ArrowPhysicsBrief, DominantPairOut, EnergyLevelsOut, SitePairOut
from ksmech.serve.testing import requires
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    sn2_state,
    synthetic_root,
    synthetic_runtime,
    synthetic_settings,
)

pytestmark = [pytest.mark.serve, requires("physics")]

RTOL, ATOL = 1e-5, 1e-11
SN2 = (16, 28)  # one-centre O lone pair -> sigma*(C-Cl), contact sites (O, C) = (1, 5)


@pytest.fixture(scope="module")
def work(synthetic_runtime, sn2_state):  # noqa: F811
    from ksmech.serve.analysis import pool_work

    return pool_work(synthetic_runtime, sn2_state)


def _valid_cells(an):
    return [tuple(int(x) for x in c) for c in np.argwhere(an.valid)]


def _z(work):
    from ksmech.physics.pairs import F
    from ksmech.physics.ranking import physical_pair_z

    return physical_pair_z(work.sample), work.sample.pair_defined, F


def _close(a, b) -> bool:
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)


def test_only_numpy_at_module_top():
    tree = ast.parse(Path(pv.__file__).read_text())
    allowed = {"__future__", "math", "re", "collections.abc", "dataclasses", "functools"}
    allowed |= {"typing", "numpy"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert {a.name for a in node.names} <= allowed
        elif isinstance(node, ast.ImportFrom):
            assert node.module in allowed, node.module


def test_brief_values_against_pair_z(work, synthetic_runtime):  # noqa: F811
    an, rt = work.an, work.rt
    zp, defined, F = _z(work)
    for i, j in _valid_cells(an):
        b = pv.brief_physics(an, rt, i, j, synthetic_runtime.occupations)
        if defined[i, j, F["log_abs_ks3"]]:
            assert _close(abs(b["ks3"]), math.exp(zp[i, j, F["log_abs_ks3"]]))
        if defined[i, j, F["log_beta2_d2.3"]]:
            assert _close(b["beta2"], math.exp(zp[i, j, F["log_beta2_d2.3"]]))
        if defined[i, j, F["log_abs_g"]]:
            assert _close(abs(b["G"]), math.exp(zp[i, j, F["log_abs_g"]]))
        assert (b["ks2"] is not None) == bool(defined[i, j, F["ks2"]])
        if b["ks2"] is not None:
            assert _close(b["ks2"], zp[i, j, F["ks2"]])
        if b["ks1"] is not None:
            assert _close(b["ks1"], math.exp(zp[i, j, F["log_abs_ks1"]]))
        assert b["floor"] == pytest.approx(-(an.ks3[i, j] + an.ks2[i, j]), abs=0)
        assert "BETA2_MISMATCH" not in b["flags"] and "G_PRECISION" not in b["flags"]


def test_brief_has_every_field(work, synthetic_runtime):  # noqa: F811
    an, rt = work.an, work.rt
    i, j = an.cell(*SN2)
    b = pv.brief_physics(an, rt, i, j, synthetic_runtime.occupations)
    assert set(b) == set(ArrowPhysicsBrief.model_fields)
    model = ArrowPhysicsBrief.model_validate(b)
    assert model.dominant is not None and model.dominant.k_label.startswith("HOMO")
    assert model.dominant.m_label.startswith("LUMO")
    assert model.ks2_form == "contact"  # MeO- and MeCl are two fragments
    assert model.contact_distance_ang == 2.3
    assert model.contact_distance_ang == synthetic_runtime.kernels.main.config.oriented.distance
    assert model.contact_sites == (1, 5)
    assert model.tau == 1.0 and model.frozen_source == "untrained_kernel"
    assert model.pair_valid and model.reason is None and "synthetic" in model.flags
    assert model.dominant.index_source == "qm_occupations"
    assert model.dominant.same_molecule is False


def test_ks2_form_difference_within_a_fragment(work, synthetic_runtime):  # noqa: F811
    an, rt = work.an, work.rt
    intra = [(i, j) for i, j in _valid_cells(an) if not an.inter[i, j] and an.ks2_ok[i, j]]
    assert intra
    i, j = intra[0]
    b = pv.brief_physics(an, rt, i, j, synthetic_runtime.occupations)
    assert b["ks2_form"] == "difference"
    d = pv.ks2_detail(an, rt, i, j)
    assert d["form"] == "difference" and d["R_geometric"] is not None
    want = 14.399645 * d["q_donor"] * d["q_acceptor"] / d["eps"]
    want *= 1.0 / d["R_contact"] - 1.0 / d["R_geometric"]
    assert _close(b["ks2"], want)


def test_ks2_detail_contact_form(work):
    an, rt = work.an, work.rt
    i, j = an.cell(*SN2)
    d = pv.ks2_detail(an, rt, i, j)
    assert d["form"] == "contact" and d["R_geometric"] is None
    assert d["main_centres"] == [1, 5]
    want = 14.399645 * d["q_donor"] * d["q_acceptor"] / (d["eps"] * d["R_contact"])
    assert _close(float(an.ks2[i, j]), want)


def test_dominant_pair_sum_equals_token_gap_kernel(work, synthetic_runtime):  # noqa: F811
    from ksmech.physics.ks_terms import token_gap_kernel

    an, rt = work.an, work.rt
    for i, j in _valid_cells(an):
        p = pv._pair(an, rt, i, j)
        ix_f = pv.block_indices(p.t_f, synthetic_runtime.occupations(p.key_f))
        ix_g = pv.block_indices(p.t_g, synthetic_runtime.occupations(p.key_g))
        d = pv.dominant_gap_pair(p.r, p.s, ix_f.occ, ix_g.virt, ix_f.homo, ix_g.lumo, 1.0)
        T, _ = pv._term_matrix(p.r, p.s)
        g = token_gap_kernel(p.r, p.s)
        assert math.isclose(float(T.sum()), g, rel_tol=1e-10, abs_tol=1e-12)
        assert d["term"] == pytest.approx(d["share"] * g, rel=1e-12)
        assert abs(d["term"]) == pytest.approx(float(np.abs(T).max()), rel=0, abs=0)
        assert d["k_canonical"] == int(ix_f.occ[d["k"]])
        assert d["m_canonical"] == int(ix_g.virt[d["m"]])
        assert d["ks3_part"] == d["term"]


def test_dominant_pair_consistency_failure(work, monkeypatch):
    from ksmech.physics import ks_terms

    an, rt = work.an, work.rt
    p = pv._pair(an, rt, *an.cell(*SN2))
    ix = pv.block_indices(p.t_f, None), pv.block_indices(p.t_g, None)
    monkeypatch.setattr(ks_terms, "token_gap_kernel", lambda r, s, *a: 123.0)
    with pytest.raises(ServeError) as e:
        pv.dominant_gap_pair(p.r, p.s, ix[0].occ, ix[1].virt, ix[0].homo, ix[1].lumo, 1.0)
    assert e.value.code == Code.CONSISTENCY_FAILED
    b = pv.brief_physics(an, rt, p.i, p.j, None)
    assert b["dominant"] is None and "CONSISTENCY_FAILED" in b["flags"]
    with pytest.raises(ServeError) as e:
        pv.arrow_physics(an, rt, p.i, p.j, 1, 5, None, 1.0, None, None)
    assert e.value.code == Code.CONSISTENCY_FAILED and e.value.status == 500


def test_canonical_indices_qm_and_aufbau(work, synthetic_runtime):  # noqa: F811
    from ksmech.qm.fukui import homo_index

    rt = work.rt
    for frag in rt.fragments:
        t, occ = frag.tokens, synthetic_runtime.occupations(frag.key)
        assert occ is not None  # the synthetic QM npz is present
        qm, auf = pv.block_indices(t, occ), pv.block_indices(t, None)
        assert qm.source == "qm_occupations" and auf.source == "aufbau"
        assert np.array_equal(qm.occ, auf.occ) and np.array_equal(qm.virt, auf.virt)
        assert qm.homo == homo_index(occ) and qm.lumo == qm.homo + 1
        for o in t.occupied():
            assert pv.spectral_consistent(o, qm.occ)
        for o in t.virtual():
            assert pv.spectral_consistent(o, qm.virt)


def test_index_source_aufbau_and_unchecked(work, synthetic_runtime):  # noqa: F811
    an, rt = work.an, work.rt
    i, j = an.cell(*SN2)
    hidden = pv.brief_physics(an, rt, i, j, lambda key: None)
    assert hidden["dominant"]["index_source"] == "aufbau"
    real = pv.brief_physics(an, rt, i, j, synthetic_runtime.occupations)
    assert hidden["dominant"]["k_canonical"] == real["dominant"]["k_canonical"]

    def shifted(key):  # same block sizes, every occupied index moved up by one
        occ = synthetic_runtime.occupations(key)
        n = int((occ >= 0.5).sum())
        out = np.zeros_like(occ)
        out[1 : n + 1] = 2.0
        return out

    b = pv.brief_physics(an, rt, i, j, shifted)
    assert b["dominant"]["index_source"] == "qm_occupations_unchecked"
    assert "index_unchecked" in b["flags"]


def test_labels():
    assert pv.homo_label(6, 6) == "HOMO" and pv.homo_label(4, 6) == "HOMO-2"
    assert pv.lumo_label(7, 7) == "LUMO" and pv.lumo_label(9, 7) == "LUMO+2"


def test_floor_rank_equals_label_rank(work):
    from ksmech.physics.ranking import label_rank

    an = work.an
    scores = np.where(an.valid, an.floor, -np.inf)
    for i, j in _valid_cells(an):
        onehot = np.zeros_like(an.valid)
        onehot[i, j] = True
        assert pv.floor_rank(an, i, j) == label_rank(scores, an.valid, onehot) + 1
    bad = tuple(int(x) for x in np.argwhere(~an.valid)[0])
    assert pv.floor_rank(an, *bad) is None


def test_label_rank_mirror_on_ties():
    from ksmech.physics.ranking import label_rank

    rng = np.random.default_rng(0)
    for _ in range(50):
        scores = rng.integers(0, 4, size=(4, 5)).astype(float)
        valid = rng.random((4, 5)) > 0.3
        positive = rng.random((4, 5)) > 0.7
        assert pv.label_rank_mirror(scores, valid, positive) == label_rank(scores, valid, positive)


def test_frozen_head_row_weights_sum_to_one(work):
    an = work.an
    for i in range(an.valid.shape[0]):
        cols = np.flatnonzero(an.valid[i])
        if not len(cols):
            continue
        weights = [pv.frozen_head_weight(an, i, int(j))["weight"] for j in cols]
        assert sum(weights) == pytest.approx(1.0, abs=1e-12)
    i, j = (int(x) for x in np.argwhere(~an.valid)[0])
    head = pv.frozen_head_weight(an, i, j)
    assert head["weight"] is None and head["logit"] == float(an.frozen[i, j])


def test_g_against_g_exact(work):
    from ksmech.physics.ks_terms import token_gap_kernel

    an, rt = work.an, work.rt
    for i, j in _valid_cells(an):
        p = pv._pair(an, rt, i, j)
        exact = token_gap_kernel(p.r, p.s)
        assert abs(float(an.gap_kernel[i, j]) - exact) <= 1e-5 * abs(exact) + 1e-9


def test_cell_rank_matches_propose_order(work):
    from ksmech.serve.contracts import PoolScores, log_softmax_valid
    from ksmech.serve.predictors import FloorPredictor

    an = work.an
    scores = np.where(an.valid, an.floor, -np.inf)
    logp = log_softmax_valid(scores, an.valid, 1.0)
    pool = PoolScores(scores, an.valid, logp, an.contact_sites, an.rows, an.cols, 1.0)
    for step in FloorPredictor("fp").propose(pool, None, k=1000):
        assert pv.cell_rank(scores, an.valid, *step.cell) == step.rank
        score, prob, rank = pv.predictor_cell(scores, an.valid, 1.0, *step.cell)
        assert rank == step.rank and prob == pytest.approx(math.exp(step.logp), rel=1e-12)


def test_energy_levels(work, synthetic_runtime):  # noqa: F811
    an, rt = work.an, work.rt
    i, j = an.cell(*SN2)
    panel = pv.arrow_physics(
        an, rt, i, j, 1, 5, an.score_floor, 1.0, synthetic_runtime.occupations, None
    )
    levels = EnergyLevelsOut.model_validate(panel["energy_levels"])
    dom = DominantPairOut.model_validate(panel["dominant_pair"])
    donor = {lv.canonical: lv for lv in levels.donor_levels}
    acceptor = {lv.canonical: lv for lv in levels.acceptor_levels}
    assert dom.k_canonical in donor and dom.m_canonical in acceptor
    assert donor[dom.k_canonical].energy == dom.E_k and acceptor[dom.m_canonical].energy == dom.E_m
    homo = max(donor)
    assert donor[homo].label == "HOMO" and {homo - 2, homo - 1} <= set(donor)
    lumo = min(acceptor)
    assert acceptor[lumo].label == "LUMO" and {lumo + 1, lumo + 2} <= set(acceptor)
    assert levels.homo_lumo_gap == pytest.approx(acceptor[lumo].energy - donor[homo].energy)
    assert dom.gap == pytest.approx(dom.E_k - dom.E_m)


def test_arrow_physics_panel(work, synthetic_runtime):  # noqa: F811
    from ksmech.serve.schemas import PhysicsOut

    an, rt = work.an, work.rt
    i, j = an.cell(*SN2)
    panel = pv.arrow_physics(
        an, rt, i, j, 1, 5, an.score_floor, 1.0, synthetic_runtime.occupations, None
    )
    assert set(PhysicsOut.model_fields) - set(panel) == {"predictor", "provenance"}
    assert panel["requested_sites_are_contact"] and panel["site_pair"] is None
    assert panel["beta2"] == panel["beta"] ** 2
    assert _close(panel["beta2"], panel["beta2_by_distance"]["2.3"])
    assert panel["G"] == float(an.gap_kernel[i, j]) and panel["ks3"] == float(an.ks3[i, j])
    assert panel["rank"] <= panel["floor_rank"]  # the floor's scores; ties count against
    assert panel["units"]["G"] == "1/eV" and panel["units"]["beta2"] == "eV^2"
    assert _close(panel["ks3"], panel["beta2"] * panel["G"])


def test_token_class_mirrors_ranking(work):
    from ksmech.physics.ranking import token_class

    for frag in work.rt.fragments:
        for o in frag.tokens.orbitals:
            assert pv.token_class(o, frag.tokens) == token_class(o, frag.tokens)


def _payload(state, donor, acceptor, **sites):
    return {"state": state.to_dict(), "donor": donor, "acceptor": acceptor, **sites}


def test_site_pairs_equal_block_contacts(work, synthetic_runtime, sn2_state):  # noqa: F811
    from ksmech.physics.ks_terms import token_gap_kernel
    from ksmech.serve.upstream import oriented_module

    block_contacts = oriented_module().block_contacts

    rt = work.rt
    out = pv.t_site_pairs(
        synthetic_runtime, _payload(sn2_state, *SN2, donor_site=1, acceptor_site=6)
    )
    assert out["note"] == pv.SITE_PAIR_NOTE and len(out["site_pairs"]) == 1
    sp = SitePairOut.model_validate(out["site_pairs"][0])
    assert not sp.is_contact and (sp.donor_site_map, sp.acceptor_site_map) == (2, 7)
    cfg = synthetic_runtime.kernels.main.config.oriented
    t_f, t_g = rt.fragments[0].tokens, rt.fragments[1].tokens
    r_loc = SN2[0] - rt.n_atoms
    s_loc = SN2[1] - rt.n_atoms - int(rt.orbital_offsets[1])
    bc = block_contacts(t_f, [r_loc], t_g, [s_loc], cfg, sites_F=[[1]], sites_G=[[1]])
    assert float(bc.S[0, 0]) == sp.S
    assert bool(bc.clash[0, 0]) == sp.clash
    assert sp.beta == pytest.approx(1.75 * float(bc.beta[0, 0]), rel=1e-12)
    g = token_gap_kernel(rt.orbital(SN2[0]), rt.orbital(SN2[1]))
    assert out["G"] == g and sp.ks3_site == pytest.approx(sp.beta**2 * g, rel=1e-12)
    assert not sp.site_rule_excluded  # two fragments: the site rule never applies


def test_site_pairs_all_centres_and_contact(work, synthetic_runtime, sn2_state):  # noqa: F811
    an = work.an
    i, j = an.cell(*SN2)
    out = pv.t_site_pairs(synthetic_runtime, _payload(sn2_state, *SN2))
    pairs = [(x["donor_site"], x["acceptor_site"]) for x in out["site_pairs"]]
    assert pairs == [(1, 5), (1, 6)]
    contact = [x for x in out["site_pairs"] if x["is_contact"]]
    assert len(contact) == 1 and (contact[0]["donor_site"], contact[0]["acceptor_site"]) == (1, 5)
    assert contact[0]["beta"] == pytest.approx(float(an.beta[i, j]), rel=1e-10)
    assert contact[0]["S"] == pytest.approx(float(an.amp[i, j]), rel=1e-10)


def test_site_pairs_intramolecular_rule(work, synthetic_runtime, sn2_state):  # noqa: F811
    from ksmech.orbitals.tokens import site_pairs_excluded

    rt = work.rt
    donor, acceptor = 12, 20  # sigma(C0-H2) -> sigma*(C0-O1), one fragment
    out = pv.t_site_pairs(synthetic_runtime, _payload(sn2_state, donor, acceptor))
    t = rt.fragments[0].tokens
    for x in out["site_pairs"]:
        want = site_pairs_excluded(
            t,
            [rt.orbital(donor)],
            [x["donor_site"]],
            [rt.orbital(acceptor)],
            [x["acceptor_site"]],
            rt.candidate.site_weight,
        )[0]
        assert x["site_rule_excluded"] == bool(want)
    assert any(x["site_rule_excluded"] for x in out["site_pairs"])


def test_site_pairs_errors(synthetic_runtime, sn2_state):  # noqa: F811
    with pytest.raises(ServeError) as e:
        pv.t_site_pairs(synthetic_runtime, _payload(sn2_state, *SN2, donor_site=0))
    assert e.value.code == Code.SITE_NOT_A_CENTRE and e.value.status == 422
    with pytest.raises(ServeError) as e:
        pv.t_site_pairs(synthetic_runtime, _payload(sn2_state, 3, SN2[1]))
    assert e.value.code == Code.NOT_AN_ORBITAL_TOKEN
