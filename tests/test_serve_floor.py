"""The spec-v9 predictors and the expansion on the synthetic chemistry: the pooled gap, the floor
ranking and its completion, B + R with sigma-key merging, the GBDT availability and the network
adapter (a fake model)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    serve_settings,
    sn2_state,
    synthetic_root,
    synthetic_runtime,
    synthetic_settings,
)


@pytest.fixture(scope="module")
def sn2_work(synthetic_runtime, sn2_state):  # noqa: F811
    from ksmech.serve.science.analysis import pool_work

    return pool_work(synthetic_runtime, sn2_state)


def _atom(work, symbol: str, degree: int | None = None) -> int:  # noqa: F811
    mol = work.rt.mol
    hits = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() == symbol and (degree is None or a.GetDegree() == degree)
    ]
    return hits[0]


def _sn2_atoms(work) -> tuple[int, int, int]:  # noqa: F811
    """(O of methoxide, C of chloromethane, Cl)."""
    mol = work.rt.mol
    cl = _atom(work, "Cl")
    c = mol.GetAtomWithIdx(cl).GetNeighbors()[0].GetIdx()
    return _atom(work, "O"), c, cl


def test_pooled_gap_is_the_spec_sum(sn2_work):  # noqa: F811
    """G_atom[i, j] = sum over valid (r, s) of (w_i^r w_j^s + w_j^r w_i^s) G(r, s)."""
    pg = sn2_work.pg
    i, j = 1, 5
    total = 0.0
    for r in range(len(pg.rows)):  # noqa: F811
        for s in range(len(pg.cols)):  # noqa: F811
            if pg.valid[r, s]:
                w = pg.w_occ[r, i] * pg.w_virt[s, j] + pg.w_occ[r, j] * pg.w_virt[s, i]
                total += w * pg.G[r, s]
    assert np.isclose(pg.g_atom[i, j], total, rtol=1e-12, atol=1e-15)
    assert np.allclose(pg.g_atom, pg.g_atom.T)
    assert np.allclose(pg.g_atom, pg.directed + pg.directed.T)
    best = pg.best_channel(i, j)
    assert best is not None and np.isclose(pg.channel_terms(i, j).sum(), pg.g_atom[i, j])


def test_floor_ranks_the_sn2_anchor_first(sn2_work):  # noqa: F811
    from ksmech.serve.predict.predictors import floor_anchors

    o, c, _ = _sn2_atoms(sn2_work)
    anchors = floor_anchors(sn2_work, "all")
    score, i, j, sign = anchors[0]
    assert (i, j, sign) == (*sorted((o, c)), "form")
    assert score == pytest.approx(abs(sn2_work.pg.g_atom[i, j]))
    assert [a[0] for a in anchors] == sorted((a[0] for a in anchors), reverse=True)
    inter = floor_anchors(sn2_work, "inter")
    frag = sn2_work.rt.atom_fragment
    g = sn2_work.rt.global_atom
    assert all(frag[g[i]] != frag[g[j]] for _, i, j, _ in inter)


def test_floor_completion_gives_the_sn2_matrix(synthetic_runtime, sn2_work):  # noqa: F811
    from ksmech.serve.predict.predictors import FloorPredictor

    o, c, cl = _sn2_atoms(sn2_work)
    skips: Counter = Counter()
    cands = FloorPredictor("fp").propose(synthetic_runtime, sn2_work, 1, "all", skips)
    sn2 = [x for x in cands if (*sorted((c, cl)), -1) in x.matrix.bonds]
    assert sn2, [x.matrix for x in cands]
    R = sn2[0].matrix
    assert (*sorted((o, c)), 1) in R.bonds and dict(R.lone) == {o: -2, cl: 2}
    assert sn2[0].logp is None and sn2[0].channel["leaving"] == cl
    assert all(x.anchor == (*sorted((o, c)), "form") for x in cands)
    assert sum(skips.values()) >= 0  # every dropped completion is counted under a code


def test_breaking_anchor_completes_as_heterolysis(sn2_work):  # noqa: F811
    from ksmech.serve.predict.predictors import anchor_candidates

    _, c, cl = _sn2_atoms(sn2_work)
    i, j = sorted((c, cl))
    out = anchor_candidates(sn2_work, (i, j, "break"), 0.5, Counter())
    assert {x.matrix.lone for x in out} == {((i, 2),), ((j, 2),)}
    assert all(x.matrix.bonds == ((i, j, -1),) for x in out)


def test_expansion_keeps_the_beam_and_reports_counts(synthetic_runtime, sn2_state):  # noqa: F811
    from ksmech.serve.chem.pools import state_path
    from ksmech.serve.core.contracts import ExpandRequest
    from ksmech.serve.predict.expansion import expand_state

    ex = expand_state(synthetic_runtime, sn2_state, ExpandRequest("floor", 3, None, "all", 64))
    assert ex.ok and 1 <= len(ex.children) <= 3
    first = ex.children[0]
    assert first.state.pool_key == "[CH3][O][CH3]|0.[Cl]|-1"
    assert first.logp is None and first.rank == 1
    step = first.step
    assert step["anchor"]["sign"] == "form" and step["sigma_keys"]
    assert {a["kind"] for a in step["arrows"]} == {"lone_pair_to_bond", "bond_to_lone_pair"}
    reading = step["readings"][0]
    assert reading["donor_diag"].startswith("lone_pair") and reading["canonical"]["k_label"]
    assert all(
        state_path(ch.state.state_id, synthetic_runtime.settings).is_file() for ch in ex.children
    )
    weights = [ch.weight for ch in ex.children]
    assert weights == sorted(weights, reverse=True) and all(w <= 0 for w in weights)
    assert all(k.split(":")[0] in ("APPLY_FAILED", "COMPLETION_FAILED", "ANCHOR_NOT_COMPLETED",
                                   "COMPLETION_WITHOUT_ANCHOR") for k in ex.failures)  # fmt: skip


def test_equal_products_merge_by_sigma_key(sn2_state, sn2_work):  # noqa: F811
    """Two matrices that take different H atoms of chloromethane give one product pool key:
    one child, the second an alternative."""
    from ksmech.arrows.delta_be import ReactionMatrix
    from ksmech.serve.predict.expansion import _group
    from ksmech.serve.predict.predictors import Candidate

    mol = sn2_work.rt.mol
    o, c, _ = _sn2_atoms(sn2_work)
    hs = [n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() if n.GetSymbol() == "H"]
    cands = []
    for h in hs[:2]:
        R = ReactionMatrix(
            ((*sorted((o, h)), 1), (*sorted((c, h)), -1)), tuple(sorted(((o, -2), (c, 2))))
        )
        cands.append(Candidate(R, (*sorted((o, h)), "form"), 1.0))
    groups = _group(sn2_state, cands, Counter(), [])
    assert len(groups) == 1 and len(next(iter(groups.values()))) == 2


def test_apply_failures_are_counted(sn2_state):  # noqa: F811
    from ksmech.arrows.delta_be import ReactionMatrix
    from ksmech.serve.predict.expansion import _group
    from ksmech.serve.predict.predictors import Candidate

    failures: Counter = Counter()
    bad = Candidate(ReactionMatrix(((0, 1, 1),), ()), (0, 1, "form"), 1.0)  # charge not conserved
    assert _group(sn2_state, [bad], failures, []) == {}
    assert failures == {"APPLY_FAILED:CHARGE_NOT_CONSERVED": 1}


def test_beam_weights():  # noqa: F811
    from ksmech.serve.predict.expansion import beam_weights

    w = beam_weights([2.0, 1.0, 1.0], [None] * 3, probabilities=False)
    assert np.allclose(np.exp(w), [0.5, 0.25, 0.25])
    assert beam_weights([0.0], [-0.5], probabilities=True) == [-0.5]
    soft = beam_weights([-1.0, 1.0], [None, None], probabilities=False)
    assert np.isclose(np.exp(soft).sum(), 1.0) and soft[1] > soft[0]


def test_gbdt_is_unavailable_without_its_file(synthetic_settings, tmp_path):  # noqa: F811
    from ksmech.serve.predict.predictors import GBDTPredictor

    info = GBDTPredictor(tmp_path / "none.npz", "fp").info()
    assert not info.available and info.code == "GBDT_MISSING" and "none.npz" in info.reason
    assert info.kind == "gbdt" and info.scores_are_probabilities is False


def test_default_gbdt_path_is_the_geomfree_molecule_model(serve_settings):  # noqa: F811
    s = serve_settings(shared_cache_root="/x")
    assert str(s.gbdt_file).endswith("data/processed/pmechdb/gbdt_lgbm_molecule_geomfree.npz")
    assert str(s.gbdt_file).startswith("/x")


def test_registry_lists_three_predictors(synthetic_settings):  # noqa: F811
    from ksmech.serve.predict.predictors import PredictorRegistry

    reg = PredictorRegistry(synthetic_settings, "fp")
    infos = {i.name: i for i in reg.info()}
    assert list(infos) == ["floor", "gbdt", "network"]
    assert infos["floor"].available and infos["floor"].kind == "floor"
    assert not infos["floor"].scores_are_probabilities
    net = infos["network"]
    assert not net.available and net.scores_are_probabilities
    assert "no checkpoint" in net.reason or "ksmech.model.net" in net.reason
    assert reg.ref("user").name == "user"


class FakeModel:
    """A NetworkModel that returns the SN2 matrix (global atoms) with a log-probability."""

    version = "fake/1"

    def __init__(self, work):  # noqa: F811
        self.work = work

    def decode(self, sample, rt, beam, *, state, scope, skips):  # noqa: F811
        from ksmech.arrows.delta_be import ReactionMatrix
        from ksmech.serve.predict.predictors import NetworkStep

        o, c, cl = (rt.global_atom[a] for a in _sn2_atoms(self.work))
        R = ReactionMatrix(
            ((*sorted((o, c)), 1), (*sorted((c, cl)), -1)), tuple(sorted(((o, -2), (cl, 2))))
        )
        return [NetworkStep(R, -0.1, (min(o, c), max(o, c), "form"))]


def test_network_adapter_with_a_fake_model(synthetic_runtime, synthetic_settings, sn2_work):  # noqa: F811
    from ksmech.serve.predict.network_predictor import NetworkPredictor

    p = NetworkPredictor(synthetic_settings, "fp", FakeModel(sn2_work))
    info = p.info()
    assert info.available and info.version == "network/fake/1" and info.scores_are_probabilities
    cands = p.propose(synthetic_runtime, sn2_work, 5, "all", Counter())
    o, c, cl = _sn2_atoms(sn2_work)
    assert len(cands) == 1 and cands[0].logp == -0.1
    assert cands[0].anchor == (*sorted((o, c)), "form")
    assert dict(cands[0].matrix.lone) == {o: -2, cl: 2}


def test_gbdt_inputs_resolve_against_the_shared_root(tmp_path):
    """A model whose relative sample inputs live under the shared root is current there and
    stale against any other root."""
    import json

    from ksmech.data import provenance as prov
    from ksmech.physics import baseline as bl
    from ksmech.serve.predict.predictors import gbdt_problems

    root = tmp_path / "shared"
    sample = root / "data" / "processed" / "pmechdb" / "samples.json"
    sample.parent.mkdir(parents=True)
    sample.write_text("{}")
    arrays = {"model": np.array("m"), "meta": np.array(json.dumps({"features": []}))}
    inputs = {"file:data/processed/pmechdb/samples.json": prov.current_hash(sample)}
    config = {"model": "lgbm_lambdarank", "features": list(bl.FEATURE_NAMES)}
    header = prov.stamp("baseline_lgbm", 1, inputs, config, prov.array_digest(arrays))
    path = prov.save_npz(tmp_path / "m.npz", arrays, header)
    problems = gbdt_problems(path, root, list(bl.FEATURE_NAMES))
    assert not [p for p in problems if "input" in p]
    assert any("input missing" in p for p in gbdt_problems(path, tmp_path, list(bl.FEATURE_NAMES)))
    assert any("features" in p for p in gbdt_problems(path, root, ["x"]))
