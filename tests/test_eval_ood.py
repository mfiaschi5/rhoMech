"""The out-of-distribution report (`eval.ood`) on samples of the mini PMechDB steps (the fixture
molecules): each probe moves only its input and only between like positions, a pass is
reproducible, the decoder keeps the stored masks, the signature ignores the atoms, and the
report has its sections."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

from ksmech.arrows.batches import collate_samples
from ksmech.arrows.steps import MechanismSample, pair_stats, sample_for_step
from ksmech.eval import ood
from ksmech.model.config import ModelConfig
from ksmech.model.inputs import fit_feature_stats, token_columns
from ksmech.model.net import KSMech

TINY = ModelConfig(
    d=32, d_pair=16, layers=2, heads=4, decoder_hidden=16, physics_hidden=8, checkpoint=False
)
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def samples(mini_steps, token_loader) -> list[MechanismSample]:
    out = [sample_for_step(mini_steps[k], token_loader) for k in mini_steps]
    out = [s for s in out if isinstance(s, MechanismSample) and s.edit_label is not None]
    assert len(out) >= 3
    assert any(int(s.atom_fragment.max()) > 0 for s in out)  # a pool of several molecules
    return out


@pytest.fixture(scope="module")
def batch(samples) -> dict:
    return collate_samples(samples)


def _model(samples, **kw) -> KSMech:
    torch.manual_seed(0)
    cfg = replace(TINY, **kw)
    stats = fit_feature_stats(samples, *token_columns(cfg))
    return KSMech(cfg, stats, pair_stats(samples)).eval()


def _gen(seed: int = 1) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_fragment_probe_relabels_every_pool_of_several_molecules(batch):
    out = ood.permute_fragments(batch, _gen())
    old, new = batch["fragment"], out["fragment"]
    assert torch.equal(new < 0, old < 0)  # the padding stays
    for k in range(len(old)):
        real = old[k] >= 0
        n = int(old[k].max()) + 1
        if n == 1:
            assert torch.equal(new[k], old[k])
            continue
        assert not torch.equal(new[k], old[k])
        pairs = set(zip(old[k][real].tolist(), new[k][real].tolist(), strict=True))
        assert len(pairs) == n  # one new label per old one: the same partition
    assert out["same_fragment"] is batch["same_fragment"]
    assert out["atom_features"] is batch["atom_features"]


def test_position_probe_moves_positions_within_a_molecule_only(batch):
    out = ood.permute_positions(batch, _gen())
    a = batch["atom_positions"].shape[1]
    frag, real = batch["fragment"], batch["token_mask"]
    for name, sl in (("atom_positions", slice(0, a)), ("orbital_centroids", slice(a, None))):
        f, r = frag[:, sl], real[:, sl]
        assert not torch.equal(out[name], batch[name])
        assert torch.equal(out[name][~r], batch[name][~r])
        for k in range(len(f)):
            for g in torch.unique(f[k][r[k]]).tolist():
                m = (f[k] == g) & r[k]
                before = Counter(map(tuple, batch[name][k][m].tolist()))
                assert before == Counter(map(tuple, out[name][k][m].tolist()))


def test_lewis_probe_is_the_same_graph_on_other_atoms(batch):
    out = ood.permute_lewis(batch, _gen())
    order, new = batch["bond_order"], out["bond_order"]
    assert not torch.equal(new, order)
    assert torch.equal(new, new.transpose(1, 2))
    a = order.shape[1]
    frag = batch["fragment"][:, :a]
    for k in range(len(order)):
        assert sorted(new[k].flatten().tolist()) == sorted(order[k].flatten().tolist())
        other = frag[k][:, None] != frag[k][None, :]
        assert not new[k][other].any()  # no bond between two molecules
        assert sorted(new[k].sum(1).tolist()) == sorted(order[k].sum(1).tolist())  # degrees
    assert out["can_form"] is batch["can_form"] and out["r_pairs"] is batch["r_pairs"]


def test_pair_probe_moves_its_columns_within_the_two_kinds_of_cell(batch):
    names = ("g", "log_abs_g")
    out = ood.permute_pair_features(batch, names, _gen())
    stored = tuple(batch["pair_features"])
    cols = [stored.index(n) for n in names]
    rest = [k for k in range(len(stored)) if k not in cols]
    assert torch.equal(out["pair_z"][..., rest], batch["pair_z"][..., rest])
    assert torch.equal(out["pair_defined"][..., rest], batch["pair_defined"][..., rest])
    assert not torch.equal(out["pair_z"][..., cols], batch["pair_z"][..., cols])
    b, r, s = batch["pair_cells"].unbind(1)
    frag = batch["fragment"]
    inter = frag[b, batch["pair_row_pos"][b, r]] != frag[b, batch["pair_col_pos"][b, s]]
    assert inter.any() and (~inter).any()
    for stratum in (inter, ~inter):
        i = (b[stratum], r[stratum], s[stratum])

        def rows(x, i=i):
            z, d = x["pair_z"][i][:, cols], x["pair_defined"][i][:, cols]
            return Counter(map(tuple, torch.cat([z, d.float()], 1).tolist()))

        assert rows(out) == rows(batch)  # values with their flags, inside the stratum
    assert ood.permute_pair_features(batch, ("not_stored",), _gen()) is batch


def test_read_probes_follow_the_features_the_model_reads():
    full = ood.read_probes(("g", "log_abs_g", "cross_coupling", "bonded", "inter"))
    assert {"distances", "lewis", "pair:gap", "pair:geometry", "pair:connectivity"} <= set(full)
    assert {"geometry", "connectivity"} <= set(full)
    few = ood.read_probes(("g", "inter"))
    assert set(few) == {"distances", "lewis", "pair:gap"}


def test_score_is_one_row_per_sample_and_reproducible(samples):
    model = _model(samples)
    loader = [collate_samples(samples[:2]), collate_samples(samples[2:])]
    base = ood.score(model, loader, CPU, None)
    assert base["step_id"] == [s.step_id for s in samples]
    assert all(len(v) == len(samples) for v in base.values())
    same = ood.score(model, loader, CPU, None, lambda b, gen: b, 3)
    assert same["rank"] == base["rank"] and same["top"] == base["top"]
    assert np.allclose(same["nll"], base["nll"])
    a = ood.score(model, loader, CPU, None, ood.permute_positions, 5)
    b = ood.score(model, loader, CPU, None, ood.permute_positions, 5)
    assert a == b
    assert np.isfinite(a["nll"]).all()  # the label stays reachable under a probe
    assert not np.allclose(a["nll"], base["nll"])


def test_the_model_as_trained_depends_on_the_fragment_order(samples):
    """The defect the symmetry probe measures: the fragment-index embedding of the token
    embedding. `ModelConfig.fragment_embedding=False` removes it (test_model.py)."""
    model = _model(samples)
    loader = [collate_samples(samples)]
    base = ood.score(model, loader, CPU, None)
    moved = ood.score(model, loader, CPU, None, ood.permute_fragments, 0)
    several = [int(s.atom_fragment.max()) > 0 for s in samples]
    diff = np.abs(np.asarray(base["nll"]) - np.asarray(moved["nll"]))
    assert (diff[several] > 1e-6).all() and (diff[~np.asarray(several)] < 1e-5).all()


def test_summary_against_and_buckets():
    scores = {
        "step_id": list("abcd"),
        "rank": [0, 2, 7, 0],
        "nll": [0.1, 1.0, 3.0, 0.2],
        "top": [4, 5, 6, 7],
        "exact": [True, False, False, True],
    }
    s = ood.summary(scores)
    assert (s["n"], s["top1"], s["top3"], s["top5"], s["r_exact"]) == (4, 0.5, 0.75, 0.75, 0.5)
    assert ood.summary(scores, [False] * 4) == {"n": 0}
    probed = scores | {"rank": [1, 2, 7, 0], "top": [9, 5, 6, 7]}
    row = ood.against(scores, [probed, scores])
    assert row["d_top1"] == pytest.approx(-0.125) and row["top_changed"] == pytest.approx(0.125)
    assert row["repeats"] == 2 and row["n"] == 4
    assert [ood.bucket(c) for c in (0, 1, ood.SEEN_OFTEN - 1, ood.SEEN_OFTEN)] == [
        "unseen", "seen_rarely", "seen_rarely", "seen_often"
    ]  # fmt: skip
    sigs = {"a": "x", "b": "y", "c": None, "d": "x"}
    cov = ood.coverage(scores, sigs, Counter({"x": 50, "y": 2}))
    assert cov["seen_often"]["n"] == 2 and cov["seen_often"]["top1"] == 1.0
    assert cov["seen_rarely"]["n"] == 1 and cov["unseen"]["n"] == 1
    assert sum(cov[b]["share"] for b in ood.BUCKETS) == pytest.approx(1.0)
    assert ood.per_sample(scores)[1] == {"step_id": "b", "rank": 2, "nll": 1.0, "exact": False}


def test_signature_ignores_the_atom_indices(samples):
    s = samples[0]
    sig = ood.signature(s)
    assert sig is not None and "|" in sig
    n = s.n_atoms
    index = {i: n - 1 - i for i in range(n)}
    z = s.atom_categorical[::-1].copy()
    moved = replace(s, atom_categorical=z, edit_label=s.edit_label.mapped(index))
    assert ood.signature(moved) == sig
    assert ood.signature(replace(s, edit_label=None)) is None
    assert len({ood.signature(x) for x in samples}) > 1


def test_training_signatures_are_cached_beside_the_view(samples, tmp_path, monkeypatch):
    view = tmp_path / "samples_molecule_train.json"
    view.write_text("{}")
    calls = []

    def fake(path, allow_stale=False, ids=None):
        calls.append(ids)
        return iter(s for s in samples if ids is None or s.step_id in ids)

    every = [s.step_id for s in samples]
    monkeypatch.setattr("ksmech.arrows.steps.iter_samples", fake)
    monkeypatch.setattr("ksmech.arrows.steps.sample_ids", lambda path, allow_stale=False: every)
    first = ood.training_signatures(view)
    assert sum(first.values()) == len(samples)
    assert (tmp_path / "edit_signatures_molecule_train.json").is_file()
    monkeypatch.setattr(ood.prov, "require_clean", lambda header, path: None)  # a clean build
    assert ood.training_signatures(view) == first and len(calls) == 1  # read from the cache
    view.write_text('{"changed": 1}')
    assert ood.training_signatures(view) == first and len(calls) == 2  # the view changed
    part = ood.training_signatures(view, subset=2)  # the run's training subset, its own cache
    assert sum(part.values()) == 2 and len(calls[-1]) == 2
    assert (tmp_path / "edit_signatures_molecule_train_sub2.json").is_file()


def test_the_terms_sum_to_the_logit_and_each_ranks_alone(samples):
    """A decoder whose pair and physics terms are zeroed ranks by the site term alone: `site`
    then equals the full ranking and `no_site` cannot tell the candidates apart."""
    model = _model(samples)
    loader = [collate_samples(samples)]
    full = ood.score(model, loader, CPU, None)["rank"]
    with torch.no_grad():
        for lin in (model.decoder.u_sign, model.decoder.physics[2]):
            lin.weight.zero_()
            lin.bias.zero_()
    ranks = ood.term_ranks(model, loader, CPU, None)
    site_only = ood.score(model, loader, CPU, None)["rank"]
    assert ranks["site"] == site_only == ranks["no_physics"] == ranks["no_pair"]
    assert site_only != full
    assert all(r > 0 for r in ranks["no_site"])  # every candidate ties: the label never first


def test_report_has_the_three_sections(samples):
    model = _model(samples)
    seen = []
    report, base = ood.ood_report(
        model, [collate_samples(samples)], CPU, None, 2, 0, lambda *a: seen.append(a[:2])
    )
    assert report["baseline"]["n"] == len(samples) == len(base["rank"])
    assert set(report["terms"]) == {"site", "pair", "physics", "no_site", "no_pair", "no_physics"}
    assert all(
        t["n"] == len(samples) and 0 <= t["top1"] <= t["top5"] for t in report["terms"].values()
    )
    assert set(report["symmetry"]) == {"fragment_order"}
    assert {"distances", "lewis", "pair:gap", "geometry", "connectivity"} <= set(report["reads"])
    assert all(r["repeats"] == 2 for r in report["reads"].values())
    assert ("symmetry", "fragment_order") in seen and len(seen) == 1 + len(report["reads"])
