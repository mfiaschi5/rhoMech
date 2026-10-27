"""The network predictor in serve (`model.decode`, `serve.network_predictor`): the
best-first enumeration, log P(R) against a brute-force enumeration, the conservation, count
and valence filters with their reason codes, the merge by product, the default checkpoint and
the predictor's reason texts, and a tiny random checkpoint decoding a synthetic MeO- + MeCl
pool end to end (every returned step applies, its log P is the model's)."""

from __future__ import annotations

import itertools
import json
import time
from collections import Counter

import numpy as np
import pytest

from ksmech.model import decode as nd
from tests.serve_fake_model import write_fake_run
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    sn2_state,
    synthetic_root,
    synthetic_runtime,
    synthetic_settings,
)

pytestmark = pytest.mark.serve


# ---- the enumeration


def test_ranked_completions_are_every_bounded_completion_in_order():
    """Every completion within the counts, one class per entry, in decreasing log P."""
    rng = np.random.default_rng(3)
    t = _random_table(rng)
    pb = nd.problem(t, 64)
    got = list(nd.ranked_completions(pb, 2, 1, 1_000_000))
    scores = [s for s, _ in got]
    assert scores == sorted(scores, reverse=True)
    sets = {tuple(sorted(c)) for _, c in got}
    assert len(sets) == len(got)
    for score, chosen in got:
        assert sum(pb.is_bond[list(chosen)]) <= 2 and sum(~pb.is_bond[list(chosen)]) <= 1
        assert len(set(pb.entry[list(chosen)].tolist())) == len(chosen)
        assert score == pytest.approx(nd.exact_logp(pb, nd.classes(pb, chosen), t.anchor_logp))
    n_bond = int(pb.is_bond.sum())
    bonds_by_entry = Counter(pb.entry[pb.is_bond].tolist())
    atoms_by_entry = Counter(pb.entry[~pb.is_bond].tolist())

    def ways(counts, k):  # choices of at most k distinct entries, one class each
        vals = list(counts.values())
        return sum(
            np.prod([vals[i] for i in idx])
            for r in range(k + 1)
            for idx in itertools.combinations(range(len(vals)), r)
        )

    assert n_bond and len(got) == ways(bonds_by_entry, 2) * ways(atoms_by_entry, 1)


def _random_table(rng, n=4, anchor=(0, 1, nd.FORM), bonded=((1, 2), (2, 3))) -> nd.EntryTable:
    """Random entry log-probabilities with the decoder's masks: -1 only on a bond, -2 only on
    an atom with a lone pair (atoms 0 and 3)."""
    pair = rng.normal(size=(n, n, 3)) * 2.0
    for i, j in itertools.combinations(range(n), 2):
        if (i, j) not in bonded:
            pair[i, j, 0] = -np.inf
    atom = rng.normal(size=(n, 3)) * 2.0
    atom[[1, 2], 0] = -np.inf

    def norm(x):
        m = np.max(x, -1, keepdims=True)
        return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))

    return nd.EntryTable(anchor, -0.7, norm(pair), norm(atom))


def _brute_force(t: nd.EntryTable, cfg: nd.DecodeConfig, valid):
    """Every assignment of every entry, scored by the joint log P, filtered as the decoder
    filters: [(logp, bonds, lone)], best first."""
    n = t.atom_lp.shape[0]
    a, b, sign = t.anchor
    pairs = [(i, j) for i, j in itertools.combinations(range(n), 2) if (i, j) != (a, b)]
    out = []
    for pv in itertools.product(range(3), repeat=len(pairs)):
        for av in itertools.product(range(3), repeat=n):
            lp = t.anchor_logp
            lp += sum(t.pair_lp[i, j, k] for (i, j), k in zip(pairs, pv, strict=True))
            lp += sum(t.atom_lp[i, k] for i, k in enumerate(av))
            if not np.isfinite(lp):
                continue
            bonds = [(a, b, 1 if sign == nd.FORM else -1)]
            bonds += [(i, j, k - 1) for (i, j), k in zip(pairs, pv, strict=True) if k != 1]
            lone = [(i, 2 * (k - 1)) for i, k in enumerate(av) if k != 1]
            bonds, lone = tuple(sorted(bonds)), tuple(sorted(lone))
            if nd.screen(bonds, lone, t.anchor, cfg) is None and valid(bonds, lone):
                out.append((float(lp), bonds, lone))
    return sorted(out, key=lambda x: -x[0])


def _fake_apply(valid):
    """apply(R): every R is its own product unless `valid` refuses it (VALENCE_EXCEEDED)."""

    def apply(bonds, lone):
        return (repr((bonds, lone)), None) if valid(bonds, lone) else (None, "VALENCE_EXCEEDED")

    return apply


def _no_double_charge(bonds, lone):
    """A stand-in valence rule: atom 3 never gains a pair."""
    return (3, 2) not in lone


@pytest.mark.parametrize(
    ("seed", "anchor"), [(0, (0, 1, nd.FORM)), (1, (0, 1, nd.FORM)), (2, (1, 2, nd.BREAK))]
)
def test_completions_match_a_brute_force_enumeration(seed, anchor):
    rng = np.random.default_rng(seed)
    t = _random_table(rng, anchor=anchor)
    cfg = nd.DecodeConfig(per_anchor=6, max_bonds=3, max_atoms=2, max_applied=10_000)
    want = _brute_force(t, cfg, _no_double_charge)[:6]
    skips: Counter = Counter()
    got = list(nd.complete_anchor(t, _fake_apply(_no_double_charge), cfg, skips).values())
    assert [(c.bonds, c.lone) for c in got] == [(b, lo) for _, b, lo in want]
    assert [c.logp for c in got] == pytest.approx([lp for lp, _, _ in want], abs=1e-9)
    # every drop is counted under its rule; the search never proposes too many changes
    assert set(skips) <= {"DECODE_DROPPED:CHARGE_NOT_CONSERVED", "DECODE_DROPPED:VALENCE_EXCEEDED"}


def test_every_kept_completion_conserves_charge_and_passes_the_valence_check():
    rng = np.random.default_rng(7)
    t = _random_table(rng)
    cfg = nd.DecodeConfig(per_anchor=20, max_bonds=8, max_atoms=2, max_applied=10_000)
    skips: Counter = Counter()
    refused: list = []

    def valid(bonds, lone):  # the first completion that reaches B + R fails its valence
        if not refused:
            refused.append((bonds, lone))
        return (bonds, lone) != refused[0] and _no_double_charge(bonds, lone)

    got = nd.complete_anchor(t, _fake_apply(valid), cfg, skips).values()
    assert got
    for c in got:
        assert sum(d for _, d in c.lone) + 2 * sum(d for *_, d in c.bonds) == 0
        assert valid(c.bonds, c.lone) and len(c.lone) <= 2
        assert (0, 1, 1) in c.bonds  # the forming anchor
    assert skips["DECODE_DROPPED:CHARGE_NOT_CONSERVED"] > 0
    assert skips["DECODE_DROPPED:VALENCE_EXCEEDED"] > 0


def test_screen_rules():
    cfg = nd.DecodeConfig(max_bonds=2, max_atoms=1)
    form, brk = (0, 1, nd.FORM), (0, 1, nd.BREAK)
    assert nd.screen(((0, 1, 1), (1, 2, -1)), (), form, cfg) is None
    assert nd.screen(((0, 1, 1),), ((0, -2),), form, cfg) is None
    assert nd.screen(((0, 1, 1),), (), form, cfg) == "CHARGE_NOT_CONSERVED"
    assert nd.screen(((0, 1, 1), (1, 2, -1), (2, 3, 1)), (), form, cfg) == "TOO_MANY_CHANGES"
    assert nd.screen(((0, 1, -1),), ((0, 2),), brk, cfg) is None
    assert nd.screen(((0, 1, -1), (2, 3, 1)), (), brk, cfg) == "ANCHOR_NOT_ADMISSIBLE"


def test_merge_keeps_the_best_logp_per_product():
    a = nd.Completion(-1.0, ((0, 1, 1),), ((0, -2),), (0, 1, 0))
    b = nd.Completion(-0.5, ((0, 2, 1),), ((0, -2),), (0, 2, 0))
    c = nd.Completion(-2.0, ((1, 2, 1),), ((1, -2),), (1, 2, 0))
    skips: Counter = Counter()
    out = nd.merge([{"p": a, "q": c}, {"p": b}], skips)
    assert [x.logp for x in out] == [-0.5, -2.0]
    assert skips["DECODE_MERGED"] == 1


def test_a_new_double_charge_on_a_non_metal_is_refused():
    from rdkit import Chem

    from ksmech.serve.network_predictor import charge_issue

    before = Chem.MolFromSmiles("C[Cl]", sanitize=False)
    after = Chem.MolFromSmiles("C[Cl-2]", sanitize=False)
    assert charge_issue(before, after, [1]) == "FORMAL_CHARGE_RANGE"
    assert charge_issue(before, after, [0]) is None  # an atom the step does not touch
    perchlorate = Chem.MolFromSmiles("[O-][Cl+3]([O-])([O-])[O-]", sanitize=False)
    assert charge_issue(perchlorate, perchlorate, [1]) is None  # carried before
    li = Chem.MolFromSmiles("[Li-2]", sanitize=False)
    assert charge_issue(Chem.MolFromSmiles("[Li]", sanitize=False), li, [0]) is None  # a metal


def test_top_anchors_respect_the_scope_mask():
    lp = np.full((3, 3, 2), -np.inf)
    lp[0, 1, 0], lp[0, 2, 0], lp[1, 2, 1] = -0.1, -1.0, -2.0
    assert [a[1:] for a in nd.top_anchors(lp, 5)] == [(0, 1, 0), (0, 2, 0), (1, 2, 1)]
    allowed = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=bool)
    assert [a[1:] for a in nd.top_anchors(lp, 5, allowed)] == [(0, 2, 0), (1, 2, 1)]


# ---- checkpoints and the predictor's states


def _settings(tmp_path, **kw):
    from ksmech.serve.settings import load_settings

    return load_settings({}, data_root=tmp_path, shared_cache_root=tmp_path, **kw)


def _run(tmp_path, name, top1=None, limit=0, dirty=False):
    run_dir = tmp_path / "data" / "models" / "pmechdb" / name
    write_fake_run(run_dir, dirty=dirty)
    doc = json.loads((run_dir / "run.json").read_text())
    doc["provenance"]["config"]["train"]["limit"] = limit
    (run_dir / "run.json").write_text(json.dumps(doc))
    if top1 is not None:
        lines = [{"epoch": 0, "step": 1, "val_top1": top1, "val_loss": 1.0}]
        (run_dir / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in lines))
    return run_dir


def test_the_default_checkpoint_is_the_best_real_run(tmp_path):
    from ksmech.serve.models import default_checkpoint, list_checkpoints

    s = _settings(tmp_path)
    _run(tmp_path, "real_a", top1=0.55)
    _run(tmp_path, "real_b", top1=0.61)
    _run(tmp_path, "smoke_x", top1=0.9)
    _run(tmp_path, "limited", top1=0.95, limit=64)
    ckpts = list_checkpoints(s)
    assert {c.run for c in ckpts if c.smoke} == {"smoke_x", "limited"}
    ck = default_checkpoint(ckpts)
    assert (ck.run, ck.file) == ("real_b", "best")
    only_smoke = [c for c in ckpts if c.smoke]
    assert default_checkpoint(only_smoke).run == "limited"  # nothing else: a smoke run
    assert default_checkpoint([]) is None


def test_reason_texts_in_every_state(tmp_path, monkeypatch):
    from ksmech.serve import models
    from ksmech.serve.errors import Code, ServeError
    from ksmech.serve.network_predictor import NetworkPredictor, record_failure

    s = _settings(tmp_path)
    p = NetworkPredictor(s, "fp")
    info = p.info()
    assert not info.available and info.code == "CHECKPOINT_NOT_FOUND"
    assert "no checkpoint <source>/<run>/<file>.pt under" in info.reason
    dirty = _run(tmp_path, "dirty_run", dirty=True)
    info = p.info()
    assert info.code == "CHECKPOINT_REFUSED" and "every local checkpoint is refused" in info.reason
    assert "dirty_run" in info.reason
    _run(tmp_path, "good")
    info = p.info()
    assert info.available and info.reason is None and info.scores_are_probabilities
    assert info.version.startswith("network/pmechdb/good/best@") and info.checkpoint_hash
    for ck in models.list_checkpoints(s):
        if ck.usable:
            record_failure(s, ck, ServeError(Code.CHECKPOINT_LOAD_FAILED, "boom"))
    info = p.info()
    assert not info.available and info.code == "NETWORK_ADAPTER_FAILED"
    assert "the adapter failed on pmechdb/good/best.pt" in info.reason and "boom" in info.reason
    later = time.time() + 3600
    monkeypatch.setattr(time, "time", lambda: later)
    assert p.info().available  # retried after RETRY_AFTER_S
    monkeypatch.setattr(models, "missing_modules", lambda: ["torch"])
    info = p.info()
    assert info.code == "TRANSFORMER_NOT_AVAILABLE" and "not importable here: torch" in info.reason
    assert dirty.is_dir()


def test_a_named_checkpoint_resolves_and_a_refused_one_is_409(tmp_path):
    from ksmech.serve.errors import ServeError
    from ksmech.serve.network_predictor import NetworkPredictor

    s = _settings(tmp_path)
    _run(tmp_path, "good", top1=0.6)
    _run(tmp_path, "other", top1=0.4)
    _run(tmp_path, "dirty_run", dirty=True)
    p = NetworkPredictor(s, "fp")
    ref = p.make_ref(None, None)
    assert ref.checkpoint["run"] == "good" and ref.checkpoint["sha256"] == ref.checkpoint_hash
    assert ref.version.endswith("/fp") and ref.checkpoint["val_top1"] == 0.6
    assert p.make_ref(None, {"run": "other", "file": "last"}).checkpoint["file"] == "last"
    with pytest.raises(ServeError) as exc:
        p.make_ref(None, {"run": "dirty_run"})
    assert exc.value.code == "CHECKPOINT_REFUSED" and exc.value.status == 409
    with pytest.raises(ServeError) as exc:
        p.make_ref(None, {"run": "nope"})
    assert exc.value.code == "CHECKPOINT_NOT_FOUND" and exc.value.status == 404


# ---- a tiny random checkpoint on the synthetic chemistry


@pytest.fixture(scope="module")
def tiny_net(tmp_path_factory, synthetic_settings):  # noqa: F811
    """The network predictor over a fake run `fake/tiny` beside the synthetic tokens."""
    from dataclasses import replace

    from ksmech.serve.network_predictor import NetworkPredictor

    root = tmp_path_factory.mktemp("netmodels")
    write_fake_run(root / "data" / "models" / "fake" / "tiny", "dense")
    s = replace(synthetic_settings, shared_cache_root=root)
    return NetworkPredictor(s, "fp")


def test_tiny_model_steps_apply_with_the_model_logp(tiny_net, synthetic_runtime, sn2_state):  # noqa: F811
    import torch

    from ksmech.model.loss import _entry_sums, _real_atoms
    from ksmech.serve.analysis import pool_work
    from ksmech.serve.contracts import ExpandRequest
    from ksmech.serve.embedding import cached_model
    from ksmech.serve.network_predictor import model_sample
    from ksmech.serve.stepapply import apply_matrix

    work = pool_work(synthetic_runtime, sn2_state)
    ref = tiny_net.make_ref(None, None)
    req = ExpandRequest("network", 4, None, "all", 64, ref.checkpoint)
    skips: Counter = Counter()
    cands = tiny_net.bind(req).propose(synthetic_runtime, work, 4, "all", skips)
    assert cands, dict(skips)
    assert [c.logp for c in cands] == sorted((c.logp for c in cands), reverse=True)
    keys = set()
    for c in cands:
        R = c.matrix
        assert sum(d for _, d in R.lone) + 2 * sum(d for *_, d in R.bonds) == 0
        applied = apply_matrix(sn2_state, R)
        assert applied.ok, applied.issues
        keys.add(applied.state.pool_key)
    assert len(keys) == len(cands)  # merged by product
    assert all(k.startswith("DECODE_") for k in skips)

    # log P of the best step, from the model's own logits
    model, _ = cached_model(*_path_sig(tiny_net, ref))
    batch, h, p, lp = nd.encode_pool(model, model_sample(work))
    best = cands[0]
    inv = {m: g for g, m in work.rt.mol_atom.items()}
    a, b = sorted((inv[best.anchor[0]], inv[best.anchor[1]]))
    sign = nd.FORM if best.anchor[2] == "form" else nd.BREAK
    n = lp.shape[0]
    r_pairs = torch.zeros(1, n, n, dtype=torch.int8)
    r_atoms = torch.zeros(1, n, dtype=torch.int8)
    for i, j, d in best.matrix.bonds:
        gi, gj = inv[i], inv[j]
        r_pairs[0, gi, gj] = r_pairs[0, gj, gi] = d
    for i, d in best.matrix.lone:
        r_atoms[0, inv[i]] = d
    with torch.no_grad():
        rows = torch.tensor([[0, a, b, sign]])
        pair, atom = model.decoder.entry_logits(h, p, batch, rows)
        sums = _entry_sums(
            torch.log_softmax(pair.double(), -1), torch.log_softmax(atom.double(), -1),
            r_pairs, r_atoms, _real_atoms(batch), torch.tensor([a]), torch.tensor([b]),
        )  # fmt: skip
    assert best.logp == pytest.approx(float(lp[a, b, sign] + sums[0]), abs=1e-6)


def _path_sig(pred, ref):
    from ksmech.serve.models import find_checkpoint

    ck = find_checkpoint(pred.settings, ref.checkpoint["run"], ref.checkpoint["file"])
    return ck.path, ck.signature


def test_a_changed_checkpoint_file_is_refused(tiny_net, synthetic_runtime, sn2_state):  # noqa: F811
    from ksmech.serve.analysis import pool_work
    from ksmech.serve.contracts import ExpandRequest
    from ksmech.serve.errors import ServeError

    work = pool_work(synthetic_runtime, sn2_state)
    ck = dict(tiny_net.make_ref(None, None).checkpoint) | {"mtime_ns": 1}
    req = ExpandRequest("network", 2, None, "all", 64, ck)
    with pytest.raises(ServeError) as exc:
        tiny_net.bind(req).propose(synthetic_runtime, work, 2, "all", Counter())
    assert exc.value.code == "CHECKPOINT_CHANGED"


# ---- through the API (jobs inline)


@pytest.fixture(scope="module")
def net_app(tmp_path_factory, synthetic_root):  # noqa: F811
    from ksmech.serve.settings import load_settings
    from tests.test_serve_fixtures_api import api_client, copy_synthetic_root

    root = copy_synthetic_root(synthetic_root, tmp_path_factory.mktemp("netapi"))
    write_fake_run(root / "data" / "models" / "fake" / "tiny", "dense")
    write_fake_run(root / "data" / "models" / "fake" / "smoke_tiny", "dense")
    s = load_settings({}, data_root=root, chemistry="synthetic")
    with api_client(s) as client:
        yield client


SN2 = {"reactants_smiles": ["C[O-]", "CCl"], "predictor": "network", "beam": 2, "max_steps": 2}


def _job(client, r) -> dict:
    assert r.status_code == 202, r.text
    job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "done", job.get("error")
    return job


def test_the_network_predicts_through_the_api(net_app):
    from ksmech.serve import schemas as S
    from ksmech.serve.context import NOTE_NETWORK_STOP

    infos = {p["name"]: p for p in net_app.get("/api/predictors").json()}
    net = infos["network"]
    assert net["available"] and net["scores_are_probabilities"]
    assert net["version"].startswith("network/fake/tiny/best@")  # the smoke run is not default
    models = S.ModelsOut.model_validate(net_app.get("/api/models").json())
    assert {c.run: c.smoke for c in models.checkpoints} == {"smoke_tiny": True, "tiny": False}
    job = _job(net_app, net_app.post("/api/predict", json=SN2))
    out = S.NetworkOut.model_validate(job["result"])
    assert out.scores_are_probabilities
    ck = out.provenance.checkpoint
    assert (ck.source, ck.run, ck.file) == ("fake", "tiny", "best")
    assert ck.sha256 == out.provenance.checkpoint_hash and ck.val_top1 == 0.5
    assert out.request.checkpoint.model_dump() == {"run": "tiny", "file": "best", "source": "fake"}
    assert NOTE_NETWORK_STOP in out.provenance.notes
    assert job["provenance"]["checkpoint"]["run"] == "tiny"
    assert out.edges and all(e.logp is not None and e.predictor == "network" for e in out.edges)
    assert all(e.score == e.logp for e in out.edges)
    for p in out.pathways:
        total = sum(next(e.logp for e in out.edges if e.id == i) for i in p.edge_ids)
        assert p.probability == pytest.approx(np.exp(total), rel=1e-6)

    # expand a frontier state: the network's own checkpoint
    leaf = next(n for n in out.nodes if n.status == "frontier" or n.terminal_reason == "MAX_STEPS")
    r = net_app.post(f"/api/jobs/{job['job_id']}/expand", json={"node_id": leaf.id})
    done = _job(net_app, r)
    assert done["result"]["predictor"] == "network"
    grown = S.NetworkOut.model_validate(net_app.get(f"/api/jobs/{job['job_id']}").json()["result"])
    versions = {e.predictor_version for e in grown.edges}
    assert versions == {out.provenance.predictor_version}  # the same checkpoint


def test_a_named_smoke_checkpoint_is_recorded_with_a_note(net_app):
    body = SN2 | {"max_steps": 1, "checkpoint": {"run": "smoke_tiny"}}
    job = _job(net_app, net_app.post("/api/predict", json=body))
    prov = job["result"]["provenance"]
    assert prov["checkpoint"]["run"] == "smoke_tiny" and prov["checkpoint"]["smoke"]
    assert any("smoke run" in n for n in job["provenance"]["notes"])


def test_checkpoint_errors_are_immediate(net_app):
    r = net_app.post("/api/predict", json=SN2 | {"checkpoint": {"run": "nope"}})
    assert r.status_code == 404 and r.json()["error"]["code"] == "CHECKPOINT_NOT_FOUND"
    body = SN2 | {"predictor": "floor", "checkpoint": {"run": "tiny"}}
    r = net_app.post("/api/predict", json=body)
    assert r.status_code == 422 and r.json()["error"]["code"] == "REQUEST_INVALID"
