"""The legacy contact sidecar (`arrows.contacts`) over a stage-A sample store."""

import json

import numpy as np
import pytest

from ksmech.arrows import contacts
from ksmech.arrows.steps import build_all_samples, load_samples, sample_for_step, samples_path
from ksmech.data import provenance as prov
from ksmech.data.io import read_training_steps, source_dir, steps_path
from ksmech.physics.pairs import PAIR_FEATURES
from tests.test_arrows_steps import mini_processed, missing_iodide  # noqa: F401


def test_train_subsample_is_fixed_by_the_seed():
    ids = [f"s{i:03d}" for i in range(50)]
    a = contacts.train_subsample(ids, 10, seed=3)
    assert a == contacts.train_subsample(list(reversed(ids)), 10, seed=3) and a == sorted(a)
    assert len(set(a)) == 10 and a != contacts.train_subsample(ids, 10, seed=4)
    assert contacts.train_subsample(ids[:5], 10) == ids[:5]


def test_sidecar_matches_the_legacy_features(
    mini_processed,  # noqa: F811
    token_loader,
    pair_kernels,
):
    loader = missing_iodide(token_loader)
    build_all_samples("pmechdb", ["molecule"], mini_processed, loader)
    report = contacts.build_all_contacts(
        "pmechdb", pair_kernels, "molecule", mini_processed, loader, n_train=1
    )
    view_dir = source_dir("pmechdb", mini_processed)
    root = contacts.contacts_dir(view_dir, "molecule")
    sub = json.loads((root / "train_subsample.json").read_text())
    assert len(sub["ids"]) == report["n_train_subsample"] == 1
    prov.verify(root / "train_subsample.json", "train_subsample", contacts.CONTACTS_VERSION)
    legacy = contacts.load_contacts(view_dir, "molecule")
    steps = {s.step_id: s for s in read_training_steps(steps_path("pmechdb", mini_processed))}
    test = load_samples(samples_path("pmechdb", "molecule_test", mini_processed))
    assert test and all(s.step_id in legacy for s in test)
    for s in test:
        joined = contacts.with_legacy(s, legacy[s.step_id])
        direct = sample_for_step(steps[s.step_id], loader, pair_kernels)
        assert joined.pair_features == PAIR_FEATURES
        valid = joined.pair_valid
        assert np.array_equal(valid, direct.pair_valid)
        assert np.array_equal(joined.pair_defined[valid], direct.pair_defined[valid])
        assert np.allclose(joined.pair_z[valid], direct.pair_z[valid], rtol=1e-3, atol=1e-3)
        sites = joined.pair_contact_sites[valid]
        assert np.array_equal(sites, direct.pair_contact_sites[valid])
    _check_selective_loading(view_dir, mini_processed, test, legacy, sub["ids"])


def _check_selective_loading(view_dir, processed, test, legacy, sub_ids):
    """`ids` filters: only those contact entries and samples are read, in view order, and
    the shard iterator yields the same samples as the list loader."""
    from ksmech.arrows.steps import iter_samples, sample_ids

    one = test[-1].step_id
    assert list(contacts.load_contacts(view_dir, "molecule", ids=[one])) == [one]
    train_path = samples_path("pmechdb", "molecule_train", processed)
    train_ids = sample_ids(train_path)
    assert len(train_ids) > 1 and set(sub_ids) <= set(train_ids)
    picked = load_samples(train_path, ids=reversed(train_ids[:2]))
    assert [s.step_id for s in picked] == train_ids[:2]  # view order, not the ids' order
    streamed = {s.step_id for s in iter_samples(train_path)}
    assert streamed == set(train_ids)
    side = contacts.load_contacts(view_dir, "molecule", ids=train_ids)
    assert set(side) == set(sub_ids)  # the training part: only the subsample has contacts
    assert set(legacy) >= set(side)


def test_ranking_refuses_stage_a_samples(mini_steps, token_loader):
    from ksmech.physics.ranking import physical_pair_z

    s = sample_for_step(mini_steps["sn2"], token_loader)
    with pytest.raises(ValueError, match="ksmech-arrows contacts"):
        physical_pair_z(s)


def test_a_missing_sidecar_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        contacts.load_contacts(tmp_path, "molecule")


def _sidecar(root):
    """Index body (its header aside), shard arrays and the inputs each shard header records."""
    index = json.loads((root / "index.json").read_text())
    shards = {}
    for name in sorted(index["shards"]):
        with np.load(root / name, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != prov.HEADER_KEY}
        shards[name] = (arrays, prov.read_header(root / name).inputs)
    return {k: v for k, v in index.items() if k != "provenance"}, shards  # built_at differs


def test_the_parallel_sidecar_matches_the_sequential_one(
    mini_processed,  # noqa: F811
    mini_token_cache,
    pair_kernels,
    monkeypatch,
):
    """Two spawned workers on chunks of one step write the shards of one process: the same
    entries, arrays and recorded inputs (token files and contact blocks), the step without
    tokens skipped in both."""
    build_all_samples("pmechdb", ["molecule"], mini_processed, mini_token_cache)
    view_dir = source_dir("pmechdb", mini_processed)
    root = contacts.contacts_dir(view_dir, "molecule")
    args = ("pmechdb", pair_kernels, "molecule", mini_processed, mini_token_cache)
    one = contacts.build_all_contacts(*args, n_train=10)  # the whole training part
    sequential = _sidecar(root)
    monkeypatch.setattr(contacts, "CHUNK", 1)
    two = contacts.build_all_contacts(*args, n_train=10, n_workers=2)
    parallel = _sidecar(root)
    assert one == two and one["n_entries"] >= 2 and one["skipped"]
    assert sequential[0] == parallel[0]
    assert sequential[1].keys() == parallel[1].keys()
    for name, (arrays, inputs) in sequential[1].items():
        other, other_inputs = parallel[1][name]
        assert inputs == other_inputs and arrays.keys() == other.keys()
        assert all(np.array_equal(arrays[k], other[k]) for k in arrays)
