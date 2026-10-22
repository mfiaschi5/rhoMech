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


def test_ranking_refuses_stage_a_samples(mini_steps, token_loader):
    from ksmech.physics.ranking import physical_pair_z

    s = sample_for_step(mini_steps["sn2"], token_loader)
    with pytest.raises(ValueError, match="ksmech-arrows contacts"):
        physical_pair_z(s)


def test_a_missing_sidecar_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        contacts.load_contacts(tmp_path, "molecule")
