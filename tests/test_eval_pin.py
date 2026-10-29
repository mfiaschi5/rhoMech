"""A pinned cohort with per-column digests (`eval.pin`, open-shell plan P0b step 1)."""

import json
from dataclasses import replace

import numpy as np
import pytest

from ksmech.arrows import store
from ksmech.arrows.steps import MechanismSample, build_all_samples, load_samples, samples_path
from ksmech.data import provenance as prov
from ksmech.eval import cohorts as co
from ksmech.eval import pin
from ksmech.eval.cli import main as eval_main
from tests.test_arrows_steps import mini_processed, missing_iodide  # noqa: F401 (fixture)


@pytest.fixture
def mini_store(mini_processed, token_loader):  # noqa: F811
    build_all_samples("pmechdb", ["molecule"], mini_processed, missing_iodide(token_loader))
    return mini_processed


def _samples(processed) -> list[MechanismSample]:
    return [
        s
        for part in ("train", "val", "test")
        for s in load_samples(samples_path("pmechdb", f"molecule_{part}", processed))
    ]


def test_every_named_column_has_a_digest(mini_steps, token_loader):
    from ksmech.arrows.steps import sample_for_step

    s = sample_for_step(mini_steps["sn2"], token_loader)
    d = pin.column_digests(s)
    assert {"ids", "atom_categorical", "pair_valid", "edit_label", "labels", "labels_all"} <= set(d)
    assert {f"atom.{n}" for n in s.atom_columns} <= set(d)
    assert {f"orbital.{n}" for n in s.orbital_columns} | {"orbital.spectra"} <= set(d)
    assert {f"pair.{n}" for n in s.pair_features} <= set(d)
    assert all(len(v) == pin.DIGEST_CHARS for v in d.values())


def test_a_digest_moves_with_its_column_only(mini_steps, token_loader):
    from ksmech.arrows.steps import sample_for_step

    s = sample_for_step(mini_steps["sn2"], token_loader)
    before = pin.column_digests(s)
    z = s.pair_z.copy()
    k = s.pair_features.index("log_abs_g")
    z[..., k] = np.where(s.pair_defined[..., k], z[..., k] + 1.0, z[..., k])
    after = pin.column_digests(replace(s, pair_z=z))
    assert {c for c in before if before[c] != after[c]} == {"pair.log_abs_g"}
    # a value where the column is undefined is not content
    hidden = s.pair_z.copy()
    hidden[..., k] = np.where(s.pair_defined[..., k], hidden[..., k], 123.0)
    assert pin.column_digests(replace(s, pair_z=hidden)) == before


def test_the_store_round_trip_keeps_every_digest(mini_steps, token_loader):
    from ksmech.arrows.steps import sample_for_step

    s = sample_for_step(mini_steps["sn2"], token_loader)
    back = store.unpack(*store.pack(s), MechanismSample)
    assert pin.column_digests(back) == pin.column_digests(
        store.unpack(*store.pack(back), MechanismSample)
    )


def test_pin_writes_the_cohort_its_gbdt_subsets_and_the_baseline(mini_store):
    out = pin.pin_cohort("pmechdb", "closed_base", processed_dir=mini_store)
    samples = _samples(mini_store)
    ids = {s.step_id for s in samples}
    assert co.load_cohort("pmechdb", "closed_base", mini_store) == ids
    with_labels = {s.step_id for s in samples if s.labels}
    assert co.load_cohort("pmechdb", "closed_base_gbdt_valid", mini_store) == with_labels
    assert sum(out[p]["samples"] for p in ("train", "val", "test")) == len(ids)
    base = pin.load_baseline("pmechdb", "closed_base", mini_store)
    assert base == pin.digests_of(samples)
    assert prov.read_header(pin.baseline_path("pmechdb", "closed_base", mini_store)).inputs == {}
    with pytest.raises(FileExistsError):
        pin.pin_cohort("pmechdb", "closed_base", processed_dir=mini_store)


def test_the_baseline_is_checked_by_its_content_hash(mini_store):
    pin.pin_cohort("pmechdb", "closed_base", processed_dir=mini_store)
    path = pin.baseline_path("pmechdb", "closed_base", mini_store)
    body = json.loads(path.read_text())
    first = next(iter(body["digests"]))
    body["digests"][first][0] = "0" * pin.DIGEST_CHARS
    path.write_text(json.dumps(body))
    with pytest.raises(prov.StaleArtifact, match="content hash"):
        pin.load_baseline("pmechdb", "closed_base", mini_store)


def test_compare_reports_columns_and_samples_by_name():
    base = {"a": {"x": "1", "y": "2"}, "b": {"x": "1", "y": "2"}, "gone": {"x": "0"}}
    now = {"a": {"x": "1", "y": "3", "z": "9"}, "b": {"x": "1", "y": "2"}, "new": {"x": "0"}}
    got = pin.compare_digests(base, now)
    assert got["n_shared"] == 2 and got["differ"] == {"y": ["a"]}
    assert got["columns_only_in_current"] == ["z"] and got["columns_only_in_baseline"] == []
    assert got["samples_only_in_baseline"] == ["gone"]
    assert got["samples_only_in_current"] == ["new"]


def test_the_pin_command(mini_store):
    argv = ["pin", "--source", "pmechdb", "--name", "closed_base"]
    assert eval_main([*argv, "--processed-dir", str(mini_store)]) == 0
    assert pin.baseline_path("pmechdb", "closed_base", mini_store).exists()
    assert co.cohort_path("pmechdb", "closed_base_gbdt_all", mini_store).exists()


@pytest.mark.realdata
def test_closed_base_reloads_and_its_digests_reproduce_on_a_rebuild():
    """closed_base (no file inputs) loads after any re-ingest and holds the molecule split's
    samples; 30 of its samples rebuilt from their tokens, through the store's packing, give the
    pinned digests on every shared column."""
    from ksmech.arrows.reactants import TokenCache
    from ksmech.arrows.steps import sample_for_step, sample_ids
    from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_training_steps, steps_path

    if not pin.baseline_path("pmechdb", "closed_base").exists():
        pytest.skip("closed_base is not pinned (ksmech-eval pin --source pmechdb ...)")
    ids = co.load_cohort("pmechdb", "closed_base")
    views = [samples_path("pmechdb", f"molecule_{p}") for p in ("train", "val", "test")]
    assert ids == {i for v in views for i in sample_ids(v)}
    base = pin.load_baseline("pmechdb", "closed_base")
    assert set(base) == ids
    picked = sorted(ids)[:: max(1, len(ids) // 30)][:30]
    steps = {
        s.step_id: s for s in read_training_steps(steps_path("pmechdb", DEFAULT_PROCESSED_DIR))
    }
    loader = TokenCache()
    rebuilt = []
    for i in picked:
        s = sample_for_step(steps[i], loader)
        rebuilt.append(store.unpack(*store.pack(s), MechanismSample))
    got = pin.compare_digests({i: base[i] for i in picked}, pin.digests_of(rebuilt))
    assert got["n_shared"] == len(picked) and not got["differ"], got["differ"]
    assert not got["columns_only_in_baseline"] and not got["columns_only_in_current"]
