"""The pretraining samples (`pretrain.samples`) on fixture molecules: the sample of a
one-molecule pool with its donor-acceptor target, the reason codes, the store round trip and
the seeded species subset."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ksmech.arrows import store
from ksmech.physics.donor_acceptor import OPEN_SHELL, donor_acceptor
from ksmech.pretrain import samples as ps
from tests.fixtures.molecules import MOLECULES
from tests.fixtures.molecules import OPEN_SHELL as OPEN_SHELL_FIXTURES

KEYS = [MOLECULES[n] for n in ("acetone", "methyl_acetate", "acrolein", "chloromethane")]  # as
#   the fixture `pretrain_cache` writes them
METHYL = OPEN_SHELL_FIXTURES["methyl"][0]


def test_sample_carries_the_target_on_its_cells(pretrain_cache, token_loader):
    for key in KEYS:
        s = ps.pretrain_sample(key, pretrain_cache)
        assert isinstance(s, ps.PretrainSample), s
        assert s.pair_target.shape == s.pair_valid.shape == s.pair_z.shape[:2]
        expected = donor_acceptor(token_loader(key)).e2
        np.testing.assert_allclose(s.pair_target, expected.astype(np.float32), rtol=1e-6)
        assert s.edit_label is None and not s.labels and len(s.bonds) > 0
        assert any(name.startswith("file:") for name in s.inputs), "token files recorded"


def test_reason_codes(pretrain_cache):
    assert ps.pretrain_sample(METHYL, pretrain_cache) == OPEN_SHELL
    assert ps.pretrain_sample("CCOC", pretrain_cache) == "MISSING_TOKENS"
    assert ps.pretrain_sample("CC(C)=O.CCl", pretrain_cache) == ps.MULTI_FRAGMENT


def test_pack_round_trip_keeps_the_target(pretrain_cache):
    s = ps.pretrain_sample(KEYS[1], pretrain_cache)
    arrays, meta = store.pack(s)
    back = store.unpack(arrays, json.loads(json.dumps(meta)), ps.PretrainSample)
    np.testing.assert_array_equal(back.pair_target, s.pair_target)
    np.testing.assert_array_equal(back.pair_z, s.pair_z.astype(np.float16))  # stored in float16


def test_store_and_views(pretrain_cache, tmp_path):
    species = tmp_path / "species.json"
    species.write_text(json.dumps({"species": [*KEYS, METHYL, "CCOC"]}))
    report = ps.build_store(species, tmp_path / "out", pretrain_cache, n_workers=1)
    assert report["n_samples"] == len(KEYS)
    assert report["skipped"] == {OPEN_SHELL: 1, "MISSING_TOKENS": 1}
    loaded = [s for p in ps.PARTS for s in ps.load_part(tmp_path / "out", p)]
    assert sorted(s.step_id for s in loaded) == sorted(ps.sample_id(k) for k in KEYS)
    for s in loaded:
        assert isinstance(s, ps.PretrainSample) and s.pair_target.size > 0
        assert ps.part_of(s.fragment_keys[0]) in ps.PARTS


def test_species_subset_is_seeded(tmp_path):
    manifest = tmp_path / "qm_manifest.parquet"
    smiles = [f"C{'C' * k}O" for k in range(40)]
    status = ["ok"] * 38 + ["failed"] * 2
    pq.write_table(pa.table({"smiles": smiles, "status": status}), manifest)
    a, b = ps.select_species(manifest, 10, 0), ps.select_species(manifest, 10, 0)
    assert a == b and len(set(a)) == 10 and not set(a) & set(smiles[38:])
    assert ps.select_species(manifest, 10, 1) != a
    with pytest.raises(ValueError):
        ps.select_species(manifest, 39, 0)
    path = ps.write_species_file(tmp_path / "species.json", manifest, 10, 0)
    assert json.loads(path.read_text())["species"] == a


def test_parts_by_hash():
    keys = [f"C{'C' * k}N" for k in range(400)]
    parts = [ps.part_of(k) for k in keys]
    assert parts == [ps.part_of(k) for k in keys]
    assert 0.01 < parts.count("val") / len(keys) < 0.12
