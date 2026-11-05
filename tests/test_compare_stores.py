"""scripts/compare_stores.py (open-shell plan P8 step 7) on two tiny stores that differ in one
named column."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np

from ksmech.data import provenance as prov

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "compare_stores", ROOT / "scripts" / "compare_stores.py"
)
cmp = importlib.util.module_from_spec(_spec)
sys.modules["compare_stores"] = cmp
_spec.loader.exec_module(cmp)


def _store(tmp: Path, samples, monkeypatch) -> Path:
    from ksmech.arrows import store
    from ksmech.arrows.steps import SAMPLE_VERSION

    monkeypatch.setattr(prov, "code_state", lambda: ("a" * 40, False, ""))
    tmp.mkdir()
    parquet = tmp / "steps.parquet"
    parquet.write_bytes(b"x")
    store.write_store(tmp, {s.step_id: s for s in samples}, {}, parquet, SAMPLE_VERSION)
    return store.store_dir(tmp)


def test_two_stores_differ_in_exactly_one_named_column(
    mini_steps, token_loader, monkeypatch, tmp_path
):
    from ksmech.arrows.steps import sample_for_step

    s = sample_for_step(mini_steps["e2"], token_loader)
    t = sample_for_step(mini_steps[next(k for k in mini_steps if k != "e2")], token_loader)
    feats = s.atom_features.copy()
    feats[0, 0] += 1.0
    changed = dataclasses.replace(s, atom_features=feats)
    a = _store(tmp_path / "a", [s, t], monkeypatch)
    b = _store(tmp_path / "b", [changed], monkeypatch)
    va, da = cmp.store_digests(a)
    vb, db = cmp.store_digests(b)
    report = cmp.compare(da, db)
    assert va == vb and report["n_both"] == 1
    assert report["only_a"] == {"n": 1, "examples": [t.step_id]}
    assert report["columns_that_differ"] == [f"atom.{s.atom_columns[0]}"]
    assert report["columns"][f"atom.{s.atom_columns[0]}"]["examples"] == [s.step_id]
    assert not report["columns_only_a"] and not report["columns_only_b"]
    assert np.array_equal(s.atom_features[1:], changed.atom_features[1:])
