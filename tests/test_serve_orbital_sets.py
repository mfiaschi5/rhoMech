"""serve/workbench/orbital_sets.py: every set and start is orthonormal (C^T S C = I) and spans the
canonical occupied space; ibo/cholesky is the model token file; canonical rows carry the
frontier labels and degenerate levels; LMO rows carry exactly the model-input descriptors; the
workbench set files are cached with provenance; bad names are REQUEST_INVALID."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ksmech.serve.core import schemas as S
from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.workbench import orbital_sets as osets
from tests.test_serve_fixtures_wb import (  # noqa: F401 (fixtures)
    canonical_key,
    no_xtb,
    synthetic_root,
    wb_runtime,
    wb_settings,
)

pytestmark = pytest.mark.serve

KEYS = ("CCO", "C=CC=O")


def _sets() -> list[tuple[str, str | None]]:
    methods, starts = osets.localize_names()
    return [(osets.CANONICAL, None)] + [(m, s) for m in methods for s in starts]


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize(("set_", "start"), _sets())
def test_every_set_is_orthonormal_and_keeps_the_occupied_space(wb_runtime, key, set_, start):  # noqa: F811
    rec, _, _ = osets.load_qm(wb_runtime, key)
    occ = np.asarray(rec.occupations) >= 0.5
    if set_ == osets.CANONICAL:
        C, kinds = rec.C, np.where(occ, "occupied", "virtual")
    else:
        t, _, _ = osets.set_tokens(wb_runtime, key, set_, start)
        C, kinds = t.C_loc, np.array([str(o.kind) for o in t.orbitals])
    assert C.shape == rec.C.shape
    np.testing.assert_allclose(C.T @ rec.S @ C, np.eye(C.shape[1]), atol=1e-9)
    D_set = C[:, kinds == "occupied"] @ C[:, kinds == "occupied"].T
    np.testing.assert_allclose(D_set, rec.C[:, occ] @ rec.C[:, occ].T, atol=1e-9)
    body = osets.orbital_set_body(wb_runtime, key, set_, start)
    st = body["stats"]
    assert st["orthonormality_error"] < 1e-9 and st["occupied_projector_error"] < 1e-9
    assert st["n_occupied"] == int(occ.sum()) and len(body["orbitals"]) == rec.C.shape[1]
    S.OrbitalSetOut(molecule_id="m" + "0" * 20, **json.loads(json.dumps(body)))


def test_ibo_cholesky_is_the_model_token_file(wb_runtime):  # noqa: F811
    from ksmech.orbitals.localize import PMConfig
    from ksmech.orbitals.tokens import build_molecule_tokens

    assert osets.model_set() == ("ibo", "cholesky") and osets.is_model_set("ibo", "cholesky")
    key = "CCO"
    t, source, digest = osets.set_tokens(wb_runtime, key, "ibo", "cholesky")
    assert source == "token_file" and digest == wb_runtime.loader.digest(key)
    assert t is wb_runtime.loader(key)
    rec, _, _ = osets.load_qm(wb_runtime, key)
    rebuilt = build_molecule_tokens(rec, pm_config=PMConfig(method="ibo", start="cholesky"))
    np.testing.assert_allclose(rebuilt.C_loc, t.C_loc, atol=1e-10)
    assert [str(o.kind) for o in rebuilt.orbitals] == [str(o.kind) for o in t.orbitals]
    body = osets.orbital_set_body(wb_runtime, key, "ibo", "cholesky")
    assert body["is_model_set"] and body["source"] == "token_file"
    assert body["provenance"]["token_digest"] == digest


def test_other_sets_differ_and_are_cached_with_provenance(wb_runtime, wb_settings):  # noqa: F811
    from ksmech.data import provenance as prov

    key = "C=CC=O"
    ibo = osets.set_tokens(wb_runtime, key, "ibo", "cholesky")[0]
    pm, source, _ = osets.set_tokens(wb_runtime, key, "pm_mulliken", "cholesky")
    assert source == "workbench" and not np.allclose(np.abs(pm.C_loc), np.abs(ibo.C_loc))
    rec, qm_path, qm_digest = osets.load_qm(wb_runtime, key)
    path = osets.set_tokens_path(
        wb_settings, key, wb_runtime.featurizer.method, qm_digest, "pm_mulliken", "cholesky"
    )
    assert path.is_file() and path.parent == wb_settings.workbench_orbitals
    header = prov.read_header(path)
    assert header.kind == "tokens" and header.config["localization"] == "pm_mulliken"
    assert prov.file_input_name(qm_path) in header.inputs
    osets._TOKENS.clear()  # the next call loads the file instead of rebuilding
    mtime = path.stat().st_mtime_ns
    again = osets.set_tokens(wb_runtime, key, "pm_mulliken", "cholesky")[0]
    np.testing.assert_array_equal(again.C_loc, pm.C_loc)
    assert path.stat().st_mtime_ns == mtime


def test_canonical_rows_labels_and_degenerate_levels(wb_runtime):  # noqa: F811
    from ksmech.qm.fukui import homo_index

    key = canonical_key("c1ccccc1")
    rec, _, _ = osets.load_qm(wb_runtime, key)
    body = osets.orbital_set_body(wb_runtime, key, osets.CANONICAL, None)
    rows = body["orbitals"]
    homo = homo_index(np.asarray(rec.occupations))
    assert body["homo_index"] == homo and body["start"] is None
    assert rows[homo]["label"] == "HOMO" and rows[homo + 1]["label"] == "LUMO"
    assert rows[homo - 2]["label"] == "HOMO-2" and rows[homo + 2]["label"] == "LUMO+1"
    homo_block = [r for r in rows if r["frontier"] == "homo"]
    assert len(homo_block) == 2  # benzene e1g
    assert len({r["level"] for r in homo_block}) == 1
    assert all(r["model_input"] is None and r["diagnostic"] is None for r in rows)
    weights = sum(c["weight"] for c in rows[homo]["centres"])
    assert 0.9 < weights <= 1.0 + 1e-9 and {c["element"] for c in rows[homo]["centres"]} == {"C"}


def test_lmo_rows_carry_the_model_input(wb_runtime):  # noqa: F811
    from ksmech.serve.chem.token_rows import orbital_model_input

    key = "CCO"
    t = wb_runtime.loader(key)
    rows = osets.orbital_set_body(wb_runtime, key, "ibo", "cholesky")["orbitals"]
    rec, _, _ = osets.load_qm(wb_runtime, key)
    occupied = set(np.flatnonzero(np.asarray(rec.occupations) >= 0.5).tolist())
    for row, o in zip(rows, t.orbitals, strict=True):
        assert row["model_input"] == orbital_model_input(o).model_dump(mode="json")
        assert row["diagnostic"]["diagnostic"] is True and row["diagnostic"]["type"]
        assert row["label"] is None and row["level"] is None
        assert abs(sum(w for _, _, w in row["spectral"]) - 1.0) < 1e-6
        in_occ = {k in occupied for k, _, _ in row["spectral"]}
        assert in_occ == {row["kind"] == "occupied"}  # an LMO mixes MOs of its own block
        assert row["centres"] and len(row["centroid"]) == 3


def test_task_writes_the_rows_once(wb_runtime, wb_settings):  # noqa: F811
    payload = {"key": "CCO", "set": "pm_mulliken", "start": "canonical"}
    first = Path(osets.t_wb_orbital_set(wb_runtime, payload)["path"])
    assert first.parent == wb_settings.workbench_orbitals and first.suffix == ".json"
    mtime = first.stat().st_mtime_ns
    assert Path(osets.t_wb_orbital_set(wb_runtime, payload)["path"]) == first
    assert first.stat().st_mtime_ns == mtime
    body = json.loads(first.read_text())
    assert body["set"] == "pm_mulliken" and body["start"] == "canonical"
    assert body["sets"] == ["canonical", "ibo", "pm_mulliken"] and not body["is_model_set"]


@pytest.mark.parametrize(("set_", "start"), [("boys", None), ("ibo", "random")])
def test_unknown_set_or_start(set_, start):
    with pytest.raises(ServeError) as info:
        osets.check_set(set_, start)
    assert info.value.code == Code.REQUEST_INVALID


def test_defaults():
    assert osets.check_set(None, None) == ("canonical", None)
    assert osets.check_set("canonical", "cholesky") == ("canonical", None)
    assert osets.check_set("pm_mulliken", None) == ("pm_mulliken", "cholesky")
