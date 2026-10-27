"""The lab's ML routes (serve.ml): the training runs under data/models (run.json, metrics per
epoch, checkpoints, reports beside them), one run's detail, the physics / eval reports under
data/processed, the size limit and unreadable files, RUN_NOT_FOUND, and `/api/models` kept in
the ml router (and open in the public profile)."""

from __future__ import annotations

import json

import pytest

from ksmech.serve.core import schemas as S
from ksmech.serve.core.settings import load_settings
from tests.test_serve_fixtures_api import api_client, no_xtb  # noqa: F401 (fixtures)

pytestmark = pytest.mark.serve

EPOCHS = [
    {"epoch": 0, "val_top1": 0.3, "val_loss": 8.0, "skipped": {"train": {"TOO_LARGE": 3}}},
    {"epoch": 1, "val_top1": 0.5, "val_loss": 6.0},
    {"epoch": 2, "val_top1": 0.4, "val_loss": 5.0},
]
RUN_JSON = {
    "provenance": {
        "kind": "train_run",
        "code_git_hash": "7b090cb3aa23043a37ea83320b1e4639beb7feb9",
        "code_dirty": False,
        "built_at": "2026-10-26T00:21:49+00:00",
        "config": {
            "model": {"encoder": "flex", "d": 256, "layers": 8, "heads": 8, "d_pair": 64},
            "train": {"limit": 0, "split": "molecule", "epochs": 3},
        },
    },
    "device": "cuda",
}
DECODE = {"provenance": {"kind": "decode"}, "top1": 0.63, "n": 1239}
FLOOR = {"provenance": {"kind": "anchor_floor"}, "val": {"top1": 0.16}, "test": {"top1": 0.18}}
GBDT = {"split": "molecule", "features": "geomfree", "results": {"val": {"top1": 0.2}}}


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def ml_root(tmp_path):
    """A data root with two runs (one with checkpoints, a report and a bad metrics line, one
    trained elsewhere: metrics only), three processed reports and one unrelated report."""
    models = tmp_path / "data" / "models" / "pmechdb"
    run = models / "flex_first"
    _write(run / "run.json", json.dumps(RUN_JSON))
    lines = [json.dumps(e) for e in EPOCHS]
    _write(run / "metrics.jsonl", "\n".join([*lines[:2], "{not json", lines[2]]) + "\n")
    for name in ("best.pt", "last.pt"):
        (run / name).write_bytes(b"not a real checkpoint")
    _write(run / "decode_report_molecule_val.json", json.dumps(DECODE))
    _write(run / "train.log", "step 1 / 10\n")
    _write(models / "dense_plain" / "metrics.jsonl", json.dumps(EPOCHS[0]) + "\n")
    _write(models / "not_a_run" / "notes.txt", "x")
    processed = tmp_path / "data" / "processed" / "pmechdb"
    _write(processed / "anchor_floor_report_molecule.json", json.dumps(FLOOR))
    _write(processed / "gbdt_lgbm_report_molecule_geomfree.json", json.dumps(GBDT))
    _write(processed / "gbdt_lgbm_report_scaffold_geomfree.json", "{truncated")
    _write(processed / "ks_rank_report_table.json", "{}")  # not a listed report
    return tmp_path


def _client(root, profile: str = "lab"):
    s = load_settings({}, data_root=root, chemistry="synthetic", profile=profile)
    return api_client(s)


def test_runs(ml_root):
    with _client(ml_root) as client:
        out = S.MlRunsOut.model_validate(client.get("/api/ml/runs").json())
    assert out.roots == [str(ml_root.resolve() / "data" / "models")]
    runs = {r.run: r for r in out.runs}
    assert set(runs) == {"flex_first", "dense_plain"}
    r = runs["flex_first"]
    assert (r.source, r.encoder, r.clean, r.smoke, r.n_epochs) == (
        "pmechdb",
        "flex",
        True,
        False,
        3,
    )
    assert r.model == {"d": 256, "d_pair": 64, "layers": 8, "heads": 8}
    assert r.config == RUN_JSON["provenance"]["config"]
    assert r.best == EPOCHS[1] and r.last == EPOCHS[2]  # passed through as written
    assert r.checkpoints == ["best", "last"] and r.logs == ["train.log"]
    assert r.reports == ["decode_report_molecule_val.json"]
    d = runs["dense_plain"]
    assert d.checkpoints == [] and d.clean is None and d.config is None and d.n_epochs == 1


def test_run_detail(ml_root):
    with _client(ml_root) as client:
        body = client.get("/api/ml/runs/pmechdb/flex_first").json()
        missing = client.get("/api/ml/runs/pmechdb/nope")
        malformed = client.get("/api/ml/runs/pmechdb/..hidden")
    out = S.MlRunDetailOut.model_validate(body)
    assert out.run_json == RUN_JSON and out.epochs == EPOCHS and out.bad_lines == 1
    (rep,) = out.reports
    assert (rep.name, rep.kind, rep.report, rep.reason) == (
        "decode_report_molecule_val.json",
        "decode",
        DECODE,
        None,
    )
    for r in (missing, malformed):
        assert r.status_code == 404 and r.json()["error"]["code"] == "RUN_NOT_FOUND"


def test_reports(ml_root, monkeypatch):
    with _client(ml_root) as client:
        out = S.MlReportsOut.model_validate(client.get("/api/ml/reports").json())
        from ksmech.serve.ml import runs

        monkeypatch.setattr(runs, "MAX_REPORT_BYTES", 4)
        small = S.MlReportsOut.model_validate(client.get("/api/ml/reports").json())
    by = {r.name: r for r in out.reports}
    assert set(by) == {
        "anchor_floor_report_molecule.json",
        "gbdt_lgbm_report_molecule_geomfree.json",
        "gbdt_lgbm_report_scaffold_geomfree.json",
    }
    assert by["anchor_floor_report_molecule.json"].report == FLOOR
    assert by["gbdt_lgbm_report_molecule_geomfree.json"].kind == "gbdt_lgbm"
    assert by["gbdt_lgbm_report_molecule_geomfree.json"].report == GBDT
    bad = by["gbdt_lgbm_report_scaffold_geomfree.json"]
    assert bad.report is None and bad.reason.startswith("REPORT_UNREADABLE")
    assert all(r.report is None and r.reason.startswith("REPORT_TOO_LARGE") for r in small.reports)


def test_models_stays_open_in_public(ml_root):
    with _client(ml_root, "public") as client:
        models = S.ModelsOut.model_validate(client.get("/api/models").json())
        assert client.get("/api/ml/runs").status_code == 503
    assert {(c.run, c.file) for c in models.checkpoints} == {
        ("flex_first", "best"),
        ("flex_first", "last"),
    }
    assert models.embedding.enabled is False  # not in the public profile
