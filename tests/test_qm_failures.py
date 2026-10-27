"""Failure memo: tombstones, the covering rule, and the batch answering a known failure from it
instead of recomputing (across sources, never on an asked-for retry)."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rich.console import Console

from ksmech.data import provenance as prov
from ksmech.qm import batch, failures
from ksmech.qm.records import QMError, ReasonCode, cache_path
from ksmech.qm.xtb_calc import QMConfig

BAD = "COC=[NH+][CuH2-]"  # 63 electrons: SPIN_PARITY, raised before any SCF


def _config(tmp_path, **kw):
    return QMConfig(cache_dir=tmp_path / "cache", **kw)


def _steps(processed, source, smiles):
    path = processed / source / "steps.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"reactants_smiles": [smiles], "products_smiles": [smiles]}), path)


def _run(processed, cfg, source, **kw):
    return batch.featurize_dataset(
        source, processed_dir=processed, config=cfg, console=Console(quiet=True), **kw
    )


def _forbid_compute(monkeypatch):
    def forbidden(task):
        raise AssertionError(f"recomputed a remembered failure: {task.smiles}")

    monkeypatch.setattr(batch, "_run_molecule", forbidden)


def test_covers_same_settings_and_smaller_budgets_only(tmp_path):
    failed = failures.config_params(_config(tmp_path, timeout_s=600, scf_maxiter=250))
    same = failures.config_params(_config(tmp_path / "elsewhere", timeout_s=600))
    assert "cache_dir" not in failed and failures.covers(failed, same)
    assert failures.covers(failed, failures.config_params(_config(tmp_path, timeout_s=300)))
    assert not failures.covers(failed, failures.config_params(_config(tmp_path, timeout_s=900)))
    assert not failures.covers(failed, failures.config_params(_config(tmp_path, scf_maxiter=500)))
    assert not failures.covers(failed, failures.config_params(_config(tmp_path, fmax=1e-3)))
    assert not failures.covers(failed, {**same, "new_setting": 1})


def test_remember_lookup_forget(tmp_path):
    cfg = _config(tmp_path)
    path = failures.remember(cfg, "CCO", 0, "SCF_NOT_CONVERGED", "250 cycles", 12.5)
    assert path == failures.tombstone_path(cfg.cache_dir, "CCO", 0, cfg.method)
    body = json.loads(path.read_text())
    assert body["reason"] == "SCF_NOT_CONVERGED" and body["params"]["scf_maxiter"] == 250
    header = prov.read_header(path)
    assert header is not None and header.kind == failures.KIND
    hit = failures.lookup(cfg.cache_dir, "CCO", 0, cfg)
    assert hit is not None and hit.detail == "250 cycles" and hit.seconds == 12.5
    assert failures.known_detail(hit).startswith(failures.KNOWN_PREFIX)
    assert failures.lookup(cfg.cache_dir, "CCO", 1, cfg) is None
    assert failures.lookup(cfg.cache_dir, "CCO", 0, _config(tmp_path, scf_maxiter=500)) is None
    failures.forget(cfg, "CCO", 0)
    assert not path.exists() and failures.lookup(cfg.cache_dir, "CCO", 0, cfg) is None


def test_only_recurring_reasons_are_remembered(tmp_path):
    cfg = _config(tmp_path)
    for reason in ("WORKER_CRASH", "UNEXPECTED", "XTB_NOT_FOUND"):
        assert failures.remember(cfg, "CCO", 0, reason, "x", 1.0) is None
    assert not cfg.cache_dir.exists()


def test_a_record_wins_over_a_tombstone(tmp_path):
    cfg = _config(tmp_path)
    failures.remember(cfg, "CCO", 0, "TIMEOUT", "> 900s", 900.0)
    record = cache_path(cfg.cache_dir, "CCO", 0, cfg.method)
    record.write_bytes(b"a record")
    assert failures.lookup(cfg.cache_dir, "CCO", 0, cfg) is None


def test_a_covering_tombstone_is_kept(tmp_path):
    big = _config(tmp_path, timeout_s=1800)
    failures.remember(big, "CCO", 0, "TIMEOUT", "> 1800s", 1800.0)
    failures.remember(_config(tmp_path, timeout_s=60), "CCO", 0, "TIMEOUT", "> 60s", 60.0)
    kept = failures.read(failures.tombstone_path(big.cache_dir, "CCO", 0, big.method))
    assert kept is not None and kept.params["timeout_s"] == 1800


def test_an_unreadable_tombstone_is_no_hit(tmp_path):
    cfg = _config(tmp_path)
    path = failures.tombstone_path(cfg.cache_dir, "CCO", 0, cfg.method)
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert failures.lookup(cfg.cache_dir, "CCO", 0, cfg) is None


def test_process_molecule_answers_from_the_memo(tmp_path, monkeypatch):
    cfg = _config(tmp_path, timeout_s=60)
    failures.remember(cfg, "CCO", 0, "TIMEOUT", "> 60s", 60.0)
    _forbid_compute(monkeypatch)
    (row,) = batch.process_molecule(batch.MolTask("CCO", 1, cfg))
    assert row["status"] == "failed" and row["reason"] == "TIMEOUT" and row["seconds"] == 0.0
    assert row["detail"].startswith(failures.KNOWN_PREFIX)


def test_a_larger_budget_or_a_retry_recomputes(tmp_path, monkeypatch):
    cfg = _config(tmp_path, timeout_s=60)
    failures.remember(cfg, "CCO", 0, "TIMEOUT", "> 60s", 60.0)
    calls = []

    def scf_fails(task):
        calls.append(task.config.timeout_s)
        raise QMError(ReasonCode.SCF_NOT_CONVERGED, "250 cycles")

    monkeypatch.setattr(batch, "_compute", scf_fails)
    batch.process_molecule(batch.MolTask("CCO", 1, _config(tmp_path, timeout_s=120)))
    batch.process_molecule(batch.MolTask("CCO", 1, cfg, memo=False))
    assert calls == [120, 60]
    hit = failures.read(failures.tombstone_path(cfg.cache_dir, "CCO", 0, cfg.method))
    assert hit is not None and hit.reason == "SCF_NOT_CONVERGED"


def test_a_success_removes_the_tombstone(tmp_path, monkeypatch):
    cfg = _config(tmp_path, timeout_s=60)
    path = failures.remember(cfg, "CCO", 0, "TIMEOUT", "> 60s", 60.0)

    def ok(task):
        return [batch._row("CCO", 0, status="ok", cache_path="x.npz")]

    monkeypatch.setattr(batch, "_compute", ok)
    batch.process_molecule(batch.MolTask("CCO", 1, _config(tmp_path, timeout_s=120)))
    assert not path.exists()


def test_multi_conformer_tasks_skip_the_memo(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    failures.remember(cfg, "CCO", 0, "SCF_NOT_CONVERGED", "x", 1.0)
    monkeypatch.setattr(batch, "_compute", lambda task: [])
    assert batch.process_molecule(batch.MolTask("CCO", 2, cfg)) == []


@pytest.fixture
def two_sources(tmp_path):
    processed = tmp_path / "processed"
    for source in ("pmechdb", "reactmech"):
        _steps(processed, source, BAD)
    return processed, _config(tmp_path, timeout_s=120)


def test_a_failure_in_one_source_is_not_recomputed_in_another(two_sources, monkeypatch):
    processed, cfg = two_sources
    first = _run(processed, cfg, "pmechdb")
    assert first["reason"].tolist() == ["SPIN_PARITY"]
    assert failures.tombstone_path(cfg.cache_dir, BAD, 0, cfg.method).is_file()
    _forbid_compute(monkeypatch)
    other = _run(processed, cfg, "reactmech")
    assert other["reason"].tolist() == ["SPIN_PARITY"]
    assert other["detail"].iloc[0].startswith(failures.KNOWN_PREFIX)


def test_a_retry_asked_for_in_the_same_source_recomputes(two_sources, monkeypatch):
    processed, cfg = two_sources
    _run(processed, cfg, "pmechdb")
    calls = []
    real = batch._run_molecule
    monkeypatch.setattr(batch, "_run_molecule", lambda t: calls.append(t.smiles) or real(t))
    _run(processed, cfg, "pmechdb")  # retry_failed by default
    assert calls == [BAD]
    _run(processed, cfg, "pmechdb", retry_reasons=["SPIN_PARITY"])
    assert calls == [BAD, BAD]
