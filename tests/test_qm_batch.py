"""Batch runner: dedup, resumability, failure rows, manifest and summary."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rich.console import Console

from ksmech.qm import batch
from ksmech.qm.records import ReasonCode, cache_path
from ksmech.qm.xtb_calc import QMConfig

REACTANTS = ["[CH3:1][OH:2].[Cl-]", "CC=O.[OH2:5]"]
PRODUCTS = ["[CH3:1][Cl:3].[OH-:2]", "CC=O.[OH2:5]"]
BAD = "COC=[NH+][CuH2-]"  # 63 electrons: SPIN_PARITY


def write_steps(processed, reactants, products, source="pmechdb"):
    path = processed / source / "steps.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"reactants_smiles": reactants, "products_smiles": products}), path)


@pytest.fixture
def tiny(tmp_path):
    processed = tmp_path / "processed"
    write_steps(processed, REACTANTS + [BAD], PRODUCTS + [BAD])
    cfg = QMConfig(cache_dir=tmp_path / "cache", timeout_s=120)
    return processed, cfg


def run(processed, cfg, **kw):
    return batch.featurize_dataset(
        "pmechdb", processed_dir=processed, config=cfg, console=Console(quiet=True), **kw
    )


def test_collect_molecules_dedups_and_drops_atom_maps(tiny):
    processed, _ = tiny
    smiles, failures = batch.collect_molecules("pmechdb", processed)
    assert failures == []
    assert smiles == sorted({"CO", "[Cl-]", "CCl", "[OH-]", "CC=O", "O", BAD})
    assert all(":" not in s for s in smiles)


def test_run_writes_cache_and_manifest(tiny):
    processed, cfg = tiny
    view = run(processed, cfg)
    assert list(view.columns) == batch.MANIFEST_COLUMNS
    assert len(view) == 7 and (view["conformer_id"] == 0).all()
    ok = view[view["status"] == "ok"]
    assert len(ok) == 6
    for row in ok.itertuples():
        assert cache_path(cfg.cache_dir, row.smiles, 0, cfg.method).is_file()
        assert row.cache_path.endswith(".npz")
    bad = view[view["status"] == "failed"].iloc[0]
    assert bad["smiles"] == BAD and bad["reason"] == "SPIN_PARITY" and bad["detail"]
    on_disk = pd.read_parquet(batch.manifest_path("pmechdb", processed))
    assert len(on_disk) == 7


def test_rerun_skips_everything_that_is_cached(tiny, monkeypatch):
    processed, cfg = tiny
    run(processed, cfg)
    calls = []
    real = batch.process_molecule
    monkeypatch.setattr(batch, "process_molecule", lambda t: calls.append(t.smiles) or real(t))
    run(processed, cfg)
    assert calls == [BAD]  # only the failure is retried
    calls.clear()
    run(processed, cfg, retry_failed=False)
    assert calls == []


def test_resume_works_from_cache_when_the_manifest_is_lost(tiny, monkeypatch):
    processed, cfg = tiny
    run(processed, cfg)
    batch.manifest_path("pmechdb", processed).unlink()
    real = batch.process_molecule

    def only_the_unknown_failure(task):
        assert task.smiles == BAD, f"recomputed a cached molecule: {task.smiles}"
        return real(task)

    monkeypatch.setattr(batch, "process_molecule", only_the_unknown_failure)
    view = run(processed, cfg, retry_failed=False)  # no manifest, so BAD has no recorded failure
    assert (view["status"] == "ok").sum() == 6


def test_limit_takes_a_reproducible_subset(tiny):
    processed, cfg = tiny
    a = run(processed, cfg, limit=3)
    b = run(processed, cfg, limit=3)
    assert len(a) == 3 and list(a["smiles"]) == list(b["smiles"])


def test_method_change_does_not_reuse_results(tiny):
    processed, cfg = tiny
    run(processed, cfg, limit=2)
    gas = QMConfig(cache_dir=cfg.cache_dir, solvent=None, timeout_s=120)
    view = run(processed, gas, limit=2)
    assert set(view["method"]) == {gas.method}
    full = pd.read_parquet(batch.manifest_path("pmechdb", processed))
    assert set(full["method"]) == {cfg.method, gas.method}


def test_process_molecule_never_raises(tmp_path):
    cfg = QMConfig(cache_dir=tmp_path)
    for smi in ("C(C", BAD, "[Na+]"):
        rows = batch.process_molecule(batch.MolTask(smi, 1, cfg))
        assert rows and all(r["smiles"] == smi for r in rows)
    assert rows[0]["status"] == "ok" and rows[0]["warnings"] == "NO_FRONTIER"


def test_timeout_is_reported_not_swallowed(tmp_path, monkeypatch):
    def slow(task):
        import time

        time.sleep(5)

    monkeypatch.setattr(batch, "_compute", slow)
    batch._init_worker()
    rows = batch.process_molecule(
        batch.MolTask("CCO", 1, QMConfig(cache_dir=tmp_path, timeout_s=1))
    )
    assert rows[0]["status"] == "failed" and rows[0]["reason"] == "TIMEOUT"
    assert rows[0]["seconds"] >= 1


def test_summary_counts_and_reason_table():
    rows = [
        batch._row("a", 0, status="ok", warnings="VBUR_FAIL", method="m"),
        batch._row("b", 0, status="ok", method="m"),
        batch._row("c", 0, reason="SPIN_PARITY", method="m"),
        batch._row("d", 0, reason="SPIN_PARITY", method="m"),
    ]
    s = batch.summarize(pd.DataFrame(rows))
    assert s["n_total"] == 4 and s["n_ok"] == 2 and s["success_fraction"] == 0.5
    assert s["reasons"] == {"SPIN_PARITY": 2} and s["warnings"] == {"VBUR_FAIL": 1}
    console = Console(record=True, width=120)
    batch.print_summary(pd.DataFrame(rows), console)
    text = console.export_text()
    assert "SPIN_PARITY" in text and "50.00%" in text and "VBUR_FAIL" in text


@pytest.mark.slow
def test_worker_pool_matches_inline_run(tiny):
    processed, cfg = tiny
    view = run(processed, cfg, n_workers=2)
    assert (view["status"] == "ok").sum() == 6
    assert view.loc[view["status"] == "failed", "reason"].tolist() == [str(ReasonCode.SPIN_PARITY)]


def test_truncated_cache_files_are_recomputed(tiny):
    processed, cfg = tiny
    run(processed, cfg)
    victim = cache_path(cfg.cache_dir, "CO", 0, cfg.method)
    victim.write_bytes(b"")  # what a crash leaves behind
    view = run(processed, cfg, retry_failed=False)
    assert victim.stat().st_size > 0
    assert view.loc[view["smiles"] == "CO", "status"].tolist() == ["ok"]


def test_retry_reasons_pick_the_failures_to_retry(tiny, monkeypatch):
    processed, cfg = tiny
    run(processed, cfg)
    calls = []
    real = batch.process_molecule
    monkeypatch.setattr(batch, "process_molecule", lambda t: calls.append(t.smiles) or real(t))
    run(processed, cfg, retry_reasons=["EMBED_FAILED"])
    assert calls == []  # BAD failed with SPIN_PARITY
    run(processed, cfg, retry_reasons=["SPIN_PARITY", "TIMEOUT"])
    assert calls == [BAD]
