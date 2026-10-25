"""Dataset build: missing QM records are skipped and counted, files load on re-run, summary."""

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rich.console import Console

from ksmech import processes
from ksmech.orbitals import build, cli
from ksmech.orbitals.build import BuildConfig, BuildRow, build_dataset, worst_molecules
from ksmech.orbitals.descriptors import feature_names
from ksmech.orbitals.tokens import atom_feature_names
from ksmech.qm.fukui import homo_index
from ksmech.qm.records import cache_path, save_record

MOLECULES = ["CO", "O", "c1ccccc1"]
MISSING = "[Cl-]"


def write_steps(path, steps):
    reactants, products, resonance = zip(*steps, strict=True)
    table = {"reactants_smiles": reactants, "products_smiles": products}
    pq.write_table(pa.table({**table, "resonance_step": resonance}), path)


@pytest.fixture
def dataset(tmp_path, sp_record):
    """A processed dir with four molecules and a QM cache holding records for three of them."""
    processed = tmp_path / "processed"
    path = processed / "pmechdb" / "steps.parquet"
    path.parent.mkdir(parents=True)
    smiles = ".".join(MOLECULES + [MISSING])
    write_steps(path, [(smiles, smiles, False)])
    config = BuildConfig(qm_cache_dir=tmp_path / "qm", out_dir=tmp_path / "tokens")
    for smi in MOLECULES:
        rec = sp_record(smi)
        save_record(rec, cache_path(config.qm_cache_dir, rec.smiles, 0, config.method))
    return processed, config


def run(processed, config, **kw):
    return build_dataset(
        "pmechdb", config, processed_dir=processed, console=Console(quiet=True), **kw
    )


def test_build_skips_missing_records_and_writes_summary(dataset):
    processed, config = dataset
    summary = run(processed, config)
    assert summary["n_molecules"] == 4 and summary["n_ok"] == 3
    assert summary["n_built"] == 3 and summary["n_missing_qm"] == 1 and summary["n_failed"] == 0
    assert len(list(config.out_dir.glob("*.npz"))) == 3
    on_disk = json.loads((config.out_dir / build.SUMMARY_NAME).read_text())
    assert on_disk["n_missing_qm"] == 1 and on_disk["method"] == config.method
    assert summary["type_counts"]["occupied"]["delocalized_pi"] == 3  # benzene
    assert summary["type_counts"]["virtual"]["sigma_star"] >= 2
    assert 0 < summary["occupied_unresolved_fraction"] < 0.5
    assert summary["pm"]["n_not_converged"] == 0 and summary["tokens_per_molecule"]["max"] > 30
    assert summary["worst_molecules"][0]["smiles"] == "c1ccccc1"
    inputs = summary["inputs"]
    assert list(inputs["orbital_inputs"]) == feature_names()
    assert list(inputs["atom_inputs"]) == atom_feature_names()
    n_orb = summary["n_occupied"] + summary["n_virtual"]
    assert all(q["n"] == n_orb for q in inputs["orbital_inputs"].values())
    assert inputs["orbital_inputs"]["occupation"]["p100"] == pytest.approx(2.0)
    assert inputs["atom_inputs"]["dual"]["n_missing"] > 0  # sp records carry no Fukui N+-1
    assert summary["masks"]["donors"] == summary["n_occupied"]
    assert summary["masks"]["d_type_label_mismatch"] == 0
    assert set(summary["chirality"]) == {"0"}
    assert summary["fermi_smeared"]["n_molecules"] == 0
    assert summary["fermi_smeared"]["steps"]["all"]["fully_tokenized"] == 0  # [Cl-] is missing
    json.dumps(summary)  # the whole summary is JSON


def test_fermi_smeared_molecules_and_steps_are_counted(dataset, sp_record):
    processed, config = dataset
    rec = sp_record("CO")
    occ = rec.occupations.copy()
    h = homo_index(occ)
    occ[h], occ[h + 1] = occ[h] - 0.3, occ[h + 1] + 0.3  # a vanishing gap, by hand
    save_record(
        replace(rec, occupations=occ), cache_path(config.qm_cache_dir, "CO", 0, config.method)
    )
    steps = [("CO.O", "CO.O", False), ("O", "O", False), ("c1ccccc1", "[Cl-]", True)]
    write_steps(processed / "pmechdb" / "steps.parquet", steps)
    fs = run(processed, config)["fermi_smeared"]
    assert fs["n_molecules"] == 1 and fs["smiles"] == ["CO"]
    keys = ("steps", "touching_smeared", "fully_tokenized", "fully_tokenized_touching_smeared")
    assert [fs["steps"]["all"][k] for k in keys] == [3, 1, 2, 1]
    assert [fs["steps"]["training"][k] for k in keys] == [2, 1, 2, 1]  # resonance step dropped


def test_rerun_loads_existing_files_unless_forced(dataset, monkeypatch):
    processed, config = dataset
    run(processed, config)
    calls = []
    real = build.build_molecule_tokens
    monkeypatch.setattr(
        build, "build_molecule_tokens", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    summary = run(processed, config)
    assert summary["n_loaded"] == 3 and summary["n_built"] == 0 and calls == []
    forced = BuildConfig(qm_cache_dir=config.qm_cache_dir, out_dir=config.out_dir, force=True)
    summary = run(processed, forced)
    assert summary["n_built"] == 3 and len(calls) == 3


def test_corrupt_record_is_a_counted_failure(dataset):
    processed, config = dataset
    cache_path(config.qm_cache_dir, "CO", 0, config.method).write_bytes(b"not an npz")
    summary = run(processed, config)
    assert summary["n_failed"] == 1 and summary["n_ok"] == 2
    assert "CO" in summary["failures"]


def test_limit_and_workers(dataset):
    processed, config = dataset
    summary = run(processed, config, limit=2, n_workers=2)
    assert summary["n_molecules"] == 2 and summary["n_ok"] + summary["n_missing_qm"] == 2


@pytest.mark.diagnostic
def test_worst_molecules_ranks_by_unresolved_fraction():
    rows = [
        BuildRow("A", "built", n_occ=10, n_unresolved_occ=1, unresolved_types={"unknown": 1}),
        BuildRow("B", "built", n_occ=4, n_unresolved_occ=2, unresolved_types={"delocalized_pi": 2}),
        BuildRow("C", "built", n_occ=5, n_unresolved_occ=0),
        BuildRow("D", "missing_qm"),
    ]
    worst = worst_molecules(rows, n=5)
    assert [w["smiles"] for w in worst] == ["B", "A"]
    assert worst[0]["fraction"] == 0.5 and worst[0]["unresolved_types"] == {"delocalized_pi": 2}


def test_cli_build_and_show(dataset, capsys):
    processed, config = dataset
    args = ["--source", "pmechdb", "--processed-dir", str(processed)]
    args += ["--qm-cache-dir", str(config.qm_cache_dir), "--out-dir", str(config.out_dir)]
    assert cli.main(["build", *args]) == 0  # benzene's delocalized pi is information, not failure
    assert cli.main(["show", "--smiles", "OC", "--qm-cache-dir", str(config.qm_cache_dir)]) == 0
    out = capsys.readouterr().out
    assert "lone_pair" in out and "sigma_star" in out
    assert cli.main(["show", "--smiles", MISSING, "--qm-cache-dir", str(config.qm_cache_dir)]) == 2


def test_cli_build_without_processed_data(tmp_path):
    rc = cli.main(["build", "--source", "pmechdb", "--processed-dir", str(tmp_path)])
    assert rc == 2


def test_workers_are_spawned_single_threaded(monkeypatch):
    for var in processes.THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    with processes.single_threaded_children():
        import os

        assert all(os.environ[v] == "1" for v in processes.THREAD_VARS)
    import os

    assert os.environ["OMP_NUM_THREADS"] == "16" and "OPENBLAS_NUM_THREADS" not in os.environ
