"""The cache index: rebuilt from the files' headers, refreshed incrementally, every unreadable
file listed with its problem, and `ksmech-qm index` over one or more roots."""

import sqlite3

import numpy as np
import pytest

from ksmech.data import provenance as prov
from ksmech.qm import failures, index
from ksmech.qm.cli import main
from ksmech.qm.records import QMRecord, cache_key, cache_path, save_record
from ksmech.qm.xtb_calc import QMConfig

METHOD = QMConfig().method


def _record(smiles: str, n: int = 2) -> QMRecord:
    nao = 2 * n
    return QMRecord(
        smiles=smiles,
        atom_order_smiles=smiles,
        conformer_id=0,
        method=METHOD,
        charge=0,
        unpaired=0,
        atomic_numbers=np.array([6] * n),
        coords_ang=np.zeros((n, 3)),
        energy_ha=-1.0,
        orbital_energies_ev=np.zeros(nao),
        occupations=np.zeros(nao),
        C=np.eye(nao),
        S=np.eye(nao),
        H_core=np.eye(nao),
        fock=np.eye(nao),
        charges=np.zeros(n),
        ao_atom=np.repeat(np.arange(n), 2),
        ao_l=np.zeros(nao, dtype=int),
        warnings="NO_FRONTIER",
    )


def _sidecar(root, kind: str, smiles: str, qm_path) -> None:
    arrays = {"x": np.arange(3)}
    header = prov.stamp(
        kind, 1, prov.file_input(qm_path), {"method": METHOD, "conformer_id": 0},
        prov.array_digest(arrays),
    )  # fmt: skip
    prov.save_npz(root / kind / f"{cache_key(smiles, 0, METHOD)}.npz", arrays, header)


@pytest.fixture
def root(tmp_path):
    """Two records (CCO with tokens and a sidecar, CC with neither), a tombstone, a corrupt
    record, a token file without a header and an atomic-write temporary."""
    root = tmp_path / "cache"
    qm = root / "qm"
    for smiles in ("CCO", "CC"):
        save_record(_record(smiles), cache_path(qm, smiles, 0, METHOD))
    cco = cache_path(qm, "CCO", 0, METHOD)
    _sidecar(root, "tokens", "CCO", cco)
    _sidecar(root, "bond_index", "CCO", cco)
    failures.remember(QMConfig(cache_dir=qm), "[CH2-]O", 0, "SCF_NOT_CONVERGED", "250", 30.0)
    (qm / f"{'0' * 64}.npz").write_bytes(b"not a zip")
    np.savez(root / "tokens" / f"{'1' * 64}.npz", x=np.zeros(1))
    (qm / ".x.npz.123.tmp").write_bytes(b"partial")
    return root


def test_refresh_lists_every_file_and_its_problems(root):
    counts = index.refresh(root)
    assert counts["qm_record:added"] == 3 and counts["qm_record:problem:CORRUPT_ARTIFACT"] == 1
    assert counts["qm_failure:added"] == 1
    assert counts["tokens:added"] == 2 and counts["tokens:problem:NO_HEADER"] == 1
    assert counts["bond_index:added"] == 1
    con = sqlite3.connect(index.index_path(root))
    row = con.execute(
        "SELECT smiles, method, n_atoms, warnings, schema_version, problem FROM artifact"
        " WHERE kind = 'qm_record' AND key = ?",
        (cache_key("CCO", 0, METHOD),),
    ).fetchone()
    assert row == ("CCO", METHOD, 2, "NO_FRONTIER", 1, None)
    failure = con.execute("SELECT smiles, reason, seconds FROM artifact WHERE kind = 'qm_failure'")
    assert failure.fetchall() == [("[CH2-]O", "SCF_NOT_CONVERGED", 30.0)]
    molecules = con.execute(
        "SELECT smiles, has_record, failure_reason, has_tokens, has_bond_index FROM molecule"
        " WHERE smiles IS NOT NULL ORDER BY smiles"
    ).fetchall()
    assert molecules == [
        ("CC", 1, None, 0, 0),
        ("CCO", 1, None, 1, 1),
        ("[CH2-]O", 0, "SCF_NOT_CONVERGED", 0, 0),
    ]


def test_refresh_reads_only_what_changed(root, monkeypatch):
    index.refresh(root)
    real = index.read_artifact
    read = []
    monkeypatch.setattr(index, "read_artifact", lambda p, k: read.append(p.name) or real(p, k))
    again = index.refresh(root)
    assert read == [] and sum(n for k, n in again.items() if not k.endswith("unchanged")) == 0
    tomb = failures.tombstone_path(root / "qm", "[CH2-]O", 0, METHOD)
    tomb.unlink()
    save_record(_record("CC", n=3), cache_path(root / "qm", "CC", 0, METHOD))
    counts = index.refresh(root)
    assert counts["qm_failure:removed"] == 1 and counts["qm_record:updated"] == 1
    assert read == [cache_path(root / "qm", "CC", 0, METHOD).name]


def test_rebuild_gives_the_same_rows(root):
    index.refresh(root)
    con = sqlite3.connect(index.index_path(root))
    before = con.execute("SELECT * FROM artifact ORDER BY kind, key").fetchall()
    con.close()
    counts = index.refresh(root, rebuild=True)
    assert counts["qm_record:added"] == 3 and counts["qm_record:unchanged"] == 0
    con = sqlite3.connect(index.index_path(root))
    assert con.execute("SELECT * FROM artifact ORDER BY kind, key").fetchall() == before


def test_another_index_version_is_recreated(root):
    index.refresh(root)
    con = sqlite3.connect(index.index_path(root))
    con.execute("PRAGMA user_version = 999")
    con.commit()
    con.close()
    with index.connect(root) as con:
        assert con.execute("SELECT COUNT(*) FROM artifact").fetchone()[0] == 0
    assert index.refresh(root)["qm_record:added"] == 3


def test_stats(root):
    index.refresh(root)
    s = index.stats(root)
    kinds = {k: (n, problems) for k, n, _, problems, _ in s["kinds"]}
    assert kinds == {
        "bond_index": (1, 0),
        "qm_failure": (1, 0),
        "qm_record": (3, 1),
        "tokens": (2, 1),
    }
    assert s["reasons"] == [("SCF_NOT_CONVERGED", 1, "[CH2-]O")]
    assert s["records_without_tokens"] == 2  # CC and the corrupt file
    assert s["tokens_without_bond_index"] == 1 and s["failures_with_a_record"] == 0
    assert s["refreshed_at"]


def test_cli(root, tmp_path, capsys):
    assert main(["index", "refresh", "--root", str(root)]) == 0
    assert main(["index", "stats", "--root", str(root), "--no-refresh"]) == 0
    out = capsys.readouterr().out
    assert "SCF_NOT_CONVERGED" in out and "CORRUPT_ARTIFACT" in out
    assert main(["index", "stats", "--root", str(tmp_path / "missing")]) == 2
