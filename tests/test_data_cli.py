import csv
import json

from ksmech.data.cli import main
from ksmech.data.io import read_steps
from tests.conftest import MINI_PMECHDB


def test_ingest_then_stats(tmp_path, capsys):
    args = ["--source", "pmechdb", "--processed-dir", str(tmp_path)]
    assert main(["ingest", "--raw", str(MINI_PMECHDB), *args]) == 0
    out = capsys.readouterr().out
    assert "Rejection reasons" in out and "UNBALANCED" in out

    out_dir = tmp_path / "pmechdb"
    assert len(read_steps(out_dir / "steps.parquet")) == 5
    with open(out_dir / "rejections.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["reason"] for r in rows] == ["UNBALANCED"]
    report = json.loads((out_dir / "ingest_report.json").read_text())
    assert report["n_records"] == 6 and report["n_rejected"] == 1

    assert main(["stats", *args]) == 0
    out = capsys.readouterr().out
    for title in ("Arrows per step", "Heavy atoms per step", "Steps per mechanism class"):
        assert title in out
    assert "Steps per pathway" not in out  # no pathway ids: the table is skipped
    stats = json.loads((out_dir / "stats.json").read_text())
    assert stats["n_steps"] == 5 and stats["n_pathways"] is None


def test_stats_without_ingest_fails_cleanly(tmp_path, capsys):
    assert main(["stats", "--source", "pmechdb", "--processed-dir", str(tmp_path)]) == 2
    assert "not found" in capsys.readouterr().out


def test_mech_uspto_ingest_not_implemented(tmp_path, capsys):
    args = ["ingest", "--source", "mech_uspto", "--processed-dir", str(tmp_path)]
    assert main(args) == 2
    assert "arrow replay" in capsys.readouterr().out
