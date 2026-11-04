"""Real data: the FlowER dry run of open-shell plan P6 step 8 on a 50-line slice, read-only."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.realdata

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("data/raw/flower/flower_new_dataset")
_spec = importlib.util.spec_from_file_location(
    "flower_label_dryrun", ROOT / "scripts" / "probes" / "flower_label_dryrun.py"
)
dry = importlib.util.module_from_spec(_spec)
sys.modules["flower_label_dryrun"] = dry
_spec.loader.exec_module(dry)


def test_the_dry_run_reads_fifty_lines_and_writes_one_report(tmp_path):
    if not (RAW / "train.txt").is_file():
        pytest.skip("no FlowER raw files")
    before = {p: p.stat().st_mtime_ns for p in RAW.iterdir()}
    dry.main(["--limit", "50", "--workers", "1", "--out", str(tmp_path)])
    assert [p.name for p in tmp_path.iterdir()] == ["label_dryrun_flower.json"]
    report = json.loads((tmp_path / "label_dryrun_flower.json").read_text())
    assert report["n_lines"] == 50 and sum(report["codes"].values()) == 50
    assert report["label_ok_fraction_of_non_resonance"] >= 0.95
    assert report["codes"].get("DISSOCIATIVE_ET_STEP", 0) == 0
    assert {p: p.stat().st_mtime_ns for p in RAW.iterdir()} == before
