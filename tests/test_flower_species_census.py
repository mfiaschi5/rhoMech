"""scripts/probes/flower_species_census.py (open-shell plan P2 step 6) on the FlowER files: the
fact-check counts the scan's species list rests on."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "flower_species_census", ROOT / "scripts" / "probes" / "flower_species_census.py"
)
census = importlib.util.module_from_spec(_spec)
sys.modules["flower_species_census"] = census
_spec.loader.exec_module(census)


@pytest.mark.realdata
def test_census_reproduces_the_fact_check_counts():
    if not (ROOT / census.RAW).exists():
        pytest.skip(f"{census.RAW} is not on this host")
    rep = census.report(census.radical_lines(ROOT / census.RAW))
    assert rep["n_species"] == 4284 and not rep["unread_lines"]
    assert rep["multi_radical"]["n"] == 56
    assert rep["named"]["CO"]["lines"] == {"RC": 5, "RS": 28}
    assert rep["named"]["N2"]["lines"].get("RC") == 5
