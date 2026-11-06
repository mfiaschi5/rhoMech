"""No real FlowER RS/RC line appears in any tracked file (open-shell plan P8 step 8): RMechDB's
lines keep their CC-BY-NC-ND license and live only in data/ (local, the private rhoMech-data and
pods, never the public repository). The fixtures are written by hand."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from ksmech.data import rmechdb

pytestmark = pytest.mark.realdata

ROOT = Path(__file__).resolve().parents[1]
RXN = re.compile(r"[^\s\"'`|,]+>>[^\s\"'`|,]+")


def _flower_reactions() -> set[str]:
    inputs = rmechdb.resolve_inputs(ROOT / rmechdb.RAW_DIR)
    return {rxn for _, _, rxn in rmechdb.radical_lines(inputs)}


def test_no_tracked_file_holds_a_flower_rs_or_rc_line():
    if not (ROOT / rmechdb.RAW_DIR).is_dir():
        pytest.skip("FlowER's files are not under data/raw/flower")
    lines = _flower_reactions()
    assert len(lines) > 4000
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    found = []
    for name in tracked:
        path = ROOT / name
        if not path.is_file() or path.stat().st_size > 20_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found += [(name, m) for m in RXN.findall(text) if m in lines]
    assert not found, found[:5]
