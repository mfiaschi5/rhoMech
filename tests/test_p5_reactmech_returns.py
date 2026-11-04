"""Real data: the ReactMech pilot after P5 (open-shell plan P5 steps 1 and 3).

P5 computed the species of the pilot steps P0a's laws and P1's V1 returned to the domain, with
no re-ingest: the pilot keeps its 10,000 ids; every species of a training step has a manifest
row under its own method, ok with its Wiberg sidecar or failed with a counted reason; a training
step without a sample holds a species without an ok record; and every pre-P5 sample keeps its
part in every frozen view and its column digests (`baseline_p5_pre.json`, pinned before P5's
sample build).
"""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from ksmech.data import chem
from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_training_steps, steps_path
from ksmech.qm.spin import method_for
from ksmech.qm.xtb_calc import QMConfig

pytestmark = pytest.mark.realdata

RM = DEFAULT_PROCESSED_DIR / "reactmech"
REFERENCE = RM / "p5_reference"  # the views and reports as they stood before P5's sample build
BASE = QMConfig().method
# the two pilot steps that stay excluded (Pd already -1 with three X ligands plus P+)
STILL_EXCLUDED = {"reactmech-cfd05e0cae5eee42", "reactmech-5e7499636a4c0e94"}


def _views(root: Path, prefix: str = "samples_") -> dict[str, Path]:
    """View name -> path of every split view under `root` (the reports left out)."""
    return {
        p.name.removeprefix(prefix).removesuffix(".json"): p
        for p in sorted(root.glob(f"{prefix}*_*.json"))
        if "report" not in p.name and not p.name.endswith("_index.json")
    }


@pytest.fixture(scope="module")
def species_of() -> dict[str, set[str]]:
    """Training step id -> the map-free canonical SMILES of its fragments."""
    if not REFERENCE.is_dir():
        pytest.skip("no p5_reference (the P5 build has not run)")
    wanted = {s.step_id for s in read_training_steps(steps_path("reactmech"))}
    columns = ["step_id", "reactants_smiles", "products_smiles"]
    out: dict[str, set[str]] = {}
    for row in pq.read_table(steps_path("reactmech"), columns=columns).to_pylist():
        if row["step_id"] in wanted:
            sides = (row["reactants_smiles"], row["products_smiles"])
            frags = [f for side in sides for f in chem.fragments(chem.parse_mapped(side))]
            out[row["step_id"]] = {chem.unmapped_canonical(f) for f in frags}
    return out


@pytest.fixture(scope="module")
def ok_species() -> tuple[set[str], set[str]]:
    """(the species with an ok record, the species with any row), each under its own method."""
    manifest = pd.read_parquet(RM / "qm_manifest.parquet")
    mine = manifest[manifest["method"] == [method_for(s, BASE) for s in manifest["smiles"]]]
    return set(mine.loc[mine["status"] == "ok", "smiles"]), set(mine["smiles"])


def test_the_pilot_keeps_its_10000_ids():
    frozen = set(json.loads((RM / "sample_frozen.json").read_text())["ids"])
    ids = set(pq.read_table(steps_path("reactmech"), columns=["step_id"]).column(0).to_pylist())
    training = {s.step_id for s in read_training_steps(steps_path("reactmech"))}
    assert len(frozen) == 10_000 and ids == frozen
    assert ids - training >= STILL_EXCLUDED


def test_every_species_has_a_row_and_every_ok_record_its_sidecar(species_of, ok_species):
    from ksmech.qm.bond_index import DEFAULT_BOND_INDEX_DIR, bond_index_path

    ok, with_row = ok_species
    species = set().union(*species_of.values())
    assert species - with_row == set()
    no_sidecar = {
        s
        for s in species & ok
        if not bond_index_path(DEFAULT_BOND_INDEX_DIR, s, 0, method_for(s, BASE)).is_file()
    }
    assert no_sidecar == set()


def test_a_training_step_without_a_sample_holds_a_species_without_a_record(species_of, ok_species):
    from ksmech.arrows.steps import sample_ids, samples_path

    ok, _ = ok_species
    report = json.loads((RM / "samples_report.json").read_text())
    assert set(report["skipped"]) <= {"MISSING_TOKENS"}, report["skipped"]
    have = {
        i
        for p in ("train", "val", "test")
        for i in sample_ids(samples_path("reactmech", f"scaffold_{p}"))
    }
    unexplained = [i for i, sp in species_of.items() if i not in have and sp <= ok]
    assert unexplained == []


def test_every_pre_p5_sample_keeps_its_part_in_every_view():
    before, now = _views(REFERENCE), _views(RM)
    assert set(before) <= set(now)
    for name, path in before.items():
        old = set(json.loads(path.read_text())["ids"])
        new = set(json.loads(now[name].read_text())["ids"])
        assert old <= new, (name, sorted(old - new)[:5])


def test_every_pre_p5_sample_keeps_its_column_digests(realdata_workers):
    from ksmech.eval import pin

    if not pin.baseline_path("reactmech", "p5_pre").exists():
        pytest.skip("baseline_p5_pre.json is not pinned")
    base = pin.load_baseline("reactmech", "p5_pre")
    views = _views(RM)
    now: dict[str, dict[str, str]] = {}
    for name in [f"{s}_{p}" for s in ("molecule", "scaffold") for p in ("train", "val", "test")]:
        now |= {i: d for i, d, _ in pin.read_view(views[name], realdata_workers)}
    got = pin.compare_digests(base, now)
    assert got["n_shared"] == len(base) and not got["samples_only_in_baseline"]
    assert not got["differ"], {c: ids[:3] for c, ids in got["differ"].items()}
    assert not got["columns_only_in_baseline"] and not got["columns_only_in_current"]
