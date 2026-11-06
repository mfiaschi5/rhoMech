"""Real data: the re-scoring of what exists on closed_base (open-shell plan P0b, step 2).

The floor, both GBDTs and the old checkpoints scored on the pinned cohort give, per sample, what
the P0a gate recorded on the whole molecule split, and every file covers exactly the cohort's
samples of its part (the GBDTs: their own subsets, `closed_base_gbdt_valid` and `_all`). The
files come from

    ksmech-eval pin --source pmechdb --name closed_base
    ksmech-eval floor --source pmechdb --cohort closed_base [--cells all]
    ksmech-eval decode --run flex_first --source pmechdb --part val|test --cohort closed_base
    ksmech-eval decode --run sparse_resonance --run-source pmechdb+reactmech --source pmechdb \\
        --part val|test --cohort closed_base
"""

import json
from pathlib import Path

import pytest

from ksmech.arrows.steps import sample_ids, samples_path
from ksmech.data.io import DEFAULT_PROCESSED_DIR
from ksmech.eval import cohorts as co

pytestmark = pytest.mark.realdata

PM = DEFAULT_PROCESSED_DIR / "pmechdb"
MODELS = Path("data/models")
DECODES = [  # (run directory, hits file on closed_base)
    ("pmechdb/flex_first", "decode_hits_molecule_{part}_closed_base.json"),
    ("pmechdb+reactmech/sparse_resonance", "decode_hits_molecule_{part}_pmechdb_closed_base.json"),
]


def _read(path: Path) -> dict:
    if not path.is_file():
        pytest.skip(f"{path} is missing (pin closed_base and run the commands of the docstring)")
    return json.loads(path.read_text())


def _cohort(name: str, part: str) -> set[str]:
    if not co.cohort_path("pmechdb", name).exists():
        pytest.skip(f"cohort {name} is not pinned")
    return co.load_cohort("pmechdb", name) & set(
        sample_ids(samples_path("pmechdb", f"molecule_{part}"))
    )


@pytest.mark.parametrize("cells", ["", "_allcells"])
@pytest.mark.parametrize("part", ["val", "test"])
def test_the_floor_on_closed_base_is_the_p0a_floor(part, cells):
    whole = _read(PM / f"anchor_floor_ranks_molecule{cells}.json")[part]
    pinned = _read(PM / f"anchor_floor_ranks_molecule_closed_base{cells}.json")[part]
    assert pinned == whole
    assert {r["step_id"] for r in pinned} == _cohort("closed_base", part)
    if part == "test" and not cells:
        ranks = [r["rank"] for r in pinned]
        tops = [round(100 * sum(r is not None and r < k for r in ranks) / len(ranks), 1)
                for k in (1, 3, 5)]  # fmt: skip
        assert len(ranks) == 1182 and tops == [17.8, 27.3, 31.6]


@pytest.mark.parametrize("cells", ["valid", "all"])
@pytest.mark.parametrize("part", ["val", "test"])
def test_the_gbdt_ranks_cover_their_own_subset(part, cells):
    tag = "" if cells == "valid" else "_allcells"
    ranked = _read(PM / f"gbdt_lgbm_ranks_molecule_geomfree{tag}.json")[part]["per_sample"]
    assert {r["step_id"] for r in ranked} == _cohort(f"closed_base_gbdt_{cells}", part)


def test_flex_first_hits_on_closed_base_val_are_the_p0a_hits():
    run = MODELS / "pmechdb" / "flex_first"
    old = _read(run / "decode_hits_molecule_val.json")["per_sample"]
    new = _read(run / "decode_hits_molecule_val_closed_base.json")["per_sample"]
    assert len(new) == 1238 and new == old


@pytest.mark.parametrize("part", ["val", "test"])
@pytest.mark.parametrize(("run", "hits"), DECODES)
def test_every_closed_base_decode_covers_the_cohort(run, hits, part):
    rows = _read(MODELS / run / hits.format(part=part))["per_sample"]
    assert {r["step_id"] for r in rows} == _cohort("closed_base", part)
