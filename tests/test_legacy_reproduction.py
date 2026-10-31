"""Real data: the legacy reproduction gate of the v18 rebuild (open-shell plan P0a, step 8).

With every new flag off (pair cells valid, no orbital extras, the 0.05 eV frontier window, the
FD columns unselected, legacy decoder caps, floor and GBDT on the valid cells) the v18 store
reproduces the P0 references on every sample P0a did not change (`p0a_changes.json`,
`changed_steps`): the floor ranks, the stored v17 GBDT model's ranks and three runs' decode hits
per sample. Every rebuilt token file kept its schema-9 fields (`tokens_summary.json`,
`replaced`), and a later build changes no closed-shell file. The references are the v17 copies
(`data/processed_v17/`, `<run>/p0_reference/`); the new files come from the CLI runs of the
rebuild:

    ksmech-eval floor --source pmechdb
    ksmech-physics gbdt --source pmechdb --score-only --allow-stale \\
        --model data/processed_v17/pmechdb/gbdt_lgbm_molecule_geomfree.npz
    ksmech-eval decode --run flex_first --source pmechdb
    ksmech-eval decode --run sparse_resonance_stop0 --source reactmech
    ksmech-eval decode --run sparse_resonance --run-source pmechdb+reactmech --source reactmech
"""

import json
from pathlib import Path

import pytest

from ksmech.data.io import DEFAULT_PROCESSED_DIR

pytestmark = pytest.mark.realdata

OLD = Path("data/processed_v17")
MODELS = Path("data/models")
GBDT_V17 = "gbdt_lgbm_ranks_molecule_geomfree_gbdt_lgbm_molecule_geomfree.json"
DECODES = [  # (run directory, hits file, source of the samples)
    ("pmechdb/flex_first", "decode_hits_molecule_val.json", "pmechdb"),
    ("reactmech/sparse_resonance_stop0", "decode_hits_molecule_val.json", "reactmech"),
    ("pmechdb+reactmech/sparse_resonance", "decode_hits_molecule_val_reactmech.json", "reactmech"),
]


def _read(path: Path) -> dict:
    if not path.is_file():
        pytest.skip(f"{path} is missing (run the P0a rebuild and the commands of the docstring)")
    return json.loads(path.read_text())


def _changed(source: str) -> set[str]:
    return set(_read(DEFAULT_PROCESSED_DIR / source / "p0a_changes.json")["changed_steps"])


def _rows(rows: list[dict]) -> dict[str, tuple]:
    return {r["step_id"]: tuple(v for k, v in sorted(r.items()) if k != "step_id") for r in rows}


def _compare(old_rows: list[dict], new_rows: list[dict], changed: set[str]) -> dict:
    """Per-sample agreement on the samples both hold that P0a did not change; every sample
    that differs must be a changed one."""
    old, new = _rows(old_rows), _rows(new_rows)
    shared = (set(old) & set(new)) - changed
    differ = sorted(i for i in set(old) & set(new) if old[i] != new[i])
    unexplained = [i for i in differ if i not in changed]
    gone = sorted((set(old) - set(new)) - changed)
    return {"n": len(shared), "unexplained": unexplained[:10], "gone": gone[:10]}


@pytest.mark.parametrize("source", ["pmechdb", "reactmech"])
def test_a_token_rebuild_changes_no_closed_shell_file(source):
    """P0a's schema-10 rebuild replaced every token file and changed no schema-9 field (gate
    passed 2026-10-29). A later build replaces only stale files: since P2b the open shells of the
    plateau rule, so every replaced file that differs is an open shell (the list is complete)."""
    from ksmech.orbitals.tokens import drawn_unpaired

    replaced = _read(DEFAULT_PROCESSED_DIR / source / "tokens_summary.json")["replaced"]
    assert len(replaced["smiles"]) == replaced["n_differing"], replaced
    assert all(drawn_unpaired(s) > 0 for s in replaced["smiles"]), replaced


@pytest.mark.parametrize("part", ["val", "test"])
def test_the_floor_reproduces_per_sample(part):
    name = "anchor_floor_ranks_molecule.json"
    old, new = _read(OLD / "pmechdb" / name)[part], _read(DEFAULT_PROCESSED_DIR / "pmechdb" / name)
    got = _compare(old, new[part], _changed("pmechdb"))
    assert got["n"] > 1000 and not got["unexplained"] and not got["gone"], got
    if part == "test":  # no PMechDB test step changed: the P0 numbers, exactly
        ranks = [r["rank"] for r in new[part]]
        tops = [round(100 * sum(r < k for r in ranks) / len(ranks), 1) for k in (1, 3, 5)]
        assert len(ranks) == 1182 and tops == [17.8, 27.3, 31.6]


@pytest.mark.parametrize("part", ["val", "test"])
def test_the_v17_gbdt_model_reproduces_per_sample(part):
    old = _read(OLD / "pmechdb" / "gbdt_lgbm_ranks_molecule_geomfree.json")[part]["per_sample"]
    new = _read(DEFAULT_PROCESSED_DIR / "pmechdb" / GBDT_V17)[part]["per_sample"]
    got = _compare(old, new, _changed("pmechdb"))
    assert got["n"] > 1000 and not got["unexplained"] and not got["gone"], got


@pytest.mark.parametrize(("run", "hits", "source"), DECODES)
def test_decode_hits_reproduce_per_sample(run, hits, source):
    old = _read(MODELS / run / "p0_reference" / hits)["per_sample"]
    new = _read(MODELS / run / hits)
    assert new["provenance"]["config"].get("caps") == "legacy", "a decode from before v18"
    got = _compare(old, new["per_sample"], _changed(source))
    assert got["n"] > 800 and not got["unexplained"] and not got["gone"], got
