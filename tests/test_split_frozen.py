"""Frozen splits (`data.splits`, `arrows.steps`): existing steps keep their parts when steps are
added, new groups are placed by hash, bridging steps are SPLIT_CONFLICT and counted, the
snapshot survives a SAMPLE_VERSION bump, and a frozen pilot never redraws (`data.ingest`)."""

import json
from collections import Counter

import pandas as pd
import pytest

from ksmech.arrows import steps as st
from ksmech.data import provenance as prov
from ksmech.data import splits
from ksmech.data.ingest import IngestResult, freeze_sample, load_frozen_sample
from ksmech.data.schema import ElementaryStep, Source


def _alcohol(k: int) -> str:
    return "C" * (k + 4) + "O"  # more than 3 heavy atoms: a non-trivial species of its own


def _step(step_id: str, *reactants: str, mechanism_class: str | None = None) -> ElementaryStep:
    return ElementaryStep(
        step_id, Source.PMECHDB, None, None, ".".join(reactants), "[OH-]", [], mechanism_class
    )


BASE = [_step(f"s{k:02d}", _alcohol(k)) for k in range(30)]
NEW = [_step(f"n{k:02d}", _alcohol(k)) for k in range(30, 40)]


def _parts(split: splits.Split) -> dict[str, str]:
    return {i: p for p in ("train", "val", "test") for i in getattr(split, p)}


def test_without_a_snapshot_the_split_is_unchanged():
    assert splits.split_by_molecule(BASE, seed=0) == splits.split_by_molecule(
        BASE, seed=0, frozen=None
    )


def test_added_groups_never_move_an_existing_step():
    frozen = _parts(splits.split_by_molecule(BASE, seed=0))
    again = splits.split_by_molecule(BASE + NEW, seed=0, frozen=frozen)
    where = _parts(again)
    assert all(where[i] == part for i, part in frozen.items())
    assert {s.step_id for s in NEW} <= set(where) and not again.conflict
    # a new group's part depends on its own key only: adding other new groups moves nothing
    alone = _parts(splits.split_by_molecule(BASE + NEW[:1], seed=0, frozen=frozen))
    assert alone[NEW[0].step_id] == where[NEW[0].step_id]


def test_a_step_joining_its_group_takes_the_frozen_part():
    frozen = _parts(splits.split_by_molecule(BASE, seed=0))
    joiner = _step("j00", _alcohol(3))  # shares s03's molecule
    where = _parts(splits.split_by_molecule([*BASE, joiner], seed=0, frozen=frozen))
    assert where["j00"] == frozen["s03"]


def test_a_bridging_step_is_a_split_conflict_and_the_frozen_steps_stay():
    frozen = _parts(splits.split_by_molecule(BASE, seed=0))
    a = next(i for i, p in frozen.items() if p == "train")
    b = next(i for i, p in frozen.items() if p != "train")
    bridge = _step("b00", _alcohol(int(a[1:])), _alcohol(int(b[1:])))
    again = splits.split_by_molecule([*BASE, bridge], seed=0, frozen=frozen)
    assert again.conflict == ["b00"]
    assert all(_parts(again)[i] == part for i, part in frozen.items())


def test_split_ids_count_conflicts_per_split_and_name_loco_snapshots():
    ids = st.split_ids(BASE, "molecule")
    frozen_parts = {i: name.rsplit("_", 1)[1] for name, v in ids.items() for i in v}
    a = next(i for i, p in frozen_parts.items() if p == "train")
    b = next(i for i, p in frozen_parts.items() if p != "train")
    bridge = _step("b00", _alcohol(int(a[1:])), _alcohol(int(b[1:])))
    conflicts: Counter = Counter()
    again = st.split_ids([*BASE, bridge], "molecule", frozen=ids, conflicts=conflicts)
    assert conflicts == {"molecule": 1}
    assert all(set(ids[k]) <= set(again[k]) for k in ids)
    classed = [
        _step(s.step_id, s.reactants_smiles, mechanism_class="AB"[i % 2])
        for i, s in enumerate(BASE)
    ]
    loco = st.split_ids(classed, "loco")
    assert set(st.split_ids(classed, "loco", frozen=loco)) == set(loco)
    assert st.split_ids(classed, "loco", frozen=loco) == loco


def test_frozen_steps_a_later_rule_excludes_are_counted():
    ids = st.split_ids(BASE, "molecule")
    gone = st.frozen_excluded(ids, BASE[2:])
    assert sum(gone.values()) == 2


def _split_file(tmp_path):
    folder = tmp_path / "pmechdb"
    folder.mkdir()
    parquet = folder / "steps.parquet"
    parquet.write_bytes(b"steps")
    ids = st.split_ids(BASE, "molecule")
    st.write_split_file(
        st.splits_path("pmechdb", "molecule", tmp_path), "molecule", 0, ids, parquet
    )
    return ids, parquet


def test_a_snapshot_survives_a_sample_version_bump_and_a_new_parquet(tmp_path, monkeypatch):
    ids, parquet = _split_file(tmp_path)
    out = st.freeze_split_file("pmechdb", "molecule", tmp_path)
    assert prov.read_header(out).inputs == {}
    parquet.write_bytes(b"re-ingested")  # the split file goes stale, the snapshot does not
    with pytest.raises(prov.StaleArtifact):
        st.load_split_file(st.splits_path("pmechdb", "molecule", tmp_path))
    monkeypatch.setattr(st, "SAMPLE_VERSION", st.SAMPLE_VERSION + 1)
    assert st.load_frozen_split("pmechdb", "molecule", tmp_path) == ids
    assert st.load_frozen_split("pmechdb", "scaffold", tmp_path) is None


def test_a_snapshot_is_never_overwritten_and_never_edited(tmp_path):
    _split_file(tmp_path)
    out = st.freeze_split_file("pmechdb", "molecule", tmp_path)
    with pytest.raises(FileExistsError):
        st.freeze_split_file("pmechdb", "molecule", tmp_path)
    body = json.loads(out.read_text())
    body["splits"]["molecule_val"].append("intruder")
    out.write_text(json.dumps(body))
    with pytest.raises(prov.StaleArtifact, match="content hash"):
        st.load_frozen_split("pmechdb", "molecule", tmp_path)


def test_a_frozen_pilot_keeps_its_ids_when_the_accepted_set_changes(tmp_path):
    folder = tmp_path / "reactmech"
    folder.mkdir()
    pilot = [s.step_id for s in BASE[:10]]
    pd.DataFrame({"step_id": pilot}).to_parquet(folder / "steps.parquet")
    (folder / "ingest_report.json").write_text(json.dumps({"sample": {"n": 10, "seed": 0}}))
    freeze_sample("reactmech", tmp_path)
    frozen = load_frozen_sample("reactmech", tmp_path)
    assert frozen == set(pilot)
    result = IngestResult(Source.REACTMECH, steps=BASE[1:] + NEW)  # s00 now rejected; more accepted
    result.keep_sample(10, 0, frozen=frozen)
    assert [s.step_id for s in result.steps] == pilot[1:]
    assert result.sample["frozen_missing"] == 1 and result.sample["of_accepted"] == 39
    assert load_frozen_sample("pmechdb", tmp_path) is None
