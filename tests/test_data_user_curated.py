"""The user_curated loader: re-ingest of the serve export parquet, counted rejections, ids,
LOADERS registration and the `ksmech-data ingest --source user_curated` CLI."""

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ksmech.data import user_curated
from ksmech.data.cli import LOADERS, main
from ksmech.data.io import read_steps, write_steps
from ksmech.data.schema import STEP_SCHEMA, Source
from ksmech.data.validation import RejectReason, make_step_id


def _rows(steps) -> list[dict]:
    return [replace(s, source=Source.USER_CURATED).to_dict() for s in steps]


def _write(rows: list[dict], path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=STEP_SCHEMA), path)
    return path


@pytest.fixture
def curated_parquet(tmp_path, mini_result):
    """The mini PMechDB fixture rewritten as a user_curated export parquet."""
    steps = [replace(s, source=Source.USER_CURATED) for s in mini_result.steps]
    return write_steps(steps, tmp_path / "export_raw" / "steps.parquet")


def test_mini_fixture_reingests_identically(curated_parquet, mini_result):
    result = user_curated.load(curated_parquet)
    assert result.source == Source.USER_CURATED
    assert not result.rejections and result.n_records == len(mini_result.steps)
    assert result.inputs == [str(curated_parquet)]
    for got, want in zip(result.steps, mini_result.steps, strict=True):
        assert got.source == Source.USER_CURATED
        assert got.step_id == want.step_id
        assert got.reactants_smiles == want.reactants_smiles
        assert got.products_smiles == want.products_smiles
        assert got.arrows == want.arrows
        assert got.resonance_step == want.resonance_step


def test_bad_rows_are_counted(tmp_path, mini_result):
    rows = _rows(mini_result.steps[:3])
    rows[0]["arrows"] = [{"source_kind": "lone_pair", "source_atoms": [10, 20], **_atom_sink()}]
    rows[1]["source"] = "pmechdb"
    rows[2]["arrows"] = [
        {"source_kind": "radical", "source_atoms": [10], "sink_kind": "atom", "sink_atoms": [20]}
    ]
    result = user_curated.load(_write(rows, tmp_path / "bad.parquet"))
    assert result.n_records == 3 and not result.steps
    assert result.reason_counts() == {"ARROW_INVALID": 2, "MALFORMED_RECORD": 1}
    radical = next(r for r in result.rejections if r.step_id == rows[2]["step_id"])
    assert radical.reason == RejectReason.ARROW_INVALID and "radical" in radical.detail
    assert all(r.raw for r in result.rejections)
    invalid = next(r for r in result.rejections if r.step_id == rows[0]["step_id"])
    assert invalid.raw.endswith(user_curated.INVALID_ARROWS)


def _atom_sink() -> dict:
    return {"sink_kind": "atom", "sink_atoms": [20]}


def test_ids_are_deterministic(tmp_path, mini_result):
    rows = _rows(mini_result.steps[:2])
    rows[0]["step_id"] = ""  # no id: derived from the record text
    path = _write(rows, tmp_path / "ids.parquet")
    a, b = user_curated.load(path), user_curated.load(path)
    assert [s.step_id for s in a.steps] == [s.step_id for s in b.steps]
    s0 = mini_result.steps[0]
    raw = f"{s0.reactants_smiles}>>{s0.products_smiles} 10=20;20,21=21"
    assert a.steps[0].step_id == make_step_id(Source.USER_CURATED, raw)
    assert a.steps[0].step_id.startswith("user_curated-")
    assert a.steps[1].step_id == rows[1]["step_id"]


def test_rejected_by_build_step(tmp_path, mini_result):
    rows = _rows(mini_result.steps[:1])
    rows[0]["products_smiles"] = rows[0]["reactants_smiles"]  # the arrows no longer reproduce
    result = user_curated.load(_write(rows, tmp_path / "mismatch.parquet"))
    assert not result.steps and result.reason_counts() == {"ARROWS_MISMATCH": 1}


def test_check_row_matches_load(tmp_path, mini_result):
    """check_row gives, per row, the rejection `load` records for it (None when accepted)."""
    rows = _rows(mini_result.steps[:5])
    rows[1]["arrows"] = [{"source_kind": "lone_pair", "source_atoms": [10, 20], **_atom_sink()}]
    rows[2]["source"] = "pmechdb"
    rows[3]["products_smiles"] = rows[3]["reactants_smiles"]  # the arrows no longer reproduce
    checks = [user_curated.check_row(r) for r in rows]
    assert checks[0] is None and checks[4] is None
    assert [c.reason for c in checks[1:4]] == [
        RejectReason.ARROW_INVALID,
        RejectReason.MALFORMED_RECORD,
        RejectReason.ARROWS_MISMATCH,
    ]
    result = user_curated.load(_write(rows, tmp_path / "mixed.parquet"))
    by_id = {r.step_id: r for r in result.rejections}
    assert {s.step_id for s in result.steps} == {rows[0]["step_id"], rows[4]["step_id"]}
    for row, check in zip(rows[1:4], checks[1:4], strict=True):
        got = by_id[row["step_id"]]
        assert (got.reason, got.detail) == (check.reason, check.detail)


def test_check_row_radical_and_unparsable_arrows(mini_result):
    (row,) = _rows(mini_result.steps[:1])
    radical = {"source_kind": "radical", "source_atoms": [10], **_atom_sink()}
    rejected = user_curated.check_row(row | {"arrows": [radical]})
    assert rejected.reason == RejectReason.ARROW_INVALID and "radical" in rejected.detail
    rejected = user_curated.check_row(row | {"arrows": [{"source_kind": "lone_pair"}]})
    assert rejected.reason == RejectReason.ARROW_INVALID and "KeyError" in rejected.detail
    assert user_curated.check_row(row | {"step_id": ""}) is None  # the id is derived


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        user_curated.load(tmp_path / "nope.parquet")


def test_directory_input(curated_parquet):
    assert len(user_curated.load(curated_parquet.parent).steps) == 5


def test_loader_registered():
    assert LOADERS[Source.USER_CURATED] is user_curated.load
    assert user_curated.RAW_FILE.as_posix() == "data/curation/export_raw/steps.parquet"


def test_cli_ingest_writes_three_outputs(curated_parquet, tmp_path, capsys):
    processed = tmp_path / "processed"
    args = ["ingest", "--source", "user_curated", "--raw", str(curated_parquet)]
    assert main([*args, "--processed-dir", str(processed)]) == 0
    out = processed / "user_curated"
    assert len(read_steps(out / "steps.parquet")) == 5
    assert (out / "rejections.csv").read_text().startswith("step_id,reason,detail,raw")
    report = json.loads((out / "ingest_report.json").read_text())
    assert report["source"] == "user_curated" and report["n_accepted"] == 5
    assert "user_curated" in capsys.readouterr().out


def test_cli_missing_raw_returns_2(tmp_path, capsys):
    args = ["ingest", "--source", "user_curated", "--raw", str(tmp_path / "none.parquet")]
    assert main([*args, "--processed-dir", str(tmp_path)]) == 2
    assert "cannot ingest" in capsys.readouterr().out
