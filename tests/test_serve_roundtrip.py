"""The data chain without HTTP: curated SN2 mechanisms -> export (raw parquet, processed
user_curated steps, PMechDB arrow-code CSV, ledger, report) -> `user_curated.load` -> alignment
and the predicted-arrow round trip on the synthetic tokens -> train-only in every split."""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from sqlmodel import select

from ksmech.data import pmechdb, user_curated
from ksmech.data.io import read_steps
from ksmech.data.schema import STEP_SCHEMA, Source
from ksmech.serve import db
from ksmech.serve.contracts import LocalJobContext
from ksmech.serve.curation import create_mechanism, set_status
from ksmech.serve.export import compact_step, run_export
from ksmech.serve.schemas import CurateIn, ExportReportOut
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    oh_sn2_state,
    serve_mini_processed,
    serve_settings,
    sn2_state,
    synthetic_context,
    synthetic_loader,
    synthetic_root,
    synthetic_runtime,
    synthetic_settings,
)
from tests.test_serve_fixtures_curation import (  # noqa: F401 (fixtures)
    curation_engine,
    meo_corrected_network,
    meo_sn2_network,
    mini_pairs,
    oh_sn2_network,
)

pytestmark = pytest.mark.serve


def _curate(engine, network, action="validate", **kw) -> int:
    req = CurateIn(
        job_id=network.network_id, pathway_id=network.pathways[0].id, action=action, **kw
    )
    with db.session(engine) as s:
        return create_mechanism(s, network, network.request.model_dump(), req, req.job_id).id


@pytest.fixture
def export_settings(serve_settings, serve_mini_processed):  # noqa: F811
    """Settings on the tmp data root whose processed/ holds the mini PMechDB parquet."""
    s = serve_settings()
    assert s.processed == serve_mini_processed
    return s


@pytest.fixture
def export(export_settings, curation_engine, serve_mini_processed):  # noqa: F811
    """export(pairs=True) -> the report; pairs=False exports without PMechDB."""
    pairs = mini_pairs(serve_mini_processed)

    def run(with_pairs: bool = True, statuses=None) -> dict:
        ctx = LocalJobContext("j" + "e" * 16, export_settings, None)
        provider = (lambda c: pairs) if with_pairs else (lambda c: None)
        request = {} if statuses is None else {"statuses": statuses}
        report = run_export(ctx, request, engine=curation_engine, pairs=provider)
        report["_skips"] = dict(ctx.skipped)
        return report

    run.settings = export_settings
    return run


@pytest.fixture
def two_mechanisms(curation_engine, oh_sn2_network, meo_sn2_network):  # noqa: F811
    return _curate(curation_engine, oh_sn2_network), _curate(curation_engine, meo_sn2_network)


def _out(s) -> Path:
    return s.processed / "user_curated"


def test_export_parquet(export, export_settings, two_mechanisms, serve_mini_processed):  # noqa: F811
    oh, meo = two_mechanisms
    report = export()
    ExportReportOut.model_validate(report)
    assert report["n_mechanisms"] == 2 and report["n_steps_considered"] == 2
    assert report["n_steps_written"] == 1 and report["duplicates_pmechdb"] == 1
    assert report["skipped"] == {"DUPLICATE_OF_PMECHDB": 1} and report["ingest_rejections"] == {}
    assert report["_skips"] == {"DUPLICATE_OF_PMECHDB": 1}
    mini_sn2 = mini_pairs(serve_mini_processed)[("CCl.[OH-]", "CO.[Cl-]")]
    by_mid = {e["mechanism_id"]: e for e in report["steps"]}
    assert by_mid[oh]["code"] == "DUPLICATE_OF_PMECHDB" and by_mid[oh]["detail"] == mini_sn2
    assert by_mid[meo]["code"] is None and by_mid[meo]["step_id"].startswith("user_curated-")
    assert by_mid[meo]["provenance"]["predictor"] == "stub"
    raw = pq.read_table(export_settings.export_raw)
    assert raw.schema.equals(STEP_SCHEMA) and raw.num_rows == 1
    (step,) = read_steps(_out(export_settings) / "steps.parquet")
    assert step.source == Source.USER_CURATED and step.step_id == by_mid[meo]["step_id"]
    assert step.pathway_id == f"user_curated-path-{meo}" and step.step_index == 0
    assert step.conditions == "water, 25 C" and step.mechanism_class is None
    for name in ("rejections.csv", "ingest_report.json", "pmechdb_arrow_codes.csv"):
        assert (_out(export_settings) / name).is_file()
    saved = json.loads((_out(export_settings) / "export_report.json").read_text())
    assert saved["sha256"] == report["sha256"] and saved["export_id"] == report["export_id"]
    assert len(read_steps(serve_mini_processed / "pmechdb" / "steps.parquet")) == 5  # untouched


def test_reingest_aligns_and_round_trips(export, export_settings, two_mechanisms, synthetic_loader):  # noqa: F811
    from ksmech.arrows.align import align_step
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.arrows.roundtrip import predicted_round_trip_step

    export(with_pairs=False)  # both SN2 steps are written
    result = user_curated.load(export_settings.export_raw)
    written = read_steps(_out(export_settings) / "steps.parquet")
    assert not result.rejections and len(result.steps) == 2
    assert [s.to_dict() for s in result.steps] == [s.to_dict() for s in written]
    for step in result.steps:
        rt = ReactantTokens.from_smiles(step.reactants_smiles, synthetic_loader)
        al = align_step(step, rt)
        assert al.complete, al.unaligned
        trip = predicted_round_trip_step(step, rt, al)
        assert trip.ok, (trip.status, trip.pattern, trip.detail)


def test_train_only(export, export_settings, two_mechanisms, serve_mini_processed):  # noqa: F811
    from ksmech.arrows.steps import split_ids

    export(with_pairs=False)
    curated = read_steps(_out(export_settings) / "steps.parquet")
    mixed = read_steps(serve_mini_processed / "pmechdb" / "steps.parquet") + curated
    user_ids = {s.step_id for s in curated}
    for strategy in ("random", "scaffold"):
        for seed in range(5):
            ids = split_ids(mixed, strategy, seed=seed)
            for name, members in ids.items():
                if name.endswith(("_val", "_test")):
                    assert not user_ids & set(members), (strategy, seed, name)
            assert user_ids <= set(ids[f"{strategy}_train"])


def test_csv_reparses_through_pmechdb(export, export_settings, two_mechanisms):  # noqa: F811
    export(with_pairs=False)
    path = _out(export_settings) / "pmechdb_arrow_codes.csv"
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and all(line.startswith('"') and line.endswith('"') for line in lines)
    parsed = pmechdb.load(path)
    assert parsed.n_records == 2 and not parsed.rejections
    written = {s.step_id: s for s in read_steps(_out(export_settings) / "steps.parquet")}
    for step in parsed.steps:
        assert "[H" not in step.reactants_smiles  # compacted: no arrow names an H
    want = sorted(pmechdb.format_arrow_code(s.arrows) for s in written.values())
    assert sorted(pmechdb.format_arrow_code(s.arrows) for s in parsed.steps) == want


def test_second_export_is_byte_identical(export, export_settings, two_mechanisms):  # noqa: F811
    first = export()
    files = ("steps.parquet", "pmechdb_arrow_codes.csv", "rejections.csv", "ingest_report.json")
    before = {f: (_out(export_settings) / f).read_bytes() for f in files}
    raw = export_settings.export_raw.read_bytes()
    second = export()
    assert second["sha256"] == first["sha256"] and second["export_id"] != first["export_id"]
    assert {f: (_out(export_settings) / f).read_bytes() for f in files} == before
    assert export_settings.export_raw.read_bytes() == raw
    assert second["changed_ids"] == 0 and second["withdrawn"] == 0


def test_reject_reports_withdrawn(export, curation_engine, two_mechanisms):  # noqa: F811
    _, meo = two_mechanisms
    first = export()
    with db.session(curation_engine) as s:
        rows = list(s.exec(select(db.ExportedStep)).all())
        assert [(r.mechanism_id, r.step_index) for r in rows] == [(meo, 0)]
        assert rows[0].first_export_id == rows[0].last_export_id == first["export_id"]
        set_status(s, s.get(db.Mechanism, meo), "rejected")
    second = export()
    assert second["withdrawn"] == 1 and second["n_steps_written"] == 0
    assert second["withdrawn_steps"][0]["mechanism_id"] == meo
    with db.session(curation_engine) as s:
        records = list(s.exec(select(db.ExportRecord)).all())
    assert {r.export_id for r in records} == {first["export_id"], second["export_id"]}


def test_duplicates(export, curation_engine, meo_sn2_network, meo_corrected_network):  # noqa: F811
    first = _curate(curation_engine, meo_sn2_network)
    corrected = _curate(curation_engine, meo_corrected_network)  # same pool keys: a duplicate
    broken = _curate(curation_engine, meo_sn2_network)
    with db.session(curation_engine) as s:
        step = s.get(db.CuratedStep, (broken, 0))
        doc = json.loads(step.arrows_json) | {"dataset_arrows": None, "encode_reason": "LEFTOVER"}
        step.arrows_json = json.dumps(doc)
    report = export()
    by_mid = {e["mechanism_id"]: e for e in report["steps"]}
    assert by_mid[first]["code"] is None
    assert by_mid[corrected]["code"] == "DUPLICATE_IN_EXPORT"
    assert by_mid[corrected]["detail"] == f"({first}, 0)"
    assert by_mid[broken] | {"provenance": None} == {
        "mechanism_id": broken,
        "step_index": 0,
        "step_id": None,
        "code": "ARROW_ENCODING_FAILED",
        "detail": "LEFTOVER",
        "provenance": None,
    }
    assert report["duplicates_in_export"] == 1 and report["n_steps_written"] == 1


def test_statuses_filter_and_corrected_export(export, curation_engine, meo_corrected_network):  # noqa: F811
    mid = _curate(curation_engine, meo_corrected_network)
    with db.session(curation_engine) as s:
        assert s.get(db.Mechanism, mid).status == "corrected"
    assert export(statuses=["validated"])["n_steps_considered"] == 0
    report = export(statuses=["corrected"])
    assert report["n_steps_written"] == 1 and report["statuses"] == ["corrected"]
    assert report["steps"][0]["provenance"]["edge_kind"] == "corrected"
    saved = _curate(curation_engine, meo_corrected_network, action="save")
    assert all(e["mechanism_id"] != saved for e in export()["steps"])  # predicted: not exported


def test_without_pmechdb_warns(export, two_mechanisms):  # noqa: F811
    report = export(with_pairs=False)
    assert report["warnings"][0].startswith("PMECHDB_UNAVAILABLE")
    assert report["duplicates_pmechdb"] == 0 and report["n_steps_written"] == 2


def test_compact_step(export, export_settings, two_mechanisms):  # noqa: F811
    export(with_pairs=False)
    for step in read_steps(_out(export_settings) / "steps.parquet"):
        small = compact_step(step)
        assert small is not None and small.step_id == step.step_id
        assert small.arrows == step.arrows and small.pathway_id == step.pathway_id
        assert small.reactants_smiles.count("[H") == 0 < step.reactants_smiles.count("[H")
        assert len(small.reactants_smiles) < len(step.reactants_smiles)


def test_compact_keeps_arrow_hydrogens(mini_result):
    proton = mini_result.steps[1]  # proton transfer: the arrows name H 20
    small = compact_step(proton)
    assert small is not None and "[H:20]" in small.reactants_smiles


def test_export_refuses_the_pmechdb_dir(export_settings, curation_engine, monkeypatch):  # noqa: F811
    """`assert_writable` guards every export write: processed/pmechdb stays read-only."""
    from ksmech.serve import export as export_mod
    from ksmech.serve.settings import assert_writable

    with pytest.raises(ValueError, match="read-only"):
        assert_writable(export_settings.processed / "pmechdb" / "x.parquet", export_settings)
    monkeypatch.setattr(export_mod, "REPORT_FILE", "../pmechdb/export_report.json")
    ctx = LocalJobContext("j" + "f" * 16, export_settings, None)
    with pytest.raises(ValueError, match="read-only"):
        run_export(ctx, {}, engine=curation_engine, pairs=lambda c: None)


def test_ingest_rejection_is_reported(export, curation_engine, meo_sn2_network):  # noqa: F811
    mid = _curate(curation_engine, meo_sn2_network)
    with db.session(curation_engine) as s:
        step = s.get(db.CuratedStep, (mid, 0))
        doc = json.loads(step.arrows_json)
        doc["dataset_arrows"] = doc["dataset_arrows"][:1]  # the C-Cl heterolysis is missing
        doc["pmechdb_code"] = None
        step.arrows_json = json.dumps(doc)
    report = export()
    assert report["n_steps_written"] == 0
    assert report["ingest_rejections"] == {"ARROWS_MISMATCH": 1}  # pentavalent C
    (entry,) = report["steps"]
    assert entry["code"] in report["ingest_rejections"] and entry["step_id"]
    assert report["_skips"] == report["ingest_rejections"]
    assert len(read_steps(_out(export.settings) / "steps.parquet")) == 0


def _truncate(engine, mid: int) -> None:
    """Drop the C-Cl heterolysis from the step's dataset arrows: the ingest rejects it."""
    with db.session(engine) as s:
        step = s.get(db.CuratedStep, (mid, 0))
        doc = json.loads(step.arrows_json)
        doc["dataset_arrows"] = doc["dataset_arrows"][:1]
        doc["pmechdb_code"] = None
        step.arrows_json = json.dumps(doc)


def test_rejected_winner_does_not_block_a_valid_duplicate(export, curation_engine, meo_sn2_network):  # noqa: F811
    """The first copy of a step fails the ingest: the second, valid copy (same pool keys) is
    written, not skipped as DUPLICATE_IN_EXPORT, and the first is reported as rejected."""
    broken = _curate(curation_engine, meo_sn2_network)
    valid = _curate(curation_engine, meo_sn2_network)
    _truncate(curation_engine, broken)
    report = export()
    assert report["n_steps_written"] == 1 and report["duplicates_in_export"] == 0
    assert report["ingest_rejections"] == {"ARROWS_MISMATCH": 1}
    assert report["skipped"] == {} and report["_skips"] == {"ARROWS_MISMATCH": 1}
    by_mid = {e["mechanism_id"]: e for e in report["steps"]}
    assert by_mid[broken]["code"] == "ARROWS_MISMATCH"
    assert by_mid[valid]["code"] is None
    (step,) = read_steps(_out(export.settings) / "steps.parquet")
    assert step.step_id == by_mid[valid]["step_id"]
    assert step.pathway_id == f"user_curated-path-{valid}"
    with db.session(curation_engine) as s:
        (ledger,) = s.exec(select(db.ExportedStep)).all()
    assert (ledger.mechanism_id, ledger.step_index) == (valid, 0)


def test_identical_rejected_copies(export, curation_engine, meo_sn2_network):  # noqa: F811
    """Two identical copies the ingest rejects: the first is reported with its ingest reason,
    the second (the same record text) as DUPLICATE_IN_EXPORT of it; a later valid copy is
    still written."""
    first, second, valid = (_curate(curation_engine, meo_sn2_network) for _ in range(3))
    _truncate(curation_engine, first)
    _truncate(curation_engine, second)
    report = export()
    by_mid = {e["mechanism_id"]: e for e in report["steps"]}
    assert by_mid[first]["code"] == "ARROWS_MISMATCH" and by_mid[first]["step_id"]
    assert by_mid[second]["code"] == "DUPLICATE_IN_EXPORT"
    assert by_mid[second]["detail"] == f"({first}, 0)" and by_mid[second]["step_id"] is None
    assert by_mid[valid]["code"] is None
    assert report["ingest_rejections"] == {"ARROWS_MISMATCH": 1}
    assert report["n_steps_written"] == 1 and report["duplicates_in_export"] == 1
    assert pq.read_table(export.settings.export_raw).num_rows == 2


def test_compact_fallback_is_counted(export, two_mechanisms, monkeypatch):  # noqa: F811
    from ksmech.serve import export as export_mod

    monkeypatch.setattr(export_mod, "compact_step", lambda step: None)
    report = export(with_pairs=False)
    assert report["skipped"] == {"COMPACT_FALLBACK": 2} and report["n_steps_written"] == 2
    assert report["_skips"] == {"COMPACT_FALLBACK": 2}
    assert any(w.startswith("COMPACT_FALLBACK") for w in report["warnings"])
    lines = (_out(export.settings) / "pmechdb_arrow_codes.csv").read_text().splitlines()
    assert len(lines) == 2 and all("[H:" in line for line in lines)  # explicit H kept
    assert not pmechdb.load(_out(export.settings) / "pmechdb_arrow_codes.csv").rejections
