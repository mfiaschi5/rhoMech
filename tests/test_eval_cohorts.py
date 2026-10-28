"""Cohorts, slices and the covered fraction of `ksmech-eval floor|decode` (`eval.cohorts`)."""

import json
from types import SimpleNamespace

import pytest

from ksmech.data import provenance as prov
from ksmech.eval import cohorts as co
from ksmech.eval.cli import build_parser


def test_a_cohort_is_pinned_once_and_checked(tmp_path):
    out = co.write_cohort("pmechdb", "closed_base", ["b", "a", "a"], {"why": "test"}, tmp_path)
    assert prov.read_header(out).inputs == {}
    assert co.load_cohort("pmechdb", "closed_base", tmp_path) == {"a", "b"}
    assert co.load_cohort("pmechdb", "all", tmp_path) is None
    with pytest.raises(FileExistsError):
        co.write_cohort("pmechdb", "closed_base", ["c"], None, tmp_path)
    body = json.loads(out.read_text())
    body["ids"].append("c")
    out.write_text(json.dumps(body))
    with pytest.raises(prov.StaleArtifact, match="content hash"):
        co.load_cohort("pmechdb", "closed_base", tmp_path)


def test_restrict_keeps_the_view_order():
    assert co.restrict(["c", "a", "b"], {"a", "c"}) == ["c", "a"]
    assert co.restrict(["c", "a"], None) == ["c", "a"]


def test_a_slice_reports_with_and_without():
    samples = {k: SimpleNamespace(fermi_smeared=k == "s") for k in "abs"}
    rows = [
        {"step_id": "a", "rank": 0},
        {"step_id": "b", "rank": None},
        {"step_id": "s", "rank": 2},
    ]
    got = co.sliced(rows, samples, "fermi_smeared")
    assert got["with"] == {"top1": 0.0, "top3": 1.0, "top5": 1.0, "n": 1}
    assert got["without"]["n"] == 2 and got["without"]["top1"] == 0.5


def test_the_covered_fraction_reads_the_view_counts(tmp_path):
    view = tmp_path / "samples_molecule_val.json"
    report = {"n_steps": 1246, "n_samples": 1239, "skipped": {"MISSING_TOKENS": 7}}
    view.write_text(json.dumps({"ids": [], "report": report}))
    got = co.covered_fraction(view)
    assert got["MISSING_TOKENS"] == 7 and got["covered"] == pytest.approx(1239 / 1246)


def test_floor_and_decode_take_a_cohort_and_a_slice():
    args = build_parser().parse_args(
        ["floor", "--source", "pmechdb", "--cohort", "closed_base", "--slice", "fermi_smeared"]
    )
    assert (args.cohort, args.slice) == ("closed_base", "fermi_smeared")
    assert build_parser().parse_args(["decode", "--run", "x"]).cohort == "all"
