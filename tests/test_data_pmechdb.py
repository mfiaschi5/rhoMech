from pathlib import Path

import pytest

from ksmech.data import pmechdb
from ksmech.data.schema import Arrow, SinkKind, SourceKind
from ksmech.data.validation import RejectReason, StepRejected, make_step_id
from tests.conftest import MINI_PMECHDB

LP, BOND = SourceKind.LONE_PAIR, SourceKind.BOND
ATOM, SBOND = SinkKind.ATOM, SinkKind.BOND
REAL_RAW = Path("data/raw/pmechrp")


# --- arrow code -------------------------------------------------------------------------------


def test_parse_arrow_code_all_shapes():
    assert pmechdb.parse_arrow_code("10=20;20,21=21,22;22,23=23") == [
        Arrow(LP, (10,), ATOM, (20,)),
        Arrow(BOND, (20, 21), SBOND, (21, 22)),
        Arrow(BOND, (22, 23), ATOM, (23,)),
    ]


def test_stray_trailing_comma_is_dropped():
    assert pmechdb.parse_arrow_code("10=20,") == [Arrow(LP, (10,), ATOM, (20,))]


@pytest.mark.parametrize("code", ["", "10-20", "10=20=30", "a=20", "10=20;;20,21=21", "1,2,3=4"])
def test_bad_arrow_grammar(code):
    with pytest.raises(StepRejected) as exc:
        pmechdb.parse_arrow_code(code)
    assert exc.value.reason is RejectReason.ARROW_PARSE


def test_meaningless_arrow_shape():
    with pytest.raises(StepRejected) as exc:
        pmechdb.parse_arrow_code("20,21=22")
    assert exc.value.reason is RejectReason.ARROW_INVALID


# --- mini fixture -----------------------------------------------------------------------------


def test_mini_fixture_counts(mini_result):
    assert mini_result.n_records == 6
    assert len(mini_result.steps) == 5
    assert [r.reason for r in mini_result.rejections] == [RejectReason.UNBALANCED]
    assert mini_result.notes["stray_trailing_comma"] == 1


def test_mini_fixture_arrows(mini_steps):
    assert mini_steps["sn2"].arrows == [
        Arrow(LP, (10,), ATOM, (20,)),
        Arrow(BOND, (20, 21), ATOM, (21,)),
    ]
    assert [a.destination() for a in mini_steps["e2"].arrows] == [
        ("bond", (10, 20)),
        ("bond", (21, 22)),
        ("lone_pair", (23,)),
    ]


def test_canonical_smiles_keep_maps_and_mapped_h(mini_steps):
    pt = mini_steps["proton_transfer"]
    assert "[H:20]" in pt.reactants_smiles and "[H:20]" in pt.products_smiles
    for step in mini_steps.values():
        for smi in (step.reactants_smiles, step.products_smiles):
            assert ":10]" in smi and ":20]" in smi


def test_step_ids_are_content_hashes(mini_result):
    lines = MINI_PMECHDB.read_text().splitlines()
    expected = [make_step_id("pmechdb", line.strip('"')) for line in lines]
    got = [s.step_id for s in mini_result.steps] + [r.step_id for r in mini_result.rejections]
    assert sorted(got) == sorted(expected)
    assert len(set(got)) == 6


# --- rejection reasons ------------------------------------------------------------------------

SN2 = "[OH-:10].[CH3:20][Cl:21]>>[OH:10][CH3:20].[Cl-:21]"


@pytest.mark.parametrize(
    "record, reason",
    [
        (SN2 + " 10=20;20,21=21 extra", RejectReason.MALFORMED_RECORD),
        ("[OH-:10].[CH3:20][Cl:21] 10=20", RejectReason.MALFORMED_RECORD),
        ("[OH-:10].[CH3:20][Cl:21]>>[OH:10][CH3:20].[Cl:21] 10=20", RejectReason.CHARGE_UNBALANCED),
        ("[OH-:10].[CH3:20][Cl:21]>>[OH:10][CH2:20].[Cl-:21] 10=20", RejectReason.H_UNBALANCED),
        (
            "[OH-:10].[CH3:20][Cl:21]>>[OH:10][CH3:20].[Cl-:22] 10=20",
            RejectReason.MAP_NOT_BIJECTIVE,
        ),
        ("[OH-:10].[CH3:10][Cl:21]>>[OH:10][CH3:10].[Cl-:21] 10=21", RejectReason.MAP_DUPLICATE),
        (SN2 + " 10=20;20,21=21;30=20", RejectReason.ARROW_ATOM_UNMAPPED),
        (SN2 + " 10=20;10,21=21", RejectReason.BOND_NOT_FOUND),
        ("[OH-:10].[CH3:20][Cl:21]>>[OH:10][CH3:20].[Cl-:21 10=20", RejectReason.SMILES_PARSE),
        ("[OH-:10].[CH5:20][Cl:21]>>[OH:10][CH3:20].[Cl-:21] 10=20", RejectReason.SANITIZE_FAILED),
    ],
)
def test_reject_reasons(record, reason):
    with pytest.raises(StepRejected) as exc:
        pmechdb.parse_record(record, "id")
    assert exc.value.reason is reason


def test_duplicate_record_rejected(tmp_path):
    line = MINI_PMECHDB.read_text().splitlines()[0]
    path = tmp_path / "dup.txt"
    path.write_text(f"{line}\n{line}\n")
    result = pmechdb.load(path)
    assert len(result.steps) == 1
    assert [r.reason for r in result.rejections] == [RejectReason.DUPLICATE_STEP]


def test_multi_field_csv_row_rejected(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text(f'"{SN2} 10=20",unexpected\n')
    result = pmechdb.load(path)
    assert [r.reason for r in result.rejections] == [RejectReason.MALFORMED_RECORD]


def test_missing_raw_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        pmechdb.resolve_inputs(tmp_path)


# --- real data --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_result():
    if not REAL_RAW.is_dir():
        pytest.skip("PMechRP raw files not in data/raw/pmechrp")
    return pmechdb.load(REAL_RAW)


@pytest.mark.realdata
def test_real_pmechdb_rejection_rate_below_2_percent(real_result):
    assert real_result.n_records == 13104
    assert real_result.rejected_fraction < 0.02


@pytest.mark.realdata
def test_real_pmechdb_pathways(real_result):
    """Measured 2026-10-19: 2,379 pathways hold 70.8% of steps (docs/decisions.md)."""
    report = real_result.report()["pathways"]
    assert report["fraction_in_pathways"] > 0.5 and report["median_pathway_size"] >= 2
    over_cap = sum(v for k, v in report["pathway_sizes"].items() if int(k) > 15)
    assert over_cap == report["n_long_chains_kept"]  # only plain chains may exceed the cap
    steps = [s for s in real_result.steps if s.pathway_id is not None]
    assert len(steps) == report["n_steps_in_pathways"]
    assert all(s.step_index is not None and s.pathway_id.startswith("pmechdb-path-") for s in steps)
