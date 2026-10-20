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


@pytest.mark.parametrize(
    "record, detail",
    [
        # the leaving-group arrow is missing: carbon would take five bonds
        (SN2 + " 10=20", "VALENCE"),
        # hydroxide attacks the proton instead of the methyl cation the products show
        ("[OH-:10].[H+:20].[CH3+:30]>>[OH:10][CH3:30].[H+:20] 10=20", "MISMATCH bond:"),
    ],
)
def test_arrows_that_miss_the_products_are_rejected(record, detail):
    with pytest.raises(StepRejected) as exc:
        pmechdb.parse_record(record, "id")
    assert exc.value.reason is RejectReason.ARROWS_MISMATCH
    assert exc.value.detail.startswith(detail)


def test_arrows_matching_only_modulo_resonance_are_accepted():
    """Styrene + H+: the curator writes the product ring in the other Kekulé form than the
    edit gives. Exact SMILES differ, resonance keys agree, so the step is accepted."""
    record = (
        "[CH2:1]=[CH:2][C:3]1=[CH:4][CH:5]=[CH:6][CH:7]=[CH:8]1.[H+:9]>>"
        "[H:9][CH2:1][CH+:2][C:3]1=[CH:8][CH:7]=[CH:6][CH:5]=[CH:4]1 1,2=1,9"
    )
    step = pmechdb.parse_record(record, "id")
    assert "[H:9]" in step.products_smiles and not step.resonance_step


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


@pytest.mark.realdata
@pytest.mark.skipif(not REAL_RAW.is_dir(), reason="PMechRP raw files not in data/raw/pmechrp")
def test_real_pmechdb_rejection_rate_below_2_percent():
    result = pmechdb.load(REAL_RAW)
    assert result.n_records == 13104
    assert result.rejected_fraction < 0.02
