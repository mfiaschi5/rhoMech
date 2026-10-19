"""ReactMech and mech-USPTO-31K loaders, written against the published formats.

The real-file tests are skipped until the raw files sit in data/raw/. The format-level tests use
only strings quoted in the papers or READMEs.
"""

import pytest

from ksmech.data import mech_uspto, reactmech
from ksmech.data.schema import Arrow, SinkKind, SourceKind
from ksmech.data.validation import RejectReason, StepRejected

REACTMECH_MISSING = not reactmech.RAW_DIR.is_dir()
MECH_USPTO_MISSING = not mech_uspto.RAW_FILE.is_file()


def test_modules_flag_untested():
    for mod in (reactmech, mech_uspto):
        assert "untested_on_real_data=True" in mod.__doc__
        assert mod.UNTESTED_ON_REAL_DATA is True


# --- ReactMech --------------------------------------------------------------------------------


@pytest.mark.realdata
@pytest.mark.skipif(
    REACTMECH_MISSING,
    reason="ReactMech raw files not in data/raw/reactmech (untested_on_real_data); download "
    "needs approval",
)
def test_reactmech_real_files():
    result = reactmech.load()
    assert result.n_records > 0
    assert result.rejected_fraction < 0.02


def test_reactmech_line_without_arrows():
    step = reactmech.parse_line("[OH-:1].[CH3:2][Cl:3]>>[OH:1][CH3:2].[Cl-:3]", "reactmech-x")
    assert step.arrows == []
    assert step.mechanism_class is None


def test_reactmech_still_validates_balance():
    with pytest.raises(StepRejected) as exc:
        reactmech.parse_line("[OH-:1].[CH3:2][Cl:3]>>[OH:1][CH2:2]C.[Cl-:3]", "reactmech-x")
    assert exc.value.reason is RejectReason.UNBALANCED


# --- mech-USPTO-31K ---------------------------------------------------------------------------

README_EXAMPLE = "[(26, 1), ([1, 9], 9), (9, [9, 1]), ([1, 101], 101)]"


@pytest.mark.realdata
@pytest.mark.skipif(
    MECH_USPTO_MISSING,
    reason="mech-USPTO-31k.csv not in data/raw/mech_uspto (untested_on_real_data); download "
    "needs approval",
)
def test_mech_uspto_real_file():
    mechanisms, result = mech_uspto.read_mechanisms()
    assert mechanisms and result.rejected_fraction < 0.02


def test_parse_readme_label():
    labels = mech_uspto.parse_mechanistic_label(README_EXAMPLE)
    assert [mech_uspto.to_arrow(x) for x in labels] == [
        Arrow(SourceKind.LONE_PAIR, (26,), SinkKind.ATOM, (1,)),
        Arrow(SourceKind.BOND, (1, 9), SinkKind.ATOM, (9,)),
        Arrow(SourceKind.LONE_PAIR, (9,), SinkKind.BOND, (1, 9)),
        Arrow(SourceKind.BOND, (1, 101), SinkKind.ATOM, (101,)),
    ]


def test_hydrogen_reference():
    (label,) = mech_uspto.parse_mechanistic_label("[(7, 5.1)]")
    assert label.sink == mech_uspto.HydrogenRef(parent_map=5, k=1)
    with pytest.raises(ValueError, match="implicit H"):
        mech_uspto.to_arrow(label)


@pytest.mark.parametrize("text", ["not python", "[]", "[(1, 2, 3)]", "[(1, 'a')]", "[(1, 5.0)]"])
def test_bad_labels(text):
    with pytest.raises(StepRejected) as exc:
        mech_uspto.parse_mechanistic_label(text)
    assert exc.value.reason is RejectReason.ARROW_PARSE


def test_read_mechanisms_checks_header(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("rxn,label\nA>>B,[(1, 2)]\n")
    with pytest.raises(ValueError, match="missing columns"):
        mech_uspto.read_mechanisms(path)


def test_read_mechanisms_published_header(tmp_path):
    path = tmp_path / "mech.csv"
    path.write_text(
        "original_reactions,updated_reaction,mechanistic_class,mechanistic_label\n"
        'X,[OH-:1].[CH3:2][Cl:3]>>[OH:1][CH3:2].[Cl-:3],sn2,"[(1, 2), ([2, 3], 3)]"\n'
        'X,A>>B,sn2,"oops"\n'
    )
    mechanisms, result = mech_uspto.read_mechanisms(path)
    assert len(mechanisms) == 1 and mechanisms[0].mechanistic_class == "sn2"
    assert [r.reason for r in result.rejections] == [RejectReason.ARROW_PARSE]


def test_mech_uspto_load_not_implemented():
    with pytest.raises(NotImplementedError, match="arrow replay"):
        mech_uspto.load()
