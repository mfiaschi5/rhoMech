"""The species table (open-shell plan P2 step 4; spin_table_v2 since 2026-10-31): rule C as a
pure function of A and B, A's bounded states, the table's schema, content hash and theory
digest, and the species the plan names."""

import copy

import pytest

from ksmech.data import spin_table as st

A_TRIPLET = st.ScanVerdict(lowest=2, gap_ev=0.9, conclusive=True)
A_TRIPLET_CLOSE = st.ScanVerdict(lowest=2, gap_ev=0.1, conclusive=False)
A_FAILED = st.ScanVerdict(lowest=None, gap_ev=None, conclusive=False)


def test_agree_assigns_even_inside_the_error_bar():
    assert st.combine(A_TRIPLET, 2) == (2, st.AGREE)
    assert st.combine(A_TRIPLET_CLOSE, 2) == (2, st.AGREE)


def test_b_undetermined_a_decides_only_when_conclusive():
    assert st.combine(A_TRIPLET, None) == (2, st.A_DECIDES)
    assert st.combine(A_TRIPLET_CLOSE, None) == (st.AMBIGUOUS, st.A_INCONCLUSIVE)


def test_every_disagreement_is_ambiguous():
    assert st.combine(A_TRIPLET, 0) == (st.AMBIGUOUS, st.DISAGREE)
    assert st.combine(A_TRIPLET_CLOSE, 0) == (st.AMBIGUOUS, st.DISAGREE)


def test_a_without_a_lowest_state_assigns_nothing():
    assert st.combine(A_FAILED, 2) == (st.AMBIGUOUS, st.A_FAILED)
    assert st.combine(A_FAILED, None) == (st.AMBIGUOUS, st.A_FAILED)


def test_scan_verdict():
    bar = st.ERROR_BARS["organic"]["ev"]
    assert st.scan_verdict({2: 0.0, 0: 0.5}, bar) == st.ScanVerdict(2, 0.5, True)
    assert st.scan_verdict({2: 0.0, 0: 0.2}, bar).conclusive is False
    assert st.scan_verdict({1: -3.0}, bar) == st.ScanVerdict(1, None, True)  # one state: by count
    assert st.scan_verdict({2: 0.0, 0: None}, bar).lowest is None


def test_a_bounded_state_decides_only_as_the_lowest():
    """A bounded energy is an upper bound: the lowest state may be bounded (its true gap is at
    least the measured one), another state may not (its true energy may be lower)."""
    bar = st.ERROR_BARS["organic"]["ev"]
    assert st.scan_verdict({0: 0.0, 2: 1.27}, bar, bounded={0}) == st.ScanVerdict(0, 1.27, True)
    assert st.scan_verdict({0: 0.0, 2: 1.27}, bar, bounded={2}).conclusive is False
    assert st.scan_verdict({0: 0.0, 2: 1.27}, bar, bounded={0, 2}).conclusive is False


def test_error_bars_are_the_published_values():
    assert st.ERROR_BARS["organic"]["ev"] == pytest.approx(0.304, abs=1e-3)
    assert st.ERROR_BARS["transition_metal"]["ev"] == pytest.approx(0.776, abs=1e-3)
    assert all(bar["citation"] for bar in st.ERROR_BARS.values())


def _table() -> tuple[dict, dict]:
    theory = {"digest": "abc"}
    entry = {"unpaired": 2, "case": st.AGREE, "states": {"2": {}, "0": {}}, "a": {}, "b": {}}
    body = {"version": st.VERSION, "schema": st.SCHEMA, "method": "m", "theory_digest": "abc"}
    return st.with_hash(body | {"error_bars": st.ERROR_BARS, "entries": {"[O][O]": entry}}), theory


def test_validate_schema_hash_and_digest():
    table, theory = _table()
    st.validate(table, theory)
    tampered = copy.deepcopy(table)
    tampered["entries"]["[O][O]"]["unpaired"] = 0
    with pytest.raises(st.SpinTableError, match="content_hash"):
        st.validate(tampered, theory)
    with pytest.raises(st.SpinTableError, match="theory_digest"):
        st.validate(table, {"digest": "other"})
    wrong = copy.deepcopy(table)
    wrong["entries"]["[O][O]"]["case"] = st.DISAGREE
    with pytest.raises(st.SpinTableError, match="case"):
        st.validate(st.with_hash(wrong), theory)


@pytest.fixture(scope="module")
def table() -> dict:
    return st.load()


TEXTBOOK_SINGLETS = (  # the dihalocarbenes, the vinylidenes and CO (v1 left three ambiguous)
    "Cl[C]Cl",
    "Br[C]Br",
    "[C]=CC",
    "[C]=C(C)C",
    "[C]=C(c1ccccc1)c1ccccc1",
    "[C]=O",
)


@pytest.mark.parametrize("smiles", TEXTBOOK_SINGLETS)
def test_carbenes_vinylidenes_and_co_are_singlets(table, smiles):
    assert st.lookup(table, smiles) == 0


def test_the_table_is_v2_and_v1_stays_as_p3_read_it():
    import json

    assert st.load()["version"] == st.VERSION == "spin_table_v2"
    v1 = json.loads(st.SPECIES_V1.read_text())
    st.validate(v1, version="spin_table_v1")
    assert st.lookup(v1, "Cl[C]Cl") == st.AMBIGUOUS


def test_triplet_oxygen(table):
    assert st.lookup(table, "[O][O]") == 2


def test_every_entry_records_every_state(table):
    for smiles, entry in table["entries"].items():
        assert entry["states"], smiles
        for u, state in entry["states"].items():
            ok = state["status"] == "ok" and state["energy_ha"] is not None
            assert ok or state["reason"], (smiles, u)


def test_fe_iii_carries_three_states_and_cu_ii_one(table):
    metals = {s: e for s, e in table["entries"].items() if "metal" in e["groups"]}
    assert metals
    for smiles, entry in metals.items():
        expect = {"5", "3", "1"} if "Fe" in smiles else {"1"}
        assert set(entry["states"]) == expect, smiles


def test_untabulated_species_is_none(table):
    assert st.lookup(table, "CC") is None
