"""Real data: P3's open-shell admission on PMechDB against the hand reading (open-shell plan P3).

The laws run on every stored step from the sides the source drew (`test_data_domain`'s
rescreen). The hand reading (`tests/fixtures/pmechdb_radical_survey.json`) is the oracle: the
admitted set is its 12 genuine steps but the differences listed in KNOWN_DIFFERENCES, each
reported to the user, and every other radical drawing carries a counted reason. After the P3
re-ingest and sample build (the resonance pairs before it copied to `p3_reference/`), the new
resonance pairs and the stretched-bond flag are checked on the written files.
"""

import json
from pathlib import Path

import pytest

from ksmech.data import domain
from ksmech.data.domain import DomainCode
from ksmech.data.io import DEFAULT_PROCESSED_DIR

pytestmark = pytest.mark.realdata

PM = DEFAULT_PROCESSED_DIR / "pmechdb"
ORACLE = Path("tests/fixtures/pmechdb_radical_survey.json")
# the species table (v1 and v2) gives the Et3B-O2 adduct u = 0 (A: the singlet 0.48 eV lower,
# B undetermined), so its step turns triplet O2 into a singlet: law (u) holds it back (plan P3,
# step 3 (b))
KNOWN_DIFFERENCES = {"pmechdb-5d007b38296827b6": DomainCode.SPIN_NOT_CONSERVED}
RESONANCE_PAIRS_GAINED = {"pmechdb-db877060093d34fc", "pmechdb-dcf0a54a98432a8c"}
# P4's: the two CO drawings (CARBENE_SINGLET) and two each of OSM_CU2 and OSM_FE3
P4_RESONANCE_CLASSES = {"CARBENE_SINGLET": 2, "OSM_CU2": 2, "OSM_FE3": 2}
BIRCH = "pmechdb-4bda5440edfee13c"


@pytest.fixture(scope="module")
def steps():
    from tests.test_data_domain import _rescreened_store

    return {s.step_id: s for s in _rescreened_store("pmechdb")}


RADICAL = {str(domain.DomainClass.RADICAL_SPECTATOR), str(domain.DomainClass.RADICAL_REACTIVE)}


def _admitted(step) -> bool:
    """Admitted as a radical (P3's classes; P4's paired carbenes and metals are not radicals)."""
    classes = domain.domain_classes(step.domain_class)
    return step.domain_exclusion is None and bool(classes & RADICAL)


def test_the_admitted_set_is_the_hand_reading_but_the_known_differences(steps):
    oracle = {r["step_id"] for r in json.loads(ORACLE.read_text())["genuine"]}
    admitted = {i for i, s in steps.items() if _admitted(s)}
    assert admitted == oracle - set(KNOWN_DIFFERENCES)
    for step_id, code in KNOWN_DIFFERENCES.items():
        assert steps[step_id].domain_exclusion == code


def test_every_other_radical_drawing_has_a_counted_reason(steps):
    curated = domain.open_shell_curation("pmechdb")
    for step_id, s in steps.items():
        if s.domain_exclusion == DomainCode.RADICAL_DRAWN:
            assert curated.get(step_id) == s.domain_class, step_id
    codes = {s.domain_exclusion for s in steps.values()}
    assert codes - {None} <= {str(c) for c in domain.PRECEDENCE}


def test_every_curated_step_passes_the_laws(steps):
    for step_id, reason in domain.open_shell_curation("pmechdb").items():
        assert steps[step_id].domain_exclusion == DomainCode.RADICAL_DRAWN
        assert steps[step_id].domain_class == reason  # the laws admitted it, the file did not


def test_every_admitted_step_with_a_u2_species_is_an_exemption(steps):
    """P8's invariant: a non-Hund u >= 2 species (MULTI_RADICAL_SPECIES) only in an exempt
    step; a Hund case (triplet O2) may stand in any admitted step."""
    exempt = {i for src, i in domain.multi_radical_exemptions() if src == "pmechdb"}
    multi = {i for i, s in steps.items() if _admitted(s) and domain.high_spin_species(s)[1]}
    assert multi <= exempt


def _report(name: str) -> dict:
    path = PM / name
    report = json.loads(path.read_text()) if path.exists() else {}
    return report


def test_the_reingest_gains_exactly_the_two_radical_resonance_pairs():
    import pyarrow.parquet as pq

    before = PM / "p3_reference" / "resonance_pairs.parquet"
    if "open_shell" not in _report("ingest_report.json") or not before.exists():
        pytest.skip("run the P3 re-ingest (resonance pairs copied to p3_reference/ first)")
    ids = lambda p: set(pq.read_table(p, columns=["step_id"]).column(0).to_pylist())  # noqa: E731
    old, new = ids(before), ids(PM / "resonance_pairs.parquet")
    p4 = PM / "p4_reference" / "resonance_pairs.parquet"
    later = new - ids(p4) if p4.exists() else set()  # the pairs P4 gained, checked below
    assert old <= new and new - old - later == RESONANCE_PAIRS_GAINED


def test_the_p4_reingest_gains_the_carbene_and_metal_resonance_pairs(steps):
    import pyarrow.parquet as pq

    before = PM / "p4_reference" / "resonance_pairs.parquet"
    if "metal_table" not in _report("ingest_report.json").get("spin", {}) or not before.exists():
        pytest.skip("run the P4 re-ingest (resonance pairs copied to p4_reference/ first)")
    ids = lambda p: set(pq.read_table(p, columns=["step_id"]).column(0).to_pylist())  # noqa: E731
    gained = ids(PM / "resonance_pairs.parquet") - ids(before)
    from collections import Counter

    assert dict(Counter(steps[i].domain_class for i in gained)) == P4_RESONANCE_CLASSES


def test_the_birch_step_is_flagged_and_no_admitted_pool_is_smeared():
    report = _report("samples_report.json")
    if "SKIPPED_BOND_STRETCHED" not in report:
        pytest.skip("run ksmech-arrows samples --source pmechdb after the P3 re-ingest")
    assert BIRCH in report["SKIPPED_BOND_STRETCHED"]
    assert "SOMO_NOT_INTEGRAL" not in report["skipped"]
