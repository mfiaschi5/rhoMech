"""Chemistry validation on the ten fixture molecules (slow: runs GFN2 optimizations)."""

import numpy as np
import pytest

from ksmech.qm import validation
from tests.fixtures.molecules import MOLECULES

pytestmark = pytest.mark.slow


def test_every_fixture_molecule_featurizes(fixture_records):
    assert list(fixture_records) == list(MOLECULES)
    assert all(rec.warnings == "" for rec in fixture_records.values()), {
        k: v.warnings for k, v in fixture_records.items() if v.warnings
    }


def test_benzene_gap_is_larger_than_acrolein_gap(fixture_records):
    assert validation.gap_ev(fixture_records["benzene"]) > validation.gap_ev(
        fixture_records["acrolein"]
    )


def test_most_negative_charge_in_acetone_is_on_oxygen(fixture_records):
    rec = fixture_records["acetone"]
    assert rec.atomic_numbers[int(np.argmin(rec.charges))] == 8


def test_fukui_minus_is_largest_on_methylamine_nitrogen(fixture_records):
    rec = fixture_records["methylamine"]
    assert rec.atomic_numbers[int(np.argmax(rec.fukui_fmo_minus))] == 7


def test_fukui_plus_is_largest_on_the_carbonyl_carbon_of_acetaldehyde(fixture_records):
    validation.check_acetaldehyde_fukui_plus_on_carbonyl_c(fixture_records)


def test_finite_difference_fukui_for_acetaldehyde_peaks_on_oxygen(fixture_records):
    """Documents the relaxed N+-1 result that motivated the frozen-orbital asserts."""
    rec = fixture_records["acetaldehyde"]
    assert rec.atomic_numbers[int(np.argmax(rec.fukui_plus))] == 8


def test_overlap_matrix_is_symmetric_with_unit_diagonal(fixture_records):
    for rec in fixture_records.values():
        assert np.allclose(rec.S, rec.S.T, atol=1e-12)
        assert np.abs(np.diag(rec.S) - 1.0).max() < validation.UNIT_DIAG_TOL


def test_mo_coefficients_are_s_orthonormal(fixture_records):
    for name, rec in fixture_records.items():
        err = np.abs(rec.C.T @ rec.S @ rec.C - np.eye(rec.C.shape[1])).max()
        assert err < 1e-6, name


def test_all_validation_checks_pass(fixture_records):
    failed = [r for r in validation.run_checks(fixture_records) if not r.passed]
    assert not failed, [(r.name, r.detail) for r in failed]
