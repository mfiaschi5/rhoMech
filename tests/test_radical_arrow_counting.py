"""Radical (half-headed) arrows are counted, never accepted unchecked or crashing: ingest
rejects the step as RADICAL_ARROW, the edit label skips it as RADICAL_ARROW, the arrow round
trip reports it as RADICAL_ARROW (open-shell plan P0; one-electron edits come in P6)."""

import pytest
from rdkit import Chem

from ksmech.arrows.delta_be import EditSkip, dataset_matrix
from ksmech.arrows.roundtrip import RADICAL_ARROW, round_trip_step
from ksmech.arrows.steps import SkipReason
from ksmech.data import chem
from ksmech.data.schema import Arrow, ElementaryStep, SinkKind, SourceKind
from ksmech.data.validation import RejectReason, StepRejected, validate

# methyl + chlorine atom -> chloromethane, drawn with two fishhooks (written by hand)
REACTANTS = "[CH3:1].[Cl:2]"
PRODUCTS = "[CH3:1][Cl:2]"
FISHHOOKS = [
    Arrow(SourceKind.RADICAL, (1,), SinkKind.BOND, (1, 2)),
    Arrow(SourceKind.RADICAL, (2,), SinkKind.BOND, (1, 2)),
]


def _step() -> ElementaryStep:
    return ElementaryStep("t", "pmechdb", None, None, REACTANTS, PRODUCTS, FISHHOOKS)


def test_ingest_rejects_a_radical_arrow_as_its_own_reason():
    with pytest.raises(StepRejected) as exc:
        validate(chem.parse_mapped(REACTANTS), chem.parse_mapped(PRODUCTS), FISHHOOKS)
    assert exc.value.reason is RejectReason.RADICAL_ARROW


def test_the_edit_label_and_the_round_trip_count_it():
    assert dataset_matrix(_step()) is EditSkip.RADICAL_ARROW
    trip = round_trip_step(_step())
    assert trip.status == RADICAL_ARROW and not trip.ok


def test_every_edit_skip_passes_through_the_sample_skip_reasons():
    """The sample builder converts EditSkip values by value: an EditSkip missing from the
    hand-kept SkipReason mirror would raise ValueError on its first step."""
    for skip in EditSkip:
        assert SkipReason(str(skip)) == skip.value


def test_the_fixture_is_a_radical_recombination():
    assert [a.GetNumRadicalElectrons() for a in Chem.MolFromSmiles(REACTANTS).GetAtoms()] == [1, 1]
