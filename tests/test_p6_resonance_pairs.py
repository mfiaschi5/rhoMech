"""Real data: every stored resonance pair of PMechDB and ReactMech stays resonance under the
electron-flow law and the identity key v2 (open-shell plan P6 step 2; v17 counts 471 and 548,
478 and 548 since P1 and P3)."""

import pandas as pd
import pytest

from ksmech.data import chem
from ksmech.data.io import DEFAULT_PROCESSED_DIR
from ksmech.data.resonance import StepKind, balanced_flow, classify_step, mapped_changes

pytestmark = pytest.mark.realdata


@pytest.mark.parametrize("source, at_least", [("pmechdb", 471), ("reactmech", 548)])
def test_the_resonance_pairs_stay_resonance(source, at_least):
    path = DEFAULT_PROCESSED_DIR / source / "resonance_pairs.parquet"
    if not path.is_file():
        pytest.skip(f"no {path}")
    pairs = pd.read_parquet(path)
    assert len(pairs) >= at_least
    kinds, unbalanced = {}, []
    for row in pairs.itertuples():
        r, p = chem.parse_mapped(row.reactants_smiles), chem.parse_mapped(row.products_smiles)
        kinds[row.step_id] = classify_step(r, p)
        if not balanced_flow(*mapped_changes(r, p)):
            unbalanced.append(row.step_id)
    assert {i: k for i, k in kinds.items() if k is not StepKind.RESONANCE} == {}
    assert unbalanced == []  # the law holds as drawn, with no enumeration
