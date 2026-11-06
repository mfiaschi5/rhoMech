"""The open-shell fixture list parses, and RDKit gives each drawing the radical count the list
states (the drawings RDKit misreads are kept on purpose: `tests/fixtures/molecules.py`)."""

import pytest
from rdkit import Chem

from tests.fixtures.molecules import MOLECULES, OPEN_SHELL


@pytest.mark.parametrize("name", sorted(OPEN_SHELL))
def test_each_open_shell_fixture_has_its_rdkit_radical_count(name):
    smiles, radicals = OPEN_SHELL[name]
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == radicals


def test_the_closed_shell_set_is_unchanged_and_apart():
    assert len(MOLECULES) == 10 and not set(MOLECULES) & set(OPEN_SHELL)
