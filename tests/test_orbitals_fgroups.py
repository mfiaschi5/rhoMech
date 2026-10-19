"""Ertl functional-group vocabulary: packaged file, building, per-atom ids."""

import pytest
from rdkit import Chem

from ksmech.orbitals.fgroups import (
    NONE_ID,
    NONE_LABEL,
    UNK_ID,
    UNK_LABEL,
    FGVocab,
    build_vocab,
    default_vocab,
    functional_groups,
)


def test_packaged_vocab():
    vocab = default_vocab()
    assert vocab.labels[:2] == [NONE_LABEL, UNK_LABEL]
    assert len(vocab) > 100 and len(set(vocab.labels)) == len(vocab)
    assert vocab.meta["source"] == "pmechdb" and vocab.meta["min_count"] == 5
    assert vocab.id_of("C=O") >= 2 and vocab.id_of("not-a-group") == UNK_ID


def test_functional_groups_on_acetone():
    assert functional_groups(Chem.MolFromSmiles("CC(=O)C")) == [((1, 2), "C=O")]
    assert functional_groups(Chem.MolFromSmiles("CC")) == []


def test_build_vocab_applies_min_count(tmp_path):
    smiles = ["CC(=O)C", "CC(=O)C", "CCO", "CS(=O)(=O)C", "bad"]
    vocab = build_vocab(smiles, min_count=2, source="test")
    assert vocab.labels == [NONE_LABEL, UNK_LABEL, "C=O"]
    assert vocab.meta["n_molecules"] == 4 and vocab.meta["n_unparsed"] == 1
    assert vocab.meta["counts"] == {"C=O": 2} and vocab.meta["n_distinct_labels"] == 3
    back = FGVocab.load(vocab.save(tmp_path / "v.json"))
    assert back.labels == vocab.labels and back.meta["source"] == "test"


def test_atom_ids_cover_explicit_hydrogens():
    vocab = default_vocab()
    ids = vocab.atom_ids(Chem.MolFromSmiles("CC(=O)C"), n_atoms=10)
    assert list(ids[1:3]) == [vocab.id_of("C=O")] * 2
    assert ids[0] == NONE_ID and ids[3] == NONE_ID and (ids[4:] == NONE_ID).all()
    with pytest.raises(ValueError):
        vocab.atom_ids(Chem.MolFromSmiles("CC(=O)C"), n_atoms=3)


def test_vocab_rejects_bad_label_lists():
    with pytest.raises(ValueError):
        FGVocab(["C=O"])
    with pytest.raises(ValueError):
        FGVocab([NONE_LABEL, UNK_LABEL, "C=O", "C=O"])
