"""Ertl functional groups (RDKit Contrib `IFG`) as an integer vocabulary index per atom.

The label of a group is the IFG `atoms` string: the canonical SMILES of the group's own atoms,
without the neighbouring carbons (`C=O`, `O=CO`, `[O-]`, `n`). The packaged vocabulary
`fg_vocab.json` is built from the unique PMechDB molecules (`ksmech-orbitals fg-vocab`). Index 0
is "no group", index 1 is the UNK bucket for labels outside the vocabulary. Hydrogens never
belong to a group. Labels are stored on the atom tokens for tests, the interface and the
"functional groups added back as input" ablation; they are not a model input
(`AtomToken.fg_id` is tagged `model_input=False`) and nothing downstream decides on them.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import date
from functools import cache
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig, rdBase

NONE_ID, UNK_ID = 0, 1
NONE_LABEL, UNK_LABEL = "<none>", "<unk>"
DEFAULT_VOCAB_PATH = Path(__file__).with_name("fg_vocab.json")


@cache
def _ifg():
    """The Contrib module, importable as a package in recent RDKit, else from the Contrib dir."""
    try:
        from rdkit.Contrib.IFG import ifg
    except ImportError:  # older RDKit: no package __init__ in Contrib
        sys.path.append(os.path.join(RDConfig.RDContribDir, "IFG"))
        import ifg
    return ifg


def functional_groups(mol: Chem.Mol) -> list[tuple[tuple[int, ...], str]]:
    """(atom indices, label) for every Ertl group of `mol` (heavy-atom mol, aromaticity on)."""
    return [(tuple(g.atomIds), g.atoms) for g in _ifg().identify_functional_groups(mol)]


class FGVocab:
    def __init__(self, labels: list[str], meta: dict | None = None) -> None:
        if labels[:2] != [NONE_LABEL, UNK_LABEL]:
            raise ValueError(f"labels must start with {NONE_LABEL!r}, {UNK_LABEL!r}")
        if len(set(labels)) != len(labels):
            raise ValueError("duplicate labels in the vocabulary")
        self.labels = list(labels)
        self.index = {label: i for i, label in enumerate(labels)}
        self.meta = dict(meta or {})

    def __len__(self) -> int:
        return len(self.labels)

    def id_of(self, label: str) -> int:
        return self.index.get(label, UNK_ID)

    @classmethod
    def load(cls, path: Path = DEFAULT_VOCAB_PATH) -> FGVocab:
        data = json.loads(Path(path).read_text())
        return cls(data["labels"], {k: v for k, v in data.items() if k != "labels"})

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**self.meta, "labels": self.labels}, indent=1) + "\n")
        return path

    def atom_ids(self, mol: Chem.Mol, n_atoms: int | None = None) -> np.ndarray:
        """Vocabulary id per atom of `mol` (heavy atoms, implicit H). Extra trailing atoms of a
        record that carries explicit hydrogens (`n_atoms` > heavy count) get `NONE_ID`."""
        n = mol.GetNumAtoms() if n_atoms is None else n_atoms
        if n < mol.GetNumAtoms():
            raise ValueError(f"n_atoms={n} < {mol.GetNumAtoms()} atoms in the mol")
        ids = np.full(n, NONE_ID, dtype=np.int32)
        for atoms, label in functional_groups(mol):
            ids[list(atoms)] = self.id_of(label)
        return ids


@cache
def default_vocab() -> FGVocab:
    return FGVocab.load()


def count_groups(smiles_iter: Iterable[str]) -> tuple[Counter[str], int, int]:
    """Label counts over molecules; also (molecules seen, molecules RDKit could not parse)."""
    counts: Counter[str] = Counter()
    n_seen = n_bad = 0
    with rdBase.BlockLogs():
        for smi in smiles_iter:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_bad += 1
                continue
            n_seen += 1
            counts.update(label for _, label in functional_groups(mol))
    return counts, n_seen, n_bad


def build_vocab(smiles_iter: Iterable[str], min_count: int, source: str = "") -> FGVocab:
    """Vocabulary of every label seen at least `min_count` times, most frequent first."""
    counts, n_seen, n_bad = count_groups(smiles_iter)
    kept = [label for label, n in counts.most_common() if n >= min_count]
    covered = sum(counts[label] for label in kept)
    total = sum(counts.values())
    meta = dict(
        source=source,
        built=date.today().isoformat(),
        min_count=min_count,
        n_molecules=n_seen,
        n_unparsed=n_bad,
        n_distinct_labels=len(counts),
        coverage=covered / total if total else 0.0,
        counts={label: counts[label] for label in kept},
    )
    return FGVocab([NONE_LABEL, UNK_LABEL, *kept], meta)
