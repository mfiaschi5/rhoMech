"""Canonical schema for one elementary mechanistic step, shared by every data source.

Atoms are always referred to by atom-map number in `reactants_smiles` / `products_smiles`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pyarrow as pa


class Source(StrEnum):
    PMECHDB = "pmechdb"
    RMECHDB = "rmechdb"
    REACTMECH = "reactmech"
    MECH_USPTO = "mech_uspto"
    FLOWER = "flower"


class SourceKind(StrEnum):
    LONE_PAIR = "lone_pair"
    BOND = "bond"
    RADICAL = "radical"


class SinkKind(StrEnum):
    ATOM = "atom"
    BOND = "bond"


_SOURCE_ARITY = {SourceKind.LONE_PAIR: 1, SourceKind.RADICAL: 1, SourceKind.BOND: 2}
_SINK_ARITY = {SinkKind.ATOM: 1, SinkKind.BOND: 2}


def _norm_atoms(atoms: tuple[int, ...], arity: int, what: str) -> tuple[int, ...]:
    atoms = tuple(int(a) for a in atoms)
    if len(atoms) != arity:
        raise ValueError(f"{what} needs {arity} atom(s), got {atoms}")
    if len(set(atoms)) != len(atoms):
        raise ValueError(f"{what} repeats an atom: {atoms}")
    if any(a <= 0 for a in atoms):
        raise ValueError(f"{what} has a non-positive atom map number: {atoms}")
    return tuple(sorted(atoms))  # a bond is an unordered pair


@dataclass(frozen=True)
class Arrow:
    """One curly arrow, written the way it is drawn: the tail sits on the source electrons and
    the head points at an atom or at a bond (an existing or a new one).

    Where the electron pair ends up depends on both ends; see `destination`.
    """

    source_kind: SourceKind
    source_atoms: tuple[int, ...]
    sink_kind: SinkKind
    sink_atoms: tuple[int, ...]

    def __post_init__(self) -> None:
        src_kind, snk_kind = SourceKind(self.source_kind), SinkKind(self.sink_kind)
        src = _norm_atoms(self.source_atoms, _SOURCE_ARITY[src_kind], "source")
        snk = _norm_atoms(self.sink_atoms, _SINK_ARITY[snk_kind], "sink")
        object.__setattr__(self, "source_kind", src_kind)
        object.__setattr__(self, "sink_kind", snk_kind)
        object.__setattr__(self, "source_atoms", src)
        object.__setattr__(self, "sink_atoms", snk)
        self.destination()  # rejects shapes that have no electron-pushing meaning

    def destination(self) -> tuple[str, tuple[int, ...]]:
        """Where the moved pair ends up: ("lone_pair", (a,)) or ("bond", (a, b))."""
        src, snk = set(self.source_atoms), set(self.sink_atoms)
        if self.source_kind is SourceKind.BOND:
            if self.sink_kind is SinkKind.ATOM and snk <= src:
                return "lone_pair", self.sink_atoms
            if self.sink_kind is SinkKind.BOND and len(src & snk) == 1:
                return "bond", self.sink_atoms
            raise ValueError(f"bond source {self.source_atoms} cannot reach {self.sink_atoms}")
        # lone pair or radical: the head names the partner atom, or the new bond itself
        if self.sink_kind is SinkKind.ATOM and not snk & src:
            return "bond", tuple(sorted(src | snk))
        if self.sink_kind is SinkKind.BOND and src < snk:
            return "bond", self.sink_atoms
        raise ValueError(f"{self.source_kind} source {self.source_atoms} cannot reach {snk}")

    @property
    def atoms(self) -> frozenset[int]:
        return frozenset(self.source_atoms) | frozenset(self.sink_atoms)

    def to_dict(self) -> dict:
        return {
            "source_kind": str(self.source_kind),
            "source_atoms": list(self.source_atoms),
            "sink_kind": str(self.sink_kind),
            "sink_atoms": list(self.sink_atoms),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Arrow:
        return cls(
            SourceKind(d["source_kind"]),
            tuple(d["source_atoms"]),
            SinkKind(d["sink_kind"]),
            tuple(d["sink_atoms"]),
        )


@dataclass
class ElementaryStep:
    step_id: str
    source: Source
    pathway_id: str | None
    step_index: int | None
    reactants_smiles: str
    products_smiles: str
    arrows: list[Arrow] = field(default_factory=list)
    mechanism_class: str | None = None
    conditions: str | None = None
    resonance_step: bool = False  # reactants and products are resonance structures

    def __post_init__(self) -> None:
        self.source = Source(self.source)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "source": str(self.source),
            "pathway_id": self.pathway_id,
            "step_index": self.step_index,
            "reactants_smiles": self.reactants_smiles,
            "products_smiles": self.products_smiles,
            "arrows": [a.to_dict() for a in self.arrows],
            "mechanism_class": self.mechanism_class,
            "conditions": self.conditions,
            "resonance_step": bool(self.resonance_step),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ElementaryStep:
        return cls(
            step_id=d["step_id"],
            source=Source(d["source"]),
            pathway_id=d["pathway_id"],
            step_index=None if d["step_index"] is None else int(d["step_index"]),
            reactants_smiles=d["reactants_smiles"],
            products_smiles=d["products_smiles"],
            arrows=[Arrow.from_dict(a) for a in d["arrows"]],
            mechanism_class=d["mechanism_class"],
            conditions=d["conditions"],
            resonance_step=bool(d.get("resonance_step") or False),
        )


ARROW_TYPE = pa.struct(
    [
        ("source_kind", pa.string()),
        ("source_atoms", pa.list_(pa.int32())),
        ("sink_kind", pa.string()),
        ("sink_atoms", pa.list_(pa.int32())),
    ]
)

STEP_SCHEMA = pa.schema(
    [
        pa.field("step_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("pathway_id", pa.string()),
        pa.field("step_index", pa.int32()),
        pa.field("reactants_smiles", pa.string(), nullable=False),
        pa.field("products_smiles", pa.string(), nullable=False),
        pa.field("arrows", pa.list_(ARROW_TYPE), nullable=False),
        pa.field("mechanism_class", pa.string()),
        pa.field("conditions", pa.string()),
        pa.field("resonance_step", pa.bool_()),
    ]
)
