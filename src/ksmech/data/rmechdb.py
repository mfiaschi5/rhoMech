"""rmechdb loader: RMechDB's radical steps as FlowER ships them (open-shell plan P8 step 2;
docs/data_schema.md, "rmechdb").

RMechDB: Tavakoli, Chiu, Baldi et al., "RMechDB: a public database of elementary radical
reaction steps", J. Chem. Inf. Model. 2023 (PMC9976277), CC-BY-NC-ND. FlowER: Joung et al.,
"Electron flow matching for generative reaction mechanism prediction" (github.com/schwallergroup
/FlowER; data figshare 32513667, MIT label on the repository, not relied on: the RMechDB lines
keep RMechDB's license). Unpacked under `data/raw/flower/flower_new_dataset/`.

What the files hold (opened 2026-11-05, `scripts/probes/flower_inspect.py`): one line per step,
`mapped reactants>>products|SUFFIX`, every atom mapped with explicit H. Only the lines whose
suffix is RS or RC are read (inferred as RMechDB's specific and core sets: RC 1,383 train / 136
test, RS 3,223 / 346, none in val); the numeric-suffix lines (about 2.14M) and PM / PC are never
ingested. The 3.1 GB train file is streamed.

Before the id, in this order: the bath gases (an unbonded noble-gas atom with one map number on
both sides: [Ar], [He]) are stripped and the line counted (NOBLE_GAS_STRIPPED, a note); then the
spin redrawings of `data.domain.redrawn_side` (PAIRED_DRAWING: [N]=[N] -> N#N, ozone;
HIGH_SPIN_DRAWING) give the key, so no id moves after the splits freeze. The id is
make_step_id('rmechdb', map-free canonical reaction), so a step drawn with other atom maps or
with a bath gas shares its twin's id. One step per id: the RC copy wins over RS (in any file
order), otherwise the first line in file order (train, val, test); every other line is
DUPLICATE_STEP. The redrawings themselves are made (and counted as repairs) by `build_step`, on
the stripped line. Columns: `flower_suffix` (RS or RC, the winner's raw suffix) and
`flower_part` (train, test, or train+test for a reaction in both FlowER parts, counted). The
steps have no arrows: R is the difference of the mapped sides.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from rdkit import Chem

from ksmech.data import chem, domain
from ksmech.data.ingest import IngestResult
from ksmech.data.schema import ElementaryStep, Source
from ksmech.data.validation import (
    RejectReason,
    StepRejected,
    build_step,
    make_step_id,
    split_reaction,
)

log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/flower/flower_new_dataset")
PARTS = ("train", "val", "test")  # file order: train before test
TAGS = ("RC", "RS")  # the RC copy of a reaction wins over its RS copy
NOBLE_GASES = frozenset([2, 10, 18, 36, 54, 86])
VIA = "flower"


@dataclass
class Line:
    part: str | None
    tag: str
    rxn: str  # as read
    stripped: str  # bath gases removed (the record build_step reads)
    step_id: str
    noble: bool = False


@dataclass
class Group:
    lines: list[Line] = field(default_factory=list)

    def winner(self) -> Line:
        return next((x for x in self.lines if x.tag == "RC"), self.lines[0])


def resolve_inputs(raw: Path | None = None) -> list[tuple[str | None, Path]]:
    """(part, file): the train, val and test files under a directory, or one file as given
    (its part from its name when it is one of them)."""
    raw = Path(raw or RAW_DIR)
    if raw.is_file():
        return [(raw.stem if raw.stem in PARTS else None, raw)]
    hits = [(p, raw / f"{p}.txt") for p in PARTS if (raw / f"{p}.txt").is_file()]
    if not hits:
        raise FileNotFoundError(f"no FlowER {{train,val,test}}.txt under {raw}")
    return hits


def radical_lines(inputs: list[tuple[str | None, Path]]) -> Iterator[tuple[str | None, str, str]]:
    """(part, suffix, reaction) of every RS / RC line, streamed in file order."""
    for part, path in inputs:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rxn, _, tag = line.rstrip("\n").rpartition("|")
                if tag in TAGS:
                    yield part, tag, rxn


def _is_bath(atom: Chem.Atom) -> bool:
    return (
        atom.GetAtomicNum() in NOBLE_GASES
        and atom.GetDegree() == 0
        and atom.GetFormalCharge() == 0
        and atom.GetAtomMapNum() > 0
    )


def strip_bath_gases(rxn: str) -> tuple[str, bool]:
    """`rxn` without its bath gases (module docstring) and whether one was stripped. A noble
    gas that is bonded, charged, or not one unbonded atom of the same map on both sides stays."""
    sides = [chem.parse_mapped(s) for s in split_reaction(rxn)]
    maps = [{a.GetAtomMapNum() for a in m.GetAtoms() if _is_bath(a)} for m in sides]
    bath = maps[0] & maps[1]
    if not bath:
        return rxn, False
    out = []
    for m in sides:
        rw = Chem.RWMol(m)
        for idx in sorted((a.GetIdx() for a in m.GetAtoms() if a.GetAtomMapNum() in bath),
                          reverse=True):  # fmt: skip
            rw.RemoveAtom(idx)
        out.append(chem.canonical_mapped_smiles(rw.GetMol()))
    return ">>".join(out), True


def reaction_key(stripped: str) -> str:
    """The map-free canonical reaction after the spin redrawings: the id's record."""
    keys = []
    for side, name in zip(split_reaction(stripped), ("reactants", "products"), strict=True):
        mol, repairs = domain.redrawn_side(chem.parse_mapped(side), name)
        if repairs:
            mol = chem.parse_mapped(chem.canonical_mapped_smiles(mol))
        keys.append(chem.unmapped_canonical(mol))
    return ">>".join(keys)


def read_line(part: str | None, tag: str, rxn: str) -> Line:
    """One line with its id; a line that does not parse keeps an id from its raw text (its
    build is rejected with the reason)."""
    try:
        stripped, noble = strip_bath_gases(rxn)
        key = reaction_key(stripped)
    except (StepRejected, chem.SmilesParseError, chem.SanitizeError, ValueError):
        return Line(part, tag, rxn, rxn, make_step_id(Source.RMECHDB, rxn))
    return Line(part, tag, rxn, stripped, make_step_id(Source.RMECHDB, key), noble)


def grouped(lines: Iterator[tuple[str | None, str, str]]) -> dict[str, Group]:
    """Lines by step id, in first-seen order."""
    groups: dict[str, Group] = {}
    for part, tag, rxn in lines:
        line = read_line(part, tag, rxn)
        groups.setdefault(line.step_id, Group()).lines.append(line)
    return groups


def _parts(group: Group) -> str | None:
    """train, test or train+test (FlowER's val holds no RS / RC line)."""
    seen = {x.part for x in group.lines}
    return "+".join(p for p in PARTS if p in seen) or None


def make_step(group: Group) -> ElementaryStep:
    win = group.winner()
    step = build_step(
        step_id=win.step_id,
        source=Source.RMECHDB,
        rxn_smiles=win.stripped,
        arrows=[],
        require_arrows=False,
    )
    return replace(step, flower_suffix=win.tag, flower_part=_parts(group))


def _count(result: IngestResult, groups: dict[str, Group]) -> None:
    n = result.notes
    for g in groups.values():
        n["lines_" + "_".join(sorted({x.tag for x in g.lines}))] += len(g.lines)
        n["NOBLE_GAS_STRIPPED"] += sum(x.noble for x in g.lines)
        n["rc_wins_over_rs"] += g.winner().tag == "RC" and any(x.tag == "RS" for x in g.lines)
        n["in_train_and_test"] += _parts(g) == "train+test"


def load(raw: Path | None = None) -> IngestResult:
    """Every RS / RC line, one step per id (module docstring)."""
    inputs = resolve_inputs(raw)
    result = IngestResult(Source.RMECHDB, inputs=[str(p) for _, p in inputs], via=VIA)
    groups = grouped(radical_lines(inputs))
    _count(result, groups)
    for step_id, group in groups.items():
        win = group.winner()
        result.add(step_id, win.rxn, lambda g=group: make_step(g))
        for line in group.lines:
            if line is win:
                continue
            result.n_records += 1
            why = "the RC copy is kept" if win.tag != line.tag else "an earlier line is kept"
            result.reject(step_id, line.rxn, StepRejected(RejectReason.DUPLICATE_STEP, why))
    result.notes["arrows_from_mapped_sides"] = len(result.steps)
    return result
