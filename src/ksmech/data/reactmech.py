"""ReactMech loader, checked against the released files (docs/data_schema.md, "ReactMech").

ReactMech comes from DeepMech: Das, Hoque, Baranwal, Sunoj, "DeepMech: a machine learning
framework for chemical reaction mechanism prediction", Chem. Sci. 2026,
doi:10.1039/d6sc02809h, arXiv:2509.15872. Code: https://github.com/alhqlearn/DeepMech. Data:
Zenodo 10.5281/zenodo.20305780, `DeepMech_Dataset_Model.zip` (CC BY 4.0), unpacked under
`data/raw/reactmech/` (the `models/` checkpoint is not needed and not unpacked).

What the files hold (opened 2026-10-26):

- `data/ID_Split42/{train,val,test}.txt` and `data/OOD_Split42/{train,val,test}.txt`: one
  atom-mapped elementary step per line, `reactants>>products`, no other field and no arrows.
  The ID split has 104,964 steps; the OOD split partitions the same steps differently and adds
  9,235 of its own (the steps of its 9 held-out classes), so the two give 114,199 distinct
  steps. Each is read once: a line already seen in another file is counted
  (`repeat_across_files`), not rejected.
- `data/OOD_Split42/OOD_CRM_dataset/OOD_Set{1,2}.csv`: `mechanistic_class,mechanisms`, one
  mechanism per row as a Python list of unmapped SMILES, one species per step boundary. They
  give the class of the OOD steps: a step whose map-free reactants and products are two
  consecutive species of a mechanism takes that mechanism's class. Only Set1's 5 classes
  (acid_halide, sul_halide, anhydride, haloformate, carbamic_halide) label steps, 9,948 of them:
  Set2's 4 (Buchwald-Hartwig aminations with Pd and Ni) match no step of the split files, whose
  Ni steps are other couplings. The other 58 of the paper's 67 classes are not released, so
  every other step has `mechanism_class` None.
- `data/ID_Split42/ID_CRM_dataset/ID_CRM.csv`: 2,751 mechanisms without a class. Not read by
  the ingest: `data.pathways` reads it, and the OOD sets, for the step order (a sidecar,
  `ksmech-data pathways`).
- `data/PrebioMech/`: a separate set of prebiotic steps. Not read.

The steps have no arrows (`arrows=[]`, `require_arrows=False`): their reaction matrix is the
difference of the two mapped sides.
"""

from __future__ import annotations

import ast
import csv
import logging
from functools import partial
from pathlib import Path

from ksmech.data import chem
from ksmech.data.ingest import IngestResult
from ksmech.data.schema import ElementaryStep, Source
from ksmech.data.validation import build_step, make_step_id, split_reaction

log = logging.getLogger(__name__)

UNTESTED_ON_REAL_DATA = False
RAW_DIR = Path("data/raw/reactmech")
SPLITS = ("ID_Split42", "OOD_Split42")
SPLIT_FILES = ("train.txt", "val.txt", "test.txt")
CLASS_FILES = ("OOD_Set1.csv", "OOD_Set2.csv")


def resolve_inputs(raw: Path | None = None, parts: tuple[str, ...] = SPLIT_FILES) -> list[Path]:
    """The step files: a file as given, else the `parts` files of both splits under a
    directory."""
    raw = Path(raw or RAW_DIR)
    if raw.is_file():
        return [raw]
    hits = [p for s in SPLITS for name in parts for p in sorted(raw.glob(f"**/{s}/{name}"))]
    if not hits:
        raise FileNotFoundError(f"no ReactMech {{ID,OOD}}_Split42/*.txt under {raw}")
    return hits


def side_key(smiles: str) -> str | None:
    """Map-free canonical SMILES of one side, or None when it does not parse (the class index
    and the pathway chains of `data.pathways` join a mechanism's species to the steps by it)."""
    try:
        return chem.unmapped_canonical(chem.parse_mapped(smiles))
    except (chem.SmilesParseError, chem.SanitizeError, ValueError):
        return None


def class_index(raw: Path | None = None) -> tuple[dict[tuple[str, str], str], int]:
    """{(reactants key, products key): class} over every consecutive species pair of the
    classified mechanisms, and the pairs dropped because two classes claim them."""
    root = Path(raw or RAW_DIR)
    root = root if root.is_dir() else root.parent
    pairs: dict[tuple[str, str], str] = {}
    clash: set[tuple[str, str]] = set()
    for path in sorted(p for name in CLASS_FILES for p in root.glob(f"**/{name}")):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                keys = [side_key(s) for s in ast.literal_eval(row["mechanisms"])]
                for pair in zip(keys, keys[1:], strict=False):
                    if None in pair:
                        continue
                    if pairs.setdefault(pair, row["mechanistic_class"]) != row["mechanistic_class"]:
                        clash.add(pair)
    for pair in clash:
        del pairs[pair]
    return pairs, len(clash)


def parse_line(text: str, step_id: str, classes: dict | None = None) -> ElementaryStep:
    rxn = text.split()[0]
    reac, prod = split_reaction(rxn)
    mechanism_class = (classes or {}).get((side_key(reac), side_key(prod))) if classes else None
    return build_step(
        step_id=step_id,
        source=Source.REACTMECH,
        rxn_smiles=rxn,
        arrows=[],
        require_arrows=False,
        mechanism_class=mechanism_class,
    )


def load(
    raw: Path | None = None, parts: tuple[str, ...] = SPLIT_FILES, n_workers: int = 1
) -> IngestResult:
    """Every step of the `parts` files of both splits, each once, with the classes of the
    classified mechanisms; `n_workers` spawned processes parse them (the same result,
    `IngestResult.add_records`)."""
    inputs = resolve_inputs(raw, parts)
    classes, clashes = class_index(raw)
    result = IngestResult(Source.REACTMECH, inputs=[str(p) for p in inputs])
    result.notes["class_pairs_claimed_twice"] = clashes
    seen: set[str] = set()
    records = []
    for path in inputs:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    result.notes["blank_line_skipped"] += 1
                    continue
                if text in seen:
                    result.notes["repeat_across_files"] += 1
                    continue
                seen.add(text)
                result.notes["extra_tokens_ignored"] += len(text.split()) > 1
                step_id = make_step_id(Source.REACTMECH, text)
                records.append((step_id, text, partial(parse_line, text, step_id, classes)))
    result.add_records(records, n_workers)
    result.notes["with_mechanism_class"] = sum(s.mechanism_class is not None for s in result.steps)
    result.notes["arrows_pending_alignment"] = len(result.steps)
    return result
