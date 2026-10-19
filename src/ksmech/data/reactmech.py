"""ReactMech loader. untested_on_real_data=True

ReactMech comes from DeepMech: Das, Hoque, Baranwal, Sunoj, "DeepMech: a machine learning
framework for chemical reaction mechanism prediction", Chem. Sci. 2026,
doi:10.1039/d6sc02809h, arXiv:2509.15872. Code: https://github.com/alhqlearn/DeepMech.

No raw file exists in data/raw/reactmech/, so this loader follows the *published* description.
It has not been run on the real files:

- Zenodo 10.5281/zenodo.20305780 holds `DeepMech_Dataset_Model.zip`, which contains
  `data/ID_Split42/{train,val,test}.txt`.
- The README says the files hold "atom-mapped elementary steps in SMILES format, one per line".
  The repo reads each line whole and splits it on '>>'. No header and no other column.
- These files have no arrows. Steps are ingested with `arrows=[]`, and the arrow-alignment phase
  derives the arrows from the mapped bond changes.
- No mechanism-class field was found in the released files: the 67 classes are only listed in the
  paper's SI. So `mechanism_class` is None and leave-one-class-out cannot run on ReactMech until a
  class source exists. The paper's out-of-distribution set (9 classes) may not be released.

Before trusting this loader, open the real file and document it in docs/data_schema.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ksmech.data.ingest import IngestResult
from ksmech.data.schema import ElementaryStep, Source
from ksmech.data.validation import build_step, make_step_id

log = logging.getLogger(__name__)

UNTESTED_ON_REAL_DATA = True
RAW_DIR = Path("data/raw/reactmech")
SPLIT_FILES = ("train.txt", "val.txt", "test.txt")


def resolve_inputs(raw: Path | None = None) -> list[Path]:
    raw = Path(raw or RAW_DIR)
    if raw.is_file():
        return [raw]
    hits = [p for name in SPLIT_FILES for p in sorted(raw.glob(f"**/ID_Split42/{name}"))]
    if not hits:
        raise FileNotFoundError(f"no ReactMech ID_Split42/{{train,val,test}}.txt under {raw}")
    return hits


def parse_line(text: str, step_id: str) -> ElementaryStep:
    rxn = text.split()[0]
    return build_step(
        step_id=step_id,
        source=Source.REACTMECH,
        rxn_smiles=rxn,
        arrows=[],
        require_arrows=False,
    )


def load(raw: Path | None = None) -> IngestResult:
    inputs = resolve_inputs(raw)
    log.warning("ReactMech loader is untested on real data; verify against the file first")
    result = IngestResult(Source.REACTMECH, inputs=[str(p) for p in inputs])
    for path in inputs:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    result.notes["blank_line_skipped"] += 1
                    continue
                result.notes["extra_tokens_ignored"] += len(text.split()) > 1
                step_id = make_step_id(Source.REACTMECH, text)
                result.add(step_id, text, lambda t=text, s=step_id: parse_line(t, s))
    result.notes["arrows_pending_alignment"] = len(result.steps)
    return result
