"""Species lists for the QM batch and the token build (`ksmech-qm run --species-from FILE`,
`ksmech-orbitals build --species-from FILE`): molecules that no source's steps hold, such as
the human benchmark's (`data.benchmark_checks.write_species_file`).

FILE is JSON: a list of SMILES, or an object whose `species` is such a list, or, as in the
human benchmark's `qm_coverage.json`, an object of status lists ({status: [SMILES]}), of which
the lists a QM batch or a token build can complete are read (`BATCH_LISTS`: `qm_missing` and
every `tokens_*` list) and the others (`qm_tombstone`, `qm_census_refused`) are counted and left
out. Each SMILES is split into fragments and every fragment canonicalized map-free
(`chem.unmapped_canonical`), the keys the QM cache and the token files use; the spin census
(`batch.census_allows`) refuses what it refuses for a source (SPIN_CENSUS_REFUSED), with no
exemption (the counted exemptions are steps, and a list has none).

A run's manifest is `<processed-dir>/species/<name>/qm_manifest.parquet` and its token summary
`<processed-dir>/species/<name>/tokens_summary.json`: `species/` is no `Source` folder (a
source's manifest is `<processed-dir>/<source>/qm_manifest.parquet`), so a list never writes
over a source's files. `name` defaults to the file's stem.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from rdkit import Chem

from ksmech.data import chem
from ksmech.data.io import DEFAULT_PROCESSED_DIR
from ksmech.qm import batch
from ksmech.qm.records import ReasonCode

log = logging.getLogger(__name__)

SPECIES_DIR = "species"
BATCH_LISTS = ("qm_missing",)  # with every `tokens_*` list: what a batch and a build complete
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def coverage_species(lists: dict[str, list[str]]) -> tuple[list[str], dict[str, int]]:
    """(the species of the status lists a QM batch or a token build can complete, {every
    other list: its size}) of a coverage file's `species` object."""
    keep = [k for k in lists if k in BATCH_LISTS or k.startswith("tokens_")]
    left = {k: len(v) for k, v in sorted(lists.items()) if k not in keep}
    return sorted({s for k in keep for s in lists[k]}), left


def read_species_file(path: Path) -> tuple[list[str], dict[str, int]]:
    """(the SMILES a species file lists, {status list left out: its size}) (module docstring);
    ValueError for any other shape."""
    body = json.loads(Path(path).read_text())
    items, left = body.get("species") if isinstance(body, dict) else body, {}
    if isinstance(items, dict) and all(isinstance(v, list) for v in items.values()):
        items, left = coverage_species(items)
        for key, n in left.items():
            log.warning("%s: list %r (%d species) left out: no batch completes it", path, key, n)
    if not isinstance(items, list) or not all(isinstance(s, str) for s in items):
        raise ValueError(
            f"{path}: neither a list of SMILES nor an object with a 'species' list or object"
        )
    return items, left


def species_dir(name: str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    """`<processed_dir>/species/<name>/`; ValueError for a name that is not one path part of
    letters, digits, '_', '.' and '-'."""
    if not _NAME.match(name) or name in (".", ".."):
        raise ValueError(f"species run name {name!r}: use letters, digits, '_', '.', '-'")
    return Path(processed_dir) / SPECIES_DIR / name


def run_name(path: Path, name: str | None) -> str:
    """The run name: `name`, else the species file's stem."""
    return name or Path(path).stem


def manifest_path(name: str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    return species_dir(name, processed_dir) / batch.MANIFEST_NAME


def _fragments(smiles: str, failed: dict[str, dict]) -> list[str]:
    """The map-free canonical SMILES of each fragment of `smiles`; a SMILES or fragment RDKit
    cannot read gets a CANONICALIZE_FAILED row in `failed` instead (conformer -1)."""
    code = str(ReasonCode.CANONICALIZE_FAILED)
    try:
        frags = chem.fragments(chem.parse_mapped(smiles))
    except (chem.SmilesParseError, chem.SanitizeError) as exc:
        failed[smiles] = batch._row(smiles, -1, reason=code, detail=str(exc))
        return []
    keys = []
    for frag in frags:
        try:
            keys.append(chem.unmapped_canonical(frag))
        except (chem.SanitizeError, ValueError) as exc:
            raw = Chem.MolToSmiles(frag)
            failed[raw] = batch._row(raw, -1, reason=code, detail=str(exc))
    return keys


def collect_species(smiles: Iterable[str]) -> tuple[list[str], list[dict]]:
    """`batch.collect_molecules` for a species list: the unique map-free canonical SMILES of
    every fragment that the spin census admits, plus a row for each one that fails to
    canonicalize or that the census refuses (no exemption)."""
    failed: dict[str, dict] = {}
    keys = {k for s in sorted(set(smiles)) for k in _fragments(s, failed)}
    admitted, refused = batch._census(keys, exempt=set())
    for row in failed.values():
        log.warning("species %s: %s (%s)", row["smiles"], row["reason"], row["detail"])
    return admitted, list(failed.values()) + refused


def featurize_species(
    smiles: Iterable[str], name: str, processed_dir: Path = DEFAULT_PROCESSED_DIR, **kw
) -> pd.DataFrame:
    """`batch.featurize_dataset` for a species list: the same tasks, census, cache keys, failure
    memo, reason codes and options (`**kw`: `batch.featurize_molecules`'s), with the manifest
    `manifest_path(name)`. Returns this run's rows."""
    path = manifest_path(name, processed_dir)  # the name is checked before any work
    admitted, failed = collect_species(smiles)
    return batch.featurize_molecules(admitted, failed, path, **kw)
