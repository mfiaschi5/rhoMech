"""ReactMech pathway chains (reaction-network plan, item G): the order of the steps, read from
the mechanism files of the DeepMech release and joined to the step ids `ksmech-data ingest
--source reactmech` gives (docs/data_schema.md, "Mechanism files"; docs/decisions.md,
2026-11-09).

A chain is one row of a mechanism file (`ID_CRM.csv`, `OOD_Set1.csv`, `OOD_Set2.csv`): a
Python list of unmapped pool SMILES, one per step boundary, so n species give n - 1 steps.
Step k of a chain is the line of the step files whose map-free canonical reactants and
products are species k and k + 1 (`reactmech.side_key`, the key of the ingest's class index).
Its id is the ingest's own (`make_step_id` of the line), so the sidecar holds for the pilot and
for a full ingest alike; `ingest_status` says whether the current steps parquet holds the step
('steps'), the ingest rejected its line ('rejected') or the pilot did not draw it
('not_sampled').

Nothing here writes the steps parquet or anything the sample store reads: `pathways.parquet` is
a separate file, guarded by `pathways_report.json`, whose provenance header carries the
parquet's sha256 as its content hash (`load_pathways` checks both). Every chain position that
finds no line, and every chain that is not whole, is counted under a reason code
(`ChainCode`). The atom maps are compared at every boundary between two matched steps
(`boundary_maps`): one mapped drawing, one after a renumbering of the map numbers, one drawing
whose atoms one side maps and the other does not (a spectator that reacts in the next step), or
not one drawing.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase

from ksmech.data import provenance as prov
from ksmech.data import reactmech
from ksmech.data.io import DEFAULT_PROCESSED_DIR, source_dir, steps_path
from ksmech.data.schema import Source
from ksmech.data.validation import make_step_id
from ksmech.processes import chunked, in_processes

log = logging.getLogger(__name__)

KIND = "reactmech_pathways"
SCHEMA_VERSION = 1
CHAIN_FILES = ("ID_CRM.csv", "OOD_Set1.csv", "OOD_Set2.csv")
TABLE_NAME = "pathways.parquet"
REPORT_NAME = "pathways_report.json"
CHUNK = 2000  # step lines per worker task

TABLE_SCHEMA = pa.schema(
    [
        pa.field("step_id", pa.string(), nullable=False),
        pa.field("pathway_id", pa.string(), nullable=False),
        pa.field("step_index", pa.int32(), nullable=False),  # 0-based position in the chain
        pa.field("n_steps", pa.int32(), nullable=False),  # the chain's species count - 1
        pa.field("is_last", pa.int8(), nullable=False),  # the stop label: 1 on the last step
        pa.field("chain_file", pa.string(), nullable=False),
        pa.field("chain_row", pa.int32(), nullable=False),  # 1-based data row in chain_file
        pa.field("mechanism_class", pa.string()),  # the OOD sets' class; null for ID_CRM
        pa.field("ingest_status", pa.string(), nullable=False),  # steps|not_sampled|rejected
        pa.field("maps_to_next", pa.string()),  # MapAgreement with step k + 1; null: none
    ]
)


class ChainCode(StrEnum):
    DUPLICATE_CHAIN = "DUPLICATE_CHAIN"  # a chain identical to an earlier one: dropped
    SPECIES_UNPARSABLE = "SPECIES_UNPARSABLE"  # RDKit cannot read a species: its steps unmatched
    NO_STEP_LINE = "NO_STEP_LINE"  # no line of the step files has the pair's map-free sides
    SEVERAL_LINES = "SEVERAL_LINES"  # several lines have them: each is a row at that position
    STEP_IN_SEVERAL_CHAINS = "STEP_IN_SEVERAL_CHAINS"  # one step id at positions of 2+ chains
    STOP_LABEL_CONFLICT = "STOP_LABEL_CONFLICT"  # a step id last in one chain, not in another
    CHAIN_UNMATCHED = "CHAIN_UNMATCHED"  # no step of the chain found a line
    CHAIN_PARTIAL = "CHAIN_PARTIAL"  # some steps found a line, some did not
    CHAIN_LAST_UNMATCHED = "CHAIN_LAST_UNMATCHED"  # the last step found none: no stop row


class MapAgreement(StrEnum):
    SAME = "same"  # step k's products and step k+1's reactants: one mapped drawing
    RENUMBERED = "renumbered"  # one mapped drawing once the map numbers are renumbered
    PARTLY_MAPPED = "partly_mapped"  # one drawing, atoms mapped on one side only: graph match
    INCONSISTENT = "inconsistent"  # not one drawing even without the maps


@dataclass(frozen=True)
class Chain:
    pathway_id: str
    chain_file: str
    row: int  # 1-based data row in its file
    species: tuple[str, ...]  # unmapped pool SMILES, one per step boundary
    mechanism_class: str | None

    @property
    def n_steps(self) -> int:
        return len(self.species) - 1


def pathway_id(species: Sequence[str]) -> str:
    """Stable id from the chain's species, independent of the file and row it sits in."""
    digest = hashlib.sha1(json.dumps(list(species)).encode()).hexdigest()[:16]
    return f"{Source.REACTMECH}-path-{digest}"


def chain_files(raw: Path | None = None) -> list[Path]:
    root = Path(raw or reactmech.RAW_DIR)
    root = root if root.is_dir() else root.parent
    hits = [p for name in CHAIN_FILES for p in sorted(root.glob(f"**/{name}"))]
    if not hits:
        raise FileNotFoundError(f"no ReactMech mechanism file ({', '.join(CHAIN_FILES)}) in {root}")
    return hits


def read_chains(paths: Iterable[Path], counts: Counter) -> list[Chain]:
    """Every chain of the mechanism files, each once: a repeat of an earlier chain's species
    list is counted (DUPLICATE_CHAIN) and dropped."""
    chains: list[Chain] = []
    seen: set[str] = set()
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for row, rec in enumerate(csv.DictReader(fh), start=1):
                species = tuple(ast.literal_eval(rec["mechanisms"]))
                pid = pathway_id(species)
                if pid in seen:
                    counts[ChainCode.DUPLICATE_CHAIN] += 1
                    counts[f"{ChainCode.DUPLICATE_CHAIN}:{path.name}"] += 1
                    continue
                seen.add(pid)
                cls = rec.get("mechanistic_class")
                chains.append(Chain(pid, path.name, row, species, cls))
    return chains


# ---- the step lines


def step_lines(raw: Path | None = None) -> dict[str, str]:
    """{step id: line} of every distinct line of the step files, read as the ingest reads them
    (`reactmech.load`: stripped, a repeat across files read once)."""
    lines: dict[str, str] = {}
    for path in reactmech.resolve_inputs(raw):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if text:
                    lines.setdefault(make_step_id(Source.REACTMECH, text), text)
    return lines


def _side_keys(texts: list[str]) -> list[tuple[str | None, str | None]]:
    out = []
    for text in texts:
        reac, _, prod = text.split()[0].partition(">>")
        out.append((reactmech.side_key(reac), reactmech.side_key(prod)))
    return out


def line_index(lines: dict[str, str], workers: int = 1) -> dict[tuple[str, str], list[str]]:
    """{(reactants key, products key): sorted step ids}, the keys from RDKit in `workers`
    spawned processes; a line with a side RDKit cannot read is in no entry."""
    ids = sorted(lines)
    chunks = chunked(ids, CHUNK)
    index: dict[tuple[str, str], list[str]] = defaultdict(list)
    tasks = ([lines[i] for i in chunk] for chunk in chunks)
    for chunk, keys in zip(chunks, in_processes(_side_keys, tasks, workers), strict=True):
        for sid, (rk, pk) in zip(chunk, keys, strict=True):
            if rk is not None and pk is not None:
                index[(rk, pk)].append(sid)
    return dict(index)


def match_chain(
    chain: Chain, index: dict[tuple[str, str], list[str]], counts: Counter
) -> list[list[str]]:
    """The step ids at each position of the chain ([] where none matched), positions
    counted by code."""
    keys = [reactmech.side_key(s) for s in chain.species]
    hits = []
    for a, b in zip(keys, keys[1:], strict=False):
        if a is None or b is None:
            counts[ChainCode.SPECIES_UNPARSABLE] += 1
            hits.append([])
            continue
        found = index.get((a, b), [])
        counts[ChainCode.NO_STEP_LINE] += not found
        counts[ChainCode.SEVERAL_LINES] += len(found) > 1
        hits.append(list(found))
    return hits


# ---- the atom maps at a boundary


def _mol(smiles: str) -> Chem.Mol | None:
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(smiles)


def _map_free(mol: Chem.Mol) -> Chem.Mol:
    m = Chem.Mol(mol)
    for atom in m.GetAtoms():
        atom.SetAtomMapNum(0)
    return m


def renumbering(products: Chem.Mol, reactants: Chem.Mol) -> dict[int, int] | None:
    """{map number in `products`: map number in `reactants`} over the atoms mapped on both
    sides, read off the canonical atom ranks of the two map-free drawings (one drawing has one
    canonical labelling), or None when the map-free drawings differ. Where the pool has
    symmetric atoms it is one renumbering of several, equal up to the pool's automorphisms."""
    flat_p, flat_r = _map_free(products), _map_free(reactants)
    if Chem.MolToSmiles(flat_p) != Chem.MolToSmiles(flat_r):
        return None
    by_rank = {rank: i for i, rank in enumerate(Chem.CanonicalRankAtoms(flat_r, breakTies=True))}
    sigma: dict[int, int] = {}
    for i, rank in enumerate(Chem.CanonicalRankAtoms(flat_p, breakTies=True)):
        a = products.GetAtomWithIdx(i).GetAtomMapNum()
        b = reactants.GetAtomWithIdx(by_rank[rank]).GetAtomMapNum()
        if a and b:
            sigma[a] = b
    return sigma


def boundary_maps(products: str, reactants: str) -> MapAgreement:
    """How the products of step k and the reactants of step k + 1 (mapped SMILES of one pool)
    agree: the same map numbers on the same atoms; a renumbering of every number; one
    drawing whose atoms one side maps and the other leaves unmapped (a spectator of one step
    that reacts in the next: only a graph match carries it across); or not one drawing."""
    p, r = _mol(products), _mol(reactants)
    if p is None or r is None:
        return MapAgreement.INCONSISTENT
    if Chem.MolToSmiles(p) == Chem.MolToSmiles(r):
        return MapAgreement.SAME
    sigma = renumbering(p, r)
    if sigma is None:
        return MapAgreement.INCONSISTENT
    relabelled = Chem.Mol(p)
    for atom in relabelled.GetAtoms():
        atom.SetAtomMapNum(sigma.get(atom.GetAtomMapNum(), 0))
    same = Chem.MolToSmiles(relabelled) == Chem.MolToSmiles(r)
    return MapAgreement.RENUMBERED if same else MapAgreement.PARTLY_MAPPED


def chain_maps(hits: list[list[str]], lines: dict[str, str]) -> list[MapAgreement | None]:
    """The agreement at each boundary k -> k + 1 (None after the last step, or where either
    side found no line); with several lines at a position, the first id stands for it."""
    out: list[MapAgreement | None] = []
    for here, nxt in zip(hits, [*hits[1:], []], strict=True):
        if not here or not nxt:
            out.append(None)
            continue
        prod = lines[here[0]].split()[0].split(">>")[1]
        reac = lines[nxt[0]].split()[0].split(">>")[0]
        out.append(boundary_maps(prod, reac))
    return out


# ---- the sidecar


def ingest_status(processed_dir: Path) -> tuple[set[str], set[str]]:
    """(the step ids of the current steps parquet, the ids its ingest rejected)."""
    folder = source_dir(Source.REACTMECH, processed_dir)
    table = pq.read_table(steps_path(Source.REACTMECH, processed_dir), columns=["step_id"])
    kept = set(table.column("step_id").to_pylist())
    with open(folder / "rejections.csv", newline="", encoding="utf-8") as fh:
        rejected = {row["step_id"] for row in csv.DictReader(fh)}
    return kept, rejected


def _status(sid: str, kept: set[str], rejected: set[str]) -> str:
    return "steps" if sid in kept else "rejected" if sid in rejected else "not_sampled"


def chain_rows(chain: Chain, hits, maps, kept: set[str], rejected: set[str]) -> list[dict]:
    """One row per (chain position, matched step id)."""
    rows = []
    for k, (ids, agree) in enumerate(zip(hits, maps, strict=True)):
        for sid in ids:
            rows.append(
                {
                    "step_id": sid,
                    "pathway_id": chain.pathway_id,
                    "step_index": k,
                    "n_steps": chain.n_steps,
                    "is_last": int(k == chain.n_steps - 1),
                    "chain_file": chain.chain_file,
                    "chain_row": chain.row,
                    "mechanism_class": chain.mechanism_class,
                    "ingest_status": _status(sid, kept, rejected),
                    "maps_to_next": None if agree is None else str(agree),
                }
            )
    return rows


def chain_codes(chain: Chain, hits: list[list[str]], counts: Counter) -> str:
    """'whole', 'partial' or 'unmatched', with the chain-level codes counted."""
    found = sum(bool(h) for h in hits)
    counts[ChainCode.CHAIN_LAST_UNMATCHED] += not hits[-1]
    if found == len(hits):
        return "whole"
    counts[ChainCode.CHAIN_UNMATCHED if not found else ChainCode.CHAIN_PARTIAL] += 1
    return "partial" if found else "unmatched"


def step_conflicts(rows: list[dict], counts: Counter) -> None:
    """Count the step ids that sit in several chains, and those a stop label disagrees on."""
    chains, last = defaultdict(set), defaultdict(set)
    for r in rows:
        chains[r["step_id"]].add(r["pathway_id"])
        last[r["step_id"]].add(r["is_last"])
    counts[ChainCode.STEP_IN_SEVERAL_CHAINS] += sum(len(v) > 1 for v in chains.values())
    counts[ChainCode.STOP_LABEL_CONFLICT] += sum(len(v) > 1 for v in last.values())


def map_summary(per_chain: list[tuple[str, list[MapAgreement | None]]]) -> dict:
    """Boundary counts, and over the whole chains with a boundary: those whose every boundary
    is one drawing (consistent: same, renumbered or partly mapped), those whose numbers alone carry
    every atom across (same or renumbered), and those whose numbers never change (same)."""
    boundaries = Counter(str(a) for _, maps in per_chain for a in maps if a is not None)
    whole = [maps[:-1] for kind, maps in per_chain if kind == "whole"]
    multi = [m for m in whole if m]  # a one-step chain has no boundary
    by_numbers = {MapAgreement.SAME, MapAgreement.RENUMBERED}
    counts = {
        "consistent_chains": sum(MapAgreement.INCONSISTENT not in m for m in multi),
        "renumbering_chains": sum(set(m) <= by_numbers for m in multi),
        "same_number_chains": sum(set(m) == {MapAgreement.SAME} for m in multi),
    }
    shares = {
        k.replace("_chains", "_share"): v / len(multi) if multi else None for k, v in counts.items()
    }
    return {
        "boundaries": dict(sorted(boundaries.items())),
        "whole_chains_with_a_boundary": len(multi),
        **counts,
        **shares,
    }


def build(raw: Path | None, processed_dir: Path, workers: int = 1) -> tuple[list[dict], dict]:
    """(the sidecar rows, the report) of every chain of the mechanism files."""
    counts: Counter = Counter()
    paths = chain_files(raw)
    chains = read_chains(paths, counts)
    lines = step_lines(raw)
    index = line_index(lines, workers)
    kept, rejected = ingest_status(processed_dir)
    rows: list[dict] = []
    per_chain: list[tuple[str, list[MapAgreement | None]]] = []
    kinds: dict[str, Counter] = defaultdict(Counter)
    for chain in chains:
        hits = match_chain(chain, index, counts)
        maps = chain_maps(hits, lines)
        kind = chain_codes(chain, hits, counts)
        kinds[chain.chain_file][kind] += 1
        kinds[chain.chain_file]["matched_steps"] += sum(bool(h) for h in hits)
        per_chain.append((kind, maps))
        rows += chain_rows(chain, hits, maps, kept, rejected)
    step_conflicts(rows, counts)
    report = build_report(chains, rows, kinds, counts, per_chain, len(lines))
    return rows, report


def build_report(chains, rows, kinds, counts, per_chain, n_lines: int) -> dict:
    """The counts of `pathways_report.json`: per chain file, codes, ingest status, maps."""
    by_file: dict[str, dict] = {}
    for c in chains:
        f = by_file.setdefault(c.chain_file, {"chains": 0, "steps": 0, "matched_rows": 0})
        f["chains"] += 1
        f["steps"] += c.n_steps
    for r in rows:
        by_file[r["chain_file"]]["matched_rows"] += 1
    for name, f in by_file.items():
        f |= {"matched_steps": 0, "whole": 0, "partial": 0, "unmatched": 0} | dict(kinds[name])
        f["duplicate_rows"] = counts[f"{ChainCode.DUPLICATE_CHAIN}:{name}"]
    status = Counter(r["ingest_status"] for r in rows)
    in_steps = defaultdict(list)
    for r in rows:
        in_steps[r["pathway_id"]].append(r["ingest_status"] == "steps")
    lengths = Counter(c.n_steps for c in chains)
    return {
        "step_lines_read": n_lines,
        "chains": len(chains),
        "chain_files": by_file,
        "steps_per_chain": {str(k): lengths[k] for k in sorted(lengths)},
        "rows": len(rows),
        "distinct_step_ids": len({r["step_id"] for r in rows}),
        "stop_rows": sum(r["is_last"] for r in rows),
        "codes": {str(k): counts[k] for k in ChainCode},
        "ingest_status": dict(sorted(status.items())),
        "chains_with_a_step_in_steps_parquet": sum(any(v) for v in in_steps.values()),
        "whole_chains_in_steps_parquet": sum(
            kind == "whole" and all(in_steps[c.pathway_id])
            for c, (kind, _) in zip(chains, per_chain, strict=True)
        ),
        "maps": map_summary(per_chain),
    }


def write_table(rows: list[dict], path: Path) -> Path:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=TABLE_SCHEMA), buf)
    return prov.atomic_write(path, buf.getvalue())


def sidecar_inputs(raw: Path | None, processed_dir: Path) -> dict[str, str]:
    folder = source_dir(Source.REACTMECH, processed_dir)
    files = [*chain_files(raw), *reactmech.resolve_inputs(raw)]
    files += [steps_path(Source.REACTMECH, processed_dir), folder / "rejections.csv"]
    return {name: sha for f in files for name, sha in prov.file_input(f).items()}


def write_pathways(
    raw: Path | None = None, processed_dir: Path = DEFAULT_PROCESSED_DIR, workers: int = 1
) -> tuple[Path, Path, dict]:
    """`pathways.parquet` and `pathways_report.json` under `<processed>/reactmech/`; the
    report's provenance header carries the parquet's sha256 as its content hash."""
    rows, report = build(raw, processed_dir, workers)
    folder = source_dir(Source.REACTMECH, processed_dir)
    table = write_table(rows, folder / TABLE_NAME)
    header = prov.stamp(
        KIND,
        SCHEMA_VERSION,
        sidecar_inputs(raw, processed_dir),
        {"chain_files": list(CHAIN_FILES), "key": "map-free canonical sides"},
        prov.file_sha256(table),
        {"table": TABLE_NAME},
    )
    out = prov.write_json(folder / REPORT_NAME, report, header)
    return table, out, report


def load_pathways(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> pa.Table:
    """The sidecar table, after its report's provenance (kind, schema, inputs, a clean build)
    and the parquet's sha256 against the report's content hash are checked."""
    folder = source_dir(Source.REACTMECH, processed_dir)
    report, table = folder / REPORT_NAME, folder / TABLE_NAME
    header = prov.verify_clean(report, KIND, SCHEMA_VERSION)
    if prov.file_sha256(table) != header.content_sha256:
        raise prov.StaleArtifact(table, [f"its sha256 differs from {REPORT_NAME}'s content hash"])
    return pq.read_table(table, schema=TABLE_SCHEMA)
