"""The PMechRP human benchmark: 350 textbook pathways (Clayden, 2nd ed.) as an evaluation-only
table (reaction-network plan, item G; docs/data_schema.md, "Pathway benchmark files").

NEVER TRAINED ON. The benchmark comes from the release PMechDB comes from, and most of its step
targets are drawn with PMechDB's reaction-centre map numbers (10, 20, ...), though few are
PMechDB products; `data.benchmark_checks.overlap` measures how much of it a PMechDB split's
training part already holds. It is no `Source`: the ingest, the sample store and `ksmech-train`
take their sources from that enum, and `ksmech-train` refuses its name besides
(`refuse_training`). Its files live under `data/processed/human_benchmark/`, beside no sample
view.

What the file holds (opened 2026-11-09): one pathway per row; `Temp (≠ RT)` (a temperature
such as `-78 °C`, or `n/a`, which the header reads as room temperature), the reactant pool
and the final product as unmapped SMILES, the product's formula, `Min Steps` (the number of
steps), the textbook citation, and `Step 1 Target` .. `Step 7 Target`: the species each step
makes. No arrows, no pools after the first, no atom maps but those some targets carry.

`read_benchmark` keeps every row and counts each normalization (`BenchmarkCode`): a cell
with surrounding spaces is stripped, a target cell that holds a whole PMechDB record
(`reactants>>products arrows`) is read as its product side (the depth files hold that side),
a blank target inside the steps is a null target, and a temperature that is neither `n/a`
nor `<number> °C` is null and counted.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

from ksmech.data import chem
from ksmech.data import provenance as prov

NAME = "human_benchmark"
EVALUATION_ONLY = frozenset({NAME})  # names `refuse_training` turns away
RAW_FILE = Path("data/raw/pmechrp/pathway_datasets/human_benchmark/pathways/350_HumanBenchmark.csv")
OUT_DIR = Path("data/processed") / NAME
TABLE_NAME = "pathways.parquet"
REPORT_NAME = "pathways_report.json"
KIND = "human_benchmark"
SCHEMA_VERSION = 1
MAX_STEPS = 7
HEADER = (
    "Temp (≠ RT)",
    "Reactants",
    "Final Product",
    "Prod Formula",
    "Min Steps",
    "Clayden, 2nd Ed Textbook Citation",
    *(f"Step {k} Target" for k in range(1, MAX_STEPS + 1)),
)
_TEMPERATURE = re.compile(r"^(?P<sign>-)?\s*(?P<value>\d+(?:\.\d+)?)\s*°C$")
_MAPPED = re.compile(r":\d+\]")

TABLE_SCHEMA = pa.schema(
    [
        pa.field("pathway_id", pa.string(), nullable=False),
        pa.field("row", pa.int32(), nullable=False),  # 1-based data row of the CSV
        pa.field("reactants_smiles", pa.string(), nullable=False),
        pa.field("final_product_smiles", pa.string(), nullable=False),
        pa.field("product_formula", pa.string(), nullable=False),
        pa.field("n_steps", pa.int32(), nullable=False),  # `Min Steps`
        pa.field("step_targets", pa.list_(pa.string()), nullable=False),  # n_steps, null = blank
        pa.field("step_target_smiles", pa.list_(pa.string()), nullable=False),  # map-free
        pa.field("temperature_c", pa.float64()),  # null: `n/a` (room temperature) or unread
        pa.field("temperature_raw", pa.string(), nullable=False),
        pa.field("citation", pa.string(), nullable=False),
        pa.field("evaluation_only", pa.bool_(), nullable=False),  # always true
    ]
)


class BenchmarkCode(StrEnum):
    WHITESPACE_STRIPPED = "WHITESPACE_STRIPPED"  # a cell with surrounding spaces (cells)
    TARGET_WAS_REACTION = "TARGET_WAS_REACTION"  # a target cell holding a record: product side
    STEP_TARGET_MISSING = "STEP_TARGET_MISSING"  # a blank target inside the steps (positions)
    PATHWAY_WITH_MISSING_TARGET = "PATHWAY_WITH_MISSING_TARGET"  # rows with such a blank
    STEP_COUNT_MISMATCH = "STEP_COUNT_MISMATCH"  # a target after `Min Steps`
    TEMPERATURE_UNREAD = "TEMPERATURE_UNREAD"  # neither `n/a` nor `<number> °C`
    TARGET_MAPPED = "TARGET_MAPPED"  # a target carrying atom-map numbers (PMechDB's drawing)
    LAST_TARGET_NOT_FINAL = "LAST_TARGET_NOT_FINAL"  # last target != final product, stereo kept
    LAST_TARGET_STEREO_ONLY = "LAST_TARGET_STEREO_ONLY"  # ... equal once stereo is dropped
    SMILES_PARSE = "SMILES_PARSE"  # RDKit cannot read a species (distinct species)
    SANITIZE_FAILED = "SANITIZE_FAILED"  # RDKit rejects its valence (distinct species)


class EvaluationOnly(ValueError):
    """A training run named an evaluation-only set."""


def refuse_training(names: Iterable[str]) -> None:
    """Raise EvaluationOnly when a training source list names the benchmark."""
    bad = sorted(set(names) & EVALUATION_ONLY)
    if bad:
        raise EvaluationOnly(f"{', '.join(bad)} is evaluation-only and never trained on")


@dataclass(frozen=True)
class BenchmarkPathway:
    pathway_id: str
    row: int
    reactants: str
    final_product: str
    formula: str
    n_steps: int
    citation: str
    temperature_c: float | None
    temperature_raw: str
    targets: tuple[str | None, ...]  # one per step; None where the cell is blank


def benchmark_id(reactants: str, final_product: str) -> str:
    """Stable id from the reactants and the final product (unique over the 350 rows)."""
    digest = hashlib.sha1(f"{reactants}>>{final_product}".encode()).hexdigest()[:16]
    return f"{NAME}-{digest}"


def parse_temperature(text: str, counts: Counter) -> float | None:
    """Degrees Celsius of `-78 °C` or `- 78 °C`; None for `n/a` or anything else (counted)."""
    if text == "n/a":
        return None
    m = _TEMPERATURE.match(text)
    if m is None:
        counts[BenchmarkCode.TEMPERATURE_UNREAD] += 1
        return None
    return float(m["value"]) * (-1 if m["sign"] else 1)


def read_target(cell: str, counts: Counter) -> str | None:
    """The species of one target cell: None when blank, the product side of a whole record."""
    if not cell:
        return None
    if ">>" in cell:
        counts[BenchmarkCode.TARGET_WAS_REACTION] += 1
        return cell.split()[0].split(">>")[1]
    counts[BenchmarkCode.TARGET_MAPPED] += bool(_MAPPED.search(cell))
    return cell


def step_targets(cells: Sequence[str], n_steps: int, counts: Counter) -> tuple[str | None, ...]:
    """`n_steps` targets (blank cells inside them None and counted); a filled cell after them
    is counted (STEP_COUNT_MISMATCH) and extends the steps."""
    filled = [k for k, c in enumerate(cells) if c]
    n = max(n_steps, filled[-1] + 1 if filled else 0)
    counts[BenchmarkCode.STEP_COUNT_MISMATCH] += n != n_steps
    targets = tuple(read_target(c, counts) for c in cells[:n])
    blanks = sum(t is None for t in targets)
    counts[BenchmarkCode.STEP_TARGET_MISSING] += blanks
    counts[BenchmarkCode.PATHWAY_WITH_MISSING_TARGET] += bool(blanks)
    return targets


def parse_row(cells: list[str], row: int, counts: Counter) -> BenchmarkPathway:
    stripped = [c.strip() for c in cells]
    counts[BenchmarkCode.WHITESPACE_STRIPPED] += sum(
        a != b for a, b in zip(cells, stripped, strict=True)
    )
    temp, reac, final, formula, n, cite, *targets = stripped
    n_steps = int(n)
    return BenchmarkPathway(
        pathway_id=benchmark_id(reac, final),
        row=row,
        reactants=reac,
        final_product=final,
        formula=formula,
        n_steps=n_steps,
        citation=cite,
        temperature_c=parse_temperature(temp, counts),
        temperature_raw=temp,
        targets=step_targets(targets, n_steps, counts),
    )


def read_benchmark(path: Path = RAW_FILE) -> tuple[list[BenchmarkPathway], Counter]:
    """Every row of the CSV, with its normalizations counted; a header other than the one
    inspected raises (the file changed and must be inspected again)."""
    counts: Counter = Counter()
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if tuple(rows[0]) != HEADER:
        raise ValueError(f"{path}: header {rows[0]} is not the inspected one {list(HEADER)}")
    out = []
    for k, cells in enumerate(rows[1:], start=1):
        if len(cells) != len(HEADER):
            raise ValueError(f"{path}: row {k} has {len(cells)} cells, not {len(HEADER)}")
        out.append(parse_row(cells, k, counts))
    if len({p.pathway_id for p in out}) != len(out):
        raise ValueError(f"{path}: two rows share reactants and final product")
    return out, counts


# ---- species


def species(smiles: str) -> list[str]:
    """Map-free canonical SMILES (stereo kept) of every fragment, as the QM batch keys them
    (`chem.unmapped_canonical` of `chem.parse_mapped`'s fragments). Raises SmilesParseError
    or SanitizeError."""
    return [chem.unmapped_canonical(f) for f in chem.fragments(chem.parse_mapped(smiles))]


def try_species(smiles: str | None, failed: dict[str, str]) -> list[str] | None:
    """`species`, or None (the reason kept in `failed`, keyed by the SMILES) when RDKit cannot
    read it or `smiles` is None."""
    if smiles is None:
        return None
    try:
        return species(smiles)
    except chem.SmilesParseError:
        failed[smiles] = str(BenchmarkCode.SMILES_PARSE)
    except (chem.SanitizeError, ValueError):
        failed[smiles] = str(BenchmarkCode.SANITIZE_FAILED)
    return None


def target_smiles(p: BenchmarkPathway, failed: dict[str, str]) -> list[str | None]:
    """Each target as one map-free canonical SMILES (fragments joined by '.'), None where the
    target is blank or unreadable."""
    out = []
    for t in p.targets:
        frags = try_species(t, failed)
        out.append(None if frags is None else ".".join(sorted(frags)))
    return out


def without_stereo(smiles: str) -> str:
    """Canonical SMILES of a canonical SMILES with its stereo dropped."""
    mol = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def last_target_check(p: BenchmarkPathway, last: str | None, counts: Counter) -> None:
    """Count a last target that is not the final product (stereo kept, then dropped)."""
    final = try_species(p.final_product, {})
    if last is None or final is None or last == ".".join(sorted(final)):
        return
    stereo_only = without_stereo(last) == without_stereo(".".join(sorted(final)))
    counts[
        BenchmarkCode.LAST_TARGET_STEREO_ONLY
        if stereo_only
        else BenchmarkCode.LAST_TARGET_NOT_FINAL
    ] += 1


def table_rows(pathways: list[BenchmarkPathway], counts: Counter) -> tuple[list[dict], dict]:
    """The table rows, and {SMILES: reason} of every species RDKit cannot read."""
    failed: dict[str, str] = {}
    rows = []
    for p in pathways:
        keys = target_smiles(p, failed)
        last_target_check(p, keys[-1] if keys else None, counts)
        try_species(p.reactants, failed)
        try_species(p.final_product, failed)
        rows.append(
            {
                "pathway_id": p.pathway_id,
                "row": p.row,
                "reactants_smiles": p.reactants,
                "final_product_smiles": p.final_product,
                "product_formula": p.formula,
                "n_steps": p.n_steps,
                "step_targets": list(p.targets),
                "step_target_smiles": keys,
                "temperature_c": p.temperature_c,
                "temperature_raw": p.temperature_raw,
                "citation": p.citation,
                "evaluation_only": True,
            }
        )
    counts.update(Counter(failed.values()))
    return rows, failed


def benchmark_species(pathways: list[BenchmarkPathway]) -> list[str]:
    """Every distinct readable species of the benchmark: reactant, target and product
    fragments."""
    out: set[str] = set()
    for p in pathways:
        for smiles in (p.reactants, p.final_product, *p.targets):
            out.update(try_species(smiles, {}) or [])
    return sorted(out)


def report(pathways: list[BenchmarkPathway], counts: Counter, failed: dict[str, str]) -> dict:
    temps = [p.temperature_c for p in pathways if p.temperature_c is not None]
    steps = Counter(p.n_steps for p in pathways)
    return {
        "evaluation_only": True,
        "pathways": len(pathways),
        "steps": sum(p.n_steps for p in pathways),
        "steps_per_pathway": {str(k): steps[k] for k in sorted(steps)},
        "with_temperature": len(temps),
        "temperature_c_range": [min(temps), max(temps)] if temps else None,
        "distinct_species": len(benchmark_species(pathways)),
        "codes": {str(k): counts[k] for k in BenchmarkCode},
        "unreadable_species": dict(sorted(failed.items())),
    }


def write_table(rows: list[dict], path: Path) -> Path:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=TABLE_SCHEMA), buf)
    return prov.atomic_write(path, buf.getvalue())


def write_benchmark(
    raw: Path = RAW_FILE, out_dir: Path = OUT_DIR
) -> tuple[Path, Path, dict, list[BenchmarkPathway]]:
    """`pathways.parquet` and `pathways_report.json` under `out_dir` (the report's provenance
    header carries the parquet's sha256 as its content hash), and the pathways written."""
    pathways, counts = read_benchmark(raw)
    rows, failed = table_rows(pathways, counts)
    table = write_table(rows, Path(out_dir) / TABLE_NAME)
    body = report(pathways, counts, failed)
    header = prov.stamp(
        KIND,
        SCHEMA_VERSION,
        prov.file_input(raw),
        {"evaluation_only": True},
        prov.file_sha256(table),
        {"table": TABLE_NAME},
    )
    return table, prov.write_json(Path(out_dir) / REPORT_NAME, body, header), body, pathways


def load_benchmark(out_dir: Path = OUT_DIR) -> pa.Table:
    """The table, after its report's provenance and the parquet's sha256 are checked."""
    report_path, table = Path(out_dir) / REPORT_NAME, Path(out_dir) / TABLE_NAME
    header = prov.verify_clean(report_path, KIND, SCHEMA_VERSION, {"evaluation_only": True})
    if prov.file_sha256(table) != header.content_sha256:
        raise prov.StaleArtifact(table, [f"its sha256 differs from {REPORT_NAME}'s content hash"])
    return pq.read_table(table, schema=TABLE_SCHEMA)


def load_pathways(out_dir: Path = OUT_DIR) -> list[BenchmarkPathway]:
    """The checked table as `BenchmarkPathway` records."""
    rows = load_benchmark(out_dir).to_pylist()
    return [
        BenchmarkPathway(
            r["pathway_id"],
            r["row"],
            r["reactants_smiles"],
            r["final_product_smiles"],
            r["product_formula"],
            r["n_steps"],
            r["citation"],
            r["temperature_c"],
            r["temperature_raw"],
            tuple(r["step_targets"]),
        )
        for r in rows
    ]
