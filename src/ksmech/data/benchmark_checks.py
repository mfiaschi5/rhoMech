"""Checks on the human benchmark (`data.human_benchmark`): its overlap with PMechDB's split
parts, and how many of its species lack a QM record or tokens (reaction-network plan, item G).

Overlap: the benchmark comes from the release PMechDB comes from, so before it serves as an
evaluation set its species are looked up among the species of a PMechDB split's parts (the
fragments of both sides of each part's steps), by map-free canonical SMILES (stereo kept) and
by the identity key v2 (`data.resonance.identity_key`: sigma skeleton, H frozen, charges
zeroed, no stereo). A benchmark step is found in a part when one of the part's steps makes its
target (every target fragment among the step's products) from the step before: the previous
target's fragments among its reactants, or for step 1 at least one non-trivial reactant of the
benchmark (lenient on purpose: a leak is better over-counted). Species of at most 3 heavy atoms
(water, Na+, Cl-, ...) are counted apart, as the molecule split treats them.

Coverage: file lookups only (no index refresh, nothing written into the caches): the QM record
and the tokens of conformer 0 under each species' own method (`qm.spin.method_for`), the
failure tombstone beside a missing record, and the spin census that keeps a species out of QM.
`write_species_file` turns the coverage into the species list of the QM batch and the token
build (`ksmech-qm run --species-from`, `ksmech-orbitals build --species-from`).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem

from ksmech.data import chem
from ksmech.data import provenance as prov
from ksmech.data.human_benchmark import (
    NAME,
    OUT_DIR,
    REPORT_NAME,
    BenchmarkPathway,
    try_species,
)
from ksmech.data.io import DEFAULT_PROCESSED_DIR, steps_path
from ksmech.data.resonance import identity_key
from ksmech.data.schema import Source
from ksmech.processes import chunked, in_processes

OVERLAP_KIND = "human_benchmark_overlap"
COVERAGE_KIND = "human_benchmark_qm_coverage"
SCHEMA_VERSION = 1
VIEWS = ("smiles", "identity_key")
COVERAGE_NAME = "qm_coverage.json"
SPECIES_KIND = "human_benchmark_qm_species"
SPECIES_NAME = "qm_species.json"
DETAIL_PARTS = ("train", "any_part")  # the parts whose per-pathway detail the file keeps
CHUNK = 500

FragmentIds = list[tuple[str, str]]


def overlap_name(split: str) -> str:
    return f"overlap_pmechdb_{split}.json"


def fragment_ids(smiles: str) -> FragmentIds | None:
    """(canonical SMILES, identity key v2) of every fragment, None when RDKit cannot read it."""
    try:
        frags = chem.fragments(chem.parse_mapped(smiles))
        return [(chem.unmapped_canonical(f), identity_key(f)) for f in frags]
    except (chem.SmilesParseError, chem.SanitizeError, ValueError):
        return None


def _fragment_ids_all(sides: list[str]) -> list[FragmentIds | None]:
    return [fragment_ids(s) for s in sides]


def side_ids(sides: Iterable[str], workers: int = 1) -> dict[str, FragmentIds | None]:
    """`fragment_ids` of every distinct side, in `workers` spawned processes."""
    distinct = sorted(set(sides))
    chunks = chunked(distinct, CHUNK)
    out: dict[str, FragmentIds | None] = {}
    for chunk, ids in zip(chunks, in_processes(_fragment_ids_all, chunks, workers), strict=True):
        out.update(zip(chunk, ids, strict=True))
    return out


def nontrivial(smiles: str) -> bool:
    """More than 3 heavy atoms (the molecule split's line between reagents and molecules)."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and chem.heavy_atom_count(mol) > chem.TRIVIAL_MAX_HEAVY_ATOMS


@dataclass
class SpeciesIndex:
    """One view (SMILES or identity key) of a part: its species, and its steps as (reactant
    ids, product ids) with an index from a product id to the steps that make it."""

    species: set[str] = field(default_factory=set)
    steps: list[tuple[frozenset[str], frozenset[str]]] = field(default_factory=list)
    by_product: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    def add(self, reactants: Iterable[str], products: Iterable[str]) -> None:
        reac, prod = frozenset(reactants), frozenset(products)
        self.species |= reac | prod
        for s in prod:
            self.by_product[s].append(len(self.steps))
        self.steps.append((reac, prod))

    def makes(self, target: set[str], before: set[str] | None, any_of: bool) -> bool:
        """A step makes every id of `target` from `before` (all of it among its reactants, or
        one of it when `any_of`; no condition when `before` is None or empty)."""
        lists = [set(self.by_product.get(t, ())) for t in target]
        for i in set.intersection(*lists) if lists else ():
            reac = self.steps[i][0]
            if not before or (reac & before if any_of else before <= reac):
                return True
        return False


def part_index(steps: list[dict], ids: Iterable[str], memo: dict, counts: Counter) -> dict:
    """{view: SpeciesIndex} of the steps `ids`; a split id missing from the steps, or a side
    RDKit cannot read, is counted."""
    by_id = {s["step_id"]: s for s in steps}
    views = {v: SpeciesIndex() for v in VIEWS}
    for sid in ids:
        s = by_id.get(sid)
        if s is None:
            counts["SPLIT_ID_NOT_IN_STEPS"] += 1
            continue
        reac, prod = memo[s["reactants_smiles"]], memo[s["products_smiles"]]
        if reac is None or prod is None:
            counts["PMECHDB_SIDE_UNREADABLE"] += 1
            continue
        for k, view in enumerate(VIEWS):
            views[view].add((f[k] for f in reac), (f[k] for f in prod))
    return views


@dataclass
class BenchmarkIds:
    """A pathway's fragments: the reactants, the final product, each target (None where blank
    or unreadable)."""

    reactants: FragmentIds
    final: FragmentIds
    targets: list[FragmentIds | None]

    def all_species(self) -> set[tuple[str, str]]:
        out = set(self.reactants) | set(self.final)
        for t in self.targets:
            out |= set(t or ())
        return out


def benchmark_ids(p: BenchmarkPathway) -> BenchmarkIds:
    return BenchmarkIds(
        fragment_ids(p.reactants) or [],
        fragment_ids(p.final_product) or [],
        [None if t is None else fragment_ids(t) for t in p.targets],
    )


def steps_found(ids: BenchmarkIds, index: SpeciesIndex, view: int) -> list[int]:
    """1-based numbers of the pathway's steps that some step of `index` makes (a step after
    a blank or unreadable target is checked on its target alone)."""
    found = []
    first = {f[view] for f in ids.reactants if nontrivial(f[0])}
    for k, target in enumerate(ids.targets):
        if target is None:
            continue
        prev = ids.targets[k - 1] if k else None
        before = first if k == 0 else (None if prev is None else {f[view] for f in prev})
        if index.makes({f[view] for f in target}, before, any_of=k == 0):
            found.append(k + 1)
    return found


def species_overlap(species: set[tuple[str, str]], indexes: dict) -> dict:
    """Distinct benchmark species found in each part, per view, all and non-trivial."""
    big = {s for s in species if nontrivial(s[0])}
    found = {
        part: {
            view: {
                "all": sum(s[k] in views[view].species for s in species),
                "nontrivial": sum(s[k] in views[view].species for s in big),
            }
            for k, view in enumerate(VIEWS)
        }
        for part, views in indexes.items()
    }
    return {"distinct": len(species), "distinct_nontrivial": len(big), "found": found}


def targets_made(ids: BenchmarkIds, index: SpeciesIndex, view: int) -> int:
    """The pathway's targets that some step of `index` makes, from whatever reactants."""
    return sum(index.makes({f[view] for f in t}, None, False) for t in ids.targets if t)


def pathway_overlap(ids: BenchmarkIds, views: dict[str, SpeciesIndex]) -> dict:
    """One pathway against one part: non-trivial species found, targets made from any
    reactants, and the steps the part makes."""
    big = {s for s in ids.all_species() if nontrivial(s[0])}
    out: dict = {"nontrivial_species": len(big)}
    for k, view in enumerate(VIEWS):
        out[f"species_found_{view}"] = sum(s[k] in views[view].species for s in big)
        out[f"targets_made_{view}"] = targets_made(ids, views[view], k)
        out[f"steps_found_{view}"] = steps_found(ids, views[view], k)
    return out


def pathway_summary(per: dict[str, dict], n_targets: dict[str, int]) -> dict:
    """Over the pathways: any / every non-trivial species found, any / every step with a
    target found, the steps found, and the targets made from any reactants."""
    out = {}
    for view in VIEWS:
        sp = [(d[f"species_found_{view}"], d["nontrivial_species"]) for d in per.values()]
        st = [(len(d[f"steps_found_{view}"]), n_targets[pid]) for pid, d in per.items()]
        out[view] = {
            "pathways_with_a_species_found": sum(f > 0 for f, _ in sp),
            "pathways_with_every_species_found": sum(f == n > 0 for f, n in sp),
            "pathways_with_a_step_found": sum(f > 0 for f, _ in st),
            "pathways_with_every_step_found": sum(f == n > 0 for f, n in st),
            "steps_found": sum(f for f, _ in st),
            "targets_made": sum(d[f"targets_made_{view}"] for d in per.values()),
        }
    return out


def pmechdb_parts(processed_dir: Path, split: str) -> tuple[Path, dict[str, list[str]]]:
    """(the split file, {part: step ids}) of PMechDB's `split`, its provenance checked
    (`arrows.steps.load_split_file`), with `any_part` the union of the parts."""
    from ksmech.arrows.steps import load_split_file, splits_path

    path = splits_path(str(Source.PMECHDB), split, processed_dir)
    parts = {k.removeprefix(f"{split}_"): v for k, v in load_split_file(path).items()}
    return path, parts | {"any_part": sorted({i for ids in parts.values() for i in ids})}


def overlap(
    pathways: list[BenchmarkPathway],
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    split: str = "molecule",
    workers: int = 1,
) -> tuple[dict, list[Path]]:
    """(the overlap report, the files it read) of the benchmark against PMechDB's split."""
    split_file, parts = pmechdb_parts(processed_dir, split)
    steps_file = steps_path(Source.PMECHDB, processed_dir)
    cols = ["step_id", "reactants_smiles", "products_smiles"]
    steps = pq.read_table(steps_file, columns=cols).to_pylist()
    memo = side_ids((s[c] for s in steps for c in cols[1:]), workers)
    counts: Counter = Counter()
    indexes = {part: part_index(steps, ids, memo, counts) for part, ids in parts.items()}
    ids = {p.pathway_id: benchmark_ids(p) for p in pathways}
    n_targets = {p.pathway_id: sum(t is not None for t in p.targets) for p in pathways}
    per = {
        part: {pid: pathway_overlap(i, indexes[part]) for pid, i in ids.items()} for part in parts
    }
    body = {
        "evaluation_only": True,
        "split": split,
        "part_steps": {part: len(v) for part, v in parts.items()},
        "species": species_overlap(set().union(*(i.all_species() for i in ids.values())), indexes),
        "pathways": {part: pathway_summary(per[part], n_targets) for part in parts},
        "codes": dict(counts),
        "per_pathway": {pid: {part: per[part][pid] for part in DETAIL_PARTS} for pid in ids},
    }
    return body, [split_file, steps_file]


def write_overlap(
    pathways: list[BenchmarkPathway],
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    out_dir: Path = OUT_DIR,
    split: str = "molecule",
    workers: int = 1,
) -> tuple[Path, dict]:
    body, read = overlap(pathways, processed_dir, split, workers)
    inputs: dict[str, str] = {}
    for f in [Path(out_dir) / REPORT_NAME, *read]:
        inputs |= prov.file_input(f)
    header = prov.stamp(OVERLAP_KIND, SCHEMA_VERSION, inputs, {"split": split, "views": VIEWS})
    return prov.write_json(Path(out_dir) / overlap_name(split), body, header), body


# ---- QM and token coverage


def roles(pathways: list[BenchmarkPathway]) -> dict[str, set[str]]:
    """The distinct readable species by role: reactants, targets, final products."""
    out: dict[str, set[str]] = {"reactants": set(), "targets": set(), "final_products": set()}
    for p in pathways:
        out["reactants"].update(try_species(p.reactants, {}) or [])
        out["final_products"].update(try_species(p.final_product, {}) or [])
        for t in p.targets:
            out["targets"].update(try_species(t, {}) or [])
    return out


def cache_status(smiles: str, qm_dir: Path, tokens_dir: Path) -> tuple[str, str]:
    """(QM: record | tombstone | census_refused | missing, tokens: current | old_schema |
    no_header | missing) of conformer 0 under the species' own method."""
    from ksmech.orbitals.build import tokens_path
    from ksmech.orbitals.tokens import SCHEMA_VERSION as TOKEN_SCHEMA
    from ksmech.qm.batch import census_allows
    from ksmech.qm.failures import tombstone_path
    from ksmech.qm.records import cache_path, is_valid_cache_file
    from ksmech.qm.spin import method_for
    from ksmech.qm.xtb_calc import QMConfig

    method = method_for(smiles, QMConfig().method)
    if is_valid_cache_file(cache_path(qm_dir, smiles, 0, method)):
        qm = "record"
    elif tombstone_path(qm_dir, smiles, 0, method).is_file():
        qm = "tombstone"
    else:
        qm = "missing" if census_allows(smiles) else "census_refused"
    path = tokens_path(tokens_dir, smiles, 0, method)
    header = prov.read_header(path) if path.is_file() else None
    if not path.is_file():
        return qm, "missing"
    if header is None:
        return qm, "no_header"
    return qm, "current" if header.schema_version == TOKEN_SCHEMA else "old_schema"


def coverage(pathways: list[BenchmarkPathway], qm_dir: Path, tokens_dir: Path) -> dict:
    """Counts by QM and token status, overall and by role, and the species of every status
    but the complete ones (a record and current tokens)."""
    by_role = roles(pathways)
    every = sorted(set().union(*by_role.values()))
    status = {s: cache_status(s, qm_dir, tokens_dir) for s in every}
    lists: dict[str, list[str]] = defaultdict(list)
    for s, (qm, tok) in status.items():
        if qm != "record":
            lists[f"qm_{qm}"].append(s)
        elif tok != "current":
            lists[f"tokens_{tok}"].append(s)
    return {
        "distinct_species": len(every),
        "qm": dict(Counter(q for q, _ in status.values())),
        "tokens": dict(Counter(t for _, t in status.values())),
        "need_qm_batch": len(lists["qm_missing"]),
        "need_token_build_only": sum(len(v) for k, v in lists.items() if k.startswith("tokens_")),
        "by_role": {
            role: {
                "species": len(sp),
                "with_qm_record": sum(status[s][0] == "record" for s in sp),
                "with_current_tokens": sum(status[s][1] == "current" for s in sp),
            }
            for role, sp in by_role.items()
        },
        "species": {k: sorted(v) for k, v in sorted(lists.items())},
    }


def write_coverage(
    pathways: list[BenchmarkPathway],
    out_dir: Path = OUT_DIR,
    qm_dir: Path = Path("data/cache/qm"),
    tokens_dir: Path = Path("data/cache/tokens"),
) -> tuple[Path, dict]:
    from ksmech.qm.xtb_calc import QMConfig

    body = coverage(pathways, qm_dir, tokens_dir)
    header = prov.stamp(
        COVERAGE_KIND,
        SCHEMA_VERSION,
        prov.file_input(Path(out_dir) / REPORT_NAME),
        {"method": QMConfig().method, "conformer_id": 0},
        extra={"qm_dir": str(qm_dir), "tokens_dir": str(tokens_dir)},
    )
    return prov.write_json(Path(out_dir) / COVERAGE_NAME, body, header), body


# ---- the species file of the QM batch and the token build


def write_species_file(out_dir: Path = OUT_DIR, path: Path | None = None) -> tuple[Path, dict]:
    """`qm_species.json` (or `path`) from the existing `qm_coverage.json` of `out_dir`, its
    provenance checked: the species a QM batch or a token build can complete (no record, or a
    record without current tokens: `qm.species.coverage_species`), the other incomplete lists
    counted under `left_out`. The input of `ksmech-qm run --species-from` and of
    `ksmech-orbitals build --species-from` (`batch_commands`)."""
    from ksmech.qm.species import coverage_species

    cov_path = Path(out_dir) / COVERAGE_NAME
    cov = prov.verify(cov_path, COVERAGE_KIND, SCHEMA_VERSION)
    found, left = coverage_species(json.loads(cov_path.read_text())["species"])
    body = {"evaluation_only": True, "n_species": len(found), "left_out": left, "species": found}
    header = prov.stamp(SPECIES_KIND, SCHEMA_VERSION, prov.file_input(cov_path), cov.config)
    return prov.write_json(path or Path(out_dir) / SPECIES_NAME, body, header), body


def batch_commands(species_file: Path, name: str = NAME) -> list[str]:
    """The commands that compute the listed species' QM records, Wiberg sidecars and tokens and
    then refresh the coverage, in order (workers as CLAUDE.md's QM and token examples)."""
    listed = f"--species-from {species_file} --name {name}"
    return [
        f"uv run ksmech-qm run {listed} --workers 8",
        "uv run ksmech-qm bond-index",
        f"uv run ksmech-orbitals build {listed} --workers 4",
        "uv run ksmech-data human-benchmark",
    ]
