"""Build token files for every unique molecule of a dataset from the GFN2 cache.

One `.npz` per (molecule, conformer) under `data/cache/tokens/`, keyed like the QM cache. A
molecule whose QM record is missing is skipped and counted, so the build can be re-run once the
QM batch has finished. Existing token files are loaded, not recomputed, unless `force` is set.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

from ksmech.data import chem
from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, steps_path
from ksmech.data.provenance import StaleArtifact
from ksmech.orbitals.classify import UNRESOLVED_TYPES, ClassifyConfig, OrbitalType
from ksmech.orbitals.descriptors import feature_names
from ksmech.orbitals.fgroups import NONE_ID, UNK_ID
from ksmech.orbitals.localize import PMConfig
from ksmech.orbitals.tokens import (
    CandidateConfig,
    MoleculeTokens,
    atom_feature_names,
    build_molecule_tokens,
    load_tokens,
    save_tokens,
    token_config,
)
from ksmech.qm.batch import collect_molecules
from ksmech.qm.records import (
    DEFAULT_CACHE_DIR,
    cache_key,
    cache_path,
    is_valid_cache_file,
    load_record,
)
from ksmech.qm.xtb_calc import QMConfig

log = logging.getLogger(__name__)

DEFAULT_TOKENS_DIR = Path("data/cache/tokens")
SUMMARY_NAME = "summary.json"
N_WORST = 20
_HIST_BIN = 10
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True)
class BuildConfig:
    qm_cache_dir: Path = DEFAULT_CACHE_DIR
    out_dir: Path = DEFAULT_TOKENS_DIR
    method: str = QMConfig().method
    conformer_id: int = 0
    force: bool = False
    pm: PMConfig = PMConfig()
    classify: ClassifyConfig = ClassifyConfig()


@dataclass
class BuildRow:
    smiles: str
    status: str  # built | loaded | missing_qm | stale_qm | failed
    detail: str = ""
    n_atoms: int = 0
    n_heavy: int = 0
    n_tokens: int = 0
    n_occ: int = 0
    n_virt: int = 0
    n_unresolved_occ: int = 0
    n_unresolved_virt: int = 0
    pm_sweeps_occ: int = 0
    pm_sweeps_virt: int = 0
    pm_converged: bool = True
    fermi_smeared: bool = False
    fg_unk: int = 0
    fg_none: int = 0
    n_metal_bound: int = 0
    types: dict[str, int] = field(default_factory=dict)
    unresolved_types: dict[str, int] = field(default_factory=dict)
    occupation_hist: dict[str, int] = field(default_factory=dict)  # occupation, 0.1 bins
    centers_hist: dict[str, int] = field(default_factory=dict)  # n_centers -> tokens
    cos_hist: dict[str, int] = field(default_factory=dict)  # cos-character bin -> tokens
    spectral_hist: dict[str, int] = field(default_factory=dict)  # spectral entries -> tokens
    masks: dict[str, int] = field(default_factory=dict)  # candidate-mask and d-type counts
    chirality: dict[str, int] = field(default_factory=dict)  # "+1" / "-1" / "0" -> atoms
    # model inputs of every token, float32, for the distributions in the summary
    orbital_inputs: np.ndarray | None = field(default=None, repr=False)
    spectral_inputs: np.ndarray | None = field(default=None, repr=False)  # top p, its E, kept
    atom_inputs: np.ndarray | None = field(default=None, repr=False)  # NaN = missing


ARRAY_FIELDS = ("orbital_inputs", "spectral_inputs", "atom_inputs")
SPECTRAL_SUMMARY = ("spectral_top_weight", "spectral_top_energy", "retained_fraction")
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def tokens_path(out_dir: Path, smiles: str, conformer_id: int, method: str) -> Path:
    return Path(out_dir) / f"{cache_key(smiles, conformer_id, method)}.npz"


def _mask_counts(t: MoleculeTokens) -> dict[str, int]:
    occ, virt = t.occupied(), t.virtual()
    d_type = [t.is_d_type(o) for o in virt]
    labelled_d = [str(o.type) == "empty_d" for o in virt]
    donors, acceptors = len(t.donor_candidates()), len(t.acceptor_candidates())
    loose = CandidateConfig(exclude_fractional=False)
    n_loose = len(t.donor_candidates(loose)) + len(t.acceptor_candidates(loose))
    return dict(
        donors=donors,
        acceptors=acceptors,
        d_type=sum(d_type),
        d_type_label_mismatch=sum(a != b for a, b in zip(d_type, labelled_d, strict=True)),
        partially_occupied=len(occ) - donors,
        fractional_band=sum(
            CandidateConfig().is_fractional(o.descriptors.occupation) for o in t.orbitals
        ),
        fractional_band_removed=n_loose - donors - acceptors,  # candidates the band takes away
        more_than_4_centres=sum(o.descriptors.n_centers > 4 for o in t.orbitals),
    )


def _input_arrays(t: MoleculeTokens) -> dict[str, np.ndarray]:
    spectral = [
        (d.spectral_weights[0], d.spectral_energies[0], d.retained_fraction)
        for d in (o.descriptors for o in t.orbitals)
    ]
    return dict(
        orbital_inputs=t.orbital_features().astype(np.float32),
        spectral_inputs=np.array(spectral, dtype=np.float32).reshape(-1, 3),
        atom_inputs=np.where(t.atom_missing(), np.nan, t.atom_features()).astype(np.float32),
    )


def _row_from_tokens(smiles: str, status: str, t: MoleculeTokens) -> BuildRow:
    occ, virt = t.occupied(), t.virtual()
    heavy = [a for a in t.atoms if a.atomic_number > 1]
    bad_occ = [o for o in occ if o.type in UNRESOLVED_TYPES]
    bad_virt = [o for o in virt if o.type in UNRESOLVED_TYPES]
    types = Counter(f"{o.kind}:{o.type}" for o in t.orbitals)
    return BuildRow(
        smiles=smiles,
        status=status,
        n_atoms=len(t.atoms),
        n_heavy=len(heavy),
        n_tokens=len(t.atoms) + len(t.orbitals),
        n_occ=len(occ),
        n_virt=len(virt),
        n_unresolved_occ=len(bad_occ),
        n_unresolved_virt=len(bad_virt),
        pm_sweeps_occ=t.pm_sweeps_occ,
        pm_sweeps_virt=t.pm_sweeps_virt,
        pm_converged=t.pm_converged,
        fermi_smeared=t.fermi_smeared,
        fg_unk=sum(a.fg_id == UNK_ID for a in heavy),
        fg_none=sum(a.fg_id == NONE_ID for a in heavy),
        n_metal_bound=sum(o.metal_bound for o in occ),
        types=dict(types),
        unresolved_types=dict(Counter(str(o.type) for o in bad_occ + bad_virt)),
        occupation_hist=dict(Counter(f"{o.descriptors.occupation:.1f}" for o in t.orbitals)),
        centers_hist=dict(Counter(str(o.descriptors.n_centers) for o in t.orbitals)),
        cos_hist=dict(Counter(_cos_bin(o) for o in t.orbitals if o.descriptors.n_centers >= 2)),
        spectral_hist=dict(Counter(str(o.descriptors.n_spectral()) for o in t.orbitals)),
        masks=_mask_counts(t),
        chirality=dict(Counter(f"{a.chirality:+.0f}" if a.chirality else "0" for a in t.atoms)),
        **_input_arrays(t),
    )


def _cos_bin(o) -> str:
    """cos-character bin of width 0.1: '0.0-0.1', ..., '0.9-1.0'."""
    lo = min(int(o.descriptors.cos_character() * 10), 9) / 10
    return f"{lo:.1f}-{lo + 0.1:.1f}"


def process_molecule(smiles: str, config: BuildConfig) -> BuildRow:
    """Build (or load) the tokens of one molecule. Never raises. A token file whose header no
    longer matches (schema, config, or the QM record it was built from) is rebuilt; a QM record
    whose header does not match this molecule and method is `stale_qm` (STALE_ARTIFACT)."""
    qm_path = cache_path(config.qm_cache_dir, smiles, config.conformer_id, config.method)
    if not is_valid_cache_file(qm_path):
        return BuildRow(smiles, "missing_qm", str(qm_path))
    out = tokens_path(config.out_dir, smiles, config.conformer_id, config.method)
    tok_config = token_config(config.pm)
    try:
        if out.is_file() and not config.force:
            try:
                return _row_from_tokens(smiles, "loaded", load_tokens(out, tok_config))
            except StaleArtifact as exc:
                log.info("rebuilding %s", exc)
        try:
            record = load_record(qm_path, (smiles, config.conformer_id, config.method))
        except StaleArtifact as exc:
            return BuildRow(smiles, "stale_qm", str(exc))
        tokens = build_molecule_tokens(record, pm_config=config.pm, classify_config=config.classify)
        save_tokens(tokens, out, prov.file_input(qm_path), tok_config)
        return _row_from_tokens(smiles, "built", tokens)
    except Exception as exc:  # noqa: BLE001 - one bad molecule must not stop the batch
        log.exception("token build failed for %s", smiles)
        return BuildRow(smiles, "failed", f"{type(exc).__name__}: {exc}")


def _worker(args: tuple[str, BuildConfig]) -> BuildRow:
    return process_molecule(*args)


@contextmanager
def single_threaded_children():
    """Spawned workers start with one BLAS / OpenMP thread each. Without it every worker's
    numpy fans out over all cores (a 3-worker build ran ~15 busy cores and, next to the QM
    batch, hard-rebooted the host; see docs/decisions.md)."""
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update(dict.fromkeys(_THREAD_VARS, "1"))
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_build(smiles_list: list[str], config: BuildConfig, n_workers: int = 1) -> list[BuildRow]:
    jobs = [(s, config) for s in smiles_list]
    if n_workers <= 1:
        return [process_molecule(*j) for j in jobs]
    with single_threaded_children(), get_context("spawn").Pool(n_workers) as pool:
        return list(pool.imap(_worker, jobs, chunksize=8))


def _hist(values: list[int], width: int) -> dict[str, int]:
    counts = Counter((v // width) * width for v in values)
    return {f"{lo}-{lo + width - 1}": counts[lo] for lo in sorted(counts)}


def _stats(values: list[int]) -> dict:
    if not values:
        return dict(min=None, median=None, max=None)
    return dict(min=min(values), median=statistics.median(values), max=max(values))


def _fraction(num: int, den: int) -> float:
    return num / den if den else float("nan")


def worst_molecules(rows: list[BuildRow], n: int = N_WORST) -> list[dict]:
    """Molecules with the largest fraction of unresolved occupied orbitals."""
    ok = [r for r in rows if r.status in ("built", "loaded") and r.n_occ]
    ok.sort(key=lambda r: (-r.n_unresolved_occ / r.n_occ, -r.n_unresolved_occ, r.smiles))
    return [
        dict(
            smiles=r.smiles,
            n_occ=r.n_occ,
            n_unresolved_occ=r.n_unresolved_occ,
            fraction=round(r.n_unresolved_occ / r.n_occ, 3),
            unresolved_types=r.unresolved_types,
        )
        for r in ok[:n]
        if r.n_unresolved_occ
    ]


def _quantiles(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)].astype(float)
    if not v.size:
        return dict(n=0)
    q = np.quantile(v, QUANTILES)
    out = {f"p{int(100 * p):d}": round(float(x), 4) for p, x in zip(QUANTILES, q, strict=True)}
    return dict(n=int(v.size), mean=round(float(v.mean()), 4), **out)


def input_distributions(rows: list[BuildRow]) -> dict:
    """Quantiles of every model input over the tokens of the built molecules."""
    blocks = {}
    for name, cols in (
        ("orbital_inputs", feature_names()),
        ("spectral_inputs", list(SPECTRAL_SUMMARY)),
        ("atom_inputs", atom_feature_names()),
    ):
        arrays = [getattr(r, name) for r in rows if getattr(r, name) is not None]
        X = np.concatenate(arrays) if arrays else np.zeros((0, len(cols)), np.float32)
        blocks[name] = {c: _quantiles(X[:, j]) for j, c in enumerate(cols)}
        if name == "atom_inputs":
            for j, c in enumerate(cols):
                blocks[name][c]["n_missing"] = int(np.isnan(X[:, j]).sum())
    return blocks


def _pm_summary(ok: list[BuildRow]) -> dict:
    occ, virt = [r.pm_sweeps_occ for r in ok], [r.pm_sweeps_virt for r in ok]
    return dict(
        n_not_converged=sum(not r.pm_converged for r in ok),
        sweeps_occ=dict(**_stats(occ), hist=_hist(occ, 5)),
        sweeps_virt=dict(**_stats(virt), hist=_hist(virt, 5)),
    )


def summarize(rows: list[BuildRow], config: BuildConfig, source: str = "") -> dict:
    ok = [r for r in rows if r.status in ("built", "loaded")]
    status = Counter(r.status for r in rows)
    types: Counter[str] = Counter()
    for r in ok:
        types.update(r.types)
    n_occ, n_virt = sum(r.n_occ for r in ok), sum(r.n_virt for r in ok)
    n_heavy = sum(r.n_heavy for r in ok)
    by_kind = {
        kind: {t: types[f"{kind}:{t}"] for t in OrbitalType if types[f"{kind}:{t}"]}
        for kind in ("occupied", "virtual")
    }
    return dict(
        source=source,
        method=config.method,
        out_dir=str(config.out_dir),
        n_molecules=len(rows),
        n_ok=len(ok),
        n_built=status["built"],
        n_loaded=status["loaded"],
        n_missing_qm=status["missing_qm"],
        n_stale_qm=status["stale_qm"],  # STALE_ARTIFACT: run `ksmech-data provenance --backfill qm`
        n_failed=status["failed"],
        failures={r.smiles: r.detail for r in rows if r.status == "failed"},
        tokens_per_molecule=dict(
            **_stats([r.n_tokens for r in ok]), hist=_hist([r.n_tokens for r in ok], _HIST_BIN)
        ),
        type_counts=by_kind,
        n_occupied=n_occ,
        n_virtual=n_virt,
        n_metal_bound_lone_pairs=sum(r.n_metal_bound for r in ok),
        occupied_unresolved_fraction=_fraction(sum(r.n_unresolved_occ for r in ok), n_occ),
        virtual_unresolved_fraction=_fraction(sum(r.n_unresolved_virt for r in ok), n_virt),
        occupied_delocalized_fraction=_fraction(
            by_kind["occupied"].get("delocalized_pi", 0), n_occ
        ),
        occupied_unknown_fraction=_fraction(by_kind["occupied"].get("unknown", 0), n_occ),
        pm=_pm_summary(ok),
        fg=dict(
            heavy_atoms=n_heavy,
            unk_fraction=_fraction(sum(r.fg_unk for r in ok), n_heavy),
            none_fraction=_fraction(sum(r.fg_none for r in ok), n_heavy),
        ),
        worst_molecules=worst_molecules(rows),
        descriptors=dict(
            occupation=_merge(r.occupation_hist for r in ok),
            centers_per_token=_merge(r.centers_hist for r in ok),
            cos_character=_merge(r.cos_hist for r in ok),
            spectral_entries_per_token=_merge(r.spectral_hist for r in ok),
        ),
        inputs=input_distributions(ok),
        masks=_sum_counts(r.masks for r in ok),
        chirality=_sum_counts(r.chirality for r in ok),
        fermi_smeared=dict(
            n_molecules=sum(r.fermi_smeared for r in ok),
            smiles=sorted(r.smiles for r in ok if r.fermi_smeared),
        ),
    )


_UNCANONICALIZABLE = "<uncanonicalizable>"  # never tokenized, so its step is never complete


def _side_keys(side: str) -> frozenset[str]:
    """Map-free canonical SMILES of the fragments of one reaction side, as `collect_molecules`
    keys them."""
    keys = set()
    for frag in chem.fragments(chem.parse_mapped(side)):
        try:
            keys.add(chem.unmapped_canonical(frag))
        except (chem.SanitizeError, ValueError):
            keys.add(_UNCANONICALIZABLE)
    return frozenset(keys)


def smeared_steps(source: str, processed_dir: Path, rows: list[BuildRow]) -> dict:
    """Steps with a Fermi-smeared reactant or product, over every accepted step ("all") and
    over the training target ("training": resonance steps dropped). `fully_tokenized` counts
    the steps whose molecules all have tokens, the denominator while the QM batch is partial."""
    cols = ["reactants_smiles", "products_smiles", "resonance_step"]
    table = pq.read_table(steps_path(source, processed_dir), columns=cols)
    ok = {r.smiles for r in rows if r.status in ("built", "loaded")}
    smeared = {r.smiles for r in rows if r.fermi_smeared}
    names = ("steps", "touching_smeared", "fully_tokenized", "fully_tokenized_touching_smeared")
    counts = {split: dict.fromkeys(names, 0) for split in ("all", "training")}
    memo: dict[str, frozenset[str]] = {}
    columns = [table.column(c).to_pylist() for c in cols]
    for reactants, products, resonance in zip(*columns, strict=True):
        keys = set()
        for side in (reactants, products):
            if side not in memo:
                memo[side] = _side_keys(side)
            keys |= memo[side]
        hit, complete = bool(keys & smeared), keys <= ok
        for split in ("all",) if resonance else ("all", "training"):
            c = counts[split]
            c["steps"] += 1
            c["touching_smeared"] += hit
            c["fully_tokenized"] += complete
            c["fully_tokenized_touching_smeared"] += hit and complete
    return counts


def _sum_counts(dicts) -> dict[str, int]:
    """Key-wise sum that keeps zero counts (Counter addition drops them)."""
    total: dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            total[k] = total.get(k, 0) + int(v)
    return total


def _merge(hists) -> dict[str, int]:
    total: Counter[str] = Counter()
    for h in hists:
        total.update(h)
    return {k: total[k] for k in sorted(total, key=lambda x: float(x.split("-")[0]))}


def _print_inputs(summary: dict, console: Console) -> None:
    """One row per model input: 5th percentile, median, 95th percentile (and missing)."""
    for block, title in (
        ("orbital_inputs", "Orbital token inputs (continuous)"),
        ("spectral_inputs", "Spectral set (top entry) and retained weight"),
        ("atom_inputs", "Atom token inputs (continuous)"),
    ):
        table = Table(title=title)
        for col in ("input", "n", "p5", "median", "p95", "missing"):
            table.add_column(col, justify="left" if col == "input" else "right")
        for name, q in summary["inputs"][block].items():
            cells = [str(q.get(k, "")) for k in ("n", "p5", "p50", "p95")]
            table.add_row(name, *cells, str(q.get("n_missing", "")))
        console.print(table)
    console.print(f"masks: {summary['masks']}")
    console.print(f"chirality (atoms): {summary['chirality']}")


def _print_type_table(summary: dict, console: Console) -> None:
    table = Table(title="Orbital types")
    for col in ("kind", "type", "count", "fraction of kind"):
        table.add_column(col, justify="right" if col in ("count", "fraction of kind") else "left")
    for kind, total in (("occupied", summary["n_occupied"]), ("virtual", summary["n_virtual"])):
        for t, n in sorted(summary["type_counts"][kind].items(), key=lambda kv: -kv[1]):
            table.add_row(kind, t, str(n), f"{_fraction(n, total):.2%}")
    console.print(table)


def _print_descriptor_distributions(summary: dict, console: Console) -> None:
    d = summary["descriptors"]
    for title, hist in (
        ("occupation (0.1 bins; 1.0 = SOMO)", d["occupation"]),
        ("centres per token", d["centers_per_token"]),
        ("cos-character (tokens with >= 2 centres)", d["cos_character"]),
        ("spectral entries per token", d["spectral_entries_per_token"]),
    ):
        total = sum(hist.values()) or 1
        console.print(
            f"{title}: " + "  ".join(f"{k}: {v} ({v / total:.1%})" for k, v in hist.items())
        )


def _print_fermi_smeared(summary: dict, console: Console) -> None:
    fs = summary["fermi_smeared"]
    console.print(
        f"Fermi-smeared molecules (an MO > 0.05 from its aufbau occupation): "
        f"{fs['n_molecules']} of {summary['n_ok']}"
    )
    for split, c in fs.get("steps", {}).items():
        console.print(
            f"  {split} steps: {c['touching_smeared']} of {c['steps']} touch one; "
            f"{c['fully_tokenized_touching_smeared']} of the {c['fully_tokenized']} steps "
            "whose molecules all have tokens"
        )


def print_summary(summary: dict, console: Console, show_worst: bool = False) -> None:
    s = summary
    console.print(
        f"[bold]Orbital tokens[/bold] ({s['method']}): {s['n_ok']}/{s['n_molecules']} molecules "
        f"({s['n_built']} built, {s['n_loaded']} loaded), {s['n_missing_qm']} without a QM "
        f"record, {s.get('n_stale_qm', 0)} with a stale QM record (STALE_ARTIFACT), "
        f"{s['n_failed']} failed"
    )
    tpm = s["tokens_per_molecule"]
    console.print(
        f"tokens per molecule: min {tpm['min']}, median {tpm['median']}, max {tpm['max']}"
    )
    console.print("  " + "  ".join(f"{k}: {v}" for k, v in tpm["hist"].items()))
    _print_type_table(s, console)
    console.print(
        f"occupied unresolved: {s['occupied_unresolved_fraction']:.2%} "
        f"(delocalized_pi {s['occupied_delocalized_fraction']:.2%}, "
        f"unknown {s['occupied_unknown_fraction']:.2%}); "
        f"virtual unresolved: {s['virtual_unresolved_fraction']:.2%}"
    )
    pm = s["pm"]
    console.print(
        f"PM sweeps: occupied median {pm['sweeps_occ']['median']} (max {pm['sweeps_occ']['max']}), "
        f"virtual median {pm['sweeps_virt']['median']} (max {pm['sweeps_virt']['max']}), "
        f"not converged: {pm['n_not_converged']}"
    )
    console.print(
        f"functional groups: {s['fg']['unk_fraction']:.2%} of heavy atoms in an UNK group, "
        f"{s['fg']['none_fraction']:.2%} in no group"
    )
    _print_descriptor_distributions(s, console)
    _print_inputs(s, console)
    _print_fermi_smeared(s, console)
    if s["failures"]:
        console.print(
            f"[red]{len(s['failures'])} failures[/red]; first: {next(iter(s['failures'].items()))}"
        )
    if show_worst and s["worst_molecules"]:
        table = Table(title=f"Worst {len(s['worst_molecules'])} molecules (unresolved occupied)")
        for col in ("smiles", "n_occ", "unresolved", "fraction", "types"):
            table.add_column(
                col, justify="right" if col in ("n_occ", "unresolved", "fraction") else "left"
            )
        for w in s["worst_molecules"]:
            types = ", ".join(f"{k} {v}" for k, v in w["unresolved_types"].items())
            table.add_row(
                w["smiles"][:70],
                str(w["n_occ"]),
                str(w["n_unresolved_occ"]),
                f"{w['fraction']:.2f}",
                types,
            )
        console.print(table)


def build_dataset(
    source: str,
    config: BuildConfig | None = None,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    n_workers: int = 1,
    limit: int | None = None,
    console: Console | None = None,
) -> dict:
    """Build tokens for every unique molecule of `source`; write and return the summary."""
    config = config or BuildConfig()
    console = console or Console()
    smiles_list, canon_failures = collect_molecules(source, processed_dir)
    if limit is not None:
        smiles_list = smiles_list[:limit]
    console.print(
        f"{len(smiles_list)} unique molecules ({len(canon_failures)} not canonicalizable), "
        f"QM cache {config.qm_cache_dir}, tokens -> {config.out_dir} ({n_workers} workers)"
    )
    rows = run_build(smiles_list, config, n_workers)
    summary = summarize(rows, config, source)
    summary["fermi_smeared"]["steps"] = smeared_steps(source, processed_dir, rows)
    out = Path(config.out_dir) / SUMMARY_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print_summary(summary, console, show_worst=summary["occupied_unresolved_fraction"] > 0.10)
    console.print(f"wrote summary: {out}")
    return summary


def rows_to_dicts(rows: list[BuildRow]) -> list[dict]:
    """Rows as plain dicts, without the per-token input arrays."""
    keep = [f.name for f in fields(BuildRow) if f.name not in ARRAY_FIELDS]
    return [{k: v for k, v in asdict(r).items() if k in keep} for r in rows]
