"""`ksmech-orbitals`: localized-orbital tokens from the GFN2 cache, of a source's molecules or of
a species list (`--species-from`, `qm.species`)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console

from ksmech.data.io import DEFAULT_PROCESSED_DIR
from ksmech.data.provenance import StaleArtifact
from ksmech.data.schema import Source
from ksmech.orbitals.build import DEFAULT_TOKENS_DIR, BuildConfig, build_dataset, build_species
from ksmech.orbitals.fgroups import DEFAULT_VOCAB_PATH, build_vocab
from ksmech.orbitals.tokens import build_molecule_tokens, token_table
from ksmech.qm import species
from ksmech.qm.batch import collect_molecules
from ksmech.qm.records import DEFAULT_CACHE_DIR, cache_path, is_valid_cache_file, load_record
from ksmech.qm.spin import method_for
from ksmech.qm.xtb_calc import QMConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-orbitals", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="one tokens file per molecule, plus a summary")
    pick = b.add_mutually_exclusive_group(required=True)
    pick.add_argument("--source", choices=[str(s) for s in Source])
    pick.add_argument(
        "--species-from",
        type=Path,
        default=None,
        help="the species a JSON file lists instead of a source's (`qm.species`), as for "
        "`ksmech-qm run --species-from`",
    )
    b.add_argument(
        "--name",
        default=None,
        help="with --species-from: the summary goes to <processed-dir>/species/<name>/ "
        "(default: the file's stem)",
    )
    b.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    b.add_argument("--qm-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    b.add_argument("--out-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--limit", type=int, default=None, help="first N molecules, for pilots")
    b.add_argument("--force", action="store_true", help="rebuild existing token files")
    v = sub.add_parser("fg-vocab", help="build the functional-group vocabulary from a dataset")
    v.add_argument("--source", required=True, choices=[str(s) for s in Source])
    v.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    v.add_argument("--min-count", type=int, default=5)
    v.add_argument("--out", type=Path, default=DEFAULT_VOCAB_PATH)
    s = sub.add_parser("show", help="print the token table of one cached molecule")
    s.add_argument("--smiles", required=True, help="map-free SMILES (canonicalized here)")
    s.add_argument("--qm-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    s.add_argument("--compute", action="store_true", help="run GFN2 if the record is missing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    console = Console()
    if args.command == "build":
        return cmd_build(args, console)
    if args.command == "fg-vocab":
        return cmd_fg_vocab(args, console)
    return cmd_show(args, console)


def cmd_build_species(args: argparse.Namespace, config: BuildConfig, console: Console) -> int:
    """`build --species-from`: the tokens of a species list's molecules (`qm.species`)."""
    try:
        name = species.run_name(args.species_from, args.name)
        species.species_dir(name, args.processed_dir)  # checks the name
        smiles, left = species.read_species_file(args.species_from)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    for key, n in left.items():
        console.print(f"[yellow]{args.species_from}: {n} species of {key!r} left out[/yellow]")
    summary = build_species(
        smiles,
        name,
        config,
        processed_dir=args.processed_dir,
        n_workers=args.workers,
        limit=args.limit,
        console=console,
    )
    return 1 if summary["n_failed"] else 0


def cmd_build(args: argparse.Namespace, console: Console) -> int:
    config = BuildConfig(qm_cache_dir=args.qm_cache_dir, out_dir=args.out_dir, force=args.force)
    if args.species_from is not None:
        return cmd_build_species(args, config, console)
    if args.name is not None:
        console.print("[red]--name names a species build: only with --species-from[/red]")
        return 2
    try:
        summary = build_dataset(
            args.source,
            config,
            processed_dir=args.processed_dir,
            n_workers=args.workers,
            limit=args.limit,
            console=console,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]; run `ksmech-data ingest --source {args.source}` first")
        return 2
    return 1 if summary["n_failed"] else 0


def cmd_fg_vocab(args: argparse.Namespace, console: Console) -> int:
    smiles_list, _ = collect_molecules(args.source, args.processed_dir)
    vocab = build_vocab(smiles_list, args.min_count, source=args.source)
    path = vocab.save(args.out)
    m = vocab.meta
    console.print(
        f"{len(vocab)} labels (incl. none/unk) from {m['n_molecules']} molecules: "
        f"{m['n_distinct_labels']} distinct, coverage {m['coverage']:.2%} at min_count "
        f"{args.min_count}; wrote {path}"
    )
    return 0


def cmd_show(args: argparse.Namespace, console: Console) -> int:
    from rdkit import Chem

    from ksmech.data import chem

    smiles = chem.unmapped_canonical(Chem.MolFromSmiles(args.smiles))
    config = QMConfig(cache_dir=args.qm_cache_dir)
    method = method_for(smiles, config.method)
    path = cache_path(config.cache_dir, smiles, 0, method)
    if not is_valid_cache_file(path):
        if not args.compute:
            console.print(f"[red]no QM record for {smiles} at {path}[/red]; add --compute")
            return 2
        from ksmech.qm.validation import compute_fixture_records

        rec = compute_fixture_records({"x": smiles}, config)["x"]
    else:
        try:
            rec = load_record(path, (smiles, 0, method))
        except StaleArtifact as exc:
            console.print(f"[red]{exc}[/red]")
            return 2
    tokens = build_molecule_tokens(rec)
    console.print(token_table(tokens))
    console.print(
        f"PM sweeps: occupied {tokens.pm_sweeps_occ}, virtual {tokens.pm_sweeps_virt}, "
        f"converged {tokens.pm_converged}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
