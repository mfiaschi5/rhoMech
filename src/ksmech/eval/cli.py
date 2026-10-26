"""`ksmech-eval`: evaluation of anchors and reaction matrices (spec v9, "Evaluation protocol").

    ksmech-eval floor --source pmechdb --split molecule
    ksmech-eval embedding --run flex_first --step-id pmechdb-018a36c64ec74e5e
    ksmech-eval embedding --run flex_first --smiles 'CC(=O)C.[OH-]'

`floor` ranks each step's anchor by |G^atom| (no learned parameter) on the validation and test
parts and writes `anchor_floor_report_<split>.json`, the reference every trained model is
compared with.

`embedding` traces one sample through a trained model (`eval.embedding`) and writes one
self-contained HTML page (`eval.embedding_page`): the tokens on the Lewis structure, the parts
of h^(0), every layer's attention and token updates, and the decoder's anchor terms. The sample
is a stored one (`--step-id`, searched in val, test, train; or `--index` of `--part`) or a pool
given as SMILES, whose missing tokens are computed first. The page goes to
`data/models/<source>/<run>/embedding/` unless `--out`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from ksmech.arrows.reactants import TokenCache
from ksmech.arrows.steps import MechanismSample, load_samples, sample_ids, samples_path
from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, source_dir, steps_path
from ksmech.data.schema import Source
from ksmech.eval.floor import floor_report
from ksmech.orbitals.build import DEFAULT_TOKENS_DIR
from ksmech.train.loop import DEFAULT_MODELS_DIR

FLOOR_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("floor", help="the |G^atom| anchor ranking on val and test")
    f.add_argument("--source", required=True, choices=[str(x) for x in Source])
    f.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    f.add_argument("--split", default="molecule", help="split strategy of the sample files")
    e = sub.add_parser("embedding", help="one sample through a trained model, as an HTML page")
    e.add_argument("--run", required=True, help="training run under <models-dir>/<source>/")
    e.add_argument("--checkpoint", default="best", choices=["best", "last"])
    e.add_argument("--source", default="pmechdb", choices=[str(x) for x in Source])
    e.add_argument("--split", default="molecule", help="split strategy of the sample files")
    e.add_argument("--part", choices=["train", "val", "test"], help="where to find the sample")
    pick = e.add_mutually_exclusive_group()
    pick.add_argument("--step-id", help="a stored sample (default: the first of --part)")
    pick.add_argument("--index", type=int, help="the k-th sample of --part (default val)")
    pick.add_argument("--smiles", help="a reactant pool, '.'-separated, no label")
    e.add_argument("--drawing", type=int, default=0, help="resonance drawing (0: the dataset's)")
    e.add_argument("--top", type=int, default=10, help="anchors listed")
    e.add_argument("--device", default="cpu")
    e.add_argument("--out", type=Path, help="the HTML page")
    e.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    e.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    e.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    return parser


def cmd_floor(args, console: Console) -> int:
    results, inputs = {}, {}
    for part in ("val", "test"):
        path = samples_path(args.source, f"{args.split}_{part}", args.processed_dir)
        inputs |= prov.file_input(path)
        results[part] = floor_report(load_samples(path))
    header = prov.stamp("anchor_floor", FLOOR_VERSION, inputs, {"split": args.split})
    out = source_dir(args.source, args.processed_dir) / f"anchor_floor_report_{args.split}.json"
    prov.write_json(out, {"scorer": "|G^atom| over valid pairs", **results}, header)
    table = Table(title=f"anchor floor, {args.source} {args.split} split")
    for col in ("part", "n", "top-1", "top-3", "top-5", "median rank", "unreachable"):
        table.add_column(col)
    for part, r in results.items():
        tops = (f"{r[f'top{k}']:.1%}" for k in (1, 3, 5))
        table.add_row(part, str(r["n"]), *tops, f"{r['median_rank']:.0f}", str(r["unreachable"]))
    console.print(table)
    console.print(f"wrote {out}")
    return 0


def _stored_sample(args) -> tuple[MechanismSample, str] | None:
    """(the sample of --step-id or --index, its part), None when there is none."""
    parts = [args.part] if args.part else (["val", "test", "train"] if args.step_id else ["val"])
    for part in parts:
        path = samples_path(args.source, f"{args.split}_{part}", args.processed_dir)
        ids = sample_ids(path)
        if args.step_id is None and 0 <= (args.index or 0) < len(ids):
            return load_samples(path, ids=[ids[args.index or 0]])[0], part
        if args.step_id in ids:
            return load_samples(path, ids=[args.step_id])[0], part
    return None


def _reaction(args, step_id: str) -> str:
    """The step's reaction SMILES from the steps parquet, "" when it has none."""
    path = steps_path(args.source, args.processed_dir)
    if not path.exists():
        return ""
    cols = ["reactants_smiles", "products_smiles"]
    df = pd.read_parquet(path, columns=cols, filters=[("step_id", "==", step_id)])
    return "" if df.empty else f"{df.iloc[0, 0]}>>{df.iloc[0, 1]}"


def _orbital_types(sample: MechanismSample, loader: TokenCache, console: Console):
    """The diagnostic orbital types of the sample's token files, None (reported) if a file is
    missing or does not match the sample."""
    from ksmech.eval.embedding_page import orbital_types

    tokens = sample.tokens(loader)
    types = None if any(t is None for t in tokens) else orbital_types(tokens)
    if types is None or len(types) != sample.n_orbitals:
        console.print("[yellow]orbital types unavailable: the page names orbitals by occupation")
        return None
    return types


def _embedding_input(args, loader: TokenCache, console: Console):
    """(sample, where, reaction) of the command, None when the sample is not found."""
    from ksmech.eval.embedding import ensure_pool, sample_from_smiles
    from ksmech.physics.textbook import fragment_keys

    if args.smiles:
        if built := ensure_pool(fragment_keys(args.smiles), loader):
            console.print(f"built tokens for {built}")
        return sample_from_smiles(args.smiles, loader), "SMILES", args.smiles
    found = _stored_sample(args)
    if found is None:
        what = args.step_id or f"index {args.index} of {args.part or 'val'}"
        console.print(f"[red]no stored sample: {what} ({args.split} split)")
        return None
    sample, part = found
    return sample, f"{args.split} {part}", _reaction(args, sample.step_id)


def cmd_embedding(args, console: Console) -> int:
    from ksmech.eval.embedding import trace
    from ksmech.eval.embedding_page import page_payload, write_page
    from ksmech.train.loop import load_model

    ckpt = args.models_dir / args.source / args.run / f"{args.checkpoint}.pt"
    model, payload = load_model(ckpt, args.device)
    loader = TokenCache(args.tokens_dir)
    found = _embedding_input(args, loader, console)
    if found is None:
        return 2
    sample, where, reaction = found
    if not 0 <= args.drawing < sample.n_drawings:
        console.print(f"[red]the sample has {sample.n_drawings} drawing(s), no {args.drawing}")
        return 2
    sample = sample.drawn(args.drawing)
    tr = trace(model, sample, args.device, args.top)
    name = sample.step_id if not args.smiles else "smiles " + args.smiles
    meta = {
        "sample": f"{name} ({where}, drawing {args.drawing})",
        "reaction": reaction,
        "run": args.run,
        "checkpoint": _checkpoint_text(args.checkpoint, payload),
        "encoder": model.cfg.encoder,
        "heads": model.cfg.heads,
    }
    slug = (
        sample.step_id
        if not args.smiles
        else "smiles-" + hashlib.sha1(args.smiles.encode()).hexdigest()[:10]
    )
    out = args.out or ckpt.parent / "embedding" / f"{slug}_d{args.drawing}.html"
    types = _orbital_types(sample, loader, console)
    write_page(page_payload(tr, sample, types, meta), out)
    _print_trace(console, tr, sample)
    console.print(f"wrote {out}")
    return 0


def _checkpoint_text(which: str, payload: dict) -> str:
    """The checkpoint, its epoch and the validation top-1 its trainer logged."""
    text = f"{which}.pt, epoch {payload.get('epoch', '?')}"
    top1 = payload.get("metrics", {}).get("val_top1")
    return text if top1 is None else f"{text}, val top-1 {top1:.1%} when trained"


def _print_trace(console: Console, tr: dict, sample: MechanismSample) -> None:
    r = tr["readout"]
    worst = max([tr["errors"]["embedding"], *tr["errors"]["layers"]])
    console.print(f"{sample.n_atoms} atoms, {sample.n_orbitals} orbitals; rebuilt layers within "
                  f"{worst:.1e} of the model")  # fmt: skip
    top = r["top"][0]
    console.print(f"top anchor {top['i']}-{top['j']} {top['side']}, P = {top['p']:.1%}")
    if "label_rank" in r:
        p_label = r["label_probability"]
        console.print(f"label rank {r['label_rank']}, P(label anchors) = {p_label:.1%}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"floor": cmd_floor, "embedding": cmd_embedding}[args.command](args, Console())


if __name__ == "__main__":
    sys.exit(main())
