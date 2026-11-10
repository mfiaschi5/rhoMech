"""`ksmech-eval roundtrip`: the round-trip check as a reranker of the forward beam
(`eval.roundtrip`; reaction-network plan, items H and I5).

    ksmech-eval roundtrip --run orb_a_s0 --backward-run bwd_s0 --cohort closed_base
    ksmech-eval roundtrip --run orb_a_s0 --cohort closed_base --species-file FILE

The forward run decodes the stored samples of one part as `decode` does (the same options, the
same file suffix); the backward run (`ksmech-train fit --direction backward`) scores -R of each
of the first `--top` completions at its child; the beam is reranked for each of `--lambdas`
(0 is always added: the forward order). It writes, beside the forward checkpoint,
`roundtrip_report_<split>_<part><suffix>_<backward run>.json` and one `decode_hits` file per
lambda, `roundtrip_hits_<split>_<part><suffix>_<backward run>_lam<lambda>.json`, which
`ksmech-eval compare` pairs with another lambda's or with the decode's hits.

`--species-file FILE` decodes only and writes the child species of the first `--top`
completions that have no usable tokens (`{"species": [...]}`, `qm.species.read_species_file`),
then prints the batch's commands: `ksmech-qm run --species-from FILE`, `ksmech-qm bond-index`,
`ksmech-orbitals build --species-from FILE`. The forward and backward checkpoints' directions
are checked (a backward run as the forward one, or the reverse, is refused).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ksmech.arrows.backward import BACKWARD, FORWARD, require_direction
from ksmech.arrows.reactants import TokenCache
from ksmech.arrows.steps import load_samples, samples_path
from ksmech.data import provenance as prov
from ksmech.orbitals.build import DEFAULT_TOKENS_DIR

ROUNDTRIP_VERSION = 1
SPECIES_KIND = "roundtrip_qm_species"


def add_parser(sub) -> None:
    from ksmech.eval.cli import decode_args

    r = sub.add_parser("roundtrip", help="the forward beam reranked by the backward model")
    decode_args(r)
    r.add_argument("--backward-run", help="the backward run under <models-dir>/<run-source>/ "
                   "(required unless --species-file)")  # fmt: skip
    r.add_argument("--backward-checkpoint", default="best", choices=["best", "last"])
    r.add_argument("--lambdas", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0],
                   help="weights of log P_b(-R | child); 0 is always added")  # fmt: skip
    r.add_argument("--top", type=int, default=5, help="completions rescored, best first")
    r.add_argument("--species-file", type=Path, help="decode only: write the child species "
                   "without tokens and print the batch commands")  # fmt: skip
    r.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    r.set_defaults(direction=FORWARD)


def _checkpoint(args, run: str, which: str, direction: str):
    """(path, model, payload) of a run's checkpoint; ValueError when it was trained in the
    other direction."""
    from ksmech.train.loop import load_model

    ckpt = args.models_dir / (args.run_source or args.source) / run / f"{which}.pt"
    model, payload = load_model(ckpt, "cpu")
    trained = payload.get("provenance", {}).get("config", {}).get("train", {})
    if trained.get("direction", FORWARD) != direction:
        raise ValueError(f"{ckpt}: trained {trained.get('direction', FORWARD)}, need {direction}")
    return ckpt, model, payload


def _setup(args, console: Console):
    """The forward checkpoint, its decoder, the part's samples and the decode config shared by
    every output (`decode`'s keys), with the inputs."""
    from ksmech.eval import cli
    from ksmech.eval.decode import sample_decoder
    from ksmech.model.decode import DecodeConfig

    caps, caps_input = cli._decode_caps(args)
    ckpt, model, payload = _checkpoint(args, args.run, args.checkpoint, FORWARD)
    path = samples_path(args.source, f"{args.split}_{args.part}", args.processed_dir)
    require_direction(path, FORWARD)
    ids = cli._cohort_ids(args, path)
    ids = ids[: args.limit] if args.limit is not None else ids
    minus1 = args.closed_minus1_filter
    cfg = DecodeConfig(anchors=args.beam, per_anchor=args.beam, closed_minus1_filter=minus1, **caps)
    keys = ("source", "split", "part", "limit", "beam", "scope", "cohort", "slice")
    config = {k: getattr(args, k) for k in (*keys, "closed_minus1_filter")}
    config |= {"run_source": args.run_source or args.source}
    config |= {"checkpoint": cli._checkpoint_text(args.checkpoint, payload)}
    config |= cli._sample_versions(path, payload, console) | {"caps": args.caps, **caps}
    inputs = prov.file_input(path) | prov.file_input(ckpt) | caps_input
    decode = sample_decoder(model, cfg, args.scope)
    return ckpt, decode, load_samples(path, ids=ids), config, inputs, caps


def _species_file(args, console: Console, decode, samples, config, inputs) -> int:
    from ksmech.eval.roundtrip import missing_species

    skips: Counter = Counter()
    found = missing_species(samples, decode, TokenCache(args.tokens_dir), args.top, skips)
    body = {"evaluation_only": True, "n_species": len(found),
            "reasons": dict(Counter(found.values())), "skips": dict(sorted(skips.items())),
            "species": sorted(found)}  # fmt: skip
    config = config | {"top": args.top, "run": args.run}
    prov.write_json(args.species_file, body, prov.stamp(SPECIES_KIND, 1, inputs, config))
    console.print(f"{len(found)} child species without tokens -> {args.species_file}")
    name = args.species_file.stem
    listed = f"--species-from {args.species_file} --name {name}"
    for line in (f"uv run ksmech-qm run {listed} --workers 8", "uv run ksmech-qm bond-index",
                 f"uv run ksmech-orbitals build {listed} --workers 4"):  # fmt: skip
        console.print(f"  {line}")
    return 0


def _print(console: Console, report: dict, title: str) -> None:
    table = Table(title=f"round-trip rerank, {title}")
    for col in ("lambda", "top-1", "top-3", "top-5", "top-1 any", "gained", "lost", "top-1 full"):
        table.add_column(col)
    for lam, r in report["by_lambda"].items():
        cells = [f"{r[f'top{k}']:.1%}" for k in (1, 3, 5)] + [f"{r['top1_any']:.1%}"]
        cells += [str(r["gained_top1"]), str(r["lost_top1"]), f"{r['full']['top1']:.1%}"]
        table.add_row(lam, *cells)
    console.print(table)
    cov = report["coverage"]
    console.print(f"backward scores: {cov['scored']} of {cov['completions']} completions; "
                  f"{report['n_full']} of {report['n']} samples fully scored")  # fmt: skip
    for code, n in cov["unscored"].items():
        console.print(f"  {code}: {n}")


def cmd_roundtrip(args, console: Console) -> int:
    from ksmech.eval import cli
    from ksmech.eval.roundtrip import roundtrip_report, roundtrip_rows
    from ksmech.network.pools import ModelExpander

    ckpt, decode, samples, config, inputs, caps = _setup(args, console)
    if args.species_file is not None:
        return _species_file(args, console, decode, samples, config, inputs)
    if not args.backward_run:
        console.print("[red]--backward-run is required (or --species-file)[/red]")
        return 2
    bckpt, bmodel, bpayload = _checkpoint(args, args.backward_run, args.backward_checkpoint,
                                          BACKWARD)  # fmt: skip
    scorer = ModelExpander(bmodel, TokenCache(args.tokens_dir))
    lambdas = sorted({0.0, *args.lambdas})
    outcomes, sibling_skips, sibling_inputs = cli._outcomes(args, console)
    skips: Counter = Counter()
    rows = roundtrip_rows(samples, decode, scorer, lambdas, args.top, outcomes, skips)
    report = roundtrip_report(rows, lambdas) | {"skips": dict(sorted(skips.items()))}
    rerank = {"backward_run": args.backward_run,
              "backward_checkpoint": cli._checkpoint_text(args.backward_checkpoint, bpayload),
              "top": args.top, "backward_score": "marginal"}  # fmt: skip
    inputs = inputs | prov.file_input(bckpt) | sibling_inputs
    suffix = cli.decode_suffix(args, args.run_source or args.source, caps)
    stem = f"{args.split}_{args.part}{suffix}_{args.backward_run}"
    out = ckpt.parent / f"roundtrip_report_{stem}.json"
    header = prov.stamp("roundtrip", ROUNDTRIP_VERSION, inputs, config | {"rerank": rerank})
    prov.write_json(out, report | {"per_sample": rows}, header)
    for lam in (f"{x:g}" for x in lambdas):
        hits = [r["by_lambda"][lam] for r in rows]
        conf = config | {"rerank": rerank | {"lambda": float(lam)}}
        path = ckpt.parent / f"roundtrip_hits_{stem}_lam{lam}.json"
        prov.write_json(path, {"per_sample": hits},
                        prov.stamp("decode_hits", cli.DECODE_VERSION, inputs, conf))  # fmt: skip
    _print(console, report, f"{args.run} + {args.backward_run}, {args.split} {args.part}")
    console.print(f"wrote {out.name} and {len(lambdas)} hit files beside it")
    return 0
