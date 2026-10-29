"""`ksmech-train`: train the model (spec v9) on one split of a source's sample store.

    ksmech-train fit --source pmechdb --split molecule --run first
    ksmech-train fit --run smoke --limit 64 --epochs 1        # a smoke run
    ksmech-train fit --run dense_plain_general_s0 --preset general --seed 0 --no-resonance-draws

`--preset general|legacy` sets the P0a flags of `model.config.PRESETS` as the defaults of their
options (GENERAL: every pair cell, the orbital extras; LEGACY: the v17 inputs); an option given
explicitly wins. run.json records the resulting flags.

On the ROCm build an RDNA2 card without kernels of its own (the RX 6600 XT, gfx1032) gets
HSA_OVERRIDE_GFX_VERSION=10.3.0 unless the variable is already set (`train.loop.rocm_override`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import fields
from pathlib import Path

from ksmech.data.io import DEFAULT_PROCESSED_DIR
from ksmech.model.config import PRESETS, ModelConfig
from ksmech.train.loop import (
    DEFAULT_MODELS_DIR,
    NonFiniteError,
    TrainConfig,
    fit,
    rocm_override,
    select_device,
)


def _add_fields(parser: argparse.ArgumentParser, cls, defaults: dict | None = None) -> None:
    """One option per field of a configuration dataclass, with its default (or `defaults`')."""
    for f in fields(cls):
        flag = "--" + f.name.replace("_", "-")
        default = (defaults or {}).get(f.name, f.default)
        if isinstance(f.default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(f.default, tuple):
            parser.add_argument(flag, nargs="*", default=list(default))
        else:
            parser.add_argument(flag, type=type(f.default), default=default)


def _preset(argv: list[str] | None) -> str | None:
    """The --preset of a command line, read before the full parser is built."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", choices=sorted(PRESETS))
    return pre.parse_known_args(sys.argv[1:] if argv is None else argv)[0].preset


def build_parser(preset: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-train", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit", help="train one run")
    f.add_argument("--run", required=True, help="run name: data/models/<source>/<run>/")
    f.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    f.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    f.add_argument("--device", default="auto", help="auto | cpu | cuda (ROCm is cuda)")
    f.add_argument("--resume", action="store_true", help="continue from the run's last.pt")
    f.add_argument("--preset", choices=sorted(PRESETS), default=preset,
                   help="the P0a flags of GENERAL or LEGACY as defaults")  # fmt: skip
    _add_fields(f, TrainConfig)
    _add_fields(f, ModelConfig, PRESETS.get(preset or "", {}))
    return parser


def _config(cls, args):
    kw = {x.name: getattr(args, x.name) for x in fields(cls)}
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in kw.items()})


def _logging(run_dir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stderr), logging.FileHandler(run_dir / "train.log")):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def cmd_fit(args) -> int:
    cfg, model_cfg = _config(TrainConfig, args), _config(ModelConfig, args)
    run_dir = Path(args.models_dir) / cfg.source / args.run
    if (run_dir / "run.json").exists() and not args.resume:
        print(f"{run_dir} exists: pass --resume or choose another --run", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    _logging(run_dir)
    if args.preset:
        flags = {k: getattr(model_cfg, k) for k in PRESETS[args.preset]}
        logging.getLogger(__name__).info("preset %s: %s", args.preset, flags)
    if override := rocm_override():
        logging.getLogger(__name__).info("set HSA_OVERRIDE_GFX_VERSION=%s (RDNA2 card)", override)
    try:
        best = fit(
            cfg, model_cfg, run_dir, select_device(args.device), args.processed_dir, args.resume
        )
    except NonFiniteError as err:  # last.pt and best.pt hold the last finite epoch
        logging.getLogger(__name__).error("NONFINITE: %s; the run stopped", err)
        return 3
    logging.getLogger(__name__).info("best epoch %s: %s", best["epoch"], best)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser(_preset(argv)).parse_args(argv)
    return {"fit": cmd_fit}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
