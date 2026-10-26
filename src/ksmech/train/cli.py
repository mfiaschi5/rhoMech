"""`ksmech-train`: train the model (spec v9) on one split of a source's sample store.

    ksmech-train fit --source pmechdb --split molecule --run first
    ksmech-train fit --run smoke --limit 64 --epochs 1        # a smoke run

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
from ksmech.model.config import ModelConfig
from ksmech.train.loop import DEFAULT_MODELS_DIR, TrainConfig, fit, rocm_override, select_device


def _add_fields(parser: argparse.ArgumentParser, cls) -> None:
    """One option per field of a configuration dataclass, with its default."""
    for f in fields(cls):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        elif isinstance(f.default, tuple):
            parser.add_argument(flag, nargs="*", default=list(f.default))
        else:
            parser.add_argument(flag, type=type(f.default), default=f.default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-train", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit", help="train one run")
    f.add_argument("--run", required=True, help="run name: data/models/<source>/<run>/")
    f.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    f.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    f.add_argument("--device", default="auto", help="auto | cpu | cuda (ROCm is cuda)")
    f.add_argument("--resume", action="store_true", help="continue from the run's last.pt")
    _add_fields(f, TrainConfig)
    _add_fields(f, ModelConfig)
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
    if override := rocm_override():
        logging.getLogger(__name__).info("set HSA_OVERRIDE_GFX_VERSION=%s (RDNA2 card)", override)
    best = fit(cfg, model_cfg, run_dir, select_device(args.device), args.processed_dir, args.resume)
    logging.getLogger(__name__).info("best epoch %s: %s", best["epoch"], best)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"fit": cmd_fit}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
