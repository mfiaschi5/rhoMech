"""`ksmech-serve ocr-setup`: install the OCR engine (MolScribe 1.1.1) under `data/ocr/`.

    ksmech-serve ocr-setup [--relock] [--uv UV] [--skip-env] [--skip-weights]

Two steps (`serve.ocr.provision`): the isolated environment (`uv lock` when `tools/ocr/uv.lock`
is missing or with `--relock`, then `uv sync --locked`, into `data/ocr/venv`), then the
checkpoint (about 1.1 GB from Hugging Face, resumable, sha256-checked). Run it again after an
interruption: what is done is kept. It downloads and installs third-party code and weights, so
it is never run by the server or the tests. A running server sees the engine at its next
request (the capability is a file check). Exit 0 installed, 1 a step failed.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ksmech.serve.core.settings import ServeSettings


def add_parser(sub) -> None:
    p = sub.add_parser(
        "ocr-setup",
        help="install the OCR engine (MolScribe) into data/ocr: its own environment and weights",
    )
    p.add_argument("--uv", default="uv", help="the uv executable (default: uv on PATH)")
    p.add_argument("--relock", action="store_true", help="re-resolve tools/ocr/uv.lock first")
    p.add_argument("--skip-env", action="store_true", help="do not build the environment")
    p.add_argument("--skip-weights", action="store_true", help="do not fetch the checkpoint")


def _say(message: str) -> None:
    print(f"ocr-setup: {message}", flush=True)


def run(args: argparse.Namespace, settings: ServeSettings, **seams) -> int:
    """`seams` (tests): `run` for `sync_env`, `opener` and `sleep` for `download_checkpoint`."""
    from ksmech.serve.core.settings import assert_writable
    from ksmech.serve.ocr import install, provision

    paths = install.OcrPaths.of(settings)
    try:
        assert_writable(paths.root, settings)
        if not args.skip_env:
            env_seams = {k: v for k, v in seams.items() if k == "run"}
            provision.sync_env(paths, uv=args.uv, relock=args.relock, say=_say, **env_seams)
        if not args.skip_weights:
            dl_seams = {k: v for k, v in seams.items() if k in ("opener", "sleep")}
            provision.download_checkpoint(paths, say=_say, **dl_seams)
    except (provision.SetupError, ValueError) as exc:
        print(f"ksmech-serve ocr-setup: {exc}", file=sys.stderr)
        return 1
    lacking = install.missing(paths)
    if lacking:
        _say(f"not installed yet: missing {'; '.join(lacking)}")
        return 1
    _say(f"{install.ENGINE} installed: the ocr capability is on")
    return 0
