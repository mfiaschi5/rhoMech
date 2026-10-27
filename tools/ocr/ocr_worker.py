"""The MolScribe worker of ksmech-serve's OCR (`ksmech.serve.ocr.engine.JsonLinesWorker`).

    data/ocr/venv/bin/python -I tools/ocr/ocr_worker.py --weights CKPT [--threads N]

It runs in the isolated environment `data/ocr/venv` (tools/ocr/pyproject.toml), never in the
project's, and imports nothing of ksmech. The model loads once; then one JSON line in, one out:

    out, first:  {"ready": true, "engine": "molscribe 1.1.1", "threads": 2}
                 (or {"ready": false, "error": "..."} and exit 1 when the model does not load)
    in:          {"id": 7, "image": "/abs/path/image.png"}
    out:         {"id": 7, "ok": true, "smiles": "...", "molfile": "...", "confidence": 0.98,
                  "width": 640, "height": 480}
                 or {"id": 7, "ok": false, "error": "..."}
    in:          {"op": "quit"}, or the end of stdin: exit 0

stdout carries only these lines: file descriptor 1 is pointed at stderr before the libraries
load, so whatever they print lands in the server's worker log. Torch uses `--threads` threads
(else KSMECH_OCR_THREADS, else 2).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections.abc import Callable
from typing import IO, Any

ENGINE = "molscribe 1.1.1"
THREADS_ENV = "KSMECH_OCR_THREADS"
DEFAULT_THREADS = 2

Predict = Callable[[str], dict]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MolScribe over JSON lines (stdin / stdout)")
    p.add_argument("--weights", required=True, help="the MolScribe checkpoint (.pth)")
    p.add_argument("--threads", type=int, default=None, help=f"torch threads ({THREADS_ENV})")
    args = p.parse_args(argv)
    if args.threads is None:
        raw = os.environ.get(THREADS_ENV, "").strip()
        args.threads = int(raw) if raw.isdigit() else DEFAULT_THREADS
    args.threads = max(1, args.threads)
    return args


def protocol_stream() -> IO[str]:
    """A private copy of stdout for the protocol; fd 1 then goes to stderr."""
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return proto


def load(weights: str, threads: int) -> Predict:
    """MolScribe on the CPU with `threads` torch threads, as path -> answer fields."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)
    import torch
    from molscribe import MolScribe
    from PIL import Image

    torch.set_num_threads(threads)
    model = MolScribe(weights, device=torch.device("cpu"))

    def predict(path: str) -> dict:
        with Image.open(path) as img:
            width, height = img.size
        out = model.predict_image_file(path, return_atoms_bonds=False, return_confidence=True)
        confidence = out.get("confidence")
        return {
            "smiles": out.get("smiles"),
            "molfile": out.get("molfile"),
            "confidence": None if confidence is None else float(confidence),
            "width": width,
            "height": height,
        }

    return predict


def answer(predict: Predict, request: Any) -> dict:
    """The answer line of one request line (already decoded)."""
    rid = request.get("id") if isinstance(request, dict) else None
    image = request.get("image") if isinstance(request, dict) else None
    if not isinstance(image, str) or not image:
        return {"id": rid, "ok": False, "error": 'expected {"id": ..., "image": PATH}'}
    if not os.path.isfile(image):
        return {"id": rid, "ok": False, "error": f"no such file: {image}"}
    try:
        return {"id": rid, "ok": True, **predict(image)}
    except Exception as exc:  # noqa: BLE001 - one bad image must not end the worker
        traceback.print_exc(file=sys.stderr)
        return {"id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _write(out: IO[str], msg: dict) -> None:
    out.write(json.dumps(msg) + "\n")
    out.flush()


def serve(predict: Predict, inp: IO[str], out: IO[str]) -> int:
    """Answer request lines until {"op": "quit"} or the end of `inp`."""
    for line in inp:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except ValueError:
            _write(out, {"id": None, "ok": False, "error": "not a JSON line"})
            continue
        if isinstance(request, dict) and request.get("op") == "quit":
            break
        _write(out, answer(predict, request))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    proto = protocol_stream()
    try:
        predict = load(args.weights, args.threads)
    except Exception as exc:  # noqa: BLE001 - reported on the protocol, then exit 1
        traceback.print_exc(file=sys.stderr)
        _write(proto, {"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    _write(proto, {"ready": True, "engine": ENGINE, "threads": args.threads})
    return serve(predict, sys.stdin, proto)


if __name__ == "__main__":
    sys.exit(main())
