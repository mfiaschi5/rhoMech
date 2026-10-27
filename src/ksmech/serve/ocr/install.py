"""Where the OCR engine lives and whether it is installed (the `ocr` capability).

Everything is under `data/ocr/` of the data root (gitignored): the isolated environment
`venv/` (built by `uv sync` of `tools/ocr` with `UV_PROJECT_ENVIRONMENT`) and the checkpoint
under `weights/`. The checkpoint counts only once its sha256 was checked: `ocr-setup` then writes
the marker `<checkpoint>.verified` (JSON: file, sha256, size, url, verified_at). The capability
is a few file stats, never a hash: the venv's python, the worker script, the checkpoint, and a
marker that names the expected sha256 and the checkpoint's current size.

Standard library only: the capability check runs in every `/api/meta`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ksmech.serve.core.settings import ServeSettings

ENGINE = "molscribe 1.1.1"
CHECKPOINT = "swin_base_char_aux_1m680k.pth"
CHECKPOINT_URL = f"https://huggingface.co/yujieq/MolScribe/resolve/main/{CHECKPOINT}"
CHECKPOINT_SHA256 = "32821fa4ba8d17f9a92e374601438c5e757947dbf1141e9ad0ef51cccfa1c501"
CHECKPOINT_BYTES = 1_134_173_494
NOT_INSTALLED = "OCR engine not installed: run `ksmech-serve ocr-setup`"


@dataclass(frozen=True)
class OcrPaths:
    root: Path  # data/ocr
    tools: Path  # <repo>/tools/ocr: the uv project of the environment and the worker

    @classmethod
    def of(cls, settings: ServeSettings) -> OcrPaths:
        return cls(settings.ocr, settings.repo_root / "tools" / "ocr")

    @property
    def venv(self) -> Path:
        return self.root / "venv"

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"

    @property
    def worker(self) -> Path:
        return self.tools / "ocr_worker.py"

    @property
    def weights(self) -> Path:
        return self.root / "weights"

    @property
    def checkpoint(self) -> Path:
        return self.weights / CHECKPOINT

    @property
    def partial(self) -> Path:
        """The download in progress (resumed from its size)."""
        return self.weights / f"{CHECKPOINT}.part"

    @property
    def marker(self) -> Path:
        return self.weights / f"{CHECKPOINT}.verified"

    @property
    def log(self) -> Path:
        """The worker's stderr (library output, tracebacks)."""
        return self.root / "worker.log"


def read_marker(paths: OcrPaths) -> dict | None:
    try:
        doc = json.loads(paths.marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def write_marker(paths: OcrPaths, sha256: str, size: int, url: str = CHECKPOINT_URL) -> None:
    doc = {"file": CHECKPOINT, "sha256": sha256, "size": int(size), "url": url}
    doc["verified_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    paths.marker.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def weights_verified(paths: OcrPaths) -> bool:
    """The checkpoint exists and its marker names the expected sha256 and its current size."""
    marker = read_marker(paths)
    if marker is None or marker.get("sha256") != CHECKPOINT_SHA256:
        return False
    try:
        return paths.checkpoint.stat().st_size == marker.get("size")
    except OSError:
        return False


def missing(paths: OcrPaths) -> list[str]:
    """What the engine still lacks (empty when installed)."""
    out = []
    if not paths.python.is_file():
        out.append(f"the environment {paths.venv}")
    if not paths.worker.is_file():
        out.append(f"the worker script {paths.worker}")
    if not weights_verified(paths):
        out.append(f"the verified checkpoint {paths.checkpoint}")
    return out


def engine_status(settings: ServeSettings) -> tuple[bool, str | None]:
    """(enabled, reason) of the `ocr` capability."""
    return (False, NOT_INSTALLED) if missing(OcrPaths.of(settings)) else (True, None)
