"""The OCR service: one engine per app, created on first use and closed at shutdown.

`recognize` takes the uploaded bytes: `images.to_png` checks and converts them, the PNG goes to
a temporary file for the worker, `structure.structure_of` checks the answer with RDKit, and the
result is `OcrOut`. The capability (`install.engine_status`) is checked by the route before any
byte is read; an injected engine (tests, another engine later) replaces MolScribe.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from ksmech.serve.core import schemas as S
from ksmech.serve.core.capabilities import status
from ksmech.serve.ocr import images
from ksmech.serve.ocr.engine import MolScribeEngine, OcrEngine
from ksmech.serve.ocr.install import ENGINE, OcrPaths
from ksmech.serve.ocr.structure import structure_of

if TYPE_CHECKING:
    from ksmech.serve.core.settings import ServeSettings

log = logging.getLogger(__name__)

TIMEOUT_S = 60.0  # one image, once the worker is ready


class OcrService:
    def __init__(
        self,
        settings: ServeSettings,
        engine: OcrEngine | None = None,
        timeout_s: float = TIMEOUT_S,
    ) -> None:
        self.settings = settings
        self.paths = OcrPaths.of(settings)
        self.timeout_s = float(timeout_s)
        self._engine = engine
        self._lock = threading.Lock()

    @property
    def engine_name(self) -> str:
        return self._engine.name if self._engine is not None else ENGINE

    def engine(self) -> OcrEngine:
        with self._lock:
            if self._engine is None:
                self._engine = MolScribeEngine(self.paths)
            return self._engine

    def status_out(self) -> S.OcrStatusOut:
        on, reason = status("ocr", self.settings)
        return S.OcrStatusOut(enabled=on, reason=reason, engine=self.engine_name)

    def _read(self, engine: OcrEngine, png: bytes):
        with tempfile.TemporaryDirectory(prefix="ksmech-ocr-") as tmp:
            path = Path(tmp) / "image.png"
            path.write_bytes(png)
            return engine.recognize(path, self.timeout_s)

    def recognize(self, data: bytes) -> S.OcrOut:
        started = time.monotonic()
        prepared = images.to_png(data)
        engine = self.engine()
        raw = self._read(engine, prepared.png)
        if raw.width is None or raw.height is None:  # the engine did not say: the PNG it read
            raw = replace(raw, width=prepared.width, height=prepared.height)
        found = structure_of(raw)
        elapsed = time.monotonic() - started
        log.info("ocr: %s image read in %.2f s by %s", prepared.format, elapsed, engine.name)
        return S.OcrOut(
            smiles=found.smiles,
            molfile=found.molfile,
            confidence=raw.confidence,
            engine=engine.name,
            elapsed_s=round(elapsed, 3),
            warnings=[*prepared.warnings, *found.warnings],
        )

    def close(self) -> None:
        with self._lock:
            engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()
