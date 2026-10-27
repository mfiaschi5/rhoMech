"""`ksmech-serve ocr-setup`: build the OCR engine's environment and fetch its verified weights.

The environment is the uv project `tools/ocr` synced into `data/ocr/venv`
(`UV_PROJECT_ENVIRONMENT`): `uv lock` first when `tools/ocr/uv.lock` does not exist yet (or with
`relock`), then `uv sync --locked`. It never touches the project's own environment or lock.

The checkpoint is downloaded to `<checkpoint>.part`, resumed from that file's size with an HTTP
Range request after an interruption (up to `attempts` tries, and across runs), then checked:
its size, then its sha256. Only a file that passes is renamed into place and gets the marker
the capability reads (`install.write_marker`); a file that fails is deleted. A checkpoint that
is already in place and marked is not downloaded again; one in place without a marker is
re-verified.

The subprocess runner and the URL opener are parameters, so the tests run all of this without
uv or a network.
"""

from __future__ import annotations

import hashlib
import http.client
import logging
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from ksmech.serve.ocr.install import (
    CHECKPOINT_BYTES,
    CHECKPOINT_SHA256,
    CHECKPOINT_URL,
    OcrPaths,
    read_marker,
    write_marker,
)

log = logging.getLogger(__name__)

USER_AGENT = "ksmech-serve ocr-setup"
CHUNK = 1 << 20
NETWORK_ERRORS = (urllib.error.URLError, OSError, http.client.HTTPException, TimeoutError)
Say = Callable[[str], None]


class SetupError(Exception):
    """A setup step failed (the message says which and why)."""


# ---- the environment


def uv_steps(paths: OcrPaths, uv: str = "uv", relock: bool = False) -> list[list[str]]:
    steps = []
    if relock or not (paths.tools / "uv.lock").is_file():
        steps.append([uv, "lock", "--project", str(paths.tools)])
    steps.append([uv, "sync", "--project", str(paths.tools), "--locked"])
    return steps


def uv_env(paths: OcrPaths) -> dict[str, str]:
    """This process's environment, pointed at `data/ocr/venv` and away from the active venv."""
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_NO_SYNC")}
    return env | {"UV_PROJECT_ENVIRONMENT": str(paths.venv)}


def sync_env(
    paths: OcrPaths,
    *,
    uv: str = "uv",
    relock: bool = False,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    say: Say = log.info,
) -> None:
    if not (paths.tools / "pyproject.toml").is_file():
        raise SetupError(f"no uv project at {paths.tools} (tools/ocr/pyproject.toml)")
    paths.root.mkdir(parents=True, exist_ok=True)
    for argv in uv_steps(paths, uv, relock):
        say(f"environment: {' '.join(argv)} (UV_PROJECT_ENVIRONMENT={paths.venv})")
        try:
            done = run(argv, env=uv_env(paths), check=False)
        except OSError as exc:
            raise SetupError(f"cannot run {argv[0]}: {exc}") from exc
        if done.returncode != 0:
            raise SetupError(f"`{' '.join(argv[:2])}` failed with exit code {done.returncode}")
    if not paths.python.is_file():
        raise SetupError(f"uv sync finished, but {paths.python} does not exist")
    say(f"environment: ready at {paths.venv}")


# ---- the weights


def sha256_of(path: Path, chunk: int = CHUNK) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class Progress:
    """A progress line every `step` of the file (and at the end)."""

    def __init__(self, total: int, done: int, say: Say, step: float = 0.05) -> None:
        self.total, self.say, self.step = max(1, total), say, step
        self.start_done, self.started = done, time.monotonic()
        self.next = (int(done / self.total / step) + 1) * step

    def update(self, done: int) -> None:
        if done / self.total < self.next and done < self.total:
            return
        self.next = (int(done / self.total / self.step) + 1) * self.step
        rate = (done - self.start_done) / max(time.monotonic() - self.started, 1e-6) / 1e6
        mb = 1e6
        self.say(
            f"weights: {done / mb:.1f} / {self.total / mb:.1f} MB "
            f"({100 * done / self.total:.0f}%), {rate:.1f} MB/s"
        )


def _request(url: str, offset: int) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.Request(url, headers=headers)


def _check_space(paths: OcrPaths, needed: int) -> None:
    free = shutil.disk_usage(paths.weights).free
    if free < needed + (64 << 20):
        raise SetupError(f"{paths.weights}: {free / 1e6:.0f} MB free, {needed / 1e6:.0f} MB needed")


def _fetch(paths: OcrPaths, url: str, size: int, opener: Callable, say: Say) -> None:
    """One download attempt, appended to the partial file from its current size."""
    offset = paths.partial.stat().st_size if paths.partial.is_file() else 0
    if offset > size:
        paths.partial.unlink()
        offset = 0
    if offset == size:
        return
    _check_space(paths, size - offset)
    try:
        response = opener(_request(url, offset), timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # nothing left past the offset: verify what is there
            return
        if 400 <= exc.code < 500:
            raise SetupError(f"{url}: HTTP {exc.code} {exc.reason}") from exc
        raise
    with response:
        resumed = offset > 0 and getattr(response, "status", 200) == 206
        if offset and not resumed:
            say("weights: the server ignored the range request: starting over")
        done = offset if resumed else 0
        say(f"weights: {'resuming at' if resumed else 'downloading from'} {done} bytes, {url}")
        progress = Progress(size, done, say)
        with open(paths.partial, "ab" if resumed else "wb") as fh:
            while block := response.read(CHUNK):
                fh.write(block)
                done += len(block)
                progress.update(done)


def _verify(paths: OcrPaths, url: str, sha256: str, size: int, say: Say) -> None:
    got = paths.partial.stat().st_size if paths.partial.is_file() else 0
    if got != size:
        if got > size:
            paths.partial.unlink()
        raise SetupError(f"weights: {got} bytes downloaded, {size} expected: run ocr-setup again")
    say("weights: checking the sha256")
    digest = sha256_of(paths.partial)
    if digest != sha256:
        paths.partial.unlink()
        raise SetupError(f"weights: sha256 {digest} is not {sha256}: the file was deleted")
    os.replace(paths.partial, paths.checkpoint)
    write_marker(paths, sha256, size, url)
    say(f"weights: verified {paths.checkpoint}")


def _already(paths: OcrPaths, sha256: str, size: int) -> bool:
    marker = read_marker(paths) or {}
    if marker.get("sha256") != sha256 or marker.get("size") != size:
        return False
    return paths.checkpoint.is_file() and paths.checkpoint.stat().st_size == size


def download_checkpoint(
    paths: OcrPaths,
    *,
    url: str = CHECKPOINT_URL,
    sha256: str = CHECKPOINT_SHA256,
    size: int = CHECKPOINT_BYTES,
    opener: Callable = urllib.request.urlopen,
    attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
    say: Say = log.info,
) -> Path:
    """The verified checkpoint (module docstring); SetupError when it cannot be had."""
    paths.weights.mkdir(parents=True, exist_ok=True)
    if _already(paths, sha256, size):
        say(f"weights: {paths.checkpoint} is already verified")
        return paths.checkpoint
    paths.marker.unlink(missing_ok=True)
    if paths.checkpoint.is_file() and not paths.partial.exists():
        os.replace(paths.checkpoint, paths.partial)  # an unmarked file: verified like a download
    for attempt in range(1, attempts + 1):
        try:
            _fetch(paths, url, size, opener, say)
            break
        except NETWORK_ERRORS as exc:
            if attempt == attempts:
                raise SetupError(f"weights: download failed {attempts} times: {exc}") from exc
            delay = min(60.0, 2.0**attempt)
            say(f"weights: interrupted ({exc}); resuming in {delay:.0f} s")
            sleep(delay)
    _verify(paths, url, sha256, size, say)
    return paths.checkpoint
