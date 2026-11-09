"""The BLAS / LAPACK that tblite runs on in the QM workers.

The PyPI wheel of tblite bundles the netlib reference BLAS and LAPACK 3.8.0
(`tblite.libs/libblas-*.so`, `liblapack-*.so`): unblocked code, about 2.4x slower than OpenBLAS
on a 70-atom GFN2 singlepoint and 4.6x on a 120-atom one, where the generalized diagonalization
(`dsygvd_`) of every SCF cycle takes most of the time. tblite looks its BLAS and LAPACK symbols
up in the process's global scope before its own bundled copies, so a library preloaded into the
process (`LD_PRELOAD`) replaces them. A singlepoint at a given geometry is the same to 1e-9; an
optimization on a flat surface can end elsewhere within its convergence threshold, since the
rounding differs (a 93-atom molecule: 0.018 A, 0.013 kcal/mol; docs/decisions.md, 2026-11-09).

The library is preloaded only into the spawned workers of the QM batch (`preloaded_children`),
never into the calling process: torch exports `sgemm_` / `dgemm_` as interposable symbols, so a
global OpenBLAS in a process that imports torch afterwards would also replace torch's own CPU
GEMM. `KSMECH_BLAS` chooses the library: unset or `auto`, the system's LP64 OpenBLAS when one is
installed (Fedora `openblas-serial`, Debian / Ubuntu `libopenblas0-serial`); `bundled`, tblite's
own copy; anything else, the name or path of a library.

`provider` names the library that answers a symbol in this process's global scope, which is
what tblite binds to; the QM records carry it in their provenance header.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache

log = logging.getLogger(__name__)

ENV = "KSMECH_BLAS"
BUNDLED = "bundled"
REQUIRED = ("dgemm_", "dsygvd_")  # BLAS level 3 and the LAPACK eigensolver tblite calls


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _library_of(func: ctypes._CFuncPtr) -> str:
    """The real path of the shared library that holds `func`."""
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    info = _DlInfo()
    if not dladdr(ctypes.cast(func, ctypes.c_void_p), ctypes.byref(info)) or not info.dli_fname:
        return "unknown"
    return os.path.realpath(info.dli_fname.decode())


def provider(symbol: str = "dsygvd_") -> str:
    """The library that answers `symbol` in this process's global scope, where tblite looks
    first; `bundled` when none does and tblite uses its own copy."""
    try:
        func = getattr(ctypes.CDLL(None), symbol)
    except AttributeError:
        return BUNDLED
    return _library_of(func)


def _unusable(lib: ctypes.CDLL) -> str | None:
    """Why `lib` cannot stand in for tblite's BLAS and LAPACK, or None."""
    missing = [s for s in REQUIRED if not hasattr(lib, s)]
    if missing:
        return f"no {', '.join(missing)} (an ILP64 or BLAS-only build?)"
    if hasattr(lib, "openblas_get_config"):
        lib.openblas_get_config.restype = ctypes.c_char_p
        config = (lib.openblas_get_config() or b"").decode()
        if "USE64BITINT" in config:
            return f"64-bit integers ({config}); tblite needs LP64"
    return None


@cache
def _resolve(want: str) -> str | None:
    if want == BUNDLED:
        return None
    name = ctypes.util.find_library("openblas") if want in ("", "auto") else want
    if name is None:
        log.warning(
            "no system OpenBLAS: tblite keeps its bundled netlib BLAS (install openblas-serial "
            "or libopenblas0-serial, or set %s to a library)",
            ENV,
        )
        return None
    try:
        lib = ctypes.CDLL(name)  # RTLD_LOCAL: this process's own symbols are left alone
    except OSError as exc:
        log.warning("%s=%r cannot be loaded (%s): tblite keeps its bundled BLAS", ENV, want, exc)
        return None
    problem = _unusable(lib)
    if problem:
        log.warning("%s cannot replace tblite's BLAS: %s", name, problem)
        return None
    return _library_of(lib.dgemm_)


def choose() -> str | None:
    """The library the QM workers preload (`KSMECH_BLAS`), or None for tblite's bundled one."""
    return _resolve(os.environ.get(ENV, "auto").strip())


def with_preload(current: str, lib: str) -> str:
    """`LD_PRELOAD` with `lib` first and the entries already there after it."""
    rest = [p for p in re.split(r"[\s:]+", current) if p and p != lib]
    return " ".join([lib, *rest])


@contextmanager
def preloaded_children() -> Iterator[str | None]:
    """Processes started inside the block preload the chosen library (yielded, or None)."""
    lib = choose()
    saved = os.environ.get("LD_PRELOAD")
    if lib is not None:
        os.environ["LD_PRELOAD"] = with_preload(saved or "", lib)
    try:
        yield lib
    finally:
        if saved is None:
            os.environ.pop("LD_PRELOAD", None)
        else:
            os.environ["LD_PRELOAD"] = saved
