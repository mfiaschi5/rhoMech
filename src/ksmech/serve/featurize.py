"""Featurizers: get or build the QM record and the tokens of one species (adapters A4-A6).

Three chemistry modes:

- `real` (`QMFeaturizer(s, allow_compute=True)`): layered caches (shared read-only first, then
  `data/serve`); a miss runs GFN2-xTB through `qm.batch.process_molecule` into
  `data/serve/qm` and builds tokens into `data/serve/tokens`. Worker only.
- `cached` (`allow_compute=False`): the same layered caches, read-only; a miss is
  `FEATURIZATION_DISABLED`; never xTB, never `load_record` (which probes tblite).
- `synthetic` (`SyntheticFeaturizer`): charge-iterated EHT (`synthetic.eht_record`) into the
  serve-owned caches under the method `SYNTHETIC-EHT/v1`; never xTB.

`preflight` runs on the API side, is instant and never computes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from ksmech.serve.contracts import LRU, to_jsonable
from ksmech.serve.errors import Code
from ksmech.serve.loader import LayeredTokenLoader
from ksmech.serve.settings import ServeSettings, append_failure, assert_writable
from ksmech.serve.synthetic import SYNTHETIC_METHOD, SyntheticError, supported_elements

log = logging.getLogger(__name__)


class FeaturizeStatus(StrEnum):
    CACHED = "cached"
    BUILT = "built"
    FAILED = "failed"


@dataclass(frozen=True)
class FeaturizeResult:
    key: str
    status: FeaturizeStatus
    code: str | None  # FEATURIZATION_FAILED | FEATURIZATION_DISABLED | TOKENS_FAILED | ...
    upstream: str | None  # qm.records.ReasonCode value when the QM step failed
    detail: str
    seconds: float
    tokens_path: str | None
    qm_path: str | None
    warnings: str

    def to_dict(self) -> dict:
        return to_jsonable({k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: dict) -> FeaturizeResult:
        return cls(
            str(d["key"]),
            FeaturizeStatus(d["status"]),
            d.get("code"),
            d.get("upstream"),
            str(d.get("detail") or ""),
            float(d.get("seconds") or 0.0),
            d.get("tokens_path"),
            d.get("qm_path"),
            str(d.get("warnings") or ""),
        )


@dataclass(frozen=True)
class SpeciesPreflight:
    key: str
    charge: int | None  # None when the key does not parse
    multiplicity: int | None
    n_atoms: int | None
    tokens_cached: bool
    qm_cached: bool
    errors: list[tuple[str, str]] = field(default_factory=list)  # (Code, message)


class Featurizer(Protocol):
    mode: Literal["real", "cached", "synthetic"]
    method: str

    def preflight(self, key: str) -> SpeciesPreflight: ...  # API side, instant, no compute

    def ensure(self, key: str) -> FeaturizeResult: ...  # worker side

    def loader(self) -> LayeredTokenLoader: ...

    def occupations(self, key: str) -> np.ndarray | None: ...  # np.load(qm npz)["occupations"]


def qm_npz_path(s: ServeSettings, key: str, method: str) -> Path | None:
    """The QM record of `key` under `method`: the shared cache first (never for the synthetic
    method), then `data/serve/qm`; None when neither holds a valid file."""
    from ksmech.qm.records import cache_path, is_valid_cache_file

    dirs = [s.serve_qm] if method == SYNTHETIC_METHOD else [s.shared_qm, s.serve_qm]
    for d in dirs:
        path = cache_path(d, key, 0, method)
        if is_valid_cache_file(path):
            return path
    return None


def featurize_label(featurizer: Featurizer, key: str) -> str:
    """A progress label such as "GFN2-xTB for CC(=O)[O-] (7 atoms)"."""
    from ksmech.qm.embed import prepare_mol
    from ksmech.qm.records import QMError

    try:
        n = f" ({prepare_mol(key).GetNumAtoms()} atoms)"
    except QMError:
        n = ""
    name = "synthetic EHT" if featurizer.method == SYNTHETIC_METHOD else "GFN2-xTB"
    return f"{name} for {key}{n}"


def _preflight(key: str, loader: LayeredTokenLoader, s: ServeSettings, method: str, synthetic):
    """Steps 1-5 of design 2.2: parse, spin, radicals, elements, cache flags."""
    from ksmech.qm.embed import prepare_mol
    from ksmech.qm.records import QMError
    from ksmech.qm.xtb_calc import resolve_charge_spin

    try:
        mol = prepare_mol(key)
    except QMError as exc:
        return SpeciesPreflight(
            key, None, None, None, False, False, [(Code.INVALID_SMILES, str(exc))]
        )
    errors: list[tuple[str, str]] = []
    charge, unpaired = None, None
    try:
        charge, unpaired = resolve_charge_spin(mol)
    except QMError as exc:
        errors.append((Code.SPIN_PARITY, f"{key}: {exc.detail}"))
    if unpaired:
        errors.append((Code.RADICAL_NOT_SUPPORTED, f"{key}: {unpaired} unpaired electrons"))
    z = {a.GetAtomicNum() for a in mol.GetAtoms()}
    if synthetic:
        allowed = supported_elements()
    else:
        from ksmech.qm.ao import gfn2_shells

        allowed = frozenset(gfn2_shells())
    bad = sorted(z - allowed)
    if bad:
        errors.append((Code.UNSUPPORTED_ELEMENT, f"{key}: no parameters for Z={bad}"))
    mult = None if unpaired is None else int(unpaired) + 1
    tokens_cached = loader.has(key)
    qm_cached = qm_npz_path(s, key, method) is not None
    return SpeciesPreflight(
        key,
        charge,
        mult,
        mol.GetNumAtoms(),
        tokens_cached,
        qm_cached,
        [(str(c), m) for c, m in errors],
    )


def _result(key, status, t0, code=None, upstream=None, detail="", tokens=None, qm=None, warn=""):
    return FeaturizeResult(
        key,
        status,
        None if code is None else str(code),
        None if upstream is None else str(upstream),
        detail,
        time.perf_counter() - t0,
        None if tokens is None else str(tokens),
        None if qm is None else str(qm),
        warn,
    )


class _Base:
    mode: str
    method: str

    def __init__(self, settings: ServeSettings, loader: LayeredTokenLoader) -> None:
        self.settings = settings
        self._loader = loader
        self._occupations: LRU[str, np.ndarray] = LRU(256)

    def loader(self) -> LayeredTokenLoader:
        return self._loader

    def preflight(self, key: str) -> SpeciesPreflight:
        return _preflight(key, self._loader, self.settings, self.method, self.mode == "synthetic")

    def occupations(self, key: str) -> np.ndarray | None:
        """Canonical MO occupations from the QM npz (np.load, never `load_record`)."""
        hit = self._occupations.get(key)
        if hit is not None:
            return hit
        path = qm_npz_path(self.settings, key, self.method)
        if path is None:
            return None
        with np.load(path, allow_pickle=False) as z:
            occ = np.asarray(z["occupations"], dtype=float)
        self._occupations.put(key, occ)
        return occ

    def _fail(self, key, t0, code, detail, upstream=None, qm=None) -> FeaturizeResult:
        log.warning("featurize %s failed [%s/%s] %s", key, code, upstream, detail)
        append_failure(self.settings, str(code), detail, key=key, upstream=upstream, mode=self.mode)
        return _result(key, FeaturizeStatus.FAILED, t0, code, upstream, detail, qm=qm)


class QMFeaturizer(_Base):
    """GFN2-xTB tokens over the layered caches; computes only with `allow_compute`."""

    def __init__(self, settings: ServeSettings, allow_compute: bool) -> None:
        from ksmech.qm.xtb_calc import QMConfig

        self.mode = "real" if allow_compute else "cached"
        self.method = QMConfig().method
        self.allow_compute = bool(allow_compute)
        dirs = [settings.shared_tokens, settings.serve_tokens]
        super().__init__(settings, LayeredTokenLoader(dirs, self.method))

    def ensure(self, key: str) -> FeaturizeResult:
        t0 = time.perf_counter()
        cached = self._loader.path_of(key)
        if cached is not None:
            qm = qm_npz_path(self.settings, key, self.method)
            return _result(key, FeaturizeStatus.CACHED, t0, tokens=cached, qm=qm)
        try:
            return self._ensure_missing(key, t0)
        except Exception as exc:  # noqa: BLE001 - reported with a reason code, never raised
            log.exception("featurize %s", key)
            detail = f"{type(exc).__name__}: {exc}"
            return self._fail(key, t0, Code.FEATURIZATION_FAILED, detail, upstream="UNEXPECTED")

    def _ensure_missing(self, key: str, t0: float) -> FeaturizeResult:
        qm = qm_npz_path(self.settings, key, self.method)
        if qm is None and not self.allow_compute:
            detail = f"no QM record for {key} in the caches (cached mode never computes)"
            return self._fail(key, t0, Code.FEATURIZATION_DISABLED, detail)
        warnings = ""
        if qm is None:
            row = self._run_qm(key)
            if row["status"] != "ok":
                return self._fail(
                    key, t0, Code.FEATURIZATION_FAILED, str(row["detail"]), str(row["reason"])
                )
            qm, warnings = Path(row["cache_path"]), str(row.get("warnings") or "")
        if not self.allow_compute:
            detail = f"{key} has a QM record but no tokens (cached mode never builds tokens)"
            return self._fail(key, t0, Code.FEATURIZATION_DISABLED, detail, qm=qm)
        return self._build_tokens(key, Path(qm), t0, warnings)

    def _run_qm(self, key: str) -> dict:
        """GFN2-xTB for one species into data/serve/qm (through the module attribute, so the
        test guard can patch it). The SIGALRM limit is `settings.qm_timeout_s`."""
        from ksmech.qm import batch
        from ksmech.qm.xtb_calc import QMConfig

        s = self.settings
        assert_writable(s.serve_qm, s)
        config = QMConfig(cache_dir=s.serve_qm, timeout_s=int(s.qm_timeout_s))
        return batch.process_molecule(batch.MolTask(key, 1, config))[0]

    def _build_tokens(self, key: str, qm: Path, t0: float, warnings: str) -> FeaturizeResult:
        from ksmech.orbitals import build

        s = self.settings
        assert_writable(s.serve_tokens, s)
        config = build.BuildConfig(
            qm_cache_dir=qm.parent, out_dir=s.serve_tokens, method=self.method
        )
        row = build.process_molecule(key, config)
        if row.status not in ("built", "loaded"):
            return self._fail(key, t0, Code.TOKENS_FAILED, f"{row.status}: {row.detail}", qm=qm)
        status = FeaturizeStatus.BUILT if row.status == "built" else FeaturizeStatus.CACHED
        tokens = build.tokens_path(s.serve_tokens, key, 0, self.method)
        return _result(key, status, t0, tokens=tokens, qm=qm, warn=warnings)


class SyntheticFeaturizer(_Base):
    """Synthetic EHT tokens in the serve-owned caches; never xTB, never `load_record`."""

    def __init__(self, settings: ServeSettings) -> None:
        self.mode = "synthetic"
        self.method = SYNTHETIC_METHOD
        super().__init__(settings, LayeredTokenLoader([settings.serve_tokens], SYNTHETIC_METHOD))

    def ensure(self, key: str) -> FeaturizeResult:
        t0 = time.perf_counter()
        cached = self._loader.path_of(key)
        if cached is not None:
            qm = qm_npz_path(self.settings, key, self.method)
            return _result(key, FeaturizeStatus.CACHED, t0, tokens=cached, qm=qm)
        try:
            qm, tokens, warn = self._build(key)
        except SyntheticError as exc:
            return self._fail(key, t0, Code.SYNTHETIC_FAILED, str(exc), upstream=exc.code)
        except Exception as exc:  # noqa: BLE001 - reported with a reason code, never raised
            log.exception("synthetic featurize %s", key)
            return self._fail(key, t0, Code.SYNTHETIC_FAILED, f"{type(exc).__name__}: {exc}")
        return _result(key, FeaturizeStatus.BUILT, t0, tokens=tokens, qm=qm, warn=warn)

    def _build(self, key: str) -> tuple[Path, Path, str]:
        from ksmech.orbitals.build import tokens_path
        from ksmech.orbitals.tokens import build_molecule_tokens, save_tokens
        from ksmech.qm.records import cache_path, save_record
        from ksmech.serve.synthetic import eht_record

        s = self.settings
        rec = eht_record(key)
        qm = cache_path(s.serve_qm, key, 0, SYNTHETIC_METHOD)
        out = tokens_path(s.serve_tokens, key, 0, SYNTHETIC_METHOD)
        for path in (qm, out):
            assert_writable(path, s)
        save_record(rec, qm)
        save_tokens(build_molecule_tokens(rec), out)
        return qm, out, rec.warnings


def make_featurizer(s: ServeSettings) -> Featurizer:
    if s.chemistry == "real":
        return QMFeaturizer(s, allow_compute=True)
    if s.chemistry == "cached":
        return QMFeaturizer(s, allow_compute=False)
    if s.chemistry == "synthetic":
        return SyntheticFeaturizer(s)
    raise ValueError(f"unknown chemistry {s.chemistry!r}")
