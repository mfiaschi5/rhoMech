"""Featurizers: get or build the QM record and the tokens of one species (adapters A4-A6).

Three chemistry modes:

- `real` (`QMFeaturizer(s, allow_compute=True)`): layered caches (shared read-only first, then
  `data/serve`); a miss runs GFN2-xTB through `qm.batch.process_molecule` into
  `data/serve/qm` and builds tokens into `data/serve/tokens`. Worker only.
- `cached` (`allow_compute=False`): the same layered caches, read-only; a miss is
  `FEATURIZATION_DISABLED`; never xTB, never `load_record` (which probes tblite).
- `synthetic` (`SyntheticFeaturizer`): charge-iterated EHT (`synthetic.eht_record`) into the
  serve-owned caches under the method `SYNTHETIC-EHT/v1`; never xTB.

A key with a transition metal takes the metal path in `real` and `cached` mode: its QM record
and tokens live under `<GFN2 method>+metal-geom-v1` (`metal_geometry.method_for`, applied by
`qm_npz_path`, the token loader and `method_for`), and a miss runs `metal_qm.
process_metal_molecule` (coordination template, metal-aware connectivity check) instead of
`qm.batch.process_molecule`. Every other key keeps the standard method and path unchanged.

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

from ksmech.serve.chem.loader import LayeredTokenLoader
from ksmech.serve.chem.metal_geometry import method_for
from ksmech.serve.chem.synthetic import SYNTHETIC_METHOD, SyntheticError, supported_elements
from ksmech.serve.core.contracts import LRU, to_jsonable
from ksmech.serve.core.errors import Code
from ksmech.serve.core.settings import ServeSettings, append_failure, assert_writable

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


class KeyedMethodLoader(LayeredTokenLoader):
    """The layered loader with each key's token file under `method_for(key, method)`."""

    def _candidates(self, key: str) -> list[Path]:
        from ksmech.orbitals.build import tokens_path

        method = method_for(key, self.method)
        return [tokens_path(d, key, 0, method) for d in self.dirs]


class Featurizer(Protocol):
    mode: Literal["real", "cached", "synthetic"]
    method: str

    def preflight(self, key: str) -> SpeciesPreflight: ...  # API side, instant, no compute

    def ensure(self, key: str) -> FeaturizeResult: ...  # worker side

    def loader(self) -> LayeredTokenLoader: ...

    def method_for(self, key: str) -> str: ...  # the record / token method of one key

    def occupations(self, key: str) -> np.ndarray | None: ...  # np.load(qm npz)["occupations"]


def qm_npz_path(s: ServeSettings, key: str, method: str) -> Path | None:
    """The QM record of `key` under `method` (the metal method for a metal key, see
    `method_for`): the shared cache first (never for the synthetic method), then
    `data/serve/qm`; None when neither holds a valid file."""
    from ksmech.qm.records import cache_path, is_valid_cache_file

    method = method_for(key, method)
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
    if method_for(key, featurizer.method) != featurizer.method:
        name += " (metal template)"
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
    if not synthetic and not unpaired:
        from ksmech.serve.chem.metal_geometry import preflight_problems

        errors += [(Code.RADICAL_NOT_SUPPORTED, f"{key}: {p}") for p in preflight_problems(mol)]
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

    def method_for(self, key: str) -> str:
        """The method of `key`'s QM record and tokens (the metal method for a metal key)."""
        return method_for(key, self.method)

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
        roots = [settings.shared_root, None]  # shared files name their QM inputs from there
        bonds = [settings.shared_bond_index, settings.serve_bond_index]
        loader = KeyedMethodLoader(dirs, self.method, roots=roots, bond_dirs=bonds)
        super().__init__(settings, loader)

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
        if self.method_for(key) != self.method:
            from ksmech.serve.chem import metal_qm

            mconfig = metal_qm.MetalQMConfig(cache_dir=s.serve_qm, timeout_s=int(s.qm_timeout_s))
            return metal_qm.process_metal_molecule(key, mconfig)
        config = QMConfig(cache_dir=s.serve_qm, timeout_s=int(s.qm_timeout_s))
        return batch.process_molecule(batch.MolTask(key, 1, config))[0]

    def _build_tokens(self, key: str, qm: Path, t0: float, warnings: str) -> FeaturizeResult:
        from ksmech.orbitals import build

        s = self.settings
        assert_writable(s.serve_tokens, s)
        method = self.method_for(key)
        config = build.BuildConfig(qm_cache_dir=qm.parent, out_dir=s.serve_tokens, method=method)
        row = build.process_molecule(key, config)
        if row.status == "stale_qm":  # a QM record without (or with another) provenance header
            detail = f"{row.detail} (run `ksmech-data provenance --backfill qm` on that cache)"
            return self._fail(key, t0, Code.TOKENS_FAILED, detail, "STALE_ARTIFACT", qm=qm)
        if row.status not in ("built", "loaded"):
            return self._fail(key, t0, Code.TOKENS_FAILED, f"{row.status}: {row.detail}", qm=qm)
        status = FeaturizeStatus.BUILT if row.status == "built" else FeaturizeStatus.CACHED
        tokens = build.tokens_path(s.serve_tokens, key, 0, method)
        failed = self._bond_sidecar(key, qm, method)
        if failed is not None:
            return self._fail(key, t0, Code.TOKENS_FAILED, failed, "NO_BOND_INDEX", qm=qm)
        return _result(key, status, t0, tokens=tokens, qm=qm, warn=warnings)

    def _bond_sidecar(self, key: str, qm: Path, method: str) -> str | None:
        """Write the Wiberg sidecar of a serve-built record (real mode: the record was just
        computed, so `load_record` may probe tblite); the reason when it cannot."""
        from ksmech.qm.bond_index import bond_index_path, save_bond_indices
        from ksmech.qm.records import load_record

        s = self.settings
        if bond_index_path(s.serve_bond_index, key, 0, method).is_file():
            return None
        if not self.allow_compute:
            return f"no Wiberg sidecar for {key} (cached mode never computes one)"
        try:
            assert_writable(s.serve_bond_index, s)
            save_bond_indices(load_record(qm), qm, s.serve_bond_index)
        except Exception as exc:  # noqa: BLE001 - reported with its reason code
            return f"Wiberg sidecar of {key}: {type(exc).__name__}: {exc}"
        return None


class SyntheticFeaturizer(_Base):
    """Synthetic EHT tokens in the serve-owned caches; never xTB, never `load_record`."""

    def __init__(self, settings: ServeSettings) -> None:
        self.mode = "synthetic"
        self.method = SYNTHETIC_METHOD
        loader = LayeredTokenLoader(
            [settings.serve_tokens], SYNTHETIC_METHOD, bond_dirs=[settings.serve_bond_index]
        )
        super().__init__(settings, loader)

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
        from ksmech.data import provenance as prov
        from ksmech.orbitals.build import tokens_path
        from ksmech.orbitals.tokens import build_molecule_tokens, save_tokens, token_config
        from ksmech.qm.bond_index import save_bond_indices
        from ksmech.qm.records import cache_path, save_record
        from ksmech.serve.chem.synthetic import eht_record

        s = self.settings
        rec = eht_record(key)
        qm = cache_path(s.serve_qm, key, 0, SYNTHETIC_METHOD)
        out = tokens_path(s.serve_tokens, key, 0, SYNTHETIC_METHOD)
        for path in (qm, out):
            assert_writable(path, s)
        save_record(rec, qm)
        save_bond_indices(rec, qm, s.serve_bond_index)  # the Wiberg sidecar the pair features need
        # stamped as `orbitals.build` does: the QM record as input, the default token config
        save_tokens(build_molecule_tokens(rec), out, prov.file_input(qm), token_config())
        return qm, out, rec.warnings


def make_featurizer(s: ServeSettings) -> Featurizer:
    if s.chemistry == "real":
        return QMFeaturizer(s, allow_compute=True)
    if s.chemistry == "cached":
        return QMFeaturizer(s, allow_compute=False)
    if s.chemistry == "synthetic":
        return SyntheticFeaturizer(s)
    raise ValueError(f"unknown chemistry {s.chemistry!r}")
