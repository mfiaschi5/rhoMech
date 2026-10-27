"""The analysis artifact of one pool state (spec v9): the pooled gap factors the floor ranks
and every reading of a step needs, computed once in the worker and stored as one npz.

`ReactantTokens` -> `reactant_sample` (the model inputs and the arrow-candidate filter) ->
`gatom.pooled_gap`: the occupied and virtual token ids, the validity filter, the gap factor
G(r, s) of every pair in float64 and the centre weights of every token, from which
`PoolAnalysis.pg` rebuilds `G_atom` and every channel without recomputation. One identity is
asserted: the filter equals the sample's `pair_valid`, and the stored G equals the sample's
`g` column to float32 precision; a failure is `CONSISTENCY_FAILED`, never a number.

Files: `pools/<state_id>/analysis-<physics fp>-<dig12>.npz` plus `.json`, with dig12 a digest
of the fragments' token files; a failure writes `analysis-failure-<fp>.json`. The digest is
taken before the tokens are read and again after the numbers are computed: when the files
changed in between, the result is TOKENS_CHANGED (503, transient, never saved or recorded).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ksmech.serve.chem.pools import atomic_write_text, state_path, tmp_sibling
from ksmech.serve.core.contracts import LRU, PoolState, Runtime
from ksmech.serve.core.errors import Code, ServeError, jsonable
from ksmech.serve.core.settings import ServeSettings, assert_writable
from ksmech.serve.core.versions import artifact_versions, physics_fingerprint
from ksmech.serve.science.gatom import PooledGap

if TYPE_CHECKING:
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.arrows.steps import MechanismSample
    from ksmech.serve.chem.loader import LayeredTokenLoader

log = logging.getLogger(__name__)

PG_MEMBERS = ("rows", "cols", "valid", "G", "w_occ", "w_virt")
TOKENS_CHANGED = "TOKENS_CHANGED"  # 503: token files changed under a computation; retry
TRANSIENT = frozenset({"WORKER_TIMEOUT", "WORKER_CRASH", "WORKER_UNSTABLE", TOKENS_CHANGED})
_LOADED: LRU[str, PoolAnalysis] = LRU(16)


@dataclass
class PoolAnalysis:
    """The members of one analysis npz (see the module docstring) and its json sidecar.
    Members are reachable as attributes (`an.G`, `an.valid`) and by name (`an["G"]`)."""

    state_id: str
    members: dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)
    path: Path | None = None

    def __getitem__(self, name: str) -> np.ndarray:
        return self.members[name]

    def __contains__(self, name: object) -> bool:
        return name in self.members

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in ("members", "meta", "path", "state_id"):
            raise AttributeError(name)
        members = self.__dict__.get("members", {})
        if name in members:
            return members[name]
        raise AttributeError(f"PoolAnalysis has no member {name!r}")

    @property
    def n_valid(self) -> int:
        return int(np.asarray(self.members["valid"]).sum())

    @property
    def fermi_smeared(self) -> bool:
        return bool(np.asarray(self.members["fermi_smeared"]))

    @property
    def pg(self) -> PooledGap:
        """The pooled-gap block (`serve.science.gatom`), rebuilt from the members."""
        cached = self.__dict__.get("_pg")
        if cached is None:
            cached = PooledGap.from_members(self.members)
            self.__dict__["_pg"] = cached
        return cached

    @property
    def mol_of_global(self) -> np.ndarray:
        """(A,) pool atom of every global atom."""
        return np.asarray(self.members["mol_of_global"], dtype=np.int64)


@dataclass
class PoolWork:
    """The in-memory bundle the worker tasks use."""

    state: PoolState
    rt: ReactantTokens
    sample: MechanismSample
    an: PoolAnalysis
    npz: Path

    @property
    def pg(self) -> PooledGap:
        return self.an.pg


# ---- computation


def _mol_of_global(rt: ReactantTokens) -> np.ndarray:
    out = np.full(rt.n_atoms, -1, dtype=np.int64)
    for g, m in rt.mol_atom.items():
        out[g] = m
    return out


def analyze_pool(state: PoolState, rt: ReactantTokens, sample: MechanismSample) -> PoolAnalysis:
    """Every member of one state's analysis (module docstring)."""
    from ksmech.orbitals.tokens import CandidateConfig
    from ksmech.physics.stage_a import A
    from ksmech.serve.science.gatom import pooled_gap

    t0 = time.perf_counter()
    pg = pooled_gap(rt)
    if not np.array_equal(pg.valid, np.asarray(sample.pair_valid, dtype=bool)):
        raise ServeError(Code.CONSISTENCY_FAILED, "the candidate filter differs from pair_valid")
    g32 = np.asarray(sample.pair_z[..., A["g"]], dtype=np.float64)
    if pg.G.shape != g32.shape or not np.allclose(pg.G, g32, rtol=1e-5, atol=1e-7):
        raise ServeError(Code.CONSISTENCY_FAILED, "G differs from the sample's pair feature g")
    members = pg.to_members()
    members["mol_of_global"] = _mol_of_global(rt)
    members["fermi_smeared"] = np.array(bool(sample.fermi_smeared))
    n_bad = int((~np.isfinite(pg.G)).sum())
    meta = {
        "state_id": state.state_id,
        "candidate_config": jsonable(asdict(rt.candidate or CandidateConfig())),
        "n_non_finite": n_bad,
        "n_valid": int(pg.valid.sum()),
        "seconds": time.perf_counter() - t0,
    }
    return PoolAnalysis(state.state_id, members, meta)


# ---- files


def analysis_paths(s: ServeSettings, state_id: str, fp: str, digest: str) -> tuple[Path, Path]:
    base = state_path(state_id, s).parent
    stem = f"analysis-{fp}-{digest}"
    return base / f"{stem}.npz", base / f"{stem}.json"


def failure_path(s: ServeSettings, state_id: str, fp: str) -> Path:
    return state_path(state_id, s).parent / f"analysis-failure-{fp}.json"


def token_digests(loader: LayeredTokenLoader, state: PoolState) -> dict[str, str | None]:
    """{key: loader.digest(key)} over the state's fragments, sorted by key."""
    return {k: loader.digest(k) for k in sorted(set(state.fragment_token_keys))}


def combined_digest(digests: dict[str, str | None]) -> str | None:
    """sha256(json of `digests`)[:12]; None when empty or when a fragment has no file."""
    if not digests or any(v is None for v in digests.values()):
        return None
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()[:12]


def analysis_digest(loader: LayeredTokenLoader, state: PoolState) -> str | None:
    """sha256(json of {key: loader.digest(key)} over the fragments)[:12]; None when a
    fragment has no token file."""
    return combined_digest(token_digests(loader, state))


def tokens_changed(state: PoolState, before: str | None, after: str | None) -> ServeError:
    """TOKENS_CHANGED (503, transient, never recorded as a failure): the token files of
    `state` changed while a result keyed by `before` was computed."""
    msg = f"token files of {state.pool_key} changed during the computation; retry"
    detail = {"state_id": state.state_id, "digest_before": before, "digest_after": after}
    return ServeError(TOKENS_CHANGED, msg, status=503, detail=detail)


def _atomic_bytes(path: Path, data: bytes, s: ServeSettings | None) -> None:
    if s is not None:
        assert_writable(path, s)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_sibling(path)
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def save_npz(path: Path, arrays: dict[str, np.ndarray], s: ServeSettings | None = None) -> Path:
    """Atomic compressed npz (no pickle)."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    _atomic_bytes(Path(path), buf.getvalue(), s)
    return Path(path)


def save_analysis(an: PoolAnalysis, npz: Path, meta: Path, s: ServeSettings | None = None) -> None:
    """The json sidecar first, the npz last (the npz is what marks the analysis as done)."""
    text = json.dumps(jsonable(an.meta), sort_keys=True, indent=1) + "\n"
    _atomic_bytes(Path(meta), text.encode(), s)
    save_npz(npz, an.members, s)
    an.path = Path(npz)


def load_analysis(npz: Path) -> PoolAnalysis:
    npz = Path(npz)
    with np.load(npz, allow_pickle=False) as z:
        members = {k: np.asarray(z[k]) for k in z.files}
    meta_path = npz.with_suffix(".json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    return PoolAnalysis(str(meta.get("state_id") or npz.parent.name), members, meta, npz)


def read_failure(s: ServeSettings, state_id: str, fp: str) -> dict | None:
    """The failure sidecar `{code, upstream, detail, digest, stage, ...}`, or None."""
    try:
        return json.loads(failure_path(s, state_id, fp).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def current_failure(s: ServeSettings, loader, state: PoolState, fp: str) -> dict | None:
    """The failure sidecar if it still applies: written for the current token digest."""
    failure = read_failure(s, state.state_id, fp)
    if failure is None or failure.get("digest") != analysis_digest(loader, state):
        return None
    return failure


def write_failure(s: ServeSettings, state: PoolState, fp: str, err: ServeError, **extra) -> None:
    """Record a non-transient failure (WORKER_TIMEOUT / CRASH / UNSTABLE are never recorded)."""
    if str(err.code) in TRANSIENT:
        return
    record = {
        "code": str(err.code),
        "upstream": err.upstream,
        "detail": err.message,
        "error_detail": jsonable(err.detail),
        "created_at": datetime.now(UTC).isoformat(),
        **extra,
    }
    text = json.dumps(jsonable(record), sort_keys=True, indent=1) + "\n"
    atomic_write_text(failure_path(s, state.state_id, fp), text, s)


def clear_failure(s: ServeSettings, state_id: str, fp: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        failure_path(s, state_id, fp).unlink()


# ---- worker entry points


def _worker_error(exc: Exception) -> ServeError:
    from ksmech.arrows.reactants import CorrespondenceError, MissingTokens

    if isinstance(exc, ServeError):
        return exc
    if isinstance(exc, MissingTokens):
        return ServeError(Code.MISSING_TOKENS, str(exc), detail={"keys": list(exc.keys)})
    if isinstance(exc, CorrespondenceError):
        return ServeError(Code.CORRESPONDENCE, str(exc))
    return ServeError(Code.ANALYSIS_FAILED, f"{type(exc).__name__}: {exc}")


def _sample(rt_ctx: Runtime, state: PoolState, rt: ReactantTokens) -> MechanismSample:
    sample = rt_ctx.sample_cache.get(state.state_id)
    if sample is None:
        from ksmech.arrows.steps import reactant_sample

        sample = reactant_sample(f"serve:{state.state_id}", "serve", rt)
        rt_ctx.sample_cache.put(state.state_id, sample)
    return sample


def _checked_rt(rt_ctx: Runtime, state: PoolState) -> ReactantTokens:
    rt = rt_ctx.reactant_tokens(state)
    if len(rt.mol_atom) != rt.n_atoms:
        msg = f"{rt.n_atoms - len(rt.mol_atom)} record atoms are not drawn (implicit H)"
        raise ServeError(Code.IMPLICIT_H_IN_POOL, msg)
    return rt


def _compute(rt_ctx: Runtime, state: PoolState, digest: str) -> PoolAnalysis:
    """The analysis of `state` for the token digest `digest` (computed before any token was
    read). The digest is taken again at the end: if the files changed meanwhile, the numbers
    may mix old and new tokens and TOKENS_CHANGED (503) is raised instead of a result."""
    rt = _checked_rt(rt_ctx, state)
    sample = _sample(rt_ctx, state, rt)
    an = analyze_pool(state, rt, sample)
    digests = token_digests(rt_ctx.loader, state)
    if combined_digest(digests) != digest:
        raise tokens_changed(state, digest, combined_digest(digests))
    an.meta |= {
        "physics_fingerprint": physics_fingerprint(),
        "token_digests": digests,
        "analysis_digest": digest,
        "versions": artifact_versions(),
        "chemistry": rt_ctx.settings.chemistry,
        "created_at": datetime.now(UTC).isoformat(),
    }
    return an


def ensure_analysis(rt_ctx: Runtime, state: PoolState) -> Path:
    """The analysis npz of `state` for the current physics fingerprint and token digest,
    computed if missing. A failure writes the failure sidecar and raises ServeError
    (MISSING_TOKENS, CORRESPONDENCE, IMPLICIT_H_IN_POOL, CONSISTENCY_FAILED, ANALYSIS_FAILED;
    TOKENS_CHANGED is transient and never recorded). Every success clears the sidecar."""
    return _ensure(rt_ctx, state)[0]


def _ensure(rt_ctx: Runtime, state: PoolState) -> tuple[Path, str]:
    """`ensure_analysis` and the token digest its npz is named by."""
    s, fp = rt_ctx.settings, physics_fingerprint()
    digest = analysis_digest(rt_ctx.loader, state)
    try:
        if digest is None:
            missing = [k for k in state.fragment_token_keys if rt_ctx.loader.digest(k) is None]
            msg = f"no tokens for {sorted(set(missing))}"
            raise ServeError(Code.MISSING_TOKENS, msg, detail={"keys": sorted(set(missing))})
        npz, meta = analysis_paths(s, state.state_id, fp, digest)
        if not npz.is_file():
            an = _compute(rt_ctx, state, digest)
            save_analysis(an, npz, meta, s)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a reason code
        err = _worker_error(exc)
        if not isinstance(exc, ServeError):
            log.exception("analysis of %s failed", state.state_id)
        write_failure(s, state, fp, err, digest=digest, stage="analyze")
        raise err from exc
    clear_failure(s, state.state_id, fp)  # every success, the cached npz included
    return npz, digest


def loaded_analysis(npz: Path) -> PoolAnalysis:
    """`load_analysis` through a small per-process LRU (by npz path)."""
    key = str(npz)
    an = _LOADED.get(key)
    if an is None:
        an = load_analysis(npz)
        _LOADED.put(key, an)
    return an


def pool_work(rt_ctx: Runtime, state: PoolState) -> PoolWork:
    """`ensure_analysis`, then the analysis with rt and the sample from the runtime's LRUs
    (the pooled gap is rebuilt from the members, never recomputed). rt and the sample must come
    from the token files the analysis was named by: a digest that changed meanwhile is
    TOKENS_CHANGED (503), never a bundle whose cells do not line up."""
    npz, digest = _ensure(rt_ctx, state)
    an = loaded_analysis(npz)
    rt = _checked_rt(rt_ctx, state)
    sample = _sample(rt_ctx, state, rt)
    after = analysis_digest(rt_ctx.loader, state)
    if after != digest:
        raise tokens_changed(state, digest, after)
    return PoolWork(state, rt, sample, an, npz)


def t_analyze(rt: Runtime, payload: dict) -> dict:
    """{"state"} -> {"npz", "meta", "n_valid", "fermi_smeared"}."""
    state = PoolState.from_dict(payload["state"])
    npz = ensure_analysis(rt, state)
    an = loaded_analysis(npz)
    return {
        "npz": str(npz),
        "meta": str(npz.with_suffix(".json")),
        "n_valid": an.n_valid,
        "fermi_smeared": an.fermi_smeared,
    }
