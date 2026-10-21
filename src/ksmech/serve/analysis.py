"""The analysis artifact of one pool state (adapter A7): every number the panels show, computed
once in the worker on the production call path and stored as one npz.

`reactant_sample` -> `pair_components` -> `SCORERS[UNTRAINED_SCORE]` gives the floor; the npz
keeps every component (`comp_<name>` for the keys without a member of their own, so `pool_work`
rebuilds `comps` without recomputation), the signed beta of the identical cached kernel call,
beta^2 at the three feature distances from the stored logs (NaN where undefined, never
exp(0) = 1), the pair reasons and every available predictor's scores. Two identities are
asserted bit for bit: `floor == -(ks3 + ks2)` and `(reasons == VALID) == pair_valid`; a
failure is `CONSISTENCY_FAILED`, never a number.

Files: `pools/<state_id>/analysis-<physics fp>-<dig12>.npz` plus `.json`, with dig12 a digest
of the fragments' token files; a failure writes `analysis-failure-<fp>.json`. The digest is
taken before the tokens are read and again after the numbers are computed: when the files
changed in between, the result is TOKENS_CHANGED (503, transient, never saved or recorded).
The stored `score_<name>` members serve only the predictor version recorded for them in
`meta["predictor_versions"]`.
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

from ksmech.serve.contracts import LRU, PoolState, Runtime, ScoringContext
from ksmech.serve.errors import Code, ServeError, jsonable
from ksmech.serve.pools import atomic_write_text, state_path, tmp_sibling
from ksmech.serve.settings import ServeSettings, assert_writable
from ksmech.serve.versions import artifact_versions, physics_fingerprint

if TYPE_CHECKING:
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.arrows.steps import MechanismSample
    from ksmech.physics.pairs import PairKernels
    from ksmech.serve.loader import LayeredTokenLoader
    from ksmech.serve.predictors import Predictor

log = logging.getLogger(__name__)

DIRECT = ("ks3", "ks2", "ks2_ok", "ks1", "ks1_ok", "gap_kernel", "amp", "clash", "inter")
IDENTITY = ("rows", "cols", "valid", "reasons", "contact_sites", "gap_sign")
TOKENS_CHANGED = "TOKENS_CHANGED"  # 503: token files changed under a computation; retry
TRANSIENT = frozenset({"WORKER_TIMEOUT", "WORKER_CRASH", "WORKER_UNSTABLE", TOKENS_CHANGED})
_LOADED: LRU[str, PoolAnalysis] = LRU(16)


@dataclass
class PoolAnalysis:
    """The members of one analysis npz (see the module docstring) and its json sidecar.
    Members are reachable as attributes (`an.ks3`, `an.score_floor`) and by name
    (`an["beta2_d2.3"]`)."""

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

    def beta2(self, distance: str) -> np.ndarray:
        """beta^2 (eV^2) at "2.0", "2.3" or "3.0" Angstrom; NaN where undefined."""
        return self.members[f"beta2_d{distance}"]

    def scores(self) -> dict[str, np.ndarray]:
        """Stored predictor scores by predictor name (whatever version produced them)."""
        return {k[6:]: v for k, v in self.members.items() if k.startswith("score_")}

    def stored_scores(self, name: str, version: str | None) -> np.ndarray | None:
        """The member `score_<name>` if `meta["predictor_versions"]` records `version` for
        it, else None (the analysis key does not cover a model file: a retrained or replaced
        model must not be served the old model's scores)."""
        member = f"score_{name}"
        recorded = ((self.meta or {}).get("predictor_versions") or {}).get(name)
        if version is None or recorded != version or member not in self.members:
            return None
        return np.asarray(self.members[member], dtype=np.float64)

    def comps(self) -> dict[str, np.ndarray]:
        """The `pair_components` dict, rebuilt exactly from the members."""
        out = {k: self.members[k] for k in DIRECT}
        out |= {k[5:]: v for k, v in self.members.items() if k.startswith("comp_")}
        return out

    def cell(self, donor: int, acceptor: int) -> tuple[int, int]:
        """(i, j) of a donor and an acceptor token id (exact match), else
        ServeError(NOT_AN_ORBITAL_TOKEN)."""
        rows, cols = self.members["rows"], self.members["cols"]
        i, j = int(np.searchsorted(rows, donor)), int(np.searchsorted(cols, acceptor))
        if i >= len(rows) or rows[i] != donor:
            raise ServeError(Code.NOT_AN_ORBITAL_TOKEN, f"{donor} is not an occupied orbital token")
        if j >= len(cols) or cols[j] != acceptor:
            msg = f"{acceptor} is not a virtual orbital token"
            raise ServeError(Code.NOT_AN_ORBITAL_TOKEN, msg)
        return i, j


@dataclass
class PoolWork:
    """The in-memory bundle the worker tasks use."""

    state: PoolState
    rt: ReactantTokens
    sample: MechanismSample
    comps: dict[str, np.ndarray]
    an: PoolAnalysis
    npz: Path


# ---- computation


def beta_matrix(rt: ReactantTokens, kernels: PairKernels) -> np.ndarray:
    """Signed beta (eV) of every (occupied, virtual) pair: the identical cached call that
    `pair_components` makes (`kernels.main.beta_matrix`, physical KSParams, same=f==g)."""
    from ksmech.physics.params import KSParams

    tokens = [f.tokens for f in rt.fragments]
    r0 = np.cumsum([0, *[len(t.occupied()) for t in tokens]])
    c0 = np.cumsum([0, *[len(t.virtual()) for t in tokens]])
    out = np.zeros((int(r0[-1]), int(c0[-1])))
    for f, t_f in enumerate(tokens):
        for g, t_g in enumerate(tokens):
            occ, virt = t_f.occupied(), t_g.virtual()
            cell = np.s_[r0[f] : r0[f + 1], c0[g] : c0[g + 1]]
            out[cell] = kernels.main.beta_matrix(occ, t_f, virt, t_g, KSParams(), f == g)
    return out


def _identity_members(rt: ReactantTokens, sample: MechanismSample) -> dict[str, np.ndarray]:
    from ksmech.serve.candidates import PairReason, pair_reasons

    valid = np.asarray(sample.pair_valid, dtype=bool)
    reasons = pair_reasons([f.tokens for f in rt.fragments], rt.candidate)
    if not np.array_equal(reasons == PairReason.VALID, valid):
        n = int(((reasons == PairReason.VALID) != valid).sum())
        raise ServeError(
            Code.CONSISTENCY_FAILED, f"pair reasons disagree with pair_valid on {n} cells"
        )
    return {
        "rows": (rt.n_atoms + np.flatnonzero(rt.occupied)).astype(np.int64),
        "cols": (rt.n_atoms + np.flatnonzero(~rt.occupied)).astype(np.int64),
        "valid": valid,
        "reasons": reasons.astype(np.int8),
        "contact_sites": np.asarray(sample.pair_contact_sites, dtype=np.int64),
        "gap_sign": np.asarray(sample.pair_gap_sign, dtype=np.int8),
    }


def _physics_members(rt, sample, comps: dict, kernels) -> dict[str, np.ndarray]:
    from ksmech.physics.pairs import FEATURE_DISTANCES, F
    from ksmech.physics.ranking import physical_pair_z

    out = {k: np.asarray(comps[k]) for k in DIRECT}
    out |= {f"comp_{k}": np.asarray(v) for k, v in comps.items() if k not in DIRECT}
    beta = beta_matrix(rt, kernels)
    if not np.allclose(beta**2 * comps["gap_kernel"], comps["ks3"], rtol=1e-12, atol=0.0):
        raise ServeError(Code.CONSISTENCY_FAILED, "signed beta does not reproduce KS3 = beta^2 G")
    out["beta"] = beta
    zp = physical_pair_z(sample)
    for d in FEATURE_DISTANCES:
        k = F[f"log_beta2_d{d}"]
        defined = np.asarray(sample.pair_defined[..., k], dtype=bool)
        out[f"beta2_d{d}"] = np.where(defined, np.exp(zp[..., k]), np.nan)
    return out


def _score_members(state, rt, sample, comps, loader, scorers) -> tuple[dict, dict, dict]:
    """floor, frozen and score_<name>; plus {name: version} and {name: error}."""
    from ksmech.physics.cli import UNTRAINED_SCORE
    from ksmech.physics.ranking import SCORERS

    floor = SCORERS[UNTRAINED_SCORE](comps)
    if not np.array_equal(floor, -(comps["ks3"] + comps["ks2"])):
        raise ServeError(Code.CONSISTENCY_FAILED, "floor != -(ks3 + ks2)")
    out = {"floor": np.asarray(floor), "frozen": np.asarray(SCORERS["frozen_head"](comps))}
    versions, errors = {}, {}
    ctx = ScoringContext(sample, rt, loader, comps)
    for name, predictor in scorers.items():
        try:
            # the version first: a model file replaced meanwhile then fails the version check
            version = predictor.info().version
            out[f"score_{name}"] = np.asarray(predictor.score(ctx), dtype=np.float64)
            versions[name] = version
        except Exception as exc:  # noqa: BLE001 - recorded in the sidecar, never silent
            errors[name] = f"{type(exc).__name__}: {exc}"
            log.warning("analysis %s: predictor %s failed: %s", state.state_id, name, exc)
    return out, versions, errors


def _n_non_finite(members: dict[str, np.ndarray]) -> int:
    names = ("ks3", "ks2", "ks1", "gap_kernel", "amp", "beta", "floor", "frozen")
    return int(sum((~np.isfinite(np.asarray(members[k], dtype=float))).sum() for k in names))


def analyze_pool(
    state: PoolState,
    rt: ReactantTokens,
    sample: MechanismSample,
    comps: dict[str, np.ndarray],
    loader: LayeredTokenLoader,
    scorers: dict[str, Predictor],
    *,
    kernels: PairKernels | None = None,
) -> PoolAnalysis:
    """Every member of design 7.1 for one state. `kernels` must be the kernels the sample and
    comps came from (default: `oriented_kernels()` with the in-memory cache only)."""
    from ksmech.orbitals.tokens import CandidateConfig
    from ksmech.physics.pairs import oriented_kernels
    from ksmech.physics.params import KSParams

    t0 = time.perf_counter()
    kernels = kernels if kernels is not None else oriented_kernels()
    members = _identity_members(rt, sample)
    members |= _physics_members(rt, sample, comps, kernels)
    scores, versions, errors = _score_members(state, rt, sample, comps, loader, scorers)
    members |= scores
    members["fermi_smeared"] = np.array(bool(sample.fermi_smeared))
    meta = {
        "state_id": state.state_id,
        "ks_params": asdict(KSParams()),
        "candidate_config": jsonable(asdict(rt.candidate or CandidateConfig())),
        "predictor_versions": versions,
        "predictor_errors": errors,
        "n_non_finite": _n_non_finite(members),
        "n_valid": int(members["valid"].sum()),
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

        sample = reactant_sample(f"serve:{state.state_id}", "serve", rt, rt_ctx.kernels)
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
    from ksmech.physics.ranking import pair_components

    rt = _checked_rt(rt_ctx, state)
    sample = _sample(rt_ctx, state, rt)
    comps = pair_components(sample, rt_ctx.loader, rt_ctx.kernels.main)
    an = analyze_pool(
        state, rt, sample, comps, rt_ctx.loader, dict(rt_ctx.scorers), kernels=rt_ctx.kernels
    )
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
    """`ensure_analysis`, then the analysis with rt and the sample from the runtime's LRUs;
    comps are rebuilt from the `comp_*` members, never recomputed. rt and the sample must come
    from the token files the analysis was named by: a digest that changed meanwhile is
    TOKENS_CHANGED (503), never a bundle whose cells do not line up."""
    npz, digest = _ensure(rt_ctx, state)
    an = loaded_analysis(npz)
    rt = _checked_rt(rt_ctx, state)
    sample = _sample(rt_ctx, state, rt)
    after = analysis_digest(rt_ctx.loader, state)
    if after != digest:
        raise tokens_changed(state, digest, after)
    return PoolWork(state, rt, sample, an.comps(), an, npz)


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


def scores_path(s: ServeSettings, state_id: str, version: str, digest: str | None = None) -> Path:
    """`scores-<sha12>.npz` of one predictor version on the token files of digest `digest`
    (the scores' shape and values follow the tokens, the version alone does not cover them)."""
    raw = version if digest is None else f"{version}|{digest}"
    sha = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return state_path(state_id, s).parent / f"scores-{sha}.npz"


def t_score(rt: Runtime, payload: dict) -> dict:
    """{"state", "predictor"} -> {"path", "version"}: `scores-<sha12(version|digest)>.npz` with
    member `scores`, and the worker-side predictor version that produced it."""
    state = PoolState.from_dict(payload["state"])
    name = str(payload["predictor"])
    predictor = rt.scorers.get(name)
    if predictor is None:
        msg = f"predictor {name!r} is not available in the worker"
        raise ServeError(Code.PREDICTOR_UNAVAILABLE, msg, status=409, detail={"code": None})
    version = str(predictor.info().version)
    digest = analysis_digest(rt.loader, state)
    path = scores_path(rt.settings, state.state_id, version, digest)
    if digest is None or not path.is_file():
        work = pool_work(rt, state)  # MISSING_TOKENS when a fragment has no token file
        ctx = ScoringContext(work.sample, work.rt, rt.loader, work.comps)
        scores = np.asarray(predictor.score(ctx), dtype=np.float64)
        after = analysis_digest(rt.loader, state)
        if after != digest:
            raise tokens_changed(state, digest, after)
        save_npz(path, {"scores": scores}, rt.settings)
    return {"path": str(path), "version": version}
