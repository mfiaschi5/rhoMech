"""The API-side context every service takes (design 6.4): settings, the scheduler, the
predictor registry, the featurizer and its loader, versions, and the API-side LRUs.

Services never compute: `ensure_analysis` and `scores` go through the scheduler with
`interactive` (202 Pending while a task runs), and the API process only `np.load`s the files
the worker wrote. Plain dicts with the schema field names are returned (no pydantic here).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ksmech.serve import API_SCHEMA, capabilities
from ksmech.serve.analysis import (
    TOKENS_CHANGED,
    PoolAnalysis,
    analysis_digest,
    analysis_paths,
    load_analysis,
    read_failure,
    write_failure,
)
from ksmech.serve.contracts import LRU, PoolState, PredictorRef
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.featurize import Featurizer, featurize_label, make_featurizer
from ksmech.serve.loader import LayeredTokenLoader
from ksmech.serve.pools import load_state
from ksmech.serve.predictors import PredictorRegistry
from ksmech.serve.settings import ServeSettings, TaskEnv
from ksmech.serve.versions import (
    SPEC_LABEL,
    artifact_versions,
    physics_fingerprint,
    spec_sha12,
    spec_version,
    token_fingerprint,
)
from ksmech.serve.worker import InlineScheduler, Scheduler, interactive

if TYPE_CHECKING:
    from ksmech.arrows.reactants import ReactantTokens

log = logging.getLogger(__name__)

ANALYZE_TIMEOUT_S = 900
SCORE_TIMEOUT_S = 300
NOTE_NO_STOP = (
    "the floor and gbdt have no END_MECH head: prediction stops at max_steps, a turnover or an "
    "empty frontier, never by a learned stop"
)
NOTE_NETWORK_STOP = (
    "the network's END_MECH head is untrained on PMechDB (lambda_stop = 0) and is ignored: "
    "prediction stops at max_steps, a turnover, a recurring pool or an empty frontier, never by "
    "a learned stop"
)
NOTE_SYNTHETIC = (
    "synthetic chemistry (SYNTHETIC-EHT/v1, charge-iterated extended Hueckel): a test vehicle, "
    "not GFN2 physics"
)
NOTE_INTER = "candidate_scope=inter masks intramolecular pairs; the physics is unchanged"


@cache
def physics_constants() -> dict[str, Any]:
    """KSParams and the default CandidateConfig, as plain values."""
    from ksmech.orbitals.tokens import CandidateConfig
    from ksmech.physics.params import KSParams

    cc = asdict(CandidateConfig())
    cc["fractional_band"] = list(cc["fractional_band"])
    return {
        "ks_params": {k: float(v) for k, v in asdict(KSParams()).items()},
        "candidate_config": cc,
    }


@dataclass
class ApiContext:
    settings: ServeSettings
    scheduler: Scheduler
    registry: PredictorRegistry
    featurizer: Featurizer
    loader: LayeredTokenLoader  # featurizer.loader()
    chemistry: str  # settings.chemistry, or "stub" when a StubChemistry is injected
    spec_version: str
    physics_fp: str
    token_fp: str
    artifact_versions: dict[str, int | str]
    rt_lru: LRU[str, ReactantTokens]  # 32 entries, by state_id
    analysis_lru: LRU[str, PoolAnalysis]  # 32 entries, by npz path
    rt_digests: dict[str, str | None] = field(default_factory=dict)  # state_id -> rt_lru digest

    def state(self, state_id: str) -> PoolState:
        """A saved state; ServeError(POOL_NOT_FOUND, 404)."""
        return load_state(state_id, self.settings)

    def reactant_tokens(self, state: PoolState) -> ReactantTokens:
        """The state's tokens in its global index space (needs every token file); missing
        tokens are POOL_NOT_ANALYZED (409). `rt_lru` serves a state only while its token
        digest is the one the entry was built with; a changed digest rebuilds it (and drops
        the state's keys from the loader memo)."""
        sid = state.state_id
        dig = analysis_digest(self.loader, state)
        if self.rt_digests.get(sid) == dig:
            rt = self.rt_lru.get(sid)
            if rt is not None:
                return rt
        else:
            self.rt_lru.pop(sid)
            if sid in self.rt_digests:  # built before with other token files
                for k in set(state.fragment_token_keys):
                    self.loader.forget(k)
        from ksmech.arrows.reactants import CorrespondenceError, MissingTokens, ReactantTokens

        try:
            rt = ReactantTokens.from_smiles(state.mapped_smiles, self.loader)
        except MissingTokens as exc:
            detail = {"code": str(Code.MISSING_TOKENS), "keys": list(exc.keys)}
            raise ServeError(Code.POOL_NOT_ANALYZED, str(exc), status=409, detail=detail) from exc
        except CorrespondenceError as exc:
            detail = {"code": str(Code.CORRESPONDENCE), "detail": str(exc)}
            raise ServeError(Code.POOL_NOT_ANALYZED, str(exc), status=409, detail=detail) from exc
        self.rt_lru.put(sid, rt)
        self.rt_digests[sid] = dig
        return rt

    def interactive(
        self, name: str, payload: dict, *, label: str, timeout_s: float, wait_s=None
    ) -> dict:
        return interactive(
            self.scheduler,
            name,
            payload,
            label=label,
            wait_s=self.settings.interactive_wait_s if wait_s is None else float(wait_s),
            timeout_s=timeout_s,
        )

    def _load(self, npz: Path) -> PoolAnalysis:
        key = str(npz)
        an = self.analysis_lru.get(key)
        if an is None:
            an = load_analysis(npz)
            self.analysis_lru.put(key, an)
        return an

    def _not_analyzed(self, state: PoolState, failure: dict) -> ServeError:
        msg = f"pool {state.state_id} could not be analyzed: {failure.get('code')}"
        if failure.get("detail"):
            msg += f": {failure['detail']}"
        return ServeError(Code.POOL_NOT_ANALYZED, msg, status=409, detail=failure)

    def _featurize_missing(self, state: PoolState, wait_s=None) -> None:
        timeout = self.settings.qm_timeout_s + 120
        for key in dict.fromkeys(state.fragment_token_keys):
            if self.loader.has(key):
                continue
            label = featurize_label(self.featurizer, key)
            try:
                self.interactive(
                    "featurize", {"key": key}, label=label, timeout_s=timeout, wait_s=wait_s
                )
            except ServeError as exc:
                extra = {"digest": None, "stage": "featurize", "key": key}
                write_failure(self.settings, state, self.physics_fp, exc, **extra)
                failure = read_failure(self.settings, state.state_id, self.physics_fp)
                fallback = {"code": str(exc.code), "upstream": exc.upstream, "detail": exc.message}
                raise self._not_analyzed(state, failure or (fallback | extra)) from exc

    def ensure_tokens(self, state: PoolState) -> ReactantTokens:
        """The state's tokens without the oriented pair analysis (spec v8 port: pools, tokens,
        geometry and cubes need only the featurized fragments). Featurizes the missing
        fragments (`Pending` while a task runs); a featurize failure recorded for the current
        token state is POOL_NOT_ANALYZED (409) with the failure as detail, as in
        `ensure_analysis`."""
        if analysis_digest(self.loader, state) is None:
            failure = read_failure(self.settings, state.state_id, self.physics_fp)
            if failure is not None and failure.get("digest") is None:
                raise self._not_analyzed(state, failure)
            self._featurize_missing(state)
        return self.reactant_tokens(state)

    def ensure_analysis(self, state: PoolState, wait_s: float | None = None) -> PoolAnalysis:
        """The state's analysis: featurize the missing fragments, then the `analyze` task, then
        `load_analysis` (LRU by npz path). Raises `Pending` while a task runs, and
        POOL_NOT_ANALYZED (409, detail = the failure sidecar) when the analysis failed for the
        current physics fingerprint and token digest; TOKENS_CHANGED (503, the token files
        changed during the analysis) passes through, since a retry succeeds."""
        s, fp = self.settings, self.physics_fp
        digest = analysis_digest(self.loader, state)
        if digest is not None:
            npz, _ = analysis_paths(s, state.state_id, fp, digest)
            if npz.is_file():
                return self._load(npz)
        failure = read_failure(s, state.state_id, fp)
        if failure is not None and failure.get("digest") == digest:
            raise self._not_analyzed(state, failure)
        self._featurize_missing(state, wait_s)
        label = f"analysis of {state.pool_key}"
        try:
            res = self.interactive(
                "analyze",
                {"state": state.to_dict()},
                label=label,
                timeout_s=ANALYZE_TIMEOUT_S,
                wait_s=wait_s,
            )
        except ServeError as exc:
            if str(exc.code) == TOKENS_CHANGED:
                raise  # transient (503, never recorded): the same request succeeds on retry
            fallback = {"code": str(exc.code), "upstream": exc.upstream, "detail": exc.message}
            raise self._not_analyzed(
                state, read_failure(s, state.state_id, fp) or fallback
            ) from exc
        return self._load(Path(res["npz"]))

    def provenance(
        self,
        predictor: str | None,
        *,
        temperature: float | None = None,
        scope: str = "all",
        notes: Sequence[str] = (),
        ref: PredictorRef | None = None,
    ) -> dict:
        """`Provenance` fields; `ref` (the job's resolved predictor reference) gives the
        version, the checkpoint hash and the network's checkpoint."""
        info = None
        if predictor is not None:
            try:
                info = self.registry.info_of(predictor)
            except ServeError:
                info = None
        if temperature is None and info is not None:
            temperature = info.default_temperature
        c = physics_constants()
        src = ref if ref is not None else info
        version = None if src is None else src.version
        sha = None if src is None else src.checkpoint_hash
        ck = None if ref is None else ref.checkpoint
        if ck and ck.get("smoke"):
            notes = [*notes, f"checkpoint {ck['run']}/{ck['file']} is a smoke run (--limit)"]
        return {
            "predictor": predictor or "none",
            "predictor_version": version,
            "checkpoint_hash": sha,
            "checkpoint": None if ck is None else dict(ck),
            "spec_version": self.spec_version,
            "physics_fingerprint": self.physics_fp,
            "artifact_versions": dict(self.artifact_versions),
            "chemistry": self.chemistry,
            "ks_params": dict(c["ks_params"]),
            "temperature": None if temperature is None else float(temperature),
            "candidate_scope": scope,
            "candidate_config": dict(c["candidate_config"]),
            "notes": self._notes(predictor, info, scope, notes),
        }

    def _notes(self, predictor, info, scope: str, notes: Sequence[str]) -> list[str]:
        out = list(notes)
        if predictor in ("floor", "gbdt", "user"):
            out.append(NOTE_NO_STOP)
        elif predictor == "network":
            out.append(NOTE_NETWORK_STOP)
        if info is not None:
            out += list(info.warnings)
        if self.chemistry == "synthetic":
            out.append(NOTE_SYNTHETIC)
        if scope == "inter":
            out.append(NOTE_INTER)
        return list(dict.fromkeys(out))


def build_context(
    settings: ServeSettings,
    *,
    scheduler: Scheduler | None = None,
    registry: PredictorRegistry | None = None,
    featurizer: Featurizer | None = None,
) -> ApiContext:
    """Defaults: InlineScheduler(TaskEnv(settings)), PredictorRegistry(settings, physics_fp),
    make_featurizer(settings)."""
    featurizer = featurizer or make_featurizer(settings)
    physics_fp = physics_fingerprint()
    return ApiContext(
        settings=settings,
        scheduler=scheduler or InlineScheduler(TaskEnv(settings)),
        registry=registry or PredictorRegistry(settings, physics_fp),
        featurizer=featurizer,
        loader=featurizer.loader(),
        chemistry=settings.chemistry,
        spec_version=spec_version(settings.spec_path),
        physics_fp=physics_fp,
        token_fp=token_fingerprint(),
        artifact_versions=dict(artifact_versions()),
        rt_lru=LRU(32),
        analysis_lru=LRU(32),
    )


def meta(ctx: ApiContext) -> dict:
    """`MetaOut` fields except `frontend_built` and `dev` (api.py adds them)."""
    import ksmech

    c = physics_constants()
    return {
        "api_schema": API_SCHEMA,
        "app_version": str(ksmech.__version__),
        "spec_label": SPEC_LABEL,
        "spec_sha256_12": spec_sha12(ctx.settings.spec_path),
        "spec_version": ctx.spec_version,
        "artifact_versions": dict(ctx.artifact_versions),
        "physics_fingerprint": ctx.physics_fp,
        "chemistry": ctx.chemistry,
        "qm_method": ctx.featurizer.method,
        "token_method": ctx.featurizer.method,
        "ks_params": dict(c["ks_params"]),
        "candidate_config": dict(c["candidate_config"]),
        "limits": ctx.settings.limits.to_dict(),
        "worker": ctx.scheduler.status().to_dict(),
        "data_root": str(ctx.settings.data_root),
        "capabilities": capabilities.as_meta(ctx.settings),
    }
