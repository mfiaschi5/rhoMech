"""Value types and protocols that cross a package boundary: the leaf module of `ksmech.serve`.

Only the standard library and numpy are imported at run time; every other serve or upstream
type appears in annotations alone (under `TYPE_CHECKING`), so this module can be imported from
anywhere without pulling in RDKit, tblite or torch.

Every dataclass that crosses the worker boundary has `to_dict()` (JSON-able: tuples become
lists, numpy scalars become Python numbers, NaN and inf become None) and `from_dict(d)`, and the
pair round-trips exactly for finite values. A None in a non-optional float field comes back as
NaN, except in the log-probability fields (`logp`), where it comes back as -inf (the only
non-finite value they take). Nothing here does I/O.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

import numpy as np

if TYPE_CHECKING:
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.arrows.steps import MechanismSample
    from ksmech.physics.pairs import PairKernels
    from ksmech.serve.featurize import Featurizer
    from ksmech.serve.loader import LayeredTokenLoader
    from ksmech.serve.predictors import Predictor
    from ksmech.serve.settings import ServeSettings
    from ksmech.serve.worker import Scheduler

log = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


def to_jsonable(value: Any) -> Any:
    """`value` as JSON-able Python: numpy to Python, tuples and sets to lists, NaN/inf to None,
    nested `to_dict()` objects expanded. Raises TypeError on anything else."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, set | frozenset):
        return [to_jsonable(v) for v in sorted(value)]
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"not JSON-able: {type(value).__name__}")


def _f(x: Any) -> float:
    """A float field from JSON: None (a NaN or inf on the way out) comes back as NaN."""
    return float("nan") if x is None else float(x)


def _logp(x: Any) -> float:
    """A log-probability from JSON: None comes back as -inf."""
    return float("-inf") if x is None else float(x)


def _opt_f(x: Any) -> float | None:
    return None if x is None else float(x)


class _Jsonable:
    """`to_dict()` over the dataclass fields, through `to_jsonable`."""

    def to_dict(self) -> dict:
        return {f.name: to_jsonable(getattr(self, f.name)) for f in fields(self)}


# ---- pools and arrows


@dataclass(frozen=True)
class SpeciesCount(_Jsonable):
    species_key: str  # map-free resonance key (keys.species_key_of)
    token_keys: tuple[str, ...]  # unmapped_canonical forms present (resonance forms can differ)
    count: int
    charge: int
    resonance_flag: str  # always "OK" (keys.RESONANCE_OK: the sigma key is exact)

    @classmethod
    def from_dict(cls, d: dict) -> SpeciesCount:
        return cls(
            str(d["species_key"]),
            tuple(str(k) for k in d["token_keys"]),
            int(d["count"]),
            int(d["charge"]),
            str(d["resonance_flag"]),
        )


@dataclass(frozen=True)
class PoolState(_Jsonable):
    state_id: str  # "s" + sha256(mapped_smiles)[:20]
    mapped_smiles: str  # chem.canonical_mapped_smiles: Kekule, explicit H, every atom mapped
    pool_key: str  # ".".join(sorted species keys with repeats)
    pool_id: str  # "p" + sha256("pool-v2|" + pool_key)[:20]
    species: tuple[SpeciesCount, ...]  # sorted by species_key
    fragment_token_keys: tuple[str, ...]  # per fragment, GetMolFrags order of parse_mapped
    n_atoms: int

    @classmethod
    def from_dict(cls, d: dict) -> PoolState:
        return cls(
            str(d["state_id"]),
            str(d["mapped_smiles"]),
            str(d["pool_key"]),
            str(d["pool_id"]),
            tuple(SpeciesCount.from_dict(s) for s in d["species"]),
            tuple(str(k) for k in d["fragment_token_keys"]),
            int(d["n_atoms"]),
        )


@dataclass(frozen=True)
class ArrowSpec(_Jsonable):
    """= `PredictedArrow` fields: global token ids and global atom ids of the source state."""

    donor: int
    donor_site: int
    acceptor: int
    acceptor_site: int

    @classmethod
    def from_dict(cls, d: Mapping) -> ArrowSpec:
        return cls(
            int(d["donor"]), int(d["donor_site"]), int(d["acceptor"]), int(d["acceptor_site"])
        )

    def as_list(self) -> list[int]:
        return [self.donor, self.donor_site, self.acceptor, self.acceptor_site]


@dataclass(frozen=True)
class ExpandRequest(_Jsonable):
    predictor: str
    beam: int
    temperature: float | None  # None: the predictor's default
    candidate_scope: str  # "all" | "inter"
    max_scan: int

    @classmethod
    def from_dict(cls, d: dict) -> ExpandRequest:
        return cls(
            str(d["predictor"]),
            int(d["beam"]),
            _opt_f(d.get("temperature")),
            str(d["candidate_scope"]),
            int(d["max_scan"]),
        )


@dataclass(frozen=True)
class Limits(_Jsonable):
    max_species: int
    max_count: int
    max_atoms: int
    max_beam: int
    max_steps: int
    max_nodes: int

    @classmethod
    def from_dict(cls, d: dict) -> Limits:
        return cls(**{f.name: int(d[f.name]) for f in fields(cls)})


@dataclass
class ProposedChild(_Jsonable):
    state: PoolState  # product state (its own maps / Lewis form); already saved
    arrows: list[dict]  # ArrowOut-shaped dicts, each with its "physics" brief
    encoded: dict | None  # EncodedStep.to_dict()
    encode_reason: str | None  # ARROW_ENCODING_FAILED detail
    touched_maps: list[int]  # edit_diff(source, product).touched_maps
    score: float  # best member's predictor score
    probability: float  # sum of p over the scanned member proposals; a lower bound
    logp: float  # log(probability)
    rank: int  # 1-based rank of the best member among valid proposals
    n_alternatives: int
    alternatives: list[dict]  # <= 5 AlternativeOut dicts

    @classmethod
    def from_dict(cls, d: dict) -> ProposedChild:
        return cls(
            state=PoolState.from_dict(d["state"]),
            arrows=list(d["arrows"]),
            encoded=d.get("encoded"),
            encode_reason=d.get("encode_reason"),
            touched_maps=[int(m) for m in d["touched_maps"]],
            score=_f(d["score"]),
            probability=_f(d["probability"]),
            logp=_logp(d["logp"]),
            rank=int(d["rank"]),
            n_alternatives=int(d["n_alternatives"]),
            alternatives=list(d["alternatives"]),
        )


@dataclass
class Expansion(_Jsonable):
    ok: bool
    code: str | None  # FEATURIZATION_FAILED | ANALYSIS_FAILED | NO_CANDIDATES | ...
    upstream: str | None  # qm ReasonCode, ApplyReason, ...
    detail: str
    children: list[ProposedChild]  # <= beam, by probability desc, ties by child pool_key
    n_valid: int
    n_scanned: int
    scanned_mass: float
    unscanned_mass: float
    failures: dict[str, int]  # APPLY_FAILED:<ApplyReason>, RESONANCE_ONLY_STEP, ...
    failed_examples: list[dict]  # <= 10 {rank, arrows, probability, code, detail}
    fermi_smeared: bool
    warnings: list[str]

    @classmethod
    def failed(cls, code: str, detail: str = "", upstream: str | None = None) -> Expansion:
        """An expansion that scanned nothing: both masses are reported as 0."""
        return cls(False, str(code), upstream, detail, [], 0, 0, 0.0, 0.0, {}, [], False, [])

    @classmethod
    def from_dict(cls, d: dict) -> Expansion:
        return cls(
            ok=bool(d["ok"]),
            code=d.get("code"),
            upstream=d.get("upstream"),
            detail=str(d.get("detail") or ""),
            children=[ProposedChild.from_dict(c) for c in d["children"]],
            n_valid=int(d["n_valid"]),
            n_scanned=int(d["n_scanned"]),
            scanned_mass=_f(d["scanned_mass"]),
            unscanned_mass=_f(d["unscanned_mass"]),
            failures={str(k): int(v) for k, v in d["failures"].items()},
            failed_examples=list(d["failed_examples"]),
            fermi_smeared=bool(d["fermi_smeared"]),
            warnings=[str(w) for w in d["warnings"]],
        )


@dataclass
class StepOutcome(_Jsonable):
    """A user-applied step (override)."""

    state: PoolState  # already saved
    arrows: list[dict]  # ArrowOut dicts with physics briefs, site_source "user"
    encoded: dict | None
    encode_reason: str | None
    touched_maps: list[int]
    predictor_view: list[dict]  # per user arrow: {score, probability, rank, pair_valid, reason}
    issues: list[dict]  # warnings from check_arrows (ErrorBody-shaped)

    @classmethod
    def from_dict(cls, d: dict) -> StepOutcome:
        return cls(
            state=PoolState.from_dict(d["state"]),
            arrows=list(d["arrows"]),
            encoded=d.get("encoded"),
            encode_reason=d.get("encode_reason"),
            touched_maps=[int(m) for m in d["touched_maps"]],
            predictor_view=list(d["predictor_view"]),
            issues=list(d["issues"]),
        )


# ---- predictors


@dataclass(frozen=True)
class PredictorInfo(_Jsonable):
    name: str
    label: str
    kind: str  # "physics_floor" | "baseline" | "model" | "stub"
    available: bool
    code: str | None
    reason: str | None
    version: str | None
    checkpoint_hash: str | None
    score_unit: str
    default_temperature: float
    warnings: tuple[str, ...]
    description: str

    @classmethod
    def from_dict(cls, d: dict) -> PredictorInfo:
        return cls(
            str(d["name"]),
            str(d["label"]),
            str(d["kind"]),
            bool(d["available"]),
            d.get("code"),
            d.get("reason"),
            d.get("version"),
            d.get("checkpoint_hash"),
            str(d["score_unit"]),
            _f(d["default_temperature"]),
            tuple(str(w) for w in d["warnings"]),
            str(d["description"]),
        )


@dataclass(frozen=True)
class PredictorRef(_Jsonable):
    """What network.py and the jobs need to know about a predictor."""

    name: str
    version: str | None
    checkpoint_hash: str | None
    temperature: float

    @classmethod
    def from_dict(cls, d: dict) -> PredictorRef:
        return cls(str(d["name"]), d.get("version"), d.get("checkpoint_hash"), _f(d["temperature"]))


@dataclass(frozen=True)
class ScoringContext:
    """Built in the worker by `analysis.pool_work`, or by the re-check. Not serialized."""

    sample: MechanismSample
    rt: ReactantTokens
    loader: LayeredTokenLoader
    comps: dict[str, np.ndarray]  # pair_components(...) at the physical KSParams


@dataclass(frozen=True)
class PoolScores:
    """A predictor's view of one pool, in the pair-block index space. Worker-side only."""

    scores: np.ndarray  # (n_occ, n_virt) float64, -inf where not valid
    valid: np.ndarray  # (n_occ, n_virt) bool (pair_valid & scope mask)
    logp: np.ndarray  # log_softmax_valid(scores, valid, temperature)
    contact_sites: np.ndarray  # (n_occ, n_virt, 2) global atoms
    rows: np.ndarray  # (n_occ,) global token id of row i = n_atoms + flatnonzero(occupied)[i]
    cols: np.ndarray  # (n_virt,)
    temperature: float


@dataclass(frozen=True)
class ProposedStep(_Jsonable):
    arrows: tuple[ArrowSpec, ...]  # floor / gbdt / stub: exactly one arrow
    site_sources: tuple[str, ...]  # per arrow: "contact" | "main_centre" | "rule"
    score: float
    logp: float
    rank: int  # 1-based among valid proposals
    cell: tuple[int, int] | None  # (i, j) of the scored pair; None for multi-arrow steps

    @classmethod
    def from_dict(cls, d: dict) -> ProposedStep:
        cell = d.get("cell")
        return cls(
            tuple(ArrowSpec.from_dict(a) for a in d["arrows"]),
            tuple(str(s) for s in d["site_sources"]),
            _f(d["score"]),
            _logp(d["logp"]),
            int(d["rank"]),
            None if cell is None else (int(cell[0]), int(cell[1])),
        )


# ---- small shared machinery


class LRU(Generic[K, V]):
    """A thread-safe least-recently-used map (OrderedDict + lock). Misses are not stored."""

    def __init__(self, maxsize: int):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = int(maxsize)
        self._data: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: K, value: V) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def pop(self, key: K) -> V | None:
        with self._lock:
            return self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._data


class ChildScan:
    """The grouping and stopping rule of a beam expansion (design section 4.3), shared by
    `expansion.expand_state` and the test chemistry.

    Proposals are added in rank order with their probability p. A proposal that gives a child
    pool key joins that child's group (the first member is the representative); a failed one
    is counted under its failure code. Every p is added to `scanned_mass`, failures included:
    p is the predictor's distribution before application.
    """

    def __init__(self, beam: int, max_scan: int):
        if beam < 1 or max_scan < 1:
            raise ValueError("beam and max_scan must be >= 1")
        self.beam = int(beam)
        self.max_scan = int(max_scan)
        self.mass: dict[str, float] = {}
        self.n_scanned = 0
        self.scanned_mass = 0.0
        self.failures: Counter = Counter()

    def add(self, p: float, child_key: str | None, failure: str | None = None) -> bool:
        """Record one scanned proposal. Returns True when it is the first member of a new
        child. Exactly one of `child_key` and `failure` must be given."""
        if (child_key is None) == (failure is None):
            raise ValueError("a scanned proposal needs exactly one of child_key and failure")
        self.n_scanned += 1
        self.scanned_mass += float(p)
        if child_key is None:
            self.failures[str(failure)] += 1
            return False
        new = child_key not in self.mass
        self.mass[child_key] = self.mass.get(child_key, 0.0) + float(p)
        return new

    @property
    def unscanned_mass(self) -> float:
        return 1.0 - self.scanned_mass

    def ranked(self) -> list[str]:
        """Every child key by mass desc, ties by key asc."""
        return sorted(self.mass, key=lambda k: (-self.mass[k], k))

    def kept(self) -> list[str]:
        """At most `beam` child keys by mass desc, ties by key asc."""
        return self.ranked()[: self.beam]

    def should_stop(self, n_remaining: int) -> str | None:
        """ "exhausted" (nothing left to scan), "max_scan", "bound" (the kept set is final:
        U < p_B - p_{B+1} with U the unscanned mass; strict, so a tie keeps scanning) or None."""
        if n_remaining <= 0:
            return "exhausted"
        if self.n_scanned >= self.max_scan:
            return "max_scan"
        masses = [self.mass[k] for k in self.ranked()]
        if len(masses) >= self.beam:
            p_b = masses[self.beam - 1]
            p_next = masses[self.beam] if len(masses) > self.beam else 0.0
            if self.unscanned_mass < p_b - p_next:
                return "bound"
        return None


def log_softmax_valid(scores: np.ndarray, valid: np.ndarray, temperature: float) -> np.ndarray:
    """log softmax of scores / T over the valid cells, in float64; -inf on invalid cells (and
    everywhere when no cell is valid). T must be > 0. A valid -inf score gets -inf; NaN or +inf
    in a valid cell is an error (ValueError), never a silent probability."""
    t = float(temperature)
    if not t > 0 or not math.isfinite(t):
        raise ValueError(f"temperature must be a finite number > 0, got {temperature!r}")
    s = np.asarray(scores, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    if s.shape != v.shape:
        raise ValueError(f"scores {s.shape} and valid {v.shape} differ in shape")
    out = np.full(s.shape, -np.inf)
    if not v.any():
        return out
    z = s[v] / t
    if np.isnan(z).any() or np.isposinf(z).any():
        raise ValueError("a valid cell has a NaN or +inf score")
    peak = z.max()
    if not np.isfinite(peak):
        return out  # every valid score is -inf
    out[v] = z - (peak + np.log(np.exp(z - peak).sum()))
    return out


# ---- protocols


class Runtime(Protocol):
    """The worker-side per-process state; `tasks.WorkerRuntime` implements it."""

    settings: ServeSettings
    featurizer: Featurizer
    loader: LayeredTokenLoader
    scorers: Mapping[str, Predictor]  # available predictors by name (asked on every call)
    rt_cache: LRU[str, ReactantTokens]  # by state_id, 16 entries
    sample_cache: LRU[str, MechanismSample]  # by state_id, 16 entries

    @property
    def kernels(self) -> PairKernels: ...  # lazy oriented_kernels(settings.oriented_dir)

    def reactant_tokens(self, state: PoolState) -> ReactantTokens: ...

    def occupations(self, key: str) -> np.ndarray | None: ...


class JobContext(Protocol):
    """What a job handler and a Chemistry call see; `jobs.RunnerJobContext` implements it."""

    job_id: str
    settings: ServeSettings
    scheduler: Scheduler

    def progress(self, **fields: Any) -> None: ...

    def cancelled(self) -> bool: ...

    def skip(self, code: str, detail: str = "", **where: Any) -> None: ...


@dataclass
class LocalJobContext:
    """A concrete `JobContext` for tests and the CLI: it records instead of writing job files.
    `progress` appends the fields, `cancelled` returns `cancel_flag`, `skip` counts the code,
    keeps the record in `skip_log` and logs it at WARNING."""

    job_id: str
    settings: ServeSettings
    scheduler: Scheduler
    progress_log: list[dict] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    cancel_flag: bool = False
    skip_log: list[dict] = field(default_factory=list)

    def progress(self, **fields: Any) -> None:
        self.progress_log.append(dict(fields))

    def cancelled(self) -> bool:
        return self.cancel_flag

    def skip(self, code: str, detail: str = "", **where: Any) -> None:
        self.skipped[str(code)] += 1
        self.skip_log.append({"code": str(code), "detail": detail, **where})
        log.warning("job %s skipped [%s] %s %s", self.job_id, code, detail, where or "")


PairsProvider = Callable[[JobContext], Mapping[tuple[str, str], str] | None]
"""(reactant_pool_key, product_pool_key) -> a PMechDB step_id; None when PMechDB is missing."""
NetworkSource = Callable[[str], tuple[Any, dict]]
"""job_id -> (NetworkOut, the job's PredictIn as dict); raises JOB_NOT_FOUND / JOB_NOT_DONE."""
JobSubmitter = Callable[[str, dict], dict]
"""(kind, request) -> JobCreatedOut as dict (JobRunner.submit(...).created_out())."""
