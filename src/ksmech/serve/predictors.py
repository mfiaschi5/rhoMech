"""Predictors: score every (occupied, virtual) cell of a pool and propose ranked steps.

`Predictor.propose(pool, ctx, k)` is the brief's `Predictor.propose(pool)`: at most k ranked
steps, each a tuple of `ArrowSpec`s with its own sites. The expansion iterates it and never
builds arrows itself, so a predictor chooses its sites and may return multi-arrow steps.

- `floor`: the untrained physics floor -(KS3 + KS2) on the oriented kernel at d = 2.3 A with the
  bonded-sites validity filter (`SCORERS[UNTRAINED_SCORE]` over `pair_components`).
- `gbdt`: the Prompt 5 ranker (`baseline.GBDTRanker` over `candidate_set`), with a feature
  compatibility check and its file hash as version (adapter A21).
- `transformer`: registered, always unavailable until a checkpoint and an inference entry point
  exist.
- `stub` (tests and e2e diagnostics only): the floor plus rule bonuses, with rule sites.

`info()` is cheap and runs on the API side; `score()` runs in the worker (lazy upstream
imports: `physics.ranking`, `physics.cli` and `physics.baseline` load torch).
"""

from __future__ import annotations

import hashlib
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ksmech.serve.contracts import (
    ArrowSpec,
    PoolScores,
    PredictorInfo,
    PredictorRef,
    ProposedStep,
    ScoringContext,
)
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.versions import file_sha256, floor_version, source_constant

if TYPE_CHECKING:
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.serve.settings import ServeSettings


def _sample_version_text() -> str:
    """SAMPLE_VERSION of the working tree's arrows/steps.py, read with ast (no torch import)."""
    try:
        return str(source_constant("arrows/steps.py", "SAMPLE_VERSION"))
    except (LookupError, OSError, SyntaxError, ValueError) as exc:
        return f"unknown ({type(exc).__name__}: {exc})"


# measured values only: the GBDT npz meta holds {n_trees, lr, features}, no sample version
GBDT_SAMPLE_WARNING = (
    "the GBDT file records no SAMPLE_VERSION; compatibility is checked by feature names and "
    f"count only (serve builds features with SAMPLE_VERSION {_sample_version_text()})"
)
TRANSFORMER_REASON = (
    "src/ksmech/model has no inference entry point (Prompt 6) and no checkpoint exists under "
    "data/models/transformer/"
)
ORDER = ("floor", "gbdt", "transformer", "stub")


def _tau() -> float:
    from ksmech.physics.params import KSParams

    return float(KSParams().tau)


def main_centre_site(rt: ReactantTokens, token_id: int) -> int:
    """Global atom of an orbital token's main centre (`center_atoms[0]`)."""
    j = int(token_id) - rt.n_atoms
    off = int(rt.atom_offsets[int(rt.orbital_fragment[j])])
    return off + int(rt.orbitals[j].descriptors.center_atoms[0])


def ranked_cells(pool: PoolScores, k: int) -> list[tuple[int, int, int]]:
    """(position, i, j) of the k best valid cells: score desc, ties by (row, col) asc."""
    flat = np.flatnonzero(np.asarray(pool.valid).ravel())
    scores = np.asarray(pool.scores, dtype=float).ravel()[flat]
    order = np.argsort(-scores, kind="stable")[: max(0, int(k))]
    n_virt = pool.valid.shape[1]
    return [(pos, int(flat[o] // n_virt), int(flat[o] % n_virt)) for pos, o in enumerate(order)]


def contact_arrow(pool: PoolScores, ctx: ScoringContext, i: int, j: int) -> tuple[ArrowSpec, str]:
    """The arrow of cell (i, j) at its contact sites; a -1 site falls back to the token's main
    centre (site source "main_centre")."""
    donor, acceptor = int(pool.rows[i]), int(pool.cols[j])
    a, b = (int(x) for x in pool.contact_sites[i, j])
    source = "contact"
    if a < 0:
        a, source = main_centre_site(ctx.rt, donor), "main_centre"
    if b < 0:
        b, source = main_centre_site(ctx.rt, acceptor), "main_centre"
    return ArrowSpec(donor, a, acceptor, b), source


class Predictor(ABC):
    name: ClassVar[str]

    @abstractmethod
    def info(self) -> PredictorInfo:
        """API side; cheap."""

    @abstractmethod
    def score(self, ctx: ScoringContext) -> np.ndarray:
        """Worker side; (n_occ, n_virt) float64, -inf where not valid."""

    def propose(self, pool: PoolScores, ctx: ScoringContext, k: int) -> list[ProposedStep]:
        """At most k ranked steps: valid cells by score desc, ties by (row, col) asc; one arrow
        per cell at its contact sites; logp = pool.logp[i, j]; rank = position + 1."""
        out = []
        for pos, i, j in ranked_cells(pool, k):
            arrow, source = contact_arrow(pool, ctx, i, j)
            out.append(self._step(pool, arrow, source, pos, i, j))
        return out

    @staticmethod
    def _step(pool: PoolScores, arrow: ArrowSpec, source: str, pos: int, i: int, j: int):
        return ProposedStep(
            (arrow,), (source,), float(pool.scores[i, j]), float(pool.logp[i, j]), pos + 1, (i, j)
        )


def floor_scores(ctx: ScoringContext) -> np.ndarray:
    """where(pair_valid, SCORERS[UNTRAINED_SCORE](comps), -inf)."""
    from ksmech.physics.cli import UNTRAINED_SCORE
    from ksmech.physics.ranking import SCORERS

    raw = SCORERS[UNTRAINED_SCORE](ctx.comps)
    return np.where(ctx.sample.pair_valid, raw, -np.inf).astype(np.float64)


class FloorPredictor(Predictor):
    name = "floor"

    def __init__(self, fp: str) -> None:
        self.fp = fp

    def info(self) -> PredictorInfo:
        return PredictorInfo(
            name=self.name,
            label="Untrained KS floor",
            kind="physics_floor",
            available=True,
            code=None,
            reason=None,
            version=floor_version(self.fp),
            checkpoint_hash=None,
            score_unit="eV",
            default_temperature=_tau(),
            warnings=(),
            description=(
                "-(KS3 + KS2) on the oriented contact kernel at d = 2.3 A with the bonded-sites "
                "validity filter; untrained physics, no END_MECH head"
            ),
        )

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return floor_scores(ctx)


@dataclass
class _GBDTState:
    code: str | None
    reason: str | None
    sha: str | None
    model: object | None
    signature: tuple[int, int] | None = None  # (st_mtime_ns, st_size) before the check


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


class GBDTPredictor(Predictor):
    """The baseline GBDT ranker; loaded (and checked) on first use, and again whenever the
    file's mtime or size changed (a retrained or replaced model is followed by every process;
    an unchanged file is never re-read)."""

    name = "gbdt"

    def __init__(self, path: Path, fp: str) -> None:
        self.path = Path(path)
        self.fp = fp
        self._state: _GBDTState | None = None
        self._lock = threading.Lock()

    def _load(self) -> _GBDTState:
        with self._lock:
            sig = _file_signature(self.path)  # before the read: a later rewrite differs
            if self._state is None or self._state.signature != sig:
                self._state = self._check()
                self._state.signature = sig
            return self._state

    def _readable(self) -> str | None:
        """None when the file is an npz with a JSON `meta` member, else the reason."""
        import json

        try:
            with np.load(self.path, allow_pickle=False) as z:
                json.loads(str(z["meta"]))
        except Exception as exc:  # noqa: BLE001 - reported as GBDT_LOAD_FAILED
            return f"{type(exc).__name__}: {exc}"
        return None

    def _check(self) -> _GBDTState:
        if not self.path.is_file():
            return _GBDTState(Code.GBDT_MISSING, f"no GBDT model at {self.path}", None, None)
        sha = file_sha256(self.path)
        unreadable = self._readable()
        if unreadable is not None:
            return _GBDTState(Code.GBDT_LOAD_FAILED, unreadable, sha, None)
        try:
            from ksmech.physics import baseline as bl
            from ksmech.serve.settings import pin_threads

            pin_threads()  # baseline imports torch
            model = bl.load_gbdt(self.path)
        except (ValueError, KeyError) as exc:
            return _GBDTState(
                Code.GBDT_FEATURES_MISMATCH, f"{type(exc).__name__}: {exc}", sha, None
            )
        except Exception as exc:  # noqa: BLE001 - reported as unavailable, never raised
            return _GBDTState(Code.GBDT_LOAD_FAILED, f"{type(exc).__name__}: {exc}", sha, None)
        n = len(bl.FEATURE_NAMES)
        bad = [
            t
            for t, tree in enumerate(model.trees)
            if np.size(tree.feature) and tree.feature.max() >= n
        ]
        if bad or len(model.edges) != n:
            reason = f"trees {bad[:5]} split on features >= {n} or {len(model.edges)} edges"
            return _GBDTState(Code.GBDT_FEATURES_MISMATCH, reason, sha, None)
        return _GBDTState(None, None, sha, model)

    def info(self) -> PredictorInfo:
        st = self._load()
        return PredictorInfo(
            name=self.name,
            label="GBDT ranker (baseline)",
            kind="baseline",
            available=st.model is not None,
            code=None if st.code is None else str(st.code),
            reason=st.reason,
            version=None if st.sha is None else f"gbdt/{st.sha[:16]}/{self.fp}",
            checkpoint_hash=st.sha,
            score_unit="logit",
            default_temperature=1.0,
            warnings=(GBDT_SAMPLE_WARNING,),
            description="gradient-boosted listwise ranker over the 61 candidate features",
        )

    def score(self, ctx: ScoringContext) -> np.ndarray:
        st = self._load()
        if st.model is None:
            detail = {"code": str(st.code), "reason": st.reason}
            raise ServeError(Code.PREDICTOR_UNAVAILABLE, str(st.reason), status=409, detail=detail)
        from ksmech.physics import baseline as bl

        cs = bl.candidate_set(ctx.sample, ctx.loader)
        out = np.full(ctx.sample.pair_valid.shape, -np.inf)
        out[cs.rows, cs.cols] = st.model.score(cs.X)
        return out


class TransformerPredictor(Predictor):
    name = "transformer"

    def info(self) -> PredictorInfo:
        return PredictorInfo(
            name=self.name,
            label="Transformer (Prompt 6)",
            kind="model",
            available=False,
            code=str(Code.TRANSFORMER_NOT_AVAILABLE),
            reason=TRANSFORMER_REASON,
            version=None,
            checkpoint_hash=None,
            score_unit="logit",
            default_temperature=1.0,
            warnings=(),
            description="KS-biased transformer over orbital tokens; disabled until a checkpoint",
        )

    def score(self, ctx: ScoringContext) -> np.ndarray:
        detail = {"code": str(Code.TRANSFORMER_NOT_AVAILABLE), "reason": TRANSFORMER_REASON}
        raise ServeError(Code.PREDICTOR_UNAVAILABLE, TRANSFORMER_REASON, status=409, detail=detail)


@dataclass(frozen=True)
class StubRule:
    """Matches cell (i, j) when the donor has a centre of `donor_site_element`, the acceptor a
    centre of `acceptor_site_element`, and each set that is not None equals the token's set of
    centre elements (`descriptors.all_center_atoms`)."""

    donor_centre_elements: frozenset[str] | None
    donor_site_element: str
    acceptor_site_element: str
    acceptor_centre_elements: frozenset[str] | None
    bonus: float

    def canonical(self) -> tuple:
        """A hash-seed independent description (frozenset reprs are not)."""
        d = None if self.donor_centre_elements is None else sorted(self.donor_centre_elements)
        a = None if self.acceptor_centre_elements is None else sorted(self.acceptor_centre_elements)
        return (d, self.donor_site_element, self.acceptor_site_element, a, float(self.bonus))

    def matches(self, donor: dict[str, float], acceptor: dict[str, float]) -> bool:
        """`donor` / `acceptor`: element -> largest centre weight of that element."""
        ok_sites = self.donor_site_element in donor and self.acceptor_site_element in acceptor
        ok_d = self.donor_centre_elements is None or set(donor) == self.donor_centre_elements
        ok_a = (
            self.acceptor_centre_elements is None or set(acceptor) == self.acceptor_centre_elements
        )
        return ok_sites and ok_d and ok_a


DEFAULT_STUB_RULES = (StubRule(frozenset({"O"}), "O", "C", frozenset({"C", "Cl"}), 50.0),)


def _centres_by_element(rt: ReactantTokens, token_id: int) -> dict[str, tuple[float, int]]:
    """element -> (largest centre weight, its global atom) over the token's centres."""
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    z = [a.atomic_number for f in rt.fragments for a in f.tokens.atoms]
    out: dict[str, tuple[float, int]] = {}
    for atom, w in sorted(rt.centres(token_id).items()):
        el = pt.GetElementSymbol(int(z[atom]))
        if el not in out or w > out[el][0]:
            out[el] = (float(w), int(atom))
    return out


class StubPredictor(Predictor):
    """The floor plus rule bonuses; a matched cell's arrow takes its sites by rule."""

    name = "stub"

    def __init__(self, fp: str, rules: tuple[StubRule, ...] = DEFAULT_STUB_RULES) -> None:
        self.fp = fp
        self.rules = tuple(rules)

    @property
    def version(self) -> str:
        digest = hashlib.sha256(repr([r.canonical() for r in self.rules]).encode()).hexdigest()
        return f"stub/{digest[:12]}/{self.fp}"

    def info(self) -> PredictorInfo:
        return PredictorInfo(
            name=self.name,
            label="Stub (tests)",
            kind="stub",
            available=True,
            code=None,
            reason=None,
            version=self.version,
            checkpoint_hash=None,
            score_unit="eV",
            default_temperature=_tau(),
            warnings=("diagnostic predictor: the floor plus hand-written rule bonuses",),
            description="the untrained floor plus rule bonuses, with rule sites; tests only",
        )

    def first_rule(self, rt: ReactantTokens, donor: int, acceptor: int) -> StubRule | None:
        d = {k: v[0] for k, v in _centres_by_element(rt, donor).items()}
        a = {k: v[0] for k, v in _centres_by_element(rt, acceptor).items()}
        return next((r for r in self.rules if r.matches(d, a)), None)

    def bonus(self, ctx: ScoringContext) -> np.ndarray:
        rows = ctx.rt.n_atoms + np.flatnonzero(ctx.rt.occupied)
        cols = ctx.rt.n_atoms + np.flatnonzero(~ctx.rt.occupied)
        out = np.zeros((len(rows), len(cols)))
        don = [{k: v[0] for k, v in _centres_by_element(ctx.rt, int(r)).items()} for r in rows]
        acc = [{k: v[0] for k, v in _centres_by_element(ctx.rt, int(c)).items()} for c in cols]
        for i, d in enumerate(don):
            for j, a in enumerate(acc):
                out[i, j] = sum(r.bonus for r in self.rules if r.matches(d, a))
        return out

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return np.where(ctx.sample.pair_valid, floor_scores(ctx) + self.bonus(ctx), -np.inf)

    def propose(self, pool: PoolScores, ctx: ScoringContext, k: int) -> list[ProposedStep]:
        out = []
        for pos, i, j in ranked_cells(pool, k):
            donor, acceptor = int(pool.rows[i]), int(pool.cols[j])
            rule = self.first_rule(ctx.rt, donor, acceptor)
            if rule is None:
                arrow, source = contact_arrow(pool, ctx, i, j)
            else:
                a = _centres_by_element(ctx.rt, donor)[rule.donor_site_element][1]
                b = _centres_by_element(ctx.rt, acceptor)[rule.acceptor_site_element][1]
                arrow, source = ArrowSpec(donor, a, acceptor, b), "rule"
            out.append(self._step(pool, arrow, source, pos, i, j))
        return out


class PredictorRegistry:
    """Every predictor of one process: floor, gbdt, transformer, and the stub when enabled."""

    def __init__(self, settings: ServeSettings, fp: str) -> None:
        self.settings = settings
        self.fp = fp
        self._predictors: dict[str, Predictor] = {
            "floor": FloorPredictor(fp),
            "gbdt": GBDTPredictor(settings.gbdt_file, fp),
            "transformer": TransformerPredictor(),
        }
        if settings.enable_stub_predictor:
            self._predictors["stub"] = StubPredictor(fp)

    def names(self) -> list[str]:
        return [n for n in ORDER if n in self._predictors]

    def info(self) -> list[PredictorInfo]:
        return [self._predictors[n].info() for n in self.names()]

    def _known(self, name: str) -> Predictor:
        if name not in self._predictors:
            msg = f"unknown predictor {name!r}; one of {self.names()}"
            raise ServeError(Code.PREDICTOR_UNKNOWN, msg, status=422)
        return self._predictors[name]

    def info_of(self, name: str) -> PredictorInfo:
        return self._known(name).info()

    def get(self, name: str) -> Predictor:
        """The predictor, if available. Raises PREDICTOR_UNKNOWN (422) or
        PREDICTOR_UNAVAILABLE (409, detail {code, reason})."""
        p = self._known(name)
        info = p.info()
        if not info.available:
            detail = {"code": info.code, "reason": info.reason}
            msg = f"predictor {name!r} is unavailable: {info.reason}"
            raise ServeError(Code.PREDICTOR_UNAVAILABLE, msg, status=409, detail=detail)
        return p

    def ref(self, name: str, temperature: float | None = None) -> PredictorRef:
        info = self.get(name).info()
        t = info.default_temperature if temperature is None else float(temperature)
        return PredictorRef(info.name, info.version, info.checkpoint_hash, t)

    def available(self) -> dict[str, Predictor]:
        """Available predictors by name (the worker's `Runtime.scorers`)."""
        return {
            n: self._predictors[n] for n in self.names() if self._predictors[n].info().available
        }
