"""Predictors (spec v9): each proposes ranked reaction matrices R for one pool state.

`Predictor.propose(rt_ctx, work, beam, scope, skips)` returns `Candidate`s on pool atoms (the
atom indices of `parse_mapped(state.mapped_smiles)`), best first; the expansion applies them,
merges them by the product's pool key and keeps the beam (`serve.predict.expansion`). Every
candidate a
predictor drops is counted in `skips` under a reason code.

- `floor`: the untrained physics floor (spec, "Evaluation protocol"). Candidate anchors are
  the atom pairs i < j of the pool: "form" when the bond order stays at most 3, "break" when
  the pair is bonded. They are ranked by |G_atom[i, j]| (`serve.science.gatom`) and the top `beam`
  are completed: a forming anchor from its dominant channels (r, s) in each direction (the
  donor token gives the source of the pair, the acceptor token the leaving bond), applied
  with upstream `apply_predicted_arrows`, one candidate per leaving choice the mask allows
  (`leaving_choices`); a breaking anchor as the heterolysis towards either atom. The score is
  |G_atom| (1/eV), not a probability.
- `gbdt`: the geometry-free LightGBM lambdarank model of `ksmech-physics gbdt`
  (`gbdt_lgbm_molecule_geomfree.npz` under the shared processed directory). It ranks the
  valid orbital pairs; the `beam` best are completed into R the same way, at the sites
  (a, b) of r and s with the largest w_a^r w_b^s. The score is the model's ranking score.
- `network`: the spec-v9 model through a local checkpoint (`serve.predict.network_predictor`, with
  its decoder `model.decode`; neither is part of the physics fingerprint, the
  network's version names the checkpoint's sha and the decoder's code instead).

`info()` is cheap and runs on the API side; `propose()` runs in the worker.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol

import numpy as np

from ksmech.serve.core.contracts import USER_PREDICTOR, USER_REF, PredictorInfo, PredictorRef
from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.core.versions import file_sha256, floor_version

if TYPE_CHECKING:
    from ksmech.arrows.delta_be import ReactionMatrix
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.arrows.steps import MechanismSample
    from ksmech.serve.core.contracts import ExpandRequest, PoolState, Runtime
    from ksmech.serve.core.settings import ServeSettings
    from ksmech.serve.science.analysis import PoolWork

log = logging.getLogger(__name__)

ORDER = ("floor", "gbdt", "network")
N_CHANNELS = 2  # channels completed per direction of a forming anchor
SCORE_NOTE = "scores rank candidates; they are not probabilities"


@dataclass(frozen=True)
class Candidate:
    """One proposed step on pool atoms."""

    matrix: ReactionMatrix
    anchor: tuple[int, int, str]  # (i, j, "form" | "break"), i < j
    score: float
    logp: float | None = None
    tie: float = 0.0  # secondary key within one score (the channel term), larger first
    channel: dict | None = None  # how the completion was built (diagnostic)


@dataclass(frozen=True)
class NetworkStep:
    """What a `NetworkModel` returns: R on the global atoms of `ReactantTokens`."""

    matrix: ReactionMatrix
    logp: float
    anchor: tuple[int, int, str] | None = None


class NetworkModel(Protocol):
    """The adapter the spec-v9 network plugs in through (pool -> ranked (R, logp)):
    `serve.predict.network_predictor.CheckpointNetwork` over a local checkpoint, or a fake in tests.
    `state` is the pool state (B + R is checked on it), `scope` "all" | "inter", and every
    completion the decoder drops is counted in `skips`."""

    version: str

    def decode(
        self,
        sample: MechanismSample,
        rt: ReactantTokens,
        beam: int,
        *,
        state: PoolState,
        scope: str,
        skips: Counter,
    ) -> list[NetworkStep]: ...


class Predictor(ABC):
    name: ClassVar[str]
    kind: ClassVar[str]

    @abstractmethod
    def info(self) -> PredictorInfo:
        """API side; cheap."""

    @abstractmethod
    def propose(
        self, rt_ctx: Runtime, work: PoolWork, beam: int, scope: str, skips: Counter
    ) -> list[Candidate]:
        """Worker side; candidates best first."""

    def bind(self, req: ExpandRequest) -> Predictor:
        """The predictor that serves one expansion request (the network binds the request's
        checkpoint); the others serve every request as they are."""
        return self

    def make_ref(self, temperature: float | None, checkpoint=None) -> PredictorRef:
        """The reference a network records. Only the network takes a checkpoint."""
        if checkpoint is not None:
            msg = f"a checkpoint applies only to the network predictor, not {self.name!r}"
            raise ServeError(Code.REQUEST_INVALID, msg, status=422)
        info = self.info()
        t = info.default_temperature if temperature is None else float(temperature)
        return PredictorRef(info.name, info.version, info.checkpoint_hash, t)


def _info(name, label, kind, available, code, reason, version, sha, unit, warnings, desc, prob):
    return PredictorInfo(
        name=name,
        label=label,
        kind=kind,
        available=available,
        code=code,
        reason=reason,
        version=version,
        checkpoint_hash=sha,
        score_unit=unit,
        default_temperature=1.0,
        warnings=tuple(warnings),
        description=desc,
        scores_are_probabilities=prob,
    )


# ---- completion of an arrow (r, a, s, b) into reaction matrices


def _order(mol, i: int, j: int) -> int:
    from ksmech.arrows.apply import TYPE_TO_ORDER

    bond = mol.GetBondBetweenAtoms(int(i), int(j))
    return 0 if bond is None else TYPE_TO_ORDER.get(bond.GetBondType(), 1)


def leaving_options(rt: ReactantTokens, arrow) -> list[int]:
    """The leaving bonds the mask allows for one arrow (`apply_predicted.leaving_choices` on
    the state after the arrow's source and new bond): NONE first when allowed, then the
    global atom x of every bond b-x. Raises ArrowApplyError when the source has no pair."""
    from rdkit import Chem

    from ksmech.arrows.align import NONE_LEAVING
    from ksmech.arrows.apply_predicted import EditState, _step, leaving_choices

    state = EditState(Chem.Mol(rt.mol))
    _, room = _step(state, rt, arrow, set())
    cands, none_ok = leaving_choices(state, rt, arrow, room)
    return ([NONE_LEAVING] if none_ok else []) + [int(c.atom) for c in cands]


def complete_arrow(rt: ReactantTokens, arrow, skips: Counter) -> list[tuple[ReactionMatrix, int]]:
    """(R on pool atoms, leaving atom) of every leaving choice of `arrow` that applies."""
    from ksmech.arrows.apply import ArrowApplyError
    from ksmech.arrows.apply_predicted import apply_predicted_arrows
    from ksmech.arrows.delta_be import matrix_between

    try:
        options = leaving_options(rt, arrow)
    except ArrowApplyError as exc:
        skips[f"COMPLETION_FAILED:{exc.reason}"] += 1
        return []
    out = []
    for x in options:
        res = apply_predicted_arrows(rt.mol, [replace(arrow, leaving=x)], rt)
        if not res.ok:
            skips[f"COMPLETION_FAILED:{res.reason}"] += 1
            continue
        out.append((matrix_between(rt.mol, res.mol), x))
    return out


def _arrow_candidates(rt, arrow, anchor, score, term, skips, logp=None) -> list[Candidate]:
    from ksmech.arrows.align import NONE_LEAVING

    out = []
    for R, x in complete_arrow(rt, arrow, skips):
        i, j = anchor[0], anchor[1]
        if not any(sorted((a, b)) == [i, j] and d > 0 for a, b, d in R.bonds):
            skips["COMPLETION_WITHOUT_ANCHOR"] += 1
            continue
        leaving = None if x == NONE_LEAVING else int(rt.mol_atom[x])
        channel = {
            "donor_token": int(arrow.donor),
            "acceptor_token": int(arrow.acceptor),
            "donor_site": int(rt.mol_atom[arrow.donor_site]),
            "acceptor_site": int(rt.mol_atom[arrow.acceptor_site]),
            "term": float(term),
            "leaving": leaving,
        }
        out.append(Candidate(R, anchor, float(score), logp, abs(float(term)), channel))
    return out


def _heterolysis(i: int, j: int, score: float) -> list[Candidate]:
    from ksmech.arrows.delta_be import ReactionMatrix

    anchor = (i, j, "break")
    return [
        Candidate(ReactionMatrix(((i, j, -1),), ((k, 2),)), anchor, float(score), None, 0.0, None)
        for k in (j, i)
    ]


def ranked(cands: list[Candidate]) -> list[Candidate]:
    """Best first (score, then the channel term), duplicates of one R dropped."""
    seen, out = set(), []
    order = sorted(cands, key=lambda c: (-c.score, -c.tie, c.anchor, c.matrix.to_list()))
    for c in order:
        if c.matrix not in seen:
            seen.add(c.matrix)
            out.append(c)
    return out


# ---- floor


def floor_anchors(work: PoolWork, scope: str) -> list[tuple[float, int, int, str]]:
    """(|G_atom|, i, j, sign) of every candidate anchor on pool atoms, best first."""
    rt, g = work.rt, np.abs(work.pg.g_atom)
    frag = rt.atom_fragment
    a, b = np.triu_indices(rt.n_atoms, k=1)
    keep = g[a, b] > 0
    if scope == "inter":
        keep &= frag[a] != frag[b]
    out = []
    for ga, gb in zip(a[keep], b[keep], strict=True):
        i, j = sorted((rt.mol_atom[int(ga)], rt.mol_atom[int(gb)]))
        order, score = _order(rt.mol, i, j), float(g[ga, gb])
        if order < 3:
            out.append((score, i, j, "form"))
        if order >= 1:
            out.append((score, i, j, "break"))
    return sorted(out, key=lambda x: (-x[0], x[3] != "form", x[1], x[2]))


def anchor_candidates(work: PoolWork, anchor, score: float, skips: Counter) -> list[Candidate]:
    """The completions of one anchor (module docstring)."""
    from ksmech.arrows.apply_predicted import PredictedArrow

    i, j, sign = anchor
    if sign == "break":
        return _heterolysis(i, j, score)
    rt, pg = work.rt, work.pg
    gi, gj = rt.global_atom[i], rt.global_atom[j]
    out: list[Candidate] = []
    for d, acc in ((gi, gj), (gj, gi)):
        for row, col, term in pg.top_directed(d, acc, N_CHANNELS):
            arrow = PredictedArrow(int(pg.rows[row]), d, int(pg.cols[col]), acc)
            out += _arrow_candidates(rt, arrow, anchor, score, term, skips)
    if not out:
        skips["ANCHOR_NOT_COMPLETED"] += 1
    return out


class FloorPredictor(Predictor):
    name = "floor"
    kind = "floor"

    def __init__(self, fp: str) -> None:
        self.fp = fp

    def info(self) -> PredictorInfo:
        desc = (
            "untrained physics floor: anchors ranked by the pooled gap factor |G_atom| over the "
            "valid donor/acceptor pairs, completed from their dominant orbital channels with "
            "the leaving-bond mask; " + SCORE_NOTE
        )
        return _info(
            self.name, "Physics floor (|G_atom|)", self.kind, True, None, None,
            floor_version(self.fp), None, "1/eV", (SCORE_NOTE,), desc, False,
        )  # fmt: skip

    def propose(self, rt_ctx, work, beam, scope, skips) -> list[Candidate]:
        out: list[Candidate] = []
        for score, i, j, sign in floor_anchors(work, scope)[: max(1, int(beam))]:
            out += anchor_candidates(work, (i, j, sign), score, skips)
        return ranked(out)


# ---- gbdt


@dataclass
class _GBDTState:
    code: str | None
    reason: str | None
    sha: str | None
    model: object | None
    signature: tuple[int, int] | None = None


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def pair_sites(rt: ReactantTokens, r: int, s: int) -> tuple[int, int] | None:
    """The global sites (a, b), a a centre of r and b one of s, a != b, whose bond order can
    still rise, with the largest w_a^r w_b^s (ties: lowest a, b); None when there is none."""
    best = None
    for a, wa in sorted(rt.centres(r).items()):
        for b, wb in sorted(rt.centres(s).items()):
            if a == b or _order(rt.mol, rt.mol_atom[a], rt.mol_atom[b]) >= 3:
                continue
            if best is None or wa * wb > best[0]:
                best = (wa * wb, a, b)
    return None if best is None else (best[1], best[2])


def _read_lgbm(path: Path, lgbm):
    """The ranker of a verified file (`load_lgbm` without its own verify, which would resolve
    the inputs against this checkout)."""
    import json

    import lightgbm as lgb

    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        booster = lgb.Booster(model_str=str(z["model"]))
    return lgbm.LGBMRanker(booster, meta["features"], meta["history"])


def gbdt_problems(path: Path, root: Path | None, features: list[str]) -> list[str]:
    """What makes the LightGBM file stale (empty when current): `physics.lgbm.load_lgbm`'s
    checks (kind, version, features, every `file:` input re-hashed, a clean build), with the
    relative inputs resolved against `root`, the shared checkout that built the model, and not
    against this checkout (`provenance.input_path`)."""
    from ksmech.data import provenance as prov
    from ksmech.physics.lgbm import LGBM_VERSION
    from ksmech.serve.chem.loader import _resolve_input

    header = prov.read_header(path)
    if header is None:
        return ["no provenance header"]
    out = []
    if header.kind != "baseline_lgbm" or header.schema_version != LGBM_VERSION:
        out.append(
            f"{header.kind} v{header.schema_version}, expected baseline_lgbm v{LGBM_VERSION}"
        )
    if header.config.get("features") != features:
        out.append("features differ from physics.baseline.FEATURE_NAMES")
    for name, was in header.inputs.items():
        if name.startswith("file:"):
            p = _resolve_input(name, root)
            if not p.is_file():
                out.append(f"input missing: {p}")
            elif prov.current_hash(p) != was:
                out.append(f"input changed: {p}")
    if header.code_dirty:
        out.append(f"built from a dirty tree (HEAD {header.code_git_hash[:12]})")
    return out


class GBDTPredictor(Predictor):
    """The geometry-free LightGBM ranker; loaded (and verified) on first use, again whenever
    the file's mtime or size changed."""

    name = "gbdt"
    kind = "gbdt"

    def __init__(self, path: Path, fp: str, root: Path | None = None) -> None:
        self.path = Path(path)
        self.fp = fp
        self.root = root  # the checkout the model was built in (its relative inputs)
        self._state: _GBDTState | None = None
        self._lock = threading.Lock()

    def _load(self) -> _GBDTState:
        with self._lock:
            sig = _file_signature(self.path)
            if self._state is None or self._state.signature != sig:
                self._state = self._check()
                self._state.signature = sig
            return self._state

    def _check(self) -> _GBDTState:
        if not self.path.is_file():
            return _GBDTState(Code.GBDT_MISSING, f"no GBDT model at {self.path}", None, None)
        sha = file_sha256(self.path)
        try:
            from ksmech.physics import baseline as bl
            from ksmech.physics import lgbm

            problems = gbdt_problems(self.path, self.root, list(bl.FEATURE_NAMES))
            if problems:
                return _GBDTState("STALE_ARTIFACT", "; ".join(problems), sha, None)
            model = _read_lgbm(self.path, lgbm)  # verified above, against the shared root
        except ValueError as exc:
            return _GBDTState(Code.GBDT_FEATURES_MISMATCH, f"{exc}", sha, None)
        except Exception as exc:  # noqa: BLE001 - reported as unavailable (StaleArtifact, ...)
            return _GBDTState(Code.GBDT_LOAD_FAILED, f"{type(exc).__name__}: {exc}", sha, None)
        return _GBDTState(None, None, sha, model)

    def info(self) -> PredictorInfo:
        st = self._load()
        version = None if st.sha is None else f"gbdt/{st.sha[:16]}/{self.fp}"
        desc = (
            "LightGBM lambdarank over the geometry-free pair features of every valid orbital "
            "pair; the best pairs are completed into reaction matrices; " + SCORE_NOTE
        )
        return _info(
            self.name, "GBDT ranker (geometry-free)", self.kind, st.model is not None,
            None if st.code is None else str(st.code), st.reason, version, st.sha, "rank score",
            (SCORE_NOTE,), desc, False,
        )  # fmt: skip

    def _pair_scores(self, rt_ctx, work) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from ksmech.physics import baseline as bl

        st = self._load()
        if st.model is None:
            raise ServeError(Code.PREDICTOR_UNAVAILABLE, str(st.reason), status=409)
        cs = bl.candidate_set(work.sample, rt_ctx.loader)
        return cs.rows, cs.cols, np.asarray(st.model.score(cs.X), dtype=np.float64)

    def propose(self, rt_ctx, work, beam, scope, skips) -> list[Candidate]:
        from ksmech.arrows.apply_predicted import PredictedArrow

        rows, cols, scores = self._pair_scores(rt_ctx, work)
        rt, pg = work.rt, work.pg
        out: list[Candidate] = []
        taken = 0
        for k in np.argsort(-scores, kind="stable"):
            r, s = int(pg.rows[rows[k]]), int(pg.cols[cols[k]])
            if (
                scope == "inter"
                and rt.orbital_fragment[r - rt.n_atoms] == rt.orbital_fragment[s - rt.n_atoms]
            ):
                continue
            sites = pair_sites(rt, r, s)
            if sites is None:
                skips["PAIR_WITHOUT_SITES"] += 1
                continue
            i, j = sorted((rt.mol_atom[sites[0]], rt.mol_atom[sites[1]]))
            arrow = PredictedArrow(r, sites[0], s, sites[1])
            term = float(pg.G[rows[k], cols[k]])
            out += _arrow_candidates(rt, arrow, (i, j, "form"), scores[k], term, skips)
            taken += 1
            if taken >= max(1, int(beam)):
                break
        return ranked(out)


class PredictorRegistry:
    """Every predictor of one process: floor, gbdt and network (`network`: a fixed
    `NetworkModel` for tests, else the local checkpoints)."""

    def __init__(self, settings: ServeSettings, fp: str, network: NetworkModel | None = None):
        from ksmech.serve.predict.network_predictor import NetworkPredictor

        self.settings = settings
        self.fp = fp
        self._predictors: dict[str, Predictor] = {
            "floor": FloorPredictor(fp),
            "gbdt": GBDTPredictor(settings.gbdt_file, fp, settings.shared_root),
            "network": NetworkPredictor(settings, fp, network),
        }

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

    def ref(self, name: str, temperature: float | None = None, checkpoint=None) -> PredictorRef:
        """The reference of an available predictor; `checkpoint` ({source?, run, file}) names
        the network's checkpoint (None: its default)."""
        if name == USER_PREDICTOR:
            return USER_REF
        return self.get(name).make_ref(temperature, checkpoint)

    def available(self) -> dict[str, Predictor]:
        """Available predictors by name (the worker's `Runtime.scorers`)."""
        return {
            n: self._predictors[n] for n in self.names() if self._predictors[n].info().available
        }
