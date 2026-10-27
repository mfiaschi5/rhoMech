"""The `network` predictor (spec v9): a local `ksmech-train fit` checkpoint (`serve.predict.models`)
through the engine's step decoder (`model.decode`). Torch is imported only in the worker.

- **Availability** (`network_status`), with the reason text of each state, in this order:
  TRANSFORMER_NOT_AVAILABLE (torch or the model code is not importable), CHECKPOINT_NOT_FOUND
  (no `<source>/<run>/<file>.pt`), CHECKPOINT_REFUSED (every checkpoint is refused: a dirty
  run), NETWORK_ADAPTER_FAILED (every usable checkpoint failed to load in the worker, with the
  error; a failure is recorded per file version in `data/serve/network/adapter_failures.json`
  and retried after RETRY_AFTER_S or when the file changes).
- **Default checkpoint**: `models.default_checkpoint` (the best usable non-smoke run by
  validation top-1) among the checkpoints without a recent adapter failure.
- **Reference** (`make_ref`, API side): the request's checkpoint ({source?, run, file}) or the
  default, hashed; the `PredictorRef` carries its sha256 and `checkpoint_ref` (run, file,
  file version, epoch, val top-1), and the version `network/<source>/<run>/<file>@<sha12>/
  dec-<code sha8>/<physics fingerprint>` (the decoder's code is not in the fingerprint).
- **Worker** (`bind(req)`): the checkpoint of the request, refused when it changed since it was
  hashed (CHECKPOINT_CHANGED), loaded once per file version (`embedding.cached_model`, CPU);
  the pool sample gets its Lewis structure and Wiberg indices (`arrows.lewis.reactant_lewis`),
  `CheckpointNetwork.decode` runs the decoder with B + R checked by `stepapply.apply_matrix`
  and `charge_issue` (no non-metal left at |formal charge| >= 2 that it did not carry).
  Scores are log P(R) (`scores_are_probabilities`); END_MECH is not read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from ksmech.model import decode
from ksmech.model.decode import NONMETALS
from ksmech.serve.core.contracts import PredictorInfo, PredictorRef
from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.predict.predictors import Candidate, NetworkModel, NetworkStep, Predictor

if TYPE_CHECKING:
    from ksmech.serve.core.contracts import ExpandRequest
    from ksmech.serve.core.settings import ServeSettings
    from ksmech.serve.predict.models import Checkpoint

log = logging.getLogger(__name__)

RETRY_AFTER_S = 600.0
LOAD_FAILURES = (Code.CHECKPOINT_LOAD_FAILED, Code.CHECKPOINT_REFUSED)
LABEL = "Network (spec v9)"
DESCRIPTION = (
    "the spec-v9 transformer: anchor pointer and entry classifiers; the B best anchors are "
    "completed by a best-first search over the entries, kept when they conserve charge and "
    "every valence, ranked by log P(R) and merged by product"
)


@cache
def decoder_sha8() -> str:
    """8 hex of sha256 over the decoder's and this module's code."""
    h = hashlib.sha256()
    for path in (Path(decode.__file__), Path(__file__)):
        h.update(path.read_bytes())
    return h.hexdigest()[:8]


def checkpoint_key(ck: Checkpoint) -> str:
    return f"{ck.source}/{ck.run}/{ck.file}"


def checkpoint_ref(ck: Checkpoint, sha: str) -> dict:
    """What a network records of its checkpoint (`PredictorRef.checkpoint`)."""
    return {
        "source": ck.source,
        "run": ck.run,
        "file": ck.file,
        "sha256": sha,
        "mtime_ns": int(ck.mtime_ns),
        "size_bytes": int(ck.size_bytes),
        "epoch": ck.epoch,
        "val_top1": ck.metrics.get("val_top1"),
        "encoder": ck.encoder,
        "smoke": bool(ck.smoke),
    }


def version_of(ck: Checkpoint, sha: str, fp: str) -> str:
    return f"network/{checkpoint_key(ck)}@{sha[:12]}/dec-{decoder_sha8()}/{fp}"


# ---- adapter failures, recorded by the worker


def failures_path(settings: ServeSettings) -> Path:
    return settings.serve / "network" / "adapter_failures.json"


def read_failures(settings: ServeSettings) -> dict[str, dict]:
    path = failures_path(settings)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("unreadable %s [ADAPTER_FAILURES_UNREADABLE]: %s", path, exc)
        return {}
    return doc if isinstance(doc, dict) else {}


def recent_failure(failures: Mapping[str, dict], ck: Checkpoint, now: float | None = None):
    """The recorded failure of this file version of `ck` within RETRY_AFTER_S, or None."""
    rec = failures.get(checkpoint_key(ck))
    if not isinstance(rec, dict) or rec.get("signature") != list(ck.signature):
        return None
    age = (time.time() if now is None else now) - float(rec.get("at") or 0.0)
    return rec if age < RETRY_AFTER_S else None


def _write_failures(settings: ServeSettings, doc: dict) -> None:
    from ksmech.serve.chem.pools import atomic_write_text

    atomic_write_text(failures_path(settings), json.dumps(doc, indent=1, sort_keys=True), settings)


def record_failure(settings: ServeSettings, ck: Checkpoint, exc: ServeError) -> None:
    doc = read_failures(settings)
    doc[checkpoint_key(ck)] = {
        "signature": list(ck.signature),
        "code": str(exc.code),
        "message": exc.message,
        "at": time.time(),
    }
    _write_failures(settings, doc)


def clear_failure(settings: ServeSettings, ck: Checkpoint) -> None:
    doc = read_failures(settings)
    if doc.pop(checkpoint_key(ck), None) is not None:
        _write_failures(settings, doc)


# ---- status and the default checkpoint


@dataclass
class NetworkStatus:
    code: str | None
    reason: str | None
    default: Checkpoint | None
    warnings: list[str] = field(default_factory=list)


def _failure_text(ck: Checkpoint, rec: dict) -> str:
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(rec.get("at") or 0)))
    return (
        f"the adapter failed on {checkpoint_key(ck)}.pt at {when}: {rec.get('code')}: "
        f"{rec.get('message')} (retried after {RETRY_AFTER_S / 60:.0f} min or when the file "
        "changes)"
    )


def network_status(settings: ServeSettings) -> NetworkStatus:
    """The network's availability and default checkpoint (module docstring)."""
    from ksmech.serve.predict.models import default_checkpoint, list_checkpoints, model_roots
    from ksmech.serve.predict.models import missing_modules as _missing

    missing = _missing()
    if missing:
        msg = f"not importable here: {', '.join(missing)} (the network needs torch and the model)"
        return NetworkStatus(str(Code.TRANSFORMER_NOT_AVAILABLE), msg, None)
    ckpts = list_checkpoints(settings)
    if not ckpts:
        where = " or ".join(str(r) for r in model_roots(settings))
        msg = f"no checkpoint <source>/<run>/<file>.pt under {where}"
        return NetworkStatus(str(Code.CHECKPOINT_NOT_FOUND), msg, None)
    usable = [c for c in ckpts if c.usable]
    if not usable:
        msg = "every local checkpoint is refused: " + "; ".join(
            f"{checkpoint_key(c)}: {c.reason}" for c in ckpts
        )
        return NetworkStatus(str(Code.CHECKPOINT_REFUSED), msg, None)
    failures = read_failures(settings)
    failed = [(c, recent_failure(failures, c)) for c in usable]
    texts = [_failure_text(c, rec) for c, rec in failed if rec]
    default = default_checkpoint([c for c, rec in failed if not rec])
    if default is None:
        return NetworkStatus(str(Code.NETWORK_ADAPTER_FAILED), "; ".join(texts), None)
    warnings = texts + ([f"the default checkpoint {checkpoint_key(default)} is a smoke run"]
                        if default.smoke else [])  # fmt: skip
    return NetworkStatus(None, None, default, warnings)


# ---- the predictor


def _info(available, code, reason, version, sha, warnings) -> PredictorInfo:
    return PredictorInfo(
        name="network",
        label=LABEL,
        kind="network",
        available=bool(available),
        code=code,
        reason=reason,
        version=version,
        checkpoint_hash=sha,
        score_unit="log P",
        default_temperature=1.0,
        warnings=tuple(warnings),
        description=DESCRIPTION,
        scores_are_probabilities=True,
    )


class NetworkPredictor(Predictor):
    """The network over the local checkpoints, or over a fixed `NetworkModel` (`model`, tests)."""

    name = "network"
    kind = "network"

    def __init__(self, settings: ServeSettings, fp: str, model: NetworkModel | None = None):
        self.settings = settings
        self.fp = fp
        self.model = model

    def info(self) -> PredictorInfo:
        from ksmech.serve.predict.models import checkpoint_sha256

        if self.model is not None:
            return _info(True, None, None, f"network/{self.model.version}", None, [])
        st = network_status(self.settings)
        if st.default is None:
            return _info(False, st.code, st.reason, None, None, [])
        sha = checkpoint_sha256(st.default)
        version = version_of(st.default, sha, self.fp)
        return _info(True, None, None, version, sha, st.warnings)

    def resolve(self, checkpoint: Mapping | None) -> Checkpoint:
        """The request's checkpoint ({source?, run, file?}: CHECKPOINT_NOT_FOUND 404,
        CHECKPOINT_REFUSED 409) or the default (PREDICTOR_UNAVAILABLE 409 without one)."""
        from ksmech.serve.embedding.service import usable_checkpoint

        if checkpoint is not None and checkpoint.get("run"):
            run, file = str(checkpoint["run"]), str(checkpoint.get("file") or "best")
            return usable_checkpoint(self.settings, run, file, checkpoint.get("source"))
        st = network_status(self.settings)
        if st.default is None:
            detail = {"code": st.code, "reason": st.reason}
            msg = f"predictor 'network' is unavailable: {st.reason}"
            raise ServeError(Code.PREDICTOR_UNAVAILABLE, msg, status=409, detail=detail)
        return st.default

    def make_ref(self, temperature: float | None, checkpoint=None) -> PredictorRef:
        from ksmech.serve.predict.models import checkpoint_sha256

        t = 1.0 if temperature is None else float(temperature)
        if self.model is not None:
            return PredictorRef(self.name, f"network/{self.model.version}", None, t)
        ck = self.resolve(checkpoint)
        sha = checkpoint_sha256(ck)
        version = version_of(ck, sha, self.fp)
        return PredictorRef(self.name, version, sha, t, checkpoint_ref(ck, sha))

    def bind(self, req: ExpandRequest) -> Predictor:
        return _Bound(self, req.checkpoint)

    def propose(self, rt_ctx, work, beam, scope, skips) -> list[Candidate]:
        return _Bound(self, None).propose(rt_ctx, work, beam, scope, skips)


class _Bound(Predictor):
    """The network for one expansion request: its checkpoint (`ref`, None: the default)."""

    name = "network"
    kind = "network"

    def __init__(self, parent: NetworkPredictor, ref: Mapping | None) -> None:
        self.parent = parent
        self.ref = ref

    def info(self) -> PredictorInfo:
        return self.parent.info()

    def _adapter(self) -> NetworkModel:
        from ksmech.serve.embedding.service import cached_model

        if self.parent.model is not None:
            return self.parent.model
        settings, ref = self.parent.settings, self.ref
        ck = self.parent.resolve(ref)
        if ref and ref.get("mtime_ns") is not None:
            want = (int(ref["mtime_ns"]), int(ref["size_bytes"]))
            if tuple(ck.signature) != want:
                msg = (
                    f"{checkpoint_key(ck)}.pt changed since this network chose it "
                    f"(sha {str(ref.get('sha256'))[:12]}); start a new prediction or expand again"
                )
                raise ServeError(Code.CHECKPOINT_CHANGED, msg, status=409)
        try:
            model, ckpt = cached_model(ck.path, ck.signature)
        except ServeError as exc:
            if exc.code in LOAD_FAILURES:
                record_failure(settings, ck, exc)
            raise
        clear_failure(settings, ck)
        return CheckpointNetwork(model, checkpoint_key(ck), ckpt.get("epoch"))

    def propose(self, rt_ctx, work, beam, scope, skips) -> list[Candidate]:
        model = self._adapter()
        sample = model_sample(work)
        steps = model.decode(sample, work.rt, int(beam), state=work.state, scope=scope, skips=skips)
        return candidates(work.rt, steps, skips)


def model_sample(work):
    """The pool's sample with its Lewis structure and Wiberg indices (the model's inputs)."""
    from ksmech.arrows.lewis import MissingBondIndex, reactant_lewis

    try:
        bonds, lone, wiberg = reactant_lewis(work.rt)
    except MissingBondIndex as exc:
        raise ServeError(Code.MISSING_BOND_INDEX, str(exc)) from exc
    return replace(work.sample, bonds=bonds, lone_pairs=lone, wiberg=wiberg)


def candidates(rt, steps: list[NetworkStep], skips: Counter) -> list[Candidate]:
    """`NetworkStep`s (global atoms) as `Candidate`s on pool atoms, best first."""
    out = []
    for step in steps:
        atoms = {a for i, j, _ in step.matrix.bonds for a in (i, j)}
        if not atoms | {i for i, _ in step.matrix.lone} <= rt.mol_atom.keys():
            skips["NETWORK_STEP_OUTSIDE_POOL"] += 1
            continue
        R = step.matrix.mapped(rt.mol_atom)
        anchor = step.anchor and tuple(sorted(rt.mol_atom[a] for a in step.anchor[:2]))
        anchor = (*anchor, step.anchor[2]) if anchor else _first_anchor(R)
        if anchor is None:
            skips["NETWORK_STEP_WITHOUT_ANCHOR"] += 1
            continue
        out.append(Candidate(R, anchor, float(step.logp), float(step.logp)))
    return sorted(out, key=lambda c: -c.score)


def _first_anchor(R) -> tuple[int, int, str] | None:
    anchors = R.anchors()
    return tuple(anchors[0]) if anchors else None


# ---- the adapter over a loaded model


class CheckpointNetwork:
    """`NetworkModel` over a loaded `KSMech` (eval mode, CPU)."""

    def __init__(self, model, key: str, epoch: int | None = None) -> None:
        self.model = model
        self.version = key if epoch is None else f"{key}@epoch{epoch}"

    def decode(self, sample, rt, beam, *, state, scope, skips) -> list[NetworkStep]:
        from ksmech.arrows.delta_be import ReactionMatrix
        from ksmech.model.decode import SIGN, DecodeConfig, decode_pool

        b = max(1, int(beam))
        cfg = DecodeConfig(anchors=b, per_anchor=b)
        try:
            found = decode_pool(self.model, sample, cfg, applier(state, rt), scope, skips)
        except ServeError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported with its code
            log.exception("network decode of %s failed", state.state_id)
            raise ServeError(Code.NETWORK_DECODE_FAILED, f"{type(exc).__name__}: {exc}") from exc
        return [
            NetworkStep(ReactionMatrix(c.bonds, c.lone), c.logp, (*c.anchor[:2], SIGN[c.anchor[2]]))
            for c in found
        ]


def charge_issue(reactant, product, atoms) -> str | None:
    """FORMAL_CHARGE_RANGE when the step leaves a non-metal it touches at |formal charge| >= 2
    that it did not carry before ([Cl-2], [O-2], [C+2]: RDKit's valence table and the octet
    check on H-Ne let them through; no PMechDB step creates one), else None."""
    for i in atoms:
        new = product.GetAtomWithIdx(int(i))
        if new.GetAtomicNum() not in NONMETALS or abs(new.GetFormalCharge()) < 2:
            continue
        if abs(reactant.GetAtomWithIdx(int(i)).GetFormalCharge()) < 2:
            return "FORMAL_CHARGE_RANGE"
    return None


def applier(state, rt):
    """B + R of a completion on global atoms, checked on the pool state (`stepapply`, then
    `charge_issue`): (pool key, None) or (None, the first issue code)."""
    from ksmech.arrows.delta_be import ReactionMatrix
    from ksmech.serve.science.stepapply import apply_matrix, matrix_atoms

    def apply(bonds, lone):
        R = ReactionMatrix(bonds, lone).mapped(rt.mol_atom)
        res = apply_matrix(state, R)
        if not res.ok:
            return None, res.issues[0].code
        why = charge_issue(rt.mol, res.product, matrix_atoms(R))
        return (None, why) if why else (res.state.pool_key, None)

    return apply
