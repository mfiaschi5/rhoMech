"""Embedding traces in serve: one pool state through a local spec-v9 checkpoint
(`ksmech.eval.embedding.trace`), drawn as the self-contained page of `eval.embedding_page`.

- `POST /api/embedding {state_id | smiles, run, file?, source?}` (`submit`) resolves the pool
  state (a SMILES becomes a saved root state, as `POST /api/pools` builds it) and the checkpoint
  (`serve.predict.models`), then queues an `embedding` job.
- The job (`embedding_handler`, a runner thread of the API process) featurizes the fragments the
  token loader lacks with the `featurize` task (GFN2-xTB, or the synthetic chemistry, into
  `data/serve/`; the shared caches are read, never written), hashes the checkpoint, and looks the
  trace up by its id. A miss runs the `embedding` task in the compute worker.
- The task (`t_embedding`) loads the model (`train.loop.load_model` on the CPU, kept per
  (path, mtime, size) in the worker; a dirty checkpoint is CHECKPOINT_REFUSED), builds the
  sample of the state's tokens with its Lewis structure and Wiberg indices, traces it and writes
  the page, the payload and the meta under `data/serve/embedding/` (gitignored). Torch is
  imported only there: the API process never loads it.

The id is "b" + sha256(version, checkpoint sha256, state id, token digest)[:24]. The state id
(sha of the mapped SMILES) fixes the pool key and the drawing whose atoms the page numbers;
the token digest changes when a fragment's tokens are rebuilt. Files: `<id>.html` (the page),
`<id>.json` (the payload, NaN as null) and `<id>.meta.json` (`EmbeddingResultOut` fields),
written in that order; the meta marks a complete trace.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ksmech.serve.chem.pools import atomic_write_text
from ksmech.serve.compute.worker import Priority
from ksmech.serve.core.contracts import LRU, PoolState, to_jsonable
from ksmech.serve.core.errors import Code, ServeError

if TYPE_CHECKING:
    from ksmech.serve.core.contracts import JobContext, Runtime
    from ksmech.serve.core.schemas import EmbeddingIn, JobCreatedOut
    from ksmech.serve.core.settings import ServeSettings
    from ksmech.serve.predict.jobs import JobRunner
    from ksmech.serve.predict.models import Checkpoint
    from ksmech.serve.runtime.context import ApiContext

log = logging.getLogger(__name__)

EMBED_VERSION = 1
TRACE_TIMEOUT_S = 900.0
FEATURIZE_SLACK_S = 120
THEMES = ("light", "dark")
_MODELS: LRU[tuple[str, int, int], tuple[Any, dict]] = LRU(1)  # worker: one model in memory


# ---- ids and files


def embedding_id(checkpoint_sha: str, state_id: str, digest: str) -> str:
    raw = f"embedding-v{EMBED_VERSION}|{checkpoint_sha}|{state_id}|{digest}"
    return "b" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def files(settings: ServeSettings, eid: str) -> dict[str, Path]:
    base = settings.embedding
    return {
        "page": base / f"{eid}.html",
        "payload": base / f"{eid}.json",
        "meta": base / f"{eid}.meta.json",
    }


def read_meta(settings: ServeSettings, eid: str) -> dict | None:
    """The meta of a complete trace (its page and payload present), None otherwise."""
    f = files(settings, eid)
    if not (f["page"].is_file() and f["payload"].is_file()):
        return None
    try:
        return json.loads(f["meta"].read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("unreadable embedding meta %s [EMBEDDING_META_UNREADABLE]: %s", eid, exc)
        return None


def _stored(settings: ServeSettings, eid: str, kind: str) -> Path:
    path = files(settings, eid)[kind]
    if read_meta(settings, eid) is None:
        raise ServeError(Code.EMBEDDING_NOT_FOUND, f"no stored embedding trace {eid}", status=404)
    return path


def payload_bytes(settings: ServeSettings, eid: str) -> bytes:
    """`GET /api/embedding/{id}`: the stored payload (JSON)."""
    return _stored(settings, eid, "payload").read_bytes()


def page_html(settings: ServeSettings, eid: str, theme: str | None = None) -> str:
    """`GET /api/embedding/{id}/page`: the self-contained page; `theme` ("light" | "dark")
    sets `data-theme` on its root element, else the page follows prefers-color-scheme."""
    html = _stored(settings, eid, "page").read_text(encoding="utf-8")
    if theme is None:
        return html
    if theme not in THEMES:
        raise ServeError(Code.REQUEST_INVALID, f"theme must be one of {THEMES}", status=422)
    return html.replace("<html", f'<html data-theme="{theme}"', 1)


def result_out(meta: dict, cached: bool, seconds: float) -> dict:
    """`EmbeddingResultOut` fields of a stored meta."""
    return meta | {"cached": bool(cached), "seconds": float(seconds)}


# ---- API side: the request


def _state_of_smiles(ctx: ApiContext, smiles: str) -> PoolState:
    """The root state of a SMILES (one field, '.'-separated species), saved; the first
    preflight error is a 422 with its code."""
    from ksmech.serve.chem.pools import SpeciesInput, preflight_inputs, save_state

    pf = preflight_inputs([SpeciesInput(smiles, 1)], ctx.featurizer, ctx.settings.limits)
    if pf.error is not None:
        err = pf.error
        raise ServeError(
            err.code, err.message, status=422, upstream=err.upstream, detail=err.detail
        )
    if pf.state is None:
        raise ServeError(Code.CONSISTENCY_FAILED, "preflight gave no state", status=500)
    save_state(pf.state, ctx.settings)
    return pf.state


def resolve_state(ctx: ApiContext, body: EmbeddingIn) -> PoolState:
    """The pool state of the request: exactly one of `state_id` and `smiles`."""
    if (body.state_id is None) == (body.smiles is None or not body.smiles.strip()):
        msg = "give exactly one of state_id and smiles"
        raise ServeError(Code.REQUEST_INVALID, msg, status=422)
    if body.state_id is not None:
        return ctx.state(body.state_id)
    return _state_of_smiles(ctx, str(body.smiles).strip())


def usable_checkpoint(settings: ServeSettings, run: str, file: str, source: str | None):
    """`models.find_checkpoint`, refused (CHECKPOINT_REFUSED, 409) when its run is dirty."""
    from ksmech.serve.predict.models import find_checkpoint

    ck = find_checkpoint(settings, run, file, source)
    if not ck.usable:
        msg = f"checkpoint {ck.source}/{ck.run}/{ck.file}.pt is refused: {ck.reason}"
        raise ServeError(Code.CHECKPOINT_REFUSED, msg, status=409, detail={"reason": ck.reason})
    return ck


def submit(ctx: ApiContext, runner: JobRunner, body: EmbeddingIn) -> JobCreatedOut:
    """`POST /api/embedding`: resolve the state and the checkpoint (errors are immediate), then
    queue the `embedding` job."""
    from ksmech.serve.core.schemas import JobCreatedOut
    from ksmech.serve.predict.jobs import JobKind

    state = resolve_state(ctx, body)
    ck = usable_checkpoint(ctx.settings, body.run, body.file, body.source)
    request = {"state": state.to_dict(), "source": ck.source, "run": ck.run, "file": ck.file}
    label = f"embedding {ck.run}/{ck.file}: {state.pool_key}"
    job = runner.submit(JobKind.EMBEDDING, request, label=label)
    return JobCreatedOut(**job.created_out())


# ---- API side: the job


def _featurize(ctx: JobContext, api: ApiContext, state: PoolState) -> None:
    """One `featurize` task per fragment key the loader lacks (JOB priority)."""
    from ksmech.serve.chem.featurize import featurize_label

    for key in dict.fromkeys(state.fragment_token_keys):
        if api.loader.has(key):
            continue
        if ctx.cancelled():
            raise ServeError(Code.CANCELLED, f"cancelled before featurizing {key}")
        label = featurize_label(api.featurizer, key)
        ctx.progress(phase="featurize", message=label)
        r = ctx.scheduler.run(
            "featurize",
            {"key": key},
            priority=Priority.JOB,
            timeout_s=ctx.settings.qm_timeout_s + FEATURIZE_SLACK_S,
            owner=ctx.job_id,
            label=label,
        )
        if not r.ok:
            raise ServeError.from_task(r)


def _token_digest(api: ApiContext, state: PoolState) -> str:
    from ksmech.serve.science.analysis import analysis_digest

    digest = analysis_digest(api.loader, state)
    if digest is None:
        missing = sorted({k for k in state.fragment_token_keys if not api.loader.has(k)})
        msg = f"no tokens for {missing} after featurization"
        raise ServeError(Code.MISSING_TOKENS, msg, detail={"keys": missing})
    return digest


def _task_payload(state: PoolState, ck: Checkpoint, sha: str, digest: str, eid: str) -> dict:
    return {
        "state": state.to_dict(),
        "path": str(ck.path),
        "signature": list(ck.signature),
        "sha256": sha,
        "digest": digest,
        "embedding_id": eid,
        "source": ck.source,
        "run": ck.run,
        "file": ck.file,
    }


def run_embedding(ctx: JobContext, request: dict, *, api: ApiContext) -> dict:
    """The `embedding` job (module docstring). Returns the `EmbeddingResultOut` fields."""
    from ksmech.serve.predict.models import checkpoint_sha256

    state = PoolState.from_dict(request["state"])
    ck = usable_checkpoint(ctx.settings, request["run"], request["file"], request["source"])
    _featurize(ctx, api, state)
    digest = _token_digest(api, state)
    ctx.progress(phase="hash", message=f"sha256 of {ck.run}/{ck.file}.pt")
    sha = checkpoint_sha256(ck)
    eid = embedding_id(sha, state.state_id, digest)
    meta = read_meta(ctx.settings, eid)
    if meta is not None:
        return result_out(meta, True, 0.0)
    label = f"embedding trace of {state.pool_key} through {ck.run}/{ck.file}"
    ctx.progress(phase="trace", message=label)
    r = ctx.scheduler.run(
        "embedding",
        _task_payload(state, ck, sha, digest, eid),
        priority=Priority.JOB,
        timeout_s=TRACE_TIMEOUT_S,
        owner=ctx.job_id,
        label=label,
    )
    if not r.ok:
        raise ServeError.from_task(r)
    return result_out(r.result or {}, False, r.seconds)


def embedding_handler(api: ApiContext):
    """The runner handler of `embedding` jobs."""

    def handle(ctx: JobContext, request: dict) -> dict:
        return run_embedding(ctx, request, api=api)

    return handle


# ---- worker side


def _signature(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return int(st.st_mtime_ns), int(st.st_size)


def _load(path: Path):
    """`train.loop.load_model` on the CPU, its refusals as reason codes."""
    from ksmech.data.provenance import StaleArtifact
    from ksmech.train.loop import load_model

    try:
        return load_model(path, "cpu")
    except StaleArtifact as exc:
        raise ServeError(Code.CHECKPOINT_REFUSED, f"load_model refused {path}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - every load failure is a reason code
        detail = f"{type(exc).__name__}: {exc}"
        raise ServeError(Code.CHECKPOINT_LOAD_FAILED, f"cannot load {path}: {detail}") from exc


def cached_model(path: Path, signature: tuple[int, int]):
    """(model, checkpoint dict) of `path` at the file version `signature`, kept in memory per
    (path, mtime, size); CHECKPOINT_CHANGED when the file is not that version."""
    key = (str(path), *signature)
    hit = _MODELS.get(key)
    if hit is not None:
        return hit
    if _signature(path) != tuple(signature):
        raise ServeError(Code.CHECKPOINT_CHANGED, f"{path} changed since it was hashed; retry")
    model, ckpt = _load(path)
    if _signature(path) != tuple(signature):
        raise ServeError(Code.CHECKPOINT_CHANGED, f"{path} changed while it was loaded; retry")
    _MODELS.put(key, (model.eval(), ckpt))
    return model, ckpt


def _pool_sample(rt: Runtime, state: PoolState, digest: str):
    """(sample with its Lewis structure, orbital types or None) of the state's tokens, which
    must still be the ones of `digest` (else TOKENS_CHANGED)."""
    from ksmech.arrows.lewis import MissingBondIndex, reactant_lewis
    from ksmech.arrows.steps import reactant_sample
    from ksmech.serve.science.analysis import (
        _checked_rt,
        _worker_error,
        analysis_digest,
        tokens_changed,
    )

    try:
        rtok = _checked_rt(rt, state)
        sample = reactant_sample(f"serve:{state.state_id}", "serve", rtok)
        sample.bonds, sample.lone_pairs, sample.wiberg = reactant_lewis(rtok)
    except MissingBondIndex as exc:
        raise ServeError(Code.MISSING_BOND_INDEX, str(exc)) from exc
    except ServeError:
        raise
    except Exception as exc:  # noqa: BLE001 - MissingTokens, CorrespondenceError, ...
        raise _worker_error(exc) from exc
    after = analysis_digest(rt.loader, state)
    if after != digest:
        raise tokens_changed(state, digest, after)
    return sample, _orbital_types(rt, sample)


def _orbital_types(rt: Runtime, sample) -> list[str] | None:
    """The diagnostic orbital types (page labels only, never a model input); None, as an event,
    when a token file does not match the sample."""
    from ksmech.eval.embedding_page import orbital_types

    tokens = sample.tokens(rt.loader)
    types = None if any(t is None for t in tokens) else orbital_types(tokens)
    if types is None or len(types) != sample.n_orbitals:
        rt.event("ORBITAL_TYPES_UNAVAILABLE", f"{sample.step_id}: orbitals named by occupation")
        return None
    return types


def _checkpoint_text(file: str, ckpt: dict) -> str:
    text = f"{file}.pt, epoch {ckpt.get('epoch', '?')}"
    top1 = (ckpt.get("metrics") or {}).get("val_top1")
    return text if top1 is None else f"{text}, val top-1 {top1:.1%} when trained"


def _page_meta(payload: dict, state: PoolState, model, ckpt: dict) -> dict:
    return {
        "sample": f"pool {state.pool_key} (serve state {state.state_id})",
        "reaction": state.mapped_smiles,
        "run": f"{payload['source']}/{payload['run']}",
        "checkpoint": _checkpoint_text(payload["file"], ckpt),
        "encoder": model.cfg.encoder,
        "heads": model.cfg.heads,
    }


def _meta(payload: dict, state: PoolState, model, ckpt: dict, tr: dict, sample) -> dict:
    """The `EmbeddingResultOut` fields of a new trace, without cached and seconds."""
    eid = payload["embedding_id"]
    errors = [tr["errors"]["embedding"], *tr["errors"]["layers"]]
    top = tr["readout"]["top"]
    return {
        "embedding_id": eid,
        "state_id": state.state_id,
        "pool_key": state.pool_key,
        "source": payload["source"],
        "run": payload["run"],
        "file": payload["file"],
        "checkpoint_sha256": payload["sha256"],
        "encoder": str(model.cfg.encoder),
        "epoch": ckpt.get("epoch"),
        "n_atoms": int(sample.n_atoms),
        "n_orbitals": int(sample.n_orbitals),
        "n_layers": int(model.cfg.layers),
        "heads": int(model.cfg.heads),
        "max_error": float(max(errors)),
        "top_anchor": to_jsonable(top[0]) if top else None,
        "page_url": f"/api/embedding/{eid}/page",
        "payload_url": f"/api/embedding/{eid}",
        "created_at": datetime.now(UTC).isoformat(),
    }


def _write(rt: Runtime, eid: str, page: dict, meta: dict) -> dict:
    """The page, the payload and the meta (in that order, each atomic); the meta with the
    page's size."""
    from ksmech.eval.embedding_page import write_page
    from ksmech.serve.chem.pools import tmp_sibling
    from ksmech.serve.core.settings import assert_writable

    s = rt.settings
    f = files(s, eid)
    tmp = tmp_sibling(f["page"])
    assert_writable(tmp, s)
    try:
        write_page(page, tmp)
        tmp.replace(f["page"])
    finally:
        tmp.unlink(missing_ok=True)
    data = json.dumps(to_jsonable(page), separators=(",", ":"), allow_nan=False)
    atomic_write_text(f["payload"], data, s)
    meta = meta | {"page_bytes": f["page"].stat().st_size}
    atomic_write_text(f["meta"], json.dumps(to_jsonable(meta), sort_keys=True, indent=1), s)
    return meta


def t_embedding(rt: Runtime, payload: dict) -> dict:
    """{state, path, signature, sha256, digest, embedding_id, source, run, file} -> the meta
    of the stored trace (`EmbeddingResultOut` fields without cached and seconds)."""
    from ksmech.eval.embedding import trace
    from ksmech.eval.embedding_page import page_payload

    t0 = time.perf_counter()
    state = PoolState.from_dict(payload["state"])
    model, ckpt = cached_model(Path(payload["path"]), tuple(payload["signature"]))
    sample, types = _pool_sample(rt, state, str(payload["digest"]))
    try:
        tr = trace(model, sample)
        page = page_payload(tr, sample, types, _page_meta(payload, state, model, ckpt))
    except Exception as exc:  # noqa: BLE001 - reported with its code
        log.exception("embedding trace of %s failed", state.state_id)
        raise ServeError(Code.EMBEDDING_FAILED, f"{type(exc).__name__}: {exc}") from exc
    meta = _write(rt, payload["embedding_id"], page, _meta(payload, state, model, ckpt, tr, sample))
    log.info("embedding %s: %s in %.1f s", meta["embedding_id"], state.pool_key,
             time.perf_counter() - t0)  # fmt: skip
    return meta
