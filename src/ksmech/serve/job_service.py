"""The API-side job service (design 6.4): predict, job status, cancel, override, pathways, and
the `NetworkSource` curation reads networks through. Services never import FastAPI; they take
an `ApiContext` and return `schemas` models.

`network_out(net, request, provenance)` builds a fully typed `NetworkOut`: species display
SMILES come from the token keys (a species key may be a charge-separated resonance form), and
the dataset arrows, PMechDB code and encoding base of an edge come from its `encoded` step.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from ksmech.serve.contracts import LRU
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.jobs import (
    Job,
    JobKind,
    JobRunner,
    JobStatus,
    override_handler,
    predict_handler,
    species_inputs,
)
from ksmech.serve.network import Network, display_smiles, network_from_dict, rank_pathways
from ksmech.serve.schemas import (
    ArrowOut,
    ExportReportOut,
    IndexStatusOut,
    JobCreatedOut,
    JobOut,
    JobSummaryOut,
    NetworkOut,
    OverrideIn,
    OverrideResultOut,
    PathwayOut,
    PredictIn,
    RecheckReportOut,
)

if TYPE_CHECKING:
    from ksmech.serve.chemistry import Chemistry
    from ksmech.serve.context import ApiContext

log = logging.getLogger(__name__)

CHEMISTRIES = ("real", "cached", "synthetic", "stub")
RESULT_MODELS = {
    JobKind.EXPORT: ExportReportOut,
    JobKind.RECHECK: RecheckReportOut,
    JobKind.INDEX: IndexStatusOut,
}
DATASET_ARROW_KEYS = ("source_kind", "source_atoms", "sink_kind", "sink_atoms")


# ---- NetworkOut


def _error_body(failure: Mapping | None) -> dict | None:
    if not failure:
        return None
    code = str(failure.get("code") or Code.UNEXPECTED)
    return {
        "code": code,
        "message": str(failure.get("detail") or code),
        "upstream": failure.get("upstream"),
        "detail": None,
        "request_id": "",
    }


def _species_out(sp) -> dict:
    return {
        "species_key": sp.species_key,
        "token_keys": list(sp.token_keys),
        "display_smiles": sp.token_keys[0] if sp.token_keys else sp.species_key,
        "count": sp.count,
        "charge": sp.charge,
        "resonance_flag": sp.resonance_flag,
    }


def _node_out(n) -> dict:
    return {
        "id": n.id,
        "pool_key": n.pool_key,
        "state_id": n.state.state_id,
        "alias_state_ids": list(n.alias_state_ids),
        "species": [_species_out(sp) for sp in n.state.species],
        "display_smiles": display_smiles(n.state),
        "depth": n.depth,
        "best_logp": n.best_logp,
        "best_parent_edge": n.best_parent_edge,
        "status": n.status.value,
        "terminal_reason": n.terminal_reason,
        "failure": _error_body(n.failure),
        "is_root": n.is_root,
        "n_valid": n.n_valid,
        "scanned_mass": n.scanned_mass,
        "unscanned_mass": n.unscanned_mass,
        "apply_failures": dict(n.apply_failures),
        "fermi_smeared": n.fermi_smeared,
        "warnings": list(n.warnings),
    }


def _dataset_arrows(encoded: Mapping) -> list[dict] | None:
    arrows = encoded.get("arrows")
    if not isinstance(arrows, list):
        return None
    out = []
    for a in arrows:
        if not isinstance(a, Mapping) or any(k not in a for k in DATASET_ARROW_KEYS):
            log.warning("encoded arrow without the dataset-arrow fields: %r", a)
            return None
        out.append({k: a[k] for k in DATASET_ARROW_KEYS})
    return out


def _failed_proposal_out(fp: Mapping) -> dict:
    """`FailedProposalOut` fields. An arrow dict not in `ArrowOut` form (a worker fallback)
    is omitted, logged, and noted in the detail, so one bad example never hides the network."""
    arrows, bad = [], 0
    for a in fp.get("arrows") or []:
        try:
            ArrowOut.model_validate(a)
            arrows.append(a)
        except ValidationError:
            bad += 1
    detail = str(fp.get("detail") or "")
    if bad:
        log.warning("failed proposal at %s: %d arrow(s) not in ArrowOut form", fp.get("node"), bad)
        detail += f" ({bad} arrow(s) not in ArrowOut form omitted)"
    return {
        "node": fp.get("node"),
        "rank": fp.get("rank"),
        "arrows": arrows,
        "probability": fp.get("probability"),
        "code": str(fp.get("code") or Code.UNEXPECTED),
        "detail": detail,
    }


def _edge_out(e, net: Network) -> dict:
    enc = e.encoded or {}
    d = {k: getattr(e, k) for k in e.__dataclass_fields__ if k != "encoded"}
    d.update(
        target_pool_key=net.nodes[e.target].pool_key,
        dataset_arrows=_dataset_arrows(enc) if enc else None,
        pmechdb_code=enc.get("code"),
        encoding_base_smiles=enc.get("base_smiles"),
    )
    return d


def _params_out(net: Network) -> dict:
    p = net.params
    t = p.temperature if p.temperature is not None else net.predictor.temperature
    return {
        "beam": p.beam,
        "max_steps": p.max_steps,
        "temperature": t,
        "stop_at_turnover": p.stop_at_turnover,
        "candidate_scope": p.candidate_scope,
        "max_scan": p.max_scan,
        "max_nodes": p.max_nodes,
        "n_pathways": p.n_pathways,
    }


def network_provenance(net: Network, provenance: Mapping) -> dict:
    """`provenance` with the predictor fields of the network's own predictor reference and
    the chemistry it was built with."""
    out = dict(provenance)
    out.update(
        predictor=net.predictor.name,
        predictor_version=net.predictor.version,
        checkpoint_hash=net.predictor.checkpoint_hash,
        temperature=net.predictor.temperature,
        candidate_scope=net.params.candidate_scope,
    )
    if net.chemistry in CHEMISTRIES:
        out["chemistry"] = net.chemistry
    return out


def network_out(net: Network, request: Mapping, provenance: Mapping) -> NetworkOut:
    """The typed `NetworkOut` of a network (raises pydantic's ValidationError)."""
    return NetworkOut.model_validate(
        {
            "network_id": net.network_id,
            "revision": net.revision,
            "complete": net.complete,
            "root": net.root,
            "nodes": [_node_out(n) for n in net.nodes.values()],
            "edges": [_edge_out(e, net) for e in net.edges.values()],
            "cycles": [c.to_dict() for c in net.cycles.values()],
            "pathways": [p.to_dict() for p in net.pathways],
            "failed_proposals": [_failed_proposal_out(fp) for fp in net.failed_proposals],
            "skipped": {k: int(v) for k, v in sorted(net.skipped.items())},
            "params": _params_out(net),
            "request": dict(request),
            "provenance": network_provenance(net, provenance),
        }
    )


# ---- the service


class JobService:
    """Jobs over HTTP (design 6.4). The constructor registers the predict and override
    handlers on the runner."""

    def __init__(self, ctx: ApiContext, runner: JobRunner, chemistry: Chemistry) -> None:
        self.ctx = ctx
        self.runner = runner
        self.chemistry = chemistry
        runner.register(JobKind.PREDICT, predict_handler(chemistry, ctx.registry.ref))
        runner.register(JobKind.OVERRIDE, override_handler(chemistry, runner))
        self._nets: LRU[str, tuple[tuple, Network]] = LRU(8)

    # ---- endpoints

    def predict(self, body: PredictIn) -> JobCreatedOut:
        """Preflight (input and predictor errors are immediate 422 / 409), then queue."""
        request = body.model_dump(mode="json")
        inputs = species_inputs(request)
        self.ctx.registry.get(body.predictor)
        self.chemistry.root(inputs)
        prov = self._provenance(body.predictor, body.temperature, body.candidate_scope)
        label = f"{body.predictor}: " + " + ".join(
            s if n == 1 else f"{n} {s}" for s, n in ((i.smiles, i.count) for i in inputs)
        )
        job = self.runner.submit(JobKind.PREDICT, request, label=label, provenance=prov)
        return JobCreatedOut(**job.created_out())

    def get(self, job_id: str, *, since_revision: int | None = None) -> JobOut:
        job = self.runner.get(job_id)
        out = self._job_base(job)
        if job.kind == JobKind.PREDICT:
            self._predict_result(job, out, since_revision)
        elif job.kind == JobKind.OVERRIDE:
            out["provenance"] = self._owner_provenance(job)
            self._stored_result(job, out, OverrideResultOut, "override")
        elif job.kind in RESULT_MODELS:
            self._stored_result(job, out, RESULT_MODELS[JobKind(job.kind)], job.kind)
        return JobOut.model_validate(out)

    def list(self, *, kind: str | None = None, status: str | None = None, limit: int = 50):
        return [
            JobSummaryOut(
                job_id=j.id,
                kind=j.kind,
                status=j.status,
                created_at=j.created_at,
                network_id=j.network_id,
                label=j.label,
            )
            for j in self.runner.list(kind=kind, status=status, limit=limit)
        ]

    def cancel(self, job_id: str, *, force: bool = False) -> JobOut:
        self.runner.cancel(job_id, force=force)
        return self.get(job_id)

    def override(self, job_id: str, body: OverrideIn) -> JobCreatedOut:
        """Queue an override of the job's network. Errors at submission: exactly one of
        edge_id / node_id (422), unknown job (404), no network yet (409 JOB_NOT_DONE),
        unknown edge or node (404), a job queued or running on the network (409)."""
        if (body.edge_id is None) == (body.node_id is None):
            msg = "give exactly one of edge_id and node_id"
            raise ServeError(Code.REQUEST_INVALID, msg, status=422)
        job = self.runner.get(job_id)
        owner, net = self._network_of_job(job)
        if body.edge_id is not None and body.edge_id not in net.edges:
            msg = f"no edge {body.edge_id} in the network of {owner.id}"
            raise ServeError(Code.EDGE_NOT_FOUND, msg, status=404)
        if body.node_id is not None and body.node_id not in net.nodes:
            msg = f"no node {body.node_id} in the network of {owner.id}"
            raise ServeError(Code.NODE_NOT_FOUND, msg, status=404)
        request = {
            "network_id": owner.id,
            "parent_job_id": job.id,
            "override": body.model_dump(mode="json"),
        }
        target = body.edge_id or body.node_id
        created = self.runner.submit(
            JobKind.OVERRIDE,
            request,
            parent_job_id=job.id,
            network_id=owner.id,
            label=f"override of {target}",
            exclusive=True,
            provenance=owner.provenance,
        )
        return JobCreatedOut(**created.created_out())

    def pathways(self, job_id: str, *, limit: int = 20, include_pruned: bool = False):
        _, net = self._network_of_job(self.runner.get(job_id))
        paths = rank_pathways(net, limit=int(limit), include_pruned=bool(include_pruned))
        return [PathwayOut.model_validate(p.to_dict()) for p in paths]

    def network_of(self, job_id: str) -> tuple[NetworkOut, dict]:
        """`contracts.NetworkSource`: (NetworkOut, the PredictIn dict) of a finished job's
        network. JOB_NOT_FOUND (404); JOB_NOT_DONE (409) unless the job and its network's
        predict job are done."""
        job = self.runner.get(job_id)
        owner, net = self._network_of_job(job)
        if job.status != JobStatus.DONE or owner.status != JobStatus.DONE:
            msg = f"job {job.id} is {job.status} (network job {owner.id}: {owner.status})"
            raise ServeError(Code.JOB_NOT_DONE, msg, status=409)
        return self._network_out(owner, net), dict(owner.request)

    def running(self) -> int:
        return self.runner.running()

    # ---- helpers

    def _provenance(self, predictor: str, temperature, scope: str) -> dict:
        prov = self.ctx.provenance(predictor, temperature=temperature, scope=scope)
        mode = getattr(self.chemistry, "mode", None)
        if mode in CHEMISTRIES:
            prov["chemistry"] = mode
        return prov

    def _owner(self, job: Job) -> Job:
        net_id = job.network_id
        owner = self.runner.get(net_id) if net_id else None
        if owner is None or owner.kind != JobKind.PREDICT:
            raise ServeError(Code.JOB_NOT_DONE, f"job {job.id} has no network", status=409)
        return owner

    def _owner_provenance(self, job: Job) -> dict | None:
        try:
            return self._owner(job).provenance
        except ServeError:
            return None

    def _network_of_job(self, job: Job) -> tuple[Job, Network]:
        owner = self._owner(job)
        net = self._network(owner.id)
        if net is None:
            msg = f"job {owner.id} ({owner.status}) has no network yet"
            raise ServeError(Code.JOB_NOT_DONE, msg, status=409)
        return owner, net

    def _network(self, network_id: str) -> Network | None:
        """The stored network, parsed once per file version (mtime, size)."""
        path = self.runner.store.network_path(network_id)
        try:
            st = path.stat()
        except FileNotFoundError:
            return None
        stamp = (st.st_mtime_ns, st.st_size)
        hit = self._nets.get(network_id)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        data = self.runner.store.read_network(network_id)
        if data is None:
            return None
        net = network_from_dict(data)
        self._nets.put(network_id, (stamp, net))
        return net

    def _network_out(self, owner: Job, net: Network) -> NetworkOut:
        prov = owner.provenance or self._provenance(
            net.predictor.name, net.predictor.temperature, net.params.candidate_scope
        )
        try:
            return network_out(net, owner.request, prov)
        except ValidationError as exc:
            log.error("network %s does not validate: %s", owner.id, exc)
            detail = {"errors": [f"{e.get('loc')}: {e.get('msg')}" for e in exc.errors()[:5]]}
            msg = f"the stored network of {owner.id} does not validate as NetworkOut"
            raise ServeError(Code.CONSISTENCY_FAILED, msg, status=500, detail=detail) from exc

    def _current_task(self, job: Job) -> dict | None:
        current = getattr(self.ctx.scheduler, "current", None)
        if callable(current):
            info = current()
            if info is None or (info.get("owners") and job.id not in info["owners"]):
                return None
            return {k: info[k] for k in ("name", "label", "elapsed_s", "queue_depth", "priority")}
        st = self.ctx.scheduler.status()
        if not st.busy or st.current_task is None:
            return None
        return {
            "name": st.current_task,
            "label": st.current_label or "",
            "elapsed_s": float(st.elapsed_s or 0.0),
            "queue_depth": st.queue_depth,
            "priority": "unknown",
        }

    def _job_base(self, job: Job) -> dict[str, Any]:
        progress = dict(job.progress)
        progress.setdefault("phase", job.status)
        progress.setdefault("updated_at", job.finished_at or job.started_at or job.created_at)
        progress["current_task"] = (
            self._current_task(job) if job.status == JobStatus.RUNNING else None
        )
        return {
            "job_id": job.id,
            "kind": job.kind,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "progress": progress,
            "error": job.error,
            "parent_job_id": job.parent_job_id,
            "network_id": job.network_id,
            "partial": False,
            "unchanged": False,
            "provenance": None,
            "result_kind": None,
            "result": None,
        }

    def _predict_result(self, job: Job, out: dict, since_revision: int | None) -> None:
        out["provenance"] = job.provenance
        net = self._network(job.network_id or job.id)
        if net is None:
            return
        out["result_kind"] = "network"
        out["partial"] = job.status != JobStatus.DONE
        if since_revision is not None and int(since_revision) == net.revision:
            out["unchanged"] = True
            return
        out["result"] = self._network_out(job, net)
        if out["provenance"] is None:
            out["provenance"] = out["result"].provenance

    def _stored_result(self, job: Job, out: dict, model, kind: str) -> None:
        data = self.runner.store.read_json(job.id, "result.json")
        if data is None:
            return
        try:
            out["result"] = model.model_validate(data)
            out["result_kind"] = kind
        except ValidationError as exc:
            log.error("job %s: result.json does not validate as %s: %s", job.id, model, exc)
