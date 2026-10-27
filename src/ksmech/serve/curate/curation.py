"""Curation: mechanisms and their steps from a network, status and order-group rules, staleness.

A mechanism is written once, from a pathway (or an explicit chain of edges) of a finished
network, and is self-contained afterwards: every `CuratedStep.arrows_json` holds a StepDoc
(format `ksmech-serve/step-v2`, built from the `EdgeOut` alone: the reaction matrix R as
`ReactionMatrix.to_list()` and as `MatrixIO`, its anchor, its arrow reading (pool atoms and,
as `dataset_arrows`, PMechDB-style map numbers), the orbital readings, the product SMILES and
sigma keys), and the mechanism stores the predictor, its version string (`checkpoint_hash`)
and the spec version, never rewritten. A user step (predictor "user") is a corrected step.

Statuses (design 9.3): `predicted` (stored, no verdict), `validated` (every step accepted as
predicted), `corrected` (accepted with at least one user-corrected step), `rejected` (kept as a
negative record, never exported). POST actions map to a status (`save`, `validate`, `reject`);
PATCH moves along `TRANSITIONS` (409 INVALID_TRANSITION otherwise) under the consistency rule
(422 STATUS_INCONSISTENT), and every status change appends an audit line to `notes`.

Order groups: one int per step, non-decreasing from 0 in steps of at most 1. Steps sharing a
group claim to be independent; when their maps are continuous (one step's product state is the
next step's reactant state) their touched maps must be disjoint (422 NOT_INDEPENDENT), and
across a merge the claim is accepted with the warning INDEPENDENCE_UNCHECKED.

Staleness: a mechanism is stale when its predictor version, the spec version or any step's
physics fingerprint differs from the current one (design 9.4).

API side: imports only the standard library, pydantic/sqlmodel and other API serve modules.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel
from sqlmodel import Session, col, select

from ksmech.serve.core.db import CuratedStep, Mechanism, Recheck, utcnow
from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.core.schemas import (
    CuratedStepOut,
    CurateIn,
    EdgeOut,
    MechanismOut,
    MechanismSummaryOut,
    NetworkOut,
    Provenance,
    RecheckRowOut,
    finite_or_none,
)

if TYPE_CHECKING:
    from ksmech.serve.predict.predictors import PredictorRegistry

log = logging.getLogger(__name__)

STEP_FORMAT = "ksmech-serve/step-v2"  # v2: reaction matrices (spec v9)
STATUSES = ("predicted", "validated", "corrected", "rejected")
TRANSITIONS = frozenset(
    {
        ("predicted", "validated"),
        ("predicted", "corrected"),
        ("predicted", "rejected"),
        ("validated", "rejected"),
        ("corrected", "rejected"),
        ("rejected", "predicted"),  # reopen
    }
)
PREDICTOR_UNAVAILABLE_REASON = "predictor_unavailable"

M = TypeVar("M", bound=BaseModel)


def as_model(cls: type[M], value: Any) -> M:
    """`value` as a `cls` instance (a model passes through; a dict is validated)."""
    return value if isinstance(value, cls) else cls.model_validate(value)


def dumps(doc: dict) -> str:
    """A StepDoc as JSON text (NaN and inf as null)."""
    return json.dumps(finite_or_none(doc), allow_nan=False)


# ---- StepDoc


def dataset_arrows(arrows: Sequence[dict], source_mapped: str) -> list[dict]:
    """The arrow reading as dataset arrows on map numbers (`data.schema.Arrow` fields):
    lone_pair_to_bond [a] -> atom b, bond_to_bond [a, c] -> bond [a, b], bond_to_lone_pair
    [a, c] -> atom c."""
    from ksmech.data import chem

    mol = chem.parse_mapped(source_mapped)
    m = lambda a: int(mol.GetAtomWithIdx(int(a)).GetAtomMapNum())  # noqa: E731
    out = []
    for a in arrows:
        src, dst = [m(x) for x in a["source"]], [m(x) for x in a["target"]]
        if a["kind"] == "lone_pair_to_bond":
            out.append({"source_kind": "lone_pair", "source_atoms": src, "sink_kind": "atom",
                        "sink_atoms": [dst[1]]})  # fmt: skip
        else:
            kind = "bond" if a["kind"] == "bond_to_bond" else "atom"
            out.append({"source_kind": "bond", "source_atoms": src, "sink_kind": kind,
                        "sink_atoms": dst})  # fmt: skip
    return out


def step_doc(edge: EdgeOut | dict, *, job_id: str, provenance: Provenance | dict) -> dict:
    """The StepDoc of one network edge, from the `EdgeOut` alone."""
    e, p = as_model(EdgeOut, edge), as_model(Provenance, provenance)
    d = e.model_dump(mode="json")
    matrix = d["matrix"]
    return {
        "format": STEP_FORMAT,
        "job_id": job_id,
        "edge_id": e.id,
        "edge_kind": e.kind,
        "reactant_state_id": e.source_state_id,
        "product_state_id": e.product_state_id,
        "reactants_mapped": e.source_mapped_smiles,
        "products_mapped": e.product_mapped_smiles,
        "matrix": matrix,
        "reaction_matrix": [matrix["bonds"], matrix["lone"]],  # ReactionMatrix.to_list()
        "anchor": d["anchor"],
        "arrows": d["arrows"],
        "readings": d["readings"],
        "dataset_arrows": dataset_arrows(d["arrows"], e.source_mapped_smiles),
        "pmechdb_code": e.pmechdb_code,
        "product_smiles": list(e.product_smiles),
        "sigma_keys": list(e.sigma_keys),
        "score": e.score,
        "logp": e.logp,
        "rank": e.rank,
        "predictor": e.predictor,
        "predictor_version": e.predictor_version,
        "spec_version": p.spec_version,
        "physics_fingerprint": p.physics_fingerprint,
        "touched_maps": sorted(int(m) for m in e.touched_maps),
    }


def original_doc(edge: EdgeOut, **_: Any) -> dict | None:
    """Spec v9 has no superseded predictions: a user step is added next to the predicted
    edges, never in place of one, so no curated step stores an original."""
    return None


# ---- chains, statuses, order groups


def chain_edges(network: NetworkOut, req: CurateIn) -> list[EdgeOut]:
    """The pathway's edges in order, or the given edge ids, checked to form a chain."""
    if (req.pathway_id is None) == (req.edge_ids is None):
        raise ServeError(Code.REQUEST_INVALID, "give exactly one of pathway_id and edge_ids")
    if req.pathway_id is not None:
        pathway = next((p for p in network.pathways if p.id == req.pathway_id), None)
        if pathway is None:
            msg = f"no pathway {req.pathway_id} in network {network.network_id}"
            raise ServeError(Code.PATHWAY_NOT_FOUND, msg, status=404)
        ids = list(pathway.edge_ids)
    else:
        ids = list(req.edge_ids or [])
    if not ids:
        raise ServeError(Code.REQUEST_INVALID, "a mechanism needs at least one step")
    by_id = {e.id: e for e in network.edges}
    missing = [i for i in ids if i not in by_id]
    if missing:
        msg = f"no edge {missing[0]} in network {network.network_id}"
        raise ServeError(Code.EDGE_NOT_FOUND, msg, status=404, detail={"edge_ids": missing})
    edges = [by_id[i] for i in ids]
    old = [e.id for e in edges if e.status == "superseded"]
    if old:
        msg = f"edge {old[0]} was superseded by a correction; curate the corrected edge"
        raise ServeError(Code.REQUEST_INVALID, msg, detail={"superseded": old})
    for a, b in zip(edges, edges[1:], strict=False):
        if a.target != b.source:
            msg = (
                f"edges do not form a chain: {a.id} ends at {a.target}, {b.id} starts at {b.source}"
            )
            raise ServeError(Code.REQUEST_INVALID, msg, detail={"edge_ids": [a.id, b.id]})
    return edges


def status_for_action(action: str, any_corrected: bool) -> str:
    """save -> predicted; validate -> validated, or corrected with a corrected step;
    reject -> rejected."""
    if action == "save":
        return "predicted"
    if action == "validate":
        return "corrected" if any_corrected else "validated"
    if action == "reject":
        return "rejected"
    raise ServeError(Code.REQUEST_INVALID, f"unknown action {action!r}")


def check_transition(old: str, new: str, any_corrected: bool) -> bool:
    """True when `old -> new` is a change the table allows; False for a no-op (same status).
    Raises INVALID_TRANSITION (409) or STATUS_INCONSISTENT (422)."""
    if new not in STATUSES:
        raise ServeError(Code.REQUEST_INVALID, f"unknown status {new!r}; one of {STATUSES}")
    if old == new:
        return False
    if (old, new) not in TRANSITIONS:
        allowed = sorted(b for a, b in TRANSITIONS if a == old)
        detail = {"from": old, "to": new, "allowed": allowed}
        msg = f"status {old} -> {new} is not allowed"
        raise ServeError(Code.INVALID_TRANSITION, msg, status=409, detail=detail)
    if new == "validated" and any_corrected:
        msg = "a mechanism with a corrected step is 'corrected', not 'validated'"
        raise ServeError(Code.STATUS_INCONSISTENT, msg, status=422)
    if new == "corrected" and not any_corrected:
        msg = "'corrected' needs at least one corrected step; use 'validated'"
        raise ServeError(Code.STATUS_INCONSISTENT, msg, status=422)
    return True


def check_order_groups(groups: Sequence[int] | None, n_steps: int) -> list[int]:
    """The order groups (default: one group per step). Raises ORDER_GROUPS_INVALID unless
    there is one int per step, starting at 0 and increasing by 0 or 1."""
    if groups is None:
        return list(range(n_steps))
    out = [int(g) for g in groups]
    bad = len(out) != n_steps or (n_steps > 0 and out[0] != 0)
    bad = bad or any(b - a not in (0, 1) for a, b in zip(out, out[1:], strict=False))
    if bad:
        msg = f"order groups {out} for {n_steps} steps: need one per step, 0, 0|+1, ..."
        raise ServeError(Code.ORDER_GROUPS_INVALID, msg, status=422, detail={"groups": out})
    return out


def independence_issues(docs: Sequence[dict], groups: Sequence[int]) -> tuple[list[str], list]:
    """(warnings, conflicts) of the independence claims. A conflict is a ServeError
    NOT_INDEPENDENT (two steps of one map-continuous run share touched maps); a run broken by
    a merge gives an INDEPENDENCE_UNCHECKED warning."""
    warnings: list[str] = []
    conflicts: list[ServeError] = []
    for g in sorted(set(groups)):
        members = [i for i, x in enumerate(groups) if x == g]
        runs = [members[:1]]
        for i, j in zip(members, members[1:], strict=False):
            if docs[i].get("product_state_id") == docs[j].get("reactant_state_id"):
                runs[-1].append(j)
                continue
            warnings.append(
                f"{Code.INDEPENDENCE_UNCHECKED}: steps {i} and {j} of order group {g} are "
                "joined by a merge (no map continuity), so their independence is not checked"
            )
            runs.append([j])
        conflicts += [c for run in runs for c in _shared_maps(docs, run, g)]
    return warnings, conflicts


def _shared_maps(docs: Sequence[dict], run: list[int], group: int) -> list[ServeError]:
    out = []
    for x, i in enumerate(run):
        for j in run[x + 1 :]:
            a, b = docs[i].get("touched_maps") or [], docs[j].get("touched_maps") or []
            shared = sorted(set(a) & set(b))
            if shared:
                msg = f"steps {i} and {j} of order group {group} touch the same atoms {shared}"
                detail = {"group": group, "steps": [i, j], "shared_maps": shared}
                out.append(ServeError(Code.NOT_INDEPENDENT, msg, status=422, detail=detail))
    return out


# ---- database operations


def steps_of(session: Session, mechanism_id: int) -> list[CuratedStep]:
    q = select(CuratedStep).where(CuratedStep.mechanism_id == mechanism_id)
    return list(session.exec(q.order_by(col(CuratedStep.step_index))).all())


def rechecks_of(session: Session, mechanism_id: int) -> list[Recheck]:
    q = select(Recheck).where(Recheck.mechanism_id == mechanism_id)
    return list(session.exec(q.order_by(col(Recheck.created_at), col(Recheck.id))).all())


def _stoichiometry(request: dict) -> list[int]:
    n = len(request.get("reactants_smiles") or [])
    return [int(x) for x in request.get("stoichiometry") or [1] * n]


def create_mechanism(
    session: Session, network: NetworkOut, request: dict, req: CurateIn, job_id: str
) -> Mechanism:
    """Write one Mechanism and its CuratedSteps from a network's chain of edges (design 9.2)."""
    net, req = as_model(NetworkOut, network), as_model(CurateIn, req)
    request = dict(request or net.request.model_dump(mode="json"))
    edges = chain_edges(net, req)
    prov = net.provenance
    docs = [step_doc(e, job_id=job_id, provenance=prov) for e in edges]
    groups = check_order_groups(req.order_groups, len(edges))
    _, conflicts = independence_issues(docs, groups)
    if conflicts:
        raise conflicts[0]
    corrected = [e.kind == "corrected" for e in edges]
    m = Mechanism(
        reactants=json.dumps(list(request.get("reactants_smiles") or [])),
        stoichiometry=json.dumps(_stoichiometry(request)),
        conditions=req.conditions if req.conditions is not None else request.get("conditions"),
        predictor=prov.predictor,
        checkpoint_hash=prov.predictor_version or prov.checkpoint_hash or "",
        spec_version=prov.spec_version,
        status=status_for_action(req.action, any(corrected)),
        notes=req.notes,
    )
    session.add(m)
    session.flush()
    _add_steps(session, m, net, edges, docs, groups, job_id)
    log.info("mechanism %s: %d steps from job %s (%s)", m.id, len(edges), job_id, m.status)
    return m


def _add_steps(session, m: Mechanism, net: NetworkOut, edges, docs, groups, job_id) -> None:
    pool_of = {n.id: n.pool_key for n in net.nodes}
    for i, (e, doc) in enumerate(zip(edges, docs, strict=True)):
        if e.source not in pool_of:
            raise ServeError(
                Code.CONSISTENCY_FAILED, f"edge {e.id}: source node {e.source} missing"
            )
        orig = original_doc(e)
        step = CuratedStep(
            mechanism_id=m.id,
            step_index=i,
            order_group=groups[i],
            reactant_pool_key=pool_of[e.source],
            product_pool_key=e.target_pool_key,
            arrows_json=dumps(doc),
            was_corrected=e.kind == "corrected",
            original_arrows_json=None if orig is None else dumps(orig),
        )
        session.add(step)
    session.flush()


def set_status(session: Session, m: Mechanism, new: str) -> None:
    """Move `m` to status `new` along the transition table; appends the audit line."""
    steps = steps_of(session, m.id)
    if not check_transition(m.status, new, any(s.was_corrected for s in steps)):
        return
    stamp = utcnow().isoformat(timespec="seconds")
    m.notes = f"{m.notes}\n[{stamp}] status {m.status} -> {new}"
    m.status = new
    session.add(m)


def set_order_groups(session: Session, m: Mechanism, groups: list[int]) -> list[str]:
    """Replace the order groups of `m` (not while rejected). Returns the warnings."""
    if m.status == "rejected":
        msg = "the order groups of a rejected mechanism are frozen; reopen it first"
        raise ServeError(Code.INVALID_TRANSITION, msg, status=409)
    steps = steps_of(session, m.id)
    groups = check_order_groups(groups, len(steps))
    docs = [json.loads(s.arrows_json) for s in steps]
    warnings, conflicts = independence_issues(docs, groups)
    if conflicts:
        raise conflicts[0]
    for s, g in zip(steps, groups, strict=True):
        if s.order_group != g:
            s.order_group = g
            session.add(s)
    return warnings


# ---- staleness and output models


def staleness(
    m: Mechanism,
    steps: Sequence[CuratedStep],
    registry: PredictorRegistry,
    spec_version: str,
    fp: str,
) -> list[str]:
    """The reasons `m` is stale against the current predictor version, spec version and
    physics fingerprint (empty: up to date)."""
    from ksmech.serve.core.contracts import USER_REF

    reasons = []
    try:
        info = registry.info_of(m.predictor)
    except ServeError:
        info = None
    if m.predictor == USER_REF.name:
        pass  # a user's mechanism has no model version to go stale
    elif info is None or not info.available or info.version is None:
        reasons.append(PREDICTOR_UNAVAILABLE_REASON)
    elif info.version != m.checkpoint_hash:
        reasons.append(f"predictor_version {m.checkpoint_hash} != {info.version}")
    if m.spec_version != spec_version:
        reasons.append(f"spec_version {m.spec_version} != {spec_version}")
    olds = {str(json.loads(s.arrows_json).get("physics_fingerprint")) for s in steps}
    reasons += [f"physics_fingerprint {old} != {fp}" for old in sorted(olds - {fp})]
    return reasons


def recheck_out(r: Recheck) -> RecheckRowOut:
    return RecheckRowOut.model_validate(r.model_dump())


def summary_out(m: Mechanism, n_steps: int, stale: bool, last_recheck: Any) -> MechanismSummaryOut:
    return MechanismSummaryOut(
        id=m.id,
        created_at=m.created_at,
        reactants=json.loads(m.reactants),
        stoichiometry=json.loads(m.stoichiometry),
        status=m.status,
        predictor=m.predictor,
        checkpoint_hash=m.checkpoint_hash,
        spec_version=m.spec_version,
        n_steps=n_steps,
        stale=stale,
        notes=m.notes,
        last_recheck=last_recheck,
    )


def mechanism_out(
    session: Session, m: Mechanism, registry: PredictorRegistry, spec_version: str, fp: str
) -> MechanismOut:
    """The full mechanism: steps with their latest re-check, every re-check, staleness and
    the independence warnings of its order groups."""
    steps = steps_of(session, m.id)
    rechecks = rechecks_of(session, m.id)
    reasons = staleness(m, steps, registry, spec_version, fp)
    latest = {r.step_index: r for r in rechecks}
    docs = [json.loads(s.arrows_json) for s in steps]
    warnings, conflicts = independence_issues(docs, [s.order_group for s in steps])
    warnings += [f"{c.code}: {c.message}" for c in conflicts]
    step_outs = [
        CuratedStepOut(
            step_index=s.step_index,
            order_group=s.order_group,
            reactant_pool_key=s.reactant_pool_key,
            product_pool_key=s.product_pool_key,
            step=doc,
            was_corrected=s.was_corrected,
            original=None if s.original_arrows_json is None else json.loads(s.original_arrows_json),
            recheck=None if s.step_index not in latest else recheck_out(latest[s.step_index]),
        )
        for s, doc in zip(steps, docs, strict=True)
    ]
    last = rechecks[-1].created_at if rechecks else None
    summary = summary_out(m, len(steps), bool(reasons), last)
    return MechanismOut(
        **summary.model_dump(),
        conditions=m.conditions,
        stale_reasons=reasons,
        steps=step_outs,
        rechecks=[recheck_out(r) for r in rechecks],
        warnings=warnings,
    )
