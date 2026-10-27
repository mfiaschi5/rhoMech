"""Compact summaries of API answers: plain dicts with rounded numbers, for the `ksmech` CLI's
tables and the MCP tools (an LLM's context is small, so a summary keeps what a chemist reads
and leaves the bookkeeping out; `detail=True` keeps more).

Atom indices in matrices, arrows and readings are pool atoms of the step's source state
(`docs/serve_v9_contract.md`, "Index spaces"); `pmechdb_code` is on atom-map numbers.
"""

from __future__ import annotations

import math
from typing import Any

from ksmech.serve.core import schemas as S

DIGITS = 4
MAX_LIST = 20


def rnd(value: Any, digits: int = DIGITS) -> Any:
    """Floats rounded to `digits` decimals (NaN / inf -> None), through lists and dicts."""
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    if isinstance(value, list | tuple):
        return [rnd(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: rnd(v, digits) for k, v in value.items()}
    return value


def compact(value: Any, max_list: int = MAX_LIST) -> Any:
    """JSON of unknown shape made small: floats rounded, lists longer than `max_list` cut to
    their first and last items with a note of how many were left out."""
    if isinstance(value, dict):
        return {k: compact(v, max_list) for k, v in value.items()}
    if isinstance(value, list):
        items = [compact(v, max_list) for v in value]
        if len(items) <= max_list:
            return items
        head, tail = items[: max_list // 2], items[-(max_list // 2) :]
        return [*head, f"... {len(items) - len(head) - len(tail)} more ...", *tail]
    return rnd(value)


def drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None and v != [] and v != {}}


# ---- steps and networks


def arrow_text(a: S.ArrowReadingOut) -> str:
    """One electron-pair move: `lp(3)->3-5`, `3-4->3-5`, `4-6->lp(6)` (pool atoms)."""
    src = f"lp({a.source[0]})" if len(a.source) == 1 else "-".join(map(str, a.source))
    dst = f"lp({a.target[0]})" if len(a.target) == 1 else "-".join(map(str, a.target))
    return f"{src}->{dst}"


def anchor_text(a: S.AnchorOut | None) -> str | None:
    return None if a is None else f"{a.sign} {a.i}-{a.j}"


def matrix_text(m: S.MatrixIO) -> str:
    """`bonds 0-4:+1 4-5:-1; lone 5:+2` (pool atoms)."""
    bonds = " ".join(f"{i}-{j}:{d:+d}" for i, j, d in m.bonds)
    lone = " ".join(f"{i}:{d:+d}" for i, d in m.lone)
    return "; ".join(
        p for p in (f"bonds {bonds}" if bonds else "", f"lone {lone}" if lone else "") if p
    )


def reading_summary(r: S.OrbitalReadingOut) -> dict[str, Any]:
    """The orbital reading of a forming bond: its gap factor and the dominant pair."""
    out: dict[str, Any] = {
        "bond": f"{r.i}-{r.j}",
        "g_atom": rnd(r.g_atom),
        "donor": r.donor_diag,
        "acceptor": r.acceptor_diag,
        "tokens": [r.donor_token, r.acceptor_token],
    }
    c = r.canonical
    if c is not None:
        out["mo_pair"] = f"{c.k_label}->{c.m_label}"
        out["gap_ev"] = rnd(c.e_m_ev - c.e_k_ev, 3)
        out["share"] = rnd(c.share, 3)
    return drop_none(out)


def edge_summary(
    edge: S.EdgeOut, nodes: dict[str, S.NodeOut], *, detail: bool = False
) -> dict[str, Any]:
    """One step: reactant and product pools, how sure the predictor is, the arrows."""
    src, dst = nodes.get(edge.source), nodes.get(edge.target)
    out: dict[str, Any] = {
        "edge": edge.id,
        "from": src.display_smiles if src else edge.source,
        "to": dst.display_smiles if dst else ".".join(edge.product_smiles),
        "kind": edge.kind,
        "rank": edge.rank,
        "p": rnd(math.exp(edge.logp), 3) if edge.logp is not None else None,
        "score": rnd(edge.score),
        "anchor": anchor_text(edge.anchor),
        "arrows": [arrow_text(a) for a in edge.arrows],
        "pmechdb_code": edge.pmechdb_code,
    }
    if detail:
        out |= {
            "source_state_id": edge.source_state_id,
            "product_state_id": edge.product_state_id,
            "source_mapped_smiles": edge.source_mapped_smiles,
            "matrix": matrix_text(edge.matrix),
            "readings": [reading_summary(r) for r in edge.readings],
            "n_alternatives": edge.n_alternatives,
            "status": edge.status,
        }
    return drop_none(out)


def pathway_summary(
    pw: S.PathwayOut, edges: dict[str, S.EdgeOut], nodes: dict[str, S.NodeOut], detail: bool
) -> dict[str, Any]:
    steps = [edge_summary(edges[e], nodes, detail=detail) for e in pw.edge_ids if e in edges]
    out = {
        "pathway": pw.id,
        "rank": pw.rank,
        "probability": rnd(pw.probability, 3),
        "logp": rnd(pw.logp, 3),
        "n_steps": pw.n_steps,
        "ends_with": pw.ends_with,
        "closes_cycle": pw.closes_cycle or None,
        "steps": steps,
    }
    return drop_none(out)


def network_summary(
    net: S.NetworkOut, *, n_pathways: int = 5, detail: bool = False
) -> dict[str, Any]:
    """The top pathways of a mechanism network, each a list of step summaries."""
    nodes = {n.id: n for n in net.nodes}
    edges = {e.id: e for e in net.edges}
    root = nodes.get(net.root)
    pathways = sorted(net.pathways, key=lambda p: p.rank)[: max(0, n_pathways)]
    out = {
        "network_id": net.network_id,
        "root": root.display_smiles if root else net.root,
        "root_state_id": root.state_id if root else None,
        "complete": net.complete,
        "scores_are_probabilities": net.scores_are_probabilities,
        "n_nodes": len(net.nodes),
        "n_edges": len(net.edges),
        "n_pathways": len(net.pathways),
        "pathways": [pathway_summary(p, edges, nodes, detail) for p in pathways],
        "skipped": dict(net.skipped) or None,
    }
    return drop_none(out)


def provenance_summary(p: S.Provenance | None) -> dict[str, Any] | None:
    if p is None:
        return None
    ckpt = p.checkpoint
    out = {
        "predictor": p.predictor,
        "version": p.predictor_version,
        "checkpoint": f"{ckpt.source}/{ckpt.run}/{ckpt.file}" if ckpt else None,
        "spec_version": p.spec_version,
        "chemistry": p.chemistry,
    }
    return drop_none(out)


def job_summary(job: S.JobOut, *, n_pathways: int = 5, detail: bool = False) -> dict[str, Any]:
    """A job: status and progress, its error, and the network summary when it has one."""
    out: dict[str, Any] = {
        "job_id": job.job_id,
        "kind": job.kind,
        "status": job.status,
        "provenance": provenance_summary(job.provenance),
    }
    if job.status not in ("done", "failed", "cancelled"):
        pr = job.progress
        out["progress"] = drop_none(
            {"phase": pr.phase, "done": pr.done, "total": pr.total, "message": pr.message or None}
        )
    if job.error is not None:
        out["error"] = {"code": job.error.code, "message": job.error.message}
    if isinstance(job.result, S.NetworkOut):
        out["network"] = network_summary(job.result, n_pathways=n_pathways, detail=detail)
    elif job.result is not None:
        out["result"] = compact(job.result.model_dump(mode="json"))
    return drop_none(out)


def step_check_summary(out: S.StepCheckOut, *, detail: bool = False) -> dict[str, Any]:
    res: dict[str, Any] = {
        "ok": out.ok,
        "issues": [f"{i.code}: {i.message}" for i in out.issues],
        "product_smiles": out.product_smiles,
        "anchor": anchor_text(out.anchor),
        "arrows": [arrow_text(a) for a in out.arrows],
        "pmechdb_code": out.pmechdb_code,
        "product_state_id": out.product_state_id,
        "readings_status": out.readings_status,
        "readings": [reading_summary(r) for r in out.readings],
    }
    if detail:
        res["sigma_keys"] = out.sigma_keys
        res["formal_charges"] = [list(fc) for fc in out.formal_charges]
    return drop_none(res) | {"ok": out.ok}


# ---- inputs, predictors, molecules


def normalize_summary(out: S.NormalizeOut) -> dict[str, Any]:
    species = [
        drop_none(
            {
                "input": s.input,
                "key": s.token_key,
                "count": s.count,
                "charge": s.charge,
                "multiplicity": s.multiplicity,
                "n_atoms": s.n_atoms,
                "errors": [f"{e.code}: {e.message}" for e in s.errors],
            }
        )
        for s in out.species
    ]
    res = {
        "ok": out.ok,
        "pool_key": out.pool_key,
        "state_id": out.state_id,
        "mapped_smiles": out.mapped_smiles,
        "n_atoms": out.n_atoms,
        "species": species,
    }
    return drop_none(res) | {"ok": out.ok}


def predictor_summary(p: S.PredictorInfoOut) -> dict[str, Any]:
    out = {
        "name": p.name,
        "available": p.available,
        "reason": p.reason,
        "scores_are_probabilities": p.scores_are_probabilities,
        "score_unit": p.score_unit,
        "version": p.version,
        "description": p.description,
    }
    return drop_none(out) | {"available": p.available}


def properties_summary(props: S.MoleculePropertiesOut, *, detail: bool = False) -> dict[str, Any]:
    """Global properties, then one row per atom (bonds and Fukui variants with `detail`)."""
    g = props.global_props
    glob = drop_none(
        rnd(
            {
                "homo_ev": g.homo_ev,
                "lumo_ev": g.lumo_ev,
                "gap_ev": g.gap_ev,
                "mu_ev": g.mu_ev,
                "eta_ev": g.eta_ev,
                "omega_ev": g.omega_ev,
                "charge": g.charge,
                "multiplicity": g.multiplicity,
                "n_atoms": g.n_atoms,
                "fermi_smeared": g.fermi_smeared or None,
            },
            3,
        )
    )
    atoms = [_atom_row(a, detail) for a in props.atoms]
    out: dict[str, Any] = {"key": props.key, "global": glob, "atoms": atoms}
    if detail:
        out["bonds"] = [
            rnd({"i": b.i, "j": b.j, "order": b.order, "wiberg": b.wiberg, "r": b.length_ang}, 3)
            for b in props.bonds
            if b.bonded
        ]
        out["method"] = props.provenance.method
    if g.warnings:
        out["warnings"] = g.warnings
    return out


def _atom_row(a: S.AtomPropsOut, detail: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "i": a.index,
        "el": a.element,
        "q_formal": a.formal_charge,
        "q_mulliken": a.mulliken_charge,
        "f_plus": a.f_plus_fmo,
        "f_minus": a.f_minus_fmo,
        "dual": a.dual_fd,
        "vbur": a.vbur,
    }
    if detail:
        row |= {
            "f_plus_fd": a.f_plus_fd,
            "f_minus_fd": a.f_minus_fd,
            "wiberg_valence": a.wiberg_valence,
            "hybridization": a.diagnostic.hybridization,
            "fg": a.diagnostic.fg_label,
        }
    return drop_none(rnd(row, 3))


def symmetry_summary(pg: S.PointGroupOut, *, detail: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "point_group": pg.point_group,
        "order": pg.order,
        "top": pg.top,
        "tolerance": pg.tolerance,
        "rms_ang": rnd(pg.rms_ang),
        "equivalent_atoms": [c.atoms for c in pg.classes if len(c.atoms) > 1],
        "looser": f"{pg.looser.point_group} at {pg.looser.tolerance:g} A" if pg.looser else None,
    }
    if detail:
        out["elements"] = [f"{e.label} atoms={e.atoms}" for e in pg.elements]
        out["notes"] = pg.notes
    return drop_none(out)


def _orbital_row(o: S.OrbitalRowOut) -> dict[str, Any]:
    centres = " ".join(f"{c.element}{c.atom}:{c.weight:.2f}" for c in o.centres[:4])
    diag = o.diagnostic.type if o.diagnostic is not None else None
    row = {
        "index": o.index,
        "label": o.label,
        "occ": rnd(o.occupation, 2),
        "e_ev": rnd(o.energy_ev, 3),
        "centres": centres,
        "type": diag,
    }
    return drop_none(row)


def orbitals_summary(
    os_: S.OrbitalSetOut, *, window: int = 4, detail: bool = False
) -> dict[str, Any]:
    """The orbitals around the frontier: `window` occupied below the HOMO boundary and
    `window` virtual above it (every orbital with `detail`)."""
    occ = [o for o in os_.orbitals if o.kind == "occupied"]
    virt = [o for o in os_.orbitals if o.kind == "virtual"]
    occ.sort(key=lambda o: o.energy_ev)
    virt.sort(key=lambda o: o.energy_ev)
    rows = os_.orbitals if detail else [*occ[-window:], *virt[:window]]
    return {
        "set": os_.set,
        "start": os_.start,
        "n_occupied": os_.stats.n_occupied,
        "n_virtual": os_.stats.n_virtual,
        "orbitals": [_orbital_row(o) for o in rows],
        "sets": os_.sets,
    }


# ---- checkpoints, curation, the dataset browser


def checkpoint_summary(c: S.CheckpointOut) -> dict[str, Any]:
    metrics = {k: v for k, v in c.metrics.items() if "top" in k or "loss" in k}
    out = {
        "checkpoint": f"{c.source}/{c.run}/{c.file}",
        "epoch": c.epoch,
        "encoder": c.encoder,
        "usable": c.usable,
        "smoke": c.smoke or None,
        "reason": c.reason,
        "metrics": rnd(metrics) or None,
        "modified_at": c.modified_at.isoformat(timespec="seconds"),
    }
    return drop_none(out) | {"usable": c.usable}


def mechanism_row(m: S.MechanismSummaryOut) -> dict[str, Any]:
    out = {
        "id": m.id,
        "status": m.status,
        "reactants": ".".join(m.reactants),
        "n_steps": m.n_steps,
        "predictor": m.predictor,
        "stale": m.stale or None,
        "created_at": m.created_at.isoformat(timespec="seconds"),
        "notes": m.notes or None,
    }
    return drop_none(out)


def mechanism_summary(m: S.MechanismOut, *, detail: bool = False) -> dict[str, Any]:
    steps = []
    for st in m.steps:
        row = {
            "step": st.step_index,
            "group": st.order_group,
            "reactants": st.reactant_pool_key,
            "products": st.product_pool_key,
            "corrected": st.was_corrected or None,
            "recheck_top1": st.recheck.top1 if st.recheck else None,
        }
        if detail:
            row["step_doc"] = compact(st.step)
        steps.append(drop_none(row))
    out = mechanism_row(m) | {"conditions": m.conditions, "steps": steps}
    if m.stale_reasons:
        out["stale_reasons"] = m.stale_reasons
    return drop_none(out)


def step_row(s: S.StepSummaryOut) -> dict[str, Any]:
    out = {
        "ref": s.ref,
        "source": s.source,
        "status": s.status,
        "reaction": f"{s.reactants_smiles}>>{s.products_smiles}",
        "n_arrows": s.n_arrows,
        "pmechdb_code": s.pmechdb_code,
        "resonance": s.resonance_step or None,
        "splits": s.splits,
    }
    return drop_none(out)


def search_summary(out: S.SearchOut) -> dict[str, Any]:
    return {
        "total": out.total,
        "truncated": out.truncated,
        "index": out.index.pmechdb.state,
        "results": [step_row(s) for s in out.results],
    }


def step_view_summary(v: S.StepViewOut, *, detail: bool = False) -> dict[str, Any]:
    arrows = [
        f"{a.source_kind}{a.source_atoms}->{a.sink_kind}{a.sink_atoms}" for a in v.dataset_arrows
    ]
    out: dict[str, Any] = step_row(v.step) | {"dataset_arrows": arrows}
    if v.mechanism is not None:
        out["mechanism"] = mechanism_row(v.mechanism)
    if detail:
        out["reactant_pool_key"] = v.step.reactant_pool_key
        out["product_pool_key"] = v.step.product_pool_key
        out["step_id"] = v.step.step_id
    return out


def index_summary(st: S.IndexStatusOut) -> dict[str, Any]:
    p = st.pmechdb
    built = p.built_at.isoformat(timespec="seconds") if p.built_at else None
    pm = drop_none({"state": p.state, "n_steps": p.n_steps, "built_at": built, "reason": p.reason})
    return {"pmechdb": pm, "curated_steps": st.curated_steps}


# ---- training runs and reports


def _metric_keys(line: dict[str, Any] | None) -> dict[str, Any] | None:
    """The epoch, the top-k and the losses of one metrics line (rounded)."""
    if not line:
        return None
    keep = {k: v for k, v in line.items() if k == "epoch" or "top" in k or "loss" in k}
    return rnd(keep) or None


def run_row(r: S.MlRunOut) -> dict[str, Any]:
    out = {
        "run": f"{r.source}/{r.run}",
        "encoder": r.encoder,
        "n_epochs": r.n_epochs,
        "best": _metric_keys(r.best),
        "last_epoch": (r.last or {}).get("epoch"),
        "checkpoints": r.checkpoints,
        "smoke": r.smoke or None,
        "clean": r.clean,
        "reason": r.reason,
        "modified_at": r.modified_at.isoformat(timespec="seconds"),
    }
    return drop_none(out)


def run_detail_summary(d: S.MlRunDetailOut, *, detail: bool = False) -> dict[str, Any]:
    """A run, its configuration, its epochs (metrics only, the first and last 5 unless
    `detail`) and its reports' names."""
    epochs = [_metric_keys(e) or {} for e in d.epochs]
    out = run_row(d.run) | {
        "config": compact(d.run.config, max_list=10),
        "epochs": compact(epochs, max_list=60 if detail else 10),
        "reports": [r.name for r in d.reports],
        "bad_lines": d.bad_lines or None,
    }
    if detail:
        out["model"] = d.run.model
    return drop_none(out)


def report_headline(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """The scalars of a report and of its one-level sections (provenance left out)."""
    if not report:
        return None
    out: dict[str, Any] = {}
    for key, value in report.items():
        if key == "provenance":
            continue
        if _flat(value):
            out[key] = rnd(value)
    return out or None


def _flat(value: Any) -> bool:
    """A scalar, or a dict of scalars."""
    if isinstance(value, dict):
        return all(not isinstance(v, dict | list) for v in value.values())
    return not isinstance(value, list)


def report_row(r: S.MlReportOut, *, detail: bool = False) -> dict[str, Any]:
    out = {
        "report": f"{r.source}/{r.name}",
        "kind": r.kind,
        "modified_at": r.modified_at.isoformat(timespec="seconds"),
        "headline": compact(r.report, max_list=10) if detail else report_headline(r.report),
        "reason": r.reason,
    }
    return drop_none(out)
