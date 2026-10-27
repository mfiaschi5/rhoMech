"""The handlers of the `ksmech` CLI: each takes the client and the parsed arguments and returns
an `Output` (the API answer for `--json`, the text of a compact table otherwise, the exit
code). They call only `KsmechClient`; the parser is `ksmech.client.cli`."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ksmech.client import summary as sm
from ksmech.client.client import KsmechClient
from ksmech.client.errors import KsmechAPIError
from ksmech.client.render import kv, table
from ksmech.serve.core import schemas as S

MOLECULE_ID = re.compile(r"m[0-9a-f]{20}")


class UsageError(Exception):
    """Arguments argparse accepts but the command cannot use (exit 2)."""


@dataclass
class Output:
    data: Any  # the API answer (models or JSON), printed by --json
    text: str  # the human rendering
    code: int = 0


Handler = Callable[[KsmechClient, argparse.Namespace], Output]


# ---- meta


def cmd_health(c: KsmechClient, a: argparse.Namespace) -> Output:
    h = c.health()
    w = h.worker
    text = kv(
        {
            "ok": h.ok,
            "chemistry": h.chemistry,
            "jobs_running": h.jobs_running,
            "worker": f"alive={w.alive} busy={w.busy} queue={w.queue_depth}",
        }
    )
    return Output(h, text, 0 if h.ok else 1)


def cmd_meta(c: KsmechClient, a: argparse.Namespace) -> Output:
    raw = c.get_json("/api/meta")
    m = S.MetaOut.model_validate(raw)
    caps = {k: v.enabled for k, v in m.capabilities.items()}
    head = {
        "url": c.url,
        "app_version": m.app_version,
        "spec": f"{m.spec_label} ({m.spec_version})",
        "profile": raw.get("profile"),
        "chemistry": m.chemistry,
        "qm_method": m.qm_method,
        "on": " ".join(k for k, on in caps.items() if on),
        "off": " ".join(k for k, on in caps.items() if not on) or None,
    }
    return Output(raw, kv(head))


def cmd_whoami(c: KsmechClient, a: argparse.Namespace) -> Output:
    who = c.whoami()
    return Output(who, kv(who.model_dump(mode="json")))


def cmd_predictors(c: KsmechClient, a: argparse.Namespace) -> Output:
    infos = c.predictors()
    rows = [sm.predictor_summary(p) for p in infos]
    cols = ["name", "available", "scores_are_probabilities", "version", "reason"]
    return Output(infos, table(rows, cols))


# ---- prediction


def _checkpoint(a: argparse.Namespace) -> dict | None:
    if not a.checkpoint:
        return None
    run, _, file = a.checkpoint.partition("/")
    return {k: v for k, v in (("run", run), ("file", file), ("source", a.source)) if v}


def predict_body(a: argparse.Namespace) -> dict[str, Any]:
    fields = {
        "reactants_smiles": list(a.smiles),
        "stoichiometry": a.stoichiometry,
        "predictor": a.predictor,
        "beam": a.beam,
        "max_steps": a.max_steps,
        "temperature": a.temperature,
        "conditions": a.conditions,
        "checkpoint": _checkpoint(a),
    }
    return {k: v for k, v in fields.items() if v is not None}


def job_text(job: S.JobOut, n_pathways: int, detail: bool) -> str:
    """The job line, its error, then each pathway with one line per step."""
    s = sm.job_summary(job, n_pathways=n_pathways, detail=detail)
    lines = [f"job {job.job_id}  {job.kind}  {job.status}"]
    if "error" in s:
        lines.append(f"error: [{s['error']['code']}] {s['error']['message']}")
    if "progress" in s:
        lines.append(f"progress: {s['progress']}")
    net = s.get("network")
    if net:
        lines.append(
            f"root {net['root']}  nodes {net['n_nodes']}  edges {net['n_edges']}  "
            f"pathways {net['n_pathways']}  complete {net['complete']}"
        )
        for pw in net.get("pathways", []):
            p = pw.get("probability", "-")
            lines.append(f"\npathway {pw['rank']}  p={p}  logp={pw['logp']}  -> {pw['ends_with']}")
            cols = ["from", "to", "rank", "p", "score", "arrows"]
            lines.append(table(pw["steps"], cols))
    return "\n".join(lines)


def cmd_predict(c: KsmechClient, a: argparse.Namespace) -> Output:
    created = c.submit_predict(predict_body(a))
    if a.no_wait:
        text = kv({"job_id": created.job_id, "status": created.status, "url": created.status_url})
        return Output(created, text)
    job = c.wait_job(created.job_id, timeout=a.wait)
    return Output(job, job_text(job, a.pathways, a.detail), 0 if job.status == "done" else 1)


def cmd_jobs_list(c: KsmechClient, a: argparse.Namespace) -> Output:
    jobs = c.jobs(kind=a.kind, status=a.status, limit=a.limit)
    rows = [j.model_dump(mode="json") for j in jobs]
    return Output(jobs, table(rows, ["job_id", "kind", "status", "created_at", "label"]))


def cmd_jobs_get(c: KsmechClient, a: argparse.Namespace) -> Output:
    job = c.job(a.job_id)
    return Output(job, job_text(job, a.pathways, a.detail))


def cmd_jobs_wait(c: KsmechClient, a: argparse.Namespace) -> Output:
    job = c.wait_job(a.job_id, timeout=a.wait)
    return Output(job, job_text(job, a.pathways, a.detail), 0 if job.status == "done" else 1)


def cmd_jobs_cancel(c: KsmechClient, a: argparse.Namespace) -> Output:
    job = c.cancel_job(a.job_id, force=a.force)
    return Output(job, f"job {job.job_id}  {job.status}")


def cmd_jobs_pathways(c: KsmechClient, a: argparse.Namespace) -> Output:
    pws = c.pathways(a.job_id, limit=a.limit, include_pruned=a.include_pruned)
    rows = [p.model_dump(mode="json") for p in pws]
    cols = ["rank", "id", "n_steps", "probability", "logp", "ends_with"]
    return Output(pws, table(rows, cols))


def parse_matrix(bonds: list[str] | None, lone: list[str] | None) -> dict[str, list]:
    """`--bond 0,4,+1 --lone 5,+2` -> a `MatrixIO` body (pool atoms; i < j for bonds)."""
    out: dict[str, list] = {"bonds": [], "lone": []}
    for key, items, size in (("bonds", bonds, 3), ("lone", lone, 2)):
        for item in items or ():
            parts = [p for p in re.split(r"[,\s]+", item.strip()) if p]
            if len(parts) != size:
                raise UsageError(f"--{key.rstrip('s')} {item!r}: expected {size} integers")
            try:
                out[key].append([int(p) for p in parts])
            except ValueError as exc:
                raise UsageError(f"--{key.rstrip('s')} {item!r}: not integers") from exc
    if not out["bonds"] and not out["lone"]:
        raise UsageError("give at least one --bond I,J,DELTA or --lone I,DELTA")
    return out


def cmd_check(c: KsmechClient, a: argparse.Namespace) -> Output:
    out = c.step_check(a.state_id, parse_matrix(a.bond, a.lone))
    s = sm.step_check_summary(out, detail=a.detail)
    readings = s.pop("readings", [])
    text = kv(s)
    if readings:
        text += "\n" + table(readings, ["bond", "g_atom", "donor", "acceptor", "mo_pair"])
    return Output(out, text, 0 if out.ok else 1)


# ---- pools and molecules


def cmd_normalize(c: KsmechClient, a: argparse.Namespace) -> Output:
    out = c.normalize(a.smiles, a.stoichiometry)
    s = sm.normalize_summary(out)
    cols = ["input", "key", "count", "charge", "n_atoms", "errors"]
    text = kv(s, ["ok", "pool_key", "state_id", "n_atoms"]) + "\n" + table(s["species"], cols)
    return Output(out, text, 0 if out.ok else 1)


def cmd_pool_create(c: KsmechClient, a: argparse.Namespace) -> Output:
    return _pool_out(c.create_pool(a.smiles, a.stoichiometry))


def cmd_pool_get(c: KsmechClient, a: argparse.Namespace) -> Output:
    return _pool_out(c.pool(a.state_id))


def _pool_out(p: S.PoolOut) -> Output:
    keys = ["state_id", "pool_key", "mapped_smiles", "n_atoms", "n_orbitals", "analysis"]
    return Output(p, kv(p.model_dump(mode="json"), keys))


def cmd_pool_svg(c: KsmechClient, a: argparse.Namespace) -> Output:
    d = c.pool_depiction(a.state_id, server_svg=True)
    svg = d.server_svg or ""
    if a.out:
        Path(a.out).write_text(svg, encoding="utf-8")
        return Output(d, f"wrote {a.out} ({len(svg)} bytes)")
    return Output(d, svg)


def molecule_of(c: KsmechClient, ref: str) -> S.MoleculeOut:
    """A molecule id (m + 20 hex) as is; a SMILES is created and waited for."""
    mol = c.molecule(ref) if MOLECULE_ID.fullmatch(ref) else c.molecule_ready(smiles=ref)
    if mol.status == "failed":
        err = mol.error
        code, msg = (err.code, err.message) if err else ("FEATURIZE_FAILED", mol.key)
        raise KsmechAPIError(code, msg)
    return mol


def cmd_molecule_create(c: KsmechClient, a: argparse.Namespace) -> Output:
    if a.molblock:
        mol = c.molecule_create(molblock=Path(a.molblock).read_text(encoding="utf-8"))
    elif a.smiles:
        mol = c.molecule_create(smiles=a.smiles)
    else:
        raise UsageError("give a SMILES or --molblock FILE")
    return _molecule_out(mol)


def cmd_molecule_get(c: KsmechClient, a: argparse.Namespace) -> Output:
    return _molecule_out(molecule_of(c, a.molecule))


def _molecule_out(m: S.MoleculeOut) -> Output:
    keys = ["molecule_id", "key", "status", "n_atoms", "charge", "multiplicity", "method"]
    return Output(m, kv(m.model_dump(mode="json"), keys))


def cmd_molecule_props(c: KsmechClient, a: argparse.Namespace) -> Output:
    props = c.molecule_properties(molecule_of(c, a.molecule).molecule_id)
    s = sm.properties_summary(props, detail=a.detail)
    cols = ["i", "el", "q_formal", "q_mulliken", "f_plus", "f_minus", "dual", "vbur"]
    text = kv({"key": s["key"], **s["global"]}) + "\n" + table(s["atoms"], cols)
    if a.detail:
        text += "\n" + table(s["bonds"], ["i", "j", "order", "wiberg", "r"])
    return Output(props, text)


def cmd_molecule_symmetry(c: KsmechClient, a: argparse.Namespace) -> Output:
    mid = molecule_of(c, a.molecule).molecule_id
    pg = c.molecule_symmetry(mid, tolerance=a.tolerance)
    return Output(pg, kv(sm.symmetry_summary(pg, detail=a.detail)))


def cmd_molecule_orbitals(c: KsmechClient, a: argparse.Namespace) -> Output:
    mid = molecule_of(c, a.molecule).molecule_id
    os_ = c.molecule_orbitals(mid, orbital_set=a.set, start=a.start)
    s = sm.orbitals_summary(os_, window=a.window, detail=a.detail)
    head = kv({k: s[k] for k in ("set", "start", "n_occupied", "n_virtual")})
    return Output(
        os_,
        head + "\n" + table(s["orbitals"], ["index", "label", "occ", "e_ev", "type", "centres"]),
    )


def cmd_depict2d(c: KsmechClient, a: argparse.Namespace) -> Output:
    molblock = Path(a.molfile).read_text(encoding="utf-8")
    d = c.depict2d(molblock, mode=a.mode, style=a.style)
    text = kv(
        {"smiles": d.smiles, "method": d.method, "stereo": d.stereo, "score": d.defects.score}
    )
    return Output(d, text)
