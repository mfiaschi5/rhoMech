"""The `ksmech` CLI handlers of the lab's data: checkpoints, training runs and reports,
embedding traces, curated mechanisms (read only) and the dataset browser."""

from __future__ import annotations

import argparse

from ksmech.client import summary as sm
from ksmech.client.client import KsmechClient
from ksmech.client.commands import Output, UsageError
from ksmech.client.render import kv, table, to_json

# ---- checkpoints, runs, reports


def cmd_models(c: KsmechClient, a: argparse.Namespace) -> Output:
    models = c.models()
    rows = [sm.checkpoint_summary(ck) for ck in models.checkpoints]
    cols = ["checkpoint", "epoch", "encoder", "usable", "smoke", "metrics"]
    emb = "on" if models.embedding.enabled else f"off ({models.embedding.reason})"
    return Output(models, table(rows, cols) + f"\nembedding: {emb}")


def cmd_ml_runs(c: KsmechClient, a: argparse.Namespace) -> Output:
    out = c.ml_runs()
    rows = [sm.run_row(r) for r in out.runs]
    cols = ["run", "encoder", "n_epochs", "best", "checkpoints", "smoke", "clean"]
    return Output(out, table(rows, cols))


def cmd_ml_run(c: KsmechClient, a: argparse.Namespace) -> Output:
    d = c.ml_run(a.source, a.run)
    s = sm.run_detail_summary(d, detail=a.detail)
    epochs = [e for e in s.pop("epochs", []) if isinstance(e, dict)]
    s.pop("config", None)
    cols = list(dict.fromkeys(k for e in epochs for k in e))[:8]
    return Output(d, kv(s) + "\n" + table(epochs, cols))


def cmd_ml_reports(c: KsmechClient, a: argparse.Namespace) -> Output:
    out = c.ml_reports()
    rows = [sm.report_row(r) for r in out.reports]
    return Output(out, table(rows, ["report", "kind", "modified_at", "headline"]))


# ---- embedding traces


def cmd_embedding_create(c: KsmechClient, a: argparse.Namespace) -> Output:
    if (a.state_id is None) == (a.smiles is None):
        raise UsageError("give exactly one of --state-id and --smiles")
    created = c.embedding_create(
        run=a.run, file=a.file, source=a.source, state_id=a.state_id, smiles=a.smiles
    )
    job = c.wait_job(created.job_id, timeout=a.wait)
    res = job.result.model_dump(mode="json") if job.result is not None else {}
    keys = ["embedding_id", "pool_key", "encoder", "n_layers", "page_url", "cached"]
    text = kv({"job_id": job.job_id, "status": job.status, **{k: res.get(k) for k in keys}})
    return Output(job, text, 0 if job.status == "done" else 1)


def cmd_embedding_get(c: KsmechClient, a: argparse.Namespace) -> Output:
    data = c.embedding(a.embedding_id)
    return Output(data, to_json(sm.compact(data, max_list=6)))


# ---- curated mechanisms (read only)


def cmd_curate_list(c: KsmechClient, a: argparse.Namespace) -> Output:
    mechs = c.curate_list(
        status=a.status, predictor=a.predictor, stale=a.stale, limit=a.limit, offset=a.offset
    )
    rows = [sm.mechanism_row(m) for m in mechs]
    cols = ["id", "status", "reactants", "n_steps", "predictor", "stale", "created_at"]
    return Output(mechs, table(rows, cols))


def cmd_curate_get(c: KsmechClient, a: argparse.Namespace) -> Output:
    m = c.curate_get(a.mechanism_id)
    s = sm.mechanism_summary(m, detail=a.detail)
    steps = s.pop("steps", [])
    cols = ["step", "group", "reactants", "products", "corrected", "recheck_top1"]
    return Output(m, kv(s) + "\n" + table(steps, cols))


# ---- the dataset browser


def cmd_browse_status(c: KsmechClient, a: argparse.Namespace) -> Output:
    st = c.browse_status()
    s = sm.index_summary(st)
    return Output(st, kv({**s["pmechdb"], "curated_steps": s["curated_steps"]}))


def cmd_browse_search(c: KsmechClient, a: argparse.Namespace) -> Output:
    out = c.browse_search(
        a.query,
        mode=a.mode,
        side=a.side,
        source=a.source,
        status=a.status,
        resonance=a.resonance,
        limit=a.limit,
        offset=a.offset,
    )
    s = sm.search_summary(out)
    head = f"{s['total']} steps (index {s['index']}{', truncated' if s['truncated'] else ''})"
    cols = ["ref", "status", "reaction", "n_arrows", "pmechdb_code"]
    return Output(out, head + "\n" + table(s["results"], cols))


def cmd_browse_step(c: KsmechClient, a: argparse.Namespace) -> Output:
    v = c.browse_step(a.ref)
    return Output(v, kv(sm.step_view_summary(v, detail=a.detail)))
