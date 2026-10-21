"""The brief's acceptance chain over HTTP (design 13.2), on a fresh synthetic data root:

1. predict MeO- + MeCl with the stub, and once with the floor (the SN2 tuple is in
   /candidates and its physics panel resolves);
2. validate the predicted SN2 pathway with POST /curate;
3. override the first root edge to a different acceptor (sigma*(C-H)) with continue_steps=1:
   the corrected target is no longer unexpanded;
4. GET /browse/search?q=COC&source=user_curated&status=validated returns curated:<mid>:0;
5. POST /curate/export writes steps.parquet holding that step (read back with read_steps);
6. a re-check of the mechanism appends a Recheck row and leaves the status alone.

The pathway is validated before the override because the override supersedes the SN2 edge,
and pathways never run through superseded edges.
"""

from __future__ import annotations

import math

import pytest

from ksmech.serve import schemas as S
from tests.test_serve_fixtures_api import (  # noqa: F401 (fixtures)
    CH_TUPLE,
    SN2_TUPLE,
    api_client,
    arrow_of,
    candidates,
    edge_to,
    fresh_synthetic_settings,
    no_xtb,
    predict,
    root_node,
    synthetic_root,
)

pytestmark = pytest.mark.serve


def _job(client, created: dict) -> dict:
    job = client.get(created["status_url"]).json()
    assert job["status"] == "done", job.get("error")
    return job


def _floor_check(client) -> None:
    job = predict(client, predictor="floor")
    assert job["provenance"]["predictor"] == "floor"
    sid = root_node(job["result"])["state_id"]
    rows = candidates(client, sid, predictor="floor", **SN2_TUPLE)
    assert rows, "the SN2 tuple (O -> sigma*(C-Cl) at sites O, C) is missing from /candidates"
    r = client.get(f"/api/pool/{sid}/physics", params=arrow_of(rows[0]))
    assert r.status_code == 200, r.text
    p = S.PhysicsOut.model_validate(r.json())
    assert p.pair_valid and math.isfinite(p.ks3) and p.floor_rank is not None
    assert p.dominant_pair.k_label and math.isfinite(p.dominant_pair.gap)


def _validate(client, job: dict) -> int:
    net = job["result"]
    sn2 = edge_to(net, "COC.[Cl-]")
    pathway = next(p for p in net["pathways"] if p["edge_ids"] == [sn2["id"]])
    body = {"job_id": job["job_id"], "pathway_id": pathway["id"], "action": "validate"}
    r = client.post("/api/curate", json=body | {"notes": "acceptance: SN2"})
    assert r.status_code == 201, r.text
    m = S.MechanismOut.model_validate(r.json())
    assert m.status == "validated" and m.n_steps == 1 and not m.stale
    assert m.steps[0].product_pool_key == "COC.[Cl-]"
    assert m.steps[0].step["pmechdb_code"] == "2=6;6,7=7"
    return m.id


def _override(client, job: dict) -> None:
    net = job["result"]
    root = root_node(net)
    first = next(e for e in net["edges"] if e["source"] == root["id"])
    rows = candidates(client, first["source_state_id"], **CH_TUPLE)
    assert rows, "no O -> sigma*(C-H) tuple"
    body = {"edge_id": first["id"], "arrows": [arrow_of(rows[0])], "continue_steps": 1}
    r = client.post(f"/api/jobs/{job['job_id']}/override", json=body)
    assert r.status_code == 202, r.text
    result = S.OverrideResultOut.model_validate(_job(client, r.json())["result"])
    after = client.get(f"/api/jobs/{job['job_id']}").json()["result"]
    edges = {e["id"]: e for e in after["edges"]}
    assert edges[result.corrected_edge_id]["kind"] == "corrected"
    assert edges[first["id"]]["status"] == "superseded"
    target = next(n for n in after["nodes"] if n["id"] == result.target_node)
    assert target["status"] in ("expanded", "failed", "terminal"), target
    assert target["status"] != "terminal" or target["terminal_reason"] != "MAX_STEPS"
    assert target["status"] != "frontier"
    assert len(after["nodes"]) > len(net["nodes"])


def _browse(client, mid: int) -> None:
    params = {"q": "COC", "source": "user_curated", "status": "validated"}
    r = client.get("/api/browse/search", params=params)
    assert r.status_code == 200, r.text
    refs = [x["ref"] for x in r.json()["results"]]
    assert f"curated:{mid}:0" in refs
    view = client.get(f"/api/browse/step/curated:{mid}:0")
    assert view.status_code == 200, view.text
    step = S.StepViewOut.model_validate(view.json())
    assert step.step.status == "validated" and step.mechanism.id == mid
    assert step.reactants_depiction.server_svg and step.dataset_arrows


def _export(client, settings, mid: int) -> None:
    from ksmech.data.io import read_steps

    r = client.post("/api/curate/export", json={})
    assert r.status_code == 202, r.text
    report = S.ExportReportOut.model_validate(_job(client, r.json())["result"])
    assert report.n_steps_written == 1 and report.n_mechanisms == 1
    path = settings.processed / "user_curated" / "steps.parquet"
    assert report.paths["steps"] == str(path)
    steps = read_steps(path)
    assert [s.step_id for s in steps] == [report.steps[0]["step_id"]]
    step = steps[0]
    assert str(step.source) == "user_curated"
    assert step.pathway_id == f"user_curated-path-{mid}" and step.step_index == 0
    doc = client.get(f"/api/curate/{mid}").json()["steps"][0]["step"]
    assert step.reactants_smiles == doc["reactants_mapped"]
    assert step.products_smiles == doc["products_mapped"]
    latest = client.get("/api/curate/export/latest").json()
    assert latest["export_id"] == report.export_id


def _recheck(client, mid: int) -> None:
    r = client.post(f"/api/curate/{mid}/recheck", json={})
    assert r.status_code == 202, r.text
    report = S.RecheckReportOut.model_validate(_job(client, r.json())["result"])
    assert report.n_steps == 1 and len(report.rows) == 1
    m = client.get(f"/api/curate/{mid}").json()
    assert m["status"] == "validated" and len(m["rechecks"]) == 1


def test_acceptance_chain(fresh_synthetic_settings):  # noqa: F811
    s = fresh_synthetic_settings
    with api_client(s) as client:
        job = predict(client)
        assert job["provenance"]["predictor"] == "stub"
        _floor_check(client)
        mid = _validate(client, job)
        _override(client, job)
        _browse(client, mid)
        _export(client, s, mid)
        _recheck(client, mid)
