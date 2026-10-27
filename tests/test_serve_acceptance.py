"""The acceptance chain over HTTP on a fresh synthetic data root (spec v9):

1. predict MeO- + MeCl with the floor: the SN2 product is an edge with its reaction matrix,
   arrow reading and orbital readings;
2. validate the SN2 pathway with POST /curate;
3. a user step from the root (O takes an H of MeCl) through POST /jobs/{id}/override, then
   POST /jobs/{id}/expand of its node: the node is expanded;
4. GET /browse/search?q=COC&source=user_curated&status=validated returns curated:<mid>:0;
5. GET /curate/export/pmechdb downloads that step as a PMechDB record;
6. a re-check of the mechanism appends a Recheck row and leaves the status alone.
"""

from __future__ import annotations

import pytest

from ksmech.serve.core import schemas as S
from ksmech.serve.core.testing import requires
from tests.test_serve_fixtures_api import (  # noqa: F401 (fixtures)
    api_client,
    edge_to,
    fresh_synthetic_settings,
    no_xtb,
    predict,
    root_node,
    synthetic_root,
)

pytestmark = [pytest.mark.serve, requires("predict"), requires("correct")]
SN2 = "[CH3][O][CH3]|0.[Cl]|-1"


def _job(client, created: dict) -> dict:
    job = client.get(created["status_url"]).json()
    assert job["status"] == "done", job.get("error")
    return job


def _validate(client, job: dict) -> int:
    net = job["result"]
    sn2 = edge_to(net, SN2)
    assert sn2["readings"] and {a["kind"] for a in sn2["arrows"]} == {
        "lone_pair_to_bond",
        "bond_to_lone_pair",
    }
    pathway = next(p for p in net["pathways"] if p["edge_ids"] == [sn2["id"]])
    body = {"job_id": job["job_id"], "pathway_id": pathway["id"], "action": "validate"}
    r = client.post("/api/curate", json=body | {"notes": "acceptance: SN2"})
    assert r.status_code == 201, r.text
    m = S.MechanismOut.model_validate(r.json())
    assert m.status == "validated" and m.n_steps == 1 and not m.stale
    assert m.steps[0].product_pool_key == SN2
    assert m.steps[0].step["pmechdb_code"] == sn2["pmechdb_code"]
    return m.id


def _user_step(client, job: dict) -> None:
    from ksmech.data import chem

    root = root_node(job["result"])
    mol = chem.parse_mapped(client.get(f"/api/pool/{root['state_id']}").json()["mapped_smiles"])
    o = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O")
    cl = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Cl")
    c = mol.GetAtomWithIdx(cl).GetNeighbors()[0].GetIdx()
    h = next(n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() if n.GetSymbol() == "H")
    matrix = {"bonds": [[*sorted((o, h)), 1], [*sorted((c, h)), -1]], "lone": [[o, -2], [c, 2]]}
    body = {"parent_state_id": root["state_id"], "matrix": matrix}
    r = client.post(f"/api/jobs/{job['job_id']}/override", json=body)
    assert r.status_code == 200, r.text
    out = S.OverrideResultOut.model_validate(r.json())
    assert out.edge.predictor == "user" and out.node.status == "frontier"
    r = client.post(f"/api/jobs/{job['job_id']}/expand", json={"node_id": out.node.id})
    assert r.status_code == 202, r.text
    _job(client, r.json())
    after = client.get(f"/api/jobs/{job['job_id']}").json()["result"]
    node = next(n for n in after["nodes"] if n["id"] == out.node.id)
    assert node["status"] in ("expanded", "terminal", "failed"), node
    assert node["status"] != "frontier"


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


def _export(client, mid: int) -> None:
    r = client.get("/api/curate/export/pmechdb")
    assert r.status_code == 200, r.text
    doc = client.get(f"/api/curate/{mid}").json()["steps"][0]["step"]
    assert (
        r.text.strip()
        == f'"{doc["reactants_mapped"]}>>{doc["products_mapped"]} {doc["pmechdb_code"]}"'
    )


def _recheck(client, mid: int) -> None:
    r = client.post(f"/api/curate/{mid}/recheck", json={})
    assert r.status_code == 202, r.text
    report = S.RecheckReportOut.model_validate(_job(client, r.json())["result"])
    assert report.n_steps == 1 and len(report.rows) == 1 and report.rows[0].reason is None
    m = client.get(f"/api/curate/{mid}").json()
    assert m["status"] == "validated" and len(m["rechecks"]) == 1


def test_acceptance_chain(fresh_synthetic_settings):  # noqa: F811
    with api_client(fresh_synthetic_settings) as client:
        job = predict(client, beam=5)
        assert job["provenance"]["predictor"] == "floor"
        mid = _validate(client, job)
        _user_step(client, job)
        _browse(client, mid)
        _export(client, mid)
        _recheck(client, mid)
