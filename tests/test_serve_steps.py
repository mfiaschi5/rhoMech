"""Correct and curate under spec v9 on the synthetic chemistry: the issues of a reaction matrix,
`POST /api/step/check`, a user network with `override` and `expand`, the predictors endpoint,
the depiction's pool atoms, and the curation round trip (StepDoc, re-check, PMechDB export)."""

from __future__ import annotations

import pytest

from ksmech.serve.testing import requires
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    serve_settings,
    sn2_state,
    synthetic_root,
    synthetic_settings,
)
from tests.test_serve_fixtures_api import (  # noqa: F401 (fixtures)
    api_client,
    fresh_synthetic_settings,
    predict,
    root_node,
    synthetic_app,
)


def _mol(state):  # noqa: F811
    from ksmech.data import chem

    return chem.parse_mapped(state.mapped_smiles)


def sn2(state) -> dict:  # noqa: F811
    """MatrixIO of MeO- + MeCl -> Me2O + Cl- on the state's pool atoms."""
    mol = _mol(state)
    o = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O")
    cl = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Cl")
    c = mol.GetAtomWithIdx(cl).GetNeighbors()[0].GetIdx()
    return {
        "bonds": [[*sorted((o, c)), 1], [*sorted((c, cl)), -1]],
        "lone": [[o, -2], [cl, 2]],
        "_atoms": (o, c, cl),
    }


def io(m: dict) -> dict:  # noqa: F811
    return {"bonds": m["bonds"], "lone": m["lone"]}


def codes(state, matrix) -> list[str]:  # noqa: F811
    from ksmech.serve import stepapply

    return [x.code for x in stepapply.apply_matrix(state, stepapply.matrix_of(matrix)).issues]


# ---- issues


def test_valid_sn2_has_no_issue(sn2_state):  # noqa: F811
    assert codes(sn2_state, io(sn2(sn2_state))) == []


def test_issue_codes_and_atoms(sn2_state):  # noqa: F811
    from ksmech.serve import stepapply

    m = sn2(sn2_state)
    o, c, cl = m["_atoms"]
    h = next(
        n.GetIdx() for n in _mol(sn2_state).GetAtomWithIdx(c).GetNeighbors() if n.GetSymbol() == "H"
    )
    assert codes(sn2_state, {"bonds": [], "lone": []}) == ["EMPTY_MATRIX"]
    assert codes(sn2_state, {"bonds": [[*sorted((o, c)), 1]], "lone": []}) == [
        "CHARGE_NOT_CONSERVED"
    ]
    assert "BREAK_UNBONDED" in codes(
        sn2_state, {"bonds": [[*sorted((o, cl)), -1]], "lone": [[o, 2]]}
    )
    assert "BOND_ORDER_RANGE" in codes(
        sn2_state, {"bonds": [[*sorted((c, cl)), -2]], "lone": [[cl, 4]]}
    )
    assert "NO_LONE_PAIR" in codes(sn2_state, {"bonds": [[*sorted((c, h)), 1]], "lone": [[c, -2]]})
    # C with five bonds: valence
    assert "VALENCE_EXCEEDED" in codes(
        sn2_state, {"bonds": [[*sorted((o, c)), 1]], "lone": [[o, -2]]}
    )
    assert codes(sn2_state, {"bonds": [[0, 999, 1]], "lone": [[0, -2]]}) == ["MATRIX_MALFORMED"]
    issue = stepapply.apply_matrix(
        sn2_state, stepapply.matrix_of({"bonds": [[*sorted((o, cl)), -1]], "lone": [[o, 2]]})
    ).issues
    assert any(x.code == "BREAK_UNBONDED" and set(x.atoms) == {o, cl} for x in issue)


def test_lone_pair_negative(serve_settings):  # noqa: F811
    from ksmech.serve.pools import SpeciesInput, initial_state

    state = initial_state([SpeciesInput("CO", 1), SpeciesInput("CCl", 1)], serve_settings().limits)
    mol = _mol(state)
    o = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O")  # 4 lone electrons
    cl = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Cl")
    got = codes(state, {"bonds": [[*sorted((o, cl)), 3]], "lone": [[o, -6]]})
    assert "LONE_PAIR_NEGATIVE" in got


def test_resonance_only_step_is_an_issue(serve_settings):  # noqa: F811
    from ksmech.serve.pools import SpeciesInput, initial_state

    s = serve_settings()
    state = initial_state([SpeciesInput("CC(=O)[O-]", 1)], s.limits)
    mol = _mol(state)
    c = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C" and a.GetDegree() == 3)
    o2 = next(
        n.GetIdx()
        for n in mol.GetAtomWithIdx(c).GetNeighbors()
        if n.GetSymbol() == "O" and n.GetFormalCharge() == 0
    )
    o1 = next(n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() if n.GetFormalCharge() == -1)
    shift = {"bonds": [[*sorted((c, o1)), 1], [*sorted((c, o2)), -1]], "lone": [[o1, -2], [o2, 2]]}
    assert codes(state, shift) == ["RESONANCE_ONLY_STEP"]


# ---- step check, predictors, depiction (API)


@requires("correct")
def test_step_check_endpoint(synthetic_app, sn2_state):  # noqa: F811
    m = sn2(sn2_state)
    r = synthetic_app.post(
        "/api/step/check", json={"state_id": sn2_state.state_id, "matrix": io(m)}
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ok"] and out["issues"] == [] and out["readings_status"] == "ready"
    assert sorted(out["sigma_keys"]) == ["[CH3][O][CH3]|0", "[Cl]|-1"]
    assert {a["kind"] for a in out["arrows"]} == {"lone_pair_to_bond", "bond_to_lone_pair"}
    o, c, cl = m["_atoms"]
    reading = out["readings"][0]
    assert {reading["i"], reading["j"]} == {o, c} and reading["canonical"]["k_label"]
    assert [cl, -1] in out["formal_charges"] and out["product_state_id"]
    assert out["anchor"] == {"i": min(o, c), "j": max(o, c), "sign": "form"}
    assert out["pmechdb_code"]
    bad = synthetic_app.post(
        "/api/step/check",
        json={"state_id": sn2_state.state_id, "matrix": {"bonds": [[o, c, 1]], "lone": []}},
    ).json()
    assert not bad["ok"] and bad["issues"][0]["code"] == "CHARGE_NOT_CONSERVED"
    assert set(bad["issues"][0]["atoms"]) == {o, c}


@requires("predict")
def test_predictors_endpoint(synthetic_app):  # noqa: F811
    rows = synthetic_app.get("/api/predictors").json()
    by = {r["name"]: r for r in rows}
    assert list(by) == ["floor", "gbdt", "network"]
    assert by["floor"]["available"] and by["floor"]["scores_are_probabilities"] is False
    assert by["network"]["available"] is False and by["network"]["reason"]
    assert all(r["kind"] == r["name"] for r in rows)


def test_depiction_maps_pool_atoms_and_hidden_h(synthetic_app, sn2_state):  # noqa: F811
    d = synthetic_app.get(f"/api/pool/{sn2_state.state_id}/depiction").json()
    seen = {a["pool_atom"] for a in d["atoms"]} | {
        h for a in d["atoms"] for h in a["hidden_h_atoms"]
    }
    assert seen == set(range(sn2_state.n_atoms))
    mol = _mol(sn2_state)
    for a in d["atoms"]:
        assert mol.GetAtomWithIdx(a["pool_atom"]).GetSymbol() == a["element"]


# ---- user network: override, expand


def _user_job(client) -> dict:  # noqa: F811
    return predict(client, predictor="user", reactants_smiles=["C[O-]", "CCl"])


@requires("correct")
def test_user_job_override_and_expand(fresh_synthetic_settings):  # noqa: F811
    with api_client(fresh_synthetic_settings) as client:
        job = _user_job(client)
        net = job["result"]
        assert len(net["nodes"]) == 1 and net["edges"] == []
        root = root_node(net)
        from ksmech.serve.pools import load_state

        state = load_state(root["state_id"], fresh_synthetic_settings)
        body = {"parent_state_id": state.state_id, "matrix": io(sn2(state))}
        r = client.post(f"/api/jobs/{job['job_id']}/override", json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        edge, node = out["edge"], out["node"]
        assert edge["predictor"] == "user" and edge["kind"] == "corrected" and edge["rank"] == 0
        assert edge["logp"] is None and edge["score"] > 0 and edge["readings"]
        assert node["pool_key"] == "[CH3][O][CH3]|0.[Cl]|-1" and node["status"] == "frontier"
        again = client.post(f"/api/jobs/{job['job_id']}/override", json=body).json()
        assert again["edge"]["id"] == edge["id"]  # idempotent
        invalid = client.post(
            f"/api/jobs/{job['job_id']}/override",
            json={"parent_state_id": state.state_id, "matrix": {"bonds": [], "lone": []}},
        )
        assert invalid.status_code == 422
        err = invalid.json()["error"]
        assert (
            err["code"] == "STEP_INVALID" and err["detail"]["issues"][0]["code"] == "EMPTY_MATRIX"
        )
        r = client.post(f"/api/jobs/{job['job_id']}/expand", json={"node_id": root["id"]})
        assert r.status_code == 202, r.text
        done = client.get(f"/api/jobs/{r.json()['job_id']}").json()
        assert done["status"] == "done", done.get("error")
        assert done["result"]["predictor"] == "floor" and done["result"]["n_edges"] > 1
        net = client.get(f"/api/jobs/{job['job_id']}").json()["result"]
        predicted = [e for e in net["edges"] if e["kind"] == "predicted"]
        assert predicted and all(e["predictor"] == "floor" and e["rank"] >= 1 for e in predicted)
        assert net["scores_are_probabilities"] is False


@requires("predict")
def test_predict_job_edges_carry_the_v9_fields(synthetic_app):  # noqa: F811
    job = predict(synthetic_app, beam=3, max_steps=1)
    net = job["result"]
    assert net["edges"], net["skipped"]
    e = net["edges"][0]
    for key in ("matrix", "anchor", "arrows", "readings", "score", "logp", "predictor", "rank",
                "product_smiles", "sigma_keys"):  # fmt: skip
        assert key in e
    assert e["predictor"] == "floor" and e["logp"] is None
    assert all(p["probability"] is None for p in net["pathways"])


# ---- curation round trip


@requires("export")
def test_curation_recheck_and_export_round_trip(fresh_synthetic_settings):  # noqa: F811
    with api_client(fresh_synthetic_settings) as client:
        job = predict(client, beam=3, max_steps=1)
        net = job["result"]
        top = min((e for e in net["edges"]), key=lambda e: e["rank"])
        body = {"job_id": job["job_id"], "edge_ids": [top["id"]], "action": "validate"}
        r = client.post("/api/curate", json=body)
        assert r.status_code == 201, r.text
        mech = r.json()
        doc = mech["steps"][0]["step"]
        assert doc["format"] == "ksmech-serve/step-v2"
        assert doc["matrix"] == top["matrix"] and doc["reaction_matrix"] == [
            top["matrix"]["bonds"],
            top["matrix"]["lone"],
        ]
        assert doc["arrows"] == top["arrows"] and doc["dataset_arrows"]
        r = client.post(f"/api/curate/{mech['id']}/recheck", json={})
        assert r.status_code == 202, r.text
        report = client.get(f"/api/jobs/{r.json()['job_id']}").json()["result"]
        row = report["rows"][0]
        assert row["reason"] is None and row["first_arrow_rank"] == top["rank"]
        assert row["product_match"] == "FOUND" and row["top1"] == (top["rank"] == 1)
        r = client.get("/api/curate/export/pmechdb")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
        lines = r.text.strip().splitlines()
        assert len(lines) == 1 and top["pmechdb_code"] in lines[0]
        assert '"n_steps_written": 1' in r.headers["x-ksmech-export-report"]


@requires("export")
def test_export_lines_parse_with_the_pmechdb_reader(fresh_synthetic_settings, tmp_path):  # noqa: F811
    from ksmech.data import pmechdb

    with api_client(fresh_synthetic_settings) as client:
        job = predict(client, beam=2, max_steps=1)
        e = min(job["result"]["edges"], key=lambda e: e["rank"])
        client.post(
            "/api/curate",
            json={"job_id": job["job_id"], "edge_ids": [e["id"]], "action": "validate"},
        )
        text = client.get("/api/curate/export/pmechdb").text
    path = tmp_path / "curated.csv"
    path.write_text(text)
    raw = text.strip().strip('"')
    step = pmechdb.parse_record(raw, "x")
    assert len(step.arrows) == len(e["arrows"])


@requires("export")
@pytest.mark.parametrize("status", ["bogus"])
def test_export_rejects_unknown_status(synthetic_app, status):  # noqa: F811
    r = synthetic_app.get("/api/curate/export/pmechdb", params={"status": status})
    assert r.status_code == 422 and r.json()["error"]["code"] == "REQUEST_INVALID"
