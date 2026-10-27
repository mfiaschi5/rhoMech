"""`ksmech-mcp`: tools registered per capability set (a public server offers no dataset,
curation or ML tool), their annotations, and round trips through the SDK's in-memory client
session against the synthetic app (normalize, predict_mechanism, step readings, resources)."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

from ksmech.client import KsmechClient
from ksmech.mcp import server as mcp_server
from tests.test_client_api import public_app  # noqa: F401 (fixture)
from tests.test_serve_fixtures_api import (  # noqa: F401 (fixtures)
    SN2_REACTANTS,
    no_xtb,
    synthetic_app,
    synthetic_root,
    synthetic_settings,
)

pytestmark = pytest.mark.serve

ALL_CAPS = ("predict", "correct", "workbench", "browse", "curate", "ml")
DATA_TOOLS = {
    "list_checkpoints",
    "list_training_runs",
    "training_run_metrics",
    "list_reports",
    "dataset_index_status",
    "search_dataset_steps",
    "view_dataset_step",
    "list_curated_mechanisms",
    "get_curated_mechanism",
}
ALWAYS = {"whoami", "normalize_reactants", "depict_svg", "get_job"}


def meta_of(on: set[str], profile: str | None = "lab") -> dict:
    caps = {name: {"enabled": name in on, "reason": None} for name in ALL_CAPS}
    meta = {"app_version": "0.1.0", "spec_label": "v9", "chemistry": "synthetic"}
    return meta | {"capabilities": caps} | ({"profile": profile} if profile else {})


def tools_of(meta: dict) -> dict:
    app = mcp_server.build_server(KsmechClient("http://unused.test"), meta)
    return {t.name: t for t in anyio.run(app.list_tools)}


def test_lab_offers_every_tool():
    tools = tools_of(meta_of(set(ALL_CAPS)))
    assert set(tools) >= DATA_TOOLS | ALWAYS
    assert {"predict_mechanism", "check_step", "molecule_properties"} <= set(tools)


def test_public_offers_no_dataset_curation_or_ml_tool():
    public = {"predict", "correct", "workbench"}
    tools = tools_of(meta_of(public, profile="public"))
    assert not set(tools) & DATA_TOOLS
    assert "predict_mechanism" in tools and "orbital_summary" in tools


def test_capabilities_gate_each_area():
    tools = set(tools_of(meta_of(set())))
    assert tools == ALWAYS
    assert "check_step" not in tools_of(meta_of({"predict"}))
    assert "check_step" in tools_of(meta_of({"correct"}))


def test_unlisted_capabilities_follow_the_profile():
    legacy = meta_of({"predict"}, profile=None)
    del legacy["capabilities"]["ml"]
    assert "list_reports" in tools_of(legacy)  # a server without profiles gated nothing
    profiled = meta_of({"predict"}, profile="lab")
    del profiled["capabilities"]["ml"]
    assert "list_reports" not in tools_of(profiled)


def test_annotations_and_descriptions():
    tools = tools_of(meta_of(set(ALL_CAPS)))
    for name, tool in tools.items():
        hints = tool.annotations
        assert hints.openWorldHint is False and hints.destructiveHint is False, name
        assert hints.title and len(tool.description) > 80, name
    assert tools["predict_mechanism"].annotations.readOnlyHint is False
    reads = [t for n, t in tools.items() if n != "predict_mechanism"]
    assert all(t.annotations.readOnlyHint and t.annotations.idempotentHint for t in reads)
    props = tools["predict_mechanism"].inputSchema["properties"]
    assert props["beam"]["anyOf"][0]["maximum"] == 10 and "description" in props["reactants"]


def test_main_fails_fast_without_the_api(capsys):
    assert mcp_server.main(["--url", "http://127.0.0.1:9", "--timeout", "2"]) == 1
    assert "cannot read http://127.0.0.1:9/api/meta" in capsys.readouterr().err


def test_instructions_name_the_profile_and_the_read_only_rule():
    app = mcp_server.build_server(KsmechClient("http://unused.test"), meta_of({"ml"}, "public"))
    text = app.instructions
    assert "profile public" in text and "capabilities on: ml" in text
    assert "no write tools" in text and "review queue" in text


# ---- round trips against the synthetic app


def session_run(client: KsmechClient, body: Callable[[ClientSession], Awaitable]) -> object:
    """Run `body(session)` on an in-memory MCP session with the server of `client`."""
    app = mcp_server.build_server(client)

    async def main():
        async with create_connected_server_and_client_session(app) as session:
            return await body(session)

    return anyio.run(main)


async def call(session: ClientSession, name: str, **arguments) -> dict | list:
    res = await session.call_tool(name, arguments)
    assert not res.isError, res.content
    assert len(res.content) == 1 and res.structuredContent is None
    return json.loads(res.content[0].text)


@pytest.fixture(scope="module")
def client(synthetic_app):  # noqa: F811
    return KsmechClient(http=synthetic_app, via="mcp")


def test_round_trip_normalize_and_predict(client):
    async def body(session):
        listed = {t.name for t in (await session.list_tools()).tools}
        norm = await call(session, "normalize_reactants", reactants=SN2_REACTANTS)
        pred = await call(
            session, "predict_mechanism", reactants=SN2_REACTANTS, beam=2, max_steps=1
        )
        edge = pred["network"]["pathways"][0]["steps"][0]["edge"]
        step = await call(session, "step_readings", job_id=pred["job_id"], edge_id=edge)
        job = await call(session, "get_job", job_id=pred["job_id"], n_pathways=1)
        who = await call(session, "whoami")
        return listed, norm, pred, step, job, who

    listed, norm, pred, step, job, who = session_run(client, body)
    assert who["via"] == "mcp"
    assert listed >= DATA_TOOLS  # the synthetic app is a lab server
    assert norm["ok"] is True and norm["pool_key"] == "[CH3][Cl]|0.[CH3][O]|-1"
    assert pred["status"] == "done" and pred["provenance"]["predictor"] == "floor"
    first = pred["network"]["pathways"][0]["steps"][0]
    assert first["from"] and first["to"] and first["arrows"]
    assert step["edge"] == first["edge"] and step["matrix"] and "source_state_id" in step
    assert len(job["network"]["pathways"]) == 1


def test_round_trip_check_step_and_errors(client):
    async def body(session):
        pred = await call(
            session, "predict_mechanism", reactants=SN2_REACTANTS, beam=1, max_steps=1
        )
        job = client.job(pred["job_id"])
        edge = job.result.edges[0]
        bonds = [list(b) for b in edge.matrix.bonds]
        lone = [list(x) for x in edge.matrix.lone]
        check = await call(
            session, "check_step", state_id=edge.source_state_id, bonds=bonds, lone=lone
        )
        bad = await session.call_tool("normalize_reactants", {"reactants": ["C1CC"]})
        return check, bad

    check, bad = session_run(client, body)
    assert check["ok"] is True and check["arrows"]
    assert bad.isError and "INVALID_SMILES" in bad.content[0].text


def test_round_trip_workbench_ml_and_resources(client):
    async def body(session):
        props = await call(session, "molecule_properties", smiles="CO")
        sym = await call(session, "molecule_symmetry", smiles="CO")
        runs = await call(session, "list_training_runs")
        resources = [str(r.uri) for r in (await session.list_resources()).resources]
        ckpts = await session.read_resource("ksmech://checkpoints")
        return props, sym, runs, resources, ckpts

    props, sym, runs, resources, ckpts = session_run(client, body)
    assert props["global"]["n_atoms"] == 6 and sym["point_group"] == "Cs"
    assert runs == [] and "ksmech://checkpoints" in resources
    assert json.loads(ckpts.contents[0].text) == []


def test_a_real_public_server_offers_no_data_tool(public_app):  # noqa: F811
    app = mcp_server.build_server(KsmechClient(http=public_app, via="mcp"))
    tools = {t.name for t in anyio.run(app.list_tools)}
    assert not tools & DATA_TOOLS and {"predict_mechanism", "whoami"} <= tools
    assert "profile public" in app.instructions
