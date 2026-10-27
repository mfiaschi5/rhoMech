"""The `ksmech` CLI: argument parsing, the compact tables and --json, the exit codes (0 ok,
1 an API error, 2 a usage error), against the synthetic app in process."""

from __future__ import annotations

import json

import pytest

from ksmech.client import KsmechClient, cli
from ksmech.client.commands import UsageError, parse_matrix
from ksmech.client.render import table
from tests.test_serve_fixtures_api import (  # noqa: F401 (fixtures)
    SN2_REACTANTS,
    no_xtb,
    synthetic_app,
    synthetic_root,
    synthetic_settings,
)

pytestmark = pytest.mark.serve


@pytest.fixture(scope="module")
def client(synthetic_app):  # noqa: F811
    return KsmechClient(http=synthetic_app, via="cli")


def run(client, capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv), client=client)
    out = capsys.readouterr()
    return code, out.out, out.err


# ---- parsing


def test_global_options_before_or_after_the_command(monkeypatch):
    monkeypatch.delenv("KSMECH_URL", raising=False)
    parser = cli.build_parser()
    a = parser.parse_args(["--json", "--url", "http://a:1", "predictors"])
    b = parser.parse_args(["predictors", "--json", "--url", "http://a:1"])
    assert (a.json, a.url) == (b.json, b.url) == (True, "http://a:1")
    c = parser.parse_args(["predictors"])
    assert (c.json, c.url, c.token) == (False, None, None)
    assert cli.client_of(c).url == "http://127.0.0.1:8000"


def test_url_and_token_from_the_environment(monkeypatch):
    monkeypatch.setenv("KSMECH_URL", "http://lab:8001")
    monkeypatch.setenv("KSMECH_TOKEN", "secret")
    c = cli.client_of(cli.build_parser().parse_args(["health"]))
    assert (c.url, c.token, c.via) == ("http://lab:8001", "secret", "cli")
    c = cli.client_of(cli.build_parser().parse_args(["health", "--token", "other"]))
    assert c.token == "other"


def test_predict_arguments():
    from ksmech.client.commands import predict_body

    a = cli.build_parser().parse_args(
        ["predict", "CO[-].CCl", "--predictor", "network", "--beam", "5", "--checkpoint", "r1/last"]
    )
    body = predict_body(a)
    assert body == {
        "reactants_smiles": ["CO[-].CCl"],
        "predictor": "network",
        "beam": 5,
        "checkpoint": {"run": "r1", "file": "last"},
    }


def test_parse_matrix():
    m = parse_matrix(["0,4,+1", "4 5 -1"], ["5,+2"])
    assert m == {"bonds": [[0, 4, 1], [4, 5, -1]], "lone": [[5, 2]]}
    for bonds, lone in ((["0,4"], None), (["a,b,c"], None), (None, None)):
        with pytest.raises(UsageError):
            parse_matrix(bonds, lone)


def test_table():
    text = table([{"a": 1.23456, "b": None}, {"a": True, "b": "x" * 100}], ["a", "b"])
    lines = text.splitlines()
    assert lines[0].split() == ["A", "B"] and lines[1].split() == ["1.235", "-"]
    assert lines[2].startswith("yes") and lines[2].endswith("...")
    assert table([], ["a"]) == "(none)"


# ---- against the app


def test_predictors_table_and_json(client, capsys):
    code, out, _ = run(client, capsys, "predictors")
    assert code == 0 and out.splitlines()[0].split()[:2] == ["NAME", "AVAILABLE"]
    code, out, _ = run(client, capsys, "predictors", "--json")
    assert code == 0 and [p["name"] for p in json.loads(out)][:3] == ["floor", "gbdt", "network"]


def test_predict_prints_the_pathways(client, capsys):
    argv = ["predict", *SN2_REACTANTS, "--beam", "2", "--max-steps", "1", "--pathways", "1"]
    code, out, _ = run(client, capsys, *argv)
    assert code == 0, out
    assert out.startswith("job j") and "done" in out.splitlines()[0]
    assert "pathway 1" in out and "ARROWS" in out
    code, out, _ = run(client, capsys, *argv, "--json")
    job = json.loads(out)
    assert job["status"] == "done" and job["result"]["pathways"]
    code, out, _ = run(client, capsys, "jobs", "list", "--kind", "predict")
    assert code == 0 and job["job_id"] in out
    code, out, _ = run(client, capsys, "jobs", "pathways", job["job_id"])
    assert code == 0 and out.splitlines()[0].split()[:2] == ["RANK", "ID"]


def test_normalize_and_check(client, capsys):
    code, out, _ = run(client, capsys, "--json", "normalize", *SN2_REACTANTS)
    assert code == 0
    state = json.loads(out)["state_id"]
    code, out, _ = run(client, capsys, "pool", "create", *SN2_REACTANTS)
    assert code == 0 and state in out
    code, out, err = run(client, capsys, "check", state, "--bond", "0,1,+1")
    assert code == 1 and "ok" in out and "issues" in out  # not a valid step: exit 1, no error


def test_molecule_commands(client, capsys):
    code, out, _ = run(client, capsys, "molecule", "props", "CO")
    assert code == 0 and "homo_ev" in out and "Q_MULLIKEN" in out
    code, out, _ = run(client, capsys, "molecule", "symmetry", "CO")
    assert code == 0 and "Cs" in out
    code, out, _ = run(client, capsys, "molecule", "orbitals", "CO", "--window", "1")
    assert code == 0 and len(out.strip().splitlines()) >= 3


def test_data_commands(client, capsys):
    assert run(client, capsys, "curate", "list")[:2] == (0, "(none)\n")
    code, out, _ = run(client, capsys, "browse", "status")
    assert code == 0 and "unavailable" in out
    code, out, _ = run(client, capsys, "models")
    assert code == 0 and "embedding:" in out
    code, out, _ = run(client, capsys, "meta")
    assert code == 0 and "workbench" in out


def test_exit_codes(client, capsys):
    code, out, err = run(client, capsys, "normalize", "C1CC")
    assert code == 1 and "INVALID_SMILES" in err and out == ""
    code, out, err = run(client, capsys, "--json", "normalize", "C1CC")
    assert code == 1 and json.loads(out)["error"]["code"] == "INVALID_SMILES"
    assert run(client, capsys, "curate", "get", "999")[0] == 1
    assert run(client, capsys, "no-such-command")[0] == 2
    assert run(client, capsys, "check", "s" + "0" * 20)[0] == 2  # no --bond / --lone
    assert run(client, capsys, "embedding", "create", "--run", "r")[0] == 2
    assert run(client, capsys, "--help")[0] == 0
