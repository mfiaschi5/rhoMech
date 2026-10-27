"""`ksmech`: the command line of the ksmech-serve API (through `KsmechClient`, so it gets the
same permissions, quotas and audit as the web UI; requests carry `X-Ksmech-Via: cli`).

    ksmech [--url URL] [--token TOKEN] [--json] COMMAND ...
    ksmech health | meta | whoami | predictors | models
    ksmech normalize SMILES... [--stoichiometry N...]
    ksmech predict SMILES... [--predictor floor|gbdt|network] [--beam 5] [--max-steps 8]
                   [--checkpoint RUN[/FILE] [--source S]] [--no-wait] [--wait 600]
    ksmech jobs list|get|wait|cancel|pathways ...
    ksmech check STATE_ID --bond I,J,DELTA ... --lone I,DELTA ...
    ksmech pool create|get|svg ...
    ksmech molecule create|get|props|symmetry|orbitals SMILES_OR_ID ...
    ksmech depict2d MOLFILE [--mode clean|match3d] [--style textbook|perspective]
    ksmech ml runs | ml run SOURCE RUN [--detail] | ml reports
    ksmech embedding create --run RUN (--state-id S | --smiles S) | embedding get ID
    ksmech curate list [...] | curate get ID
    ksmech browse status | browse search [QUERY] [...] | browse step REF

`--url` defaults to `KSMECH_URL` (else http://127.0.0.1:8000), `--token` to `KSMECH_TOKEN`.
`--json` prints the API's answer as JSON; otherwise a compact table. The global options may
come before or after the command. Exit codes: 0 ok, 1 an API error (or a failed job, an invalid
step), 2 a usage error.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

from ksmech.client import commands as C
from ksmech.client import commands_data as D
from ksmech.client.client import DEFAULT_URL, KsmechClient
from ksmech.client.errors import KsmechAPIError
from ksmech.client.render import to_json

PROG = "ksmech"


def _common(parser: argparse.ArgumentParser, top: bool) -> None:
    """The global options: real defaults on the top parser, SUPPRESS on the leaves (so an
    option given after the command overrides, and an absent one keeps the top value)."""

    def d(value):
        return value if top else argparse.SUPPRESS

    url_help = f"API root (env KSMECH_URL, default {DEFAULT_URL})"
    parser.add_argument("--url", default=d(None), help=url_help)
    parser.add_argument("--token", default=d(None), help="bearer token (env KSMECH_TOKEN)")
    parser.add_argument(
        "--json", action="store_true", default=d(False), help="print the API answer as JSON"
    )
    parser.add_argument(
        "--timeout", type=float, default=d(60.0), help="seconds per request (default 60)"
    )


class _Parsers:
    """Adds leaf commands with the global options and their handler."""

    def __init__(self, sub) -> None:
        self.sub = sub

    def add(self, name: str, handler: Callable, help: str) -> argparse.ArgumentParser:
        p = self.sub.add_parser(name, help=help, description=help)
        _common(p, top=False)
        p.set_defaults(handler=handler)
        return p

    def group(self, name: str, help: str) -> _Parsers:
        p = self.sub.add_parser(name, help=help, description=help)
        return _Parsers(p.add_subparsers(dest=f"{name}_command", required=True, metavar="ACTION"))


def _job_view(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pathways", type=int, default=5, help="pathways shown (default 5)")
    p.add_argument("--detail", action="store_true", help="matrices and orbital readings too")


def _wait(p: argparse.ArgumentParser, default: float = 600.0) -> None:
    p.add_argument("--wait", type=float, default=default, help=f"seconds (default {default:g})")


def _reactants(p: argparse.ArgumentParser) -> None:
    p.add_argument("smiles", nargs="+", help="reactant SMILES (one field each, '.' allowed)")
    p.add_argument("--stoichiometry", type=int, nargs="+", default=None, metavar="N")


def _add_predict(r: _Parsers) -> None:
    p = r.add("predict", C.cmd_predict, "predict a mechanism and wait for the job")
    _reactants(p)
    p.add_argument("--predictor", default=None, help="floor | gbdt | network | user")
    p.add_argument("--beam", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--conditions", default=None, help="recorded only")
    p.add_argument("--checkpoint", default=None, metavar="RUN[/FILE]", help="network only")
    p.add_argument("--source", default=None, help="the checkpoint's data source")
    p.add_argument("--no-wait", action="store_true", help="print the job id and return")
    _wait(p)
    _job_view(p)


def _add_jobs(r: _Parsers) -> None:
    g = r.group("jobs", "list, read, wait for and cancel jobs")
    p = g.add("list", C.cmd_jobs_list, "recent jobs")
    p.add_argument("--kind", default=None)
    p.add_argument("--status", default=None)
    p.add_argument("--limit", type=int, default=50)
    for name, handler, text in (
        ("get", C.cmd_jobs_get, "one job and its top pathways"),
        ("wait", C.cmd_jobs_wait, "wait until the job finishes"),
    ):
        p = g.add(name, handler, text)
        p.add_argument("job_id")
        _job_view(p)
        if name == "wait":
            _wait(p)
    p = g.add("cancel", C.cmd_jobs_cancel, "cancel a job")
    p.add_argument("job_id")
    p.add_argument("--force", action="store_true", help="kill the running task")
    p = g.add("pathways", C.cmd_jobs_pathways, "the pathways of a job's network")
    p.add_argument("job_id")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--include-pruned", action="store_true")


def _add_chem(r: _Parsers) -> None:
    p = r.add("normalize", C.cmd_normalize, "check and canonicalize reactants")
    _reactants(p)
    p = r.add("check", C.cmd_check, "validate a reaction matrix on a pool state")
    p.add_argument("state_id")
    p.add_argument("--bond", action="append", metavar="I,J,DELTA", help="repeatable")
    p.add_argument("--lone", action="append", metavar="I,DELTA", help="repeatable")
    p.add_argument("--detail", action="store_true")
    g = r.group("pool", "pool states")
    _reactants(g.add("create", C.cmd_pool_create, "build and save a pool state"))
    g.add("get", C.cmd_pool_get, "one pool state").add_argument("state_id")
    p = g.add("svg", C.cmd_pool_svg, "the server's SVG depiction of a pool state")
    p.add_argument("state_id")
    p.add_argument("--out", default=None, help="write the SVG to this file")
    p = r.add("depict2d", C.cmd_depict2d, "a 2D drawing of a molfile")
    p.add_argument("molfile")
    p.add_argument("--mode", choices=("clean", "match3d"), default=None)
    p.add_argument("--style", choices=("textbook", "perspective"), default=None)


def _add_molecule(r: _Parsers) -> None:
    g = r.group("molecule", "the molecule workbench (SMILES or a molecule id)")
    p = g.add("create", C.cmd_molecule_create, "register a molecule (featurization starts)")
    p.add_argument("smiles", nargs="?", default=None)
    p.add_argument("--molblock", default=None, metavar="FILE")
    g.add("get", C.cmd_molecule_get, "a molecule's status").add_argument("molecule")
    p = g.add("props", C.cmd_molecule_props, "GFN2 properties: frontier levels, atoms")
    p.add_argument("molecule")
    p.add_argument("--detail", action="store_true")
    p = g.add("symmetry", C.cmd_molecule_symmetry, "the point group")
    p.add_argument("molecule")
    p.add_argument("--tolerance", type=float, default=None)
    p.add_argument("--detail", action="store_true")
    p = g.add("orbitals", C.cmd_molecule_orbitals, "orbitals around the frontier")
    p.add_argument("molecule")
    p.add_argument("--set", default=None, help="canonical | ibo | pm_mulliken")
    p.add_argument("--start", default=None, help="cholesky | canonical")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--detail", action="store_true")


def _add_ml(r: _Parsers) -> None:
    r.add("models", D.cmd_models, "the local checkpoints")
    g = r.group("ml", "training runs and reports")
    g.add("runs", D.cmd_ml_runs, "the training runs")
    p = g.add("run", D.cmd_ml_run, "one run and its metrics per epoch")
    p.add_argument("source")
    p.add_argument("run")
    p.add_argument("--detail", action="store_true", help="up to 60 epochs, the model config")
    g.add("reports", D.cmd_ml_reports, "the evaluation reports")
    g = r.group("embedding", "embedding traces through a checkpoint")
    p = g.add("create", D.cmd_embedding_create, "trace a pool state or a SMILES")
    p.add_argument("--run", required=True)
    p.add_argument("--file", default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--state-id", default=None)
    p.add_argument("--smiles", default=None)
    _wait(p)
    g.add("get", D.cmd_embedding_get, "a stored trace").add_argument("embedding_id")


def _add_data(r: _Parsers) -> None:
    g = r.group("curate", "curated mechanisms (read only)")
    p = g.add("list", D.cmd_curate_list, "curated mechanisms")
    p.add_argument("--status", default=None)
    p.add_argument("--predictor", default=None)
    p.add_argument("--stale", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--offset", type=int, default=0)
    p = g.add("get", D.cmd_curate_get, "one mechanism and its steps")
    p.add_argument("mechanism_id", type=int)
    p.add_argument("--detail", action="store_true")
    g = r.group("browse", "the dataset browser")
    g.add("status", D.cmd_browse_status, "the search index")
    p = g.add("search", D.cmd_browse_search, "search dataset and curated steps")
    p.add_argument("query", nargs="?", default="", help="SMILES or SMARTS")
    p.add_argument("--mode", default=None, help="exact | substructure | ...")
    p.add_argument("--side", default=None, help="reactants | products | either")
    p.add_argument("--source", action="append", default=None, help="repeatable")
    p.add_argument("--status", default=None)
    p.add_argument("--resonance", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)
    p = g.add("step", D.cmd_browse_step, "one step with its dataset arrows")
    p.add_argument("ref")
    p.add_argument("--detail", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    _common(parser, top=True)
    r = _Parsers(parser.add_subparsers(dest="command", required=True, metavar="COMMAND"))
    r.add("health", C.cmd_health, "server health")
    r.add("meta", C.cmd_meta, "versions, profile and capabilities")
    r.add("whoami", C.cmd_whoami, "the principal the server sees for this token")
    r.add("predictors", C.cmd_predictors, "the predictors and whether they are available")
    for add in (_add_predict, _add_jobs, _add_chem, _add_molecule, _add_ml, _add_data):
        add(r)
    return parser


def client_of(args: argparse.Namespace) -> KsmechClient:
    url = args.url or os.environ.get("KSMECH_URL") or DEFAULT_URL
    token = args.token or os.environ.get("KSMECH_TOKEN") or None
    return KsmechClient(url, token, via="cli", timeout=args.timeout)


def run(args: argparse.Namespace, client: KsmechClient) -> int:
    """Run the parsed command on `client`; print the answer; the exit code."""
    try:
        out = args.handler(client, args)
    except C.UsageError as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 2
    except KsmechAPIError as exc:
        if args.json:
            print(to_json({"error": exc.to_dict()}))
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 1
    print(to_json(out.data) if args.json else out.text)
    return out.code


def main(argv: list[str] | None = None, client: KsmechClient | None = None) -> int:
    """The CLI; `client` replaces the one built from the options (tests)."""
    try:
        args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:  # argparse: 2 on a usage error, 0 for --help
        return int(exc.code or 0)
    own = client is None
    client = client or client_of(args)
    try:
        return run(args, client)
    finally:
        if own:
            client.close()
