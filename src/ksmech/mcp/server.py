"""`ksmech-mcp`: an MCP server (FastMCP, the official `mcp` SDK) over the ksmech-serve API.

    ksmech-mcp [--url URL] [--token TOKEN] [--transport stdio|http] [--host H] [--port P]

It calls the API only through `ksmech.client.KsmechClient`, with `X-Ksmech-Via: mcp` and the
bearer token of `--token` / `KSMECH_TOKEN`, so an LLM gets exactly the permissions, quotas and
audit of that token, as the web UI and the `ksmech` CLI do. At startup it reads `/api/meta` and
registers only the tools whose capability the server enables (`AREAS`): a `public` server never
offers the dataset, curation or ML tools. A capability the server does not list counts as on
only when the server predates profiles (its meta has no `profile`): such a server gated nothing.

Transports: stdio (the default; Claude Code or Claude Desktop starts the process) and streamable
HTTP (`--transport http`, endpoint `http://HOST:PORT/mcp`, bound to 127.0.0.1 by default with
FastMCP's DNS-rebinding check). Over HTTP every MCP client acts with the server's one token.
Logs go to stderr (stdout is the stdio transport).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Mapping
from typing import Any

from mcp.server.fastmcp import FastMCP

from ksmech.client import DEFAULT_URL, KsmechAPIError, KsmechClient
from ksmech.mcp.tools import Enabled, chem, curate, datasets, meta, ml, predict

log = logging.getLogger(__name__)

NAME = "ksmech"
DEFAULT_PORT = 8765
AREAS: tuple[tuple[str, Callable[..., list[str]]], ...] = (
    ("meta", meta.register),
    ("chem", chem.register),
    ("predict", predict.register),
    ("ml", ml.register),
    ("datasets", datasets.register),
    ("curate", curate.register),
)
"""(area, register); each register gates its tools on capabilities: meta (whoami) always,
chem `workbench` (normalize and depict_svg always), predict `predict` / `correct` (get_job
always), ml `ml`, datasets `browse`, curate `curate`."""

INSTRUCTIONS = """ksmech predicts organic reaction mechanisms as sequences of elementary steps.
Each step is a reaction matrix R (bond-order changes of +-1 on atom pairs, lone-pair changes
of +-2 on atoms) on the 'pool atoms' of its source state: the atom indices of the state's
atom-mapped SMILES, explicit H included. Curved arrows are a reading of R ('lp(3)->3-5': the
lone pair of atom 3 forms bond 3-5). Typical use: normalize_reactants to check the input,
predict_mechanism to get ranked pathways (only the 'network' predictor gives probabilities;
'floor' ranks steps by the Klopman-Salem gap factor), step_readings for the orbital picture
of one step, check_step to test a step of your own. Molecule tools compute GFN2-xTB properties
on first use (a few seconds to minutes) and cache them. Outputs are compact summaries; pass
detail=true for more. There are no write tools for curated mechanisms or datasets: such
changes will go through a review queue."""


def enabled_from(meta: Mapping[str, Any]) -> Enabled:
    """`enabled(capability)` of an `/api/meta` answer (raw JSON, so new names are kept)."""
    caps = meta.get("capabilities") or {}
    profiled = "profile" in meta

    def enabled(name: str) -> bool:
        if name in caps:
            return bool((caps[name] or {}).get("enabled"))
        return not profiled

    return enabled


def instructions_of(meta: Mapping[str, Any]) -> str:
    """`INSTRUCTIONS` and a line on the server: versions, profile, capabilities on."""
    caps = meta.get("capabilities") or {}
    on = [name for name, c in caps.items() if (c or {}).get("enabled")]
    head = (
        f"Server {meta.get('app_version', '?')}, spec {meta.get('spec_label', '?')}, "
        f"chemistry {meta.get('chemistry', '?')}, profile {meta.get('profile') or 'lab'}; "
        f"capabilities on: {', '.join(on) or 'none'}."
    )
    return f"{INSTRUCTIONS}\n{head}"


def build_server(
    client: KsmechClient, meta: Mapping[str, Any] | None = None, **settings: Any
) -> FastMCP:
    """The FastMCP server of `client`'s API: the tools its `/api/meta` capabilities enable
    (`meta` is read from the API when not given). `settings` go to FastMCP (host, port)."""
    meta = client.get_json("/api/meta") if meta is None else meta
    enabled = enabled_from(meta)
    app = FastMCP(NAME, instructions=instructions_of(meta), **settings)
    names: list[str] = []
    for area, register in AREAS:
        added = register(app, client, enabled)
        log.info("ksmech-mcp: area %s: %s", area, ", ".join(added) or "off")
        names += added
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-mcp", description=__doc__.split("\n\n")[0])
    url_help = f"ksmech-serve API root (env KSMECH_URL, default {DEFAULT_URL})"
    parser.add_argument("--url", default=None, help=url_help)
    parser.add_argument("--token", default=None, help="bearer token (env KSMECH_TOKEN)")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP transport: bind address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP transport: port")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per API request")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s"
    )
    url = args.url or os.environ.get("KSMECH_URL") or DEFAULT_URL
    token = args.token or os.environ.get("KSMECH_TOKEN") or None
    client = KsmechClient(url, token, via="mcp", timeout=args.timeout)
    try:
        app = build_server(client, host=args.host, port=args.port)
    except KsmechAPIError as exc:
        print(f"ksmech-mcp: cannot read {url}/api/meta: {exc}", file=sys.stderr)
        client.close()
        return 1
    try:
        app.run("stdio" if args.transport == "stdio" else "streamable-http")
    finally:
        client.close()
    return 0
