"""Server tools: who the API sees for this MCP server's token (always offered)."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ksmech.client import KsmechClient
from ksmech.mcp.tools import READ, Enabled, add, blocking, dumps


class MetaTools:
    """The server tools over one client."""

    def __init__(self, client: KsmechClient) -> None:
        self.client = client

    async def whoami(self) -> str:
        """The principal the ksmech API sees for this MCP server: its name, its role (reader,
        user, curator or admin: what the tools may do; a call beyond the role answers
        ROLE_INSUFFICIENT) and the 'via' the audit log records (mcp)."""
        who = await blocking(self.client.whoami)
        return dumps(who.model_dump(mode="json"))


def register(app: FastMCP, client: KsmechClient, enabled: Enabled) -> list[str]:
    return [add(app, MetaTools(client).whoami, "Who am I", READ)]
