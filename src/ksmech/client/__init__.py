"""The HTTP client of the ksmech-serve API (`KsmechClient`) and the `ksmech` CLI over it.

It imports only httpx, pydantic and `ksmech.serve.core.schemas`: no web server, no RDKit, no
torch, so scripts and the MCP server (`ksmech.mcp`) can use it on any machine.
"""

from ksmech.client.client import DEFAULT_URL, KsmechClient
from ksmech.client.errors import KsmechAPIError, KsmechPending, KsmechTimeout

__all__ = ["DEFAULT_URL", "KsmechAPIError", "KsmechClient", "KsmechPending", "KsmechTimeout"]
