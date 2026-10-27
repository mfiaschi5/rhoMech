"""The MCP tools of `ksmech-mcp`, one module per area (chem, predict, ml, datasets, curate).

Each area's `register(app, client, enabled)` adds its tools to the FastMCP `app` when
`enabled(capability)` says the API serves them, and returns the names it added. Every tool
calls the API through `ksmech.client.KsmechClient` only (in a worker thread, so a long wait
never blocks the server's event loop) and answers compact JSON text (an LLM's context is small:
summaries, not raw dumps; `detail=true` keeps more).
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import Any

import anyio.to_thread
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

Enabled = Callable[[str], bool]

READ = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
"""A tool that only reads (it may make the server cache what it computed, never more)."""

COMPUTE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
)
"""A tool that submits compute (a job is created and kept), never changes or deletes data."""


def dumps(value: Any) -> str:
    """Compact JSON (no indentation), the text every tool answers."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)


async def blocking(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """`fn(*args, **kwargs)` in a worker thread (the client is synchronous)."""
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


def add(app: FastMCP, fn: Callable[..., Any], title: str, annotations: ToolAnnotations) -> str:
    """Register `fn` (its docstring is the description the LLM reads) as an unstructured
    tool (the compact JSON text only, no second structured copy) and return its name."""
    hints = annotations.model_copy(update={"title": title})
    app.add_tool(fn, name=fn.__name__, title=title, annotations=hints, structured_output=False)
    return fn.__name__
