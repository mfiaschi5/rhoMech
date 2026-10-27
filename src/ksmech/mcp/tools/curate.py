"""Curation tools: list and read the curated mechanisms (read only; the `curate` capability).
Saving, validating or editing a mechanism is not offered: such changes will go through a
review queue."""

from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ksmech.client import KsmechClient
from ksmech.client import summary as sm
from ksmech.mcp.tools import READ, Enabled, add, blocking, dumps


class CurateTools:
    """The curation read tools over one client."""

    def __init__(self, client: KsmechClient) -> None:
        self.client = client

    async def list_curated_mechanisms(
        self,
        status: Annotated[
            str | None, Field(description="predicted | validated | corrected | rejected.")
        ] = None,
        predictor: Annotated[
            str | None, Field(description="floor | gbdt | network | user.")
        ] = None,
        stale: Annotated[
            bool | None, Field(description="Only mechanisms whose physics version changed.")
        ] = None,
        limit: Annotated[int, Field(ge=1, le=200)] = 20,
        offset: Annotated[int, Field(ge=0)] = 0,
    ) -> str:
        """The mechanisms chemists saved from predictions (newest first): id, status,
        reactants, number of steps, the predictor, and whether a newer physics version made
        them stale."""
        rows = await blocking(
            self.client.curate_list,
            status=status,
            predictor=predictor,
            stale=stale,
            limit=limit,
            offset=offset,
        )
        return dumps([sm.mechanism_row(m) for m in rows])

    async def get_curated_mechanism(
        self,
        mechanism_id: Annotated[int, Field(ge=0, description="The mechanism id.")],
        detail: Annotated[bool, Field(description="Each step's full step document.")] = False,
    ) -> str:
        """One curated mechanism: its reactants, conditions, status and notes, and its steps
        in order (reactant and product pool keys, whether the chemist corrected the step,
        the last recheck's top-1 hit)."""
        m = await blocking(self.client.curate_get, mechanism_id)
        return dumps(sm.mechanism_summary(m, detail=detail))


def register(app: FastMCP, client: KsmechClient, enabled: Enabled) -> list[str]:
    if not enabled("curate"):
        return []
    t = CurateTools(client)
    return [
        add(app, t.list_curated_mechanisms, "List curated mechanisms", READ),
        add(app, t.get_curated_mechanism, "Read a curated mechanism", READ),
    ]
