"""Dataset tools: the browse index, step search and the view of one dataset step (read only;
the `browse` capability). No tool writes a dataset: changes will go through a review queue."""

from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ksmech.client import KsmechClient
from ksmech.client import summary as sm
from ksmech.mcp.tools import READ, Enabled, add, blocking, dumps


class DatasetTools:
    """The dataset browser tools over one client."""

    def __init__(self, client: KsmechClient) -> None:
        self.client = client

    async def dataset_index_status(self) -> str:
        """The state of the search index over PMechDB (ready, building, missing, stale,
        unavailable), how many dataset steps it holds and how many curated steps exist."""
        return dumps(sm.index_summary(await blocking(self.client.browse_status)))

    async def search_dataset_steps(
        self,
        query: Annotated[
            str, Field(description="SMILES (exact mode) or SMARTS (substructure); '' lists all.")
        ] = "",
        mode: Annotated[str, Field(description="exact | substructure.")] = "exact",
        side: Annotated[str, Field(description="reactants | products | either.")] = "either",
        sources: Annotated[
            list[str] | None, Field(description="pmechdb and/or user_curated (default both).")
        ] = None,
        resonance: Annotated[
            bool | None, Field(description="Only (true) or no (false) resonance steps.")
        ] = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 10,
        offset: Annotated[int, Field(ge=0)] = 0,
    ) -> str:
        """Search the elementary steps of PMechDB and of the curated mechanisms by a species
        (exact SMILES) or a substructure (SMARTS) on the reactant or product side. Each
        result has its ref (for view_dataset_step), the reaction SMILES, the number of
        arrows, the PMechDB arrow code and its split membership."""
        out = await blocking(
            self.client.browse_search,
            query,
            mode=mode,
            side=side,
            source=sources,
            resonance=resonance,
            limit=limit,
            offset=offset,
        )
        return dumps(sm.search_summary(out))

    async def view_dataset_step(
        self,
        ref: Annotated[
            str, Field(description="A step ref: 'pmechdb:pmechdb-<16 hex>' or 'curated:<m>:<i>'.")
        ],
        detail: Annotated[bool, Field(description="Also the pool keys and the step id.")] = False,
    ) -> str:
        """One dataset step: the reaction SMILES, its dataset arrows (source and sink kind
        with atom-map numbers), the PMechDB arrow code, and the curated mechanism it belongs
        to, if any (no drawing: use depict_svg on the reactants)."""
        view = await blocking(self.client.browse_step, ref)
        return dumps(sm.step_view_summary(view, detail=detail))


def register(app: FastMCP, client: KsmechClient, enabled: Enabled) -> list[str]:
    if not enabled("browse"):
        return []
    t = DatasetTools(client)
    return [
        add(app, t.dataset_index_status, "Dataset index status", READ),
        add(app, t.search_dataset_steps, "Search dataset steps", READ),
        add(app, t.view_dataset_step, "View a dataset step", READ),
    ]
