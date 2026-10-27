"""Routes of embedding traces: submit a trace job, read its payload and its page."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, Response

from ksmech.serve.core import schemas as S
from ksmech.serve.core.capabilities import require
from ksmech.serve.core.http import (
    TRACE_CACHE,
    EmbeddingId,
    Svc,
    blank,
    check_id,
)

router = APIRouter(prefix="/api")


@router.post(
    "/embedding",
    response_model=S.JobCreatedOut,
    status_code=202,
    dependencies=[require("embedding")],
)
def create_embedding(svc: Svc, body: S.EmbeddingIn) -> S.JobCreatedOut:
    """Trace a pool state (or a SMILES) through a local checkpoint: an `embedding` job whose
    result is `EmbeddingResultOut`."""
    from ksmech.serve.embedding.service import submit

    if body.state_id is not None:
        check_id("state_id", body.state_id)
    return submit(svc.ctx, svc.runner, body)


@router.get("/embedding/{embedding_id}", response_class=Response)
def get_embedding(svc: Svc, embedding_id: EmbeddingId) -> Response:
    """The stored payload of a trace (what the page draws), as JSON."""
    from ksmech.serve.embedding.service import payload_bytes

    data = payload_bytes(svc.settings, check_id("embedding_id", embedding_id))
    headers = {"Cache-Control": TRACE_CACHE}
    return Response(data, media_type="application/json", headers=headers)


@router.get("/embedding/{embedding_id}/page", response_class=HTMLResponse)
def get_embedding_page(
    svc: Svc, embedding_id: EmbeddingId, theme: str | None = None
) -> HTMLResponse:
    """The self-contained trace page (no external script), for an iframe or a tab; `theme`
    light | dark sets its colour scheme, else it follows the browser's."""
    from ksmech.serve.embedding.service import page_html

    html = page_html(svc.settings, check_id("embedding_id", embedding_id), blank(theme))
    return HTMLResponse(html, headers={"Cache-Control": TRACE_CACHE})
