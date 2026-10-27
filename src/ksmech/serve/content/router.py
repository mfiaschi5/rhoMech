"""Routes of the wiki: the index (with search), one page, and `llms.txt`. Public reads behind
the `wiki` capability (on by default, in every profile). `llms.txt` is registered before
`{slug}`, which would otherwise take it (a slug never holds a dot)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import PlainTextResponse

from ksmech.serve.content.service import WikiService
from ksmech.serve.core import schemas as S
from ksmech.serve.core.capabilities import require
from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.core.http import Services, Svc, check_id

router = APIRouter(prefix="/api", dependencies=[require("wiki")])


def _wiki(svc: Services) -> WikiService:
    if svc.wiki is None:
        raise ServeError(Code.FEATURE_DISABLED, "the wiki is not set up in this app")
    return svc.wiki


@router.get("/wiki", response_model=S.WikiIndexOut)
def wiki_index(svc: Svc, q: Annotated[str | None, Query(max_length=200)] = None) -> S.WikiIndexOut:
    """The pages by `order` then title; `q` keeps those whose title, summary or body holds it
    (ignoring case)."""
    return _wiki(svc).index(q)


@router.get("/wiki/llms.txt", response_class=PlainTextResponse)
def wiki_llms(request: Request, svc: Svc) -> PlainTextResponse:
    """The page index as plain text for LLM readers (llmstxt.org)."""
    return PlainTextResponse(_wiki(svc).llms_txt(str(request.base_url)))


@router.get("/wiki/{slug}", response_model=S.WikiPageOut)
def wiki_page(svc: Svc, slug: str) -> S.WikiPageOut:
    """One page: its metadata, the server-rendered HTML, its Markdown, scenes and headings."""
    return _wiki(svc).page(check_id("wiki_slug", slug))
