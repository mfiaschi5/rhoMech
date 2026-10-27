"""HTTP plumbing shared by every router: the services one app holds (`Services`, the `Svc`
dependency), the path-id types and their check (design 6.1: a mismatch is a 404 with the
endpoint's not-found code before any service runs), and small response helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Path, Request
from fastapi.responses import PlainTextResponse

from ksmech.serve.core.errors import Code, ServeError

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from ksmech.serve.browse.service import BrowseService
    from ksmech.serve.chem.chemistry import Chemistry
    from ksmech.serve.compute.worker import Scheduler
    from ksmech.serve.core.settings import ServeSettings
    from ksmech.serve.curate.service import CurateService
    from ksmech.serve.predict.jobs import JobRunner
    from ksmech.serve.predict.service import JobService
    from ksmech.serve.runtime.context import ApiContext

ID_RULES: dict[str, tuple[re.Pattern[str], Code]] = {
    "state_id": (re.compile(r"s[0-9a-f]{20}"), Code.POOL_NOT_FOUND),
    "node_id": (re.compile(r"p[0-9a-f]{20}"), Code.NODE_NOT_FOUND),
    "edge_id": (re.compile(r"[ec][0-9a-f]{16}"), Code.EDGE_NOT_FOUND),
    "job_id": (re.compile(r"j[0-9a-f]{16}"), Code.JOB_NOT_FOUND),
    "pathway_id": (re.compile(r"w[0-9a-f]{12}"), Code.PATHWAY_NOT_FOUND),
    "ref": (
        re.compile(r"pmechdb:pmechdb-[0-9a-f]{16}|curated:[0-9]+:[0-9]+"),
        Code.STEP_NOT_FOUND,
    ),
    "which": (re.compile(r"donor|acceptor"), Code.POOL_NOT_FOUND),
    "molecule_id": (re.compile(r"m[0-9a-f]{20}"), Code.MOLECULE_NOT_FOUND),
    "embedding_id": (re.compile(r"b[0-9a-f]{24}"), Code.EMBEDDING_NOT_FOUND),
}


def check_id(kind: str, value: str) -> str:
    """`value` if it matches the `kind` pattern (full match); else ServeError 404 with that
    kind's not-found code. No service or file is touched before this check."""
    pattern, code = ID_RULES[kind]
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        shown = value if isinstance(value, str) and len(value) <= 64 else f"{str(value)[:64]}..."
        raise ServeError(code, f"no such {kind.replace('_', ' ')}: {shown!r}", status=404)
    return value


@dataclass
class Services:
    settings: ServeSettings
    ctx: ApiContext
    scheduler: Scheduler
    chemistry: Chemistry
    engine: Engine
    runner: JobRunner
    jobs: JobService
    browse: BrowseService
    curate: CurateService
    frontend_built: bool = False


def _services(request: Request) -> Services:
    return request.app.state.services


Svc = Annotated[Services, Depends(_services)]


StateId = Annotated[str, Path(description="state id (s + 20 hex)")]


JobId = Annotated[str, Path(description="job id (j + 16 hex)")]


TokenId = Annotated[int, Path(ge=0)]


Mid = Annotated[int, Path(ge=0)]


MoleculeId = Annotated[str, Path(description="molecule id (m + 20 hex)")]


EmbeddingId = Annotated[str, Path(description="embedding trace id (b + 24 hex)")]


TRACE_CACHE = "private, max-age=86400"  # a trace id names immutable content


def blank(value: str | None) -> str | None:
    """Query strings: an empty value means "not given"."""
    return None if value is None or not value.strip() else value


def cube_response(text: str, filename: str) -> PlainTextResponse:
    headers = {"Content-Disposition": f"inline; filename={filename}"}
    return PlainTextResponse(text, headers=headers)
