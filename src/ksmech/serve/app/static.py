"""Serving the built frontend (design 12): `apps/ui/dist` mounted at `/` after the `/api` routers.

`mount_frontend(app, dist)` registers `.wasm` as `application/wasm` (RDKit.js compiles its wasm
with `instantiateStreaming`, which refuses any other type) and mounts
`SPAStaticFiles(directory=dist, html=True)`: a path the build does not hold answers
`index.html` (design 6.1, "every other path serves the built SPA"), so a typed or bookmarked
`/predict` or `/curate/12` opens the app. Unknown `/api/...` paths and missing assets (a last
segment with a dot, `/assets/missing.js`) keep their JSON 404. Without a build, `/` serves a
short page with the npm commands that build it, and `/api/meta.frontend_built` is false. Dev
mode (`--dev`) mounts no files and `/` points at Vite on :5173, which proxies `/api` here (no
CORS needed). `--no-ui` serves the API only.
"""

from __future__ import annotations

import html
import mimetypes
from pathlib import Path, PurePath

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

BUILD_COMMAND = "npm --prefix apps/ui ci && npm --prefix apps/ui run build"
NOT_BUILT = f"frontend not built: run `{BUILD_COMMAND}`"
DEV_BANNER = "dev: use Vite on :5173"
NO_UI = "ksmech-serve runs with --no-ui: only the API under /api is served"
SPA_INDEX = "index.html"

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ksmech-serve</title>
<style>
:root {{ --bg: #ffffff; --fg: #1d2330; --muted: #5c6475; --code: #eef1f6; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #14171d; --fg: #e4e7ee; --muted: #9aa3b5; --code: #232833; }}
}}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif;
       margin: 0; padding: 2rem 16px; }}
main {{ max-width: 46rem; margin: 0 auto; }}
code {{ background: var(--code); padding: 0.1rem 0.3rem; border-radius: 4px;
        overflow-wrap: anywhere; }}
p.muted {{ color: var(--muted); }}
</style>
</head>
<body>
<main data-testid="serve-placeholder" data-reason="{reason}">
<h1>ksmech-serve</h1>
<p>{message}</p>
<p class="muted">The API is up: <a href="/api/health">/api/health</a>,
<a href="/api/meta">/api/meta</a>, <a href="/api/docs">/api/docs</a>.</p>
</main>
</body>
</html>
"""


def _code_spans(text: str) -> str:
    """Escape `text`, then render its backtick spans as <code>."""
    parts = html.escape(text).split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def placeholder_page(message: str, reason: str) -> str:
    return _PAGE.format(message=_code_spans(message), reason=html.escape(reason))


def mount_placeholder(app: FastAPI, message: str, reason: str) -> None:
    """`/` (and `/index.html`) answer a small page with `message`."""
    page = placeholder_page(message, reason)

    async def placeholder() -> HTMLResponse:
        return HTMLResponse(page)

    for path in ("/", "/index.html"):
        app.add_api_route(path, placeholder, methods=["GET"], include_in_schema=False)


def spa_fallback(path: str) -> bool:
    """Whether a path the build does not hold is an app route (served `index.html`): not under
    `api/`, and its last segment has no dot (a missing asset keeps its 404)."""
    parts = PurePath(path).parts
    if not parts or parts[0] == "api":
        return False
    return "." not in parts[-1]


class SPAStaticFiles(StaticFiles):
    """`StaticFiles` whose 404 on an app route (`spa_fallback`) serves the SPA's index."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or not spa_fallback(path):
                raise
            return await super().get_response(SPA_INDEX, scope)


def mount_frontend(app: FastAPI, dist: Path) -> bool:
    """Mount the built SPA at `/` (call it after every `/api` router). Returns True when
    `dist/index.html` exists; otherwise `/` serves the "frontend not built" page."""
    mimetypes.add_type("application/wasm", ".wasm")
    dist = Path(dist)
    if not (dist / "index.html").is_file():
        mount_placeholder(app, NOT_BUILT, "not-built")
        return False
    app.mount("/", SPAStaticFiles(directory=dist, html=True), name="frontend")
    return True


def mount_dev_banner(app: FastAPI) -> None:
    """Dev mode: no static files; `/` points at the Vite dev server."""
    mount_placeholder(
        app,
        f"{DEV_BANNER} (`npm --prefix apps/ui run dev`, then open http://localhost:5173 "
        "through `ssh -L 5173:localhost:5173 <server>`)",
        "dev",
    )
