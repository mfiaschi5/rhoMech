"""The wiki service: the pages of one directory, each parsed and rendered once per file version.

Every call lists `*.md` and stats each file; a page is parsed and rendered again only when its
modification time or size changed, and a deleted file leaves the index. A file whose name is
not a slug, or whose content does not follow the format, is left out of the index and logged
once per version (no silent failures); asking for it by slug is WIKI_PAGE_INVALID (500).
`updated` is the file's modification time (in a git checkout: when it was checked out or
edited). The search `q` matches the title, the summary and the body, ignoring case.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ksmech.serve.content.pages import SLUG_RE, FrontMatter, PageError, parse_page
from ksmech.serve.content.render import Rendered, render
from ksmech.serve.core import schemas as S
from ksmech.serve.core.errors import Code, ServeError

log = logging.getLogger(__name__)

LLMS_HEADER = (
    "# ChemRho wiki\n\n"
    "> How ChemRho predicts organic reaction mechanisms as sequences of elementary steps: the\n"
    "> reaction matrix, the orbital tokens and the Klopman-Salem pair features. Each link below\n"
    '> returns a page as JSON; its "markdown" field is the page source.\n'
)


@dataclass(frozen=True)
class Page:
    slug: str
    meta: FrontMatter
    body: str
    rendered: Rendered
    updated: datetime
    haystack: str  # title, summary and body, lower-case (the search)

    def brief(self) -> S.WikiPageBriefOut:
        m = self.meta
        return S.WikiPageBriefOut(slug=self.slug, title=m.title, summary=m.summary, tags=[*m.tags])


def load_page(path: Path) -> Page:
    """One page file, parsed and rendered; PageError (or OSError) when it cannot be."""
    meta, body = parse_page(path.read_text(encoding="utf-8"))
    rendered = render(body)
    updated = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    haystack = "\n".join((meta.title, meta.summary, body)).lower()
    return Page(path.stem, meta, body, rendered, updated, haystack)


class WikiService:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._cache: dict[Path, tuple[tuple[int, int], Page | str]] = {}
        self._lock = threading.Lock()

    def _entry(self, path: Path) -> Page | str:
        """The page of `path`, or the reason it is not one (cached by mtime and size)."""
        st = path.stat()
        key = (st.st_mtime_ns, st.st_size)
        cached = self._cache.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            entry: Page | str = load_page(path)
        except (PageError, OSError, UnicodeDecodeError) as exc:
            entry = f"{type(exc).__name__}: {exc}"
            log.warning("wiki: %s is left out: %s", path, entry)
        self._cache[path] = (key, entry)
        return entry

    def _scan(self) -> dict[str, Page | str]:
        out: dict[str, Page | str] = {}
        files = sorted(self.directory.glob("*.md")) if self.directory.is_dir() else []
        with self._lock:
            for path in files:
                if not SLUG_RE.fullmatch(path.stem):
                    if path not in self._cache:
                        log.warning("wiki: %s is left out: its name is not a slug", path)
                        self._cache[path] = ((0, 0), "not a slug")
                    continue
                try:
                    out[path.stem] = self._entry(path)
                except OSError as exc:  # deleted between the listing and the stat
                    log.warning("wiki: %s is left out: %s", path, exc)
            for gone in set(self._cache) - set(files):
                del self._cache[gone]
        return out

    def pages(self) -> list[Page]:
        """The valid pages, by order then title."""
        found = [p for p in self._scan().values() if isinstance(p, Page)]
        return sorted(found, key=lambda p: (p.meta.order, p.meta.title.lower(), p.slug))

    def index(self, q: str | None = None) -> S.WikiIndexOut:
        needle = (q or "").strip().lower()
        pages = [p for p in self.pages() if not needle or needle in p.haystack]
        return S.WikiIndexOut(pages=[p.brief() for p in pages])

    def page(self, slug: str) -> S.WikiPageOut:
        entry = self._scan().get(slug)
        if entry is None:
            raise ServeError(Code.WIKI_PAGE_NOT_FOUND, f"no wiki page {slug!r}", status=404)
        if isinstance(entry, str):
            msg = f"the wiki page {slug!r} does not parse: {entry}"
            raise ServeError(Code.WIKI_PAGE_INVALID, msg, detail={"slug": slug})
        r = entry.rendered
        return S.WikiPageOut(
            **entry.brief().model_dump(),
            html=r.html,
            markdown=entry.body,
            scenes=list(r.scenes),
            toc=[S.WikiHeadingOut(level=h.level, id=h.id, text=h.text) for h in r.toc],
            updated=entry.updated,
        )

    def llms_txt(self, base_url: str) -> str:
        """A plain-text index for LLM readers (llmstxt.org): title, link and summary per page."""
        base = base_url.rstrip("/")
        lines = [LLMS_HEADER, "## Pages", ""]
        for p in self.pages():
            summary = f": {p.meta.summary}" if p.meta.summary else ""
            lines.append(f"- [{p.meta.title}]({base}/api/wiki/{p.slug}){summary}")
        return "\n".join(lines) + "\n"
