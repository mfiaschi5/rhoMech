"""A wiki page body to HTML: markdown-it-py's CommonMark with the wiki's three additions.

- Raw HTML is off: a `<script>` in the source is shown as text, never run. Tables and
  strikethrough are on; links keep markdown-it's own check (no `javascript:` URLs).
- Every heading gets an `id`, its text slugified (`slugify`; a repeated one gets `-2`, `-3`),
  and goes into the table of contents.
- `[[slug]]` and `[[slug|text]]` link to another page: `<a class="wiki-link"
  href="#/wiki/slug">` (the frontend's hash route). Text that does not match stays text.
- A fenced block with the info string `scene` holds one scene path relative to
  `apps/ui/public` (`SCENE_RE`: `scenes/...json`, no `..`). It renders as
  `<figure class="wiki-scene" data-scene-src="scenes/sn2-step.json"></figure>`, which the frontend
  fills with the scene's view; any other path is a `PageError`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.renderer import RendererHTML

from ksmech.serve.content.pages import PageError

WIKILINK_RE = re.compile(r"\[\[([a-z0-9][a-z0-9-]{0,63})(?:\|([^\]\n|]+))?\]\]")
SCENE_RE = re.compile(r"scenes/(?:[A-Za-z0-9][A-Za-z0-9_.-]*/)*[A-Za-z0-9][A-Za-z0-9_.-]*\.json")
SCENE_INFO = "scene"


@dataclass(frozen=True)
class Heading:
    level: int
    id: str
    text: str


@dataclass(frozen=True)
class Rendered:
    html: str
    toc: list[Heading]
    scenes: list[str]
    links: list[str]  # the slugs of the [[...]] links, in order


def slugify(text: str) -> str:
    """Lower-case ASCII words joined by `-` ("section" when nothing is left): accents are
    dropped (é to e), any other non-ASCII character (a dash, a Greek letter) separates words."""
    decomposed = unicodedata.normalize("NFKD", text)
    kept = "".join(
        c if c.isascii() else ("" if unicodedata.combining(c) else " ") for c in decomposed
    )
    slug = re.sub(r"[^a-z0-9]+", "-", kept.lower()).strip("-")
    return slug or "section"


def _wikilink(state, silent: bool) -> bool:
    if not state.src.startswith("[[", state.pos):
        return False
    m = WIKILINK_RE.match(state.src, state.pos)
    if m is None:
        return False
    if not silent:
        slug, text = m.group(1), (m.group(2) or m.group(1)).strip()
        opening = state.push("link_open", "a", 1)
        opening.attrs = {"href": f"#/wiki/{slug}", "class": "wiki-link"}
        state.push("text", "", 0).content = text
        state.push("link_close", "a", -1)
        state.env.setdefault("links", []).append(slug)
    state.pos = m.end()
    return True


def _inline_text(token) -> str:
    kinds = ("text", "code_inline")
    return "".join(c.content for c in token.children or () if c.type in kinds).strip()


def _heading_ids(state) -> None:
    seen: dict[str, int] = {}
    toc = state.env.setdefault("toc", [])
    for i, token in enumerate(state.tokens):
        if token.type != "heading_open":
            continue
        text = _inline_text(state.tokens[i + 1])
        base = slugify(text)
        seen[base] = seen.get(base, 0) + 1
        anchor = base if seen[base] == 1 else f"{base}-{seen[base]}"
        token.attrSet("id", anchor)
        toc.append(Heading(int(token.tag[1]), anchor, text))


def _fence(self, tokens, idx, options, env) -> str:
    token = tokens[idx]
    info = token.info.strip().split()
    if not info or info[0] != SCENE_INFO:
        return RendererHTML.fence(self, tokens, idx, options, env)
    path = token.content.strip()
    if not SCENE_RE.fullmatch(path):
        raise PageError(f"a scene block holds one path scenes/....json, got {path[:80]!r}")
    env.setdefault("scenes", []).append(path)
    return f'<figure class="wiki-scene" data-scene-src="{escapeHtml(path)}"></figure>\n'


def markdown() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False})
    md.enable(["table", "strikethrough"])
    md.inline.ruler.before("link", "wikilink", _wikilink)
    md.core.ruler.push("heading_ids", _heading_ids)
    md.add_render_rule("fence", _fence)
    return md


_MD = markdown()


def render(body: str) -> Rendered:
    """The HTML of a page body, with its headings, scenes and wiki links."""
    env: dict = {}
    html = _MD.render(body, env)
    return Rendered(html, env.get("toc", []), env.get("scenes", []), env.get("links", []))
