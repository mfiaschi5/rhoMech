"""One wiki page file: its slug, its TOML front matter and its Markdown body.

    +++
    title = "The reaction matrix"
    summary = "B, R = ΔB and what one elementary step is."
    tags = ["representation"]
    order = 20
    +++
    # The reaction matrix
    ...

The front matter is the file's first lines, between two lines that are exactly `+++`, read with
`tomllib`. `title` is required; `summary` (default ""), `tags` (default []) and `order` (default
1000, the index sorts by it, then by title) are optional, and any other key is an error (a typo
would otherwise vanish). The slug is the file name without `.md` and must match `SLUG_RE`.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from typing import Any

SLUG_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
FENCE = "+++"
DEFAULT_ORDER = 1000
KEYS = ("title", "summary", "tags", "order")


class PageError(ValueError):
    """A page file that does not follow the format (the message says where)."""


@dataclass(frozen=True)
class FrontMatter:
    title: str
    summary: str = ""
    tags: tuple[str, ...] = ()
    order: int = DEFAULT_ORDER


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """(the parsed TOML, the Markdown body) of a page's text."""
    lines = text.lstrip("﻿").splitlines(keepends=True)
    if not lines or lines[0].strip() != FENCE:
        raise PageError(f"a page starts with a {FENCE} line (its TOML front matter)")
    end = next((k for k in range(1, len(lines)) if lines[k].strip() == FENCE), None)
    if end is None:
        raise PageError(f"the front matter has no closing {FENCE} line")
    try:
        meta = tomllib.loads("".join(lines[1:end]))
    except tomllib.TOMLDecodeError as exc:
        raise PageError(f"the front matter is not valid TOML: {exc}") from exc
    return meta, "".join(lines[end + 1 :])


def _text(meta: dict, key: str, default: str | None) -> str:
    value = meta.get(key, default)
    if not isinstance(value, str) or (default is None and not value.strip()):
        raise PageError(f"front matter: {key} must be a non-empty string")
    return value.strip()


def front_matter(meta: dict[str, Any]) -> FrontMatter:
    unknown = sorted(set(meta) - set(KEYS))
    if unknown:
        raise PageError(f"front matter: unknown keys {unknown} (known: {', '.join(KEYS)})")
    tags = meta.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) and t.strip() for t in tags):
        raise PageError("front matter: tags must be a list of non-empty strings")
    order = meta.get("order", DEFAULT_ORDER)
    if not isinstance(order, int) or isinstance(order, bool):
        raise PageError("front matter: order must be an integer")
    return FrontMatter(
        title=_text(meta, "title", None),
        summary=_text(meta, "summary", ""),
        tags=tuple(t.strip() for t in tags),
        order=order,
    )


def parse_page(text: str) -> tuple[FrontMatter, str]:
    meta, body = split_front_matter(text)
    return front_matter(meta), body
