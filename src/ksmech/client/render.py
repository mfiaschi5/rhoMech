"""Plain-text rendering of the `ksmech` CLI: compact tables and key-value blocks (no colour, no
third-party package), and the JSON of `--json`."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

MAX_CELL = 60


def jsonable(value: Any) -> Any:
    """Pydantic models (and lists of them) as JSON data; anything else unchanged."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    return value


def to_json(value: Any) -> str:
    return json.dumps(jsonable(value), indent=2, sort_keys=False, default=str)


def cell(value: Any, width: int = MAX_CELL) -> str:
    """One table cell: None as "-", floats with 4 significant digits, long text cut."""
    if value is None:
        text = "-"
    elif isinstance(value, bool):
        text = "yes" if value else "no"
    elif isinstance(value, float):
        text = f"{value:.4g}"
    elif isinstance(value, list | tuple):
        text = ", ".join(cell(v, width) for v in value)
    elif isinstance(value, dict):
        text = " ".join(f"{k}={cell(v, width)}" for k, v in value.items())
    else:
        text = str(value)
    return text if len(text) <= width else text[: width - 3] + "..."


def table(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """A left-aligned table of `columns` (a missing key is "-"); "(none)" when empty."""
    body = [[cell(r.get(c)) for c in columns] for r in rows]
    if not body:
        return "(none)"
    widths = [max(len(c), *(len(row[i]) for row in body)) for i, c in enumerate(columns)]
    lines = ["  ".join(c.upper().ljust(w) for c, w in zip(columns, widths, strict=True))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)) for row in body]
    return "\n".join(line.rstrip() for line in lines)


def kv(data: Mapping[str, Any], keys: Sequence[str] | None = None) -> str:
    """`key: value` lines (the given keys in order, else every key; None values skipped)."""
    names = list(keys) if keys is not None else list(data)
    width = max((len(k) for k in names), default=0)
    lines = [f"{k.ljust(width)}  {cell(data[k], 100)}" for k in names if data.get(k) is not None]
    return "\n".join(lines)
