"""The wiki: read-only Markdown pages from `content/wiki/*.md`, rendered on the server.

A page is a Markdown file whose name is its slug, with TOML front matter between two `+++`
lines (`pages.py`: title, summary, tags, order). `render.py` turns the body into HTML with
markdown-it-py (raw HTML escaped, tables, heading ids, `[[slug]]` links between pages, `scene`
blocks for the views of `@ksmech/viz`); `service.py` caches each page by its file's
modification time and answers the index, a page and `llms.txt`. The authoring format is
`docs/wiki.md`.
"""
