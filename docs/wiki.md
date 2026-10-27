# The wiki: authoring format (`serve.content`)

The wiki is a directory of Markdown files, `content/wiki/*.md`, tracked in git. The server
renders it read-only (`KSMECH_SERVE_WIKI` points at another directory). The repository is
public: a page never quotes dataset records or samples (no PMechDB step, no row of any
dataset). Examples are steps computed by ChemRho itself from SMILES.

## A page

````markdown
+++
title = "The reaction matrix"
summary = "B, R = ΔB, elementary steps, anchors, and resonance modulo the sigma key."
tags = ["representation", "steps"]
order = 10
+++
# The reaction matrix

Text, tables, [[klopman-salem|links to other pages]], and scenes:

```scene
scenes/sn2-step.json
```
````

- **The slug is the file name** without `.md`. It must match `^[a-z0-9][a-z0-9-]{0,63}$`
  (`reaction-matrix.md` is `/api/wiki/reaction-matrix` and `#/wiki/reaction-matrix` in the
  app).
- **The front matter** is TOML between two lines that are exactly `+++`, at the top of the
  file. It is parsed with `tomllib`.

  | Key | Type | Default | Use |
  |---|---|---|---|
  | `title` | string | required | the page title, in the index and in `llms.txt` |
  | `summary` | string | `""` | one line under the title in the index |
  | `tags` | list of strings | `[]` | shown in the index |
  | `order` | integer | 1000 | the index sorts by it, then by title |

  Any other key is an error, so a typo does not vanish.
- **Start the body with one `#` heading**, the title again. The tests check it.

## Markdown

CommonMark, rendered on the server by markdown-it-py 4.2.0:

- **Raw HTML is off.** `<b>` or `<script>` in the source is shown as text, never run.
  Links keep markdown-it's own URL check (no `javascript:`).
- **Tables and strikethrough** are on.
- **Heading ids.** Every heading gets an `id`: its text, lower-case, accents dropped, other
  characters turned into `-`. A repeated heading gets `-2`, `-3`. The headings are the page's
  `toc`.
- **No math renderer.** Write formulas in plain text or Unicode (ΔB, ε, σ\*), or in a code
  block.

## Wiki links

`[[slug]]` links to another page with its slug as the text; `[[slug|text]]` with your text. Both
render as `<a class="wiki-link" href="#/wiki/slug">`. Something that does not match (a slug
with capitals or spaces) stays plain text. The tests check that every wiki link of a tracked
page names an existing page.

## Scenes

A fenced block whose info string is `scene` holds one path, relative to `apps/ui/public`:

````markdown
```scene
scenes/sn2-step.json
```
````

It renders as `<figure class="wiki-scene" data-scene-src="scenes/sn2-step.json"></figure>`.
The frontend fills the figure with the scene's view (`@ksmech/viz`, scene format v1,
`docs/serve_architecture.md`), with no API call.

- The path must be `scenes/...json`, made of letters, digits, `_`, `.` and `-`, without `..`.
  Anything else makes the page invalid.
- The tests check that the file exists.
- Make a scene with `npm --prefix apps/ui run scene:export` from a running server, and check
  its provenance: a scene computed by ChemRho from SMILES, never a dataset record.

## The API

| Route | Answer |
|---|---|
| `GET /api/wiki?q=` | `WikiIndexOut {pages: [{slug, title, summary, tags}]}`, by `order` then title; `q` (at most 200 characters) keeps the pages whose title, summary or body holds it, ignoring case |
| `GET /api/wiki/{slug}` | `WikiPageOut {slug, title, summary, tags, html, markdown, scenes, toc: [{level, id, text}], updated}` |
| `GET /api/wiki/llms.txt` | a plain-text index for LLM readers (llmstxt.org): title, link and summary per page |

- `markdown` is the body without its front matter.
- `updated` is the file's modification time. In a git checkout that is when the file was
  checked out or last edited, not the last commit.

Errors:

| Code | Status | When |
|---|---|---|
| `WIKI_PAGE_NOT_FOUND` | 404 | no such page, or a slug that does not match the pattern |
| `WIKI_PAGE_INVALID` | 500 | the page file does not parse; the reason is in the message |

All three routes are public reads behind the `wiki` capability. It is on in every profile, and
off ("no wiki pages on this server") only when the directory is missing.

## Rendering and caching

The server lists `*.md` at every request and parses and renders a page again only when the
file's modification time or size changed, so an edit shows at the next request with no restart.
A file that does not parse, or whose name is not a slug, is left out of the index and logged
once per version.
