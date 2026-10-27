"""serve.content: the wiki. Front matter, the server-side render (scene figures, wiki links,
escaped raw HTML, heading ids), the mtime cache, the routes (index and search, a page, the 404,
llms.txt, slug checks), and every tracked page in content/wiki: it parses, its scenes exist
in apps/ui/public and its wiki links name existing pages."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ksmech.serve.content.pages import DEFAULT_ORDER, FrontMatter, PageError, parse_page
from ksmech.serve.content.render import render, slugify
from ksmech.serve.content.service import WikiService
from ksmech.serve.core.capabilities import status
from ksmech.serve.core.errors import STATUS, Code, ServeError
from ksmech.serve.core.settings import load_settings
from tests.test_serve_fixtures_api import api_client, no_xtb  # noqa: F401 (fixtures)

pytestmark = pytest.mark.serve

REPO = Path(__file__).resolve().parents[1]
WIKI = REPO / "content" / "wiki"
PUBLIC = REPO / "apps" / "ui" / "public"


def _page(title: str = "A page", body: str = "Body.\n", **meta) -> str:
    lines = [f'title = "{title}"'] + [f"{k} = {v}" for k, v in meta.items()]
    return "+++\n" + "\n".join(lines) + "\n+++\n" + body


def _error(r, status_code: int, code: str) -> dict:
    assert r.status_code == status_code, r.text
    body = r.json()["error"]
    assert body["code"] == code, body
    return body


@pytest.fixture
def wiki_dir(tmp_path) -> Path:
    d = tmp_path / "wiki"
    d.mkdir()
    (d / "alpha.md").write_text(
        _page("Alpha", "# Alpha\n\nAbout the sigma key.\n", summary='"First"', order=2)
    )
    (d / "beta.md").write_text(_page("Beta", "# Beta\n\nSee [[alpha]].\n", tags='["x", "y"]'))
    (d / "early.md").write_text(_page("Zulu", "Early.\n", order=1))
    return d


@pytest.fixture
def wiki_app(tmp_path, wiki_dir):
    settings = load_settings(
        {}, data_root=tmp_path / "root", chemistry="synthetic", profile="public", wiki_dir=wiki_dir
    )
    with api_client(settings) as client:
        yield client


# ---- front matter


def test_front_matter():
    meta, body = parse_page(_page("T", "# H\n", summary='"S"', tags='["a"]', order=5))
    assert meta == FrontMatter("T", "S", ("a",), 5) and body == "# H\n"
    meta, _ = parse_page("﻿" + _page("Only a title"))
    assert meta == FrontMatter("Only a title", "", (), DEFAULT_ORDER)
    bad = [
        ("no front matter\n", "starts with"),
        ("+++\ntitle = 'x'\n", "no closing"),
        ("+++\ntitle = \n+++\n", "not valid TOML"),
        (_page("T", titel='"typo"'), "unknown keys"),
        ("+++\nsummary = 'no title'\n+++\n", "title"),
        (_page("T", tags='"not-a-list"'), "tags"),
        (_page("T", order='"first"'), "order"),
        (_page("T", order="true"), "order"),
    ]
    for text, match in bad:
        with pytest.raises(PageError, match=match):
            parse_page(text)


# ---- rendering


def test_scene_blocks_become_figures():
    r = render("Text.\n\n```scene\nscenes/sn2-step.json\n```\n\n```python\nx = 1\n```\n")
    assert '<figure class="wiki-scene" data-scene-src="scenes/sn2-step.json"></figure>' in r.html
    assert r.scenes == ["scenes/sn2-step.json"]
    assert '<code class="language-python">x = 1' in r.html  # other fences: plain code
    for bad in ("../secrets.json", "scenes/../x.json", "/etc/passwd", "scenes/a.js", "a\nb"):
        with pytest.raises(PageError, match="scene"):
            render(f"```scene\n{bad}\n```\n")


def test_wiki_links():
    r = render("See [[reaction-matrix]] and [[klopman-salem|the pair features]].\n")
    assert '<a href="#/wiki/reaction-matrix" class="wiki-link">reaction-matrix</a>' in r.html
    assert '<a href="#/wiki/klopman-salem" class="wiki-link">the pair features</a>' in r.html
    assert r.links == ["reaction-matrix", "klopman-salem"]
    plain = render("Not links: [[Bad Slug]], [[a|]], [single](https://example.org).\n")
    assert plain.links == [] and "[[Bad Slug]]" in plain.html
    assert '<a href="https://example.org">single</a>' in plain.html


def test_raw_html_is_escaped():
    r = render(
        '<script>alert(1)</script>\n\nA <b onclick="x()">tag</b>.\n\n[x](javascript:alert(1))\n'
    )
    assert "<script>" not in r.html and "&lt;script&gt;alert(1)&lt;/script&gt;" in r.html
    assert "<b " not in r.html and "&lt;b onclick" in r.html
    assert 'href="javascript' not in r.html


def test_heading_anchors_and_tables():
    r = render(
        "# The R matrix\n\n## Résonance\n\n## The R matrix\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    )
    assert '<h1 id="the-r-matrix">' in r.html and '<h2 id="resonance">' in r.html
    assert '<h2 id="the-r-matrix-2">' in r.html
    assert [(h.level, h.id, h.text) for h in r.toc] == [
        (1, "the-r-matrix", "The R matrix"),
        (2, "resonance", "Résonance"),
        (2, "the-r-matrix-2", "The R matrix"),
    ]
    assert "<table>" in r.html and "<td>2</td>" in r.html
    assert slugify("Klopman–Salem: G & |G|") == "klopman-salem-g-g" and slugify("—") == "section"


# ---- the service


def test_index_order_search_and_cache(wiki_dir):
    wiki = WikiService(wiki_dir)
    assert [p.slug for p in wiki.index().pages] == ["early", "alpha", "beta"]  # order, title
    assert [p.slug for p in wiki.index("SIGMA KEY").pages] == ["alpha"]  # the body
    assert [p.slug for p in wiki.index("first").pages] == ["alpha"]  # the summary
    assert [p.slug for p in wiki.index("zulu").pages] == ["early"]  # the title
    assert wiki.index("nowhere").pages == []
    first = wiki.pages()[1]
    assert wiki.pages()[1] is first  # cached: the same object while the file is unchanged
    path = wiki_dir / "alpha.md"
    path.write_text(_page("Alpha 2", "New body.\n", order=2))
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))
    assert wiki.page("alpha").title == "Alpha 2"
    (wiki_dir / "beta.md").unlink()
    assert [p.slug for p in wiki.index().pages] == ["early", "alpha"]


def test_invalid_pages_are_left_out_and_reported(wiki_dir, caplog):
    (wiki_dir / "broken.md").write_text("no front matter\n")
    (wiki_dir / "Not A Slug.md").write_text(_page("X"))
    wiki = WikiService(wiki_dir)
    assert [p.slug for p in wiki.index().pages] == ["early", "alpha", "beta"]
    assert "broken.md is left out" in caplog.text and "not a slug" in caplog.text
    with pytest.raises(ServeError) as exc:
        wiki.page("broken")
    assert exc.value.code == Code.WIKI_PAGE_INVALID and exc.value.status == 500
    assert STATUS[Code.WIKI_PAGE_NOT_FOUND] == 404


# ---- the routes


def test_routes(wiki_app):
    index = wiki_app.get("/api/wiki").json()
    assert index["pages"][1] == {"slug": "alpha", "title": "Alpha", "summary": "First", "tags": []}
    assert [
        p["slug"] for p in wiki_app.get("/api/wiki", params={"q": "sigma"}).json()["pages"]
    ] == ["alpha"]
    page = wiki_app.get("/api/wiki/beta").json()
    assert set(page) == {
        "slug", "title", "summary", "tags", "html", "markdown", "scenes", "toc", "updated",
    }  # fmt: skip
    assert page["tags"] == ["x", "y"] and page["markdown"] == "# Beta\n\nSee [[alpha]].\n"
    assert page["toc"] == [{"level": 1, "id": "beta", "text": "Beta"}]
    assert 'href="#/wiki/alpha"' in page["html"] and page["updated"].endswith("Z")
    _error(wiki_app.get("/api/wiki/gamma"), 404, "WIKI_PAGE_NOT_FOUND")
    for bad in ("Alpha", "-alpha", "a" * 65, "alpha.md", "al_pha"):
        _error(wiki_app.get(f"/api/wiki/{bad}"), 404, "WIKI_PAGE_NOT_FOUND")
    _error(wiki_app.get("/api/wiki", params={"q": "x" * 201}), 422, "REQUEST_INVALID")


def test_llms_txt(wiki_app):
    r = wiki_app.get("/api/wiki/llms.txt")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    lines = r.text.splitlines()
    assert lines[0] == "# ChemRho wiki"
    entries = [x for x in lines if x.startswith("- [")]
    assert entries == [
        "- [Zulu](http://testserver/api/wiki/early)",
        "- [Alpha](http://testserver/api/wiki/alpha): First",
        "- [Beta](http://testserver/api/wiki/beta)",
    ]


def test_the_capability(tmp_path):
    settings = load_settings({}, data_root=tmp_path, wiki_dir=tmp_path / "none", profile="public")
    assert status("wiki", settings) == (False, "no wiki pages on this server")
    assert status("wiki", load_settings({}, data_root=tmp_path)) == (True, None)
    assert load_settings({}, data_root=tmp_path).wiki == WIKI
    with api_client(
        load_settings({}, data_root=tmp_path, chemistry="synthetic", wiki_dir=tmp_path / "none")
    ) as client:
        err = _error(client.get("/api/wiki"), 503, "FEATURE_DISABLED")
        assert err["detail"]["capability"] == "wiki"


# ---- the tracked pages


def test_every_tracked_page_parses_and_its_scenes_and_links_exist():
    wiki = WikiService(WIKI)
    files = sorted(p.stem for p in WIKI.glob("*.md"))
    slugs = [p.slug for p in wiki.pages()]
    assert sorted(slugs) == files  # none left out
    assert {"index", "reaction-matrix", "orbital-tokens", "klopman-salem", "sn2-example"} <= set(
        slugs
    )
    for page in wiki.pages():
        r = page.rendered
        assert r.toc and r.toc[0].level == 1, page.slug  # every page opens with its title
        for scene in r.scenes:
            assert (PUBLIC / scene).is_file(), (page.slug, scene)
        assert set(r.links) <= set(slugs), (page.slug, set(r.links) - set(slugs))
    assert wiki.page("sn2-example").scenes == ["scenes/sn2-step.json"]
