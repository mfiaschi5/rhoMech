"""serve.ocr: image uploads, the `ocr` capability, the JSON-lines worker, RDKit's check of the
engine's answer, the route and its job quota, and `ksmech-serve ocr-setup`. Everything runs on
a fake engine, a stub worker script or mocked subprocess and HTTP calls: MolScribe is never
installed or downloaded here. The one test of the real engine skips unless `ocr-setup` has run
on this data root."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import io
import json
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ksmech.serve.accounts import users
from ksmech.serve.core.capabilities import status
from ksmech.serve.core.errors import STATUS, Code, ServeError
from ksmech.serve.core.settings import load_settings
from ksmech.serve.ocr import images, install, provision
from ksmech.serve.ocr.engine import JsonLinesWorker, MolScribeEngine, RawOcr, raw_of
from ksmech.serve.ocr.install import CHECKPOINT_SHA256, NOT_INSTALLED, OcrPaths, write_marker
from ksmech.serve.ocr.service import OcrService
from ksmech.serve.ocr.structure import BOND_LENGTH, structure_of
from tests.test_serve_fixtures_api import api_client, no_xtb  # noqa: F401 (fixtures)

pytestmark = pytest.mark.serve

REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "tools" / "ocr" / "ocr_worker.py"
PASSWORD = "correct horse battery"
ETHANOL = "CCO"


def _settings(tmp_path, profile: str = "lab", auth: str | None = None):
    path = None
    if auth is not None:
        path = tmp_path / "auth.toml"
        path.write_text(auth, encoding="utf-8")
    root = tmp_path / "root"
    return load_settings({}, data_root=root, chemistry="synthetic", profile=profile, auth_file=path)


def _error(r, status_code: int, code: str) -> dict:
    assert r.status_code == status_code, r.text
    body = r.json()["error"]
    assert body["code"] == code, body
    return body


def _image(fmt: str = "PNG", size=(60, 40), mode: str = "RGB", color=(0, 0, 0), **save) -> bytes:
    out = io.BytesIO()
    Image.new(mode, size, color).save(out, format=fmt, **save)
    return out.getvalue()


def _noise_png(side: int = 200) -> bytes:
    pixels = np.random.default_rng(0).integers(0, 255, (side, side, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return buf.getvalue()


def fake_install(settings) -> OcrPaths:
    """The files the capability checks: a venv python, a checkpoint and its marker."""
    paths = OcrPaths.of(settings)
    paths.python.parent.mkdir(parents=True, exist_ok=True)
    paths.python.write_text("", encoding="utf-8")
    paths.weights.mkdir(parents=True, exist_ok=True)
    paths.checkpoint.write_bytes(b"x" * 10)
    write_marker(paths, CHECKPOINT_SHA256, 10)
    return paths


def _molblock(smiles: str = ETHANOL, scale: float = 0.1) -> str:
    """A molfile of `smiles` whose coordinates are small fractions (as MolScribe writes them)."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (0.5 + p.x * scale, 0.5 + p.y * scale, 0.0))
    return Chem.MolToMolBlock(mol)


class FakeEngine:
    """An `OcrEngine` that answers `answer(png_path)` and records what it was given."""

    name = "fake 1.0"

    def __init__(self, answer=None) -> None:
        self.answer = answer or (lambda png: RawOcr(ETHANOL, _molblock(), 0.9))
        self.seen: list[tuple[str, tuple[int, int], str]] = []
        self.closed = False

    def recognize(self, png: Path, timeout_s: float) -> RawOcr:
        with Image.open(png) as img:
            self.seen.append((img.format, img.size, img.mode))
        return self.answer(png)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def ocr_app(tmp_path):
    settings = _settings(tmp_path)
    fake_install(settings)
    with api_client(settings) as client:
        engine = FakeEngine()
        client.app.state.services.ocr = OcrService(settings, engine=engine)
        client.engine = engine
        yield client


# ---- upload validation


def test_accepted_images_become_flat_pngs():
    for fmt in ("PNG", "JPEG", "WEBP", "BMP", "TIFF", "GIF"):
        mode = "P" if fmt == "GIF" else "RGB"
        out = images.to_png(_image(fmt, mode=mode, color=0 if mode == "P" else (9, 9, 9)))
        assert out.format == images.ACCEPTED[fmt] and (out.width, out.height) == (60, 40)
        with Image.open(io.BytesIO(out.png)) as img:
            assert img.format == "PNG" and img.mode == "RGB"
    clear = images.to_png(_image("PNG", mode="RGBA", color=(0, 0, 0, 0)))
    with Image.open(io.BytesIO(clear.png)) as img:
        assert img.getpixel((5, 5)) == (255, 255, 255)  # transparent: white, not black
    frames = [Image.new("L", (20, 20), v) for v in (0, 255)]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:])
    assert "only the first was read" in images.to_png(buf.getvalue()).warnings[0]


def test_rejected_uploads_have_their_codes(monkeypatch):
    cases = [
        (b"", Code.IMAGE_INVALID),
        (b"just some text, not an image", Code.IMAGE_TYPE_UNSUPPORTED),
        (_image("ICO", size=(32, 32)), Code.IMAGE_TYPE_UNSUPPORTED),
        (_noise_png()[:2000], Code.IMAGE_INVALID),  # truncated: identified, not decodable
    ]
    for data, code in cases:
        with pytest.raises(ServeError) as exc:
            images.to_png(data)
        assert exc.value.code == code, data[:20]
    assert STATUS[Code.IMAGE_TYPE_UNSUPPORTED] == 415 and STATUS[Code.IMAGE_INVALID] == 422
    monkeypatch.setattr(images, "MAX_BYTES", 100)
    with pytest.raises(ServeError) as exc:
        images.to_png(_image("BMP"))
    assert exc.value.code == Code.IMAGE_TOO_LARGE and exc.value.status == 413
    monkeypatch.setattr(images, "MAX_BYTES", 10 << 20)
    monkeypatch.setattr(images, "MAX_PIXELS", 1000)
    with pytest.raises(ServeError) as exc:
        images.to_png(_image("PNG"))
    assert exc.value.code == Code.IMAGE_TOO_LARGE and exc.value.detail["max_pixels"] == 1000
    monkeypatch.setattr(images, "MAX_PIXELS", 10**8)
    monkeypatch.setattr(images, "MAX_SIDE", 30)
    small = images.to_png(_image("PNG"))
    assert (small.width, small.height) == (30, 20)  # the aspect ratio is kept


def test_the_streamed_upload_stops_at_the_limit():
    from ksmech.serve.ocr.router import _limited

    async def chunks():
        for _ in range(10):
            yield b"x" * 100

    async def drain(limit: int) -> int:
        return sum([len(c) async for c in _limited(chunks(), limit)])

    assert asyncio.run(drain(1000)) == 1000
    with pytest.raises(ServeError) as exc:
        asyncio.run(drain(999))
    assert exc.value.code == Code.IMAGE_TOO_LARGE


# ---- the capability


def test_the_capability_needs_the_installed_engine(tmp_path):
    settings = _settings(tmp_path)
    assert status("ocr", settings) == (False, NOT_INSTALLED)
    paths = fake_install(settings)
    assert status("ocr", settings) == (True, None)
    write_marker(paths, "0" * 64, 10)  # a marker of another file
    assert status("ocr", settings) == (False, NOT_INSTALLED)
    write_marker(paths, CHECKPOINT_SHA256, 11)  # the checkpoint changed size since the check
    assert status("ocr", settings) == (False, NOT_INSTALLED)
    write_marker(paths, CHECKPOINT_SHA256, 10)
    paths.python.unlink()
    assert install.missing(paths) == [f"the environment {paths.venv}"]
    assert status("ocr", _settings(tmp_path, "public")) == (False, NOT_INSTALLED)


def test_routes_while_the_engine_is_not_installed(tmp_path):
    with api_client(_settings(tmp_path)) as client:
        out = client.get("/api/ocr").json()
        assert out == {"enabled": False, "reason": NOT_INSTALLED, "engine": install.ENGINE}
        r = client.post("/api/ocr", files={"image": ("a.png", _image(), "image/png")})
        err = _error(r, 503, "FEATURE_DISABLED")
        assert err["detail"] == {"capability": "ocr", "reason": NOT_INSTALLED}
        meta = client.get("/api/meta").json()["capabilities"]
        assert meta["ocr"] == {"enabled": False, "reason": NOT_INSTALLED}


# ---- the route


def test_the_route_reads_one_image(ocr_app):
    assert ocr_app.get("/api/ocr").json() == {"enabled": True, "reason": None, "engine": "fake 1.0"}
    r = ocr_app.post("/api/ocr", files={"image": ("photo.jpg", _image("JPEG"), "image/jpeg")})
    assert r.status_code == 200, r.text
    out = r.json()
    assert set(out) == {"smiles", "molfile", "confidence", "engine", "elapsed_s", "warnings"}
    assert out["smiles"] == ETHANOL and out["confidence"] == 0.9 and out["engine"] == "fake 1.0"
    assert out["warnings"] == [] and "V2000" in out["molfile"]
    assert ocr_app.engine.seen == [("PNG", (60, 40), "RGB")]  # the engine always gets a PNG


def test_the_route_refuses_bad_uploads(ocr_app, monkeypatch):
    _error(ocr_app.post("/api/ocr", json={"image": "x"}), 422, "REQUEST_INVALID")
    r = ocr_app.post("/api/ocr", files={"picture": ("a.png", _image(), "image/png")})
    assert "image" in _error(r, 422, "REQUEST_INVALID")["message"]
    two = [("image", ("a.png", _image(), "image/png")), ("image", ("b.png", _image(), "x"))]
    _error(ocr_app.post("/api/ocr", files=two), 422, "REQUEST_INVALID")
    r = ocr_app.post("/api/ocr", files={"image": ("a.pdf", b"%PDF-1.4 ...", "application/pdf")})
    _error(r, 415, "IMAGE_TYPE_UNSUPPORTED")
    monkeypatch.setattr(images, "MAX_BYTES", 1000)
    r = ocr_app.post("/api/ocr", files={"image": ("big.png", _noise_png(), "image/png")})
    _error(r, 413, "IMAGE_TOO_LARGE")  # over MAX_BYTES + the framing: by Content-Length
    r = ocr_app.post("/api/ocr", files={"image": ("big.png", _image("BMP"), "image/bmp")})
    _error(r, 413, "IMAGE_TOO_LARGE")  # within the framing, over MAX_BYTES itself
    assert ocr_app.engine.seen == []


def test_nothing_parses_is_ocr_failed(ocr_app):
    ocr_app.engine.answer = lambda png: RawOcr("<invalid>", "", None)
    r = ocr_app.post("/api/ocr", files={"image": ("a.png", _image(), "image/png")})
    _error(r, 422, "OCR_FAILED")


def test_ocr_jobs_count_against_the_quota(tmp_path):
    """public profile: anonymous readers may not; a user's third image of a minute is refused
    (an invalid upload counts too); an admin is never limited."""
    settings = _settings(tmp_path, "public", auth="[quota.user]\nper_day = 50\nper_minute = 2\n")
    fake_install(settings)
    upload = {"image": ("a.png", _image(), "image/png")}
    with api_client(settings) as client:
        client.app.state.services.ocr = OcrService(settings, engine=FakeEngine())
        assert client.get("/api/ocr").json()["enabled"] is True  # a reader sees the status
        _error(client.post("/api/ocr", files=upload), 401, "AUTH_REQUIRED")
        engine = client.app.state.services.accounts.engine
        for login, role in (("fay", "user"), ("gil", "admin")):
            users.create_user(engine, login, login.title(), role, password=PASSWORD)
        login = {"login": "fay", "password": PASSWORD}
        assert client.post("/api/auth/login", json=login).status_code == 200
        assert client.post("/api/ocr", files=upload).status_code == 200
        bad = {"image": ("a.txt", b"not an image", "text/plain")}
        _error(client.post("/api/ocr", files=bad), 415, "IMAGE_TYPE_UNSUPPORTED")
        err = _error(client.post("/api/ocr", files=upload), 429, "QUOTA_EXCEEDED")
        assert err["detail"]["window"] == "minute"
        assert client.get("/api/auth/me").json()["quota"]["used_today"] == 2
        client.post("/api/auth/logout")
        login = {"login": "gil", "password": PASSWORD}
        assert client.post("/api/auth/login", json=login).status_code == 200
        for _ in range(4):
            assert client.post("/api/ocr", files=upload).status_code == 200


def test_the_client_and_its_command(ocr_app, tmp_path, capsys):
    from ksmech.client import cli
    from ksmech.client.client import KsmechClient

    client = KsmechClient(http=ocr_app, via="cli")
    assert client.ocr_status().enabled is True
    assert client.ocr(_image("WEBP"), "x.webp").smiles == ETHANOL
    image, molfile = tmp_path / "drawn.png", tmp_path / "out.mol"
    image.write_bytes(_image())
    assert cli.main(["ocr", str(image), "--molfile", str(molfile)], client=client) == 0
    assert "smiles" in capsys.readouterr().out and "V2000" in molfile.read_text()
    assert cli.main(["ocr", str(tmp_path / "missing.png")], client=client) == 2
    text = tmp_path / "notes.txt"
    text.write_text("not an image")
    assert cli.main(["ocr", str(text)], client=client) == 1
    assert "IMAGE_TYPE_UNSUPPORTED" in capsys.readouterr().err


# ---- RDKit's check of the answer


def _coords(molfile: str) -> np.ndarray:
    from rdkit import Chem

    mol = Chem.MolFromMolBlock(molfile)
    return np.asarray(mol.GetConformer().GetPositions())[:, :2], mol


def test_the_drawn_layout_is_kept_and_scaled():
    from rdkit import Chem

    block = Chem.MolToMolBlock(Chem.MolFromSmiles("CC"))  # no coordinates: set them by hand
    lines = block.splitlines()
    lines[4] = f"{0.25:10.4f}{0.5:10.4f}{0.0:10.4f}" + lines[4][30:]
    lines[5] = f"{0.75:10.4f}{0.5:10.4f}{0.0:10.4f}" + lines[5][30:]
    out = structure_of(RawOcr("CC", "\n".join(lines), 0.5, width=400, height=100))
    xy, _ = _coords(out.molfile)
    assert np.isclose(np.linalg.norm(xy[0] - xy[1]), BOND_LENGTH, atol=1e-3)
    assert np.allclose(xy.mean(axis=0), 0.0, atol=1e-3) and xy[0, 0] < xy[1, 0]
    assert out.smiles == "CC" and out.warnings == []
    tall = structure_of(RawOcr(ETHANOL, _molblock(), None, width=100, height=400))
    wide = structure_of(RawOcr(ETHANOL, _molblock(), None, width=400, height=100))
    t, _ = _coords(tall.molfile)
    w, _ = _coords(wide.molfile)
    span = lambda a: np.ptp(a, axis=0)  # noqa: E731
    assert span(t)[1] / span(t)[0] > span(w)[1] / span(w)[0]  # the image's aspect ratio counts


def test_answers_that_need_rdkit_to_step_in():
    from rdkit import Chem

    abbreviated = structure_of(RawOcr("CCO", _molblock("C*"), 0.7, 300, 300))
    assert abbreviated.smiles == "CCO" and "RDKit's" in abbreviated.warnings[0]
    no_smiles = structure_of(RawOcr("<invalid>", _molblock(), 0.7, 300, 300))
    assert no_smiles.smiles == ETHANOL and "does not parse" in no_smiles.warnings[0]
    no_block = structure_of(RawOcr("c1ccccc1", None, None))
    assert no_block.smiles == "c1ccccc1" and Chem.MolFromMolBlock(no_block.molfile) is not None
    pentavalent = Chem.RWMol(Chem.MolFromMolBlock(_molblock("CC(C)(C)C")))
    pentavalent.AddAtom(Chem.Atom(6))
    pentavalent.AddBond(1, 5, Chem.BondType.SINGLE)
    pentavalent.GetConformer().SetAtomPosition(5, (0.6, 0.6, 0.0))
    pentavalent.UpdatePropertyCache(strict=False)
    block = Chem.MolToMolBlock(pentavalent, kekulize=False)
    bad = structure_of(RawOcr("<invalid>", block, 0.1, 100, 100))
    assert "not valid for RDKit" in bad.warnings[0] and "V2000" in bad.molfile
    with pytest.raises(ServeError) as exc:
        structure_of(RawOcr("<invalid>", "not a molfile", None))
    assert exc.value.code == Code.OCR_FAILED and exc.value.status == 422
    with pytest.raises(ServeError) as exc:
        raw_of({"id": 1, "ok": False, "error": "no structure"})
    assert exc.value.code == Code.OCR_FAILED and "no structure" in exc.value.message


# ---- the JSON-lines worker

STUB = textwrap.dedent(
    """
    import json, os, sys, time
    print("a library banner on stdout, not a protocol line", flush=True)
    if os.environ.get("STUB_FAIL_START"):
        print(json.dumps({"ready": False, "error": "no weights"}), flush=True)
        sys.exit(1)
    print(json.dumps({"ready": True, "engine": "stub", "pid": os.getpid()}), flush=True)
    for line in sys.stdin:
        msg = json.loads(line)
        if msg.get("op") == "quit":
            break
        image = msg["image"]
        if "crash" in image:
            os._exit(3)
        if "hang" in image:
            time.sleep(60)
        print(json.dumps({"id": 0, "ok": True, "smiles": "stale"}), flush=True)  # another id
        out = {"id": msg["id"], "ok": "fail" not in image, "smiles": "CCO", "pid": os.getpid()}
        print(json.dumps(out | {"error": "cannot read"}), flush=True)
    """
)


def _stub_worker(tmp_path, **kw) -> JsonLinesWorker:
    script = tmp_path / "stub_worker.py"
    script.write_text(STUB, encoding="utf-8")
    kw.setdefault("ready_timeout_s", 30)
    return JsonLinesWorker([sys.executable, "-I", script], log_path=tmp_path / "w.log", **kw)


def test_the_worker_answers_json_lines_and_stays_up(tmp_path):
    worker = _stub_worker(tmp_path)
    try:
        assert not worker.alive() and worker.pid is None  # spawned lazily
        first = worker.request({"image": "a.png"}, timeout_s=10)
        assert first["ok"] and first["smiles"] == "CCO" and first["id"] == 1
        assert worker.ready["engine"] == "stub"
        second = worker.request({"image": "b.png"}, timeout_s=10)
        assert second["pid"] == first["pid"] and second["id"] == 2  # one persistent process
        assert raw_of(second) == RawOcr("CCO", None, None)
        assert not worker.request({"image": "fail.png"}, timeout_s=10)["ok"]
    finally:
        worker.close()
    assert not worker.alive()


def test_the_worker_restarts_after_a_crash_and_a_timeout(tmp_path):
    worker = _stub_worker(tmp_path)
    try:
        pid = worker.request({"image": "a.png"}, timeout_s=10)["pid"]
        with pytest.raises(ServeError) as exc:
            worker.request({"image": "crash.png"}, timeout_s=10)
        assert exc.value.code == Code.OCR_ENGINE_FAILED and "code 3" in exc.value.message
        again = worker.request({"image": "a.png"}, timeout_s=10)
        assert again["pid"] != pid and worker.restarts == 1
        with pytest.raises(ServeError) as exc:
            worker.request({"image": "hang.png"}, timeout_s=0.5)
        assert exc.value.code == Code.OCR_TIMEOUT and exc.value.status == 504
        assert not worker.alive()  # the stuck worker was stopped
        assert worker.request({"image": "a.png"}, timeout_s=10)["ok"] and worker.restarts == 2
    finally:
        worker.close()


def test_the_worker_start_failure_and_the_busy_lock(tmp_path):
    worker = _stub_worker(tmp_path, env={"STUB_FAIL_START": "1"})
    with pytest.raises(ServeError) as exc:
        worker.request({"image": "a.png"}, timeout_s=10)
    assert exc.value.code == Code.OCR_ENGINE_FAILED and "no weights" in exc.value.message
    busy = _stub_worker(tmp_path, busy_timeout_s=0.1)
    busy._lock.acquire()
    try:
        with pytest.raises(ServeError) as exc:
            busy.request({"image": "a.png"}, timeout_s=10)
        assert exc.value.code == Code.OCR_BUSY and exc.value.status == 503
    finally:
        busy._lock.release()
        busy.close()


def test_one_request_at_a_time(tmp_path):
    worker = _stub_worker(tmp_path)
    answers: list[dict] = []

    def ask(k: int) -> None:
        answers.append(worker.request({"image": f"{k}.png"}, timeout_s=20))

    try:
        threads = [threading.Thread(target=ask, args=(k,)) for k in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert sorted(a["id"] for a in answers) == [1, 2, 3, 4]
        assert len({a["pid"] for a in answers}) == 1
    finally:
        worker.close()


def _load_worker_module():
    spec = importlib.util.spec_from_file_location("ocr_worker_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_real_worker_loop_with_a_fake_model(tmp_path):
    """tools/ocr/ocr_worker.py's protocol, in process, with a fake predictor for MolScribe."""
    mod = _load_worker_module()
    image = tmp_path / "x.png"
    image.write_bytes(_image())
    lines = [{"id": 1, "image": str(image)}, {"id": 2, "image": str(tmp_path / "none.png")}]
    lines += [{"id": 3}, {"op": "quit"}, {"id": 4, "image": str(image)}]
    inp = io.StringIO("\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
    out = io.StringIO()

    def predict(path: str) -> dict:
        if path.endswith("x.png"):
            return {"smiles": "C", "molfile": "", "confidence": 1.0, "width": 60, "height": 40}
        raise AssertionError(path)

    assert mod.serve(predict, inp, out) == 0
    answers = [json.loads(x) for x in out.getvalue().splitlines()]
    assert [a["id"] for a in answers] == [1, 2, 3]  # stopped at quit
    assert answers[0] == {"id": 1, "ok": True, "smiles": "C", "molfile": "", "confidence": 1.0,
                          "width": 60, "height": 40}  # fmt: skip
    assert "no such file" in answers[1]["error"] and not answers[2]["ok"]
    assert mod.parse_args(["--weights", "w.pth"]).threads == 2
    assert mod.parse_args(["--weights", "w.pth", "--threads", "3"]).threads == 3


def test_the_molscribe_engine_reports_a_worker_that_cannot_load(tmp_path):
    """The real worker under an interpreter without MolScribe: `ready: false`, then exit 1."""
    paths = OcrPaths(tmp_path / "ocr", REPO / "tools" / "ocr")
    paths.python.parent.mkdir(parents=True)
    paths.python.symlink_to(sys.executable)
    engine = MolScribeEngine(paths, threads=1, ready_timeout_s=60)
    argv = engine.worker.argv
    assert argv[:3] == [str(paths.python), "-I", str(WORKER)] and argv[-2:] == ["--threads", "1"]
    assert engine.worker.env["OMP_NUM_THREADS"] == "1" and "VIRTUAL_ENV" not in engine.worker.env
    with pytest.raises(ServeError) as exc:
        engine.recognize(tmp_path / "a.png", timeout_s=10)
    assert exc.value.code == Code.OCR_ENGINE_FAILED
    assert "could not start" in exc.value.message and "ModuleNotFoundError" in exc.value.message
    engine.close()


# ---- ksmech-serve ocr-setup


class FakeRun:
    """`subprocess.run` for uv: records the commands; `uv sync` creates the venv's python."""

    def __init__(self, paths: OcrPaths, fail: str | None = None) -> None:
        self.paths, self.fail, self.calls = paths, fail, []

    def __call__(self, argv, env=None, check=False):
        self.calls.append((list(argv), dict(env or {})))
        if argv[1] == self.fail:
            return SimpleNamespace(returncode=2)
        if argv[1] == "sync":
            self.paths.python.parent.mkdir(parents=True, exist_ok=True)
            self.paths.python.write_text("", encoding="utf-8")
        return SimpleNamespace(returncode=0)


def _tool_paths(tmp_path, locked: bool) -> OcrPaths:
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    if locked:
        (tools / "uv.lock").write_text("", encoding="utf-8")
    return OcrPaths(tmp_path / "data" / "ocr", tools)


def test_the_environment_is_synced_with_uv(tmp_path):
    paths = _tool_paths(tmp_path, locked=False)
    run = FakeRun(paths)
    provision.sync_env(paths, uv="/bin/uv", run=run, say=lambda m: None)
    assert [c[0][:2] for c in run.calls] == [["/bin/uv", "lock"], ["/bin/uv", "sync"]]
    argv, env = run.calls[1]
    assert argv == ["/bin/uv", "sync", "--project", str(paths.tools), "--locked"]
    assert env["UV_PROJECT_ENVIRONMENT"] == str(paths.venv) and "VIRTUAL_ENV" not in env
    (paths.tools / "uv.lock").write_text("", encoding="utf-8")
    assert [s[1] for s in provision.uv_steps(paths)] == ["sync"]
    assert [s[1] for s in provision.uv_steps(paths, relock=True)] == ["lock", "sync"]
    with pytest.raises(provision.SetupError, match="exit code 2"):
        provision.sync_env(paths, run=FakeRun(paths, fail="sync"), say=lambda m: None)


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, status: int, fail_after: int | None = None) -> None:
        super().__init__(data)
        self.status, self.fail_after, self.sent = status, fail_after, 0

    def read(self, n: int = -1) -> bytes:
        if self.fail_after is not None and self.sent >= self.fail_after:
            raise ConnectionResetError("connection reset by peer")
        block = super().read(min(n, 1 << 16) if n > 0 else -1)
        self.sent += len(block)
        return block


class FakeOpener:
    """`urlopen` over `payload`: honours Range (206) unless `ignore_range`; the first response
    breaks after `break_after` bytes."""

    def __init__(self, payload: bytes, break_after: int | None = None, ignore_range=False):
        self.payload, self.break_after, self.ignore_range = payload, break_after, ignore_range
        self.ranges: list[str | None] = []

    def __call__(self, request, timeout=None):
        rng = request.get_header("Range")
        self.ranges.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng and not self.ignore_range else 0
        fail, self.break_after = self.break_after, None
        return FakeResponse(self.payload[start:], 206 if start else 200, fail)


def _download(paths, opener, payload: bytes, **kw):
    sha = hashlib.sha256(payload).hexdigest()
    return provision.download_checkpoint(
        paths, url="https://example.org/w.pth", sha256=sha, size=len(payload), opener=opener,
        sleep=lambda s: None, say=lambda m: None, **kw,
    )  # fmt: skip


def test_the_checkpoint_download_resumes_and_is_verified(tmp_path):
    paths = OcrPaths(tmp_path / "ocr", tmp_path / "tools")
    payload = np.random.default_rng(1).bytes(300_000)
    opener = FakeOpener(payload, break_after=100_000)
    assert _download(paths, opener, payload) == paths.checkpoint
    assert paths.checkpoint.read_bytes() == payload and not paths.partial.exists()
    assert opener.ranges[0] is None and opener.ranges[1].startswith("bytes=")
    assert int(opener.ranges[1][6:-1]) >= 100_000  # resumed, not restarted
    marker = install.read_marker(paths)
    assert marker["sha256"] == hashlib.sha256(payload).hexdigest()
    assert marker["size"] == len(payload)
    again = FakeOpener(payload)
    _download(paths, again, payload)
    assert again.ranges == []  # already verified: nothing fetched


def test_a_bad_download_is_deleted_and_a_partial_one_restarts(tmp_path):
    paths = OcrPaths(tmp_path / "ocr", tmp_path / "tools")
    payload = b"w" * 50_000
    wrong = FakeOpener(b"v" * 50_000)
    with pytest.raises(provision.SetupError, match="sha256"):
        _download(paths, wrong, payload)
    assert not paths.partial.exists() and not paths.checkpoint.exists()
    paths.partial.write_bytes(payload[:1000])
    ignoring = FakeOpener(payload, ignore_range=True)
    _download(paths, ignoring, payload)  # the server answered 200: written from the start
    assert paths.checkpoint.read_bytes() == payload
    paths.marker.unlink()  # an unmarked checkpoint is verified again, without a download
    none = FakeOpener(payload)
    _download(paths, none, payload)
    assert install.read_marker(paths)["size"] == len(payload) and none.ranges == []


def test_the_setup_command(tmp_path, capsys, monkeypatch):
    from ksmech.serve.app import cli, ocr_cli

    settings = _settings(tmp_path)
    paths = OcrPaths.of(settings)
    args = cli.build_parser().parse_args(["ocr-setup", "--skip-weights", "--uv", "/bin/uv"])
    assert cli.normalize_argv(["ocr-setup"]) == ["ocr-setup"]
    run = FakeRun(paths)
    assert ocr_cli.run(args, settings, run=run) == 1  # the environment, but no weights yet
    assert paths.python.is_file() and run.calls[-1][0][1] == "sync"
    assert "missing the verified checkpoint" in capsys.readouterr().out
    paths.weights.mkdir(parents=True)
    with open(paths.checkpoint, "wb") as fh:  # sparse: the size of the real checkpoint
        fh.truncate(install.CHECKPOINT_BYTES)
    write_marker(paths, CHECKPOINT_SHA256, install.CHECKPOINT_BYTES)
    args = cli.build_parser().parse_args(["ocr-setup", "--skip-env"])

    def no_network(*a, **k):
        raise AssertionError("the network was used")

    assert ocr_cli.run(args, settings, opener=no_network) == 0
    out = capsys.readouterr().out
    assert "already verified" in out and "the ocr capability is on" in out
    assert status("ocr", settings) == (True, None)
    args = cli.build_parser().parse_args(["ocr-setup", "--skip-weights"])
    # tools/ocr is locked, so the failing step is the sync (a lock runs only without uv.lock)
    assert ocr_cli.run(args, settings, run=FakeRun(paths, fail="sync")) == 1
    assert "failed with exit code 2" in capsys.readouterr().err


# ---- the real engine (skipped until `ksmech-serve ocr-setup` has run on this data root)

_REAL = OcrPaths.of(load_settings({}, data_root=REPO))


@pytest.mark.skipif(bool(install.missing(_REAL)), reason="OCR engine not installed (ocr-setup)")
def test_real_molscribe_reads_aspirin(tmp_path):
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 400)
    drawer.DrawMolecule(aspirin)
    drawer.FinishDrawing()
    service = OcrService(load_settings({}, data_root=REPO))
    try:
        out = service.recognize(drawer.GetDrawingText())
    finally:
        service.close()
    assert Chem.CanonSmiles(out.smiles) == Chem.MolToSmiles(aspirin)
    assert Chem.MolFromMolBlock(out.molfile) is not None and out.engine == install.ENGINE
