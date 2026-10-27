"""Embedding traces in serve (`serve.ml.checkpoints`, `serve.embedding.service`): the checkpoint
listing, the
`embedding` capability, `POST /api/embedding` as a job through the worker task, the stored
payload and page, the cache by checkpoint and pool state, the refusals with their codes, and
the worker's model cache. The checkpoints are tiny random models (`tests/serve_fake_model.py`)
on the synthetic chemistry; torch is never imported by the API modules."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from ksmech.serve.core import schemas as S
from tests.serve_fake_model import write_fake_run
from tests.test_serve_fixtures import no_xtb, synthetic_root  # noqa: F401 (fixtures)
from tests.test_serve_fixtures_api import (  # noqa: F401 (fixtures)
    api_client,
    copy_synthetic_root,
    services,
    synthetic_app,
    synthetic_settings,
)

pytestmark = pytest.mark.serve

SN2 = "C[O-].CCl"


def error_of(r, status: int, code: str) -> dict:
    assert r.status_code == status, r.text
    err = r.json()["error"]
    assert err["code"] == code, err
    return err


@pytest.fixture(scope="module")
def emb_settings(tmp_path_factory, synthetic_root):  # noqa: F811
    """Synthetic tokens and a clean fake run `fake/tiny` (dense) on a new data root."""
    from ksmech.serve.core.settings import load_settings

    root = copy_synthetic_root(synthetic_root, tmp_path_factory.mktemp("embedding"))
    write_fake_run(root / "data" / "models" / "fake" / "tiny", "dense")
    return load_settings({}, data_root=root, chemistry="synthetic")


@pytest.fixture(scope="module")
def emb_app(emb_settings):
    with api_client(emb_settings) as client:
        yield client


def _trace(client, **body) -> dict:
    """POST /api/embedding (jobs run inline) -> the done job's EmbeddingResultOut dict."""
    r = client.post("/api/embedding", json={"run": "tiny"} | body)
    assert r.status_code == 202, r.text
    job = S.JobOut.model_validate(client.get(r.json()["status_url"]).json())
    assert job.kind == "embedding" and job.status == "done", job.error
    assert job.result_kind == "embedding"
    assert isinstance(job.result, S.EmbeddingResultOut)
    return job.result.model_dump(mode="json")


def test_models_lists_the_checkpoints_with_their_epoch_and_provenance(emb_app):
    out = S.ModelsOut.model_validate(emb_app.get("/api/models").json())
    assert out.embedding.enabled and out.embedding.reason is None
    by_file = {c.file: c for c in out.checkpoints if c.run == "tiny"}
    assert set(by_file) == {"best", "last"}
    best, last = by_file["best"], by_file["last"]
    assert best.source == "fake" and best.encoder == "dense" and best.model["layers"] == 2
    assert best.epoch == 1 and best.metrics["val_top1"] == 0.5  # the trainer's best line
    assert last.epoch == 2  # the last line
    assert best.clean is True and best.usable and best.reason is None


def test_the_capability_needs_a_checkpoint(synthetic_app, emb_app):  # noqa: F811
    on = emb_app.get("/api/meta").json()["capabilities"]["embedding"]
    assert on == {"enabled": True, "reason": None}
    off = synthetic_app.get("/api/meta").json()["capabilities"]["embedding"]
    assert off["enabled"] is False and "no checkpoint" in off["reason"]
    err = error_of(synthetic_app.post("/api/embedding", json={"smiles": SN2, "run": "x"}), 503,
                   "FEATURE_DISABLED")  # fmt: skip
    assert err["detail"] == {"capability": "embedding", "reason": off["reason"]}


def test_the_capability_is_off_without_torch(emb_app, monkeypatch):
    from ksmech.serve.core import capabilities

    monkeypatch.setitem(capabilities.CAPABILITIES, "embedding", (False, "no torch (test)"))
    meta = emb_app.get("/api/meta").json()["capabilities"]["embedding"]
    assert meta == {"enabled": False, "reason": "no torch (test)"}
    error_of(emb_app.post("/api/embedding", json={"smiles": SN2, "run": "tiny"}), 503,
             "FEATURE_DISABLED")  # fmt: skip


def test_a_trace_is_a_job_whose_payload_and_page_are_stored(emb_app, emb_settings):
    res = _trace(emb_app, smiles=SN2)
    eid = res["embedding_id"]
    assert not res["cached"] and res["seconds"] > 0
    assert res["encoder"] == "dense" and res["epoch"] == 1 and res["n_layers"] == 2
    assert res["max_error"] < 1e-4  # the rebuilt layers are the model's
    assert res["page_url"] == f"/api/embedding/{eid}/page"
    data = emb_app.get(res["payload_url"]).json()
    n = res["n_atoms"] + res["n_orbitals"] + 1
    assert len(data["tokens"]) == n and data["tokens"][0]["kind"] == "mech"
    assert data["meta"]["run"] == "fake/tiny" and data["layers"]["h"]["shape"][1] == n
    html = emb_app.get(res["page_url"]).text
    assert "/*PAYLOAD*/" not in html and "<script src" not in html
    assert len(html.encode()) == res["page_bytes"]
    dark = emb_app.get(res["page_url"], params={"theme": "dark"}).text
    assert dark.startswith('<!doctype html>\n<html data-theme="dark"')
    error_of(emb_app.get(res["page_url"], params={"theme": "blue"}), 422, "REQUEST_INVALID")
    stored = sorted(p.name for p in emb_settings.embedding.iterdir())
    assert stored == [f"{eid}.html", f"{eid}.json", f"{eid}.meta.json"]
    assert emb_settings.embedding.is_relative_to(emb_settings.data_root / "data" / "serve")


def test_a_trace_is_cached_by_checkpoint_and_pool_state(emb_app):
    first = _trace(emb_app, smiles=SN2)
    tasks = services(emb_app).scheduler.history
    n_traced = sum(name == "embedding" for name, _ in tasks)
    again = _trace(emb_app, state_id=first["state_id"])
    assert again["cached"] and again["seconds"] == 0
    assert again["embedding_id"] == first["embedding_id"]
    assert sum(name == "embedding" for name, _ in tasks) == n_traced  # no second task
    other = _trace(emb_app, smiles=SN2, file="last")
    assert other["embedding_id"] != first["embedding_id"] and not other["cached"]


def test_refusals_have_codes(emb_app, emb_settings):
    post = emb_app.post
    error_of(post("/api/embedding", json={"run": "tiny"}), 422, "REQUEST_INVALID")
    both = {"run": "tiny", "smiles": SN2, "state_id": "s" + "0" * 20}
    error_of(post("/api/embedding", json=both), 422, "REQUEST_INVALID")
    error_of(post("/api/embedding", json={"run": "nope", "smiles": SN2}), 404,
             "CHECKPOINT_NOT_FOUND")  # fmt: skip
    error_of(post("/api/embedding", json={"run": "../x", "smiles": SN2}), 422, "REQUEST_INVALID")
    error_of(post("/api/embedding", json={"run": "tiny", "smiles": "C1CC"}), 422,
             "INVALID_SMILES")  # fmt: skip
    error_of(post("/api/embedding", json={"run": "tiny", "state_id": "s" + "0" * 20}), 404,
             "POOL_NOT_FOUND")  # fmt: skip
    error_of(emb_app.get("/api/embedding/b" + "0" * 24), 404, "EMBEDDING_NOT_FOUND")
    error_of(emb_app.get("/api/embedding/x/page"), 404, "EMBEDDING_NOT_FOUND")


def test_a_dirty_run_is_listed_with_its_reason_and_refused(emb_app, emb_settings):
    write_fake_run(emb_settings.data_root / "data" / "models" / "fake" / "dirty", dirty=True)
    rows = [c for c in emb_app.get("/api/models").json()["checkpoints"] if c["run"] == "dirty"]
    assert rows and all(not c["usable"] and "dirty tree" in c["reason"] for c in rows)
    err = error_of(emb_app.post("/api/embedding", json={"run": "dirty", "smiles": SN2}), 409,
                   "CHECKPOINT_REFUSED")  # fmt: skip
    assert "dirty tree" in err["message"]


def test_the_worker_refuses_a_dirty_header_that_run_json_hides(tmp_path):
    """run.json says clean, the checkpoint's own header says dirty: load_model refuses it in
    the worker and the job fails with CHECKPOINT_REFUSED."""
    from ksmech.serve.core.errors import ServeError
    from ksmech.serve.embedding.service import _signature, cached_model

    run = tmp_path / "models" / "fake" / "mixed"
    write_fake_run(run, dirty=True, files=("best",))
    doc = json.loads((run / "run.json").read_text())
    doc["provenance"]["code_dirty"] = False
    (run / "run.json").write_text(json.dumps(doc))
    with pytest.raises(ServeError) as exc:
        cached_model(run / "best.pt", _signature(run / "best.pt"))
    assert exc.value.code == "CHECKPOINT_REFUSED" and "load_model refused" in exc.value.message


def test_the_worker_keeps_one_model_per_file_version(tmp_path):
    from ksmech.serve.core.errors import ServeError
    from ksmech.serve.embedding.service import _signature, cached_model

    (path,) = write_fake_run(tmp_path / "fake" / "cache", files=("best",))
    sig = _signature(path)
    model, ckpt = cached_model(path, sig)
    assert cached_model(path, sig)[0] is model and ckpt["epoch"] == 1
    os.utime(path, ns=(sig[0] + 10**9, sig[0] + 10**9))
    with pytest.raises(ServeError) as exc:
        cached_model(path, (sig[0] - 1, sig[1]))
    assert exc.value.code == "CHECKPOINT_CHANGED"


def test_the_api_modules_never_import_torch():
    code = (
        "import sys\n"
        "import ksmech.serve.app.api, ksmech.serve.embedding.service, ksmech.serve.ml.checkpoints\n"
        "import ksmech.serve.ml.runs, ksmech.serve.ml.router\n"
        "import ksmech.serve.predict.network_predictor, ksmech.model.decode\n"
        "from ksmech.serve.ml.checkpoints import missing_modules\n"
        "missing_modules()\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
