"""API fixtures of the serve app (w3-integration): the synthetic app and its HTTP helpers.

Import them with the repo's pattern:

    from tests.test_serve_fixtures_api import no_xtb, synthetic_app  # noqa: F401 (fixtures)

`synthetic_app` is a `TestClient` (lifespan entered) over
`create_app(synthetic_settings, scheduler=InlineScheduler(TaskEnv(synthetic_settings)),
jobs_inline=True)`: synthetic EHT chemistry, every job run inline
before its POST returns. `fresh_synthetic_settings` is a new data root that holds a copy of the
synthetic tokens, QM records and Wiberg sidecars, so a test gets an empty curation database and
job store without rebuilding any token. `NeverScheduler` never finishes a task (202 Pending).
"""

from __future__ import annotations

import contextlib
import shutil
import threading
from pathlib import Path

import pytest

from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    serve_settings,
    synthetic_root,
    synthetic_settings,
)

pytestmark = pytest.mark.serve

SN2_REACTANTS = ["C[O-]", "CCl"]
SN2_REQUEST = {"reactants_smiles": SN2_REACTANTS, "predictor": "floor", "beam": 2, "max_steps": 1}
SN2_TUPLE = {
    "donor_species": "C[O-]",
    "donor_site_element": "O",
    "acceptor_species": "CCl",
    "acceptor_site_element": "C",
    "acceptor_centre_elements": ["C", "Cl"],
    "valid": True,
}
CH_TUPLE = SN2_TUPLE | {"acceptor_site_element": "H", "acceptor_centre_elements": ["C", "H"]}
ARROW_KEYS = ("donor", "donor_site", "acceptor", "acceptor_site")
SPEC_RE = r"^v9\+[0-9a-f]{12};tok\d+;smp\d+;qm\d+$"
COPIED_DIRS = ("tokens", "qm", "bond_index")


class NeverScheduler:
    """A `Scheduler` whose tasks never finish: every submit returns a pending handle (the same
    one for the same key), so `worker.interactive` raises `Pending`. Records every submit."""

    def __init__(self) -> None:
        from ksmech.serve.worker import TaskHandle

        self._handle = TaskHandle
        self.handles: dict[str, object] = {}
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def submit(self, name, payload, *, priority, timeout_s, owner="", key=None, label=""):
        from ksmech.serve.worker import task_key

        key = key or task_key(name, payload)
        with self._lock:
            self.calls.append((name, label))
            if key not in self.handles:
                self.handles[key] = self._handle(f"never-{len(self.handles) + 1}", label)
            return self.handles[key]

    def run(self, name, payload, *, priority, timeout_s, owner="", label=""):
        raise AssertionError("NeverScheduler.run would block forever")

    def find(self, key):
        with self._lock:
            return self.handles.get(key)

    def status(self):
        from ksmech.serve.worker import WorkerStatus

        label = self.calls[-1][1] if self.calls else None
        return WorkerStatus(True, 1, [], bool(self.calls), None, label, None, 0, 0, False)

    def kill_owner(self, owner: str) -> bool:
        return False

    def close(self) -> None:
        return None


@contextlib.contextmanager
def api_client(settings, **kwargs):
    """A TestClient (lifespan entered) over `create_app(settings, ...)`; the scheduler defaults
    to an InlineScheduler and jobs run inline."""
    from fastapi.testclient import TestClient

    from ksmech.serve.api import create_app
    from ksmech.serve.settings import TaskEnv
    from ksmech.serve.worker import InlineScheduler

    kwargs.setdefault("scheduler", InlineScheduler(TaskEnv(settings)))
    kwargs.setdefault("jobs_inline", True)
    app = create_app(settings, **kwargs)
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def synthetic_app(synthetic_settings):  # noqa: F811
    with api_client(synthetic_settings) as client:
        yield client


def copy_synthetic_root(src: Path, dst: Path) -> Path:
    """Copy the synthetic tokens, QM records and Wiberg sidecars of `src` into the data root
    `dst` (nothing else: no pools, jobs or curation database). Returns `dst`."""
    for name in COPIED_DIRS:
        source = Path(src) / "data" / "serve" / name
        if source.is_dir():
            shutil.copytree(source, Path(dst) / "data" / "serve" / name, dirs_exist_ok=True)
    return Path(dst)


@pytest.fixture
def fresh_synthetic_settings(tmp_path, synthetic_root):  # noqa: F811
    """Synthetic settings (stub predictor on) on a new data root holding a copy of the
    synthetic tokens; the curation database, job store and exports start empty."""
    from ksmech.serve.settings import load_settings

    root = copy_synthetic_root(synthetic_root, tmp_path / "root")
    return load_settings({}, data_root=root, chemistry="synthetic")


def services(client):
    """The `api.Services` of the client's app."""
    return client.app.state.services


def predict(client, **overrides) -> dict:
    """POST /api/predict (default: the stub on MeO- + MeCl, beam 2, max steps 1) and return
    the finished JobOut; the job must be done."""
    body = SN2_REQUEST | overrides
    r = client.post("/api/predict", json=body)
    assert r.status_code == 202, r.text
    job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "done", job.get("error")
    return job


def root_node(net: dict) -> dict:
    return next(n for n in net["nodes"] if n["is_root"])


def edge_to(net: dict, pool_key: str) -> dict:
    return next(e for e in net["edges"] if e["target_pool_key"] == pool_key)


def candidates(client, state_id: str, *, predictor: str = "floor", **match) -> list[dict]:
    """Every candidate tuple of the state (excluded ones too) whose fields equal `match`."""
    params = {"predictor": predictor, "limit": 10000, "include_excluded": "true"}
    r = client.get(f"/api/pool/{state_id}/candidates", params=params)
    assert r.status_code == 200, r.text
    rows = r.json()["candidates"]
    return [c for c in rows if all(c.get(k) == v for k, v in match.items())]


def arrow_of(tuple_row: dict) -> dict:
    return {k: int(tuple_row[k]) for k in ARROW_KEYS}


# ---- sanity tests of the fixtures themselves


def test_synthetic_app_answers(synthetic_app):
    r = synthetic_app.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["chemistry"] == "synthetic"
    assert r.headers["x-request-id"]
    svc = services(synthetic_app)
    assert svc.runner.inline is True
    assert type(svc.scheduler).__name__ == "InlineScheduler"


def test_fresh_settings_hold_the_tokens(fresh_synthetic_settings, synthetic_settings):  # noqa: F811
    from ksmech.serve.featurize import make_featurizer

    s = fresh_synthetic_settings
    assert s.data_root != synthetic_settings.data_root
    loader = make_featurizer(s).loader()
    assert all(loader.has(k) for k in SN2_REACTANTS)
    assert not s.curation_db.exists()
    assert not s.jobs.exists()


def test_never_scheduler_is_pending():
    from ksmech.serve.errors import Pending
    from ksmech.serve.worker import interactive

    sched = NeverScheduler()
    with pytest.raises(Pending) as first:
        interactive(sched, "ping", {}, label="never", wait_s=0.01, timeout_s=1)
    with pytest.raises(Pending) as again:
        interactive(sched, "ping", {}, label="never", wait_s=0.01, timeout_s=1)
    assert first.value.task_id == again.value.task_id
    assert len(sched.handles) == 1
