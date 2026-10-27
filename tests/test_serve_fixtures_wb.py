"""Fixtures of the molecule workbench tests: a synthetic data root of their own, its worker
runtime, the keys it holds, and `DeferredScheduler` (202 until the test lets the tasks run).

Import them with the repo's pattern:

    from tests.test_serve_fixtures_wb import no_xtb, wb_settings  # noqa: F401 (fixtures)

`wb_settings` is a module-scoped copy of the synthetic tokens and QM records (never the shared
session root, so workbench caches start empty in every module), synthetic EHT chemistry, no
xTB. `wb_key(settings, smiles)` featurizes a species that is not in the synthetic root.
"""

from __future__ import annotations

import threading

import pytest

from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    synthetic_root,
    xtb_forbidden,
)
from tests.test_serve_fixtures_api import copy_synthetic_root

pytestmark = pytest.mark.serve


@pytest.fixture(scope="module")
def wb_settings(tmp_path_factory, synthetic_root):  # noqa: F811
    from ksmech.serve.core.settings import load_settings

    root = copy_synthetic_root(synthetic_root, tmp_path_factory.mktemp("wb") / "root")
    return load_settings({}, data_root=root, chemistry="synthetic")


@pytest.fixture(scope="module")
def wb_runtime(wb_settings):
    from ksmech.serve.core.settings import TaskEnv
    from ksmech.serve.runtime.tasks import runtime

    return runtime(TaskEnv(wb_settings))


def canonical_key(smiles: str) -> str:
    from rdkit import Chem

    from ksmech.data import chem

    return chem.unmapped_canonical(Chem.MolFromSmiles(smiles))


def wb_key(rt, smiles: str) -> str:
    """The key of `smiles`, featurized with the synthetic chemistry of runtime `rt`."""
    from ksmech.serve.chem.featurize import FeaturizeStatus

    key = canonical_key(smiles)
    with xtb_forbidden():
        res = rt.featurizer.ensure(key)
    assert res.status != FeaturizeStatus.FAILED, (key, res.code, res.detail)
    return key


class DeferredScheduler:
    """Tasks wait until `drain()`: a submit returns a pending handle (the same one per key,
    and `find` returns it), so the API answers 202; `drain()` runs every queued task through
    an `InlineScheduler` and completes the handles, so the same request then answers 200."""

    def __init__(self, env) -> None:
        from ksmech.serve.compute.worker import InlineScheduler

        self.inline = InlineScheduler(env)
        self.queued: dict[str, tuple[str, dict, object]] = {}
        self.names: list[str] = []
        self._lock = threading.Lock()

    def submit(self, name, payload, *, priority, timeout_s, owner="", key=None, label=""):
        from ksmech.serve.compute.worker import TaskHandle, task_key

        key = key or task_key(name, payload)
        with self._lock:
            if key not in self.queued:
                self.queued[key] = (name, payload, TaskHandle(f"deferred-{len(self.names)}", label))
                self.names.append(name)
            return self.queued[key][2]

    def run(self, name, payload, *, priority, timeout_s, owner="", label=""):
        return self.inline.run(name, payload, priority=priority, timeout_s=timeout_s, label=label)

    def find(self, key):
        with self._lock:
            entry = self.queued.get(key)
        return None if entry is None else entry[2]

    def drain(self) -> int:
        with self._lock:
            items, self.queued = list(self.queued.values()), {}
        for name, payload, handle in items:
            done = self.inline.submit(name, payload, priority=0, timeout_s=60)
            handle.set_result(done.wait(None))
        return len(items)

    def status(self):
        return self.inline.status()

    def kill_owner(self, owner: str) -> bool:
        return False

    def close(self) -> None:
        return None


def test_wb_settings_hold_the_synthetic_tokens(wb_settings, wb_runtime):
    assert wb_runtime.loader.has(canonical_key("CO"))
    assert not wb_settings.workbench.exists() or not any(wb_settings.workbench_cubes.glob("*"))
