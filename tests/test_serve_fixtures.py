"""Shared serve fixtures and the no-xTB guard (a binding interface for every serve test module).

Import them with the repo's pattern:

    from tests.test_serve_fixtures import no_xtb, synthetic_context  # noqa: F401 (fixtures)

`no_xtb` is module-scoped and autouse: importing it into a module guards that module only, so
the guard never leaks into the xTB tests that run later in the full suite. `synthetic_root`
lives at a fixed path under the session's base temp directory and is built once per session
(reused when `<root>/.ready` exists), inside `xtb_forbidden()` whatever the fixture order.
Serve tests never use the conftest fixtures qm_record, fixture_records, sp_record,
token_loader or pair_kernels.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from tests.fixtures.molecules import MOLECULES

FORBIDDEN_MESSAGE = "xTB is forbidden in serve tests"
# (module, attribute path) of every entry point that runs xTB or a tblite SCF
XTB_TARGETS = (
    ("ksmech.qm.xtb_calc", "GFN2Calculator.run"),
    ("ksmech.qm.batch", "process_molecule"),
    ("ksmech.qm.ao", "aos_per_element"),
    ("tblite.interface", "Calculator"),
    ("ksmech.qm.geometry", "Calculator"),
    ("ksmech.qm.validation", "compute_fixture_records"),
)
SYNTHETIC_SPECIES = ("C[O-]", "CCl", "[OH-]", "[Cl-]", "CO", "COC")
SYNTHETIC_ROOT_NAME = "serve-synthetic"


def _forbidden(*args, **kwargs):
    raise AssertionError(FORBIDDEN_MESSAGE)


@contextlib.contextmanager
def xtb_forbidden():
    """Patch every xTB / tblite-SCF entry point to raise AssertionError. Only the tblite
    parameter export (`qm.ao.gfn2_shells`) stays allowed."""
    import importlib

    with pytest.MonkeyPatch.context() as mp:
        for module_name, attr in XTB_TARGETS:
            target = importlib.import_module(module_name)
            *owners, name = attr.split(".")
            for owner in owners:
                target = getattr(target, owner)
            mp.setattr(target, name, _forbidden)
        yield


@pytest.fixture(scope="module", autouse=True)
def no_xtb():
    with xtb_forbidden():
        yield


@pytest.fixture
def serve_settings(tmp_path):
    """A factory make(**overrides) -> ServeSettings over a fresh data root (tmp_path)."""
    from ksmech.serve.settings import load_settings

    def make(**overrides):
        return load_settings({}, data_root=tmp_path, **overrides)

    return make


def synthetic_keys() -> list[str]:
    """Token keys of the synthetic root: the SN2 species and the ten fixture molecules."""
    from rdkit import Chem

    from ksmech.data import chem

    smiles = list(SYNTHETIC_SPECIES) + list(MOLECULES.values())
    keys = [chem.unmapped_canonical(Chem.MolFromSmiles(s)) for s in smiles]
    return list(dict.fromkeys(keys))


def ensure_synthetic_root(root: Path) -> bool:
    """Build the synthetic tokens under `root` unless `<root>/.ready` exists. Returns True
    when it built. `.ready` is written last; the build runs inside `xtb_forbidden()`."""
    from ksmech.serve.featurize import FeaturizeStatus, SyntheticFeaturizer
    from ksmech.serve.settings import load_settings

    root = Path(root)
    if (root / ".ready").exists():
        return False
    settings = load_settings({}, data_root=root, chemistry="synthetic")
    with xtb_forbidden():
        featurizer = SyntheticFeaturizer(settings)
        for key in synthetic_keys():
            res = featurizer.ensure(key)
            assert res.status != FeaturizeStatus.FAILED, (key, res.code, res.detail)
    (root / ".ready").write_text("synthetic tokens built\n")
    return True


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.getbasetemp() / SYNTHETIC_ROOT_NAME
    ensure_synthetic_root(root)
    return root


@pytest.fixture(scope="session")
def synthetic_settings(synthetic_root):
    from ksmech.serve.settings import load_settings

    return load_settings({}, data_root=synthetic_root, chemistry="synthetic")


@pytest.fixture(scope="session")
def synthetic_runtime(synthetic_settings):
    from ksmech.serve.settings import TaskEnv
    from ksmech.serve.tasks import runtime

    return runtime(TaskEnv(synthetic_settings))


@pytest.fixture(scope="session")
def synthetic_loader(synthetic_runtime):
    return synthetic_runtime.loader


@pytest.fixture(scope="session")
def synthetic_context(synthetic_settings):
    """`build_context(synthetic_settings)`: InlineScheduler, synthetic featurizer, stub on."""
    from ksmech.serve.context import build_context

    return build_context(synthetic_settings)


def _saved_state(settings, pairs):
    from ksmech.serve.pools import SpeciesInput, initial_state, save_state

    state = initial_state([SpeciesInput(s, n) for s, n in pairs], settings.limits)
    save_state(state, settings)
    return state


@pytest.fixture(scope="session")
def sn2_state(synthetic_settings):
    """MeO- + MeCl, saved under the synthetic root."""
    return _saved_state(synthetic_settings, [("C[O-]", 1), ("CCl", 1)])


@pytest.fixture(scope="session")
def oh_sn2_state(synthetic_settings):
    """HO- + MeCl, saved under the synthetic root."""
    return _saved_state(synthetic_settings, [("[OH-]", 1), ("CCl", 1)])


@pytest.fixture
def serve_mini_processed(tmp_path, mini_result) -> Path:
    """A tmp `<data_root>/data/processed` (data_root = tmp_path) holding
    `pmechdb/steps.parquet` from the mini PMechDB fixture (`write_outputs`)."""
    from ksmech.data.ingest import write_outputs

    processed = tmp_path / "data" / "processed"
    write_outputs(mini_result, processed)
    return processed


# ---- sanity tests of the fixtures themselves


def test_guard_trips_inside_the_module():
    from ksmech.qm import batch
    from ksmech.qm.xtb_calc import GFN2Calculator, QMConfig

    with pytest.raises(AssertionError, match=FORBIDDEN_MESSAGE):
        batch.process_molecule(batch.MolTask("C", 1, QMConfig()))
    with pytest.raises(AssertionError, match=FORBIDDEN_MESSAGE):
        GFN2Calculator().run(None)
    import tblite.interface

    with pytest.raises(AssertionError, match=FORBIDDEN_MESSAGE):
        tblite.interface.Calculator("GFN2-xTB", None, None)
    import ksmech.qm.ao

    with pytest.raises(AssertionError, match=FORBIDDEN_MESSAGE):
        ksmech.qm.ao.aos_per_element(6)


def test_guard_blocks_load_record_probe(tmp_path):
    """`ao_layout` (what `load_record` calls) cannot reach tblite's SCF."""
    import numpy as np

    from ksmech.qm.ao import ao_layout

    with pytest.raises(AssertionError, match=FORBIDDEN_MESSAGE):
        ao_layout(np.array([6, 1]))


def test_guard_allows_the_parameter_export():
    from ksmech.qm.ao import element_ao_l, gfn2_shells

    assert 6 in gfn2_shells()
    assert element_ao_l(6) == (0, 1, 1, 1)


def test_xtb_forbidden_restores_on_exit(monkeypatch):
    from ksmech.qm import batch

    def sentinel(task):
        return []

    monkeypatch.setattr(batch, "process_molecule", sentinel)
    with xtb_forbidden():
        assert batch.process_molecule is _forbidden
    assert batch.process_molecule is sentinel


def test_synthetic_root_is_reused(synthetic_root):
    ready = synthetic_root / ".ready"
    assert ready.is_file()
    before = ready.stat().st_mtime_ns
    assert ensure_synthetic_root(synthetic_root) is False  # second request: no rebuild
    assert ready.stat().st_mtime_ns == before
    assert synthetic_root == synthetic_root.parent / SYNTHETIC_ROOT_NAME


def test_synthetic_root_holds_every_key(synthetic_settings, synthetic_loader):
    for key in synthetic_keys():
        assert synthetic_loader.has(key), key
    assert synthetic_settings.chemistry == "synthetic"


def test_states_are_saved(synthetic_settings, sn2_state, oh_sn2_state):
    from ksmech.serve.pools import load_state, state_path

    for state in (sn2_state, oh_sn2_state):
        assert state_path(state.state_id, synthetic_settings).is_file()
        assert load_state(state.state_id, synthetic_settings) == state
    assert sn2_state.pool_key == "[CH3][Cl]|0.[CH3][O]|-1"
    assert oh_sn2_state.pool_key == "[CH3][Cl]|0.[OH]|-1"


def test_serve_settings_factory(serve_settings, tmp_path):
    s = serve_settings(chemistry="cached")
    assert s.data_root == tmp_path.resolve() and s.chemistry == "cached"
    assert s.shared_cache_root == tmp_path.resolve()


def test_serve_mini_processed(serve_mini_processed, serve_settings):
    s = serve_settings()
    assert serve_mini_processed == s.processed
    assert (s.processed / "pmechdb" / "steps.parquet").is_file()


def test_synthetic_context_and_runtime(synthetic_context, synthetic_runtime, synthetic_settings):
    from ksmech.serve.worker import InlineScheduler

    assert isinstance(synthetic_context.scheduler, InlineScheduler)
    assert synthetic_context.chemistry == "synthetic"
    assert synthetic_runtime.settings == synthetic_settings
