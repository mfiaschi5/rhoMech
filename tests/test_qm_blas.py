"""qm.blas: the BLAS / LAPACK tblite runs on in the QM workers, and that it changes no result."""

import ctypes.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from ksmech.qm import blas
from ksmech.qm.batch import execute
from ksmech.qm.embed import embed, prepare_mol
from ksmech.qm.geometry import BOHR_ANG
from tests.fixtures.molecules import MOLECULES

needs_openblas = pytest.mark.skipif(blas.choose() is None, reason="no system LP64 OpenBLAS")

SINGLEPOINTS = """
import json, sys
import numpy as np
from ksmech.qm import blas
from ksmech.qm.geometry import new_calculator
out = {"lapack": blas.provider("dsygvd_")}
for name, (z, xyz) in json.load(sys.stdin).items():
    res = new_calculator(np.array(z, dtype=np.int32), np.array(xyz), 0, 0, "water").singlepoint()
    keys = ("energy", "orbital-energies", "charges", "gradient")
    out[name] = {k: np.asarray(res.get(k)).tolist() for k in keys}
print(json.dumps(out))
"""


@pytest.fixture(autouse=True)
def fresh_choice():
    blas._resolve.cache_clear()
    yield
    blas._resolve.cache_clear()


def test_with_preload_puts_the_library_first_once():
    assert blas.with_preload("", "/l/a.so") == "/l/a.so"
    assert blas.with_preload("/l/m.so:/l/a.so /l/n.so", "/l/a.so") == "/l/a.so /l/m.so /l/n.so"


def test_bundled_or_unloadable_choices_keep_the_bundled_copy(monkeypatch, caplog):
    monkeypatch.setenv(blas.ENV, "bundled")
    assert blas.choose() is None
    monkeypatch.setenv(blas.ENV, "/no/such/libblas.so")
    assert blas.choose() is None and "cannot be loaded" in caplog.text


def test_a_library_without_lapack_is_refused(monkeypatch, caplog):
    monkeypatch.setenv(blas.ENV, ctypes.util.find_library("m"))
    assert blas.choose() is None and "no dgemm_, dsygvd_" in caplog.text


def test_provider_names_the_library_or_bundled():
    assert blas.provider("ksmech_no_such_symbol_") == blas.BUNDLED
    assert os.path.basename(blas.provider("dladdr")).startswith("libc")


@needs_openblas
def test_spawned_workers_bind_tblite_to_the_chosen_library():
    before, got = os.environ.get("LD_PRELOAD"), []
    execute(["dsygvd_", "dgemm_"], 2, got.append, worker=blas.provider)
    assert got == [blas.choose()] * 2
    assert os.environ.get("LD_PRELOAD") == before  # the calling process is left as it was


def _singlepoints(geometries: dict, preload: bool) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    if preload:
        env["LD_PRELOAD"] = blas.choose()
    proc = subprocess.run(
        [sys.executable, "-c", SINGLEPOINTS],
        input=json.dumps(geometries),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return json.loads(proc.stdout)


@needs_openblas
def test_openblas_gives_the_bundled_singlepoints_on_the_fixture_molecules():
    geometries = {}
    for name, smi in MOLECULES.items():
        mol = prepare_mol(smi)
        cid = embed(mol, 1, seed=0)[0]
        z = [a.GetAtomicNum() for a in mol.GetAtoms()]
        geometries[name] = (z, (mol.GetConformer(cid).GetPositions() / BOHR_ANG).tolist())
    fast, ref = _singlepoints(geometries, True), _singlepoints(geometries, False)
    assert fast.pop("lapack") == blas.choose() and ref.pop("lapack") == blas.BUNDLED
    for name in MOLECULES:
        for key, value in ref[name].items():
            assert np.allclose(fast[name][key], value, rtol=0, atol=1e-9), (name, key)
