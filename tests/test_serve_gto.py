"""The vendored GFN2 Gaussian basis (`serve.chem.gto`, from `physics/gto.py` at 46adf6b) against
stored values: the code and its STO-nG table must stay bit-identical to the original."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.test_serve_fixtures import no_xtb  # noqa: F401 (fixture)

ORIGIN = "46adf6b"
STO_NG_SHA256 = "c66bf02df8ea7b8f46541c4d9d7ad811b028ab6a49f576f76ee08ff44bc8fa6a"
# methyl chloride-like geometry (Angstrom): C, three H, Cl
Z = np.array([6, 1, 1, 1, 17])
XYZ = np.array(
    [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63], [1.0, -1.0, -1.0]]
)
# stored with the original module (float.hex, exact)
S_ROW0 = ["0x1.b1025cb72f77fp-2"] * 3 + ["0x1.21901ffd69e88p-3"]  # S[0, 4:8]
S_SUM = "0x1.574bf76ec9f32p+4"
AO_SUM = "0x1.7a0cfdc8ab1aep+0"
AO_ROW0 = [0.33256539, 0.1291286, -0.38738579, 0.25825719]


def test_table_is_the_original():
    from ksmech.serve.chem import gto

    assert Path(gto.__file__).parent / "data" / "sto_ng.json" == gto.STO_NG_PATH
    assert hashlib.sha256(gto.STO_NG_PATH.read_bytes()).hexdigest() == STO_NG_SHA256


def test_stewart_sto3g_1s():
    from ksmech.qm.ao import gfn2_shells
    from ksmech.serve.chem.gto import element_shells, slater_to_gauss

    sh = slater_to_gauss(3, 1, 0, 1.0)
    assert np.allclose(sh.alpha, [2.227660584, 0.4057711562, 0.1098175104])
    raw = sh.coeff / (2 * sh.alpha / np.pi) ** 0.75
    assert np.allclose(raw, [0.1543289673, 0.5353281423, 0.4446345422])
    zeta = gfn2_shells()[1][0][2]
    assert np.allclose(element_shells(1)[0].alpha, sh.alpha * zeta**2)


def test_overlap_and_ao_values_match_stored_values():
    from ksmech.serve.chem.gto import ao_values, overlap_matrix

    S = overlap_matrix(Z, XYZ)
    assert S.shape == (16, 16) and np.allclose(S, S.T, atol=0)
    want = np.array([float.fromhex(x) for x in S_ROW0])
    assert np.allclose(S[0, 4:8], want, rtol=1e-13, atol=0)
    assert np.isclose(S.sum(), float.fromhex(S_SUM), rtol=1e-13, atol=0)
    v = ao_values(Z, XYZ, np.array([[0.2, 0.1, -0.3], [0.5, 0.5, 0.5]]))
    assert v.shape == (2, 16)
    assert np.isclose(v.sum(), float.fromhex(AO_SUM), rtol=1e-13, atol=0)
    assert np.allclose(v[0, :4], AO_ROW0, atol=1e-8)


def _origin_source() -> str | None:
    root = Path(__file__).resolve().parents[1]
    try:
        out = subprocess.run(
            ["git", "show", f"{ORIGIN}:src/ksmech/physics/gto.py"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def test_code_is_the_original_below_the_docstring():
    from ksmech.serve.chem import gto

    original = _origin_source()
    if original is None:
        pytest.skip(f"git object {ORIGIN} not available")
    here = Path(gto.__file__).read_text()
    assert here.split('"""', 2)[2] == original.split('"""', 2)[2]
