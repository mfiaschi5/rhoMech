"""The GFN2-xTB valence basis as contracted Gaussians, and analytic two-centre overlaps.

Vendored into serve, bit-identical, from `src/ksmech/physics/gto.py` at commit 46adf6b (with
its table `physics/data/sto_ng.json`, now `serve/chem/data/sto_ng.json`): main removed the module
with the contact model (spec v9; the contact code is archived on `archive/stage-b`), and the
workbench's orbital and density surfaces (`cubes`, `densities`) and the synthetic chemistry
(`synthetic`) still evaluate AOs and overlaps with it. Only this paragraph differs from the
original (the table path is relative to the module); `tests/test_serve_gto.py` pins stored
values and compares the code with the original.

Every shell is tblite's STO-nG expansion of a Slater function:

- the Slater exponent, principal quantum number and number of primitives come from the GFN2
  parametrization that tblite itself exports (`ksmech.qm.ao.gfn2_shells`);
- the expansion (Gaussian exponents for zeta = 1, contraction coefficients) is Stewart's table
  as tblite ships it (`data/sto_ng.json`, extracted from tblite v0.7.0 `basis/slater.f90`, its
  sha256 recorded in the file); exponents are scaled by zeta^2;
- each primitive is normalized as in tblite: (2 alpha / pi)^(3/4) (4 alpha)^(l/2) / sqrt((2l-1)!!).

Cartesian components and the spherical transformation also follow tblite: p in the order
(y, z, x); d from (xx, yy, zz, xy, xz, yz) by tblite's `dtrafo` to (xy, yz, z2, xz, x2-y2).
Inside this module lengths are in bohr; the public functions take Angstrom.

Overlaps use the Gaussian product rule with the Obara-Saika recursion for the three 1-D
factors, vectorized over primitive pairs and over any number of centre pairs at once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from ksmech.qm.ao import gfn2_shells
from ksmech.qm.geometry import BOHR_ANG

STO_NG_PATH = Path(__file__).parent / "data" / "sto_ng.json"
_DFACT = {0: 1.0, 1: 1.0, 2: 3.0}  # (2l - 1)!!

# cartesian exponents (lx, ly, lz) per component, in tblite's order
CART = {
    0: np.array([[0, 0, 0]]),
    1: np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),  # y, z, x
    2: np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1]]),
}
_S3 = np.sqrt(3.0)
# spherical <- cartesian, rows (xy, yz, z2, xz, x2-y2), columns (xx, yy, zz, xy, xz, yz)
SPH = {
    0: np.eye(1),
    1: np.eye(3),
    2: np.array(
        [
            [0.0, 0.0, 0.0, _S3, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, _S3],
            [-0.5, -0.5, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, _S3, 0.0],
            [_S3 / 2, -_S3 / 2, 0.0, 0.0, 0.0, 0.0],
        ]
    ),
}


@dataclass(frozen=True)
class Shell:
    l: int  # noqa: E741 - the angular momentum quantum number
    alpha: np.ndarray  # (ng,) bohr^-2
    coeff: np.ndarray  # (ng,) contraction coefficients including primitive normalization

    @property
    def n_sph(self) -> int:
        return 2 * self.l + 1


@cache
def _sto_tables() -> dict:
    return json.loads(STO_NG_PATH.read_text())


def slater_to_gauss(ng: int, n: int, l: int, zeta: float) -> Shell:  # noqa: E741
    """tblite's `slater_to_gauss` with normalization: an STO-ngG shell of exponent zeta."""
    data = _sto_tables()
    if ng == 6 and n == 6:
        entry = data["six_primitive_n6"][f"6{'sp'[l]}"]
        alpha, coeff = np.array(entry["alpha"]), np.array(entry["coeff"])
    else:
        ityp = {0: n, 1: 4 + n, 2: 7 + n}[l]
        table = data["tables"][str(ng)]
        alpha = np.array(table["alpha"][ityp - 1], dtype=float)
        coeff = np.array(table["coeff"][ityp - 1], dtype=float)
    alpha = alpha * zeta**2
    norm = (2.0 * alpha / np.pi) ** 0.75 * np.sqrt(4.0 * alpha) ** l / np.sqrt(_DFACT[l])
    return Shell(l, alpha, coeff * norm)


@cache
def element_shells(z: int) -> tuple[Shell, ...]:
    """The shells of element `z`, in tblite's AO order."""
    return tuple(slater_to_gauss(ng, n, ell, zeta) for n, ell, zeta, ng in gfn2_shells()[int(z)])


def _overlap_1d(la: int, lb: int, X, p, xpa, xpb, s00) -> list[list[np.ndarray]]:
    """Obara-Saika 1-D overlaps s[i][j], i <= la, j <= lb, for one cartesian direction."""
    s = [[None] * (lb + 1) for _ in range(la + 1)]
    s[0][0] = s00
    half = 0.5 / p
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            if j == 0:
                prev = s[i - 1][0]
                term = (i - 1) * s[i - 2][0] if i >= 2 else 0.0
                s[i][0] = xpa * prev + half * term
            else:
                prev = s[i][j - 1]
                term = (i * s[i - 1][j - 1] if i else 0.0) + (
                    (j - 1) * s[i][j - 2] if j >= 2 else 0.0
                )
                s[i][j] = xpb * prev + half * term
    return s


def shell_overlap(sa: Shell, sb: Shell, R: np.ndarray) -> np.ndarray:
    """(N, n_sph_a, n_sph_b) overlaps of shell `sa` at the origin with `sb` at each R (bohr)."""
    R = np.asarray(R, dtype=float).reshape(-1, 3)
    a, b = sa.alpha[None, :, None], sb.alpha[None, None, :]
    p = a + b
    mu = a * b / p
    tables = []
    for d in range(3):
        X = R[:, d][:, None, None]
        s00 = np.sqrt(np.pi / p) * np.exp(-mu * X**2)
        tables.append(_overlap_1d(sa.l, sb.l, X, p, b / p * X, -a / p * X, s00))
    w = sa.coeff[:, None] * sb.coeff[None, :]
    cart = np.empty((len(R), len(CART[sa.l]), len(CART[sb.l])))
    for i, ea in enumerate(CART[sa.l]):
        for j, eb in enumerate(CART[sb.l]):
            prim = tables[0][ea[0]][eb[0]] * tables[1][ea[1]][eb[1]] * tables[2][ea[2]][eb[2]]
            cart[:, i, j] = np.einsum("nab,ab->n", prim, w)
    return np.einsum("ic,ncd,jd->nij", SPH[sa.l], cart, SPH[sb.l])


def atom_ao_slices(z: np.ndarray) -> list[slice]:
    starts = np.cumsum([0] + [sum(s.n_sph for s in element_shells(int(x))) for x in z])
    return [slice(int(starts[i]), int(starts[i + 1])) for i in range(len(z))]


def atom_pair_blocks(za: int, zb: int, R: np.ndarray) -> np.ndarray:
    """(N, nao_a, nao_b) overlap blocks of an atom of element za at the origin with an atom of
    element zb at each R (bohr), every shell pair."""
    sh_a, sh_b = element_shells(za), element_shells(zb)
    na, nb = sum(s.n_sph for s in sh_a), sum(s.n_sph for s in sh_b)
    out = np.zeros((len(R), na, nb))
    ia = 0
    for sa in sh_a:
        ib = 0
        for sb in sh_b:
            out[:, ia : ia + sa.n_sph, ib : ib + sb.n_sph] = shell_overlap(sa, sb, R)
            ib += sb.n_sph
        ia += sa.n_sph
    return out


def overlap_matrix(z: np.ndarray, xyz_ang: np.ndarray) -> np.ndarray:
    """The full AO overlap matrix of a molecule (tblite's AO order)."""
    z = np.asarray(z, dtype=int)
    xyz = np.asarray(xyz_ang, dtype=float) / BOHR_ANG
    sl = atom_ao_slices(z)
    S = np.zeros((sl[-1].stop, sl[-1].stop))
    for za in np.unique(z):
        for zb in np.unique(z):
            ii, jj = np.meshgrid(np.flatnonzero(z == za), np.flatnonzero(z == zb), indexing="ij")
            ii, jj = ii.ravel(), jj.ravel()
            blocks = atom_pair_blocks(int(za), int(zb), xyz[jj] - xyz[ii])
            for k, (i, j) in enumerate(zip(ii, jj, strict=True)):
                S[sl[i], sl[j]] = blocks[k]
    return S


def ao_values(z: np.ndarray, xyz_ang: np.ndarray, points_ang: np.ndarray) -> np.ndarray:
    """(n_points, nao) values of every AO at the points (Angstrom in, bohr^-3/2 out)."""
    z = np.asarray(z, dtype=int)
    xyz = np.asarray(xyz_ang, dtype=float) / BOHR_ANG
    pts = np.asarray(points_ang, dtype=float).reshape(-1, 3) / BOHR_ANG
    cols = []
    for zi, centre in zip(z, xyz, strict=True):
        r = pts - centre
        r2 = np.sum(r * r, axis=1)
        for sh in element_shells(int(zi)):
            radial = np.exp(-r2[:, None] * sh.alpha[None, :]) @ sh.coeff
            mono = np.prod(r[:, None, :] ** CART[sh.l][None, :, :], axis=2)
            cols.append((radial[:, None] * mono) @ SPH[sh.l].T)
    return np.concatenate(cols, axis=1)
