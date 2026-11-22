"""The GFN2-xTB basis of tblite 0.7.0, built from `gfn2_params.json`.

Every shell is a contracted Gaussian fitted to a Slater function (STO-nG, `slater_to_gauss`
in tblite's basis/slater.f90), each primitive normalized as tblite does it. Shells keep
tblite's order within an element (a transition metal lists d before s and p), so the AO
order of a molecule is tblite's: atoms in input order, shells in element order, and within a
shell the spherical functions in tblite's order (p: y, z, x; d: xy, yz, z2, xz, x2-y2).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cache, cached_property, lru_cache
from pathlib import Path

import numpy as np

PARAMS_PATH = Path(__file__).with_name("gfn2_params.json")
MAX_ELEM = 86
DOUBLE_FACTORIAL = (1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0)  # (2l-1)!!
N_SPH = (1, 3, 5)
N_CART = (1, 3, 6)


@cache
def params() -> dict:
    """The parsed parameter file: {section: {name: ndarray in Fortran dims}}."""
    raw = json.loads(PARAMS_PATH.read_text())
    out = {}
    for section, table in raw.items():
        if not isinstance(table, dict):
            continue
        out[section] = {
            name: np.reshape(np.array(v["values"]), v["dims"], order="F")
            for name, v in table.items()
        }
    return out


def scalar(name: str) -> float:
    return float(params()["gfn2"][name][0])


@dataclass(frozen=True)
class CGTO:
    """One contracted shell: angular momentum, Slater exponent, primitives."""

    l: int  # noqa: E741 - the quantum number's name
    zeta: float
    alpha: np.ndarray
    coeff: np.ndarray  # contraction coefficient x primitive normalization

    @property
    def nprim(self) -> int:
        return len(self.alpha)


def _sto_ng(ng: int, n: int, l: int, zeta: float) -> tuple[np.ndarray, np.ndarray]:  # noqa: E741
    """tblite's slater_to_gauss: exponents and normalized coefficients."""
    sl = params()["slater"]
    if ng == 6 and n == 6:
        key = {0: "6s", 1: "6p"}[l]
        alpha, coeff = sl[f"pAlpha{key}"], sl[f"pCoeff{key}"]
    else:
        if n > 5 or n <= l:
            raise ValueError(f"no STO-{ng}G fit for n={n}, l={l}")
        ityp = {0: n, 1: 4 + n, 2: 7 + n}[l] - 1
        alpha = sl[f"pAlpha{ng}"][..., ityp] if ng > 1 else sl["pAlpha1"][ityp : ityp + 1]
        coeff = sl[f"pCoeff{ng}"][..., ityp] if ng > 1 else np.ones(1)
    alpha = np.asarray(alpha, dtype=float) * zeta**2
    norm = (2.0 / math.pi * alpha) ** 0.75 * np.sqrt(4.0 * alpha) ** l
    norm /= math.sqrt(DOUBLE_FACTORIAL[l])
    return alpha, np.asarray(coeff, dtype=float) * norm


@cache
def element_shells(z: int) -> tuple[CGTO, ...]:
    """The shells of element z in tblite's order."""
    if not 1 <= z <= MAX_ELEM:
        raise ValueError(f"GFN2-xTB has no parameters for Z={z}")
    g = params()["gfn2"]
    shells = []
    for ish in range(int(g["nshell"][z - 1])):
        l = int(g["ang_shell"][ish, z - 1])  # noqa: E741
        n = int(g["principal_quantum_number"][ish, z - 1])
        ng = int(g["number_of_primitives"][ish, z - 1])
        zeta = float(g["slater_exponent"][ish, z - 1])
        alpha, coeff = _sto_ng(ng, n, l, zeta)
        shells.append(CGTO(l, zeta, alpha, coeff))
    return tuple(shells)


@dataclass(frozen=True)
class MolBasis:
    """The shells of one molecule: owning atom, element shell index, first AO."""

    z: np.ndarray
    sh_atom: np.ndarray
    sh_elem: np.ndarray  # index of the shell within its element
    sh_ao: np.ndarray
    nao: int

    def shell(self, s: int) -> CGTO:
        return element_shells(int(self.z[self.sh_atom[s]]))[int(self.sh_elem[s])]

    @cached_property
    def ao_atom(self) -> np.ndarray:
        out = np.empty(self.nao, dtype=int)
        for s in range(len(self.sh_atom)):
            n = N_SPH[self.shell(s).l]
            out[self.sh_ao[s] : self.sh_ao[s] + n] = self.sh_atom[s]
        return out


def mol_basis(z: np.ndarray) -> MolBasis:
    """The basis of a molecule, cached by its element sequence (treat it as read-only)."""
    return _mol_basis(tuple(int(x) for x in np.asarray(z).ravel()))


@lru_cache(maxsize=8192)
def _mol_basis(zt: tuple[int, ...]) -> MolBasis:
    z = np.array(zt, dtype=int)
    atom, elem, ao, nao = [], [], [], 0
    for iat, zi in enumerate(z):
        for ish, cg in enumerate(element_shells(int(zi))):
            atom.append(iat)
            elem.append(ish)
            ao.append(nao)
            nao += N_SPH[cg.l]
    return MolBasis(z, np.array(atom), np.array(elem), np.array(ao), nao)


def integral_cutoff() -> float:
    """tblite's primitive screening threshold on a*b/(a+b) R^2. `integral_cutoff(acc)` clips acc
    between bounds given in the wrong order, so it returns 25 for every accuracy."""
    return 25.0


def pair_cutoff(z: np.ndarray) -> float:
    """tblite's `get_cutoff`: the neighbour-list cutoff of a molecule in Bohr."""
    min_alpha = min(float(cg.alpha.min()) for zi in set(z.tolist()) for cg in element_shells(zi))
    return min(math.sqrt(2.0 * integral_cutoff() / min_alpha), 40.0)
