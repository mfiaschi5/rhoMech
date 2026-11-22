"""A torch reference (fp64, CPU) of the GFN2-xTB terms beyond the one-electron integrals.

Each term is written as an energy of the geometry and of the SCF variables (shell charges qsh,
atomic charges qat, atomic dipoles dp and quadrupoles qp), so autograd gives both its gradient
at fixed SCF variables and its potentials. It follows tblite 0.7.0 with dftd4 4.2.0 and
mctc-lib 0.5.2 term by term and is slow: it is the oracle the CUDA kernels are checked
against, not an engine. Units are Hartree and Bohr.

- `cn_dexp`: the double-exponential coordination number of H0 and of the multipole radii.
- `selfenergies`: the shell self-energies of H0, se = se0 - kcn * CN.
- `e_rep`: the screened Coulomb repulsion.
- `e_es2`, `e_es3`: the shell-resolved isotropic electrostatics and onsite third order.
- `e_aes`: the damped anisotropic (dipole and quadrupole) electrostatics, its radii set by CN.
- `cn_d4`, `e_d4_two`, `e_d4_three`: D4 dispersion, its two-body part with the SCF charges,
  the three-body (ATM) part with zero charges.
- `e_alpb`, `e_cds`, `SHIFT`: ALPB solvation in water, polar, non-polar (with the H-bond term
  on the charges) and the constant shift.
- `electronic_entropy`: the Fermi smearing term -TS of tblite at 300 K.

Conventions: dp is (nat, 3), qp is (nat, 6) with tblite's components (xx, xy, yy, xz, yz, zz).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from itertools import combinations

import numpy as np
import torch
from scipy.integrate import lebedev_rule

from ksmech.qm.gpu.basis import MolBasis, params, scalar

F64 = torch.float64
AATOAU = 1.0 / 0.529177210903
# mctc-lib's kcal/mol -> Hartree from CODATA 2018 (hartree = me c^2 alpha^2, NA, 4.184 J/cal)
_HARTREE_KCAL = 9.1093837015e-31 * 299792458.0**2 * 7.2973525693e-3**2 * 6.02214076e23 / 4184
KCALTOAU = 1.0 / _HARTREE_KCAL
KT_300K = 300.0 * 3.166808578545117e-06  # tblite's default electronic temperature, Hartree

CN_DEXP_CUTOFF = 25.0  # mctc-lib dexp default
CN_D4_CUTOFF = 30.0  # dftd4 realspace_cutoff%cn default
REP_CUTOFF = 25.0  # tb_repulsion default
DISP2_CUTOFF, DISP3_CUTOFF, DISP_WIDTH = 50.0, 25.0, 0.05  # tblite's D4 setup
BORN_CUTOFF = 35.0 * AATOAU  # born.f90 lrcut_default
ATM_ALPHA = 16.0  # rational_damping_param%alp
D4_GA, D4_GC, D4_WF = 3.0, 2.0, 6.0


def _tensor(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=F64)


def _pairs(xyz: torch.Tensor, cutoff: float | None = None):
    """Atom pairs i > j with 1e-12 <= r^2 (<= cutoff^2): i, j, x_i - x_j, r."""
    i, j = np.tril_indices(xyz.shape[0], -1)
    vec = xyz[i] - xyz[j]
    r2 = (vec * vec).sum(1)
    r2n = r2.detach().numpy()
    keep = r2n >= 1e-12 if cutoff is None else (r2n >= 1e-12) & (r2n <= cutoff * cutoff)
    return i[keep], j[keep], vec[keep], torch.sqrt(r2[keep])


def _scatter(n: int, i, j, val: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(n, dtype=F64)
    return out.index_add(0, torch.as_tensor(i), val).index_add(0, torch.as_tensor(j), val)


def cn_dexp(z: np.ndarray, xyz: torch.Tensor) -> torch.Tensor:
    """mctc-lib's dexp coordination number (ka = 10, kb = 20, shift 2 Bohr, covalent_rad_d3)."""
    rcov = _tensor(params()["covalent_rad"]["covalent_rad_d3"][z - 1])
    i, j, _, r = _pairs(xyz, CN_DEXP_CUTOFF)
    rc = rcov[i] + rcov[j]
    count = torch.sigmoid(10.0 * (rc / r - 1.0)) * torch.sigmoid(20.0 * ((rc + 2.0) / r - 1.0))
    return _scatter(len(z), i, j, count)


def shell_info(basis: MolBasis) -> tuple[np.ndarray, np.ndarray]:
    """(angular momentum, atomic number) of every shell."""
    ls = np.array([basis.shell(s).l for s in range(len(basis.sh_atom))])
    return ls, basis.z[basis.sh_atom]


def selfenergies(basis: MolBasis, cn: torch.Tensor) -> torch.Tensor:
    """tblite's get_selfenergy: se = selfenergy(ish, Z) - kcn(l, Z) * CN."""
    g = params()["gfn2"]
    ls, zs = shell_info(basis)
    se0 = _tensor(g["p_selfenergy"][basis.sh_elem, zs - 1])
    kcn = _tensor(g["p_kcn"][ls, zs - 1])
    return se0 - kcn * cn[torch.as_tensor(basis.sh_atom)]


def e_rep(z: np.ndarray, xyz: torch.Tensor) -> torch.Tensor:
    g = params()["gfn2"]
    alpha, zeff = _tensor(g["rep_alpha"][z - 1]), _tensor(g["rep_zeff"][z - 1])
    i, j, _, r = _pairs(xyz, REP_CUTOFF)
    heavy = torch.as_tensor((z[i] > 2) | (z[j] > 2))
    kexp = torch.where(heavy, scalar("rep_kexp"), scalar("rep_kexp_light"))
    e = zeff[i] * zeff[j] * torch.exp(-torch.sqrt(alpha[i] * alpha[j]) * r**kexp)
    return (e / r ** scalar("rep_rexp")).sum()


def _shell_hardness(basis: MolBasis) -> torch.Tensor:
    g = params()["gfn2"]
    ls, zs = shell_info(basis)
    return _tensor(g["hubbard_parameter"][zs - 1] * g["shell_hubbard"][ls, zs - 1])


def e_es2(basis: MolBasis, xyz: torch.Tensor, qsh: torch.Tensor) -> torch.Tensor:
    """0.5 q^T A q, A = (r^g + eta^-g)^(-1/g), eta the arithmetic mean of shell hardnesses."""
    eta = _shell_hardness(basis)
    gam = 0.5 * (eta[:, None] + eta[None, :])
    at = torch.as_tensor(basis.sh_atom)
    i, j, _, r = _pairs(xyz)
    rmat = torch.zeros(len(basis.z), len(basis.z), dtype=F64)
    rmat = rmat.index_put((torch.as_tensor(i), torch.as_tensor(j)), r)
    rmat = rmat + rmat.T
    rsh = rmat[at][:, at]
    gexp = scalar("gexp")
    amat = 1.0 / (rsh**gexp + gam ** (-gexp)) ** (1.0 / gexp)
    amat = torch.where(at[:, None] == at[None, :], gam, amat)
    return 0.5 * qsh @ amat @ qsh


def e_es3(basis: MolBasis, qsh: torch.Tensor) -> torch.Tensor:
    g = params()["gfn2"]
    ls, zs = shell_info(basis)
    hd = _tensor(g["p_hubbard_derivs"][zs - 1] * g["shell_hubbard_derivs"][ls])
    return (qsh**3 * hd).sum() / 3.0


QP_SCALE = (1.0, 2.0, 1.0, 2.0, 2.0, 1.0)  # off-diagonal quadrupole components count twice


def multipole_radii(z: np.ndarray, cn: torch.Tensor) -> torch.Tensor:
    g = params()["gfn2"]
    rad, vcn = _tensor(g["p_rad"][z - 1]), _tensor(g["p_vcn"][z - 1])
    arg = cn - vcn - scalar("mp_shift")
    return rad + (scalar("mp_rmax") - rad) / (1.0 + torch.exp(-scalar("mp_kexp") * arg))


def e_aes(z, xyz, qat, dp, qp, cn) -> torch.Tensor:
    """tblite's damped multipole energy (get_energy of tblite_coulomb_multipole, 0d)."""
    g = params()["gfn2"]
    mrad = multipole_radii(z, cn)
    i, j, vec, r = _pairs(xyz)
    rr = 0.5 * (mrad[i] + mrad[j]) / r
    f3 = 1.0 / (1.0 + 6.0 * rr ** scalar("mp_dmp3"))
    f5 = 1.0 / (1.0 + 6.0 * rr ** scalar("mp_dmp5"))
    g3, g5 = f3 / r**3, f5 / r**5
    # charge-dipole: both orders of the pair; vec_ij = x_i - x_j, amat_sd(:, j, i) = vec g3
    e = ((dp[j] * vec).sum(1) * qat[i] - (dp[i] * vec).sum(1) * qat[j]) @ g3
    dd = (dp[i] * dp[j]).sum(1) * f5 / r**3 - 3.0 * (dp[i] * vec).sum(1) * (dp[j] * vec).sum(1) * g5
    e = e + dd.sum()
    x, y, zc = vec[:, 0], vec[:, 1], vec[:, 2]
    tc = torch.stack([x * x, 2 * x * y, y * y, 2 * x * zc, 2 * y * zc, zc * zc], 1)
    e = e + (((qp[j] * tc).sum(1) * qat[i] + (qp[i] * tc).sum(1) * qat[j]) * g5).sum()
    dk, qk = _tensor(g["p_dkernel"][z - 1]), _tensor(g["p_qkernel"][z - 1])
    e = e + (dk * (dp * dp).sum(1)).sum()
    return e + (qk * (qp * qp * _tensor(QP_SCALE)).sum(1)).sum()


# --- D4 dispersion ---------------------------------------------------------------------------

_K4, _K5, _K6 = 4.10451, 19.08857, 2 * 11.28174**2  # mctc-lib erf_dftd4 EN factor
_FREQ = np.array([0.000001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4,
                  1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0])  # fmt: skip
_TRAPZD = 0.5 * np.concatenate([[_FREQ[1] - _FREQ[0]], _FREQ[2:] - _FREQ[:-2],
                                [_FREQ[-1] - _FREQ[-2]]])  # fmt: skip


def _zeta(a, c, qref, qmod):
    """dftd4's charge scaling exp(a (1 - exp(c (1 - qref / qmod)))), exp(a) for qmod < 0."""
    safe = torch.where(qmod < 0, torch.ones_like(qmod), qmod) if torch.is_tensor(qmod) else qmod
    val = torch.exp(a * (1.0 - torch.exp(c * (1.0 - qref / safe))))
    return torch.where(qmod < 0, torch.full_like(val, math.exp(a)), val)


@dataclass(frozen=True)
class D4Element:
    nref: int
    refcn: np.ndarray  # (nref,) the reference CN of the weights (refcovcn)
    ngw: np.ndarray  # (nref,) Gaussian count of each reference (from refcn)
    refq: np.ndarray  # (nref,) GFN2 reference charges
    aiw: np.ndarray  # (nref, 23) reference polarizabilities alpha(i omega)


def _d4(name: str) -> np.ndarray:
    return params()["dftd4"][name]


@cache
def d4_element(zz: int) -> D4Element:
    """dftd4's set_refcn, set_refgw, set_refq_gfn2 and set_refalpha_gfn2 for one element."""
    nref = int(_d4("refn")[zz - 1])
    refcn = _d4("refcn")[:nref, zz - 1]
    cnc = np.zeros(20, dtype=int)
    cnc[0] = 1
    icn = np.minimum(np.floor(refcn + 0.5).astype(int), 19)  # Fortran nint, cnc(0:19)
    for k in icn:
        cnc[k] += 1
    ngw = np.array([cnc[k] * (cnc[k] + 1) // 2 for k in icn])
    aiw = np.zeros((nref, 23))
    for ir in range(nref):
        is_ = int(_d4("refsys")[ir, zz - 1])
        if is_ == 0:
            continue
        iz = _d4("effective_nuclear_charge")[is_ - 1]
        eta = _d4("chemical_hardness")[is_ - 1]
        sec = (
            _d4("sscale")[is_ - 1]
            * _d4("secaiw")[:, is_ - 1]
            * math.exp(
                D4_GA * (1.0 - math.exp(D4_GC * eta * (1.0 - iz / (_d4("refh")[ir, zz - 1] + iz))))
            )
        )
        diff = _d4("alphaiw")[:, ir, zz - 1] - _d4("hcount")[ir, zz - 1] * sec
        aiw[ir] = np.maximum(_d4("ascale")[ir, zz - 1] * diff, 0.0)
    return D4Element(nref, _d4("refcovcn")[:nref, zz - 1], ngw, _d4("refq")[:nref, zz - 1], aiw)


@cache
def d4_c6ref(za: int, zb: int) -> np.ndarray:
    """(nref_a, nref_b) reference C6 by the Casimir-Polder trapezoid of dftd4's trapzd."""
    a, b = d4_element(za), d4_element(zb)
    return 3.0 / math.pi * np.einsum("aw,bw,w->ab", a.aiw, b.aiw, _TRAPZD)


def cn_d4(z: np.ndarray, xyz: torch.Tensor) -> torch.Tensor:
    """mctc-lib's erf_dftd4 CN: erf count (kcn 7.5) times the electronegativity factor."""
    rcov = _tensor(_d4("covalent_rad_d3")[z - 1])
    en = _d4("pauling_en")[z - 1]
    i, j, _, r = _pairs(xyz, CN_D4_CUTOFF)
    rc = rcov[i] + rcov[j]
    den = _tensor(_K4 * np.exp(-((np.abs(en[i] - en[j]) + _K5) ** 2) / _K6))
    count = den * 0.5 * (1.0 + torch.erf(-7.5 * (r - rc) / rc))
    return _scatter(len(z), i, j, count)


def d4_weights(z: np.ndarray, cn: torch.Tensor, q: torch.Tensor) -> list[torch.Tensor]:
    """dftd4's weight_references: per atom the (nref,) Gaussian CN weights times zeta(q)."""
    out = []
    for a, zz in enumerate(z.tolist()):
        el = d4_element(zz)
        refcn = _tensor(el.refcn)
        expw = torch.stack([
            sum(torch.exp(-k * D4_WF * (cn[a] - refcn[r]) ** 2) for k in range(1, el.ngw[r] + 1))
            for r in range(el.nref)])  # fmt: skip
        norm = expw.sum()
        gwk = expw / norm
        if not bool(torch.isfinite(gwk).all()) or float(norm) <= math.sqrt(np.finfo(float).tiny):
            gwk = _tensor((np.abs(el.refcn - el.refcn.max()) < 1e-12).astype(float))
        zi = float(_d4("effective_nuclear_charge")[zz - 1])
        gi = float(_d4("chemical_hardness")[zz - 1]) * D4_GC
        out.append(gwk * _zeta(D4_GA, gi, _tensor(el.refq) + zi, q[a] + zi))
    return out


def d4_c6_matrix(z: np.ndarray, gw: list[torch.Tensor]) -> torch.Tensor:
    """(nat, nat) C6 from the reference weights, sum_ab gw_i[a] C6ref[a, b] gw_j[b]."""
    nat = len(z)
    w = torch.stack([torch.cat([g, torch.zeros(7 - len(g), dtype=F64)]) for g in gw])
    ref = np.zeros((nat, nat, 7, 7))
    for za in set(z.tolist()):
        for zb in set(z.tolist()):
            c = d4_c6ref(za, zb)
            ref[np.ix_(z == za, z == zb)] = np.pad(c, ((0, 7 - c.shape[0]), (0, 7 - c.shape[1])))
    return torch.einsum("ia,ijab,jb->ij", w, _tensor(ref), w)


def _r4r2(z: np.ndarray) -> torch.Tensor:
    return _tensor(np.sqrt(0.5 * _d4("r4_over_r2")[z - 1] * np.sqrt(z)))


def _switch(r: torch.Tensor, cutoff: float, width: float) -> torch.Tensor:
    """dftd4's smooth_cutoff: 1 below cutoff - width, the quintic smoothstep up to cutoff."""
    x = ((cutoff - r) / width).clamp(0.0, 1.0)
    return torch.where(r <= cutoff - width, torch.ones_like(r), x**3 * (10 + x * (-15 + 6 * x)))


def e_d4_two(z: np.ndarray, xyz: torch.Tensor, cn: torch.Tensor, q: torch.Tensor):
    """The rational-damped two-body D4 energy with the SCF charges q."""
    i, j, _, r = _pairs(xyz, DISP2_CUTOFF)
    if not len(i):
        return xyz.sum() * 0.0
    c6 = d4_c6_matrix(z, d4_weights(z, cn, q))[i, j]
    r4r2 = _r4r2(z)
    rrij = 3.0 * r4r2[i] * r4r2[j]
    r0 = scalar("a1") * torch.sqrt(rrij) + scalar("a2")
    t6, t8 = 1.0 / (r**6 + r0**6), 1.0 / (r**8 + r0**8)
    sw = _switch(r, DISP2_CUTOFF, DISP_WIDTH)
    return -(c6 * sw * (scalar("s6") * t6 + scalar("s8") * rrij * t8)).sum()


def e_d4_three(z: np.ndarray, xyz: torch.Tensor, cn: torch.Tensor) -> torch.Tensor:
    """The Axilrod-Teller-Muto three-body energy with zero-charge C6 (tblite's non-SC part)."""
    nat = len(z)
    if nat < 3:
        return xyz.sum() * 0.0
    gw = d4_weights(z, cn, torch.zeros(nat, dtype=F64))
    k, j, i = np.array(list(combinations(range(nat), 3))).T  # i > j > k
    d = {}
    for name, (a, b) in {"ij": (i, j), "ik": (i, k), "jk": (j, k)}.items():
        v = xyz[a] - xyz[b]
        d[name] = (v * v).sum(1)
    ok = np.all([(t.detach().numpy() <= DISP3_CUTOFF**2) & (t.detach().numpy() >= 1e-16)
                 for t in d.values()], axis=0)  # fmt: skip
    i, j, k = i[ok], j[ok], k[ok]
    r2ij, r2ik, r2jk = (d[n][ok] for n in ("ij", "ik", "jk"))
    c6 = d4_c6_matrix(z, gw)
    c6ij, c6ik, c6jk = c6[i, j], c6[i, k], c6[j, k]
    r4r2 = _r4r2(z)
    a1, a2 = scalar("a1"), scalar("a2")

    def r0_of(a, b):
        return a1 * torch.sqrt(3 * r4r2[a] * r4r2[b]) + a2

    r0 = r0_of(i, j) * r0_of(i, k) * r0_of(j, k)
    r1 = torch.sqrt(r2ij * r2ik * r2jk)
    fdmp = 1.0 / (1.0 + 6.0 * (r0 / r1) ** (ATM_ALPHA / 3.0))
    ang = (0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik) / r1**5
           + 1.0 / r1**3)  # fmt: skip
    sw = (_switch(torch.sqrt(r2ij), DISP3_CUTOFF, DISP_WIDTH)
          * _switch(torch.sqrt(r2ik), DISP3_CUTOFF, DISP_WIDTH)
          * _switch(torch.sqrt(r2jk), DISP3_CUTOFF, DISP_WIDTH))  # fmt: skip
    c9 = scalar("s9") * torch.sqrt(torch.abs(c6ij * c6ik * c6jk))
    return (c9 * ang * fdmp * sw).sum()


# --- ALPB solvation (water, GFN2 parameters) ---------------------------------------------------

ZETA_P16 = 1.028
ALPHA_ALPB = 0.571412
OBC = (1.0, 0.8, 4.85)


def _solv(name: str) -> np.ndarray:
    return params()["solvation"][name]


def _born_g(r, rho_other, rvdw_self):
    """The descreening integral of one neighbour (compute_psi): outside, partial overlap, 0."""
    ap, am = r + rho_other, r - rho_other
    ov = (r < rvdw_self + rho_other).detach()
    inside = (ap <= rvdw_self).detach()
    am_out = torch.where(ov, torch.ones_like(am), am)  # keeps the unused branch finite
    outside = rho_other / (ap * am_out) + 0.5 * torch.log(am_out / ap) / r
    rh1 = 1.0 / rvdw_self
    partial = rh1 - 1.0 / ap + 0.5 / r * (0.5 * am * (1.0 / ap - ap * rh1**2) - torch.log(ap * rh1))
    zero = torch.zeros_like(r)
    return torch.where(~ov, outside, torch.where(inside, zero, partial))


def born_radii(z: np.ndarray, xyz: torch.Tensor) -> torch.Tensor:
    """tblite's GBOBC Born radii with the ALPB water descreening and scale."""
    rvdw = _tensor(_solv("vdw_rad_d3")[z - 1])
    rho = rvdw * _tensor(_solv("sx")[z - 1])
    svdw = rvdw - float(_solv("soset")[0]) * 0.1 * AATOAU
    i, j, _, r = _pairs(xyz, BORN_CUTOFF)
    psi = torch.zeros(len(z), dtype=F64)
    psi = psi.index_add(0, torch.as_tensor(i), _born_g(r, rho[j], rvdw[i]))
    psi = psi.index_add(0, torch.as_tensor(j), _born_g(r, rho[i], rvdw[j]))
    br = psi * 0.5 * svdw
    arg = br * (OBC[0] + br * (OBC[2] * br - OBC[1]))
    return float(_solv("c1")[0]) / (1.0 / svdw - torch.tanh(arg) / rvdw)


def _alpb_consts() -> tuple[float, float]:
    eps = float(_solv("epsv")[0])
    alpbet = ALPHA_ALPB / eps
    return (1.0 / eps - 1.0) / (1.0 + alpbet), alpbet


def _adet(z: np.ndarray, xyz: torch.Tensor) -> torch.Tensor:
    """The ALPB shape descriptor A_det from the inertia of the vdW spheres."""
    rad = _tensor(_solv("vdw_rad_d3")[z - 1])
    r3 = rad**3
    vol = r3.sum()
    centre = (xyz * r3[:, None]).sum(0) / vol
    vec = xyz - centre
    r2 = (vec * vec).sum(1)
    eye = torch.eye(3, dtype=F64)
    inertia = (r3[:, None, None] * ((r2 + 0.4 * rad**2)[:, None, None] * eye
                                    - vec[:, :, None] * vec[:, None, :])).sum(0)  # fmt: skip
    return torch.sqrt(torch.linalg.det(inertia) ** (1.0 / 3.0) / (0.4 * vol))


def e_alpb(z: np.ndarray, xyz: torch.Tensor, qat: torch.Tensor) -> torch.Tensor:
    """0.5 q^T J q with the P16 Born kernel plus the ALPB term keps alpha / A_det."""
    keps, alpbet = _alpb_consts()
    brad = born_radii(z, xyz)
    i, j, _, r = _pairs(xyz)
    ab = torch.sqrt(brad[i] * brad[j])
    fgb = r + ab * (ab / (ab + ZETA_P16 / 16.0 * r)) ** 16
    e = keps * (qat[i] * qat[j] / fgb).sum() + 0.5 * keps * (qat**2 / brad).sum()
    return e + 0.5 * keps * alpbet / _adet(z, xyz) * qat.sum() ** 2


# --- CDS: surface tension and H-bond term --------------------------------------------------------

SASA_SMOOTH = 0.3 * AATOAU
SASA_OFFSET = 2.0 * AATOAU
TOLSESP = 1e-6


@cache
def _lebedev230() -> tuple[np.ndarray, np.ndarray]:
    x, w = lebedev_rule(25)  # the 230-point grid; weights sum to 4 pi
    return x.T.copy(), w.copy()


def _sasa_consts(z: np.ndarray):
    probe = float(_solv("rprobe")[0]) * AATOAU
    vdwsa = _solv("vdw_rad_d3")[z - 1] + probe
    w = SASA_SMOOTH
    ah0, ah1, ah3 = 0.5, 3.0 / (4.0 * w), -1.0 / (4.0 * w**3)

    def part(r):
        return (0.25 / w + 3.0 * ah3 * (0.2 * r * r - 0.5 * r * vdwsa + vdwsa**2 / 3.0)) * r**3

    wrp = part(vdwsa + w) - part(vdwsa - w)
    srcut = 2.0 * (w + vdwsa.max()) + SASA_OFFSET
    return vdwsa, wrp, (ah0, ah1, ah3), srcut


def surface_areas(z: np.ndarray, xyz: torch.Tensor) -> torch.Tensor:
    """tblite's smoothed SASA per atom (compute_surface) on the 230-point Lebedev grid."""
    vdwsa, wrp, (ah0, ah1, ah3), srcut = _sasa_consts(z)
    grid, weight = (_tensor(a) for a in _lebedev230())
    w = SASA_SMOOTH
    lo2, hi2 = (vdwsa - w) ** 2, (vdwsa + w) ** 2
    xn = xyz.detach().numpy()
    out = []
    for a in range(len(z)):
        d2 = ((xn - xn[a]) ** 2).sum(1)
        nb = np.flatnonzero((d2 <= srcut**2) & (d2 >= np.finfo(float).eps))
        pts = xyz[a] + float(vdwsa[a]) * grid  # (230, 3)
        t = pts[:, None, :] - xyz[nb][None, :, :]
        t2 = (t * t).sum(2)
        u = torch.sqrt(t2.clamp(min=1e-300)) - _tensor(vdwsa[nb])
        f = ah0 + (ah1 + ah3 * u * u) * u
        f = torch.where(t2 >= _tensor(hi2[nb]), torch.ones_like(f), f)
        f = torch.where(t2 <= _tensor(lo2[nb]), torch.zeros_like(f), f)
        sasap = f.prod(1)
        sasap = torch.where(sasap > TOLSESP, sasap, torch.zeros_like(sasap))
        out.append(float(wrp[a]) * (weight * sasap).sum())
    return torch.stack(out)


def e_cds(z: np.ndarray, xyz: torch.Tensor, qat: torch.Tensor) -> torch.Tensor:
    """Surface tension (geometry only) plus the H-bond term hbond * surface * q^2."""
    probe = float(_solv("rprobe")[0]) * AATOAU
    rad = _solv("vdw_rad_d3")[z - 1]
    tension = _tensor(_solv("gamscale")[z - 1] * 1e-5)
    hbond = _tensor(
        -KCALTOAU * _solv("sqrtGhbond")[z - 1] ** 2 / (4 * math.pi * (rad + probe) ** 2)
    )
    area = surface_areas(z, xyz)
    return (area * (tension + hbond * qat**2)).sum()


def solvent_shift() -> float:
    """The constant ALPB shift (solution state gsolv: no state correction)."""
    return float(_solv("gshift")[0]) * KCALTOAU


# --- electronic entropy, SCF variables and the total energy -------------------------------------


def fermi_filling(emo: np.ndarray, nel: float, kt: float = KT_300K) -> np.ndarray:
    """tblite's get_fermi_filling for one spin channel (Newton steps on the Fermi level)."""
    occ = np.zeros(len(emo))
    homo = math.floor(nel)
    occ[: min(homo, len(occ))] = 1.0
    if homo < len(occ):
        occ[homo] = nel % 1.0
    homo = homo + 1 if nel % 1.0 > 0.5 else homo
    if homo <= 0:
        return occ
    ef = 0.5 * (emo[max(homo, 1) - 1] + emo[min(homo + 1, len(emo)) - 1])
    for _ in range(200):
        x = (emo - ef) / kt
        live = x < 50
        ex = np.exp(np.where(live, x, 0.0))
        occ = np.where(live, 1.0 / (ex + 1.0), 0.0)
        dfermi = np.where(live, ex / (kt * (ex + 1.0) ** 2), 0.0).sum()
        total = occ.sum()
        ef += (homo - total) / dfermi if dfermi > math.sqrt(np.finfo(float).tiny) else 0.0
        if abs(homo - total) <= math.sqrt(np.finfo(float).eps):
            break
    return occ


def electronic_entropy(focc: np.ndarray, kt: float = KT_300K) -> float:
    """tblite's -TS term, kt * sum log(f^f (1-f)^(1-f)), of one spin channel."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(kt * np.log(focc**focc * (1.0 - focc) ** (1.0 - focc)).sum())


@dataclass(frozen=True)
class SCFVariables:
    qsh: np.ndarray  # (nsh,) Mulliken shell charges (reference occupation minus population)
    qat: np.ndarray  # (nat,)
    dp: np.ndarray  # (nat, 3) atomic dipoles
    qp: np.ndarray  # (nat, 6) atomic traceless quadrupoles


def reference_occupations(basis: MolBasis) -> np.ndarray:
    ls, zs = shell_info(basis)
    return params()["gfn2"]["reference_occ"][ls, zs - 1]


def scf_variables(basis: MolBasis, ints, density: np.ndarray) -> SCFVariables:
    """tblite's Mulliken shell charges and atomic multipoles of a density matrix."""
    nat, ao_atom = len(basis.z), basis.ao_atom
    pop_ao = (density * ints.S).sum(0)
    ao_shell = np.repeat(np.arange(len(basis.sh_atom)), np.diff(np.append(basis.sh_ao, basis.nao)))
    qsh = reference_occupations(basis) - np.bincount(ao_shell, pop_ao, len(basis.sh_atom))
    qat = np.bincount(basis.sh_atom, qsh, nat)
    dp = np.stack([np.bincount(ao_atom, -(density * d).sum(0), nat) for d in ints.D], 1)
    qp = np.stack([np.bincount(ao_atom, -(density * c).sum(0), nat) for c in ints.Q], 1)
    return SCFVariables(qsh, qat, dp, qp)


def energy_terms(basis: MolBasis, xyz: torch.Tensor, v: dict, solvent: bool) -> dict:
    """Every term except the band energy tr(H0 P) and the entropy, as torch scalars. `v` holds
    torch tensors qsh, qat, dp, qp."""
    z = basis.z
    cn, cn4 = cn_dexp(z, xyz), cn_d4(z, xyz)
    out = {
        "repulsion": e_rep(z, xyz),
        "es2": e_es2(basis, xyz, v["qsh"]),
        "es3": e_es3(basis, v["qsh"]),
        "aes": e_aes(z, xyz, v["qat"], v["dp"], v["qp"], cn),
        "d4_two": e_d4_two(z, xyz, cn4, v["qat"]),
        "d4_three": e_d4_three(z, xyz, cn4),
    }
    if solvent:
        out["alpb"] = e_alpb(z, xyz, v["qat"])
        out["cds"] = e_cds(z, xyz, v["qat"])
        out["shift"] = torch.tensor(solvent_shift(), dtype=F64)
    return out


def _leaves(sv: SCFVariables) -> dict:
    return {k: _tensor(getattr(sv, k)).requires_grad_() for k in ("qsh", "qat", "dp", "qp")}


def total_energy(basis, xyz, density, focc_spins, solvent: bool) -> tuple[float, dict]:
    """tblite's total energy of a density: band energy + entropy + every other term."""
    from ksmech.qm.gpu import reference as ref

    xt = _tensor(xyz)
    se = selfenergies(basis, cn_dexp(basis.z, xt)).numpy()
    ints = ref.integrals(basis, xyz, se)
    terms = {k: float(e) for k, e in energy_terms(basis, xt, _leaves(scf_variables(
        basis, ints, density)), solvent).items()}  # fmt: skip
    terms["band"] = float((ints.H0 * density).sum())
    terms["entropy"] = sum(electronic_entropy(f) for f in focc_spins)
    return sum(terms.values()), terms


def total_gradient(basis, xyz, density, wdensity, solvent: bool) -> np.ndarray:
    """dE/dR (nat, 3) at a converged density: every term at fixed SCF variables, plus the H0,
    overlap and multipole-integral derivatives contracted with the potentials (stage 1's
    `h0_gradient`), plus dE/dCN of the self-energies through the CN."""
    from ksmech.qm.gpu import reference as ref

    xt = _tensor(xyz).requires_grad_()
    se_t = selfenergies(basis, cn_dexp(basis.z, xt.detach()))
    ints = ref.integrals(basis, xyz, se_t.numpy())
    leaves = _leaves(scf_variables(basis, ints, density))
    e = sum(energy_terms(basis, xt, leaves, solvent).values())
    g = torch.autograd.grad(e, [xt, *leaves.values()], allow_unused=True)
    wrt = [xt, *leaves.values()]
    g_xyz, gsh, gat, gdp, gqp = (
        torch.zeros_like(t) if d is None else d for d, t in zip(g, wrt, strict=True)
    )
    vsh = (gsh + gat[torch.as_tensor(basis.sh_atom)]).numpy()
    nao_sh = np.diff(np.append(basis.sh_ao, basis.nao))
    ls, zs = shell_info(basis)
    gi = ref.GradInputs(
        se=se_t.numpy(), dsedcn=-params()["gfn2"]["p_kcn"][ls, zs - 1], P=density, X=wdensity,
        vao=np.repeat(vsh, nao_sh), vdp=gdp.numpy().T.copy(), vqp=gqp.numpy().T.copy(),
    )  # fmt: skip
    grad_h0, dedcn = ref.h0_gradient(basis, xyz, gi)
    xc = _tensor(xyz).requires_grad_()
    (g_cn,) = torch.autograd.grad((cn_dexp(basis.z, xc) * _tensor(dedcn)).sum(), [xc])
    return g_xyz.numpy() + grad_h0 + g_cn.numpy()
