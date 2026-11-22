"""A plain numpy reference of the GFN2 one-electron integrals and their gradient contraction.

It follows tblite 0.7.0 (integral/multipole.f90, xtb/h0.f90) term by term and is slow: it is
the oracle the CUDA kernels are checked against, not an engine.

Conventions (tblite's):
- `S[mu, nu]` overlap; `H0[mu, nu]` the core Hamiltonian.
- `D[k, mu, nu] = <mu| (r - R_atom(nu))_k |nu>`: the dipole integral centred on the atom of
  the column index. `Q[c, mu, nu]` the traceless quadrupole centred the same way, components
  (xx, xy, yy, xz, yz, zz), each 3/2 r_a r_b - r^2/2 delta_ab.
- The gradient contraction is `get_hamiltonian_gradient` for a closed shell: the density P,
  the energy-weighted density X, the AO potential vao, the atomic dipole and quadrupole
  potentials vdp (3, nat) and vqp (6, nat), and d(selfenergy)/dCN; it returns the gradient
  (nat, 3) and dE/dCN (nat).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ksmech.qm.gpu.basis import (
    CGTO,
    N_CART,
    MolBasis,
    integral_cutoff,
    pair_cutoff,
    params,
    scalar,
)

S3 = math.sqrt(3.0)
# Cartesian exponents in tblite's order (lx): p = y, z, x; d = xx, yy, zz, xy, xz, yz
CART = (
    ((0, 0, 0),),
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    ((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)),
)
# Cartesian -> spherical (rows spherical, columns cartesian), tblite's dtrafo
TRAFO = (
    np.eye(1),
    np.eye(3),
    np.array(
        [
            [0, 0, 0, S3, 0, 0],  # xy
            [0, 0, 0, 0, 0, S3],  # yz
            [-0.5, -0.5, 1, 0, 0, 0],  # z2
            [0, 0, 0, 0, S3, 0],  # xz
            [S3 / 2, -S3 / 2, 0, 0, 0, 0],  # x2-y2
        ]
    ),
)
# Operator exponents: overlap, dipole x/y/z, quadrupole xx, xy, yy, xz, yz, zz
OPS = (
    (0, 0, 0),
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (2, 0, 0), (1, 1, 0), (0, 2, 0), (1, 0, 1), (0, 1, 1), (0, 0, 2),
)  # fmt: skip


def _poly_pow(p: np.ndarray, a: int) -> np.ndarray:
    """Coefficients in t of (t + p)^a, shape (n, a + 1)."""
    out = np.zeros((len(p), a + 1))
    for m in range(a + 1):
        out[:, m] = math.comb(a, m) * p ** (a - m)
    return out


def _poly_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros((x.shape[0], x.shape[1] + y.shape[1] - 1))
    for i in range(x.shape[1]):
        out[:, i : i + y.shape[1]] += x[:, i : i + 1] * y
    return out


def _moments(eab: np.ndarray, mmax: int) -> np.ndarray:
    """int t^m exp(-eab t^2) dt / sqrt(pi / eab), m = 0..mmax."""
    out = np.zeros((len(eab), mmax + 1))
    for m in range(0, mmax + 1, 2):
        out[:, m] = (0.5 / eab) ** (m // 2) * [1, 1, 3, 15, 105, 945, 10395][m // 2]
    return out


def _m1d(a, b, k, pa, pb, pc, s1d):
    """int (x-A)^a (x-B)^b (x-C)^k over the product Gaussian, per primitive pair."""
    if a < 0:
        return np.zeros(len(pa))
    poly = _poly_mul(_poly_mul(_poly_pow(pa, a), _poly_pow(pb, b)), _poly_pow(pc, k))
    return (poly * s1d[:, : poly.shape[1]]).sum(axis=1)


def _dim_tables(a, b, alpha_i, pa, pb, s1d):
    """Per dimension: values and d/dA of the 1D moments k = 0..2, about A ('i') and B ('j')."""
    val, dval = {}, {}
    for centre, pc in (("i", pa), ("j", pb)):
        for k in range(3):
            v = _m1d(a, b, k, pa, pb, pc, s1d)
            g = -a * _m1d(a - 1, b, k, pa, pb, pc, s1d) + 2 * alpha_i * _m1d(
                a + 1, b, k, pa, pb, pc, s1d
            )
            if centre == "i" and k > 0:  # the operator centre moves with A
                g = g - k * _m1d(a, b, k - 1, pa, pb, pc, s1d)
            val[centre, k], dval[centre, k] = v, g
    return val, dval


def _prim_pairs(ci: CGTO, cj: CGTO, vec: np.ndarray):
    """Primitive pair data with tblite's screening (screened pairs get cc = 0)."""
    ai, aj = np.meshgrid(ci.alpha, cj.alpha, indexing="ij")
    ki, kj = np.meshgrid(ci.coeff, cj.coeff, indexing="ij")
    ai, aj, ki, kj = ai.ravel(), aj.ravel(), ki.ravel(), kj.ravel()
    eab = ai + aj
    est = ai * aj * float(vec @ vec) / eab
    cc = ki * kj * np.exp(-est) * math.pi**1.5 / eab**1.5
    cc[est > integral_cutoff()] = 0.0
    pa = -np.outer(aj / eab, vec)  # P - A, with vec = A - B
    pb = np.outer(ai / eab, vec)  # P - B
    return ai, eab, cc, pa, pb


def _cart_block(ci: CGTO, cj: CGTO, vec: np.ndarray, centre: str, grad: bool):
    """Cartesian (10, ncart_i, ncart_j) operator block about A or B, and d/dA (3, 10, ...)."""
    ai, eab, cc, pa, pb = _prim_pairs(ci, cj, vec)
    s1d = _moments(eab, ci.l + cj.l + 5)
    out = np.zeros((10, N_CART[ci.l], N_CART[cj.l]))
    dout = np.zeros((3, 10, N_CART[ci.l], N_CART[cj.l]))
    for mi, ei in enumerate(CART[ci.l]):
        for mj, ej in enumerate(CART[cj.l]):
            tabs = [_dim_tables(ei[d], ej[d], ai, pa[:, d], pb[:, d], s1d) for d in range(3)]
            for c, op in enumerate(OPS):
                vals = [tabs[d][0][centre, op[d]] for d in range(3)]
                out[c, mi, mj] = (cc * vals[0] * vals[1] * vals[2]).sum()
                if not grad:
                    continue
                for e in range(3):
                    prod = tabs[e][1][centre, op[e]]
                    for d in range(3):
                        prod = prod if d == e else prod * vals[d]
                    dout[e, c, mi, mj] = (cc * prod).sum()
    return out, dout


def _traceless(q: np.ndarray) -> np.ndarray:
    """(6, ...) cartesian second moments -> tblite's traceless quadrupole."""
    tr = 0.5 * (q[0] + q[2] + q[5])
    out = 1.5 * q
    out[0] -= tr
    out[2] -= tr
    out[5] -= tr
    return out


def _to_sph(block: np.ndarray, li: int, lj: int) -> np.ndarray:
    return np.einsum("ma,...ab,nb->...mn", TRAFO[li], block, TRAFO[lj])


def shell_pair(ci: CGTO, cj: CGTO, vec: np.ndarray, centre: str = "i", grad: bool = False):
    """Spherical S, D, Q of shell i on A and shell j on B (vec = A - B), the multipoles about
    A ('i') or B ('j'); with grad also their derivatives with respect to A, (3, comp, ni, nj)."""
    cart, dcart = _cart_block(ci, cj, vec, centre, grad)
    sph = _to_sph(cart, ci.l, cj.l)
    s, d, q = sph[0], sph[1:4], _traceless(sph[4:10])
    if not grad:
        return s, d, q
    dsph = _to_sph(dcart, ci.l, cj.l)
    ds, dd, dq = dsph[:, 0], dsph[:, 1:4], np.stack([_traceless(x) for x in dsph[:, 4:10]])
    return s, d, q, ds, dd, dq


@dataclass(frozen=True)
class H0Params:
    """The geometry-independent parts of the GFN2 H0 for one molecule."""

    hscale: np.ndarray  # (nsh, nsh): kpair * kshell * EN factor * zeta factor, by shell pair
    shpoly: np.ndarray  # (nsh,)
    rad: np.ndarray  # (nat,) atomic radii, Bohr


def _kshell(li: int, lj: int) -> float:
    kdiag = params()["gfn2"]["kdiag"]
    if (lj == 2 and li in (0, 1)) or (li == 2 and lj in (0, 1)):
        return 2.0
    return 0.5 * (kdiag[li] + kdiag[lj])


def h0_params(basis: MolBasis) -> H0Params:
    g, en = params()["gfn2"], params()["pauling_en"]["pauling_en"]
    rad_tab = params()["atomic_rad"]["atomic_rad"]
    nsh = len(basis.sh_atom)
    zs = basis.z[basis.sh_atom]
    cg = [basis.shell(s) for s in range(nsh)]
    zeta = np.array([c.zeta for c in cg])
    ls = np.array([c.l for c in cg])
    kshell = np.array([[_kshell(li, lj) for lj in range(3)] for li in range(3)])
    zij = (2 * np.sqrt(zeta[:, None] * zeta[None, :]) / (zeta[:, None] + zeta[None, :])) ** scalar(
        "wexp"
    )
    ens = en[zs - 1]
    enp = 1.0 + scalar("enscale") * (ens[:, None] - ens[None, :]) ** 2
    hscale = zij * kshell[ls[:, None], ls[None, :]] * enp
    shpoly = np.array([g["p_shpoly"][c.l, z - 1] for c, z in zip(cg, zs, strict=True)])
    return H0Params(hscale, shpoly, rad_tab[basis.z - 1])


def _neighbour_pairs(basis: MolBasis, xyz: np.ndarray):
    """Unordered atom pairs within tblite's cutoff, (A, B) with A > B, as tblite lists them."""
    cut2 = pair_cutoff(basis.z) ** 2
    nat = len(basis.z)
    for a in range(nat):
        for b in range(a):
            vec = xyz[a] - xyz[b]
            if vec @ vec <= cut2:
                yield a, b, vec


def _atom_shells(basis: MolBasis, atom: int) -> np.ndarray:
    return np.flatnonzero(basis.sh_atom == atom)


def _sh_slice(basis: MolBasis, s: int) -> slice:
    n = 2 * basis.shell(s).l + 1
    return slice(int(basis.sh_ao[s]), int(basis.sh_ao[s]) + n)


def _hij_offsite(hp: H0Params, se: np.ndarray, si: int, sj: int, a: int, b: int, r2: float):
    """H0 scale of an off-site shell pair and the pieces of its distance derivative."""
    rr = math.sqrt(math.sqrt(r2) / (hp.rad[a] + hp.rad[b]))
    pi, pj = 1.0 + hp.shpoly[si] * rr, 1.0 + hp.shpoly[sj] * rr
    hs = hp.hscale[sj, si] * pi * pj
    hij = 0.5 * (se[si] + se[sj]) * hs
    dlog = (pi * hp.shpoly[sj] + pj * hp.shpoly[si]) * 0.5 * rr / r2 / (pi * pj)
    return hij, hs, dlog


@dataclass
class Integrals:
    S: np.ndarray
    D: np.ndarray
    Q: np.ndarray
    H0: np.ndarray


def integrals(basis: MolBasis, xyz: np.ndarray, se: np.ndarray) -> Integrals:
    """S, D, Q and H0 of one molecule (xyz in Bohr, se the shell self-energies in Hartree)."""
    n = basis.nao
    out = Integrals(np.zeros((n, n)), np.zeros((3, n, n)), np.zeros((6, n, n)), np.zeros((n, n)))
    hp = h0_params(basis)
    for a, b, vec in _neighbour_pairs(basis, xyz):
        for si in _atom_shells(basis, a):
            for sj in _atom_shells(basis, b):
                ci, cj = basis.shell(si), basis.shell(sj)
                s, d_i, q_i = shell_pair(ci, cj, vec, "i")
                _, d_j, q_j = shell_pair(ci, cj, vec, "j")
                hij = _hij_offsite(hp, se, si, sj, a, b, float(vec @ vec))[0]
                ii, jj = _sh_slice(basis, si), _sh_slice(basis, sj)
                out.S[ii, jj], out.S[jj, ii] = s, s.T
                out.H0[ii, jj], out.H0[jj, ii] = s * hij, (s * hij).T
                out.D[:, jj, ii], out.D[:, ii, jj] = d_i.transpose(0, 2, 1), d_j
                out.Q[:, jj, ii], out.Q[:, ii, jj] = q_i.transpose(0, 2, 1), q_j
    for a in range(len(basis.z)):
        for si in _atom_shells(basis, a):
            for sj in _atom_shells(basis, a):
                s, d, q = shell_pair(basis.shell(si), basis.shell(sj), np.zeros(3), "i")
                ii, jj = _sh_slice(basis, si), _sh_slice(basis, sj)
                out.S[ii, jj], out.D[:, ii, jj], out.Q[:, ii, jj] = s, d, q
                out.H0[ii, jj] = s * 0.5 * (se[si] + se[sj])
    return out


@dataclass
class GradInputs:
    """What `get_hamiltonian_gradient` contracts the integral derivatives with."""

    se: np.ndarray  # (nsh,) self-energies
    dsedcn: np.ndarray  # (nsh,)
    P: np.ndarray  # (nao, nao) density
    X: np.ndarray  # (nao, nao) energy-weighted density
    vao: np.ndarray  # (nao,)
    vdp: np.ndarray  # (3, nat)
    vqp: np.ndarray  # (6, nat)


def _pair_terms(basis, hp, g: GradInputs, si, sj, a, b, vec):
    """dG (d/dR_A) and the dE/dCN parts of one off-site shell pair, tblite's h0 gradient."""
    ci, cj = basis.shell(si), basis.shell(sj)
    s, _, _, ds, dd_i, dq_i = shell_pair(ci, cj, vec, "i", grad=True)
    *_, dd_j, dq_j = shell_pair(ci, cj, vec, "j", grad=True)
    hij, hs, dlog = _hij_offsite(hp, g.se, si, sj, a, b, float(vec @ vec))
    ii, jj = _sh_slice(basis, si), _sh_slice(basis, sj)
    p, x = g.P[ii, jj], g.X[ii, jj]
    w = -p * (g.vao[ii, None] + g.vao[None, jj]) + 2 * p * hij - 2 * x
    dg = np.einsum("ij,eij->e", w, ds) + 2 * hij * float((p * s).sum()) * dlog * vec
    dg -= np.einsum("ij,ecij,c->e", p, dd_i, g.vdp[:, a])
    dg -= np.einsum("ij,ecij,c->e", p, dd_j, g.vdp[:, b])
    dg -= np.einsum("ij,ecij,c->e", p, dq_i, g.vqp[:, a])
    dg -= np.einsum("ij,ecij,c->e", p, dq_j, g.vqp[:, b])
    ps = float((p * s).sum())
    return dg, g.dsedcn[si] * hs * ps, g.dsedcn[sj] * hs * ps


def h0_gradient(basis: MolBasis, xyz: np.ndarray, g: GradInputs):
    """The gradient (nat, 3) and dE/dCN (nat,) of `get_hamiltonian_gradient`."""
    nat = len(basis.z)
    grad, dedcn = np.zeros((nat, 3)), np.zeros(nat)
    hp = h0_params(basis)
    for a, b, vec in _neighbour_pairs(basis, xyz):
        for si in _atom_shells(basis, a):
            for sj in _atom_shells(basis, b):
                dg, dci, dcj = _pair_terms(basis, hp, g, si, sj, a, b, vec)
                grad[a] += dg
                grad[b] -= dg
                dedcn[a] += dci
                dedcn[b] += dcj
    for s in range(len(basis.sh_atom)):
        ii = _sh_slice(basis, s)
        dedcn[basis.sh_atom[s]] += g.dsedcn[s] * float(np.trace(g.P[ii, ii]))
    return grad, dedcn


def h0_functional(basis: MolBasis, xyz: np.ndarray, g: GradInputs) -> float:
    """The scalar whose R-gradient (self-energies fixed) `h0_gradient` returns."""
    ints = integrals(basis, xyz, g.se)
    ao_atom = basis.ao_atom
    offsite = ao_atom[:, None] != ao_atom[None, :]
    p, x = g.P * offsite, g.X * offsite
    e = float((-p * (g.vao[:, None] + g.vao[None, :]) * ints.S).sum()) * 0.5
    e += float((p * ints.H0).sum()) - float((x * ints.S).sum())
    # D[:, mu, nu] is centred on atom(nu): sum P_mu,nu D_mu,nu . v_atom(nu) over off-site pairs
    e -= float(np.einsum("mn,cmn,cn->", p, ints.D, g.vdp[:, ao_atom]))
    e -= float(np.einsum("mn,cmn,cn->", p, ints.Q, g.vqp[:, ao_atom]))
    return e
