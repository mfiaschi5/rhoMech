"""Stage 3: the batched GFN2 SCF on the GPU, as tblite 0.7.0 runs it.

A batch of molecules is held twice: packed (stage 1's per-molecule blocks, the CUDA kernels'
layout) and padded to the batch's largest AO count (B, n, n) for the dense linear algebra. Each
SCF cycle, as tblite's `next_scf`:

1. potentials of the input SCF variables (stage 2's `evaluate`);
2. the Fock matrix H0 - S (v_i + v_j) / 2 - (W + W^T) / 2, W the dipole and quadrupole
   potentials contracted with the multipole integrals (tblite's add_vao_to_h1, add_vmp_to_h1);
3. the generalized eigenproblem through the Cholesky factor of S;
4. Fermi occupations at 300 K for each spin channel from the same orbitals, the density, the
   electronic entropy;
5. the output Mulliken shell charges and atomic multipoles, their difference from the input
   (the mixer residual) and the energy of the output;
6. tblite's modified Broyden mixing of (qsh, dp, qp) for the next input.

Converged: |dE| < 1e-6 accuracy and the residual's RMS < 2e-5 accuracy, tblite's criteria. The
guess is tblite's SAD: charge / nat on every atom, split over its shells by reference occupation.
Padded AOs get the diagonal `PAD_LEVEL` in F (identity in S): empty, far above the Fermi level.

Precision, per molecule: the cycles run in fp32 (transform, eigensolve, density) until the
residual's RMS falls below `SCFBatch.SWITCH`, then with fp64-accurate orbitals, and convergence
is only declared after two such cycles, so the energy test compares fp64 energies. A cycle makes
one fp32 eigensolve for all its active molecules; for the precise ones the occupied orbitals are
then refined in fp64 (`refine_occupied`, Ogita-Aishima restricted to pairs of different
occupation), an fp64 eigensolve only where the refinement is ill-conditioned (fixed cost per
call: 4 ms in fp64, docs/decisions.md 2026-11-23). The Fock build, the Mulliken sums and the
Fermi filling are fused kernels (native/ksxtb/src/scf.cu).
"""

from __future__ import annotations

import ctypes as C
import math
from dataclasses import dataclass
from functools import cache

import numpy as np
import torch

from ksmech.qm.gpu import classical as K
from ksmech.qm.gpu import native
from ksmech.qm.gpu import terms as T
from ksmech.qm.gpu.basis import mol_basis

PAD_LEVEL = 5.0  # Hartree; a padded AO's orbital energy
F64 = torch.float64


@dataclass
class Layout:
    """Padded views of a packed batch."""

    nmax: int
    nao: torch.Tensor  # (B,)
    loc: torch.Tensor  # (B, n, n) mu * nao + nu within the molecule's block, -1 for padding
    off: torch.Tensor  # (B,) stage 1's mol_mat offset
    ao_atom: torch.Tensor  # (B, n) global atom of each AO, nat for padding
    ao_shell: torch.Tensor  # (B, n) global shell, nsh for padding
    valid: torch.Tensor  # (B, n) bool
    sh_atom: torch.Tensor  # (nsh,) global atom of each shell

    @property
    def valid_u8(self) -> torch.Tensor:
        return self.valid.to(torch.uint8).contiguous()


def layout(hb: native.HostBatch, mols, device="cuda") -> Layout:
    nao = hb.arrays["mol_nao"].astype(np.int64)
    n, b = int(nao.max()), len(nao)
    loc = np.full((b, n, n), -1, dtype=np.int64)
    ao_atom = np.full((b, n), hb.nat, dtype=np.int64)
    ao_shell = np.full((b, n), hb.nsh, dtype=np.int64)
    at0 = sh0 = 0
    for m, (z, _) in enumerate(mols):
        basis = mol_basis(np.asarray(z, dtype=int))
        k = basis.nao
        loc[m, :k, :k] = np.arange(k * k).reshape(k, k)
        ao_atom[m, :k] = basis.ao_atom + at0
        sizes = np.diff(np.append(basis.sh_ao, k))
        ao_shell[m, :k] = np.repeat(np.arange(len(sizes)), sizes) + sh0
        at0, sh0 = at0 + len(z), sh0 + len(sizes)
    t = {k: torch.as_tensor(v, device=device) for k, v in
         {"nao": nao, "loc": loc, "off": hb.arrays["mol_mat"].astype(np.int64),
          "ao_atom": ao_atom, "ao_shell": ao_shell}.items()}  # fmt: skip
    sh_atom = torch.as_tensor(hb.arrays["sh_atom"].astype(np.int64), device=device)
    valid = t["loc"][:, 0] >= 0
    return Layout(n, t["nao"], t["loc"], t["off"], t["ao_atom"], t["ao_shell"], valid, sh_atom)


def pad(flat: torch.Tensor, lay: Layout, ncomp: int = 1) -> torch.Tensor:
    """(B, ncomp, n, n) padded copy of a packed ncomp-component matrix, zeros in the padding."""
    ext = torch.cat([flat, flat.new_zeros(1)])
    nn = (lay.nao * lay.nao)[:, None, None, None]
    comp = torch.arange(ncomp, device=flat.device)[None, :, None, None]
    idx = ncomp * lay.off[:, None, None, None] + comp * nn + lay.loc[:, None]
    idx = torch.where(lay.loc[:, None] >= 0, idx, len(flat))
    return ext[idx]


def unpad_into(flat: torch.Tensor, padded: torch.Tensor, lay: Layout) -> None:
    """Write a (B, n, n) padded matrix back into a packed single-component buffer."""
    sel = lay.loc >= 0
    flat[(lay.off[:, None, None] + lay.loc)[sel]] = padded[sel]


@dataclass
class Geometry:
    """The padded matrices of one geometry."""

    S: torch.Tensor  # (B, n, n), identity on the padding
    H0: torch.Tensor  # (B, n, n)
    D: torch.Tensor  # (B, 3, n, n)
    Q: torch.Tensor  # (B, 6, n, n)
    Linv: torch.Tensor  # (B, n, n) inverse Cholesky factor of S
    Linv32: torch.Tensor  # the same in fp32, for the fp32 cycles
    bad: torch.Tensor  # (B,) non-finite integrals or S not positive definite


def padded_geometry(db: native.DeviceBatch, lay: Layout) -> Geometry:
    """The padded matrices; a molecule with non-finite integrals or an S that is not positive
    definite (coincident atoms) is flagged `bad` and given harmless stand-ins (S = 1, H0 = 0),
    so it cannot fail the batch."""
    s = pad(db.S, lay)[:, 0] + torch.diag_embed((~lay.valid).to(F64))
    h0, d, q = pad(db.H0, lay)[:, 0], pad(db.D, lay, 3), pad(db.Q, lay, 6)
    bad = ~(torch.isfinite(s).all((1, 2)) & torch.isfinite(h0).all((1, 2))
            & torch.isfinite(d).all((1, 2, 3)) & torch.isfinite(q).all((1, 2, 3)))  # fmt: skip
    eye_n = torch.eye(lay.nmax, dtype=F64, device=s.device).expand_as(s)
    s = torch.where(bad[:, None, None], eye_n, s)
    chol, info = torch.linalg.cholesky_ex(s)
    bad = bad | (info != 0)
    m3, m4 = bad[:, None, None], bad[:, None, None, None]
    s, chol = torch.where(m3, eye_n, s), torch.where(m3, eye_n, chol)
    h0, d, q = torch.where(m3, 0.0, h0), torch.where(m4, 0.0, d), torch.where(m4, 0.0, q)
    linv = torch.linalg.solve_triangular(chol, eye_n, upper=False)
    return Geometry(s, h0, d, q, linv, linv.float(), bad)


def eigh_safe(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched eigh that cannot fail the batch: a non-finite matrix, or one the solver fails on,
    gets eigenvalues PAD_LEVEL and unit vectors and ok = False."""
    ok = torch.isfinite(a).all((1, 2))
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device).expand_as(a)
    a = torch.where(ok[:, None, None], a, PAD_LEVEL * eye)
    try:
        e, c = torch.linalg.eigh(a)
    except torch.linalg.LinAlgError:
        e, c = torch.empty_like(a[..., 0]), torch.empty_like(a)
        for i in range(a.shape[0]):
            try:
                e[i], c[i] = torch.linalg.eigh(a[i])
            except torch.linalg.LinAlgError:
                ok[i], e[i], c[i] = False, PAD_LEVEL, eye[i]
    return e, c, ok


class KsFock(C.Structure):
    _fields_ = [(n, C.c_void_p) for n in ("H0", "S", "D", "Q", "vao", "vmp", "valid", "idx",
                                          "fock")] + [("pad", C.c_double), ("n", C.c_int),
                                                      ("nsel", C.c_int)]  # fmt: skip


class KsMulliken(C.Structure):
    _fields_ = [(n, C.c_void_p) for n in ("P", "S", "D", "Q", "out", "idx")] + [
        ("n", C.c_int), ("nsel", C.c_int)]  # fmt: skip


class KsFermi(C.Structure):
    _fields_ = [(n, C.c_void_p) for n in ("emo", "nel", "mol", "occ", "ts")] + [
        ("kt", C.c_double), ("n", C.c_int), ("njob", C.c_int)]  # fmt: skip


@cache
def _lib():
    so = native.lib()
    so.ksxtb_fock.argtypes = [C.POINTER(KsFock), C.c_void_p]
    so.ksxtb_mulliken.argtypes = [C.POINTER(KsMulliken), C.c_void_p]
    so.ksxtb_fermi.argtypes = [C.POINTER(KsFermi), C.c_void_p]
    so.ksxtb_fock.restype = so.ksxtb_mulliken.restype = so.ksxtb_fermi.restype = C.c_int
    return so


def occupations(emo: torch.Tensor, nel_spin) -> tuple[torch.Tensor, torch.Tensor]:
    """Both spin channels' Fermi filling of each molecule (B, n), summed, and the entropy
    term (B,): scf.cu's warp-per-channel kernel (`fermi` in torch, the reference)."""
    b, n = emo.shape
    e = emo.contiguous()
    nel = torch.cat([nel_spin[0], nel_spin[1]]).contiguous()
    mol = torch.arange(b, device=emo.device).repeat(2).contiguous()
    occ = torch.empty(2 * b, n, dtype=F64, device=emo.device)
    ts = torch.empty(2 * b, dtype=F64, device=emo.device)
    args = KsFermi(*[t.data_ptr() for t in (e, nel, mol, occ, ts)], T.KT_300K, n, 2 * b)
    native._check(_lib().ksxtb_fermi(C.byref(args), _stream()), "ksxtb_fermi")
    return occ[:b] + occ[b:], ts[:b] + ts[b:]


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def fock(g: Geometry, lay: Layout, vsh, vat, vdp, vqp, idx=None) -> torch.Tensor:
    """tblite's H1 = H0 + add_vao_to_h1 + add_vmp_to_h1 (dipole, quadrupole), padded, for the
    molecules `idx` (default all): the fused kernel of scf.cu (`fock_reference` in torch)."""
    if idx is None:
        idx = torch.arange(g.S.shape[0], device=g.S.device)
    vao = torch.cat([vsh + vat[lay.sh_atom], vsh.new_zeros(1)])[lay.ao_shell].contiguous()
    vmp = torch.cat([torch.cat([vdp, vqp], 1), vdp.new_zeros(1, 9)])[lay.ao_atom].contiguous()
    out = torch.empty(len(idx), lay.nmax, lay.nmax, dtype=F64, device=g.S.device)
    idx = idx.contiguous()
    args = KsFock(*[t.data_ptr() for t in (g.H0, g.S, g.D, g.Q, vao, vmp, lay.valid_u8, idx, out)],
                  PAD_LEVEL, lay.nmax, len(idx))  # fmt: skip
    native._check(_lib().ksxtb_fock(C.byref(args), _stream()), "ksxtb_fock")
    return out


def fock_reference(g: Geometry, lay: Layout, vsh, vat, vdp, vqp, idx=None) -> torch.Tensor:
    """The same in torch."""
    sel = slice(None) if idx is None else idx
    ao_shell, ao_atom, valid = lay.ao_shell[sel], lay.ao_atom[sel], lay.valid[sel]
    s_, h0, d, q = g.S[sel], g.H0[sel], g.D[sel], g.Q[sel]
    vsh_ext = torch.cat([vsh + vat[lay.sh_atom], vsh.new_zeros(1)])
    vao = vsh_ext[ao_shell]  # (B, n)
    f = h0 - 0.5 * s_ * (vao[:, :, None] + vao[:, None, :])
    vdp_ext = torch.cat([vdp, vdp.new_zeros(1, 3)])[ao_atom]  # (B, n, 3) at atom(nu)
    vqp_ext = torch.cat([vqp, vqp.new_zeros(1, 6)])[ao_atom]
    w = torch.einsum("bcmn,bnc->bmn", d, vdp_ext) + torch.einsum("bcmn,bnc->bmn", q, vqp_ext)
    f = f - 0.5 * (w + w.transpose(1, 2))
    return f + torch.diag_embed((~valid).to(F64) * PAD_LEVEL)


def solve(g: Geometry, f: torch.Tensor, idx=None):
    """Orbital energies (B, n), coefficients (B, n, n), columns the orbitals (F C = S C e), and
    whether each solve succeeded (`eigh_safe`), in fp64 throughout."""
    linv = g.Linv if idx is None else g.Linv[idx]
    fp = linv @ f @ linv.transpose(1, 2)
    e, c, ok = eigh_safe(0.5 * (fp + fp.transpose(1, 2)))
    return e, linv.transpose(1, 2) @ c, ok


def solve_fp32(g: Geometry, f: torch.Tensor, idx):
    """The same in fp32 (transform, eigensolve, back-transform): the early SCF cycles. The
    orbital energies come back in fp64, the coefficients in fp32."""
    e, c, ok = eig_fp32(g, f, idx)
    return e, g.Linv32[idx].transpose(1, 2) @ c, ok


def eig_fp32(g: Geometry, f: torch.Tensor, idx):
    """The fp32 eigensolve in the orthogonalized basis: orbital energies (fp64) and the
    orthonormal eigenvectors c of L^-1 F L^-T (fp32, before the back-transform)."""
    linv = g.Linv32[idx]
    fp = linv @ f.float() @ linv.transpose(1, 2)
    e, c, ok = eigh_safe(0.5 * (fp + fp.transpose(1, 2)))
    return e.double(), c, ok


REFINE_TINY = 1e-12  # occupations within this of each other count as equal
REFINE_NEAR = 1e-3  # Hartree: orbital pairs closer than this are near-degenerate
REFINE_FLAT = 1e-8  # occupations farther than this from an integer are fractional
REFINE_CHECK = 1e-7  # the last refinement step's largest rotation; above it, an fp64 eigh


def refine_occupied(g: Geometry, f, c32, emo32, occ, idx, steps: int = 2):
    """fp64-accurate occupied orbitals from an fp32 eigensolve, without an fp64 eigensolve.

    Ogita and Aishima's refinement of an approximate eigenbasis, in the AO basis (F x = e S x)
    and restricted to the columns the density needs: the first K orbitals, K the most occupied
    orbitals of any molecule. Per step, with X the K columns and C the whole basis (X first):
    R = I - C^T S X, M = C^T F X, e_j = M_jj / (1 - R_jj), and X += C E with
    E_ij = (M_ij + e_j R_ij) / (e_j - e_i) for orbitals of different occupation, R_ij / 2 (the
    orthonormalization) for equal ones. The virtual orbitals beyond K keep their fp32 energies
    and only serve as the basis. Near-degenerate pairs (closer than REFINE_NEAR) are not
    rotated: between orbitals of equal integer occupation the rotation does not move the
    density. A molecule with a near-degenerate pair that holds a fractionally occupied orbital
    (more than REFINE_FLAT from an integer: there the split of the occupations follows the
    fp32 eigenvalues' noise) goes to an fp64 eigensolve, as does one whose last step still
    rotated by more than REFINE_CHECK.

    Returns the orbital energies (b, n) and the basis (b, n, n, the first K columns refined) in
    fp64, the number K, and ok (b,)."""
    s, linv = g.S[idx], g.Linv[idx]
    cf = linv.transpose(1, 2) @ c32.double()
    k = max(int((occ > REFINE_TINY).sum(1).max()), 1)
    dfo = (occ[:, :, None] - occ[:, None, :k]).abs()  # (b, n, k)
    near = (emo32[:, :, None] - emo32[:, None, :k]).abs() < REFINE_NEAR
    near &= ~torch.eye(*near.shape[1:], dtype=torch.bool, device=near.device)
    frac = (occ - occ.round()).abs() > REFINE_FLAT
    differ = (dfo > REFINE_TINY) & ~near
    clean = ~(near & (frac[:, :, None] | frac[:, None, :k])).any((1, 2))
    eye = torch.eye(cf.shape[1], k, dtype=F64, device=cf.device)
    sf = torch.cat([s, f], 1)  # one GEMM for S X and F X
    emo = emo32.clone()
    rot = torch.zeros_like(emo32[:, 0])
    for step in range(steps):
        x = cf[:, :, :k]
        rm = cf.transpose(1, 2) @ (sf @ x).view(len(cf), 2, -1, k).transpose(1, 2).reshape(
            len(cf), -1, 2 * k)  # fmt: skip
        r, m = eye - rm[:, :, :k], rm[:, :, k:]
        lam = torch.diagonal(m[:, :k], dim1=1, dim2=2) / (1 - torch.diagonal(r[:, :k], dim1=1,
                                                                              dim2=2))  # fmt: skip
        emo[:, :k] = lam
        den = lam[:, None, :] - emo[:, :, None]
        e = torch.where(differ, (m + lam[:, None, :] * r) / torch.where(differ, den, 1.0), r / 2)
        if step == steps - 1:
            rot = torch.where(differ, e.abs(), 0.0).amax((1, 2))
        cf[:, :, :k] = x + cf @ e
    return emo, cf, k, clean & (rot <= REFINE_CHECK) & torch.isfinite(rot)


def fermi(emo: torch.Tensor, nel: torch.Tensor, kt: float = T.KT_300K) -> torch.Tensor:
    """tblite's get_fermi_filling for each molecule of a batch, one spin channel (B, n)."""
    b, n = emo.shape
    ar = torch.arange(n, device=emo.device)[None, :]
    homo = torch.floor(nel).long()
    frac = torch.remainder(nel, 1.0)
    occ = (ar < homo[:, None]).to(F64) + torch.where(ar == homo[:, None], frac[:, None], 0.0)
    homo = torch.where(frac > 0.5, homo + 1, homo)
    lo = emo.gather(1, (homo.clamp(min=1) - 1)[:, None])[:, 0]
    hi = emo.gather(1, (homo + 1).clamp(max=n)[:, None] - 1)[:, 0]
    ef = 0.5 * (lo + hi)
    occt = homo.to(F64)
    live_mol = homo > 0
    thr, tiny = math.sqrt(np.finfo(float).eps), math.sqrt(np.finfo(float).tiny)
    done = ~live_mol
    for it in range(200):
        x = (emo - ef[:, None]) / kt
        live = x < 50
        ex = torch.exp(torch.where(live, x, 0.0))
        new = torch.where(live, 1.0 / (ex + 1.0), 0.0)
        occ = torch.where(done[:, None], occ, new)
        dfermi = torch.where(live, ex / (kt * (ex + 1.0) ** 2), 0.0).sum(1)
        total = new.sum(1)
        step = torch.where(dfermi > tiny, (occt - total) / dfermi, 0.0)
        ok = (occt - total).abs() <= thr
        ef = torch.where(done, ef, ef + step)
        done = done | ok
        if it % 8 == 7 and bool(done.all()):  # one host sync per 8 Newton steps
            break
    return occ


def entropy(occ: torch.Tensor, kt: float = T.KT_300K) -> torch.Tensor:
    """tblite's -TS of one spin channel per molecule (B,)."""
    s = occ * torch.log(occ.clamp(min=1e-300)) + (1 - occ) * torch.log((1 - occ).clamp(min=1e-300))
    return kt * torch.where((occ > 0) & (occ < 1), s, 0.0).sum(1)


def mulliken(g: Geometry, lay: Layout, p: torch.Tensor, n0sh: torch.Tensor, nat: int, idx=None):
    """Output shell charges (nsh,), atomic dipoles (nat, 3) and quadrupoles (nat, 6): the per-AO
    sums of scf.cu's kernel, reduced to shells and atoms (`mulliken_reference` in torch). With
    `idx` only those molecules are summed; the others' charges are the reference occupations
    and their multipoles zero (an SCF discards them)."""
    b, n = p.shape[0], lay.nmax
    if idx is None:
        idx = torch.arange(b, device=p.device)
    per = torch.zeros(b, n, 10, dtype=F64, device=p.device)
    pc, idx = p.contiguous(), idx.contiguous()
    args = KsMulliken(*[t.data_ptr() for t in (pc, g.S, g.D, g.Q, per, idx)], n, len(idx))
    native._check(_lib().ksxtb_mulliken(C.byref(args), _stream()), "ksxtb_mulliken")
    nsh = len(n0sh)
    pop = torch.zeros(nsh + 1, dtype=F64, device=p.device).index_add(
        0, lay.ao_shell.reshape(-1), per[..., 0].reshape(-1))[:nsh]  # fmt: skip
    mp = torch.zeros(nat + 1, 9, dtype=F64, device=p.device).index_add(
        0, lay.ao_atom.reshape(-1), per[..., 1:].reshape(-1, 9))[:nat]  # fmt: skip
    return n0sh - pop, mp[:, :3].contiguous(), mp[:, 3:].contiguous()


def mulliken_reference(g: Geometry, lay: Layout, p: torch.Tensor, n0sh: torch.Tensor, nat: int):
    """The same in torch."""
    pop = (p * g.S).sum(1)  # column sums, (B, n)
    nsh = len(n0sh)
    sh = lay.ao_shell.reshape(-1)
    qsh = n0sh - torch.zeros(nsh + 1, dtype=F64, device=p.device).index_add(
        0, sh, pop.reshape(-1))[:nsh]  # fmt: skip
    at = lay.ao_atom.reshape(-1)
    out = []
    for mat, k in ((g.D, 3), (g.Q, 6)):
        per = -(p[:, None] * mat).sum(2)  # (B, k, n)
        acc = torch.zeros(nat + 1, k, dtype=F64, device=p.device)
        out.append(acc.index_add(0, at, per.transpose(1, 2).reshape(-1, k))[:nat])
    return qsh, out[0], out[1]


# --- the mixed SCF variables and tblite's modified Broyden mixer ------------------------------


class VarMap:
    """(qsh, dp, qp) of a batch <-> one padded vector per molecule (B, V), tblite's order:
    the shell charges, then the dipoles, then the quadrupoles of each molecule."""

    def __init__(self, mols, device="cuda") -> None:
        sh_idx, dp_idx, qp_idx, nvar = [], [], [], []
        for m, (z, _) in enumerate(mols):
            nsh, nat = len(mol_basis(np.asarray(z, dtype=int)).sh_atom), len(z)
            nvar.append(nsh + 9 * nat)
            sh_idx.append((m, np.arange(nsh)))
            dp_idx.append((m, nsh + np.arange(3 * nat)))
            qp_idx.append((m, nsh + 3 * nat + np.arange(6 * nat)))
        self.v = int(max(nvar))
        self.nvar = torch.as_tensor(nvar, dtype=F64, device=device)

        def flat(parts):
            idx = np.concatenate([m * self.v + i for m, i in parts])
            return torch.as_tensor(idx, device=device)

        self.sh, self.dp, self.qp = flat(sh_idx), flat(dp_idx), flat(qp_idx)
        self.b = len(mols)

    def pack(self, qsh, dp, qp) -> torch.Tensor:
        out = torch.zeros(self.b * self.v, dtype=F64, device=qsh.device)
        out[self.sh], out[self.dp], out[self.qp] = qsh, dp.reshape(-1), qp.reshape(-1)
        return out.view(self.b, self.v)

    def unpack(self, vec: torch.Tensor):
        flat = vec.reshape(-1)
        return flat[self.sh], flat[self.dp].view(-1, 3), flat[self.qp].view(-1, 6)


class Broyden:
    """tblite's broyden (Johnson's modified Broyden, damping 0.4, the whole history) for every
    molecule of a batch at once, each molecule at its own iteration: `reset` restarts some
    molecules from new inputs while the others continue, and molecules outside `active` keep
    their state unchanged (a later call continues them).

    The history lives in preallocated buffers (grown by doubling, shrunk back once the molecules
    still stepping use under a quarter of them: one long SCF does not leave a batch's mixer
    wide for the rest of its life, `_reserve`), slot j of molecule m holding
    its j-th normalized residual difference; the Gram matrix is updated by one row and column
    per iteration, and each molecule's small system, over its own h_m slots (the unused ones
    identity rows with g = 0, so y = 0 there), is solved in its equilibrated form: with
    W = diag(omega), tblite's (W A W + 1e-4 I) c = W g is (A + 1e-4 W^-2) y = g for y = W c, the
    products the update needs, an SPD matrix with unit diagonal solved by Cholesky (LU, as
    tblite, for a molecule where that fails). Equal to `BroydenReference` (the list form) run
    on each molecule alone, to rounding."""

    def __init__(self, q0: torch.Tensor, damping: float = 0.4, chunk: int = 16) -> None:
        b, v = q0.shape
        self.q, self.alpha = q0.clone(), damping
        self.qlast, self.dqlast = torch.zeros_like(q0), torch.zeros_like(q0)
        self.it = torch.zeros(b, dtype=torch.long, device=q0.device)  # iterations done
        self.h = torch.zeros_like(self.it)  # history slots in use
        self.hbound = 0  # >= h.max(), exact after each solve (read in its host sync)
        self.started = False  # no molecule past its first iteration yet: the plain step
        self.chunk = chunk
        self.read = None  # the `also` values of the last `next`'s host sync (None: no solve)
        self.df, self.u = q0.new_zeros(b, chunk, v), q0.new_zeros(b, chunk, v)
        self.winv2, self.a = q0.new_zeros(b, chunk), q0.new_zeros(b, chunk, chunk)

    def reset(self, which: torch.Tensor, q0: torch.Tensor) -> None:
        """Restart the molecules `which` (B,) bool from q0's rows: iteration 0, an empty
        history. Their history rows are zeroed, so a failed run's NaNs cannot reach a later
        solve (y = 0 on an unused slot still multiplies its row)."""
        w = which[:, None]
        self.q = torch.where(w, q0, self.q)
        self.it = torch.where(which, 0, self.it)
        self.h = torch.where(which, 0, self.h)
        for t in (self.qlast, self.dqlast, self.winv2):
            t.masked_fill_(w, 0.0)
        for t in (self.df, self.u, self.a):
            t.masked_fill_(w[:, :, None], 0.0)

    def retire(self, which: torch.Tensor) -> None:
        """The molecules `which` stop: their history no longer counts toward the solve's width
        (`hbound`). A retired molecule is `reset` before it steps again."""
        self.it = torch.where(which, 0, self.it)
        self.h = torch.where(which, 0, self.h)

    def _reserve(self) -> None:
        """Room for one more slot of every molecule (h <= hbound < capacity): the buffers
        double when full, and shrink to the smallest chunk multiple of twice the slots in use
        when that is a quarter of them or less (slots >= hbound are unused: dropping them loses
        nothing)."""
        cap = self.df.shape[1]
        if self.hbound >= cap:
            pad = torch.nn.functional.pad
            self.df, self.u = pad(self.df, (0, 0, 0, cap)), pad(self.u, (0, 0, 0, cap))
            self.winv2, self.a = pad(self.winv2, (0, cap)), pad(self.a, (0, cap, 0, cap))
            return
        new = self.chunk
        while new <= 2 * self.hbound:
            new *= 2
        if new <= cap // 4:
            own = {"memory_format": torch.contiguous_format}
            self.df, self.u = self.df[:, :new].clone(**own), self.u[:, :new].clone(**own)
            self.winv2, self.a = (
                self.winv2[:, :new].clone(**own),
                self.a[:, :new, :new].clone(**own),
            )

    def next(self, dq: torch.Tensor, active: torch.Tensor, retire=None, also=None) -> torch.Tensor:
        """The next input of the active molecules from their residuals dq (B, V). `retire`: the
        molecules whose SCF has just ended, retired (`retire`) before the solve reads its width.
        `also`: a 1-d integer tensor read in the solve's host sync, into `self.read` (left None
        when no solve ran, so the caller reads it itself)."""
        if retire is not None:
            self.retire(retire)
        self.read = None
        q, act = self.q, active[:, None]
        first = active & (self.it == 0)
        self.it = self.it + active
        step = q + self.alpha * dq
        if self.started:
            later = active & ~first
            step = torch.where(later[:, None], self._update(dq, later, also), step)
        self.started = True
        self.dqlast, self.qlast = torch.where(act, dq, self.dqlast), torch.where(act, q, self.qlast)
        self.q = torch.where(act, step, q)
        return self.q

    def _update(self, dq: torch.Tensor, later: torch.Tensor, also=None) -> torch.Tensor:
        """The Broyden step of the molecules `later` (past their first iteration): a new slot
        h_m each, the Gram row and column, the system over slots < h_m + 1."""
        q, lat = self.q, later[:, None]
        norm = torch.sqrt((dq * dq).sum(1))
        om = torch.where(norm > 0.01 / 1e5, 0.01 / norm.clamp(min=1e-300), 1e5).clamp(min=1.0)
        df = dq - self.dqlast
        inv = 1.0 / torch.sqrt((df * df).sum(1)).clamp(min=np.finfo(float).eps)
        df = inv[:, None] * df
        self._reserve()
        hn, k, ar = self.hbound + 1, self.h, torch.arange(len(q), device=q.device)
        self.df[ar, k] = torch.where(lat, df, self.df[ar, k])
        self.winv2[ar, k] = torch.where(later, 1.0 / (om * om), self.winv2[ar, k])
        self.h = self.h + later
        valid = torch.arange(hn, device=q.device)[None, :] < self.h[:, None]  # (B, hn)
        dfs = self.df[:, :hn]
        row = torch.where(valid & lat, (dfs @ df[:, :, None])[:, :, 0], 0.0)
        self.a[ar, k, :hn] = torch.where(lat, row, self.a[ar, k, :hn])
        self.a[ar, :hn, k] = torch.where(lat, row, self.a[ar, :hn, k])
        g = torch.where(valid & lat, (dfs @ dq[:, :, None])[:, :, 0], 0.0)
        pair = (valid & lat)[:, :, None] & valid[:, None, :]
        eye = torch.eye(hn, dtype=q.dtype, device=q.device)
        m = torch.where(pair, self.a[:, :hn, :hn] + torch.diag_embed(1e-4 * self.winv2[:, :hn]),
                        eye)  # fmt: skip
        extra = self.h.max()[None] if also is None else torch.cat([self.h.max()[None], also])
        y, read = _spd_solve(m, g, also=extra)
        self.hbound, self.read = read[0], (read[1:] if also is not None else None)
        self.u[ar, k] = torch.where(lat, self.alpha * df + inv[:, None] * (q - self.qlast),
                                    self.u[ar, k])  # fmt: skip
        return q + self.alpha * dq - (y[:, None, :] @ self.u[:, :hn])[:, 0]


def _spd_solve(a: torch.Tensor, g: torch.Tensor, also=None):
    """a y = g for a batch of SPD matrices: Cholesky, LU where Cholesky fails. `also` (a 1-d
    integer tensor or None) is read in the same host sync as the failure check; returns y and
    its values (a list, or None)."""
    chol, info = torch.linalg.cholesky_ex(a)
    y = torch.cholesky_solve(g[:, :, None], chol)[:, :, 0]
    bad = info != 0
    extra = bad.new_zeros(0, dtype=torch.long) if also is None else also.long().reshape(-1)
    nbad, *values = torch.cat([bad.sum()[None].long(), extra]).tolist()
    if nbad:
        y[bad] = torch.linalg.solve(a[bad], g[bad][:, :, None])[:, :, 0]
    return y, (values if also is not None else None)


class BroydenReference:
    """The list form of the mixer, tblite's arithmetic as written (the test reference)."""

    def __init__(self, q0: torch.Tensor, damping: float = 0.4) -> None:
        self.q, self.alpha, self.iter = q0.clone(), damping, 0
        self.qlast = self.dqlast = None
        self.df, self.u, self.omega = [], [], []
        self.a = q0.new_zeros(q0.shape[0], 0, 0)

    def next(self, dq: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        self.iter += 1
        q, act = self.q, active[:, None]
        if self.iter == 1:
            self.dqlast, self.qlast = dq.clone(), q.clone()
            self.q = torch.where(act, q + self.alpha * dq, q)
            return self.q
        norm = torch.sqrt((dq * dq).sum(1))
        om = torch.where(norm > 0.01 / 1e5, 0.01 / norm.clamp(min=1e-300), 1e5).clamp(min=1.0)
        df = dq - self.dqlast
        inv = 1.0 / torch.sqrt((df * df).sum(1)).clamp(min=np.finfo(float).eps)
        df = inv[:, None] * df
        self.omega.append(om)
        self.df.append(df)
        dfs = torch.stack(self.df, 1)  # (B, h, V)
        h = dfs.shape[1]
        a = dfs @ dfs.transpose(1, 2)
        w = torch.stack(self.omega, 1)  # (B, h)
        c = w * (dfs @ dq[:, :, None])[:, :, 0]
        beta = w[:, :, None] * w[:, None, :] * a + 1e-4 * torch.eye(h, dtype=F64, device=q.device)
        coef = torch.linalg.solve(beta, c[:, :, None])[:, :, 0]
        self.u.append(self.alpha * df + inv[:, None] * (q - self.qlast))
        us = torch.stack(self.u, 1)
        new = q + self.alpha * dq - ((w * coef)[:, :, None] * us).sum(1)
        self.dqlast, self.qlast = torch.where(act, dq, self.dqlast), torch.where(act, q, self.qlast)
        self.q = torch.where(act, new, q)
        return self.q


# --- the SCF ---------------------------------------------------------------------------------


@dataclass
class SCFResult:
    energy: torch.Tensor  # (B,) total energy of the output density, Hartree
    converged: torch.Tensor  # (B,) bool
    iterations: torch.Tensor  # (B,) cycles run
    density: torch.Tensor  # (B, n, n)
    wdensity: torch.Tensor  # (B, n, n) energy-weighted density
    emo: torch.Tensor  # (B, n); a refined molecule's occupied energies are Rayleigh quotients
    occ: torch.Tensor  # (B, n) both spins
    qsh: torch.Tensor
    dp: torch.Tensor
    qp: torch.Tensor
    status: torch.Tensor  # (B,) SCF_OK, SCF_MAXITER, SCF_BAD_GEOMETRY or SCF_NONFINITE
    vin: dict  # the last cycle's input potentials (vsh, vat, vdp, vqp): tblite's gradient's


SCF_OK, SCF_MAXITER, SCF_BAD_GEOMETRY, SCF_NONFINITE = 0, 1, 2, 3
SCF_STATUS = {SCF_MAXITER: "no convergence within the cycle limit",
              SCF_BAD_GEOMETRY: "non-finite integrals or S not positive definite",
              SCF_NONFINITE: "non-finite Fock matrix, eigensolve or energy"}  # fmt: skip


@dataclass
class SCFRun:
    """The SCF state of a batch, per molecule, kept between `SCFBatch.advance` calls: the
    mixer, the orbital buffers (c, p, x, f, emo, occ, ts of `SCFBatch._cycle`), each
    molecule's input potentials of its last cycle, output variables, energies, cycle count and
    status, and which molecules run, are in the precise phase, may still fall back to the SAD
    guess, and fell back."""

    mixer: Broyden
    st: dict
    vin: dict
    qout: torch.Tensor  # (B, V) the last cycle's output variables
    energy: torch.Tensor  # (B,)
    elast: torch.Tensor  # (B,) the last precise cycle's energy, inf after an fp32 one
    iters: torch.Tensor  # (B,) cycles of the current SCF
    status: torch.Tensor  # (B,) SCF_MAXITER while running
    running: torch.Tensor  # (B,) bool
    precise: torch.Tensor  # (B,) bool
    may_fall_back: torch.Tensor  # (B,) bool
    fell: torch.Tensor  # (B,) bool


class SCFBatch:
    """A batch of molecules (z, xyz Bohr, charge, unpaired electrons) and its GFN2 SCF."""

    def __init__(self, mols, solvent: bool, accuracy: float = 1.0, max_iter: int = 250) -> None:
        self.geo_mols = [(np.asarray(z, dtype=int), np.asarray(x, float)) for z, x, _, _ in mols]
        self.charge = torch.tensor([float(c) for *_, c, _ in mols], dtype=F64, device="cuda")
        uhf = torch.tensor([float(u) for *_, u in mols], dtype=F64, device="cuda")
        self.solvent, self.max_iter = solvent, max_iter
        self.fallbacks = 0  # precise-phase eigensolves that needed fp64 eigh (`_precise`)
        self.econv, self.pconv = 1e-6 * accuracy, 2e-5 * accuracy
        self.db = native.DeviceBatch(native.host_batch(self.geo_mols))
        self.cb = K.ClassicalBatch(self.geo_mols, solvent)
        self.lay = layout(self.db.hb, self.geo_mols)
        self.vars = VarMap(self.geo_mols)
        n0 = [T.reference_occupations(mol_basis(z)) for z, _ in self.geo_mols]
        self.n0sh = torch.as_tensor(np.concatenate(n0), device="cuda")
        self.sh_atom = self.lay.sh_atom
        self.n0at = torch.zeros(self.db.hb.nat, **_f64()).index_add(0, self.sh_atom, self.n0sh)
        self.atom_mol = self.cb.t["atom_mol"].long()
        nel = torch.stack([torch.as_tensor(x.sum()) for x in n0]).to("cuda") - self.charge
        self.nel_spin = ((nel + uhf) / 2, (nel - uhf) / 2)
        self.se0, self.kcn = (torch.from_numpy(a).cuda()
                              for a in K._shell_selfenergy_tables(self.geo_mols))  # fmt: skip
        self.set_geometry(None)

    def set_geometry(self, xyz) -> None:
        """Recompute everything geometric (xyz: concatenated (nat, 3) Bohr, None = as built)."""
        if xyz is not None:
            self.db.set_xyz(xyz)
            self.cb.set_xyz(xyz)
        self.cb.geometry()
        self.db.t["sh_se"].copy_(self.se0 - self.kcn * self.cb.w["cn"][self.sh_atom])
        self.db.t["sh_dsedcn"].copy_(-self.kcn)
        self.db.forward()
        self.g = padded_geometry(self.db, self.lay)

    def guess(self):
        """tblite's SAD guess: charge / nat per atom, split over shells by reference occupation."""
        nat = torch.bincount(self.atom_mol).to(F64)
        qat = (self.charge / nat)[self.atom_mol]
        qsh = self.n0sh / self.n0at[self.sh_atom] * qat[self.sh_atom]
        nat_t = self.db.hb.nat
        return qsh, torch.zeros(nat_t, 3, **_f64()), torch.zeros(nat_t, 6, **_f64())

    def qat(self, qsh: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.db.hb.nat, **_f64()).index_add(0, self.sh_atom, qsh)

    def _potentials(self, qsh, dp, qp, slot: str = "in") -> dict:
        """Energies and potentials (no gradient) at the variables, into the slot's buffers."""
        if not hasattr(self, "_slots"):
            self._slots = {k: self.cb.new_outputs() for k in ("in", "out")}
        return self.cb.evaluate(qsh, self.qat(qsh), dp, qp, gradient=False,
                                out=self._slots[slot])  # fmt: skip

    def _occupations(self, emo, idx):
        return occupations(emo, [n[idx] for n in self.nel_spin])

    def _energy(self, p, ts, out_pot) -> torch.Tensor:
        band = (p * self.g.H0).sum((1, 2))
        shift = T.solvent_shift() if self.solvent else 0.0
        return band + ts + out_pot["energy"].sum(1) + shift

    def _cycle(self, st: dict, pot: dict, active, precise, idx=None) -> torch.Tensor:
        """Fock, eigensolve and density of the active molecules, into the state buffers;
        returns the molecules whose eigensolve failed (B,)."""
        failed = torch.zeros_like(active)
        idx = torch.nonzero(active).squeeze(1) if idx is None else idx
        if not len(idx):
            return failed
        f = fock(self.g, self.lay, pot["vsh"], pot["vat"], pot["vdp"], pot["vqp"], idx)
        emo, c, ok = eig_fp32(self.g, f, idx)  # one fp32 eigensolve for every active molecule
        failed[idx] = ~ok
        occ, ts = self._occupations(emo, idx)
        sel = precise[idx]
        lo, hi = torch.nonzero(~sel).squeeze(1), torch.nonzero(sel).squeeze(1)
        if len(lo):  # the fp32 phase: density in fp32
            i = idx[lo]
            cc = self.g.Linv32[i].transpose(1, 2) @ c[lo]
            st["emo"][i], st["occ"][i], st["ts"][i] = emo[lo], occ[lo], ts[lo]
            st["p"][i] = ((cc * occ[lo][:, None, :].float()) @ cc.transpose(1, 2)).double()
        if len(hi):
            self._precise(st, f[hi], c[hi], emo[hi], occ[hi], idx[hi], failed)
        return failed

    REFINE_STEPS = 2

    def _precise(self, st, f, c32, emo32, occ32, idx, failed) -> None:
        """The precise phase's orbitals: the fp32 eigenvectors refined in fp64
        (`refine_occupied`), an fp64 eigensolve for the molecules it flags; the density from
        the occupied columns, C kept for the energy-weighted density."""
        emo, cf, k, ok = refine_occupied(self.g, f, c32, emo32, occ32, idx, self.REFINE_STEPS)
        back = torch.nonzero(~ok).squeeze(1)
        if len(back):
            e64, c64, ok64 = solve(self.g, f[back], idx[back])
            emo[back], cf[back] = e64, c64
            failed[idx[back]] |= ~ok64
            self.fallbacks += len(back)
        occ, ts = self._occupations(emo, idx)
        if len(back):
            k = max(k, int((occ > 0).sum(1).max()))
        x = cf[:, :, :k]
        st["emo"][idx], st["occ"][idx], st["ts"][idx] = emo, occ, ts
        st["p"][idx] = (x * occ[:, None, :k]) @ x.transpose(1, 2)
        st["c"][idx], st["f"][idx] = cf, f

    def _wdensity(self, st: dict, idx) -> None:
        """tblite's energy-weighted density C diag(f e) C^T of the last (fp64) cycle, as
        C_k sqrt(f) (C_k^T F C_k) sqrt(f) C_k^T over the occupied columns: the same at exact
        eigenvectors, and unchanged by rotations among orbitals of equal occupation, which the
        refinement (`refine_occupied`) leaves unresolved."""
        if not len(idx):
            return
        occ = st["occ"][idx]
        k = max(int((occ > 0).sum(1).max()), 1)
        c, f = st["c"][idx, :, :k], st["f"][idx]
        y = c * occ[:, None, :k].sqrt()
        st["x"][idx] = y @ (c.transpose(1, 2) @ (f @ c)) @ y.transpose(1, 2)

    SWITCH = 1e-4  # residual RMS below which a molecule's cycles turn fp64-accurate

    def new_run(self, start=None) -> SCFRun:
        """An SCF state with no molecule running: each molecule's SCF variables at `start`
        (qsh, dp, qp), default the SAD guess, its status SCF_MAXITER (SCF_BAD_GEOMETRY for a
        bad geometry), its energy NaN and no cycle run, as a molecule `run` leaves inactive."""
        q0 = self.vars.pack(*(start if start is not None else self.guess()))
        b, n = self.vars.b, self.lay.nmax
        st = {k: torch.zeros(b, n, n, **_f64()) for k in ("c", "p", "x", "f")}
        st.update(emo=torch.zeros(b, n, **_f64()), occ=torch.zeros(b, n, **_f64()),
                  ts=torch.zeros(b, **_f64()))  # fmt: skip
        no = torch.zeros(b, dtype=torch.bool, device="cuda")
        status = torch.where(self.g.bad, SCF_BAD_GEOMETRY, SCF_MAXITER)
        return SCFRun(Broyden(q0), st, self._zero_potentials(), q0.clone(),
                      torch.full((b,), torch.nan, **_f64()), torch.full((b,), torch.inf, **_f64()),
                      torch.zeros(b, dtype=torch.long, device="cuda"), status, no.clone(),
                      no.clone(), no.clone(), no.clone())  # fmt: skip

    def start(self, state: SCFRun, which, start=None, fallback=False) -> None:
        """(Re)start the SCF of the molecules `which` (B,) bool from the rows of `start` (full
        batch (qsh, dp, qp)), default the SAD guess, at the current geometry; the others keep
        their state, mid-run or not. A bad geometry gets SCF_BAD_GEOMETRY and does not run.
        With `fallback` (True, or a (B,) bool mask of the molecules it applies to) and a
        `start`, a molecule whose SCF then ends unconverged (SCF_MAXITER, SCF_NONFINITE)
        restarts once from the guess within the same evaluation, as `TbliteEnergy` restarts
        tblite, and is marked in `state.fell`. A molecule whose `start` is the guess itself
        (an optimization's first point) takes no fallback: tblite's would rerun the same SCF."""
        bad = self.g.bad
        go = which & ~bad
        self._reset(state, which, self.vars.pack(*(start if start is not None else self.guess())))
        state.status = torch.where(which, torch.where(bad, SCF_BAD_GEOMETRY, SCF_MAXITER),
                                   state.status)  # fmt: skip
        state.running = (state.running & ~which) | go
        fb = go & fallback if start is not None else torch.zeros_like(go)
        state.may_fall_back = torch.where(which, fb, state.may_fall_back)
        state.fell = state.fell & ~which

    def _reset(self, s: SCFRun, which, q0) -> None:
        """Every per-SCF quantity of the molecules `which` as a fresh SCF from q0 has it."""
        s.mixer.reset(which, q0)
        s.qout = torch.where(which[:, None], q0, s.qout)
        s.energy = torch.where(which, torch.nan, s.energy)
        s.elast = torch.where(which, torch.inf, s.elast)
        s.iters = torch.where(which, 0, s.iters)
        s.precise = s.precise & ~which
        for t in s.st.values():
            t.masked_fill_(which.view(-1, *([1] * (t.dim() - 1))), 0.0)
        sh, at = which[self.atom_mol[self.sh_atom]], which[self.atom_mol]
        for k, t in s.vin.items():
            s.vin[k] = torch.where((sh if k == "vsh" else at).view(-1, *([1] * (t.dim() - 1))),
                                   0.0, t)  # fmt: skip

    def advance(self, state: SCFRun, carry: float | None = None) -> torch.Tensor:
        """SCF cycles of the running molecules; returns the molecules (B,) bool whose SCF ended
        during the call (converged or failed; a fallback restart is not an end). carry=None
        runs until none is running; with a number, the call returns once at most carry x the
        molecules running at entry still run (after one cycle at least; carry >= 1: exactly
        one), and those keep their state for the next call."""
        ended = torch.zeros_like(state.running)
        idx = torch.nonzero(state.running).squeeze(1)
        n0, cycles = len(idx), 0
        while len(idx) and not (carry is not None and cycles and len(idx) <= carry * n0):
            ended |= self._iterate(state, idx)
            cycles += 1
            idx = torch.nonzero(state.running).squeeze(1)
        return ended

    def _iterate(self, s: SCFRun, idx) -> torch.Tensor:
        """One cycle of the running molecules (indices `idx`); returns those whose SCF ended."""
        on, this = s.running, s.precise.clone()
        s.iters = s.iters + on
        pot = self._potentials(*self.vars.unpack(s.mixer.q))
        _keep_input(s.vin, pot, on[self.atom_mol[self.sh_atom]], on[self.atom_mol])
        failed = self._cycle(s.st, pot, on, this, idx)
        qo = mulliken(self.g, self.lay, s.st["p"], self.n0sh, self.db.hb.nat, idx)
        s.qout = torch.where(on[:, None], self.vars.pack(*qo), s.qout)
        dq = s.qout - s.mixer.q
        enew = self._energy(s.st["p"], s.st["ts"], self._potentials(*qo, slot="out"))
        s.energy = torch.where(on, enew, s.energy)
        err = torch.sqrt((dq * dq).sum(1) / self.vars.nvar)
        bad = on & (failed | ~torch.isfinite(s.energy) | ~torch.isfinite(err))
        done = on & this & ((s.energy - s.elast).abs() < self.econv) & (err < self.pconv) & ~bad
        s.elast = torch.where(on, torch.where(this, s.energy, torch.inf), s.elast)
        unconv = bad | (on & ~done & (s.iters >= self.max_iter))
        s.status = torch.where(bad, SCF_NONFINITE, torch.where(done, SCF_OK, s.status))
        retry = unconv & s.may_fall_back
        final = done | (unconv & ~retry)
        s.running = on & ~final & ~retry
        s.precise = s.precise | (s.running & (err < self.SWITCH))
        counts = torch.stack([final.sum(), retry.sum()])  # read in the mixer's solve's sync
        s.mixer.next(torch.where(s.running[:, None], dq, 0.0), s.running, final | retry, counts)
        nfinal, nretry = s.mixer.read if s.mixer.read is not None else counts.tolist()
        if nfinal:
            self._wdensity(s.st, torch.nonzero(final).squeeze(1))
        if nretry:  # the fallback: the same evaluation again from the SAD guess
            self._reset(s, retry, self.vars.pack(*self.guess()))
            s.status = torch.where(retry, SCF_MAXITER, s.status)
            s.running, s.fell = s.running | retry, s.fell | retry
            s.may_fall_back = s.may_fall_back & ~retry
        return final

    def result(self, state: SCFRun) -> SCFResult:
        """Every molecule's SCF result as it stands (a running one's last cycle), as copies:
        later `start` and `advance` calls do not change it."""
        qsh, dp, qp = self.vars.unpack(state.qout.clone())
        st = {k: state.st[k].clone() for k in ("p", "x", "emo", "occ")}
        return SCFResult(state.energy.clone(), state.status == SCF_OK, state.iters.clone(),
                         st["p"], st["x"], st["emo"], st["occ"], qsh, dp, qp,
                         state.status.clone(), dict(state.vin))  # fmt: skip

    def run(self, start=None, active=None) -> SCFResult:
        """The SCF from `start` (qsh, dp, qp), default the SAD guess, for the molecules
        `active` (default all; bad geometries never run). A molecule that fails stops alone:
        its status says why (SCF_STATUS)."""
        state = self.new_run(start)
        self.start(state, self._which(active), start)
        self.advance(state)
        return self.result(state)

    def run_with_fallback(self, start, active=None) -> tuple[SCFResult, torch.Tensor]:
        """`run` from `start`; the molecules it leaves unconverged run again from the SAD guess
        (as `TbliteEnergy` restarts tblite). Returns the result and who fell back."""
        state = self.new_run(start)
        self.start(state, self._which(active), start, fallback=True)
        self.advance(state)
        return self.result(state), state.fell.clone()

    def _which(self, active) -> torch.Tensor:
        if active is None:
            return torch.ones(self.vars.b, dtype=torch.bool, device="cuda")
        return active.clone()

    def merge(self, a: SCFResult, b: SCFResult, take_b: torch.Tensor) -> SCFResult:
        """`a` with the molecules `take_b` from `b`."""
        sh_mol = self.atom_mol[self.sh_atom]

        def pick(x, y, mask):
            return torch.where(mask.view(-1, *([1] * (x.dim() - 1))), y, x)

        mol = {k: pick(getattr(a, k), getattr(b, k), take_b) for k in
               ("energy", "converged", "iterations", "density", "wdensity", "emo", "occ",
                "status")}  # fmt: skip
        sh, at = take_b[sh_mol], take_b[self.atom_mol]
        vin = {k: pick(a.vin[k], b.vin[k], sh if k == "vsh" else at) for k in a.vin}
        return SCFResult(**mol, qsh=pick(a.qsh, b.qsh, sh), dp=pick(a.dp, b.dp, at),
                         qp=pick(a.qp, b.qp, at), vin=vin)  # fmt: skip

    def _zero_potentials(self) -> dict:
        nat, nsh = self.db.hb.nat, len(self.sh_atom)
        z = _f64()
        return {"vsh": torch.zeros(nsh, **z), "vat": torch.zeros(nat, **z),
                "vdp": torch.zeros(nat, 3, **z), "vqp": torch.zeros(nat, 6, **z)}  # fmt: skip

    def gradient(
        self, r: SCFResult, potentials: str = "input"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy and gradient (nat, 3) at the SCF result: the stage-2 terms at the output
        variables, stage 1's H0 gradient, the CN chain of its dE/dCN. The H0 gradient is
        contracted with the potentials of the output variables ("output", consistent with the
        density) or, as tblite does, of the last cycle's input ("input": right after `run`)."""
        qat = self.qat(r.qsh)
        out_pot = potentials != "input"
        pot = self._potentials(r.qsh, r.dp, r.qp, slot="out") if out_pot else r.vin
        p, x = (torch.zeros(self.db.hb.mat_total, **_f64()) for _ in range(2))
        unpad_into(p, r.density, self.lay)
        unpad_into(x, r.wdensity, self.lay)
        vao = K._shell_to_ao(self.db, pot["vsh"] + pot["vat"][self.sh_atom])
        self.db.h0_gradient(p, x, vao, pot["vdp"].reshape(-1).contiguous(),
                            pot["vqp"].reshape(-1).contiguous())  # fmt: skip
        out = self.cb.evaluate(r.qsh, qat, r.dp, r.qp, self.db.dedcn)
        return r.energy, out["grad"] + self.db.grad.view(-1, 3)


def _f64() -> dict:
    return {"dtype": F64, "device": "cuda"}


def _keep_input(vin: dict, pot: dict, sh_active, at_active) -> None:
    """Copy the active molecules' input potentials of this cycle into `vin`."""
    vin["vsh"] = torch.where(sh_active, pot["vsh"], vin["vsh"])
    vin["vat"] = torch.where(at_active, pot["vat"], vin["vat"])
    vin["vdp"] = torch.where(at_active[:, None], pot["vdp"], vin["vdp"])
    vin["vqp"] = torch.where(at_active[:, None], pot["vqp"], vin["vqp"])
