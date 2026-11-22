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
residual's RMS falls below `SCFBatch.SWITCH`, then in fp64, and convergence is only declared
after two fp64 cycles, so the energy test compares fp64 energies. An fp32 eigenvector
refinement in fp64 (Ogita-Aishima) was tried and dropped: on this GPU's fp64 rate its GEMMs
cost more than an fp64 eigensolve (docs/decisions.md). The Fock build, the Mulliken sums and
the Fermi filling are fused kernels (native/ksxtb/src/scf.cu).
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


def padded_geometry(db: native.DeviceBatch, lay: Layout) -> Geometry:
    eye = torch.diag_embed((~lay.valid).to(F64))
    s = pad(db.S, lay)[:, 0] + eye
    chol = torch.linalg.cholesky(s)
    eye_n = torch.eye(lay.nmax, dtype=F64, device=s.device).expand_as(s)
    linv = torch.linalg.solve_triangular(chol, eye_n, upper=False)
    h0, d, q = pad(db.H0, lay)[:, 0], pad(db.D, lay, 3), pad(db.Q, lay, 6)
    return Geometry(s, h0, d, q, linv, linv.float())


class KsFock(C.Structure):
    _fields_ = [(n, C.c_void_p) for n in ("H0", "S", "D", "Q", "vao", "vmp", "valid", "idx",
                                          "fock")] + [("pad", C.c_double), ("n", C.c_int),
                                                      ("nsel", C.c_int)]  # fmt: skip


class KsMulliken(C.Structure):
    _fields_ = [(n, C.c_void_p) for n in ("P", "S", "D", "Q", "out")] + [
        ("n", C.c_int), ("nmol", C.c_int)]  # fmt: skip


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


def solve(g: Geometry, f: torch.Tensor, idx=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Orbital energies (B, n) and coefficients (B, n, n), columns the orbitals: F C = S C e,
    in fp64 throughout (the exact reference of the faster solvers)."""
    linv = g.Linv if idx is None else g.Linv[idx]
    fp = linv @ f @ linv.transpose(1, 2)
    e, c = torch.linalg.eigh(0.5 * (fp + fp.transpose(1, 2)))
    return e, linv.transpose(1, 2) @ c


def solve_fp32(g: Geometry, f: torch.Tensor, idx) -> tuple[torch.Tensor, torch.Tensor]:
    """The same in fp32 (transform, eigensolve, back-transform): the early SCF cycles. The
    orbital energies come back in fp64, the coefficients in fp32."""
    linv = g.Linv32[idx]
    fp = linv @ f.float() @ linv.transpose(1, 2)
    e, c = torch.linalg.eigh(0.5 * (fp + fp.transpose(1, 2)))
    return e.double(), linv.transpose(1, 2) @ c


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


def mulliken(g: Geometry, lay: Layout, p: torch.Tensor, n0sh: torch.Tensor, nat: int):
    """Output shell charges (nsh,), atomic dipoles (nat, 3) and quadrupoles (nat, 6): the per-AO
    sums of scf.cu's kernel, reduced to shells and atoms (`mulliken_reference` in torch)."""
    b, n = p.shape[0], lay.nmax
    per = torch.empty(b, n, 10, dtype=F64, device=p.device)
    pc = p.contiguous()
    args = KsMulliken(*[t.data_ptr() for t in (pc, g.S, g.D, g.Q, per)], n, b)
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
    molecule of a batch at once; molecules outside `active` keep their input unchanged."""

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
    emo: torch.Tensor  # (B, n)
    occ: torch.Tensor  # (B, n) both spins
    qsh: torch.Tensor
    dp: torch.Tensor
    qp: torch.Tensor


class SCFBatch:
    """A batch of molecules (z, xyz Bohr, charge, unpaired electrons) and its GFN2 SCF."""

    def __init__(self, mols, solvent: bool, accuracy: float = 1.0, max_iter: int = 250) -> None:
        self.geo_mols = [(np.asarray(z, dtype=int), np.asarray(x, float)) for z, x, _, _ in mols]
        self.charge = torch.tensor([float(c) for *_, c, _ in mols], dtype=F64, device="cuda")
        uhf = torch.tensor([float(u) for *_, u in mols], dtype=F64, device="cuda")
        self.solvent, self.max_iter = solvent, max_iter
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

    def _cycle(self, st: dict, pot: dict, active, precise) -> None:
        """Fock, eigensolve and density of the active molecules, into the state buffers."""
        for phase in (False, True):
            idx = torch.nonzero(active & (precise == phase)).squeeze(1)
            if not len(idx):
                continue
            f = fock(self.g, self.lay, pot["vsh"], pot["vat"], pot["vdp"], pot["vqp"], idx)
            if phase:
                emo, c = solve(self.g, f, idx)
            else:
                emo, c = solve_fp32(self.g, f, idx)
            occ, ts = self._occupations(emo, idx)
            st["emo"][idx], st["occ"][idx], st["ts"][idx] = emo, occ, ts
            # the fp32 phase's density in fp32; C and F kept only for the precise phase's X
            st["p"][idx] = ((c * occ[:, None, :].to(c.dtype)) @ c.transpose(1, 2)).double()
            if phase:
                st["c"][idx] = c

    def _wdensity(self, st: dict, idx) -> None:
        """X = C diag(f e) C^T of the last (fp64) cycle, tblite's energy-weighted density."""
        c, w = st["c"][idx], (st["occ"] * st["emo"])[idx]
        st["x"][idx] = (c * w[:, None, :]) @ c.transpose(1, 2)

    SWITCH = 1e-4  # residual RMS below which a molecule's cycles turn fp64-accurate

    def run(self, start=None) -> SCFResult:
        """The SCF from `start` (qsh, dp, qp), default the SAD guess."""
        qsh, dp, qp = start if start is not None else self.guess()
        mixer = Broyden(self.vars.pack(qsh, dp, qp))
        b, n = self.vars.b, self.lay.nmax
        st = {k: torch.zeros(b, n, n, **_f64()) for k in ("c", "p", "x")}
        st.update(emo=torch.zeros(b, n, **_f64()), occ=torch.zeros(b, n, **_f64()),
                  ts=torch.zeros(b, **_f64()))  # fmt: skip
        flags = {k: torch.zeros(b, dtype=torch.bool, device="cuda") for k in ("conv", "precise")}
        active = torch.ones(b, dtype=torch.bool, device="cuda")
        elast = torch.full((b,), torch.inf, **_f64())
        iters = torch.zeros(b, dtype=torch.long, device="cuda")
        energy = qout = None
        for it in range(1, self.max_iter + 1):
            precise = flags["precise"].clone()
            self._cycle(st, self._potentials(*self.vars.unpack(mixer.q)), active, precise)
            qo = mulliken(self.g, self.lay, st["p"], self.n0sh, self.db.hb.nat)
            qnew = self.vars.pack(*qo)
            qout = qnew if qout is None else torch.where(active[:, None], qnew, qout)
            dq = qout - mixer.q
            enew = self._energy(st["p"], st["ts"], self._potentials(*qo, slot="out"))
            energy = enew if energy is None else torch.where(active, enew, energy)
            err = torch.sqrt((dq * dq).sum(1) / self.vars.nvar)
            done = precise & ((energy - elast).abs() < self.econv) & (err < self.pconv)
            elast = torch.where(precise, energy, torch.inf)
            iters = torch.where(active, it, iters)
            finished = active & (done | ~torch.isfinite(energy))
            flags["conv"] |= active & done & torch.isfinite(energy)
            if bool(finished.any()):
                self._wdensity(st, torch.nonzero(finished).squeeze(1))
            active = active & ~finished
            flags["precise"] |= active & (err < self.SWITCH)
            if not bool(active.any()):
                break
            mixer.next(torch.where(active[:, None], dq, 0.0), active)
        if bool(active.any()):
            self._wdensity(st, torch.nonzero(active).squeeze(1))
        qsh, dp, qp = self.vars.unpack(qout)
        return SCFResult(energy, flags["conv"], iters, st["p"], st["x"], st["emo"], st["occ"],
                         qsh, dp, qp)  # fmt: skip

    def gradient(
        self, r: SCFResult, potentials: str = "input"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy and gradient (nat, 3) at the SCF result: the stage-2 terms at the output
        variables, stage 1's H0 gradient, the CN chain of its dE/dCN. The H0 gradient is
        contracted with the potentials of the output variables ("output", consistent with the
        density) or, as tblite does, of the last cycle's input ("input": right after `run`)."""
        qat = self.qat(r.qsh)
        if potentials == "input":
            raw = self._slots["in"]  # untouched since the last cycle; frozen molecules' too
            pot = {"vsh": raw["vsh"], "vat": raw["vat"], "vdp": raw["vdp"].view(-1, 3),
                   "vqp": raw["vqp"].view(-1, 6)}  # fmt: skip
        else:
            pot = self._potentials(r.qsh, r.dp, r.qp, slot="out")
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
