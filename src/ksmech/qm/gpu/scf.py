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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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


def padded_geometry(db: native.DeviceBatch, lay: Layout) -> Geometry:
    eye = torch.diag_embed((~lay.valid).to(F64))
    s = pad(db.S, lay)[:, 0] + eye
    chol = torch.linalg.cholesky(s)
    eye_n = torch.eye(lay.nmax, dtype=F64, device=s.device).expand_as(s)
    linv = torch.linalg.solve_triangular(chol, eye_n, upper=False)
    return Geometry(s, pad(db.H0, lay)[:, 0], pad(db.D, lay, 3), pad(db.Q, lay, 6), linv)


def fock(g: Geometry, lay: Layout, vsh, vat, vdp, vqp) -> torch.Tensor:
    """tblite's H1 = H0 + add_vao_to_h1 + add_vmp_to_h1 (dipole, quadrupole), padded."""
    vsh_ext = torch.cat([vsh + vat[lay.sh_atom], vsh.new_zeros(1)])
    vao = vsh_ext[lay.ao_shell]  # (B, n)
    f = g.H0 - 0.5 * g.S * (vao[:, :, None] + vao[:, None, :])
    vdp_ext = torch.cat([vdp, vdp.new_zeros(1, 3)])[lay.ao_atom]  # (B, n, 3) at atom(nu)
    vqp_ext = torch.cat([vqp, vqp.new_zeros(1, 6)])[lay.ao_atom]
    w = torch.einsum("bcmn,bnc->bmn", g.D, vdp_ext) + torch.einsum("bcmn,bnc->bmn", g.Q, vqp_ext)
    f = f - 0.5 * (w + w.transpose(1, 2))
    return f + torch.diag_embed((~lay.valid).to(F64) * PAD_LEVEL)


def solve(g: Geometry, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Orbital energies (B, n) and coefficients (B, n, n), columns the orbitals: F C = S C e."""
    fp = g.Linv @ f @ g.Linv.transpose(1, 2)
    e, c = torch.linalg.eigh(0.5 * (fp + fp.transpose(1, 2)))
    return e, g.Linv.transpose(1, 2) @ c


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
    for _ in range(200):
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
        if bool(done.all()):
            break
    return occ


def entropy(occ: torch.Tensor, kt: float = T.KT_300K) -> torch.Tensor:
    """tblite's -TS of one spin channel per molecule (B,)."""
    s = occ * torch.log(occ.clamp(min=1e-300)) + (1 - occ) * torch.log((1 - occ).clamp(min=1e-300))
    return kt * torch.where((occ > 0) & (occ < 1), s, 0.0).sum(1)


def mulliken(g: Geometry, lay: Layout, p: torch.Tensor, n0sh: torch.Tensor, nat: int):
    """Output shell charges (nsh,), atomic dipoles (nat, 3) and quadrupoles (nat, 6)."""
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

    def _potentials(self, qsh, dp, qp) -> dict:
        out = self.cb.evaluate(qsh, self.qat(qsh), dp, qp)
        return {k: v.clone() for k, v in out.items()}

    def _diagonalize(self, pot):
        f = fock(self.g, self.lay, pot["vsh"], pot["vat"], pot["vdp"], pot["vqp"])
        emo, c = solve(self.g, f)
        occs = [fermi(emo, n) for n in self.nel_spin]
        occ = occs[0] + occs[1]
        ts = entropy(occs[0]) + entropy(occs[1])
        p = (c * occ[:, None, :]) @ c.transpose(1, 2)
        return emo, c, occ, ts, p

    def _energy(self, p, ts, out_pot) -> torch.Tensor:
        band = (p * self.g.H0).sum((1, 2))
        shift = T.solvent_shift() if self.solvent else 0.0
        return band + ts + out_pot["energy"].sum(1) + shift

    def run(self, start=None) -> SCFResult:
        """The SCF from `start` (qsh, dp, qp), default the SAD guess."""
        qsh, dp, qp = start if start is not None else self.guess()
        mixer = Broyden(self.vars.pack(qsh, dp, qp))
        b = self.vars.b
        active = torch.ones(b, dtype=torch.bool, device="cuda")
        elast = torch.zeros(b, **_f64())
        res = None
        iters = torch.zeros(b, dtype=torch.long, device="cuda")
        conv = torch.zeros(b, dtype=torch.bool, device="cuda")
        for it in range(1, self.max_iter + 1):
            pot = self._potentials(*self.vars.unpack(mixer.q))
            emo, c, occ, ts, p = self._diagonalize(pot)
            qo = mulliken(self.g, self.lay, p, self.n0sh, self.db.hb.nat)
            qout = self.vars.pack(*qo)
            dq = qout - mixer.q
            energy = self._energy(p, ts, self._potentials(*qo))
            err = torch.sqrt((dq * dq).sum(1) / self.vars.nvar)
            done = ((energy - elast).abs() < self.econv) & (err < self.pconv)
            elast = energy
            cur = (energy, emo, c, occ, p, qout)
            res = cur if res is None else _merge(res, cur, active)
            iters = torch.where(active, it, iters)
            conv = conv | (active & done & torch.isfinite(energy))
            active = active & ~done & torch.isfinite(energy)
            if not bool(active.any()):
                break
            mixer.next(torch.where(active[:, None], dq, 0.0), active)
        energy, emo, c, occ, p, qout = res
        x = (c * (occ * emo)[:, None, :]) @ c.transpose(1, 2)
        qsh, dp, qp = self.vars.unpack(qout)
        return SCFResult(energy, conv, iters, p, x, emo, occ, qsh, dp, qp)

    def gradient(self, r: SCFResult) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy and gradient (nat, 3) at the SCF result: the stage-2 terms at the output
        variables, stage 1's H0 gradient with their potentials, the CN chain of its dE/dCN."""
        qat = self.qat(r.qsh)
        pot = self._potentials(r.qsh, r.dp, r.qp)
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


def _merge(old, new, active):
    """Keep converged molecules' last values, take the active ones' new values."""
    out = []
    for o, n in zip(old, new, strict=True):
        mask = active.view(-1, *([1] * (n.dim() - 1)))
        out.append(torch.where(mask, n, o))
    return tuple(out)
