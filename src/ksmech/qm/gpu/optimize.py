"""Stage 4: the batched ANC / RFO geometry optimizer on the GPU, `qm.anc` for many molecules.

Every molecule runs `qm.anc.optimize_anc`'s algorithm with its own state: the Lindh model
Hessian once at the start, the approximate normal coordinates (ANC) rebuilt every `micro` steps
from the learned Hessian, RFO steps cut to a trust radius that grows and shrinks as there,
uphill steps taken back, BFGS updates in the ANC basis, convergence on the largest gradient
component (walls included). The energies and gradients of all still-running molecules come
from one batched GFN2 evaluation (`SCFBatch`), each SCF restarted from the molecule's last
evaluated point and, if that fails, from the guess (as `geometry.TbliteEnergy`). The bond
walls and the dissociation guard of `geometry` apply per molecule: the guard sees every trial
geometry, the first included, before it is evaluated.

Coordinates are padded per molecule to the batch's largest 3N; padded ANC modes are inert
(curvature 1, zero gradient), so a molecule's steps do not depend on the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ksmech.qm.anc import H_LOW, AncSettings
from ksmech.qm.geometry import BOHR_ANG
from ksmech.qm.gpu import scf as S
from ksmech.qm.gpu.lindh import lindh_hessian_fast
from ksmech.qm.records import ReasonCode

F64 = torch.float64
PAD_CURVATURE = 1.0  # Hartree / Bohr^2 on padded ANC modes: inert, never the RFO's lowest


@dataclass
class OptSpec:
    """One molecule to optimize: atoms, start (Bohr), charge, unpaired electrons, and the
    optional walls (pairs (k, 2), onsets d0 (k,) Bohr, force constant) and guard (pairs (k, 2),
    covalent-radius sums (k,) Angstrom, factor, patience)."""

    z: np.ndarray
    start: np.ndarray
    charge: int
    unpaired: int
    walls: tuple[np.ndarray, np.ndarray, float] | None = None
    guard: tuple[np.ndarray, np.ndarray, float, int] | None = None


@dataclass
class OptOutcome:
    xyz: np.ndarray  # Bohr, the last accepted point
    converged: bool
    reason: ReasonCode | None
    detail: str
    n_evals: int
    gmax: float
    energy: float
    fresh_starts: int = 0
    trace: list = field(default_factory=list)  # (energy, radius) per evaluation, when traced


# --- batched linear algebra of `qm.anc` ------------------------------------------------------


def anc_basis(h_cart: torch.Tensor, xyz: torch.Tensor, real: torch.Tensor):
    """`qm.anc.anc_basis` for a batch: h_cart (b, P, P), xyz (b, N, 3) padded, real (b, N)
    bool. Returns vectors (b, P, M) and curvatures (b, M), M = max over the batch of 3n - 6
    (3n - 5 if linear); columns past a molecule's own count are zero, curvature PAD_CURVATURE."""
    b, n = real.shape
    pdim = 3 * n
    w = real.to(F64)
    centre = (xyz * w[..., None]).sum(1) / w.sum(1)[:, None]
    c = (xyz - centre[:, None]) * w[..., None]
    eye3 = torch.eye(3, dtype=F64, device=xyz.device)
    trans = [(eye3[k][None, None, :] * w[..., None]).reshape(b, pdim) for k in range(3)]
    rots = [torch.cross(eye3[k].expand_as(c), c, dim=-1).reshape(b, pdim) for k in range(3)]
    u, sv, _ = torch.linalg.svd(torch.stack(trans + rots, 2), full_matrices=False)
    keep = sv > 1e-6 * sv.amax(1, keepdim=True)
    t = u * keep[:, None, :]
    tt = t @ t.transpose(1, 2)
    real3 = real.repeat_interleave(3, 1).to(F64)
    proj = torch.diag_embed(real3) - tt
    shift = 1e3 + h_cart.abs().sum(2).amax(1)
    a = proj @ h_cart @ proj + shift[:, None, None] * tt
    a = a + torch.diag_embed((1.0 - real3) * 10.0 * shift[:, None])
    lam, vec = torch.linalg.eigh(0.5 * (a + a.transpose(1, 2)))
    nmode = (3 * real.sum(1) - keep.sum(1)).clamp(min=0)
    mmax = int(nmode.max()) if b else 0
    live = torch.arange(mmax, device=xyz.device)[None, :] < nmode[:, None]
    vecs = vec[:, :, :mmax] * live[:, None, :]
    lam = torch.where(live, lam[:, :mmax].clamp(min=H_LOW), PAD_CURVATURE)
    return vecs, lam, live


def rfo_step(h: torch.Tensor, g: torch.Tensor, dtype=F64) -> torch.Tensor:
    """`qm.anc.rfo_step` for a batch: the lowest eigenvector of [[H, g], [g^T, 0]], scaled;
    steepest descent -g / max(diag H, H_LOW) where it has no usable last component."""
    b, m = g.shape
    aug = torch.zeros(b, m + 1, m + 1, dtype=F64, device=g.device)
    aug[:, :m, :m], aug[:, :m, m], aug[:, m, :m] = h, g, g
    _, v = torch.linalg.eigh(aug.to(dtype))
    v = v[:, :, 0].double()
    last = v[:, m]
    sd = -g / torch.diagonal(h, dim1=1, dim2=2).clamp(min=H_LOW)
    ok = last.abs() >= 1e-8
    return torch.where(ok[:, None], v[:, :m] / torch.where(ok, last, 1.0)[:, None], sd)


def bfgs(h: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """`qm.anc._bfgs` for a batch: skipped where the curvature condition fails."""
    sy = (s * y).sum(1)
    hs = (h @ s[:, :, None])[:, :, 0]
    shs = (s * hs).sum(1)
    ok = (sy > 1e-10) & (shs > 1e-12)
    upd = (
        y[:, :, None] * y[:, None, :] / torch.where(ok, sy, 1.0)[:, None, None]
        - hs[:, :, None] * hs[:, None, :] / torch.where(ok, shs, 1.0)[:, None, None]
    )
    return torch.where(ok[:, None, None], h + upd, h)


# --- the optimizer -----------------------------------------------------------------------------

FAIL_NONE, FAIL_GUARD, FAIL_SCF = 0, 1, 2


def _pair_tensors(specs, offsets, which: str):
    """Concatenated walls or guard pairs (global atom indices) with their molecule."""
    rows = [(m, getattr(sp, which)) for m, sp in enumerate(specs) if getattr(sp, which) is not None]
    rows = [(m, w) for m, w in rows if len(w[0])]
    if not rows:
        return None
    pairs = np.concatenate([w[0] + offsets[m] for m, w in rows])
    mol = np.concatenate([np.full(len(w[0]), m) for m, w in rows])
    out = {
        "i": pairs[:, 0],
        "j": pairs[:, 1],
        "mol": mol,
        "ref": np.concatenate([w[1] for _, w in rows]),
    }
    out["k"] = np.concatenate([np.full(len(w[0]), float(w[2])) for _, w in rows])
    if which == "guard":
        out["patience"] = np.array([sp.guard[3] if sp.guard else 0 for sp in specs])
    return {k: torch.as_tensor(v, device="cuda") for k, v in out.items()}


class BatchedANC:
    """`qm.anc.optimize_anc` (with `geometry.TbliteEnergy`'s restarts, walls and guard) for a
    batch of molecules on the GPU. `rfo_dtype` float32 runs the RFO eigensolve in fp32."""

    def __init__(
        self,
        specs: list[OptSpec],
        solvent: bool = True,
        cfg: AncSettings | None = None,
        accuracy: float = 1.0,
        rfo_dtype=F64,
        trace: bool = False,
    ) -> None:
        self.specs, self.cfg, self.rfo_dtype, self.tracing = (
            specs,
            cfg or AncSettings(),
            rfo_dtype,
            trace,
        )
        mols = [(sp.z, sp.start, sp.charge, sp.unpaired) for sp in specs]
        self.sb = S.SCFBatch(mols, solvent, accuracy=accuracy)
        nat = np.array([len(sp.z) for sp in specs])
        self.offsets = np.concatenate([[0], np.cumsum(nat)[:-1]])
        self.b, self.nmax = len(specs), int(nat.max())
        idx = np.full((self.b, self.nmax), -1)
        for m, (o, n) in enumerate(zip(self.offsets, nat, strict=True)):
            idx[m, :n] = np.arange(o, o + n)
        self.idx = torch.as_tensor(idx, device="cuda")
        self.real = self.idx >= 0
        self.nat = torch.as_tensor(nat, device="cuda")
        self.atom_mol = self.sb.atom_mol
        self.walls = _pair_tensors(specs, self.offsets, "walls")
        self.guard = _pair_tensors(specs, self.offsets, "guard")
        self.streak = torch.zeros(self.b, dtype=torch.long, device="cuda")

    def _pad(self, flat: torch.Tensor) -> torch.Tensor:
        ext = torch.cat([flat, flat.new_zeros(1, 3)])
        return ext[torch.where(self.real, self.idx, len(flat))]  # (b, N, 3)

    def _unpad_into(
        self, flat: torch.Tensor, padded: torch.Tensor, mols: torch.Tensor
    ) -> torch.Tensor:
        sel = self.real & mols[:, None]
        out = flat.clone()
        out[self.idx[sel]] = padded[sel]
        return out

    def _wall_energy(self, x: torch.Tensor):
        e = torch.zeros(self.b, dtype=F64, device=x.device)
        g = torch.zeros_like(x)
        w = self.walls
        if w is None:
            return e, g
        vec = x[w["i"]] - x[w["j"]]
        d = torch.linalg.norm(vec, dim=1)
        over = (d - w["ref"]).clamp(min=0.0)
        e = e.index_add(0, w["mol"], w["k"] * over * over)
        pull = (2.0 * w["k"] * over / d.clamp(min=1e-12))[:, None] * vec
        return e, g.index_add(0, w["i"], pull).index_add(0, w["j"], -pull)

    def _guard(self, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """DissociationGuard on the trial geometries of the active molecules: True = stop."""
        gd = self.guard
        if gd is None:
            return torch.zeros_like(active)
        d = torch.linalg.norm(x[gd["i"]] - x[gd["j"]], dim=1) * BOHR_ANG / gd["ref"]
        worst = torch.full((self.b,), -1.0, dtype=F64, device=x.device)
        worst = worst.scatter_reduce(0, gd["mol"], d, "amax")
        over = worst > gd["k"].new_zeros(self.b).index_reduce(
            0, gd["mol"], gd["k"], "amax", include_self=False
        )
        self.streak = torch.where(active, torch.where(over, self.streak + 1, 0), self.streak)
        self.last_ratio = worst
        return active & (gd["patience"] > 0) & (self.streak >= gd["patience"])

    def _evaluate(self, x: torch.Tensor, active: torch.Tensor):
        """Energies (b,), gradients (nat, 3) and failure codes (b,) at x for the active
        molecules: the guard first, then the SCF from each molecule's last evaluated point
        (from the guess if that fails), the gradient, the walls."""
        fail = torch.where(self._guard(x, active), FAIL_GUARD, FAIL_NONE)
        go = active & (fail == FAIL_NONE)
        self.sb.set_geometry(x)
        r, fell = self.sb.run_with_fallback(self.start, go)
        self.fresh_starts += fell.long()
        e, g = self.sb.gradient(r)
        ew, gw = self._wall_energy(x)
        fail = torch.where(go & ~r.converged, FAIL_SCF, fail)
        keep = go & r.converged
        if self.start is None:
            self.start = (r.qsh.clone(), r.dp.clone(), r.qp.clone())
        else:  # each molecule's next SCF starts from its last evaluated point
            sh, at = keep[self.atom_mol[self.sb.sh_atom]], keep[self.atom_mol]
            self.start = (
                torch.where(sh, r.qsh, self.start[0]),
                torch.where(at[:, None], r.dp, self.start[1]),
                torch.where(at[:, None], r.qp, self.start[2]),
            )
        return e + ew, g + gw, fail

    def _rebuild(self, st: dict, which: torch.Tensor) -> None:
        """New ANC bases of the molecules `which` from their Cartesian Hessians (the Lindh model
        at the start, the learned one after): `qm.anc.optimize_anc`'s outer loop."""
        idx = torch.nonzero(which).squeeze(1)
        if not len(idx):
            return
        vecs, lam, _ = anc_basis(st["hcart"][idx], self._pad(st["x"])[idx], self.real[idx])
        mm, m = st["vecs"].shape[2], vecs.shape[2]
        st["vecs"][idx] = torch.nn.functional.pad(vecs, (0, mm - m))
        st["h"][idx] = torch.diag_embed(
            torch.nn.functional.pad(lam, (0, mm - m), value=PAD_CURVATURE)
        )
        st["micro"][idx] = 0

    def _step(self, st: dict, active: torch.Tensor) -> torch.Tensor:
        """One RFO step of every active molecule (`qm.anc._State.step`); returns failures."""
        cfg, vecs, h = self.cfg, st["vecs"], st["h"]
        gq = (vecs.transpose(1, 2) @ self._pad(st["g"]).reshape(self.b, -1, 1))[:, :, 0]
        s = rfo_step(h, gq, self.rfo_dtype)
        ns = torch.linalg.norm(s, dim=1)
        cut = torch.where(ns > st["radius"], st["radius"] / ns.clamp(min=1e-300), 1.0)
        s = s * cut[:, None]
        ns = torch.linalg.norm(s, dim=1)
        predicted = (gq * s).sum(1) + 0.5 * (s * (h @ s[:, :, None])[:, :, 0]).sum(1)
        trial_pad = self._pad(st["x"]) + (vecs @ s[:, :, None]).reshape(self.b, self.nmax, 3)
        trial = self._unpad_into(st["x"], trial_pad, active)
        e_new, g_new, fail = self._evaluate(trial, active)
        st["n_evals"] += active.long()
        st["micro"] += active.long()
        de = e_new - st["e"]
        ratio = torch.where(predicted < 0, de / torch.where(predicted < 0, predicted, -1.0), -1.0)
        ok = active & (fail == FAIL_NONE)
        reject = ok & (de > cfg.uphill) & (ratio < 0)
        acc = ok & ~reject
        y = (vecs.transpose(1, 2) @ self._pad(g_new - st["g"]).reshape(self.b, -1, 1))[:, :, 0]
        st["h"] = torch.where(acc[:, None, None], bfgs(h, s, y), h)
        r = st["radius"]
        grow = acc & (ratio > 0.75) & (ns > 0.8 * r)
        shrink = reject | (acc & ~grow & (ratio < 0.25))
        r = torch.where(grow, (2 * r).clamp(max=cfg.radius_max), r)
        st["radius"] = torch.where(shrink, (r / 4).clamp(min=cfg.radius_min), r)
        am = acc[self.atom_mol]
        st["x"] = torch.where(am[:, None], trial, st["x"])
        st["g"] = torch.where(am[:, None], g_new, st["g"])
        st["e"] = torch.where(acc, e_new, st["e"])
        if self.tracing:
            self.trace.append((e_new.cpu().numpy(), st["radius"].cpu().numpy(), acc.cpu().numpy()))
        return fail

    def _initial(self) -> tuple[dict, torch.Tensor]:
        """The start of every molecule: one evaluation, the Lindh model Hessian."""
        b, pdim = self.b, 3 * self.nmax
        x = torch.as_tensor(
            np.concatenate([sp.start for sp in self.specs]), dtype=F64, device="cuda"
        )
        self.start, self.trace = None, []
        self.fresh_starts = torch.zeros(b, dtype=torch.long, device="cuda")
        active = self.nat > 1  # a single atom returns at once (optimize_tblite_anc)
        st = {
            "x": x,
            "radius": torch.full((b,), self.cfg.radius, dtype=F64, device="cuda"),
            "n_evals": torch.zeros(b, dtype=torch.long, device="cuda"),
            "micro": torch.zeros(b, dtype=torch.long, device="cuda"),
        }
        st["e"], st["g"], fail = self._evaluate(x, active)
        st["n_evals"] += active.long()
        hcart = np.zeros((b, pdim, pdim))
        for m, sp in enumerate(self.specs):
            k = 3 * len(sp.z)
            if len(sp.z) > 1:
                hcart[m, :k, :k] = lindh_hessian_fast(np.asarray(sp.z), np.asarray(sp.start))
        st["hcart"] = torch.as_tensor(hcart, device="cuda")
        mm = max(pdim - 5, 1)
        st["vecs"] = torch.zeros(b, pdim, mm, dtype=F64, device="cuda")
        st["h"] = PAD_CURVATURE * torch.eye(mm, dtype=F64, device="cuda").repeat(b, 1, 1)
        return st, fail

    def run(self) -> list[OptOutcome]:
        """Optimize every molecule; one batched evaluation per step for all still running."""
        cfg = self.cfg
        st, fail = self._initial()
        reason = torch.where(fail != FAIL_NONE, fail, FAIL_NONE)
        active = (self.nat > 1) & (fail == FAIL_NONE)

        def finished(active):
            gmax = self._gmax(st["g"])
            return active & ((gmax <= cfg.fmax) | (st["n_evals"] >= cfg.maxiter))

        active = active & ~finished(active)
        self._rebuild(st, active)
        while bool(active.any()):
            fail = self._step(st, active)
            reason = torch.where(active & (fail != FAIL_NONE), fail, reason)
            active = active & (fail == FAIL_NONE)
            active = active & ~finished(active)
            due = active & (st["micro"] >= cfg.micro)
            if bool(due.any()):
                v, h = st["vecs"], st["h"]
                st["hcart"] = torch.where(
                    due[:, None, None], v @ h @ v.transpose(1, 2), st["hcart"]
                )
                self._rebuild(st, due)
        return self._outcomes(st, reason)

    def _gmax(self, g: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.b, dtype=F64, device=g.device).scatter_reduce(
            0, self.atom_mol, g.abs().amax(1), "amax"
        )

    def _outcomes(self, st: dict, reason: torch.Tensor) -> list[OptOutcome]:
        gmax = self._gmax(st["g"]).cpu().numpy()
        x, e = st["x"].cpu().numpy(), st["e"].cpu().numpy()
        n_evals, reason = st["n_evals"].cpu().numpy(), reason.cpu().numpy()
        fresh = self.fresh_starts.cpu().numpy()
        out = []
        for m, sp in enumerate(self.specs):
            xyz = x[self.offsets[m] : self.offsets[m] + len(sp.z)]
            conv = bool(reason[m] == FAIL_NONE and (len(sp.z) == 1 or gmax[m] <= self.cfg.fmax))
            code, detail = self._reason(m, reason[m], conv, gmax[m], n_evals[m])
            out.append(
                OptOutcome(
                    xyz,
                    conv,
                    code,
                    detail,
                    int(n_evals[m]),
                    float(gmax[m]),
                    float(e[m]),
                    int(fresh[m]),
                )
            )
        return out

    def _reason(self, m, code, conv, gmax, n_evals):
        if conv:
            return None, ""
        if code == FAIL_GUARD:
            pat = self.specs[m].guard
            ratio = float(self.last_ratio[m])
            return ReasonCode.CONNECTIVITY_CHANGED, (
                f"a bond at {ratio:.2f}x, beyond {pat[2]}x for {pat[3]} steps of the optimization"
            )
        if code == FAIL_SCF:
            return ReasonCode.SCF_NOT_CONVERGED, "during optimization (GPU SCF, from the guess too)"
        return ReasonCode.OPT_NOT_CONVERGED, f"max|grad|={gmax:.2e} after {n_evals} energies (ANC)"
