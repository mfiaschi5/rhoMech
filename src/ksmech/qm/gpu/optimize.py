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
    state: MolState | None = None  # resume an optimization a previous batch left unfinished


@dataclass
class MolState:
    """Everything one molecule's optimization carries from one batch to the next."""

    x: np.ndarray  # (n, 3) the current point, Bohr
    e: float
    g: np.ndarray  # (n, 3)
    radius: float
    n_evals: int
    micro: int  # steps since the last basis
    hcart: np.ndarray  # (3n, 3n) the Hessian the next basis is built from
    vecs: np.ndarray  # (3n, 3n - 5) the ANC basis (zero columns past its modes)
    h: np.ndarray  # (3n - 5, 3n - 5) the Hessian in that basis
    qsh: np.ndarray  # the last evaluated point's SCF variables, the next SCF's start
    dp: np.ndarray
    qp: np.ndarray
    streak: int
    fresh_starts: int


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
        """Every molecule's start: resumed molecules from their state; new ones get their
        first evaluation (from the SAD guess) and the Lindh model Hessian."""
        b, pdim = self.b, 3 * self.nmax
        mm = max(pdim - 5, 1)
        dev = {"device": "cuda"}
        self.trace, self.fresh_starts = [], torch.zeros(b, dtype=torch.long, **dev)
        x = torch.as_tensor(
            np.concatenate([sp.state.x if sp.state else sp.start for sp in self.specs]),
            dtype=F64,
            **dev,
        )
        st = {
            "x": x,
            "radius": torch.full((b,), self.cfg.radius, dtype=F64, **dev),
            "n_evals": torch.zeros(b, dtype=torch.long, **dev),
            "micro": torch.zeros(b, dtype=torch.long, **dev),
            "e": torch.zeros(b, dtype=F64, **dev),
            "g": torch.zeros_like(x),
            "vecs": torch.zeros(b, pdim, mm, dtype=F64, **dev),
            "h": PAD_CURVATURE * torch.eye(mm, dtype=F64, **dev).repeat(b, 1, 1),
        }
        hcart = np.zeros((b, pdim, pdim))
        resumed = np.array([sp.state is not None for sp in self.specs])
        for m, sp in enumerate(self.specs):
            k = 3 * len(sp.z)
            if sp.state is None and len(sp.z) > 1:
                hcart[m, :k, :k] = lindh_hessian_fast(np.asarray(sp.z), np.asarray(sp.start))
            elif sp.state is not None:
                hcart[m, :k, :k] = sp.state.hcart
        st["hcart"] = torch.as_tensor(hcart, **dev)
        self.start = self.sb.guess()
        self._restore(st, resumed)
        new = torch.as_tensor(~resumed, **dev) & (self.nat > 1)
        e, g, fail = self._evaluate(st["x"], new)
        am = new[self.atom_mol]
        st["e"], st["g"] = torch.where(new, e, st["e"]), torch.where(am[:, None], g, st["g"])
        st["n_evals"] += new.long()
        return st, torch.where(new, fail, FAIL_NONE), new

    def _restore(self, st: dict, resumed: np.ndarray) -> None:
        """Write the resumed molecules' states into the batch's tensors."""
        sh_mol = self.atom_mol[self.sb.sh_atom].cpu().numpy()
        qsh, dp, qp = (t.clone() for t in self.start)
        for m in np.flatnonzero(resumed):
            sp, s_ = self.specs[m], self.specs[m].state
            o, n, k = self.offsets[m], len(sp.z), 3 * len(sp.z)
            st["x"][o : o + n] = torch.as_tensor(s_.x)
            st["g"][o : o + n] = torch.as_tensor(s_.g)
            st["e"][m], st["radius"][m] = s_.e, s_.radius
            st["n_evals"][m], st["micro"][m] = s_.n_evals, s_.micro
            w = s_.vecs.shape[1]
            st["vecs"][m, :k, :w] = torch.as_tensor(s_.vecs)
            st["h"][m, :w, :w] = torch.as_tensor(s_.h)
            shells = torch.as_tensor(np.flatnonzero(sh_mol == m), device="cuda")
            qsh[shells] = torch.as_tensor(s_.qsh, device="cuda")
            dp[o : o + n], qp[o : o + n] = torch.as_tensor(s_.dp), torch.as_tensor(s_.qp)
            self.streak[m], self.fresh_starts[m] = s_.streak, s_.fresh_starts
        self.start = (qsh, dp, qp)

    def _extract(self, st: dict, m: int) -> MolState:
        sp = self.specs[m]
        o, n, k = self.offsets[m], len(sp.z), 3 * len(sp.z)
        w = max(k - 5, 1)
        shells = torch.nonzero(self.atom_mol[self.sb.sh_atom] == m).squeeze(1)
        cpu = lambda t: t.detach().cpu().numpy().copy()  # noqa: E731
        return MolState(
            cpu(st["x"][o : o + n]),
            float(st["e"][m]),
            cpu(st["g"][o : o + n]),
            float(st["radius"][m]),
            int(st["n_evals"][m]),
            int(st["micro"][m]),
            cpu(st["hcart"][m, :k, :k]),
            cpu(st["vecs"][m, :k, :w]),
            cpu(st["h"][m, :w, :w]),
            cpu(self.start[0][shells]),
            cpu(self.start[1][o : o + n]),
            cpu(self.start[2][o : o + n]),
            int(self.streak[m]),
            int(self.fresh_starts[m]),
        )

    def run(self, stop_below: int | None = None):
        """Optimize every molecule, one batched evaluation per step for all still running.
        Without `stop_below`: the outcomes (one per spec). With it: stop once fewer molecules
        than that are running and return ({index: outcome} of the finished, {index: MolState}
        of the rest), for a driver to refill the batch."""
        cfg = self.cfg
        st, fail, new = self._initial()
        reason = fail.clone()
        active = (self.nat > 1) & (fail == FAIL_NONE)

        def finished(active):
            gmax = self._gmax(st["g"])
            return active & ((gmax <= cfg.fmax) | (st["n_evals"] >= cfg.maxiter))

        active = active & ~finished(active)
        self._rebuild(st, active & new)
        while bool(active.any()):
            if stop_below is not None and int(active.sum()) < stop_below:
                break
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
        outcomes = self._outcomes(st, reason)
        if stop_below is None:
            return outcomes
        running = active.cpu().numpy()
        done = {m: o for m, o in enumerate(outcomes) if not running[m]}
        return done, {int(m): self._extract(st, int(m)) for m in np.flatnonzero(running)}

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


# --- production's relax for many molecules, with refill --------------------------------------


@dataclass
class RelaxTask:
    key: str
    mol: object  # rdkit Mol, prepared and embedded as qm.batch._compute does
    z: np.ndarray
    start: np.ndarray  # Bohr
    charge: int
    unpaired: int


@dataclass
class RelaxOutcome:
    key: str
    xyz: np.ndarray | None
    restrained: bool
    reason: ReasonCode | None
    detail: str
    n_evals: int  # every energy of both runs, as the reference counts them
    energy: float
    gmax: float
    fresh_starts: int


def _guard_spec(mol, factor: float):
    from ksmech.qm.geometry import DissociationGuard

    g = DissociationGuard(mol, factor)
    return (g.pairs, g.rcov, float(g.factor), int(g.patience))


def _wall_spec(mol, wall_factor: float, k: float):
    from ksmech.qm.geometry import bond_walls

    w = bond_walls(mol, wall_factor, k)
    return (w.pairs, w.d0_bohr, float(w.k))


def _connectivity(task: RelaxTask, xyz: np.ndarray, factor: float) -> str | None:
    from ksmech.qm.geometry import check_connectivity
    from ksmech.qm.records import QMError

    try:
        check_connectivity(task.mol, xyz * BOHR_ANG, factor)
    except QMError as exc:
        return exc.detail
    return None


class _Job:
    """One task's place in `relax_many`: the free run, then perhaps the walled one."""

    def __init__(self, task: RelaxTask, config) -> None:
        self.task, self.cfg, self.phase, self.evals_before, self.fresh_before = (
            task,
            config,
            0,
            0,
            0,
        )
        self.spec = OptSpec(
            task.z,
            task.start,
            task.charge,
            task.unpaired,
            guard=_guard_spec(task.mol, config.conn_cov_factor),
        )

    def to_walls(self, out: OptOutcome) -> None:
        self.phase, self.evals_before = 1, self.evals_before + out.n_evals
        self.fresh_before += out.fresh_starts
        t = self.task
        self.spec = OptSpec(
            t.z,
            t.start,
            t.charge,
            t.unpaired,
            walls=_wall_spec(t.mol, self.cfg.wall_factor, self.cfg.wall_k),
        )

    def finish(self, out: OptOutcome) -> RelaxOutcome | None:
        """The outcome, or None when the walled run is still to come (`qm.xtb_calc._relax`)."""
        cfg, broke = self.cfg, None
        if out.converged:
            broke = _connectivity(self.task, out.xyz, cfg.conn_cov_factor)
            if broke is None:
                return self._done(out, None, "")
        elif out.reason is not ReasonCode.CONNECTIVITY_CHANGED:
            return self._done(out, out.reason, out.detail)
        detail = broke or out.detail
        if self.phase == 0 and cfg.restrain:
            self.to_walls(out)
            return None
        return self._done(out, ReasonCode.CONNECTIVITY_CHANGED, detail)

    def _done(self, out: OptOutcome, reason, detail) -> RelaxOutcome:
        ok = reason is None
        return RelaxOutcome(
            self.task.key,
            out.xyz if ok else None,
            self.phase == 1,
            reason,
            detail,
            self.evals_before + out.n_evals,
            out.energy,
            out.gmax,
            self.fresh_before + out.fresh_starts,
        )


MIN_COMPACT = 8  # a batch smaller than this runs to its end


def _nao(z: np.ndarray) -> int:
    from ksmech.qm.gpu.basis import mol_basis

    return mol_basis(np.asarray(z)).nao


def molecule_bytes(z: np.ndarray) -> int:
    """A batch member's peak device memory: ~50 n_ao^2 doubles for the SCF (the padded
    matrices, the eigensolver's workspace and transient copies; measured 47-49 at 21-117 atoms)
    and the optimizer's (3N)^2 Hessians and basis."""
    from ksmech.qm.gpu.basis import mol_basis

    nao, p = mol_basis(np.asarray(z)).nao, 3 * len(z)
    return 8 * (50 * nao * nao + 4 * p * p)


def relax_many(
    tasks: list[RelaxTask],
    config=None,
    capacity: int = 1024,
    refill: float = 0.75,
    solvent: bool = True,
    rfo_dtype=F64,
    on_batch=None,
    memory_bytes: float | None = None,
    size_ratio: float = 1.5,
) -> list[RelaxOutcome]:
    """`qm.xtb_calc.GFN2Calculator._relax` with the tblite-anc settings of `config` (a QMConfig)
    for many molecules: a free run with the dissociation guard, `check_connectivity`, and on a
    broken bond a walled run from the start geometry, checked again. The batch holds up to
    `capacity` runs and at most `memory_bytes` (`molecule_bytes` of its largest member times
    its size; default 60% of the device memory free at the start, as another process may hold
    the rest); when fewer than `refill` x its size still run, it is rebuilt with them (their
    state carried over) and the next tasks, smallest first. A batch takes a molecule only if
    its AO count is within `size_ratio` of the batch's smallest: every member pays for the
    padding to the largest."""
    from ksmech.qm.xtb_calc import QMConfig

    config = config or QMConfig(optimizer="tblite-anc")
    if memory_bytes is None:
        torch.cuda.empty_cache()
        memory_bytes = 0.6 * torch.cuda.mem_get_info()[0]
    cfg = AncSettings(fmax=config.fmax, maxiter=config.maxiter)
    queue = sorted((_Job(t, config) for t in tasks), key=lambda j: _nao(j.task.z))
    running: list[_Job] = []
    results: dict[str, RelaxOutcome] = {}
    while queue or running:
        while queue and len(running) < capacity:
            biggest = max([molecule_bytes(j.task.z) for j in running] + [0])
            need = max(biggest, molecule_bytes(queue[0].task.z)) * (len(running) + 1)
            smallest = min([_nao(j.task.z) for j in running] + [_nao(queue[0].task.z)])
            if running and (need > memory_bytes or _nao(queue[0].task.z) > size_ratio * smallest):
                break
            running.append(queue.pop(0))
        opt = BatchedANC([j.spec for j in running], solvent, cfg, rfo_dtype=rfo_dtype)
        # stop to refill, or with nothing queued to compact: a finished molecule left in the
        # batch would still pay its geometry and gradient at every step
        stop = int(refill * len(running)) if len(running) >= MIN_COMPACT else None
        got = opt.run(stop_below=stop) if stop else (dict(enumerate(opt.run())), {})
        done, states = got
        nxt = []
        for m, job in enumerate(running):
            if m in states:
                job.spec.state = states[m]
                nxt.append(job)
                continue
            res = job.finish(done[m])
            if res is None:
                nxt.append(job)  # the walled run, from the start
            else:
                results[job.task.key] = res
        running = nxt
        if on_batch is not None:
            on_batch(len(results), len(running), len(queue))
    return [results[t.key] for t in tasks]
