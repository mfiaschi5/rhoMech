"""Stage 4: the batched ANC / RFO geometry optimizer on the GPU, `qm.anc` for many molecules.

Every molecule runs `qm.anc.optimize_anc`'s algorithm with its own state: the Lindh model
Hessian once at the start, the approximate normal coordinates (ANC) rebuilt every `micro` steps
from the learned Hessian, RFO steps cut to a trust radius that grows and shrinks as there,
uphill steps taken back, BFGS updates in the ANC basis, convergence on the largest gradient
component (walls included). The energies and gradients of all still-running molecules come
from one batched GFN2 SCF state (`SCFBatch`), each SCF restarted from the molecule's last
evaluated point and, if that fails, from the guess (as `geometry.TbliteEnergy`). The bond
walls and the dissociation guard of `geometry` apply per molecule: the guard sees every trial
geometry, the first included, once, before it is evaluated. With a `carry`, an SCF still
running when most of its round has ended continues into the next round while the others step
(`BatchedANC`).

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
    trace: list = field(default_factory=list)  # (energy, radius) per step evaluation, traced


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
    batch of molecules on the GPU. `rfo_dtype` float32 runs the RFO eigensolve in fp32.

    The optimization runs in rounds. In each, every running molecule that is not waiting for an
    evaluation proposes its next trial (stored, and frozen while its evaluation is pending), the
    guard checks that trial once, its SCF starts from the molecule's last evaluated point, and
    the batch's SCFs advance; each molecule whose evaluation ended then takes the step's update
    (accept or reject, BFGS, trust radius, convergence test, basis). `carry` None: a round waits
    for every SCF, one round per step as `qm.anc` steps. A number c: the round ends once at most
    c x its running SCFs still run (`SCFBatch.advance`), and those carry their SCF state into the
    next round beside the SCFs the finished molecules start. Per molecule the computation is the
    same either way (trial, SCF start, mixer history, convergence rules)."""

    def __init__(
        self,
        specs: list[OptSpec],
        solvent: bool = True,
        cfg: AncSettings | None = None,
        accuracy: float = 1.0,
        rfo_dtype=F64,
        trace: bool = False,
        carry: float | None = None,
        max_iter: int = 250,
    ) -> None:
        self.specs, self.cfg, self.rfo_dtype, self.tracing, self.carry = (
            specs,
            cfg or AncSettings(),
            rfo_dtype,
            trace,
            carry,
        )
        mols = [(sp.z, sp.start, sp.charge, sp.unpaired) for sp in specs]
        self.sb = S.SCFBatch(mols, solvent, accuracy=accuracy, max_iter=max_iter)
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
        self.last_ratio = torch.full((self.b,), -1.0, dtype=F64, device="cuda")
        self.drained = 0  # evaluations in flight that a refill stop finished (`run`)
        self.rounds = 0

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
        """DissociationGuard on the new trial geometries of the molecules `active`, each trial
        counted once, when it is proposed: True = stop. `last_ratio` keeps each molecule's
        worst bond ratio at its last checked trial."""
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
        self.last_ratio = torch.where(active, worst, self.last_ratio)
        return active & (gd["patience"] > 0) & (self.streak >= gd["patience"])

    # --- one round: proposals, evaluations started, SCFs advanced, updates ---------------------

    def _propose(self, st: dict, which: torch.Tensor) -> None:
        """The next trial of the molecules `which` (`qm.anc._State.step` up to its evaluation):
        the RFO step in the ANC basis, cut to the trust radius, its predicted energy change and
        the trial point, stored per molecule and left unchanged while the evaluation runs. The
        RFO eigensolve covers only the molecules `which` (one host sync per round): with a carry,
        the pending and the finished molecules would be solved for nothing; a molecule's step
        moves with the solve's batch at rounding level only (1e-15)."""
        vecs, h = st["vecs"], st["h"]
        gq = (vecs.transpose(1, 2) @ self._pad(st["g"]).reshape(self.b, -1, 1))[:, :, 0]
        idx = torch.nonzero(which).squeeze(1)
        if not len(idx):
            return
        s = torch.zeros_like(gq)
        s[idx] = rfo_step(h[idx], gq[idx], self.rfo_dtype)
        ns = torch.linalg.norm(s, dim=1)
        cut = torch.where(ns > st["radius"], st["radius"] / ns.clamp(min=1e-300), 1.0)
        s = s * cut[:, None]
        ns = torch.linalg.norm(s, dim=1)
        predicted = (gq * s).sum(1) + 0.5 * (s * (h @ s[:, :, None])[:, :, 0]).sum(1)
        trial_pad = self._pad(st["x"]) + (vecs @ s[:, :, None]).reshape(self.b, self.nmax, 3)
        st["trial"] = self._unpad_into(st["trial"], trial_pad, which)
        st["s"] = torch.where(which[:, None], s, st["s"])
        st["pred"] = torch.where(which, predicted, st["pred"])
        st["ns"] = torch.where(which, ns, st["ns"])

    def _launch(self, st: dict, init: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Start the evaluations of the molecules `init` (their first, at x) and `step` (at
        their proposed trial): the guard checks each new trial once; the SCFs of those it lets
        through start from the molecule's last evaluated point (a step's from the guess if that
        fails, within the same evaluation; a first evaluation starts from the guess and fails
        alone, as `TbliteEnergy` re-raises without a previous result); the geometry takes the
        new trials and keeps the pending ones. Returns the guard's stops."""
        newly = init | step
        st["trial"] = torch.where(init[self.atom_mol][:, None], st["x"], st["trial"])
        stop = self._guard(st["trial"], newly)
        go = newly & ~stop
        at = (self.pending | newly)[self.atom_mol][:, None]
        self.xeval = torch.where(at, st["trial"], st["x"])
        self.sb.set_geometry(self.xeval)
        self.sb.start(self.scf, go, self.start, fallback=step)
        self.pending = self.pending | go
        return stop

    def _collect(self, ended: torch.Tensor):
        """Energies (b,), gradients (nat, 3) (walls included) and failure codes (b,) of the
        evaluations that ended (`ended`; the other rows are not meaningful); each of those
        molecules' next SCF starts from this one's variables when it converged."""
        r = self.sb.result(self.scf)
        self.fresh_starts += (ended & self.scf.fell).long()
        e, g = self.sb.gradient(r)
        ew, gw = self._wall_energy(self.xeval)
        fail = torch.where(ended & ~r.converged, FAIL_SCF, FAIL_NONE)
        keep = ended & r.converged
        sh, at = keep[self.atom_mol[self.sb.sh_atom]], keep[self.atom_mol]
        self.start = (
            torch.where(sh, r.qsh, self.start[0]),
            torch.where(at[:, None], r.dp, self.start[1]),
            torch.where(at[:, None], r.qp, self.start[2]),
        )
        return e + ew, g + gw, fail

    def _apply_first(self, st: dict, first, e, g) -> None:
        """A new molecule's first evaluation: its energy and gradient, one energy counted."""
        am = first[self.atom_mol]
        st["e"], st["g"] = torch.where(first, e, st["e"]), torch.where(am[:, None], g, st["g"])
        st["n_evals"] += first.long()

    def _apply_step(self, st: dict, done, e_new, g_new, fail) -> None:
        """The rest of `qm.anc._State.step` for the molecules `done` whose evaluation of their
        stored trial ended (or the guard stopped): the energy counted, the uphill rejection,
        BFGS, the trust radius, the new point."""
        cfg, vecs, h, s = self.cfg, st["vecs"], st["h"], st["s"]
        st["n_evals"] += done.long()
        st["micro"] += done.long()
        de, predicted = e_new - st["e"], st["pred"]
        ratio = torch.where(predicted < 0, de / torch.where(predicted < 0, predicted, -1.0), -1.0)
        ok = done & (fail == FAIL_NONE)
        reject = ok & (de > cfg.uphill) & (ratio < 0)
        acc = ok & ~reject
        y = (vecs.transpose(1, 2) @ self._pad(g_new - st["g"]).reshape(self.b, -1, 1))[:, :, 0]
        st["h"] = torch.where(acc[:, None, None], bfgs(h, s, y), h)
        r, ns = st["radius"], st["ns"]
        grow = acc & (ratio > 0.75) & (ns > 0.8 * r)
        shrink = reject | (acc & ~grow & (ratio < 0.25))
        r = torch.where(grow, (2 * r).clamp(max=cfg.radius_max), r)
        st["radius"] = torch.where(shrink, (r / 4).clamp(min=cfg.radius_min), r)
        am = acc[self.atom_mol]
        st["x"] = torch.where(am[:, None], st["trial"], st["x"])
        st["g"] = torch.where(am[:, None], g_new, st["g"])
        st["e"] = torch.where(acc, e_new, st["e"])
        if self.tracing:
            self.trace.append(tuple(t.cpu().numpy() for t in (e_new, st["radius"], acc, done)))

    def _round(self, st: dict, active, reason, mode: str, launch: bool = True):
        """One round. "init": the new molecules' first evaluations start; "step": also every
        other running molecule not waiting proposes and starts its next one; "drain": nothing
        starts and every evaluation in flight runs to its end. Then each molecule whose
        evaluation ended is updated. Returns the running molecules and the failure codes."""
        self.rounds += 1
        stop = torch.zeros_like(active)
        if launch and mode != "drain":
            free = active & ~self.pending
            step = free & ~self.need if mode == "step" else torch.zeros_like(free)
            if mode == "step":
                self._propose(st, step)
            stop = self._launch(st, free & self.need, step)
        self.sb.advance(self.scf, None if mode == "drain" else self.carry)
        ended = self.pending & ~self.scf.running  # a geometry the SCF refused ends here too
        self.pending = self.pending & ~ended
        if int(ended.sum()):
            e, g, fail = self._collect(ended)
            e = torch.where(ended, e, torch.nan)  # a guard stop's energy: never evaluated
        else:
            e = torch.full((self.b,), torch.nan, dtype=F64, device="cuda")
            g, fail = torch.zeros_like(st["x"]), torch.zeros_like(reason)
        fail, done = torch.where(stop, FAIL_GUARD, fail), ended | stop
        first = done & self.need
        self._apply_first(st, first, e, g)
        self.need = self.need & ~first
        if mode != "init":
            self._apply_step(st, done & ~first, e, g, fail)
        reason = torch.where(done & (fail != FAIL_NONE), fail, reason)
        active = active & (fail == FAIL_NONE)
        active = active & ~self._finished(st, active & ~self.pending)
        self._rebuild(st, first & active, active & (st["micro"] >= self.cfg.micro))
        return active, reason

    def _finished(self, st: dict, which: torch.Tensor) -> torch.Tensor:
        gmax = self._gmax(st["g"])
        return which & ((gmax <= self.cfg.fmax) | (st["n_evals"] >= self.cfg.maxiter))

    def _rebuild(self, st: dict, new: torch.Tensor, due: torch.Tensor) -> None:
        """New ANC bases: of the molecules `new` from the Lindh model Hessian, of those `due`
        (`micro` steps since their last basis) from the learned Hessian in the old basis:
        `qm.anc.optimize_anc`'s outer loop."""
        which = new | due
        idx = torch.nonzero(which).squeeze(1)
        if not len(idx):
            return
        v, h = st["vecs"], st["h"]
        st["hcart"] = torch.where(due[:, None, None], v @ h @ v.transpose(1, 2), st["hcart"])
        vecs, lam, _ = anc_basis(st["hcart"][idx], self._pad(st["x"])[idx], self.real[idx])
        mm, m = st["vecs"].shape[2], vecs.shape[2]
        st["vecs"][idx] = torch.nn.functional.pad(vecs, (0, mm - m))
        st["h"][idx] = torch.diag_embed(
            torch.nn.functional.pad(lam, (0, mm - m), value=PAD_CURVATURE)
        )
        st["micro"][idx] = 0

    # --- the batch's start, its states, the loop -----------------------------------------------

    def _initial(self) -> tuple[dict, torch.Tensor]:
        """Every molecule's start: resumed molecules from their state; new ones the Lindh model
        Hessian, their first evaluation (from the SAD guess) still to come (`self.need`)."""
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
            "trial": x.clone(),  # each molecule's last proposed point (pending: being evaluated)
            "s": torch.zeros(b, mm, dtype=F64, **dev),  # its step, predicted change and |step|
            "pred": torch.zeros(b, dtype=F64, **dev),
            "ns": torch.zeros(b, dtype=F64, **dev),
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
        self.need = torch.as_tensor(~resumed, **dev) & (self.nat > 1)
        self.scf, self.pending, self.xeval = self.sb.new_run(), torch.zeros_like(self.need), x
        return st, self.need.clone()

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
        """Optimize every molecule. Without `stop_below`: the outcomes (one per spec). With it:
        stop once fewer molecules than that are running and return ({index: outcome} of the
        finished, {index: MolState} of the rest), for a driver to refill the batch. A `MolState`
        holds no SCF in flight: at the stop, the evaluations still pending (with a `carry`) run
        to their end and their steps are taken first (`drained` counts them), so nothing a
        round started is lost; a molecule those steps converge is among the finished."""
        st, _ = self._initial()
        reason = torch.full((self.b,), FAIL_NONE, dtype=torch.long, device="cuda")
        active, reason = self._round(st, self.nat > 1, reason, "init")
        while True:
            n_active, n_free = torch.stack([active.sum(), (active & ~self.pending).sum()]).tolist()
            if not n_active:
                break
            if stop_below is not None and n_active < stop_below:
                if n_active > n_free:
                    self.drained += n_active - n_free
                    active, reason = self._round(st, active, reason, "drain")
                break
            active, reason = self._round(st, active, reason, "step", launch=n_free > 0)
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
                    [(float(t[0][m]), float(t[1][m])) for t in self.trace if t[3][m]],
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
CARRY = 0.1  # relax_many's round: done once at most a tenth of its SCFs still run


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


def _fill(queue: list, running: list, capacity: int, memory_bytes: float, size_ratio: float,
          claim=None):  # fmt: skip
    """Move tasks from the queue (smallest first) into the batch while they fit. A task whose
    `claim(key)` is False (another worker has it) leaves the queue without running."""
    while queue and len(running) < capacity:
        biggest = max([molecule_bytes(j.task.z) for j in running] + [0])
        need = max(biggest, molecule_bytes(queue[0].task.z)) * (len(running) + 1)
        smallest = min([_nao(j.task.z) for j in running] + [_nao(queue[0].task.z)])
        if running and (need > memory_bytes or _nao(queue[0].task.z) > size_ratio * smallest):
            break
        job = queue.pop(0)
        if claim is None or claim(job.task.key):
            running.append(job)


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
    carry: float | None = CARRY,
    on_result=None,
    more=None,
    claim=None,
) -> list[RelaxOutcome]:
    """`qm.xtb_calc.GFN2Calculator._relax` with the tblite-anc settings of `config` (a QMConfig)
    for many molecules: a free run with the dissociation guard, `check_connectivity`, and on a
    broken bond a walled run from the start geometry, checked again. The batch holds up to
    `capacity` runs and at most `memory_bytes` (`molecule_bytes` of its largest member times
    its size; default 60% of the device memory free at the start, as another process may hold
    the rest); when fewer than `refill` x its size still run, it is rebuilt with them (their
    state carried over) and the next tasks, smallest first. A batch takes a molecule only if
    its AO count is within `size_ratio` of the batch's smallest: every member pays for the
    padding to the largest. `carry` is `BatchedANC`'s (None: every step waits for its slowest
    SCF); the SCF cycle limit is `config.scf_maxiter`. `on_result(outcome)` is called with each
    task's outcome once it is final, while the other tasks still run; the outcomes are then
    not kept, and the result is []. `more(block)`, if given, is asked for further tasks at every
    refill (blocking when nothing else is left to run): a list (they join the queue in size
    order), or None once there will be no more. `claim(key)`, if given, is asked for each task as
    a batch takes it: False drops the task (work sharing, `qm.gpu_run`: a CPU worker has it), and
    it has no outcome."""
    from ksmech.qm.xtb_calc import QMConfig

    config = config or QMConfig(optimizer="tblite-anc")
    if memory_bytes is None:
        torch.cuda.empty_cache()
        memory_bytes = 0.6 * torch.cuda.mem_get_info()[0]
    cfg = AncSettings(fmax=config.fmax, maxiter=config.maxiter)
    queue, keys = _queued([], tasks, config), [t.key for t in tasks]
    running: list[_Job] = []
    results: dict[str, RelaxOutcome] = {}
    n_done = 0
    while queue or running or more is not None:
        if more is not None:
            got = more(not queue and not running)
            more = None if got is None else more
            queue = _queued(queue, got or [], config)
            keys += [t.key for t in got or []] if on_result is None else []
            if not queue and not running:
                continue
        _fill(queue, running, capacity, memory_bytes, size_ratio, claim)
        if not running:  # every task the fill met was claimed elsewhere
            continue
        opt = BatchedANC([j.spec for j in running], solvent, cfg, rfo_dtype=rfo_dtype,
                         carry=carry, max_iter=config.scf_maxiter)  # fmt: skip
        running, ended = _run_batch(opt, running, refill)
        del opt  # its SCF state and integrals, before the next batch allocates its own
        for res in ended:
            n_done += 1
            if on_result is None:
                results[res.key] = res
            else:  # handed over, not kept: a long fed run would hold every outcome
                on_result(res)
        if on_batch is not None:
            on_batch(n_done, len(running), len(queue))
    return [results[k] for k in keys if k in results] if on_result is None else []


def _queued(queue: list, new: list, config) -> list:
    """The queue with the tasks `new` joined, smallest AO count first."""
    if not new:
        return queue
    return sorted(queue + [_Job(t, config) for t in new], key=lambda j: _nao(j.task.z))


def _run_batch(opt: BatchedANC, running: list, refill: float):
    """Run one batch: stop to refill, or with nothing queued to compact (a finished molecule
    left in the batch would still pay its geometry and gradient at every step). Returns (the
    jobs still to run: resumed, or their walled run to start; the outcomes that are final)."""
    stop = int(refill * len(running)) if len(running) >= MIN_COMPACT else None
    done, states = opt.run(stop_below=stop) if stop else (dict(enumerate(opt.run())), {})
    nxt, ended = [], []
    for m, job in enumerate(running):
        if m in states:
            job.spec.state = states[m]
            nxt.append(job)
            continue
        res = job.finish(done[m])
        if res is None:
            nxt.append(job)  # the walled run, from the start
        else:
            ended.append(res)
    return nxt, ended
