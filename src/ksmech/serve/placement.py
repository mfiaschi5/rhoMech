"""The oriented contact placement that produced a token pair's overlap (adapters A13, A14).

The oriented module (`physics.contact.legacy_fixed23`, through `serve.upstream`) evaluates
every placement of the acceptor's cap around the donor and keeps only the winning |S| and beta;
the placement itself (tier, slots, roll, rotation) is discarded.
`reconstruct_contact` replays `_evaluate_block` and `_site_pair_values` for ONE site pair with
the same private building blocks and the same tie rule, and keeps the winner. Those names are
pinned to `ORIENTED_VERSION == 5`; another version (or a missing name) is
`PHYSICS_VERSION_MISMATCH` (409), checked when a placement is requested.

`verify_placement` proves on every request that the placement is the one the kernel evaluated:
(1) S, beta and CLASH equal the production per-site path `block_contacts(..., sites_F=[[a]],
sites_G=[[b]])` (rel 1e-10); (2) at the analysis' contact sites, S equals the cached
`oriented_block` value and K beta equals the analysis beta; (3) site b sits at the configured
distance from site a and the placed acceptor lobe is antiparallel to the donor lobe. A failure
of 1-3 is `PLACEMENT_MISMATCH` (500) and no geometry is shown. (4) A soft analytic check
recomputes S with `gto.overlap_matrix` on the placed atoms and flags `ANALYTIC_OVERLAP_DRIFT`
(screening at tol = 1e-6 per atom pair can drop that much).

`t_contact` writes `data/serve/contact/<sha>.json` (the `ContactOut` fields minus provenance and
cubes) and the two cubes on one shared grid (the donor cap and the placed acceptor cap +- 4
Angstrom): the whole donor LMO, and the cap-truncated acceptor LMO, moved rigidly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ksmech.serve.contracts import PoolState, Runtime, to_jsonable
from ksmech.serve.errors import Code, ServeError

if TYPE_CHECKING:
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.orbitals.tokens import MoleculeTokens
    from ksmech.physics.contact.legacy_fixed23 import OrientedConfig
    from ksmech.serve.analysis import PoolAnalysis
    from ksmech.serve.settings import ServeSettings

PLACEMENT_VERSION = 1
PINNED_ORIENTED_VERSION = 5
PRIVATE_NAMES = ("_Side", "_entries", "_tier", "_concat", "_overlaps", "_placed", "TIER1", "TIER2")
REL_TOL = 1e-10  # checks 1 and 2 (S, beta1 against the production paths)
BETA_TOL = 1e-9  # check 2: K beta1 against the analysis beta (eV)
GEOM_TOL = 1e-9  # check 3: distance (Angstrom) and lobe antiparallelism
CONTEXT_NOTE = "context only: the kernel placed only the cap"


def oriented_module():
    """The upstream oriented module, after checking the pinned version and the private names."""
    from ksmech.serve.upstream import ORIENTED_SOURCE
    from ksmech.serve.upstream import oriented_module as _upstream

    ori = _upstream()

    version = getattr(ori, "ORIENTED_VERSION", None)
    missing = [n for n in PRIVATE_NAMES if not hasattr(ori, n)]
    if version != PINNED_ORIENTED_VERSION or missing:
        msg = (
            f"the placement reconstruction is pinned to ORIENTED_VERSION "
            f"{PINNED_ORIENTED_VERSION}; {ORIENTED_SOURCE} has {version}"
        )
        if missing:
            msg += f" and lacks {missing}"
        detail = {"oriented_version": version, "pinned": PINNED_ORIENTED_VERSION}
        raise ServeError(Code.PHYSICS_VERSION_MISMATCH, msg, status=409, detail=detail)
    return ori


def check_source_version() -> None:
    """The version pin read from the source with `ast` (API side: the oriented module, and with it
    tblite, is not imported); the worker checks the imported module again."""
    from ksmech.serve.upstream import ORIENTED_SOURCE
    from ksmech.serve.versions import source_constant

    try:
        version = source_constant(ORIENTED_SOURCE, "ORIENTED_VERSION")
    except LookupError:
        version = None
    if version != PINNED_ORIENTED_VERSION:
        msg = (
            f"the placement reconstruction is pinned to ORIENTED_VERSION {PINNED_ORIENTED_VERSION}"
        )
        detail = {"oriented_version": version, "pinned": PINNED_ORIENTED_VERSION}
        raise ServeError(Code.PHYSICS_VERSION_MISMATCH, msg, status=409, detail=detail)


@dataclass
class Placement:
    """The winning placement of one (r, a, s, b) site pair, in bohr where named so."""

    k: int  # index of the winner among the concatenated tier-1 + tier-2 placements
    tier: int  # 1: the tied lobes, 2: an opposite face
    roll: int  # roll step about the approach axis (0: the smallest rotation)
    u: np.ndarray  # (3,) the donor lobe, the approach axis
    rot: np.ndarray  # (3, 3) rotation of the acceptor cap
    anchor_bohr: np.ndarray  # (3,) the acceptor site in its own frame (rotation centre)
    shift_bohr: np.ndarray  # (3,) where the acceptor site goes (x_a + d u)
    cap_atoms: np.ndarray  # acceptor record atoms of the cap, ascending
    cap_xyz_ang: np.ndarray  # (n_cap, 3) placed cap positions, donor frame
    S: float  # |S| of the winner
    beta1: float  # AO-level resonance integral at K = 1 (eV), S's phase
    clash: bool  # every placement of the site pair clashed
    slots: tuple[int, int]  # (donor slot p, acceptor slot q): 0, 1 tied lobes, 2 opposite
    donor_dirs: np.ndarray  # (3, 3) the donor site's slot directions
    donor_valid: np.ndarray  # (3,)
    acceptor_dirs: np.ndarray  # (3, 3) the acceptor site's slot directions, own frame
    acceptor_valid: np.ndarray  # (3,)

    def placed(self, xyz_ang: np.ndarray) -> np.ndarray:
        """Acceptor-frame positions (Angstrom) under the placement's rigid motion."""
        from ksmech.serve.cubes import bohr_ang

        b = bohr_ang()
        rel = np.asarray(xyz_ang, dtype=float).reshape(-1, 3) - self.anchor_bohr * b
        return rel @ self.rot.T + self.shift_bohr * b


def _slots(ori, EF, EG, u: np.ndarray, rot: np.ndarray, tier: int) -> tuple[int, int]:
    """The direction slots (p, q) of the tier whose donor lobe is u and whose acceptor lobe,
    rotated, is -u (within 1e-9)."""
    for p, q in ori.TIER1 if tier == 1 else ori.TIER2:
        if not (EF.valid[0, p] and EG.valid[0, q]):
            continue
        same_u = np.abs(EF.dirs[0, p] - u).max() <= GEOM_TOL
        if same_u and np.abs(rot @ EG.dirs[0, q] + u).max() <= GEOM_TOL:
            return int(p), int(q)
    raise ServeError(
        Code.PLACEMENT_MISMATCH, "no direction slot reproduces the winning placement", status=500
    )


def _winner(ori, F, G, caps, EF, EG, cfg) -> tuple[Any, int, int, float, float, bool]:
    """`_evaluate_block` for one site pair, keeping the placement: (placements, winner k,
    len(tier 1), |S|, beta1, failed). Same evaluated set, key and stable tie rule."""
    d, lim = cfg.distance / ori.BOHR_ANG, cfg.clash / ori.BOHR_ANG
    pl1, c1 = ori._tier(F, G, caps, EF, EG, np.arange(1), ori.TIER1, d, lim)
    ok1 = np.zeros(1, dtype=bool)
    ok1[pl1.sp[~c1]] = True
    pl2, c2 = ori._tier(F, G, caps, EF, EG, np.flatnonzero(~ok1), ori.TIER2, d, lim)
    ok2 = np.zeros(1, dtype=bool)
    ok2[pl2.sp[~c2]] = True
    failed = bool((~ok1 & ~ok2)[0])
    pl, clash = ori._concat(pl1, pl2), np.concatenate([c1, c2])
    in_tier1 = np.arange(len(clash)) < len(c1)
    evaluated = (~clash & (in_tier1 | ~ok1[pl.sp])) | (failed & (pl.roll == 0))
    idx = np.flatnonzero(evaluated)
    values, betas = ori._overlaps(F, caps, pl, idx, cfg.tol)
    key = -values if failed else values  # the largest |S|, or the smallest when all clashed
    key = np.where(clash[idx] & (not failed), -np.inf, key)
    win = int(np.lexsort((key, pl.sp[idx]))[-1])  # stable: the last of equal keys wins
    return pl, int(idx[win]), len(c1), float(values[win]), float(betas[win]), failed


def reconstruct_contact(
    t_F: MoleculeTokens, r: int, a: int, t_G: MoleculeTokens, s: int, b: int, cfg: OrientedConfig
) -> Placement:
    """The placement the oriented kernel evaluated for donor token r (position in
    t_F.orbitals) at site a and acceptor token s at site b (record atom indices)."""
    ori = oriented_module()
    F, G = ori._Side.of(t_F, [r]), ori._Side.of(t_G, [s])
    caps = {0: ori._Side.cap(t_G, s)}
    EF, EG = ori._entries(t_F, [r], [[a]]), ori._entries(t_G, [s], [[b]])
    pl, k, n1, S, beta1, failed = _winner(ori, F, G, caps, EF, EG, cfg)
    tier = 1 if k < n1 else 2
    u, rot = pl.axis[k].copy(), pl.rot[k].copy()
    return Placement(
        k=k,
        tier=tier,
        roll=int(pl.roll[k]),
        u=u,
        rot=rot,
        anchor_bohr=pl.anchor[k].copy(),
        shift_bohr=pl.shift[k].copy(),
        cap_atoms=np.asarray(ori.cap_atoms(t_G.orbitals[s], t_G), dtype=int),
        cap_xyz_ang=ori._placed(caps[0], pl, np.array([k]))[0] * ori.BOHR_ANG,
        S=S,
        beta1=beta1,
        clash=failed,
        slots=_slots(ori, EF, EG, u, rot, tier),
        donor_dirs=EF.dirs[0].copy(),
        donor_valid=EF.valid[0].copy(),
        acceptor_dirs=EG.dirs[0].copy(),
        acceptor_valid=EG.valid[0].copy(),
    )


def placed_cap_coefficients(t_G: MoleculeTokens, s: int, rot: np.ndarray) -> np.ndarray:
    """The cap-truncated acceptor coefficients (cap atoms ascending, each atom's AOs in tblite
    order) expressed in the placed frame: per shell `rotate_coeffs(l, c, rot)`, the convention
    `_group_overlap` uses with M = Q @ rot."""
    from ksmech.physics.gto import element_shells

    ori = oriented_module()
    c = np.asarray(t_G.orbitals[s].coeff_vector, dtype=float)
    parts = []
    for atom in ori.cap_atoms(t_G.orbitals[s], t_G):
        ca, k = c[np.flatnonzero(t_G.ao_atom == atom)], 0
        for sh in element_shells(int(t_G.atoms[int(atom)].atomic_number)):
            parts.append(ori.rotate_coeffs(sh.l, ca[None, k : k + sh.n_sph], rot[None])[0])
            k += sh.n_sph
        if k != len(ca):
            raise ServeError(Code.CONSISTENCY_FAILED, f"atom {atom}: {len(ca)} AOs, shells {k}")
    return np.concatenate(parts)


def _block_position(t: MoleculeTokens, k: int) -> int:
    """Position of token k among the tokens of its own kind (its oriented_block row / col)."""
    kind = t.orbitals[k].kind
    return [i for i, o in enumerate(t.orbitals) if o.kind == kind].index(k)


def _production_checks(pl, t_F, r, a, t_G, s, b, cfg, intra, an_beta, is_contact) -> dict:
    """Checks 1 and 2 (hard)."""
    from ksmech.physics.params import KSParams

    ori = oriented_module()
    bc = ori.block_contacts(t_F, [r], t_G, [s], cfg, sites_F=[[a]], sites_G=[[b]], intra=intra)
    out: dict[str, float | bool] = {
        "S_production": float(bc.S[0, 0]),
        "dS_production": abs(pl.S - float(bc.S[0, 0])),
        "dbeta1_production": abs(pl.beta1 - float(bc.beta[0, 0])),
        "clash_matches": bool(pl.clash == bool(bc.clash[0, 0])),
    }
    ok = out["dS_production"] <= REL_TOL * max(1.0, pl.S) and out["clash_matches"]
    ok = ok and out["dbeta1_production"] <= REL_TOL * max(1.0, abs(pl.beta1))
    if is_contact:
        ob = ori.oriented_block(t_F, t_G, "occ_virt", cfg, intra=intra)
        out["dS_block"] = abs(pl.S - float(ob.S[_block_position(t_F, r), _block_position(t_G, s)]))
        ok = ok and out["dS_block"] <= REL_TOL
        if an_beta is not None:
            out["dbeta_analysis"] = abs(KSParams().K * pl.beta1 - float(an_beta))
            ok = ok and out["dbeta_analysis"] <= BETA_TOL
    out["production_ok"] = bool(ok)
    return out


def _geometry_checks(pl: Placement, t_F, a: int, b: int, cfg) -> dict:
    """Check 3 (hard): |placed b - x_a| = d, and rot v = -u."""
    x_a = np.asarray(t_F.atoms[a].position, dtype=float)
    placed_b = pl.cap_xyz_ang[list(pl.cap_atoms).index(b)]
    v = pl.acceptor_dirs[pl.slots[1]]
    out: dict[str, float | bool] = {
        "distance_ang": float(np.linalg.norm(placed_b - x_a)),
        "lobe_antiparallel_err": float(np.abs(pl.rot @ v + pl.u).max()),
    }
    out["d_distance"] = abs(out["distance_ang"] - cfg.distance)
    ok = out["d_distance"] <= GEOM_TOL and out["lobe_antiparallel_err"] <= GEOM_TOL
    out["geometry_ok"] = bool(ok)
    return out


def _analytic_check(pl: Placement, t_F, r: int, t_G, s: int) -> dict:
    """Check 4 (soft): l_r^T S_full[F, cap] c_placed against S."""
    from ksmech.physics.gto import overlap_matrix

    z_F = np.array([x.atomic_number for x in t_F.atoms], dtype=int)
    x_F = np.array([x.position for x in t_F.atoms], dtype=float).reshape(-1, 3)
    z_cap = np.array([t_G.atoms[int(i)].atomic_number for i in pl.cap_atoms], dtype=int)
    S_full = overlap_matrix(np.concatenate([z_F, z_cap]), np.vstack([x_F, pl.cap_xyz_ang]))
    l_r = np.asarray(t_F.orbitals[r].coeff_vector, dtype=float)
    n_F = len(l_r)
    S_an = float(l_r @ S_full[:n_F, n_F:] @ placed_cap_coefficients(t_G, s, pl.rot))
    drift = abs(abs(S_an) - pl.S)
    return {
        "S_analytic": abs(S_an),
        "analytic_drift": drift,
        "analytic_ok": bool(drift <= 2e-5 + 1e-4 * pl.S),
    }


def verify_placement(
    pl: Placement,
    t_F: MoleculeTokens,
    r: int,
    a: int,
    t_G: MoleculeTokens,
    s: int,
    b: int,
    cfg: OrientedConfig,
    intra: bool,
    an_beta: float | None,
    is_contact: bool,
) -> dict[str, float | bool]:
    """Every check value; "consistent" is true only when checks 1-3 pass. A failed soft check
    adds "ANALYTIC_OVERLAP_DRIFT": True."""
    out = _production_checks(pl, t_F, r, a, t_G, s, b, cfg, intra, an_beta, is_contact)
    out |= _geometry_checks(pl, t_F, a, b, cfg)
    out |= _analytic_check(pl, t_F, r, t_G, s)
    if not out["analytic_ok"]:
        out[str(Code.ANALYTIC_OVERLAP_DRIFT)] = True
    out["consistent"] = bool(out["production_ok"] and out["geometry_ok"])
    return out


# ---- the contact payload


@dataclass
class ContactParts:
    """What the contact cubes need besides the JSON payload."""

    t_F: MoleculeTokens
    r: int
    t_G: MoleculeTokens
    s: int
    pl: Placement


def resolve_ids(rt: ReactantTokens, an: PoolAnalysis, donor, donor_site, acceptor, acceptor_site):
    """(i, j, f, r, a, g, s, b) of the request; NOT_AN_ORBITAL_TOKEN / SITE_NOT_A_CENTRE."""
    from ksmech.serve.geometry import orbital_position

    i, j = an.cell(donor, acceptor)
    (f, r), (g, s) = orbital_position(rt, donor), orbital_position(rt, acceptor)
    sites = []
    for token, frag, k, site in ((donor, f, r, donor_site), (acceptor, g, s, acceptor_site)):
        t = rt.fragments[frag].tokens
        local = int(site) - int(rt.atom_offsets[frag])
        centres = {int(c) for c in t.orbitals[k].descriptors.all_center_atoms}
        if local not in centres:
            msg = f"site {site} is not a centre of token {token}"
            detail = {"token": int(token), "site": int(site), "centres": sorted(centres)}
            raise ServeError(Code.SITE_NOT_A_CENTRE, msg, detail=detail)
        sites.append(local)
    return i, j, f, r, sites[0], g, s, sites[1]


def _lobe(site: int, origin, direction, dirs: np.ndarray, valid: np.ndarray, M=None) -> dict:
    """`LobeOut` fields; M rotates the slot directions (placed frame)."""
    M = np.eye(3) if M is None else M
    tied = [tuple(float(x) for x in M @ dirs[p]) for p in (0, 1) if valid[p]]
    opposite = tuple(float(x) for x in M @ dirs[2]) if valid[2] else None
    return {
        "site": int(site),
        "origin": tuple(float(x) for x in origin),
        "direction": tuple(float(x) for x in direction),
        "tied": tied,
        "opposite": opposite,
    }


def _cap_out(rt: ReactantTokens, g: int, atoms, xyz_ang) -> dict:
    """`CapOut` fields of acceptor record atoms `atoms` at `xyz_ang`."""
    from ksmech.serve.geometry import atom_rows, bond_rows, molblock_subset

    record = rt.fragments[g].record
    atoms = [int(x) for x in atoms]
    return {
        "atoms": atom_rows(rt, g, atoms, xyz_ang),
        "bonds": bond_rows(record, atoms),
        "molblock": molblock_subset(record, atoms, xyz_ang),
        "local_ids": atoms,
    }


def _lobes(pl: Placement, t_F, t_G, sites) -> dict:
    donor_site, acceptor_site, a, b = sites
    x_a = np.asarray(t_F.atoms[a].position, dtype=float)
    x_b = np.asarray(t_G.atoms[b].position, dtype=float)
    placed_b = pl.cap_xyz_ang[list(pl.cap_atoms).index(b)]
    return {
        "donor_lobe": _lobe(donor_site, x_a, pl.u, pl.donor_dirs, pl.donor_valid),
        "acceptor_lobe_placed": _lobe(
            acceptor_site, placed_b, -pl.u, pl.acceptor_dirs, pl.acceptor_valid, pl.rot
        ),
        "acceptor_lobe_own_frame": _lobe(
            acceptor_site, x_b, -(pl.rot.T @ pl.u), pl.acceptor_dirs, pl.acceptor_valid
        ),
    }


def _context(rt: ReactantTokens, g: int, pl: Placement) -> dict:
    """Every acceptor atom under the cap's rigid motion (`CapOut` fields)."""
    t_G = rt.fragments[g].tokens
    xyz = np.array([x.position for x in t_G.atoms], dtype=float).reshape(-1, 3)
    return _cap_out(rt, g, range(len(xyz)), pl.placed(xyz))


def _checked(pl, t_F, r, a, t_G, s, b, cfg, intra, an_beta, is_contact) -> dict:
    checks = verify_placement(pl, t_F, r, a, t_G, s, b, cfg, intra, an_beta, is_contact)
    if not checks["consistent"]:
        msg = "the reconstructed placement does not reproduce the oriented kernel"
        raise ServeError(Code.PLACEMENT_MISMATCH, msg, status=500, detail=to_jsonable(checks))
    return checks


def _contact(rt_ctx: Runtime, state: PoolState, an, rt, ids) -> tuple[dict, ContactParts]:
    """The `ContactOut` fields (minus provenance, cube_urls, cubes) and the cube inputs."""
    from ksmech.physics.ks_terms import token_gap_kernel
    from ksmech.physics.params import KSParams
    from ksmech.serve.geometry import fragment_geometry

    donor, donor_site, acceptor, acceptor_site = (int(x) for x in ids)
    i, j, f, r, a, g, s, b = resolve_ids(rt, an, *ids)
    t_F, t_G = rt.fragments[f].tokens, rt.fragments[g].tokens
    cfg = rt_ctx.kernels.main.config.oriented
    pl = reconstruct_contact(t_F, r, a, t_G, s, b, cfg)
    is_contact = tuple(int(x) for x in an.contact_sites[i, j]) == (donor_site, acceptor_site)
    an_beta = float(an.beta[i, j]) if is_contact else None
    checks = _checked(pl, t_F, r, a, t_G, s, b, cfg, f == g, an_beta, is_contact)
    beta = KSParams().K * pl.beta1
    G = token_gap_kernel(t_F.orbitals[r], t_G.orbitals[s])
    body = {
        "state_id": state.state_id,
        "donor": donor,
        "donor_site": donor_site,
        "acceptor": acceptor,
        "acceptor_site": acceptor_site,
        "same_fragment": f == g,
        "is_contact_argmax": is_contact,
        "donor_fragment": fragment_geometry(rt, f),
        "cap": _cap_out(rt, g, pl.cap_atoms, pl.cap_xyz_ang),
        "acceptor_context": None if f == g else _context(rt, g, pl),
        "acceptor_context_note": None if f == g else CONTEXT_NOTE,
        **_lobes(pl, t_F, t_G, (donor_site, acceptor_site, a, b)),
        "distance_ang": float(cfg.distance),
        "clash_limit_ang": float(cfg.clash),
        "S": pl.S,
        "beta": float(beta),
        "clash": pl.clash,
        "tier": pl.tier,
        "roll_step": pl.roll,
        "roll_deg": 360.0 / _n_rolls() * pl.roll,
        "slots": pl.slots,
        "ks3_site": float(beta**2 * G),
        "G": float(G),
        "consistent": True,
        "checks": checks,
    }
    return to_jsonable(body), ContactParts(t_F, r, t_G, s, pl)


def _n_rolls() -> int:
    return int(oriented_module().N_ROLLS)


def contact_payload(
    rt_ctx: Runtime,
    state: PoolState,
    an: PoolAnalysis,
    donor: int,
    donor_site: int,
    acceptor: int,
    acceptor_site: int,
) -> dict:
    """The `ContactOut` fields minus provenance, cube_urls, cubes and iso_default (worker)."""
    from ksmech.serve.cubes import worker_tokens

    rt = worker_tokens(rt_ctx, state)
    body, _ = _contact(rt_ctx, state, an, rt, (donor, donor_site, acceptor, acceptor_site))
    return body


# ---- cache and task


def contact_key(state_id: str, ids, fp: str, digest: str) -> str:
    from ksmech.serve.cubes import CUBE_VERSION, DEFAULT_PADDING_ANG, DEFAULT_SPACING_ANG

    key = ["contact", state_id, *[int(x) for x in ids], fp, digest]
    key += [DEFAULT_SPACING_ANG, DEFAULT_PADDING_ANG, PLACEMENT_VERSION, CUBE_VERSION]
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:32]


def contact_path(s: ServeSettings, state_id: str, ids, fp: str, digest: str) -> Path:
    """`data/serve/contact/<sha>.json` of one (donor, donor_site, acceptor, acceptor_site)."""
    return s.contact / f"{contact_key(state_id, ids, fp, digest)}.json"


def contact_cubes(parts: ContactParts, header: str) -> dict[str, str]:
    """Both cube texts on one grid: the donor cap and the placed acceptor cap +- 4 Angstrom."""
    from ksmech.serve import cubes
    from ksmech.serve.upstream import oriented_module as _upstream

    cap_atoms = _upstream().cap_atoms

    t_F, t_G, pl = parts.t_F, parts.t_G, parts.pl
    z_F, x_F = cubes.fragment_arrays(t_F)
    donor_cap = x_F[cap_atoms(t_F.orbitals[parts.r], t_F)]
    g = cubes.grid_around(np.vstack([donor_cap, pl.cap_xyz_ang]))
    z_cap = np.array([t_G.atoms[int(i)].atomic_number for i in pl.cap_atoms], dtype=int)
    z_all, x_all = np.concatenate([z_F, z_cap]), np.vstack([x_F, pl.cap_xyz_ang])
    values = {
        "donor": cubes.lmo_values(t_F, parts.r, g),
        "acceptor": cubes.placed_cap_values(t_G, parts.s, pl, g),
    }
    note = "frame: donor record (Angstrom in, bohr out); acceptor = its placed cap only"
    return {
        which: cubes.write_cube(
            g, z_all, x_all, v, (f"{header} which={which}", f"{cubes.UNITS_COMMENT}; {note}")
        )
        for which, v in values.items()
    }


def t_contact(rt: Runtime, payload: dict) -> dict:
    """{"state", "donor", "donor_site", "acceptor", "acceptor_site"} -> {"path"}: the contact
    JSON (with "cube_paths") under data/serve/contact and both cubes under data/serve/cubes."""
    from ksmech.serve.analysis import analysis_digest, pool_work
    from ksmech.serve.cubes import CUBE_VERSION, worker_tokens
    from ksmech.serve.pools import atomic_write_text
    from ksmech.serve.versions import physics_fingerprint

    oriented_module()
    state = PoolState.from_dict(payload["state"])
    ids = tuple(int(payload[k]) for k in ("donor", "donor_site", "acceptor", "acceptor_site"))
    digest = analysis_digest(rt.loader, state)
    if digest is None:
        raise ServeError(Code.MISSING_TOKENS, f"no tokens for every fragment of {state.pool_key}")
    s = rt.settings
    path = contact_path(s, state.state_id, ids, physics_fingerprint(), digest)
    if path.is_file():
        return {"path": str(path)}
    work = pool_work(rt, state)
    body, parts = _contact(rt, state, work.an, worker_tokens(rt, state), ids)
    contact = "/".join(str(x) for x in ids)
    header = f"ksmech-serve cube v{CUBE_VERSION} state={state.state_id} contact={contact}"
    paths = {}
    for which, text in contact_cubes(parts, header).items():
        cube = s.cubes / f"{path.stem}-{which}.cube"
        paths[which] = str(atomic_write_text(cube, text, s))
    body |= {"cube_paths": paths, "placement_version": PLACEMENT_VERSION}
    body["created_at"] = datetime.now(UTC).isoformat()
    atomic_write_text(path, json.dumps(body, sort_keys=True) + "\n", s)
    return {"path": str(path)}
