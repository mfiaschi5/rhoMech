"""API-side pool services (design 6.4): preflight, pools, tokens, candidates, the physics panel
and the site-pair block. Every function takes an `ApiContext` and returns a `schemas` model.

Services never compute: the analysis comes through `ApiContext.ensure_analysis` (202 Pending
while the worker runs it), extra predictor scores through `ApiContext.scores`, and the
site-pair block through the `site_pairs` worker task. What runs here is `np.load` of tokens
and analyses, `ReactantTokens` correspondence and per-arrow numpy (`physics_view`).

Classifier labels (orbital type, token class, hybridization, aromaticity, functional groups)
appear only in `diagnostic` blocks; `model_input` holds exactly the descriptors the model
reads.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from ksmech.serve import physics_view as pv
from ksmech.serve import schemas as S
from ksmech.serve.analysis import (
    PoolAnalysis,
    analysis_digest,
    analysis_paths,
    current_failure,
    load_analysis,
)
from ksmech.serve.candidates import SCOPE_EXCLUDED, SELF_SITE_PAIR, reason_name, scope_mask
from ksmech.serve.contracts import PoolState, log_softmax_valid
from ksmech.serve.errors import Code, ServeError, error_body
from ksmech.serve.pools import SpeciesInput, preflight_inputs, save_state
from ksmech.serve.worker import task_key

if TYPE_CHECKING:
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.orbitals.tokens import MoleculeTokens, OrbitalToken
    from ksmech.serve.context import ApiContext

log = logging.getLogger(__name__)

SITE_PAIRS_TIMEOUT_S = 180


# ---- preflight and pools


def _inputs(body: S.NormalizeIn) -> list[SpeciesInput]:
    smiles = list(body.reactants_smiles)
    counts = [1] * len(smiles) if body.stoichiometry is None else list(body.stoichiometry)
    if len(counts) != len(smiles):
        msg = f"{len(counts)} stoichiometry entries for {len(smiles)} reactants"
        raise ServeError(Code.STOICHIOMETRY_MISMATCH, msg, status=422)
    return [SpeciesInput(s, int(n)) for s, n in zip(smiles, counts, strict=True)]


def _preflight(ctx: ApiContext, body: S.NormalizeIn):
    """(NormalizeOut, Preflight)."""
    pf = preflight_inputs(_inputs(body), ctx.featurizer, ctx.settings.limits)
    st = pf.state
    out = S.NormalizeOut(
        ok=pf.error is None,
        species=[S.SpeciesPreflightOut(**row) for row in pf.species],
        pool_key=None if st is None else st.pool_key,
        pool_id=None if st is None else st.pool_id,
        state_id=None if st is None else st.state_id,
        mapped_smiles=None if st is None else st.mapped_smiles,
        n_atoms=None if st is None else st.n_atoms,
    )
    return out, pf


def _raise_preflight(out: S.NormalizeOut, err: ServeError) -> None:
    """The first preflight error as a 422, with every species row in detail["normalize"]."""
    detail = dict(err.detail or {})
    detail["normalize"] = out.model_dump(mode="json")
    raise ServeError(err.code, err.message, status=422, upstream=err.upstream, detail=detail)


def normalize(ctx: ApiContext, body: S.NormalizeIn) -> S.NormalizeOut:
    """POST /normalize: the preflight of the input (never computes). The first error is a
    422 with its code (`detail.field`, and the full rows in `detail.normalize`)."""
    out, pf = _preflight(ctx, body)
    if pf.error is not None:
        _raise_preflight(out, pf.error)
    return out


def create_pool(ctx: ApiContext, body: S.NormalizeIn) -> S.PoolOut:
    """POST /pools: the root state of the input, saved, once its analysis is ready (raises
    `Pending` while featurize / analyze run)."""
    out, pf = _preflight(ctx, body)
    if pf.error is not None:
        _raise_preflight(out, pf.error)
    save_state(pf.state, ctx.settings)
    ctx.ensure_analysis(pf.state)
    return pool_out(ctx, pf.state.state_id)


def species_out(state: PoolState) -> list[S.SpeciesCountOut]:
    """Display SMILES come from the token keys (a species key can be a charge-separated
    resonance form)."""
    return [
        S.SpeciesCountOut(
            species_key=sp.species_key,
            token_keys=list(sp.token_keys),
            display_smiles=sp.token_keys[0] if sp.token_keys else sp.species_key,
            count=sp.count,
            charge=sp.charge,
            resonance_flag=sp.resonance_flag,
        )
        for sp in state.species
    ]


def _ready_analysis(ctx: ApiContext, state: PoolState) -> PoolAnalysis | None:
    digest = analysis_digest(ctx.loader, state)
    if digest is None:
        return None
    npz, _ = analysis_paths(ctx.settings, state.state_id, ctx.physics_fp, digest)
    if not npz.is_file():
        return None
    an = ctx.analysis_lru.get(str(npz))
    if an is None:
        an = load_analysis(npz)
        ctx.analysis_lru.put(str(npz), an)
    return an


def _running(ctx: ApiContext, state: PoolState) -> bool:
    keys = [task_key("analyze", {"state": state.to_dict()})]
    keys += [task_key("featurize", {"key": k}) for k in dict.fromkeys(state.fragment_token_keys)]
    return any(ctx.scheduler.find(k) is not None for k in keys)


def _failure_body(failure: dict) -> S.ErrorBody:
    message = str(failure.get("detail") or failure.get("code") or "analysis failed")
    body = error_body(
        str(failure.get("code") or Code.ANALYSIS_FAILED),
        message,
        upstream=failure.get("upstream"),
        detail=failure,
    )
    return S.ErrorBody(**body)


def _tokens_or_none(ctx: ApiContext, state: PoolState) -> ReactantTokens | None:
    try:
        return ctx.reactant_tokens(state)
    except ServeError:
        return None


def _pool_warnings(ctx: ApiContext, state: PoolState, rt) -> list[str]:
    out = [f"TOKENS_STALE: {k}" for k in state.fragment_token_keys if k in ctx.loader.stale]
    if rt is not None:
        for frag in rt.fragments:
            if frag.tokens.qm_warnings:
                out.append(f"{frag.key}: {frag.tokens.qm_warnings}")
    return list(dict.fromkeys(out))


def pool_out(ctx: ApiContext, state_id: str) -> S.PoolOut:
    """GET /pool/{state_id}: the state, and its analysis status (ready, running: a task with
    the analyze or featurize key is queued, failed: the failure sidecar of the current
    fingerprint and token digest, missing). Never computes."""
    state = ctx.state(state_id)
    an = _ready_analysis(ctx, state)
    rt = _tokens_or_none(ctx, state)
    failure = (
        None if an is not None else current_failure(ctx.settings, ctx.loader, state, ctx.physics_fp)
    )
    if an is not None:
        status = "ready"
    elif failure is not None:
        status = "failed"
    else:
        status = "running" if _running(ctx, state) else "missing"
    fermi = an.fermi_smeared if an is not None else None
    if fermi is None and rt is not None:
        fermi = any(f.tokens.fermi_smeared for f in rt.fragments)
    return S.PoolOut(
        state_id=state.state_id,
        pool_id=state.pool_id,
        pool_key=state.pool_key,
        mapped_smiles=state.mapped_smiles,
        species=species_out(state),
        n_atoms=state.n_atoms,
        n_orbitals=None if rt is None else rt.n_orbitals,
        analysis=status,
        analysis_error=None if failure is None else _failure_body(failure),
        fermi_smeared=fermi,
        warnings=_pool_warnings(ctx, state, rt),
        chemistry=ctx.chemistry,
    )


# ---- tokens


def _analyzed(ctx: ApiContext, state_id: str) -> tuple[PoolState, PoolAnalysis, ReactantTokens]:
    state = ctx.state(state_id)
    an = ctx.ensure_analysis(state)
    return state, an, ctx.reactant_tokens(state)


def _site_map(rt: ReactantTokens, site: int) -> int:
    idx = rt.mol_atom.get(int(site))
    return 0 if idx is None else int(rt.mol.GetAtomWithIdx(idx).GetAtomMapNum())


def _symbol(rt: ReactantTokens, site: int) -> str:
    f = int(rt.atom_fragment[int(site)])
    return rt.fragments[f].tokens.symbol(int(site) - int(rt.atom_offsets[f]))


def centres_out(rt: ReactantTokens, token: int) -> list[S.CentreOut]:
    return [
        S.CentreOut(site=a, map=_site_map(rt, a), element=_symbol(rt, a), weight=w)
        for a, w in rt.centres(int(token)).items()
    ]


def _overlap_sign(x: float) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _spectral_set(d) -> list[tuple[float, float]]:
    w = np.asarray(d.spectral_weights, dtype=float)
    e = np.asarray(d.spectral_energies, dtype=float)
    return [(float(e[k]), float(w[k])) for k in np.flatnonzero(w > 0)]


def orbital_model_input(o: OrbitalToken) -> S.OrbitalModelInputOut:
    """Exactly the model-input descriptors (`orbitals/descriptors.py`, `model_input=True`);
    the spectral set as (E_k, weight) rows with weight > 0, the overlap sign alongside."""
    d = o.descriptors
    return S.OrbitalModelInputOut(
        occupation=float(d.occupation),
        n_centers=int(d.n_centers),
        center_weights=[float(x) for x in np.asarray(d.center_weights).ravel()],
        center_spd=np.asarray(d.center_spd, dtype=float).tolist(),
        cos_a=float(d.cos_a),
        cos_b=float(d.cos_b),
        overlap_population=float(d.overlap_population),
        overlap_sign=_overlap_sign(float(d.overlap_population)),
        distance_ab=float(d.distance_ab),
        radius_gyration=float(d.radius_gyration),
        spectral_set=_spectral_set(d),
    )


@dataclass(frozen=True)
class _Tok:
    """An orbital token resolved in the pool's index space."""

    token_id: int
    fragment: int
    local: int
    o: OrbitalToken
    t: MoleculeTokens


def _tok(rt: ReactantTokens, token: int) -> _Tok:
    j = int(token) - rt.n_atoms
    f = int(rt.orbital_fragment[j])
    return _Tok(
        int(token), f, j - int(rt.orbital_offsets[f]), rt.orbitals[j], rt.fragments[f].tokens
    )


def _bookkeeping(rt: ReactantTokens, k: _Tok) -> S.OrbitalBookkeepingOut:
    d = k.o.descriptors
    w = np.asarray(d.spectral_weights, dtype=float)
    return S.OrbitalBookkeepingOut(
        energy_ev=float(d.energy_ev),
        centres=centres_out(rt, k.token_id),
        centroid=[float(x) for x in np.asarray(d.centroid).ravel()],
        spectral_indices=[int(d.spectral_indices[e]) for e in np.flatnonzero(w > 0)],
        retained_fraction=float(d.retained_fraction),
        n_block=len(d.full_energies),
        block=str(k.o.kind),
        fragment=k.fragment,
        local_position=k.local,
    )


def _masks(rt: ReactantTokens, k: _Tok) -> S.OrbitalMasksOut:
    from ksmech.orbitals.localize import DOUBLE_THRESHOLD

    j = k.token_id - rt.n_atoms
    occ = float(k.o.descriptors.occupation)
    occupied = bool(rt.occupied[j])
    return S.OrbitalMasksOut(
        donor_ok=bool(rt.donor_ok[j]),
        acceptor_ok=bool(rt.acceptor_ok[j]),
        d_type=bool(rt.d_type[j]),
        fractional=bool(rt.candidate.is_fractional(occ)),
        somo=occupied and occ < DOUBLE_THRESHOLD,
    )


def _diag_float(x: float) -> float | None:
    return None if x is None or not math.isfinite(float(x)) else float(x)


def _parents(k: _Tok) -> str:
    return ",".join(
        f"{k.t.symbol(int(a))}{int(a)}:{float(w):.2f}"
        for a, w in zip(k.o.parent_atoms, k.o.parent_weights, strict=False)
    )


def _orbital_diagnostic(k: _Tok) -> S.OrbitalDiagnosticOut:
    return S.OrbitalDiagnosticOut(
        type=str(k.o.type),
        token_class=pv.token_class(k.o, k.t),
        parents=_parents(k),
        s_share=_diag_float(k.o.s_share),
        p_share=_diag_float(k.o.p_share),
        d_share=_diag_float(k.o.d_share),
        classifier_overlap_population=_diag_float(k.o.overlap_population),
        metal_bound=bool(k.o.metal_bound),
    )


def orbital_out(rt: ReactantTokens, token: int) -> S.OrbitalTokenOut:
    k = _tok(rt, token)
    return S.OrbitalTokenOut(
        token_id=k.token_id,
        kind=str(k.o.kind),
        model_input=orbital_model_input(k.o),
        bookkeeping=_bookkeeping(rt, k),
        masks=_masks(rt, k),
        diagnostic=_orbital_diagnostic(k),
    )


def _jsonish(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonish(v) for v in value.tolist()]
    if isinstance(value, list | tuple):
        return [_jsonish(v) for v in value]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _fg_label(fg_id: int) -> str:
    from ksmech.orbitals.fgroups import default_vocab

    labels = default_vocab().labels
    return labels[fg_id] if 0 <= fg_id < len(labels) else f"fg{fg_id}"


def atom_out(rt: ReactantTokens, site: int) -> S.AtomTokenOut:
    """`model_input`: the AtomToken fields tagged model_input=True (in their order);
    `diagnostic`: f+ / f- (finite difference), hybridization, aromatic, the FG label."""
    from ksmech.orbitals.descriptors import MODEL_INPUT

    f = int(rt.atom_fragment[int(site)])
    local = int(site) - int(rt.atom_offsets[f])
    a = rt.fragments[f].tokens.atoms[local]
    model = {x.name: _jsonish(getattr(a, x.name)) for x in fields(a) if x.metadata[MODEL_INPUT]}
    diagnostic = {
        "f_plus": _jsonish(a.f_plus),
        "f_minus": _jsonish(a.f_minus),
        "hybridization": str(a.hybridization),
        "aromatic": bool(a.aromatic),
        "fg_label": _fg_label(int(a.fg_id)),
    }
    return S.AtomTokenOut(
        token_id=int(site),
        fragment=f,
        record_index=local,
        map=_site_map(rt, site),
        element=_symbol(rt, site),
        model_input=model,
        diagnostic=diagnostic,
    )


def _fragments_out(ctx: ApiContext, state: PoolState, rt: ReactantTokens) -> list:
    from ksmech.serve.keys import species_key_of

    return [
        S.FragmentOut(
            index=f,
            species_key=species_key_of(frag.key)[0],
            token_key=frag.key,
            atom_offset=int(rt.atom_offsets[f]),
            orbital_offset=rt.n_atoms + int(rt.orbital_offsets[f]),
            fermi_smeared=bool(frag.tokens.fermi_smeared),
            pm_converged=bool(frag.tokens.pm_converged),
            qm_warnings=str(frag.tokens.qm_warnings or ""),
            token_digest=str(ctx.loader.digest(frag.key) or ""),
        )
        for f, frag in enumerate(rt.fragments)
    ]


def tokens_out(ctx: ApiContext, state_id: str) -> S.TokensOut:
    """GET /pool/{state_id}/tokens: every atom and orbital token of the pool (global ids).
    Runs the analysis first (202 while it runs; 409 POOL_NOT_ANALYZED when it failed)."""
    from ksmech.orbitals.descriptors import feature_names
    from ksmech.orbitals.tokens import atom_feature_names

    state, _, rt = _analyzed(ctx, state_id)
    return S.TokensOut(
        state_id=state.state_id,
        n_atoms=rt.n_atoms,
        n_orbitals=rt.n_orbitals,
        orbital_feature_names=list(feature_names()),
        atom_feature_names=list(atom_feature_names()),
        fragments=_fragments_out(ctx, state, rt),
        atoms=[atom_out(rt, a) for a in range(rt.n_atoms)],
        orbitals=[orbital_out(rt, rt.n_atoms + j) for j in range(rt.n_orbitals)],
        provenance=S.Provenance(**ctx.provenance(None)),
    )


# ---- candidates


@dataclass(frozen=True)
class _Tuple:
    """One (donor, donor site, acceptor, acceptor site) tuple, before rendering."""

    i: int
    j: int
    a: int
    b: int
    valid: bool
    reason: str | None
    contact: bool
    weight: float  # w_a * w_b, the site-pair order within a cell


@dataclass
class _Scored:
    scores: np.ndarray  # (n_occ, n_virt) the predictor's values
    valid: np.ndarray  # pair_valid & scope
    logp: np.ndarray
    pair_rank: dict[tuple[int, int], int]


def _scored(scores: np.ndarray, valid: np.ndarray, temperature: float) -> _Scored:
    valid = np.asarray(valid, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    masked = np.where(valid, s, -np.inf)
    try:
        logp = log_softmax_valid(masked, valid, temperature)
    except ValueError as exc:
        raise ServeError(Code.REQUEST_INVALID, str(exc), status=422) from exc
    flat = np.flatnonzero(valid.ravel())
    order = flat[np.argsort(-masked.ravel()[flat], kind="stable")]
    n_virt = valid.shape[1]
    ranks = {(int(x // n_virt), int(x % n_virt)): r + 1 for r, x in enumerate(order)}
    return _Scored(s, valid, logp, ranks)


def _cell_tuples(rt, an, sc: _Scored, i: int, j: int, bonded: dict) -> list[_Tuple]:
    from ksmech.orbitals.tokens import bonded_matrix, site_pairs_excluded

    donor, acceptor = int(an.rows[i]), int(an.cols[j])
    d_c, a_c = rt.centres(donor), rt.centres(acceptor)
    pairs = [(a, b) for a in d_c for b in a_c]
    contact = tuple(int(x) for x in an.contact_sites[i, j])
    excluded = np.zeros(len(pairs), dtype=bool)
    f = int(rt.orbital_fragment[donor - rt.n_atoms])
    if bool(an.valid[i, j]) and f == int(rt.orbital_fragment[acceptor - rt.n_atoms]) and pairs:
        t, off = rt.fragments[f].tokens, int(rt.atom_offsets[f])
        if f not in bonded:
            bonded[f] = bonded_matrix(t)
        r, s = rt.orbital(donor), rt.orbital(acceptor)
        av, bv = [a - off for a, _ in pairs], [b - off for _, b in pairs]
        w = float(rt.candidate.site_weight)
        excluded = site_pairs_excluded(t, [r] * len(pairs), av, [s] * len(pairs), bv, w, bonded[f])
    base_reason = None if an.valid[i, j] else reason_name(int(an.reasons[i, j]))
    if base_reason is None and not sc.valid[i, j]:
        base_reason = SCOPE_EXCLUDED
    out = []
    for k, (a, b) in enumerate(pairs):
        reason = base_reason or (SELF_SITE_PAIR if excluded[k] else None)
        out.append(_Tuple(i, j, a, b, reason is None, reason, (a, b) == contact, d_c[a] * a_c[b]))
    return out


def donor_row(an: PoolAnalysis, donor: int) -> int:
    """Row i of an occupied orbital token, else ServeError(NOT_AN_ORBITAL_TOKEN)."""
    rows = np.asarray(an.rows)
    i = int(np.searchsorted(rows, int(donor)))
    if i >= len(rows) or int(rows[i]) != int(donor):
        msg = f"{donor} is not an occupied orbital token"
        raise ServeError(Code.NOT_AN_ORBITAL_TOKEN, msg, status=422)
    return i


def _tuples(rt, an, sc: _Scored, donor: int | None) -> list[_Tuple]:
    rows = range(len(an.rows)) if donor is None else [donor_row(an, donor)]
    bonded: dict = {}
    return [
        t for i in rows for j in range(len(an.cols)) for t in _cell_tuples(rt, an, sc, i, j, bonded)
    ]


def _sorted(tuples: list[_Tuple], sc: _Scored) -> list[_Tuple]:
    def key(t: _Tuple):
        if t.valid:
            return (0, sc.pair_rank[(t.i, t.j)], not t.contact, -t.weight, t.a, t.b)
        return (1, t.i, t.j, 0, 0.0, t.a, t.b)

    return sorted(tuples, key=key)


def _desc_brief(rt: ReactantTokens, token: int) -> S.OrbitalDescBrief:
    d = rt.orbital(int(token)).descriptors
    spd = np.asarray(d.center_spd, dtype=float)
    return S.OrbitalDescBrief(
        occupation=float(d.occupation),
        n_centers=int(d.n_centers),
        centres=centres_out(rt, token),
        spd0=[float(x) for x in spd[0]] if len(spd) else [],
        cos_a=float(d.cos_a),
        cos_b=float(d.cos_b),
        overlap_sign=_overlap_sign(float(d.overlap_population)),
        spectral_set=_spectral_set(d),
    )


class _Rows:
    """Renders tuples to `CandidateOut` dicts, caching per-token parts."""

    def __init__(self, rt: ReactantTokens, an: PoolAnalysis, sc: _Scored) -> None:
        from ksmech.serve.keys import species_key_of

        self.rt, self.an, self.sc = rt, an, sc
        self.species = [species_key_of(f.key)[0] for f in rt.fragments]
        self._desc: dict[int, S.OrbitalDescBrief] = {}
        self._class: dict[int, str] = {}

    def desc(self, token: int) -> S.OrbitalDescBrief:
        if token not in self._desc:
            self._desc[token] = _desc_brief(self.rt, token)
        return self._desc[token]

    def klass(self, token: int) -> str:
        if token not in self._class:
            k = _tok(self.rt, token)
            self._class[token] = pv.token_class(k.o, k.t)
        return self._class[token]

    def sp(self, token: int) -> str:
        return self.species[int(self.rt.orbital_fragment[token - self.rt.n_atoms])]

    def render(self, t: _Tuple, rank: int | None) -> dict:
        rt, an, sc = self.rt, self.an, self.sc
        donor, acceptor = int(an.rows[t.i]), int(an.cols[t.j])
        in_softmax = bool(sc.valid[t.i, t.j])
        score = float(sc.scores[t.i, t.j])
        elements = sorted({_symbol(rt, x) for x in rt.centres(acceptor)})
        return {
            "rank": rank,
            "pair_rank": sc.pair_rank.get((t.i, t.j)),
            "row": t.i,
            "col": t.j,
            "donor": donor,
            "donor_site": t.a,
            "acceptor": acceptor,
            "acceptor_site": t.b,
            "donor_site_map": _site_map(rt, t.a),
            "acceptor_site_map": _site_map(rt, t.b),
            "donor_site_element": _symbol(rt, t.a),
            "acceptor_site_element": _symbol(rt, t.b),
            "donor_species": self.sp(donor),
            "acceptor_species": self.sp(acceptor),
            "acceptor_centre_elements": elements,
            "is_contact_site_pair": t.contact,
            "contact_sites": [int(x) for x in an.contact_sites[t.i, t.j]],
            "score": score if math.isfinite(score) else None,
            "probability": float(np.exp(sc.logp[t.i, t.j])) if in_softmax else None,
            "ks1": float(an.ks1[t.i, t.j]) if an.ks1_ok[t.i, t.j] else None,
            "ks2": float(an.ks2[t.i, t.j]) if an.ks2_ok[t.i, t.j] else None,
            "ks3": float(an.ks3[t.i, t.j]),
            "clash": bool(an.clash[t.i, t.j]),
            "donor_desc": self.desc(donor),
            "acceptor_desc": self.desc(acceptor),
            "diagnostic": {
                "donor_class": self.klass(donor),
                "acceptor_class": self.klass(acceptor),
            },
            "valid": t.valid,
            "reason": t.reason,
        }


def _ranked(tuples: list[_Tuple], sc: _Scored) -> list[tuple[_Tuple, int | None]]:
    out, n = [], 0
    for t in _sorted(tuples, sc):
        n += t.valid
        out.append((t, n if t.valid else None))
    return out


def tuple_rows(
    rt: ReactantTokens,
    an: PoolAnalysis,
    scores: np.ndarray,
    valid: np.ndarray,
    temperature: float,
    donor: int | None,
    include_excluded: bool,
) -> list[dict]:
    """`CandidateOut` dicts of every (donor, site, acceptor, site) tuple (each centre of the
    donor x each centre of the acceptor of every cell), valid ones first by the predictor's
    rank (cell rank, the contact site pair first, then by centre weights), with the filter
    reason of the excluded ones on request. `valid` = pair_valid & scope."""
    sc = _scored(scores, valid, temperature)
    rows = _Rows(rt, an, sc)
    ranked = _ranked(_tuples(rt, an, sc, donor), sc)
    return [rows.render(t, r) for t, r in ranked if include_excluded or t.valid]


def _temperature_of(ctx: ApiContext, predictor: str, temperature: float | None) -> float:
    if temperature is None:
        return float(ctx.registry.info_of(predictor).default_temperature)
    t = float(temperature)
    if not t > 0 or not math.isfinite(t):
        raise ServeError(Code.REQUEST_INVALID, f"temperature must be > 0, got {temperature}")
    return t


def candidates_out(
    ctx: ApiContext,
    state_id: str,
    *,
    predictor: str = "floor",
    donor: int | None = None,
    include_excluded: bool = False,
    offset: int = 0,
    limit: int = 200,
    temperature: float | None = None,
    scope: str = "all",
) -> S.CandidatesOut:
    """GET /pool/{state_id}/candidates: the ranked tuples of the selected predictor."""
    state, an, rt = _analyzed(ctx, state_id)
    scores = ctx.scores(state, an, predictor)
    t = _temperature_of(ctx, predictor, temperature)
    valid = np.asarray(an.valid, dtype=bool) & scope_mask(rt, scope)
    if donor is not None:
        donor_row(an, donor)  # NOT_AN_ORBITAL_TOKEN before any work
    sc = _scored(scores, valid, t)
    ranked = _ranked(_tuples(rt, an, sc, donor), sc)
    counts = Counter(x.reason for x, _ in ranked if not x.valid)
    kept = [(x, r) for x, r in ranked if include_excluded or x.valid]
    offset, limit = max(0, int(offset)), max(0, int(limit))
    rows = _Rows(rt, an, sc)
    page = [S.CandidateOut(**rows.render(x, r)) for x, r in kept[offset : offset + limit]]
    return S.CandidatesOut(
        state_id=state.state_id,
        predictor=predictor,
        temperature=t,
        scope=scope,
        n_valid_pairs=int(valid.sum()),
        n_tuples=sum(x.valid for x, _ in ranked),
        total=len(kept),
        offset=offset,
        limit=limit,
        candidates=page,
        excluded_counts={str(k): int(v) for k, v in sorted(counts.items())},
        provenance=S.Provenance(**ctx.provenance(predictor, temperature=t, scope=scope)),
    )


# ---- physics panel and site pairs


def _default_site(rt: ReactantTokens, token: int, contact: int) -> int:
    if contact >= 0:
        return int(contact)
    from ksmech.serve.predictors import main_centre_site

    return main_centre_site(rt, token)


def _sites(rt, an, i: int, j: int, donor_site, acceptor_site) -> tuple[int, int]:
    donor, acceptor = int(an.rows[i]), int(an.cols[j])
    ca, cb = (int(x) for x in an.contact_sites[i, j])
    a = _default_site(rt, donor, ca) if donor_site is None else int(donor_site)
    b = _default_site(rt, acceptor, cb) if acceptor_site is None else int(acceptor_site)
    for role, token, site in (("donor", donor, a), ("acceptor", acceptor, b)):
        centres = rt.centres(token)
        if site not in centres:
            msg = f"{role} site {site} is not a centre of token {token} ({sorted(centres)})"
            detail = {"token": token, "site": site, "centres": sorted(centres)}
            raise ServeError(Code.SITE_NOT_A_CENTRE, msg, status=422, detail=detail)
    return a, b


def _cached_site_pairs(an: PoolAnalysis, payload: dict) -> dict | None:
    """The site-pair block the worker cached next to the analysis npz, or None."""
    if an.path is None:
        return None
    path = pv.site_pairs_path(
        an.path,
        payload["donor"],
        payload["acceptor"],
        payload.get("donor_site"),
        payload.get("acceptor_site"),
    )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:  # rewritten by the next site_pairs task
        log.warning("[%s] unreadable site-pair cache %s: %s", Code.CONSISTENCY_FAILED, path, exc)
        return None


def _site_pair(ctx: ApiContext, state: PoolState, an: PoolAnalysis, payload: dict) -> dict:
    """The `site_pairs` block of the payload: the cached JSON (as view3d reads cubes and
    contacts), else the worker task, which writes it."""
    cached = _cached_site_pairs(an, payload)
    if cached is not None:
        return cached
    label = f"site pairs of {payload['donor']} -> {payload['acceptor']} in {state.pool_key}"
    return ctx.interactive("site_pairs", payload, label=label, timeout_s=SITE_PAIRS_TIMEOUT_S)


def physics_out(
    ctx: ApiContext,
    state_id: str,
    *,
    donor: int,
    acceptor: int,
    donor_site: int | None = None,
    acceptor_site: int | None = None,
    predictor: str = "floor",
    temperature: float | None = None,
) -> S.PhysicsOut:
    """GET /pool/{state_id}/physics: the full panel of one arrow. Sites default to the
    contact sites; other sites add the `site_pairs` task's block (design 7.4)."""
    state, an, rt = _analyzed(ctx, state_id)
    i, j = an.cell(int(donor), int(acceptor))
    a, b = _sites(rt, an, i, j, donor_site, acceptor_site)
    scores = ctx.scores(state, an, predictor)
    t = _temperature_of(ctx, predictor, temperature)
    site_pair = None
    if (a, b) != tuple(int(x) for x in an.contact_sites[i, j]):
        payload = {"state": state.to_dict(), "donor": int(donor), "acceptor": int(acceptor)}
        payload |= {"donor_site": a, "acceptor_site": b}
        site_pair = _site_pair(ctx, state, an, payload)["site_pairs"][0]
    body = pv.arrow_physics(an, rt, i, j, a, b, scores, t, ctx.featurizer.occupations, site_pair)
    body["predictor"] = predictor
    body["provenance"] = ctx.provenance(predictor, temperature=t)
    return S.PhysicsOut(**body)


def site_pairs_out(ctx: ApiContext, state_id: str, *, donor: int, acceptor: int) -> S.SitePairsOut:
    """GET /pool/{state_id}/site_pairs: every centre x centre pair of (donor, acceptor) with
    its oriented contact, `is_contact` marking the argmax."""
    state, an, _ = _analyzed(ctx, state_id)
    an.cell(int(donor), int(acceptor))
    payload = {"state": state.to_dict(), "donor": int(donor), "acceptor": int(acceptor)}
    block = _site_pair(ctx, state, an, payload)
    return S.SitePairsOut(**block, provenance=S.Provenance(**ctx.provenance(None)))
