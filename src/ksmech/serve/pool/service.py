"""API-side pool services (design 6.4): preflight, pools and tokens. Every function takes an
`ApiContext` and returns a `schemas` model.

Services never compute: the analysis (spec v9: the pooled gap factors of `serve.science.analysis`)
comes through `ApiContext.ensure_analysis` (202 Pending while the worker runs it). What runs
here is `np.load` of tokens and analyses and `ReactantTokens` correspondence.

Classifier labels (orbital type, token class, hybridization, aromaticity, functional groups)
appear only in `diagnostic` blocks; `model_input` holds exactly the descriptors the model
reads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from ksmech.serve.chem.pools import SpeciesInput, preflight_inputs, save_state
from ksmech.serve.chem.token_rows import diag_float as _diag_float
from ksmech.serve.chem.token_rows import fg_label as _fg_label
from ksmech.serve.chem.token_rows import jsonish as _jsonish
from ksmech.serve.chem.token_rows import orbital_model_input, parents_text
from ksmech.serve.compute.worker import task_key
from ksmech.serve.core import capabilities
from ksmech.serve.core import schemas as S
from ksmech.serve.core.contracts import PoolState
from ksmech.serve.core.errors import Code, ServeError, error_body
from ksmech.serve.science.analysis import (
    PoolAnalysis,
    analysis_digest,
    analysis_paths,
    current_failure,
    load_analysis,
)

if TYPE_CHECKING:
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.orbitals.tokens import MoleculeTokens, OrbitalToken
    from ksmech.serve.runtime.context import ApiContext

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
    """POST /pools: the root state of the input, saved, once its tokens and (with the
    predict capability) its pooled-gap analysis are ready (raises `Pending` while a task
    runs): the state the Correct page checks reaction matrices on (`POST /api/step/check`)."""
    out, pf = _preflight(ctx, body)
    if pf.error is not None:
        _raise_preflight(out, pf.error)
    save_state(pf.state, ctx.settings)
    ctx.ensure_tokens(pf.state)
    if capabilities.enabled("predict"):
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
    fingerprint and token digest, missing; disabled: tokens ready but the predict capability
    is off, so no analysis will run). Never computes."""
    state = ctx.state(state_id)
    physics = capabilities.enabled("predict")
    an = _ready_analysis(ctx, state) if physics else None
    rt = _tokens_or_none(ctx, state)
    settled = an is not None or (not physics and rt is not None)  # nothing left to fail
    failure = None if settled else current_failure(ctx.settings, ctx.loader, state, ctx.physics_fp)
    if an is not None:
        status = "ready"
    elif failure is not None:
        status = "failed"
    elif _running(ctx, state):
        status = "running"
    elif rt is not None and not physics:
        status = "disabled"
    else:
        status = "missing"
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


def token_class(o: OrbitalToken, t: MoleculeTokens) -> str:
    """Diagnostic description of a token, display only: the classifier type and the elements
    of the two leading centres."""
    centres = [a for a in o.descriptors.center_atoms[:2] if a >= 0]
    return f"{o.type}({'-'.join(t.symbol(int(a)) for a in centres)})"


def _orbital_diagnostic(k: _Tok) -> S.OrbitalDiagnosticOut:
    return S.OrbitalDiagnosticOut(
        type=str(k.o.type),
        token_class=token_class(k.o, k.t),
        parents=parents_text(k.t, k.o),
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
    from ksmech.serve.core.keys import species_key_of

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
    Featurizes first (202 while it runs; 409 POOL_NOT_ANALYZED when it failed); never needs
    the oriented analysis."""
    from ksmech.orbitals.descriptors import feature_names
    from ksmech.orbitals.tokens import atom_feature_names

    state = ctx.state(state_id)
    rt = ctx.ensure_tokens(state)
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
