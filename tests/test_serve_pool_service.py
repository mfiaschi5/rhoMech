"""pool_service.py over the synthetic ApiContext (InlineScheduler): preflight, pools and their
analysis states, tokens (model input vs diagnostics), candidate tuples, the physics panel and
the site-pair block."""

import json
from dataclasses import fields

import pytest

from ksmech.serve import pool_service as ps
from ksmech.serve import schemas as S
from ksmech.serve.capabilities import CAPABILITIES
from ksmech.serve.errors import Code, ServeError
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    sn2_state,
    synthetic_context,
    synthetic_root,
    synthetic_settings,
)

pytestmark = pytest.mark.serve

PHYSICS = CAPABILITIES["predict"][0]  # the pool analysis serves predict and correct
ANALYZED = "ready" if PHYSICS else "disabled"  # pool status once its tokens exist
SN2_DONOR, SN2_ACCEPTOR = 16, 28  # n(O) -> sigma*(C-Cl); contact sites (O, C) = (1, 5)


def _in(*smiles, counts=None):
    return S.NormalizeIn(reactants_smiles=list(smiles), stoichiometry=counts)


def _raises(code, fn, *args, **kw) -> ServeError:
    with pytest.raises(ServeError) as e:
        fn(*args, **kw)
    assert e.value.code == code, (e.value.code, e.value.message)
    return e.value


# ---- normalize and pools


def test_normalize_success(synthetic_context, sn2_state):  # noqa: F811
    out = ps.normalize(synthetic_context, _in("C[O-]", "CCl"))
    assert (
        out.ok and out.state_id == sn2_state.state_id and out.pool_key == "[CH3][Cl]|0.[CH3][O]|-1"
    )
    assert out.n_atoms == 10 and out.mapped_smiles == sn2_state.mapped_smiles
    assert [r.token_key for r in out.species] == ["C[O-]", "CCl"]
    assert all(r.tokens_cached and r.qm_cached and not r.errors for r in out.species)


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (_in("C[O-", "CCl"), Code.INVALID_SMILES),
        (_in("[CH3]"), Code.RADICAL_NOT_SUPPORTED),
        (_in("C[O-]", "CCl", counts=[1]), Code.STOICHIOMETRY_MISMATCH),
        (_in("C[O-]", counts=[9]), Code.LIMIT_EXCEEDED),
        (_in(""), Code.EMPTY_INPUT),
    ],
)
def test_normalize_errors(synthetic_context, body, code):  # noqa: F811
    err = _raises(code, ps.normalize, synthetic_context, body)
    assert err.status == 422
    if code != Code.STOICHIOMETRY_MISMATCH:
        rows = err.detail["normalize"]["species"]
        assert err.detail["normalize"]["ok"] is False and isinstance(rows, list)


def test_normalize_rows_carry_the_field_error(synthetic_context):  # noqa: F811
    err = _raises(Code.INVALID_SMILES, ps.normalize, synthetic_context, _in("CCl", "C[O-"))
    assert err.detail["field"] == 1
    rows = err.detail["normalize"]["species"]
    assert rows[0]["errors"] == [] and rows[1]["errors"][0]["code"] == Code.INVALID_SMILES


def test_create_pool_is_ready_and_saved(synthetic_context, sn2_state, synthetic_settings):  # noqa: F811
    from ksmech.serve.pools import state_path

    out = ps.create_pool(synthetic_context, _in("C[O-]", "CCl"))
    assert isinstance(out, S.PoolOut) and out.state_id == sn2_state.state_id
    swapped = ps.create_pool(synthetic_context, _in("CCl", "C[O-]"))  # other maps, same pool
    assert swapped.pool_id == out.pool_id and swapped.state_id != out.state_id
    assert out.analysis == ANALYZED and out.analysis_error is None
    assert out.n_orbitals == 27 and out.fermi_smeared is False and out.chemistry == "synthetic"
    assert state_path(out.state_id, synthetic_settings).is_file()
    assert [(s.species_key, s.display_smiles, s.count) for s in out.species] == [
        ("[CH3][Cl]|0", "CCl", 1),
        ("[CH3][O]|-1", "C[O-]", 1),
    ]


def test_pool_out_missing_then_ready(synthetic_context, synthetic_settings):  # noqa: F811
    from ksmech.serve.pools import SpeciesInput, initial_state, save_state

    state = initial_state(
        [SpeciesInput("CCO", 1), SpeciesInput("CN", 1)], synthetic_settings.limits
    )
    save_state(state, synthetic_settings)
    before = ps.pool_out(synthetic_context, state.state_id)
    assert before.analysis == ("missing" if PHYSICS else "disabled")
    assert before.analysis_error is None
    assert before.n_orbitals is not None  # the tokens exist, the analysis does not
    after = ps.create_pool(synthetic_context, _in("CCO", "CN"))
    assert after.state_id == state.state_id and after.analysis == ANALYZED


def test_featurize_failure_without_tokens(synthetic_context, synthetic_settings, monkeypatch):  # noqa: F811
    """A recorded featurize failure (digest None: the tokens are missing) is `failed` and makes
    `ensure_tokens` a 409 POOL_NOT_ANALYZED without submitting the task again."""
    from ksmech.serve.analysis import clear_failure, write_failure
    from ksmech.serve.pools import SpeciesInput, initial_state, save_state

    ctx = synthetic_context
    state = initial_state([SpeciesInput("CCCCC", 1)], synthetic_settings.limits)
    save_state(state, synthetic_settings)
    assert not ctx.loader.has(state.fragment_token_keys[0])
    err = ServeError(Code.SYNTHETIC_FAILED, "no tokens", upstream="TEST")
    write_failure(synthetic_settings, state, ctx.physics_fp, err, digest=None, stage="featurize")

    def submitted(*args, **kwargs):
        raise AssertionError("a task was submitted")

    monkeypatch.setattr(ctx, "interactive", submitted)
    try:
        out = ps.pool_out(ctx, state.state_id)
        assert out.analysis == "failed" and out.analysis_error.code == Code.SYNTHETIC_FAILED
        e = _raises(Code.POOL_NOT_ANALYZED, ctx.ensure_tokens, state)
        assert e.status == 409 and e.detail["stage"] == "featurize"
    finally:
        clear_failure(synthetic_settings, state.state_id, ctx.physics_fp)


def test_pool_out_unknown_state(synthetic_context):  # noqa: F811
    e = _raises(Code.POOL_NOT_FOUND, ps.pool_out, synthetic_context, "s" + "0" * 20)
    assert e.status == 404
    _raises(Code.POOL_NOT_FOUND, ps.pool_out, synthetic_context, "../etc")


# ---- tokens


@pytest.fixture(scope="module")
def tokens(synthetic_context, sn2_state):  # noqa: F811
    return ps.tokens_out(synthetic_context, sn2_state.state_id)


def test_tokens_model_input_is_exactly_the_descriptors(tokens):
    from ksmech.orbitals.descriptors import MODEL_INPUT, SPECTRAL_FIELDS, OrbitalDescriptors
    from ksmech.orbitals.tokens import AtomToken

    names = {f.name for f in fields(OrbitalDescriptors) if f.metadata.get(MODEL_INPUT)}
    keys = set(S.OrbitalModelInputOut.model_fields)
    assert keys - {"spectral_set", "overlap_sign"} == names - set(SPECTRAL_FIELDS)
    atom_inputs = {f.name for f in fields(AtomToken) if f.metadata[MODEL_INPUT]}
    dumped = tokens.model_dump(mode="json")
    for o in dumped["orbitals"]:
        assert set(o["model_input"]) == keys
        assert "type" not in o["model_input"] and o["diagnostic"]["diagnostic"] is True
    for a in dumped["atoms"]:
        assert set(a["model_input"]) == atom_inputs
        assert not set(a["model_input"]) & set(a["diagnostic"])
        assert {"hybridization", "aromatic", "fg_label"} <= set(a["diagnostic"])


def test_tokens_values_and_layout(tokens, synthetic_context, sn2_state):  # noqa: F811
    from ksmech.orbitals.descriptors import feature_names

    rt = synthetic_context.reactant_tokens(sn2_state)
    assert (tokens.n_atoms, tokens.n_orbitals) == (10, 27)
    assert [a.token_id for a in tokens.atoms] == list(range(10))
    assert [o.token_id for o in tokens.orbitals] == list(range(10, 37))
    assert tokens.orbital_feature_names == feature_names()
    o = tokens.orbitals[SN2_DONOR - 10]
    d = rt.orbital(SN2_DONOR).descriptors
    assert o.kind == "occupied" and o.model_input.occupation == float(d.occupation)
    assert o.model_input.n_centers == 1 and o.model_input.overlap_sign == 0
    assert [c.site for c in o.bookkeeping.centres] == [1] and o.bookkeeping.centres[0].map == 2
    assert o.masks.donor_ok and not o.masks.acceptor_ok and not o.masks.somo
    assert sum(w for _, w in o.model_input.spectral_set) == pytest.approx(1.0)
    acc = tokens.orbitals[SN2_ACCEPTOR - 10]
    assert acc.model_input.overlap_sign == -1 and acc.masks.acceptor_ok
    assert acc.diagnostic.token_class == "sigma_star(C-Cl)"
    assert [f.token_key for f in tokens.fragments] == ["C[O-]", "CCl"]
    assert [f.orbital_offset for f in tokens.fragments] == [10, 21]
    d_type = [x for x in tokens.orbitals if x.masks.d_type]
    assert d_type and all(not x.masks.acceptor_ok for x in d_type)
    assert tokens.provenance.chemistry == "synthetic" and tokens.provenance.predictor == "none"


def test_tokens_nan_is_null(tokens):
    text = tokens.model_dump_json()
    assert "NaN" not in text and "Infinity" not in text
    dumped = json.loads(text)
    duals = [a["model_input"]["dual"] for a in dumped["atoms"]]
    assert None in duals  # the synthetic records carry no finite-difference dual descriptor


# ---- candidates


def _sn2_row(rows):
    return [
        r
        for r in rows
        if (r.donor, r.donor_site, r.acceptor, r.acceptor_site) == (SN2_DONOR, 1, SN2_ACCEPTOR, 5)
    ]


# ---- physics panel and site pairs
