"""pool_service.py over the synthetic ApiContext (InlineScheduler): preflight, pools and their
analysis states, tokens (model input vs diagnostics), candidate tuples, the physics panel and
the site-pair block."""

import json
from dataclasses import fields

import numpy as np
import pytest

from ksmech.serve import pool_service as ps
from ksmech.serve import schemas as S
from ksmech.serve.errors import Code, ServeError
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    sn2_state,
    synthetic_context,
    synthetic_root,
    synthetic_settings,
)

pytestmark = pytest.mark.serve

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
    assert out.ok and out.state_id == sn2_state.state_id and out.pool_key == "CCl.C[O-]"
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
    assert out.analysis == "ready" and out.analysis_error is None
    assert out.n_orbitals == 27 and out.fermi_smeared is False and out.chemistry == "synthetic"
    assert state_path(out.state_id, synthetic_settings).is_file()
    assert [(s.species_key, s.display_smiles, s.count) for s in out.species] == [
        ("CCl", "CCl", 1),
        ("C[O-]", "C[O-]", 1),
    ]


def test_pool_out_missing_then_ready(synthetic_context, synthetic_settings):  # noqa: F811
    from ksmech.serve.pools import SpeciesInput, initial_state, save_state

    state = initial_state(
        [SpeciesInput("CCO", 1), SpeciesInput("CN", 1)], synthetic_settings.limits
    )
    save_state(state, synthetic_settings)
    before = ps.pool_out(synthetic_context, state.state_id)
    assert before.analysis == "missing" and before.analysis_error is None
    assert before.n_orbitals is not None  # the tokens exist, the analysis does not
    after = ps.create_pool(synthetic_context, _in("CCO", "CN"))
    assert after.state_id == state.state_id and after.analysis == "ready"


def test_pool_out_failed_sidecar(synthetic_context, synthetic_settings):  # noqa: F811
    from ksmech.serve.analysis import analysis_digest, clear_failure, write_failure
    from ksmech.serve.pools import SpeciesInput, initial_state, save_state

    ctx = synthetic_context
    state = initial_state(
        [SpeciesInput("CC(=O)OC", 1), SpeciesInput("CN", 1)], synthetic_settings.limits
    )
    save_state(state, synthetic_settings)
    err = ServeError(Code.ANALYSIS_FAILED, "boom", upstream="TEST")
    digest = analysis_digest(ctx.loader, state)
    write_failure(synthetic_settings, state, ctx.physics_fp, err, digest=digest, stage="analyze")
    try:
        out = ps.pool_out(ctx, state.state_id)
        assert out.analysis == "failed"
        assert out.analysis_error.code == Code.ANALYSIS_FAILED
        assert out.analysis_error.message == "boom" and out.analysis_error.upstream == "TEST"
        e = _raises(Code.POOL_NOT_ANALYZED, ps.tokens_out, ctx, state.state_id)
        assert e.status == 409 and e.detail["code"] == Code.ANALYSIS_FAILED
    finally:
        clear_failure(synthetic_settings, state.state_id, ctx.physics_fp)
    assert ps.pool_out(ctx, state.state_id).analysis == "missing"


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


def test_candidates_hold_the_sn2_tuple(synthetic_context, sn2_state):  # noqa: F811
    out = ps.candidates_out(synthetic_context, sn2_state.state_id, predictor="stub", limit=1000)
    (row,) = _sn2_row(out.candidates)
    assert row.valid and row.reason is None and row.rank == 1 and row.pair_rank == 1
    assert row.acceptor_centre_elements == ["C", "Cl"] and row.is_contact_site_pair
    assert (row.donor_site_element, row.acceptor_site_element) == ("O", "C")
    assert (row.donor_site_map, row.acceptor_site_map) == (2, 6)
    assert row.contact_sites == (1, 5) and row.score > 40 and 0 < row.probability <= 1
    assert row.donor_desc.n_centers == 1 and row.acceptor_desc.overlap_sign == -1
    assert row.diagnostic == {"donor_class": "lone_pair(O)", "acceptor_class": "sigma_star(C-Cl)"}
    floor = ps.candidates_out(synthetic_context, sn2_state.state_id, limit=1000)
    (frow,) = _sn2_row(floor.candidates)
    assert frow.valid and frow.score != row.score
    assert out.provenance.predictor == "stub" and floor.provenance.predictor == "floor"


def test_candidates_counts_and_order(synthetic_context, sn2_state):  # noqa: F811
    out = ps.candidates_out(synthetic_context, sn2_state.state_id, limit=10_000)
    assert out.n_valid_pairs == 98 and out.total == out.n_tuples == len(out.candidates)
    assert [r.rank for r in out.candidates] == list(range(1, out.total + 1))
    ranks = [r.pair_rank for r in out.candidates]
    assert ranks == sorted(ranks)
    cells = {(r.row, r.col): r.probability for r in out.candidates}
    assert len(cells) == 98 and sum(cells.values()) == pytest.approx(1.0, abs=1e-12)
    for r in out.candidates:
        assert r.acceptor_centre_elements == sorted(set(r.acceptor_centre_elements))
    first = {}
    for r in out.candidates:  # the contact site pair leads its cell
        first.setdefault((r.row, r.col), r)
    assert all(r.is_contact_site_pair for r in first.values())


def test_candidates_excluded_on_request(synthetic_context, sn2_state):  # noqa: F811
    ctx, sid = synthetic_context, sn2_state.state_id
    valid_only = ps.candidates_out(ctx, sid, limit=10_000)
    out = ps.candidates_out(ctx, sid, include_excluded=True, limit=10_000)
    assert out.n_tuples == valid_only.n_tuples
    assert out.total == out.n_tuples + sum(out.excluded_counts.values())
    assert out.excluded_counts == valid_only.excluded_counts
    reasons = {r.reason for r in out.candidates if not r.valid}
    assert {"ACCEPTOR_D_TYPE", "OWN_ANTIBOND", "SELF_SITE_PAIR"} <= reasons
    assert all(r.rank is None for r in out.candidates if not r.valid)
    assert all(r.valid for r in out.candidates[: out.n_tuples])


def test_candidates_paging_donor_and_scope(synthetic_context, sn2_state):  # noqa: F811
    ctx, sid = synthetic_context, sn2_state.state_id
    ten = ps.candidates_out(ctx, sid, limit=10).candidates
    a = ps.candidates_out(ctx, sid, limit=5).candidates
    b = ps.candidates_out(ctx, sid, offset=5, limit=5).candidates
    assert [r.model_dump() for r in a + b] == [r.model_dump() for r in ten]
    rows = ps.candidates_out(ctx, sid, donor=SN2_DONOR, limit=1000).candidates
    assert rows and {r.donor for r in rows} == {SN2_DONOR}
    e = _raises(Code.NOT_AN_ORBITAL_TOKEN, ps.candidates_out, ctx, sid, donor=SN2_ACCEPTOR)
    assert e.status == 422
    inter = ps.candidates_out(ctx, sid, scope="inter", include_excluded=True, limit=10_000)
    assert inter.provenance.candidate_scope == "inter"
    assert all(r.donor_species != r.acceptor_species for r in inter.candidates if r.valid)
    assert inter.excluded_counts.get("SCOPE_EXCLUDED", 0) > 0
    _raises(Code.REQUEST_INVALID, ps.candidates_out, ctx, sid, temperature=-1.0)


def test_candidates_gbdt_missing_is_409(synthetic_context, sn2_state):  # noqa: F811
    e = _raises(
        Code.PREDICTOR_UNAVAILABLE,
        ps.candidates_out,
        synthetic_context,
        sn2_state.state_id,
        predictor="gbdt",
    )
    assert e.status == 409 and e.detail["code"] == Code.GBDT_MISSING
    _raises(
        Code.PREDICTOR_UNKNOWN,
        ps.candidates_out,
        synthetic_context,
        sn2_state.state_id,
        predictor="nope",
    )


def test_tuple_rows_matches_candidates(synthetic_context, sn2_state):  # noqa: F811
    from ksmech.serve.candidates import scope_mask

    ctx = synthetic_context
    an = ctx.ensure_analysis(sn2_state)
    rt = ctx.reactant_tokens(sn2_state)
    valid = np.asarray(an.valid) & scope_mask(rt, "all")
    rows = ps.tuple_rows(rt, an, an.score_floor, valid, 1.0, None, False)
    page = ps.candidates_out(ctx, sn2_state.state_id, limit=10_000).candidates
    assert [S.CandidateOut(**r).model_dump() for r in rows] == [r.model_dump() for r in page]


# ---- physics panel and site pairs


def test_physics_out_at_contact_sites(synthetic_context, sn2_state):  # noqa: F811
    out = ps.physics_out(
        synthetic_context, sn2_state.state_id, donor=SN2_DONOR, acceptor=SN2_ACCEPTOR
    )
    assert (out.donor_site, out.acceptor_site) == (1, 5) and out.contact_sites == (1, 5)
    assert out.requested_sites_are_contact and out.site_pair is None
    assert out.predictor == "floor" and out.provenance.predictor == "floor"
    assert out.provenance.temperature == 1.0 and out.contact_distance_ang == 2.3
    assert out.pair_valid and out.ks2_detail.form == "contact"
    assert out.dominant_pair.index_source == "qm_occupations"
    assert out.score == out.floor and out.rank is not None and 0 < out.probability < 1
    assert out.frozen_head.weight is not None and out.frozen_head.tau == 1.0
    assert out.ks3 == pytest.approx(out.beta2 * out.G, rel=1e-12)


def test_physics_out_at_user_sites(synthetic_context, sn2_state):  # noqa: F811
    out = ps.physics_out(
        synthetic_context,
        sn2_state.state_id,
        donor=SN2_DONOR,
        acceptor=SN2_ACCEPTOR,
        acceptor_site=6,
        predictor="stub",
    )
    assert not out.requested_sites_are_contact and out.acceptor_site == 6
    sp = out.site_pair
    assert sp is not None and (sp.donor_site, sp.acceptor_site) == (1, 6) and not sp.is_contact
    assert sp.ks3_site == pytest.approx(sp.beta**2 * out.G_exact, rel=1e-12)
    assert (
        out.ks3
        == ps.physics_out(
            synthetic_context, sn2_state.state_id, donor=SN2_DONOR, acceptor=SN2_ACCEPTOR
        ).ks3
    )  # the token pair's physics does not depend on the requested sites
    assert out.predictor == "stub" and out.score > out.floor


def test_physics_out_errors(synthetic_context, sn2_state):  # noqa: F811
    ctx, sid = synthetic_context, sn2_state.state_id
    e = _raises(
        Code.SITE_NOT_A_CENTRE,
        ps.physics_out,
        ctx,
        sid,
        donor=SN2_DONOR,
        acceptor=SN2_ACCEPTOR,
        donor_site=0,
    )
    assert e.status == 422
    _raises(Code.NOT_AN_ORBITAL_TOKEN, ps.physics_out, ctx, sid, donor=3, acceptor=SN2_ACCEPTOR)
    _raises(
        Code.NOT_AN_ORBITAL_TOKEN, ps.physics_out, ctx, sid, donor=SN2_ACCEPTOR, acceptor=SN2_DONOR
    )


def test_physics_out_of_an_invalid_pair(synthetic_context, sn2_state):  # noqa: F811
    out = ps.physics_out(synthetic_context, sn2_state.state_id, donor=25, acceptor=SN2_ACCEPTOR)
    assert not out.pair_valid and out.reason == "OWN_ANTIBOND"
    assert out.floor_rank is None and out.score is None and out.rank is None
    assert "pair_invalid:OWN_ANTIBOND" in out.flags and out.frozen_head.weight is None


def test_site_pairs_out(synthetic_context, sn2_state):  # noqa: F811
    out = ps.site_pairs_out(
        synthetic_context, sn2_state.state_id, donor=SN2_DONOR, acceptor=SN2_ACCEPTOR
    )
    assert isinstance(out, S.SitePairsOut) and out.state_id == sn2_state.state_id
    assert [(x.donor_site, x.acceptor_site, x.is_contact) for x in out.site_pairs] == [
        (1, 5, True),
        (1, 6, False),
    ]
    assert out.note and out.G < 0
    _raises(
        Code.NOT_AN_ORBITAL_TOKEN,
        ps.site_pairs_out,
        synthetic_context,
        sn2_state.state_id,
        donor=3,
        acceptor=SN2_ACCEPTOR,
    )


def test_site_pairs_are_cached_next_to_the_analysis(synthetic_context, sn2_state, monkeypatch):  # noqa: F811
    from ksmech.serve import physics_view as pv

    ctx, sid = synthetic_context, sn2_state.state_id
    an = ctx.ensure_analysis(sn2_state)
    one = pv.site_pairs_path(an.path, SN2_DONOR, SN2_ACCEPTOR, 1, 6)
    every = pv.site_pairs_path(an.path, SN2_DONOR, SN2_ACCEPTOR, None, None)
    assert one.parent == an.path.parent and one != every
    assert one.name.startswith("site-pairs-") and len(one.name) == len("site-pairs-.json") + 16
    for path in (one, every):
        path.unlink(missing_ok=True)
    submitted = []
    real_submit = ctx.scheduler.submit

    def counting_submit(name, payload, **kw):
        submitted.append(name)
        return real_submit(name, payload, **kw)

    monkeypatch.setattr(ctx.scheduler, "submit", counting_submit)
    kw = dict(donor=SN2_DONOR, acceptor=SN2_ACCEPTOR, acceptor_site=6)
    first = ps.physics_out(ctx, sid, **kw)
    assert submitted.count("site_pairs") == 1 and one.is_file()
    again = ps.physics_out(ctx, sid, **kw)
    assert submitted.count("site_pairs") == 1  # read from site-pairs-*.json: no task
    assert again.site_pair == first.site_pair and again.site_pair.acceptor_site == 6
    assert json.loads(one.read_text())["site_pairs"][0] == first.site_pair.model_dump()
    block = ps.site_pairs_out(ctx, sid, donor=SN2_DONOR, acceptor=SN2_ACCEPTOR)
    assert submitted.count("site_pairs") == 2 and every.is_file()
    cached = ps.site_pairs_out(ctx, sid, donor=SN2_DONOR, acceptor=SN2_ACCEPTOR)
    assert submitted.count("site_pairs") == 2 and cached == block
    assert block.provenance is not None and block.provenance.predictor == "none"
    assert block.provenance.spec_version == ctx.spec_version
