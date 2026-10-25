"""expansion.py on MeO- + MeCl (synthetic chemistry, floor and stub predictors): children come
from `propose`, the stub's SN2 child, the scan masses, persisted child states, complete arrow
briefs, counted failures, a user step onto sigma*(C-H), and the task bodies through
InlineScheduler."""

import math

import numpy as np
import pytest

from ksmech.serve import expansion
from ksmech.serve import stepapply as sa
from ksmech.serve.contracts import (
    ArrowSpec,
    ExpandRequest,
    Expansion,
    ProposedStep,
    ScoringContext,
    StepOutcome,
)
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.pools import load_state, state_path
from ksmech.serve.schemas import AlternativeOut, ArrowOut, ArrowPhysicsBrief, PredictorViewOut
from ksmech.serve.testing import requires
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    sn2_state,
    synthetic_root,
    synthetic_runtime,
    synthetic_settings,
)

pytestmark = [pytest.mark.serve, requires("predict")]

SN2 = ArrowSpec(16, 1, 28, 5)
CH_ACCEPTOR = ArrowSpec(16, 1, 34, 5)  # n(O) -> sigma*(C-H9) at site C: expels H-
NO_PAIR = ArrowSpec(12, 2, 20, 0)  # H2 has no pair beyond its C0 bond: APPLY_FAILED:VALENCE


def _req(predictor="stub", beam=3, temperature=None, scope="all", max_scan=64):
    return ExpandRequest(predictor, beam, temperature, scope, max_scan)


@pytest.fixture(scope="module")
def stub_ex(synthetic_runtime, sn2_state):  # noqa: F811
    return expansion.expand_state(synthetic_runtime, sn2_state, _req())


@pytest.fixture(scope="module")
def floor_ex(synthetic_runtime, sn2_state):  # noqa: F811
    return expansion.expand_state(synthetic_runtime, sn2_state, _req("floor", beam=4))


def _all_arrows(ex: Expansion):
    for child in ex.children:
        yield from child.arrows
        for alt in child.alternatives:
            yield from alt["arrows"]


def test_stub_top_child_is_the_sn2(stub_ex):
    assert stub_ex.ok and stub_ex.code is None
    top = stub_ex.children[0]
    assert top.state.pool_key == "COC.[Cl-]" and top.rank == 1
    (arrow,) = top.arrows
    assert (arrow["donor_site"], arrow["acceptor_site"]) == (1, 5)
    assert (arrow["donor_element"], arrow["acceptor_element"]) == ("O", "C")
    assert arrow["site_source"] == "rule" and arrow["source_kind"] == "lone_pair"
    assert top.encoded["code"] == "2=6;6,7=7" and top.encode_reason is None
    assert top.touched_maps == [2, 6, 7]


def test_masses_and_probabilities(stub_ex, floor_ex):
    for ex in (stub_ex, floor_ex):
        assert ex.scanned_mass + ex.unscanned_mass == pytest.approx(1.0, abs=1e-12)
        assert sum(c.probability for c in ex.children) <= ex.scanned_mass + 1e-12
        probs = [c.probability for c in ex.children]
        assert probs == sorted(probs, reverse=True)
        for c in ex.children:
            assert c.logp == pytest.approx(math.log(c.probability))
            assert c.probability + ex.unscanned_mass <= 1.0 + 1e-12
        assert 0 < ex.n_scanned <= 64 and ex.n_valid == 98
        assert ex.detail.startswith("scan stopped: ")
    assert len(floor_ex.children) == 4 and len(stub_ex.children) == 3


def test_child_states_are_saved(stub_ex, floor_ex, synthetic_settings):  # noqa: F811
    for ex in (stub_ex, floor_ex):
        for child in ex.children:
            assert state_path(child.state.state_id, synthetic_settings).is_file()
            assert load_state(child.state.state_id, synthetic_settings) == child.state


def test_every_arrow_carries_a_complete_brief(stub_ex, floor_ex):
    for ex in (stub_ex, floor_ex):
        for arrow in _all_arrows(ex):
            # site_pair (optional, default None) is attached to user-applied arrows only
            assert set(arrow) <= set(ArrowOut.model_fields)
            assert set(ArrowOut.model_fields) - set(arrow) <= {"site_pair"}
            ArrowOut.model_validate(arrow)
            brief = ArrowPhysicsBrief.model_validate(arrow["physics"])
            assert set(arrow["physics"]) == set(ArrowPhysicsBrief.model_fields)
            assert brief.dominant is not None and brief.contact_distance_ang == 2.3
            assert brief.ks2_form in ("contact", "difference", "undefined") and brief.tau == 1.0
        for child in ex.children:
            for alt in child.alternatives:
                AlternativeOut.model_validate(alt)
            assert child.n_alternatives >= len(child.alternatives)


def test_floor_children_follow_the_contact_sites(floor_ex, synthetic_runtime, sn2_state):  # noqa: F811
    from ksmech.serve.analysis import pool_work

    an = pool_work(synthetic_runtime, sn2_state).an
    for child in floor_ex.children:
        (arrow,) = child.arrows
        i, j = an.cell(arrow["donor"], arrow["acceptor"])
        assert arrow["site_source"] == "contact"
        assert [arrow["donor_site"], arrow["acceptor_site"]] == an.contact_sites[i, j].tolist()
        assert arrow["physics"]["floor"] == float(an.floor[i, j])
        assert child.score == float(an.floor[i, j])  # the floor's score is the floor


def test_children_come_from_propose(synthetic_runtime, sn2_state, monkeypatch):  # noqa: F811
    stub = synthetic_runtime.scorers["stub"]
    hand = ProposedStep((CH_ACCEPTOR,), ("rule",), 7.0, math.log(0.25), 1, (6, 10))
    seen = {}

    def propose(pool, ctx, k):
        seen["k"], seen["T"] = k, pool.temperature
        return [hand]

    monkeypatch.setattr(stub, "propose", propose)
    ex = expansion.expand_state(synthetic_runtime, sn2_state, _req(temperature=2.0))
    assert seen == {"k": 64, "T": 2.0}
    (child,) = ex.children
    (arrow,) = child.arrows
    got = [arrow["donor"], arrow["donor_site"], arrow["acceptor"], arrow["acceptor_site"]]
    assert got == CH_ACCEPTOR.as_list() and arrow["site_source"] == "rule"
    assert child.state.pool_key == "COCCl.[H-]"
    assert (child.score, child.rank) == (7.0, 1)
    assert child.probability == pytest.approx(0.25) and ex.scanned_mass == pytest.approx(0.25)
    assert ex.unscanned_mass == pytest.approx(0.75) and ex.n_scanned == 1


def test_failures_are_counted(synthetic_runtime, sn2_state, monkeypatch):  # noqa: F811
    stub = synthetic_runtime.scorers["stub"]
    steps = [
        ProposedStep((NO_PAIR,), ("contact",), 3.0, math.log(0.2), 1, None),
        ProposedStep((SN2,), ("rule",), 2.0, math.log(0.3), 2, None),
        ProposedStep((ArrowSpec(15, 1, 28, 5),), ("rule",), 1.0, math.log(0.1), 3, None),
    ]
    monkeypatch.setattr(stub, "propose", lambda pool, ctx, k: steps)
    real = sa.apply_step

    def apply_step(rt, arrows):  # the third proposal: pretend it only moved a resonance form
        out = real(rt, arrows)
        if arrows[0].donor == 15:
            out = sa.ApplyOutcome(True, None, None, "", out.product, sn2_state)
        return out

    monkeypatch.setattr(sa, "apply_step", apply_step)
    ex = expansion.expand_state(synthetic_runtime, sn2_state, _req(beam=2))
    assert ex.failures == {"APPLY_FAILED:VALENCE": 1, "RESONANCE_ONLY_STEP": 1}
    assert [c.state.pool_key for c in ex.children] == ["COC.[Cl-]"]
    assert ex.children[0].rank == 2 and ex.n_scanned == 3
    assert ex.scanned_mass == pytest.approx(0.6)
    codes = [(f["rank"], f["code"]) for f in ex.failed_examples]
    assert codes == [(1, "APPLY_FAILED:VALENCE"), (3, "RESONANCE_ONLY_STEP")]
    assert ex.failed_examples[0]["probability"] == pytest.approx(0.2)
    assert ex.failed_examples[0]["arrows"][0]["physics"] is not None


def test_children_merge_by_pool_key(synthetic_runtime, sn2_state, monkeypatch):  # noqa: F811
    stub = synthetic_runtime.scorers["stub"]
    second = ArrowSpec(15, 1, 28, 5)  # the other O lone pair: the same child
    steps = [
        ProposedStep((SN2,), ("rule",), 2.0, math.log(0.4), 1, None),
        ProposedStep((second,), ("rule",), 1.5, math.log(0.3), 2, None),
    ]
    monkeypatch.setattr(stub, "propose", lambda pool, ctx, k: steps)
    ex = expansion.expand_state(synthetic_runtime, sn2_state, _req(beam=1))
    (child,) = ex.children
    assert child.probability == pytest.approx(0.7) and child.n_alternatives == 1
    alt = child.alternatives[0]
    assert alt["rank"] == 2 and alt["probability"] == pytest.approx(0.3)
    assert alt["arrows"][0]["donor"] == 15 and child.arrows[0]["donor"] == 16


def test_failed_expansions(synthetic_runtime, sn2_state):  # noqa: F811
    from ksmech.serve.contracts import PoolState

    ex = expansion.expand_state(synthetic_runtime, sn2_state, _req("gbdt"))
    assert not ex.ok and ex.code == Code.PREDICTOR_UNAVAILABLE
    ex = expansion.expand_state(synthetic_runtime, sn2_state, _req(scope="nope"))
    assert not ex.ok and ex.code == Code.REQUEST_INVALID
    ex = expansion.expand_state(synthetic_runtime, sn2_state, _req(temperature=-1.0))
    assert not ex.ok and ex.code == Code.REQUEST_INVALID
    fake = {"state_id": "s" + "f" * 20, "fragment_token_keys": ["C[O-]", "Xx"]}
    broken = PoolState.from_dict(sn2_state.to_dict() | fake)  # its own (failure) directory
    ex = expansion.expand_state(synthetic_runtime, broken, _req())
    assert not ex.ok and ex.code == Code.ANALYSIS_FAILED and ex.upstream == Code.MISSING_TOKENS


def test_no_candidates_in_scope(synthetic_runtime, synthetic_settings):  # noqa: F811
    from ksmech.serve.pools import SpeciesInput, initial_state, save_state

    state = initial_state([SpeciesInput("CO", 1)], synthetic_settings.limits)
    save_state(state, synthetic_settings)
    ex = expansion.expand_state(synthetic_runtime, state, _req("floor", scope="inter"))
    assert ex.ok and ex.children == [] and ex.n_valid == 0 and ex.n_scanned == 0


def test_apply_user_step_on_sigma_star_ch(synthetic_runtime, sn2_state, synthetic_settings):  # noqa: F811
    out = expansion.apply_user_step(
        synthetic_runtime, sn2_state, [CH_ACCEPTOR], False, "stub", None
    )
    assert out.state.pool_key == "COCCl.[H-]"
    assert state_path(out.state.state_id, synthetic_settings).is_file()
    (arrow,) = out.arrows
    assert arrow["site_source"] == "user" and arrow["sink_partner"] == 9
    ArrowPhysicsBrief.model_validate(arrow["physics"])
    assert out.encoded["code"] == "2=6;6,10=10" and out.touched_maps == [2, 6, 10]
    (view,) = out.predictor_view
    PredictorViewOut.model_validate(view)
    assert view["pair_valid"] and view["rank"] >= 1 and 0 < view["probability"] < 1
    assert out.issues == []
    contact = [arrow["donor_site"], arrow["acceptor_site"]]
    assert contact == arrow["physics"]["contact_sites"] or arrow["site_pair"] is not None


def test_apply_user_step_site_pair_block(synthetic_runtime, sn2_state):  # noqa: F811
    off_contact = ArrowSpec(16, 1, 28, 6)  # sigma*(C-Cl) at Cl, not its contact site C
    out = expansion.apply_user_step(
        synthetic_runtime, sn2_state, [off_contact], False, "floor", None
    )
    assert out.state.pool_key == "COCl.[CH3-]" and out.encoded["code"] == "2=7;6,7=6"
    (arrow,) = out.arrows
    assert expansion.USER_SITES_FLAG in arrow["physics"]["flags"]
    sp = arrow["site_pair"]
    assert (sp["donor_site"], sp["acceptor_site"]) == (1, 6) and not sp["is_contact"]
    assert (sp["donor_site_map"], sp["acceptor_site_map"]) == (2, 7)


def test_apply_user_step_errors(synthetic_runtime, sn2_state):  # noqa: F811
    with pytest.raises(ServeError) as e:
        expansion.apply_user_step(
            synthetic_runtime, sn2_state, [ArrowSpec(1, 1, 28, 5)], True, "floor", None
        )
    assert e.value.code == Code.NOT_AN_ORBITAL_TOKEN and e.value.status == 422
    own = ArrowSpec(25, 6, 28, 5)
    with pytest.raises(ServeError) as e:
        expansion.apply_user_step(synthetic_runtime, sn2_state, [own], False, "floor", None)
    assert e.value.code == Code.PAIR_EXCLUDED and e.value.detail["reason"] == "OWN_ANTIBOND"
    with pytest.raises(ServeError) as e:
        expansion.apply_user_step(synthetic_runtime, sn2_state, [NO_PAIR], False, "floor", None)
    assert e.value.code == Code.APPLY_FAILED and e.value.upstream == "VALENCE"


def test_apply_user_step_with_an_unavailable_predictor(synthetic_runtime, sn2_state):  # noqa: F811
    off = ArrowSpec(16, 1, 34, 9)  # sigma*(C-H9) at its H: a centre, the pair is valid
    out = expansion.apply_user_step(synthetic_runtime, sn2_state, [off], False, "gbdt", None)
    codes = [x["code"] for x in out.issues]
    assert codes == [Code.PREDICTOR_UNAVAILABLE]
    assert out.predictor_view[0]["score"] is None and out.predictor_view[0]["pair_valid"]


def test_apply_user_step_resonance_only(synthetic_runtime, sn2_state, monkeypatch):  # noqa: F811
    monkeypatch.setattr(
        sa, "apply_step", lambda rt, arrows: sa.ApplyOutcome(True, None, None, "", None, sn2_state)
    )
    with pytest.raises(ServeError) as e:
        expansion.apply_user_step(synthetic_runtime, sn2_state, [SN2], False, "floor", None)
    assert e.value.code == Code.RESONANCE_ONLY_STEP and e.value.status == 422


def test_tasks_round_trip_through_inline_scheduler(synthetic_settings, sn2_state):  # noqa: F811
    from ksmech.serve.settings import TaskEnv
    from ksmech.serve.worker import InlineScheduler, Priority

    sched = InlineScheduler(TaskEnv(synthetic_settings))
    req = _req(beam=2)
    r = sched.run(
        "expand",
        {"state": sn2_state.to_dict(), "req": req.to_dict()},
        priority=Priority.JOB,
        timeout_s=900,
    )
    assert r.ok, r.detail
    ex = Expansion.from_dict(r.result)
    assert ex.to_dict() == r.result
    assert ex.children[0].state.pool_key == "COC.[Cl-]" and len(ex.children) == 2
    payload = {
        "state": sn2_state.to_dict(),
        "arrows": [CH_ACCEPTOR.to_dict()],
        "allow_invalid": False,
        "predictor": "stub",
        "temperature": None,
    }
    r = sched.run("apply", payload, priority=Priority.JOB, timeout_s=900)
    assert r.ok, r.detail
    out = StepOutcome.from_dict(r.result)
    assert out.to_dict() == r.result and out.state.pool_key == "COCCl.[H-]"
    bad = payload | {"arrows": [ArrowSpec(1, 1, 28, 5).to_dict()]}
    r = sched.run("apply", bad, priority=Priority.JOB, timeout_s=900)
    assert not r.ok and r.code == Code.NOT_AN_ORBITAL_TOKEN
    err = ServeError.from_task(r)
    assert err.status == 422 and err.detail["arrow"] == 0


def test_alternative_scores_are_lower(stub_ex):
    for child in stub_ex.children:
        for alt in child.alternatives:
            assert alt["rank"] > child.rank and alt["score"] <= child.score
            assert np.isfinite(alt["probability"])


class _Retrained:
    """A predictor named like the stub whose current version and scores differ from what the
    analysis stored (a model retrained after the analysis)."""

    def __init__(self, inner, version: str, shift: float = 1.0) -> None:
        self.name, self.inner, self.version, self.shift = inner.name, inner, version, shift

    def info(self):
        import dataclasses

        return dataclasses.replace(self.inner.info(), version=self.version)

    def score(self, ctx):
        return self.inner.score(ctx) + self.shift


def test_stored_scores_need_the_current_version(synthetic_runtime, sn2_state):  # noqa: F811
    from ksmech.serve.analysis import pool_work

    work = pool_work(synthetic_runtime, sn2_state)
    stub = synthetic_runtime.scorers["stub"]
    ctx = ScoringContext(work.sample, work.rt, synthetic_runtime.loader, work.comps)
    recorded = work.an.meta["predictor_versions"]["stub"]
    assert recorded == stub.info().version
    same = _Retrained(stub, recorded)  # the recorded version: the stored member is used
    assert np.array_equal(expansion._scores(synthetic_runtime, work, same, ctx), work.an.score_stub)
    newer = _Retrained(stub, "stub/retrained")
    got = expansion._scores(synthetic_runtime, work, newer, ctx)
    assert np.array_equal(got, newer.score(ctx)) and not np.array_equal(got, work.an.score_stub)
