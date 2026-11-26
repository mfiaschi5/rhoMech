"""`network.rerank`, the walk's rerank, the search's first-step hit and the human benchmark's
frozen halves (the cheap tests of 2026-11-26)."""

from collections import Counter
from types import SimpleNamespace

import pytest

from ksmech.network.benchmarks import Case, freeze_halves, load_half
from ksmech.network.evaluate import evaluate_case, summary
from ksmech.network.pools import pool_state
from ksmech.network.rerank import NO_RECORD, EnergyScorer, Rerank, Score, reorder_places
from ksmech.network.search import Proposal, SearchParams
from ksmech.network.walk import walk_case, walk_summary
from tests.test_network_evaluate import DETOUR, SN2, START, TOY


def test_reorder_places_permutes_only_the_scored_places():
    logps, scores = [-0.1, -1.0, -2.0, -3.0], [0.0, None, 5.0, 1.0]
    assert reorder_places(logps, scores, 0.0) == [0, 1, 2, 3]
    # lambda 1: 0 -> -0.1, 2 -> 3.0, 3 -> -2.0; place 1 (unscored) stays
    assert reorder_places(logps, scores, 1.0) == [2, 1, 0, 3]
    assert reorder_places(logps, [None] * 4, 2.0) == [0, 1, 2, 3]


def test_a_rerank_gives_the_hit_rank_per_lambda_and_counts_the_unscored():
    props = [Proposal(pool_state(x), lp) for x, lp in ((SN2, -0.2), (DETOUR, -1.0), ("CO", -4.0))]
    score = {pool_state(SN2).key: 0.0, pool_state(DETOUR).key: 2.0}
    rr = Rerank(lambda p: score.get(p.child.key, NO_RECORD), (1.0,), top=3)
    assert rr.lambdas == (0.0, 1.0)
    skips = Counter()
    out = rr.ranks(props, 1, skips)
    assert out == {"full": False, "by_lambda": {"0": 2, "1": 1}}
    assert skips == {f"RERANK_UNSCORED:{NO_RECORD}": 1}
    assert rr.ranks(props, None, Counter())["by_lambda"] == {"0": None, "1": None}
    assert Rerank(rr.score, (1.0,), top=1).ranks(props, 2, Counter())["by_lambda"]["1"] == 3


def test_the_walk_reads_every_lambda_and_still_moves_to_the_target_child():
    favour_detour = Rerank(lambda p: 1.0 if p.child.key == pool_state(DETOUR).key else 0.0,
                           (2.0,))  # fmt: skip
    row = walk_case(TOY, Case("w", START, "CO", None, ("CO",), "benchmark"), favour_detour)
    assert row["steps"][0]["rank"] == 1
    assert row["steps"][0]["rerank"] == {"full": True, "by_lambda": {"0": 1, "2": 2}}
    detour = walk_case(TOY, Case("d", START, "CO", None, ("[CH2-]Cl", "CO"), "benchmark"),
                       favour_detour)  # fmt: skip
    assert [s["rank"] for s in detour["steps"]] == [2, 1] and detour["stop"] is None
    s = walk_summary([row, detour])
    assert s["first_step"]["top1"] == 0.5
    assert s["first_step_rerank"]["0"]["top1"] == 0.5 and s["first_step_rerank"]["2"]["top1"] == 0.5
    assert s["first_step_rerank"]["2"]["full"]["n"] == 2
    plain = walk_case(TOY, Case("p", START, "CO", None, ("CO",), "b"))
    assert "first_step_rerank" not in walk_summary([plain])


def test_the_energy_scorer_sums_the_fragments_and_passes_failures_on(monkeypatch):
    scorer = EnergyScorer(temperature_k=298.15)
    energies = {"CO": -100.0, "[Cl-]": -50.0}
    monkeypatch.setattr(scorer, "_load", lambda key: energies.get(key, NO_RECORD))
    state = pool_state(SN2)
    assert scorer.energy(state) == pytest.approx(-150.0)
    s = scorer.step_logp(state)
    assert isinstance(s, Score) and s.marginal == pytest.approx(150.0 / scorer.rt)
    assert scorer.step_logp(pool_state(DETOUR)) == NO_RECORD
    assert set(scorer.memo) >= {"CO", "[Cl-]"}


def test_the_search_reports_whether_its_best_pathway_makes_the_first_target():
    hit = evaluate_case(TOY, Case("h", START, "CO", None, ("CO",)), SearchParams(k=3))
    detour = evaluate_case(TOY, Case("m", START, "CO", None, ("[CH2-]Cl", "CO")),
                           SearchParams(k=3))  # fmt: skip
    held = evaluate_case(TOY, Case("x", START, "CO", None, ("CCl", "CO")), SearchParams(k=3))
    lost = evaluate_case(TOY, Case("l", START, "CCO", None, ("CCO",)), SearchParams(k=3))
    assert (hit["first_step_hit"], detour["first_step_hit"]) == (True, False)
    assert (held["first_step_hit"], lost["first_step_hit"]) == (None, False)
    s = summary([hit, detour, held, lost], k=3)
    assert (s["n_first_step"], s["first_step_top1"]) == (3, 1 / 3)


def test_the_halves_put_the_inspected_in_dev_and_are_read_back(tmp_path, monkeypatch):
    ids = [f"p{i}" for i in range(40)]
    halves = freeze_halves(ids, ["p3", "p7"])
    assert sorted(halves["dev"] + halves["test"]) == sorted(ids)
    assert {"p3", "p7"} <= set(halves["dev"]) and halves["test"]
    assert freeze_halves(ids, ["p3", "p7"]) == halves
    with pytest.raises(ValueError):
        freeze_halves(ids, ["nope"])
    from ksmech.data import human_benchmark as hb

    (tmp_path / hb.NAME).mkdir()
    (tmp_path / hb.NAME / "halves.json").write_text(__import__("json").dumps(halves))
    assert load_half(tmp_path, "test") == set(halves["test"])

    from ksmech.network import benchmarks
    from ksmech.network.cli import _cases

    cases = [Case(i, START, "CO") for i in ids]
    monkeypatch.setattr(benchmarks, "human_benchmark_cases", lambda *a, **k: list(cases))
    args = SimpleNamespace(set="human_benchmark", processed_dir=tmp_path, split="molecule",
                           part="all", limit=None, half="dev")  # fmt: skip
    assert {c.case_id for c in _cases(args, Counter())} == set(halves["dev"])
    with pytest.raises(ValueError):
        _cases(SimpleNamespace(**{**vars(args), "set": "reactmech"}), Counter())
