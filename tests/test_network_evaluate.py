"""`network.benchmarks` and `network.evaluate`: the cases of the pathway benchmarks and the
search towards their products, scored (reaction-network plan, section 5)."""

from collections import Counter

import pytest

from ksmech.network.benchmarks import Case, _chains, _part, unmapped
from ksmech.network.evaluate import evaluate_case, recorded_rank, summary
from ksmech.network.pools import pool_state
from ksmech.network.search import Expansion, Proposal, SearchParams
from ksmech.network.target import EXACT

START = "CCl.[OH-]"
SN2 = "CO.[Cl-]"
DETOUR = "[CH2-]Cl.O"


class ToyExpander:
    """Fixed steps between pool states given as SMILES."""

    def __init__(self, steps):
        self.steps = {pool_state(a).key: [(pool_state(b), lp) for b, lp in outs]
                      for a, outs in steps.items()}  # fmt: skip

    def key(self, state):
        return state.key

    def expand(self, state):
        return Expansion([Proposal(c, lp) for c, lp in self.steps.get(state.key, [])])


TOY = ToyExpander({START: [(SN2, -0.2), (DETOUR, -1.0)], DETOUR: [(SN2, -0.3)]})


def test_a_case_scores_arrival_recorded_rank_and_step_targets():
    recorded = (pool_state(START).key, pool_state(DETOUR).key, pool_state(SN2).key)
    case = Case("c1", START, "CO", recorded, ("[CH2-]Cl", "CO"), "val")
    row = evaluate_case(TOY, case, SearchParams(k=3))
    assert (row["arrival"], row["n_pathways"], row["best_steps"]) == (EXACT, 2, 1)
    assert row["recorded_rank"] == 2 and row["recorded_steps"] == 2
    assert (row["step_targets"], row["step_targets_passed"]) == (2, 1)  # the best skips the detour
    missing = evaluate_case(TOY, Case("c2", START, "CCO"), SearchParams(k=3))
    assert missing["arrival"] is None and missing["recorded_rank"] is None
    bad = evaluate_case(TOY, Case("c3", "C(", "CO"))
    assert bad["failure"].startswith("UNPARSED")
    s = summary([row, missing, bad], k=3)
    assert (s["n"], s["failed"], s["arrival_exact"], s["n_with_recorded"]) == (2, 1, 0.5, 1)
    assert (s["recorded_top1"], s["recorded_top3"]) == (0.0, 1.0)


def test_recorded_rank_matches_the_whole_sequence():
    class Res:
        class search:
            pathways = [type("P", (), {"nodes": ("a", "b")}), type("P", (), {"nodes": ("a", "c")})]

    assert recorded_rank(Res, ("a", "c")) == 2
    assert recorded_rank(Res, ("a",)) is None and recorded_rank(Res, None) is None


def test_chain_helpers():
    rows = [{"pathway_id": "p", "step_index": 1, "step_id": "b"},
            {"pathway_id": "p", "step_index": 0, "step_id": "a"}]  # fmt: skip
    assert [r["step_id"] for r in _chains(rows)["p"]] == ["a", "b"]
    assert _part(["a", "b"], {"a": "val", "b": "val"}) == "val"
    assert _part(["a", "b"], {"a": "val"}) == "mixed"
    assert unmapped("[CH3:1][OH:2]") == "CO" and unmapped("C(") is None


@pytest.mark.realdata
def test_the_reactmech_chains_ingested_whole_are_cases():
    from ksmech.network.benchmarks import CHAIN_NOT_INGESTED, reactmech_cases

    counts = Counter()
    cases = reactmech_cases(counts=counts)
    assert cases and counts[CHAIN_NOT_INGESTED] > 0
    for c in cases:
        assert c.recorded[0] == pool_state(c.start).key and len(c.recorded) >= 2
        assert c.part in ("train", "val", "test", "mixed", "none")


def test_the_part_option_selects_the_cases_of_that_part(monkeypatch):
    from types import SimpleNamespace

    from ksmech.network import benchmarks
    from ksmech.network.cli import _cases

    cases = [Case("a", START, "CO", part="val"), Case("b", START, "CO", part="train"),
             Case("c", START, "CO", part="mixed")]  # fmt: skip
    monkeypatch.setattr(benchmarks, "reactmech_cases", lambda *a, **k: list(cases))
    args = SimpleNamespace(set="reactmech", processed_dir=None, split="molecule", part="val",
                           limit=None)  # fmt: skip
    assert [c.case_id for c in _cases(args, Counter())] == ["a"]
    assert len(_cases(SimpleNamespace(**{**vars(args), "part": "all"}), Counter())) == 3


def test_the_human_benchmark_is_read_under_processed_dir(monkeypatch, tmp_path):
    from ksmech.data import human_benchmark as hb
    from ksmech.network.benchmarks import human_benchmark_cases

    seen = []
    monkeypatch.setattr(hb, "load_pathways", lambda out_dir: seen.append(out_dir) or [])
    assert human_benchmark_cases(tmp_path) == [] and seen == [tmp_path / hb.NAME]


class ToyBackward:
    """The previous pool of a species, given by hand: CO came from CCl + OH-."""

    def expand(self, pool):
        if pool.key == pool_state("CO").key:
            return Expansion([Proposal(pool_state(START), -0.4, "back", frozenset({1}))])
        return Expansion([])


def test_a_retro_case_recovers_the_recorded_reactants_and_is_verified_forward():
    from ksmech.network.evaluate import retro_case, retro_summary
    from ksmech.network.retro import RetroParams

    case = Case("r1", START, "CO", None, (), "benchmark")
    row = retro_case(ToyBackward(), case, [], RetroParams(), 0, TOY, SearchParams(k=1))
    assert (row["solved"], row["recovered"], row["verified"]) == (True, True, True)
    assert row["cost"] == pytest.approx(0.4) and row["expanded"] == 1
    other = retro_case(ToyBackward(), Case("r2", "CCBr.[OH-]", "CO"), [], RetroParams(), 0)
    assert other["solved"] is False and other["recovered"] is False and other["verified"] is None
    bad = retro_case(ToyBackward(), Case("r3", "C(", "CO"), [], RetroParams(), 0)
    s = retro_summary([row, other, bad])
    assert (s["n"], s["failed"], s["solved"], s["recovered"]) == (2, 1, 0.5, 0.5)
    assert (s["n_verified_checked"], s["verified"]) == (1, 1.0)
