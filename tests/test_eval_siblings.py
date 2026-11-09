"""Multi-reference evaluation (reaction-network plan, item A): the recorded outcomes of a
reactant pool (`eval.siblings`), `rank_any` in the decode report and `--metric any` in the
paired comparison."""

from collections import Counter
from types import SimpleNamespace

import pytest

from ksmech.arrows.delta_be import ReactionMatrix
from ksmech.data import provenance as prov
from ksmech.eval import paired
from ksmech.eval.cli import DECODE_VERSION
from ksmech.eval.decode import any_rank, decode_report
from ksmech.eval.siblings import (
    SIBLING_KEY_FAILED,
    Outcomes,
    outcome_summary,
    outcome_table,
    parts_of,
    side_keys,
)
from ksmech.model.decode import Completion

# methoxide + acetone: addition to the carbonyl and deprotonation of the alpha carbon
POOL = "[CH3:1][O-:2].[CH3:3][C:4](=[O:5])[CH3:6]"
POOL_RENUMBERED = "[CH3:6][C:5](=[O:4])[CH3:3].[CH3:2][O-:1]"
ADDUCT = "[CH3:1][O:2][C:4]([CH3:3])([O-:5])[CH3:6]"
ADDUCT_RENUMBERED = "[CH3:2][O:1][C:5]([CH3:6])([O-:4])[CH3:3]"
ENOLATE = "[CH3:1][OH:2].[CH2-:3][C:4](=[O:5])[CH3:6]"
ENOLATE_O = "[CH3:1][OH:2].[CH2:3]=[C:4]([O-:5])[CH3:6]"  # the same products, other drawing


def _step(step_id, reactants, products):
    return SimpleNamespace(step_id=step_id, reactants_smiles=reactants, products_smiles=products)


STEPS = [
    _step("add", POOL, ADDUCT),
    _step("dep", POOL, ENOLATE),
    _step("add2", POOL_RENUMBERED, ADDUCT_RENUMBERED),
    _step("dep_o", POOL, ENOLATE_O),
    _step("other", "[CH3:1][Cl:2].[OH-:3]", "[CH3:1][OH:3].[Cl-:2]"),
]
PARTS = {"add": "val", "dep": "train", "add2": "test", "dep_o": "train", "other": "val"}


def test_pools_and_products_are_compared_by_map_free_identity_key():
    skips = Counter()
    keys = side_keys(STEPS, skips)
    assert not skips
    assert keys["add"] == keys["add2"]  # renumbered atoms, the same pool and product
    assert keys["dep"] == keys["dep_o"]  # resonance drawings of the enolate agree
    assert keys["add"][0] == keys["dep"][0] and keys["add"][1] != keys["dep"][1]


def test_outcome_table_groups_by_reactant_pool_and_counts_other_parts():
    table = outcome_table(side_keys(STEPS, Counter()), PARTS)
    add, dep = table["add"], table["dep"]
    assert add.products == dep.products and len(add.products) == 2
    assert add.multi and not table["other"].multi
    assert add.other_parts == ("train", "train")  # dep and dep_o record the enolate
    assert dep.other_parts == ("test", "val")  # add2 and add record the adduct
    assert table["other"].products == frozenset({table["other"].own})


def test_a_step_rdkit_cannot_read_is_counted():
    skips = Counter()
    keys = side_keys([_step("bad", "[CH3:1][Xx]", "C"), *STEPS[:1]], skips)
    assert skips == {SIBLING_KEY_FAILED: 1} and set(keys) == {"add"}


def test_outcome_summary_counts_siblings_by_part():
    table = outcome_table(side_keys(STEPS, Counter()), PARTS)
    s = outcome_summary(table, ["add", "other", "missing"], "val")
    assert (s["n_with_key"], s["n_multi"], s["max_outcomes"]) == (2, 1, 2)
    assert (s["siblings_same_part"], s["siblings_other_part"]) == (0, 2)
    assert parts_of({"train": ["a"], "val": ["b"]}) == {"a": "train", "b": "val"}


SN2 = (((0, 5, 1), (0, 1, -1)), ((1, 2), (5, -2)))
OTHER = (((0, 5, 1), (0, 2, -1)), ((5, -2),))


def _completion(matrix, key):
    return Completion(0.0, tuple(matrix[0]), tuple(matrix[1]), (0, 5, 0), key)


def test_any_rank_takes_the_first_recorded_product_or_the_own_label():
    own, sibling = _completion(SN2, "ADD"), _completion(OTHER, "DEP")
    assert any_rank([sibling, own], 1, frozenset({"ADD", "DEP"})) == 0
    assert any_rank([sibling, own], 1, frozenset({"ADD"})) == 1
    assert any_rank([sibling], None, frozenset({"ADD"})) is None
    keyless = Completion(0.0, (), (), (0, 1, 0))  # a completion built without a key
    assert any_rank([keyless, own], 1, frozenset({"ADD"})) == 1


def test_report_scores_rank_any_and_the_multi_outcome_subset():
    label = SimpleNamespace(variants=[ReactionMatrix(*SN2)])
    own, sibling = _completion(SN2, "ADD"), _completion(OTHER, "DEP")
    samples = [
        SimpleNamespace(step_id="a", edit_label=label),
        SimpleNamespace(step_id="x", edit_label=label, product_key="ADD"),
        SimpleNamespace(step_id="m", edit_label=label),
    ]
    outcomes = {
        "a": Outcomes("ADD", frozenset({"ADD", "DEP"}), ("train",)),
        "m": Outcomes("ZZZ", frozenset({"ZZZ"})),  # its own hit has another key
    }
    r = decode_report(samples, lambda s, skips: [sibling, own], outcomes=outcomes)
    rows = {row["step_id"]: row for row in r["per_sample"]}
    assert (rows["a"]["rank"], rows["a"]["rank_any"], rows["a"]["n_outcomes"]) == (1, 0, 2)
    assert (rows["x"]["rank"], rows["x"]["rank_any"], rows["x"]["n_outcomes"]) == (1, 1, 1)
    assert (r["top1"], r["top1_any"]) == (0.0, pytest.approx(1 / 3))
    assert r["multi_outcome"] == {"n": 1, "top1": 0.0, "top3": 1.0, "top5": 1.0,
                                  "top1_any": 1.0, "top3_any": 1.0, "top5_any": 1.0}  # fmt: skip
    assert r["skips"]["OWN_KEY_MISMATCH"] == 1


def _hits(path, ranks, any_ranks=None):
    rows = [{"step_id": f"s{k}", "rank": r} for k, r in enumerate(ranks)]
    for row, r in zip(rows, any_ranks or [], strict=False):
        row["rank_any"] = r
    config = {"source": "pmechdb", "split": "molecule", "part": "val"}
    return prov.write_json(
        path, {"per_sample": rows}, prov.stamp("decode_hits", DECODE_VERSION, {}, config)
    )


def test_paired_comparison_reads_rank_any_under_metric_any(tmp_path):
    a = _hits(tmp_path / "a.json", [None, 1, 0], [0, 0, 0])
    b = _hits(tmp_path / "b.json", [0, None, 0], [0, None, 0])
    own = paired.compare_arms([a], [b], k=1)
    anyk = paired.compare_arms([a], [b], k=1, metric="any")
    assert own["top_k"]["a"] == [pytest.approx(100 / 3)] and own["metric"] == "own"
    assert anyk["top_k"]["a"] == [100.0] and anyk["top_k"]["b"] == [pytest.approx(200 / 3)]


def test_a_hits_file_without_rank_any_is_refused_for_metric_any(tmp_path):
    a = _hits(tmp_path / "a.json", [0, 1])
    with pytest.raises(ValueError, match="rank_any"):
        paired.compare_arms([a], [a], metric="any")
    with pytest.raises(ValueError, match="metric"):
        paired.read_hits(a, metric="best")
