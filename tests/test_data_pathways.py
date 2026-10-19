"""Pathway reconstruction on small hand-made chains."""

import hashlib

import pytest

from ksmech.data import pathways
from ksmech.data.schema import ElementaryStep, Source

# Acid-catalysed hydration of acetone, plus distractors. Species of <= 3 heavy atoms are ignored,
# so s3 still follows s2 although s3 adds a chloride that s2 never made.
CHAIN = {
    "s1": "CC(C)=O.[OH3+]>>CC(C)=[OH+].O",
    "s2": "CC(C)=[OH+].O>>CC(C)(O)[OH2+]",
    "s3": "CC(C)(O)[OH2+].[Cl-]>>CC(C)(O)O.Cl",
    "rev": "CC(C)=[OH+].O>>CC(C)=O.[OH3+]",  # undoes s1
    "other": "CCOC(C)=O.[OH-]>>CCOC(C)([O-])O",  # unrelated
    "t1": "[OH3+].[Cl-]>>O.Cl",  # trivial species only: must never link,
    "t2": "O.Cl>>[OH3+].[Cl-]",  # although t1 -> t2 -> t1 would match exactly
}


def _step(step_id: str, rxn: str, pathway_id: str | None = None) -> ElementaryStep:
    reactants, products = rxn.split(">>")
    return ElementaryStep(step_id, Source.PMECHDB, pathway_id, None, reactants, products)


def _assign(names, **kw):
    steps, report = pathways.assign_pathways([_step(n, CHAIN[n]) for n in names], **kw)
    return {s.step_id: (s.pathway_id, s.step_index) for s in steps}, report


def test_chain_is_one_pathway_ordered_from_its_first_step():
    got, report = _assign(CHAIN)
    pid = "pmechdb-path-" + hashlib.sha1(b"s1").hexdigest()[:16]
    assert got["s1"] == (pid, 0)
    assert got["s2"] == (pid, 1) and got["rev"] == (pid, 1)  # the reverse step branches off s1
    assert got["s3"] == (pid, 2)
    assert got["other"] == got["t1"] == got["t2"] == (None, None)
    assert report.as_dict()["n_pathways"] == 1 and report.pathway_sizes == [4]
    assert (report.n_links, report.n_reverse_pairs, report.n_branching_steps) == (4, 1, 0)
    assert report.n_empty_reactant_key == report.n_empty_product_key == 2


def test_result_does_not_depend_on_input_order():
    assert _assign(CHAIN)[0] == _assign(reversed(list(CHAIN)))[0]


def test_existing_pathway_and_bad_smiles_are_left_alone():
    steps = [
        _step("s1", CHAIN["s1"], pathway_id="known"),
        _step("s2", CHAIN["s2"]),
        _step("bad", "C1CC>>CCC"),  # unclosed ring: RDKit cannot parse it
    ]
    out, report = pathways.assign_pathways(steps)
    expected = [("known", None), (None, None), (None, None)]
    assert [(s.pathway_id, s.step_index) for s in out] == expected
    assert report.skipped == {"HAS_PATHWAY": 1, "KEY_FAILED": 1}


def test_side_key_strips_maps_and_small_species():
    assert pathways.side_key("[CH3:1][C:2](C)=[O:3].[OH2:4].[Cl-]") == ("CC(C)=O",)
    assert pathways.side_key("O.[Na+].[Cl-]") == ()


# Keys are written by hand below; they only need to be valid SMILES (heavy atoms are counted).
SMALL, BIG = ("CC(C)=O",), ("CCCCCCCCCC",)


def _keys(spec: dict[str, tuple[str, str]]) -> pathways.Keys:
    return {sid: ((r,) if r else (), (p,)) for sid, (r, p) in spec.items()}


HUB = {  # h makes acetone; two unrelated reactions start from it
    "h": ("CCCCCC", "CC(C)=O"),
    "a1": ("CC(C)=O", "CCCCCCCC"),
    "a2": ("CCCCCCCC", "CCCCCCCCO"),
    "b1": ("CC(C)=O", "CCCCCCCCN"),
    "b2": ("CCCCCCCCN", "CCCCCCCCCl"),
}


def test_oversize_component_is_cut_at_the_shared_intermediate():
    keys = _keys(HUB)
    links = pathways.find_links(keys)
    assert links == {"h": ["a1", "b1"], "a1": ["a2"], "b1": ["b2"]}
    pieces, inner, cuts = pathways.split_oversize(sorted(keys), links, keys, max_steps=3)
    assert pieces == [["a1", "a2"], ["b1", "b2"], ["h"]] and cuts == 1
    assert inner == {"a1": ["a2"], "b1": ["b2"]}
    pieces, _, cuts = pathways.split_oversize(sorted(keys), links, keys, max_steps=5)
    assert pieces == [sorted(keys)] and cuts == 0


def test_smallest_shared_intermediate_is_cut_first():
    spec = {**HUB, "a1": ("CC(C)=O", BIG[0]), "a2": (BIG[0], "CCCCO"), "a3": (BIG[0], "CCCCN")}
    del spec["b2"]
    keys = _keys(spec)
    pieces, _, cuts = pathways.split_oversize(
        sorted(keys), pathways.find_links(keys), keys, max_steps=4
    )
    assert cuts == 1 and ["a1", "a2", "a3"] in pieces  # the 10-carbon hub survives


def test_plain_chain_is_kept_whole_however_long():
    keys = _keys({f"c{i}": (f"C{'C' * (4 + i)}", f"C{'C' * (5 + i)}") for i in range(5)})
    report = pathways.PathwayReport(n_steps=5)
    built = pathways.build_pathways(keys, report, max_steps=2)
    assert [members for members, _ in built] == [sorted(keys)]
    assert (report.n_components_cut, report.n_long_chains_kept) == (0, 1)


def test_build_pathways_drops_singletons_after_cutting():
    report = pathways.PathwayReport(n_steps=5)
    built = pathways.build_pathways(_keys(HUB), report, max_steps=3)
    assert [members for members, _ in built] == [["a1", "a2"], ["b1", "b2"]]
    assert (report.n_components_cut, report.n_intermediates_cut) == (1, 1)
    assert report.as_dict()["median_pathway_size"] == 2


@pytest.mark.parametrize(
    "links, expected",
    [
        ({"a": ["b"], "b": ["c"]}, {"a": 0, "b": 1, "c": 2}),
        ({"a": ["b"], "b": ["a"]}, {"a": 0, "b": 1}),  # pure reverse pair: smallest id first
        ({"a": ["c"], "b": ["c"]}, {"a": 0, "b": 0, "c": 1}),  # two starts merge
    ],
)
def test_order_pathway(links, expected):
    assert pathways.order_pathway(sorted(expected), links) == expected
