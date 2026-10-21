from dataclasses import replace

import pytest

from ksmech.data import splits
from ksmech.data.schema import ElementaryStep, Source
from tests.fixtures.molecules import MOLECULES


def _synthetic_steps(n: int = 60) -> list[ElementaryStep]:
    """Steps whose largest reactant cycles through the fixture molecules; 3 steps per pathway,
    every 4th pathway unknown, classes cycling over A/B/C (one unlabelled step per 10)."""
    names = sorted(MOLECULES)
    out = []
    for i in range(n):
        pathway = None if (i // 3) % 4 == 3 else f"p{i // 3}"
        out.append(
            ElementaryStep(
                step_id=f"s{i:03d}",
                source=Source.PMECHDB,
                pathway_id=pathway,
                step_index=None if pathway is None else i % 3,
                reactants_smiles=f"{MOLECULES[names[i % len(names)]]}.[OH-:1]",
                products_smiles="[OH-:1]",
                arrows=[],
                mechanism_class=None if i % 10 == 9 else "ABC"[i % 3],
            )
        )
    return out


STEPS = _synthetic_steps()
ALL_IDS = {s.step_id for s in STEPS}


def _check_partition(split: splits.Split, ids=ALL_IDS):
    parts = [set(split.train), set(split.val), set(split.test), set(split.excluded)]
    assert sum(len(p) for p in parts) == len(ids)
    assert set().union(*parts) == ids


@pytest.mark.parametrize(
    "fn", [splits.split_by_pathway, splits.split_by_scaffold, splits.split_by_molecule]
)
def test_deterministic_and_partition(fn):
    a, b = fn(STEPS, seed=7), fn(STEPS, seed=7)
    assert a == b
    _check_partition(a)
    assert all(lst == sorted(lst) for lst in (a.train, a.val, a.test))


def test_pathway_seed_changes_split():
    assert splits.split_by_pathway(STEPS, seed=0) != splits.split_by_pathway(STEPS, seed=1)


def test_pathway_split_never_breaks_a_pathway():
    for seed in range(5):
        sp = splits.split_by_pathway(STEPS, seed=seed)
        where = {i: name for name, ids in sp.as_dict().items() for i in ids}
        by_pathway: dict[str, set[str]] = {}
        for s in STEPS:
            if s.pathway_id is not None:
                by_pathway.setdefault(s.pathway_id, set()).add(where[s.step_id])
        assert all(len(v) == 1 for v in by_pathway.values())


def test_pathway_split_without_any_pathway_id_is_per_step():
    bare = [replace(s, pathway_id=None, step_index=None) for s in STEPS]
    sp = splits.split_by_pathway(bare, seed=0)
    _check_partition(sp)
    assert (len(sp.train), len(sp.val), len(sp.test)) == (48, 6, 6)  # 60 one-step groups
    _check_partition(splits.split_by_scaffold(bare, seed=0))


def test_pathway_split_fractions():
    sp = splits.split_by_pathway(STEPS, (0.5, 0.25, 0.25), seed=3)
    # greedy group packing never overshoots a target and misses it by less than one group (3)
    assert 27 < len(sp.train) <= 30 and 12 < len(sp.val) <= 15


def test_scaffold_split_never_breaks_a_scaffold():
    unknown = [s for s in STEPS if s.pathway_id is None]
    sp = splits.split_by_scaffold(unknown, seed=0)
    where = {i: name for name, ids in sp.as_dict().items() for i in ids}
    by_scaffold: dict[str, set[str]] = {}
    for s in unknown:
        key = splits.largest_reactant_scaffold(s.reactants_smiles)
        by_scaffold.setdefault(key, set()).add(where[s.step_id])
    assert all(len(v) == 1 for v in by_scaffold.values())


def test_largest_reactant_scaffold():
    assert splits.largest_reactant_scaffold("[CH3:1][OH:2].C1=CC=CC=C1C") == "c1ccccc1"
    assert splits.largest_reactant_scaffold("CCCCCCCO.c1ccccc1") == ""  # largest is acyclic


def test_leave_one_class_out():
    sp = splits.split_leave_one_class_out(STEPS, "A", val_fraction=0.2, seed=0)
    _check_partition(sp)
    cls = {s.step_id: s for s in STEPS}
    assert {cls[i].mechanism_class for i in sp.test} == {"A"}
    assert {cls[i].mechanism_class for i in sp.train + sp.val} <= {"B", "C"}
    held_pathways = {cls[i].pathway_id for i in sp.test} - {None}
    for i in sp.excluded:  # unlabelled, or would leak a held-out pathway
        assert cls[i].mechanism_class is None or cls[i].pathway_id in held_pathways
    assert sp == splits.split_leave_one_class_out(STEPS, "A", val_fraction=0.2, seed=0)


def test_leave_one_class_out_all_classes():
    folds = splits.leave_one_class_out_splits(STEPS)
    assert sorted(folds) == ["A", "B", "C"]


def test_leave_one_class_out_needs_labels():
    unlabelled = [replace(s, mechanism_class=None) for s in STEPS]
    with pytest.raises(ValueError, match="no step has a mechanism_class"):
        splits.split_leave_one_class_out(unlabelled, "A")
    with pytest.raises(ValueError, match="not present"):
        splits.split_leave_one_class_out(STEPS, "Z")


def test_bad_fractions():
    with pytest.raises(ValueError):
        splits.split_by_pathway(STEPS, (0.8, 0.1, 0.2))


def _where(split: splits.Split) -> dict[str, str]:
    return {i: name for name, ids in split.as_dict().items() for i in ids}


def test_molecule_split_never_shares_a_molecule_or_pathway():
    for seed in range(5):
        where = _where(splits.split_by_molecule(STEPS, seed=seed))
        seen: dict[str, set[str]] = {}
        for s in STEPS:
            tokens = set(splits.step_molecules(s.reactants_smiles, s.products_smiles))
            tokens |= {f"pathway:{s.pathway_id}"} if s.pathway_id else set()
            for tok in tokens:
                seen.setdefault(tok, set()).add(where[s.step_id])
        assert seen and all(len(v) == 1 for v in seen.values())


def _plain(step_id: str, rxn: str) -> ElementaryStep:
    reactants, products = rxn.split(">>")
    return ElementaryStep(step_id, Source.PMECHDB, None, None, reactants, products)


def test_molecule_groups_join_consecutive_steps_through_their_products():
    steps = [
        _plain("a", "CC(C)=O.[OH3+]>>CC(C)=[OH+].O"),  # makes protonated acetone
        _plain("b", "CC(C)=[OH+].O>>CC(C)(O)[OH2+]"),  # shares no reactant with a
        _plain("c", "CCOC(C)=O.[OH-]>>CCOC(C)([O-])O"),
        _plain("d", "O.[OH3+]>>[OH3+].O"),  # trivial species only: its own group
    ]
    groups = sorted(splits.molecule_groups(steps).values())
    assert groups == [["a", "b"], ["c"], ["d"]]


def test_default_split_per_source():
    assert splits.default_split("pmechdb") is splits.split_by_molecule
    assert splits.default_split(Source.REACTMECH) is splits.split_by_pathway


def test_group_bigger_than_train_target_still_goes_to_train(caplog):
    big = [replace(s, pathway_id="big") for s in STEPS[:40]]
    split = splits.split_by_pathway(big + STEPS[40:], (0.5, 0.25, 0.25), seed=0)
    assert {s.step_id for s in big} <= set(split.train)
    assert "train target" in caplog.text
