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


@pytest.mark.parametrize("fn", [splits.split_by_pathway, splits.split_by_scaffold])
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


# ---- train-only pinning (steps curated in ksmech-serve)


def _pinned_steps(every: int = 7) -> list[ElementaryStep]:
    """STEPS with every `every`-th step relabelled user_curated (train-only)."""
    return [
        replace(s, source=Source.USER_CURATED) if i % every == 0 else s for i, s in enumerate(STEPS)
    ]


PINNED_STEPS = _pinned_steps()
PINNED_IDS = {s.step_id for s in PINNED_STEPS if s.source == Source.USER_CURATED}


def test_split_off_pinned():
    unpinned, pinned = splits.split_off_pinned(PINNED_STEPS)
    assert set(pinned) == PINNED_IDS and len(pinned) == 9
    assert all(s.source == Source.PMECHDB for s in unpinned)
    assert frozenset({Source.USER_CURATED}) == splits.TRAIN_ONLY_SOURCES


@pytest.mark.parametrize("fn", [splits.split_by_pathway, splits.split_by_scaffold])
def test_pinned_steps_never_in_val_or_test(fn):
    unpinned = [s for s in PINNED_STEPS if s.step_id not in PINNED_IDS]
    for seed in range(10):
        sp = fn(PINNED_STEPS, seed=seed)
        _check_partition(sp)
        assert not PINNED_IDS & (set(sp.val) | set(sp.test))
        assert set(sp.train) >= PINNED_IDS
        alone = fn(unpinned, seed=seed)  # the unpinned steps split as if the others were absent
        assert (sp.val, sp.test, sp.excluded) == (alone.val, alone.test, alone.excluded)
        assert sp.train == sorted(alone.train + sorted(PINNED_IDS))


@pytest.mark.parametrize(
    "fn, inner",
    [
        (splits.split_by_pathway, splits._split_by_pathway),
        (splits.split_by_scaffold, splits._split_by_scaffold),
    ],
)
def test_pure_lists_are_unchanged(fn, inner):
    """No pinned step: the split equals the pre-pinning body (`_split_by_*`)."""
    for seed in range(3):
        assert fn(STEPS, seed=seed) == inner(STEPS, (0.8, 0.1, 0.1), seed)


@pytest.mark.parametrize("fn", [splits.split_by_pathway, splits.split_by_scaffold])
def test_all_pinned_gives_train_only(fn):
    every = [replace(s, source=Source.USER_CURATED) for s in STEPS]
    sp = fn(every, seed=0)
    assert sp.train == sorted(ALL_IDS) and not (sp.val or sp.test or sp.excluded)


def test_leave_one_class_out_with_pinned_steps():
    steps = list(STEPS)
    # a pinned held-out step, a pinned unlabelled step, a pinned step of another class
    for i, cls in ((1, "A"), (2, "B"), (4, None)):
        steps[i] = replace(
            steps[i],
            source=Source.USER_CURATED,
            mechanism_class=cls,
            pathway_id=f"user_curated-path-{i}",
        )
    pinned = {steps[i].step_id for i in (1, 2, 4)}
    sp = splits.split_leave_one_class_out(steps, "A", val_fraction=0.2, seed=0)
    _check_partition(sp)
    assert not pinned & (set(sp.val) | set(sp.test))
    assert steps[1].step_id in sp.excluded and steps[4].step_id in sp.excluded
    assert steps[2].step_id in sp.train
    unpinned = [s for s in steps if s.step_id not in pinned]
    alone = splits.split_leave_one_class_out(unpinned, "A", val_fraction=0.2, seed=0)
    assert (sp.val, sp.test) == (alone.val, alone.test)
    assert set(sp.train) == set(alone.train) | {steps[2].step_id}


def test_leave_one_class_out_pinned_shares_held_out_pathway():
    steps = [
        replace(s, pathway_id="user_curated-path-9") if i in (0, 3) else s
        for i, s in enumerate(STEPS)
    ]
    steps[0] = replace(steps[0], source=Source.USER_CURATED, mechanism_class="A")
    steps[3] = replace(steps[3], source=Source.USER_CURATED, mechanism_class="B")
    sp = splits.split_leave_one_class_out(steps, "A", seed=0)
    assert {steps[0].step_id, steps[3].step_id} <= set(sp.excluded)  # same pathway as an A step


def test_leave_one_class_out_all_pinned():
    every = [replace(s, source=Source.USER_CURATED) for s in STEPS]
    sp = splits.split_leave_one_class_out(every, "A", seed=0)
    _check_partition(sp)
    assert not (sp.val or sp.test)
    labelled = {s.step_id for s in every if s.mechanism_class in ("B", "C")}
    assert set(sp.train) <= labelled
    assert splits.leave_one_class_out_splits(every) == {}


def test_loco_folds_take_classes_from_unpinned_steps():
    steps = [
        replace(s, source=Source.USER_CURATED) if s.mechanism_class == "C" else s for s in STEPS
    ]
    folds = splits.leave_one_class_out_splits(steps)
    assert sorted(folds) == ["A", "B"]
    pinned = {s.step_id for s in steps if s.source == Source.USER_CURATED}
    for sp in folds.values():
        assert not pinned & (set(sp.val) | set(sp.test))


def test_split_ids_never_puts_pinned_in_val_or_test():
    from ksmech.arrows.steps import split_ids

    for strategy in ("random", "scaffold", "loco"):
        for seed in (0, 1):
            ids = split_ids(PINNED_STEPS, strategy, seed=seed)
            for name, members in ids.items():
                if name.endswith(("_val", "_test")):
                    assert not PINNED_IDS & set(members), (strategy, name)
            if strategy != "loco":
                assert set(ids[f"{strategy}_train"]) >= PINNED_IDS
