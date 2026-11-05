"""rmechdb's split strategies (open-shell plan P8 step 4): rc_molecule (the default) and
flower_given, on synthetic steps whose molecules are chains of 4+ carbons (non-trivial)."""

from __future__ import annotations

from collections import Counter

import pytest

from ksmech.arrows.steps import STRATEGIES, default_strategies, split_ids
from ksmech.data import splits
from ksmech.data.schema import ElementaryStep, Source

N_RC = 30


def step(sid: str, n: int, suffix: str | None, part: str | None = "train") -> ElementaryStep:
    """A step on the alcohol with n + 4 carbons (its own molecules, unless shared on purpose)."""
    chain = "C" * (n + 4)
    return ElementaryStep(
        step_id=sid,
        source=Source.RMECHDB if suffix else Source.PMECHDB,
        pathway_id=None,
        step_index=None,
        reactants_smiles=f"{chain}O",
        products_smiles=f"{chain}[O].[H]",
        flower_suffix=suffix,
        flower_part=part,
    )


def corpus() -> list[ElementaryStep]:
    rc = [step(f"rc{i:02d}", i, "RC") for i in range(N_RC)]
    rs = [step(f"rs{i:02d}", i, "RS") for i in range(N_RC)]  # each shares rc<i>'s molecules
    return rc + rs + [step("rs_own", 99, "RS")]


def molecules(steps, ids) -> set[str]:
    by_id = {s.step_id: s for s in steps}
    out: set[str] = set()
    for i in ids:
        out |= splits.step_molecules(by_id[i].reactants_smiles, by_id[i].products_smiles)
    return out


def test_rc_molecule_shares_no_species_between_training_and_rc_val_test():
    steps = corpus()
    sp = splits.split_rc_molecule(steps, seed=0)
    assert molecules(steps, sp.train).isdisjoint(molecules(steps, sp.val + sp.test))
    assert all(i.startswith("rc") for i in sp.val + sp.test)
    held = {i[2:] for i in sp.val + sp.test}
    assert sp.excluded == sorted(f"rs{k}" for k in held)  # LEAK_GUARD
    assert "rs_own" in sp.train
    assert len(sp.train) + len(sp.val) + len(sp.test) + len(sp.excluded) == len(steps)


def test_rc_molecule_is_seeded_deterministic_and_frozen():
    steps = corpus()
    a, b = splits.split_rc_molecule(steps, seed=0), splits.split_rc_molecule(steps, seed=0)
    assert a == b
    frozen = {i: p for p in ("train", "val", "test") for i in getattr(a, p) if i.startswith("rc")}
    more = [*steps, step("rc_new", 50, "RC")]
    c = splits.split_rc_molecule(more, seed=0, frozen=frozen)
    for part in ("train", "val", "test"):
        assert set(getattr(a, part)) & frozen.keys() <= set(getattr(c, part))
    assert sum("rc_new" in getattr(c, p) for p in ("train", "val", "test")) == 1


def test_flower_given_keeps_flowers_test_and_draws_val_from_rc_train_groups():
    steps = [step(f"rc{i:02d}", i, "RC") for i in range(N_RC)]
    steps += [step("rs_tr", 70, "RS"), step("rs_te", 71, "RS", "test")]
    steps += [step("rc_te", 72, "RC", "test"), step("rc_both", 73, "RC", "train+test")]
    sp = splits.split_flower_given(steps, seed=0)
    assert sp.test == ["rc_both", "rc_te", "rs_te"]
    assert sp.val and all(i.startswith("rc") and i in {s.step_id for s in steps} for i in sp.val)
    assert "rs_tr" in sp.train and not sp.excluded
    assert len(sp.train) + len(sp.val) + len(sp.test) == len(steps)


def test_flower_splits_need_flower_columns():
    plain = [step(f"p{i}", i, None) for i in range(5)]
    for fn in (splits.split_rc_molecule, splits.split_flower_given):
        with pytest.raises(ValueError, match="flower_suffix"):
            fn(plain)


def test_the_strategies_are_dispatched_by_name_and_count_leak_guard():
    assert {"rc_molecule", "flower_given"} <= set(STRATEGIES)
    assert default_strategies("rmechdb") == ("rc_molecule", "flower_given")
    assert default_strategies("pmechdb") == ("molecule", "scaffold", "loco")
    assert splits.default_split("rmechdb") is splits.split_rc_molecule
    excluded: Counter = Counter()
    ids = split_ids(corpus(), "rc_molecule", excluded=excluded)
    assert set(ids) == {"rc_molecule_train", "rc_molecule_val", "rc_molecule_test"}
    assert excluded["rc_molecule"] == len(ids["rc_molecule_val"]) + len(ids["rc_molecule_test"])
