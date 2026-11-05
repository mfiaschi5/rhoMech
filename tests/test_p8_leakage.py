"""The rmechdb leakage audit (open-shell plan P8 step 6): every overlapping step id, in both
directions, with species matched through the identity key v2."""

from __future__ import annotations

from ksmech.data import leakage


def test_species_match_through_key_v2_and_trivial_ones_are_dropped():
    assert leakage.species_key("C=C[CH]CC") == leakage.species_key("[CH2]C=CCC")
    assert leakage.species_key("CC[O]") is None  # 3 heavy atoms: trivial
    assert leakage.side_species("[CH3:1][CH2:2][CH2:3][CH2:4][OH:5].[OH2:6]") == frozenset(
        [leakage.species_key("CCCCO")]
    )


def test_the_audit_lists_every_overlapping_id_in_both_directions():
    a, b, c, d = (leakage.species_key(s) for s in ("CCCCO", "CCCCCO", "CCCCCCO", "CCCCCCCO"))
    species = {
        "rc_val": frozenset([a]),
        "rc_test": frozenset([d]),
        "rm_train1": frozenset([b]),
        "rm_train2": frozenset([c]),
        "pm_train1": frozenset([a]),
        "pm_train2": frozenset([c]),
        "pm_val": frozenset([b]),
        "pm_test": frozenset(),
    }
    report = leakage.audit(
        rc={"val": ["rc_val"], "test": ["rc_test"]},
        rm_train=["rm_train1", "rm_train2"],
        closed={"train": ["pm_train1", "pm_train2"], "val": ["pm_val"], "test": ["pm_test"]},
        species=species,
        others={"reactmech_any": frozenset([d]), "mech_uspto": None},
        reactions={"rc_val": "x>>y", "rc_test": "z>>w"},
        other_reactions={"pmechdb": frozenset(["x>>y"])},
    )
    assert report["rc_heldout_shares_pmechdb_train"] == {"n": 1, "ids": ["rc_val"]}
    assert report["rmechdb_train_shares_closed_base_heldout"]["ids"] == ["rm_train1"]
    assert report["pmechdb_train_shares_rc_heldout"]["ids"] == ["pm_train1"]
    assert report["closed_base_heldout_shares_rmechdb_train"]["ids"] == ["pm_val"]
    assert report["rc_heldout_shares_reactmech_any"]["ids"] == ["rc_test"]
    assert report["rc_heldout_shares_mech_uspto"] is None  # the raw file is absent
    assert report["rc_heldout_reaction_in_pmechdb"]["ids"] == ["rc_val"]
    assert set(leakage.COHORTS) == {
        "leak_guard_joint", "rc_heldout_overlap", "rc_heldout_shared",
        "closed_base_rmechdb_overlap",
    }  # fmt: skip


# ---- the R-shape breakdown of the cohort reports (open-shell plan, evaluation (6))


def _sample(r_bonds, r_lone, bonds):
    from types import SimpleNamespace

    import numpy as np

    from ksmech.arrows.delta_be import ReactionMatrix

    r = ReactionMatrix(tuple(r_bonds), tuple(r_lone))
    label = SimpleNamespace(variants=(r,))
    return SimpleNamespace(edit_label=label, bonds=np.array(bonds, dtype=np.int16).reshape(-1, 3))


def test_r_shapes_from_r_and_the_reactant_bond_orders():
    from ksmech.eval.cohorts import by_r_shape, r_shape

    homolysis = _sample([(0, 1, -1)], [(0, 1), (1, 1)], [(0, 1, 1)])
    recombination = _sample([(0, 1, 1)], [(0, -1), (1, -1)], [])
    # X. + C1=C2 -> X-C1-C2.
    addition = _sample([(0, 1, 1), (1, 2, -1)], [(0, -1), (2, 1)], [(1, 2, 2)])
    # X. + H-Y -> X-H + Y.
    abstraction = _sample([(0, 1, 1), (1, 2, -1)], [(0, -1), (2, 1)], [(1, 2, 1)])
    # X-C1-C2. -> X. + C1=C2
    beta = _sample([(0, 1, -1), (1, 2, 1)], [(0, 1), (2, -1)], [(0, 1, 1), (1, 2, 1)])
    polar = _sample([(0, 1, 1)], [(0, -2)], [])
    shapes = [r_shape(s) for s in (homolysis, recombination, addition, abstraction, beta, polar)]
    assert shapes == [
        "homolysis", "recombination", "addition", "abstraction_sh2", "beta_scission", "other"
    ]  # fmt: skip
    rows = [{"step_id": "a", "rank": 0}, {"step_id": "b", "rank": 4}]
    report = by_r_shape(rows, {"a": homolysis, "b": homolysis})
    assert report == {"homolysis": {"top1": 0.5, "top3": 0.5, "top5": 1.0, "n": 2}}
