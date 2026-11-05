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
