"""Real data: the frozen split snapshots of PMechDB and ReactMech reproduce today's split
views id for id when the strategy is re-run through the frozen path (open-shell plan P0)."""

import pytest

from ksmech.arrows import steps as st
from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_training_steps, steps_path

pytestmark = pytest.mark.realdata

FROZEN = [
    ("pmechdb", "molecule"),
    ("pmechdb", "scaffold"),
    ("reactmech", "molecule"),
    ("reactmech", "scaffold"),
    ("reactmech", "loco"),
    ("reactmech", "loco_all"),
]


@pytest.mark.parametrize(("source", "strategy"), FROZEN)
def test_a_frozen_split_reproduces_its_parts(source, strategy):
    frozen = st.load_frozen_split(source, strategy, DEFAULT_PROCESSED_DIR)
    if frozen is None:
        pytest.skip(f"no split_frozen_{strategy}.json for {source} (`ksmech-arrows freeze-splits`)")
    steps = read_training_steps(steps_path(source, DEFAULT_PROCESSED_DIR))
    again = st.split_ids(steps, strategy, frozen=frozen)
    gone = st.frozen_excluded(frozen, steps)
    for name, ids in frozen.items():
        have = set(again[name])
        assert sum(i in have for i in ids) + gone.get(name, 0) == len(ids), name
