"""Real data: P4's tagged QM records (open-shell plan P4 steps 2, 3 and 6).

Every species of PMechDB that `data.spin` reassigns has a record under its tagged method that
verifies against that method, with its Wiberg sidecar and tokens the sample builder finds, or a
counted failure reason; every other species keeps its untagged key, and the only untagged
records that a tag now bypasses are those of the species the species table pairs.
"""

import random

import pandas as pd
import pytest

from ksmech.data import spin
from ksmech.data.io import DEFAULT_PROCESSED_DIR
from ksmech.qm.records import cache_path, is_valid_cache_file, verify_record
from ksmech.qm.spin import method_for, spin_tag
from ksmech.qm.xtb_calc import QMConfig

pytestmark = pytest.mark.realdata

BASE = QMConfig().method
MANIFEST = DEFAULT_PROCESSED_DIR / "pmechdb" / "qm_manifest.parquet"
# the P4 run (decisions.md, 2026-11-03): tagged species of PMechDB by tag
TAGGED = {"spin-v1-u0": 8, "spin-v1-u1": 28, "spin-v1-u5": 13}
# species whose untagged record (computed at the drawn count, a triplet) a tag now bypasses:
# each is a species the species table pairs
PAIRED_BEFORE = {
    "Cl[C]Cl", "Br[C]Br", "[C]=CC", "[C]=C(C)C", "[C]=C(c1ccccc1)c1ccccc1", "[C]=O",
    "CC[B-](CC)(CC)[O+][O]", "C=[O+][Sn-]",
}  # fmt: skip


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    if not MANIFEST.exists():
        pytest.skip("no PMechDB QM manifest")
    return pd.read_parquet(MANIFEST)


@pytest.fixture(scope="module")
def tagged(manifest) -> dict[str, str]:
    species = set(manifest.loc[manifest["conformer_id"] >= 0, "smiles"])
    return {s: t for s in species if (t := spin_tag(s))}


def test_the_tagged_species_are_the_ones_p4_computed(tagged):
    from collections import Counter

    assert dict(Counter(tagged.values())) == TAGGED


def test_every_tagged_species_has_a_verified_record_or_a_counted_reason(manifest, tagged):
    for smiles in tagged:
        method = method_for(smiles, BASE)
        rows = manifest[(manifest["smiles"] == smiles) & (manifest["method"] == method)]
        assert len(rows) == 1, smiles
        row = rows.iloc[0]
        if row["status"] == "ok":
            path = cache_path(QMConfig().cache_dir, smiles, 0, method)
            header = verify_record(path, smiles, 0, method)
            assert not header.code_dirty, smiles
        else:
            assert row["reason"], smiles


def test_a_tagged_record_has_its_sidecar_and_tokens_for_the_sample_builder(manifest, tagged):
    from ksmech.arrows.reactants import TokenCache

    loader = TokenCache()
    cache = QMConfig().cache_dir
    ok = [s for s in tagged if is_valid_cache_file(cache_path(cache, s, 0, method_for(s, BASE)))]
    assert ok
    for smiles in ok:
        tokens = loader(smiles)
        assert tokens is not None, (smiles, loader.missing_reason(smiles))
        assert tokens.method == method_for(smiles, BASE)


def test_no_untagged_record_changes_key_but_the_paired_species(manifest):
    untagged_ok = manifest[(manifest["status"] == "ok") & (manifest["method"] == BASE)]
    moved = {s for s in untagged_ok["smiles"] if method_for(s, BASE) != BASE}
    assert moved == PAIRED_BEFORE
    for s in moved:
        a = spin.assign_smiles(s)
        assert a.source == spin.SPECIES and a.unpaired == 0


def test_qmugs_keeps_every_key():
    """A sample of QMugs' given-geometry records (0 radicals, no odd metal): no tag."""
    from ksmech.data import qmugs

    path = qmugs.PROCESSED_DIR / "qm_manifest.parquet"
    if not path.exists():
        pytest.skip("no QMugs manifest")
    smiles = sorted(set(pd.read_parquet(path, columns=["smiles"])["smiles"].dropna()))
    picked = random.Random(0).sample(smiles, min(5000, len(smiles)))
    assert not [s for s in picked if spin_tag(s)]
