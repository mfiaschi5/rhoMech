"""`eval.decode`: B + R on a stored sample's Lewis structure, the exact-R hit rank and the
top-k report of the engine's beam decoder."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
from rdkit import Chem

from ksmech.arrows.delta_be import ReactionMatrix
from ksmech.data.resonance import sigma_key
from ksmech.eval.decode import decode_report, hit_rank, lewis_apply
from ksmech.model.decode import Completion


def _sample(z, q, bonds, lone_pairs, label=None):
    return SimpleNamespace(
        atom_categorical=np.stack([np.asarray(z), np.asarray(q)], axis=1).astype(np.int64),
        bonds=np.asarray(bonds, dtype=np.int16).reshape(-1, 3),
        lone_pairs=np.asarray(lone_pairs, dtype=np.int8),
        edit_label=label,
    )


# chloromethane + hydroxide: C0 Cl1 H2 H3 H4 O5 H6
MECL_OH = _sample(
    [6, 17, 1, 1, 1, 8, 1],
    [0, 0, 0, 0, 0, -1, 0],
    [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (5, 6, 1)],
    [0, 3, 0, 0, 0, 3, 0],
)
SN2 = (((0, 5, 1), (0, 1, -1)), ((1, 2), (5, -2)))


def test_sn2_gives_methanol_and_chloride():
    key, why = lewis_apply(MECL_OH)(*SN2)
    assert why is None
    assert key == sigma_key(Chem.MolFromSmiles("CO.[Cl-]"))


def test_a_missing_bond_cannot_break():
    assert lewis_apply(MECL_OH)(((1, 5, -1), (0, 5, 1)), ()) == (None, "BOND_ORDER")


def test_a_fifth_bond_on_carbon_exceeds_the_octet():
    assert lewis_apply(MECL_OH)(((0, 5, 1),), ((5, -2),)) == (None, "OCTET_EXCEEDED")


def test_a_new_dianion_on_a_nonmetal_is_refused():
    assert lewis_apply(MECL_OH)(((5, 6, -1),), ((5, 2),)) == (None, "FORMAL_CHARGE_RANGE")


def test_moving_only_the_pi_bond_is_resonance():
    formaldehyde = _sample(
        [6, 8, 1, 1], [0, 0, 0, 0], [(0, 1, 2), (0, 2, 1), (0, 3, 1)], [0, 2, 0, 0]
    )
    assert lewis_apply(formaldehyde)(((0, 1, -1),), ((1, 2),)) == (None, "RESONANCE_ONLY_STEP")


def _completion(bonds, lone, logp=0.0):
    return Completion(logp, tuple(bonds), tuple(lone), (0, 5, 0))


def _label(*variants):
    return SimpleNamespace(variants=[ReactionMatrix(b, lo) for b, lo in variants])


def test_hit_rank_finds_any_variant_of_the_class():
    wrong = _completion([(0, 5, 1), (0, 2, -1)], [(5, -2)])
    right = _completion([(0, 1, -1), (0, 5, 1)], [(5, -2), (1, 2)])
    label = _label(([(0, 5, 1), (0, 1, -1)], [(1, 2), (5, -2)]))
    assert hit_rank([wrong, right], label) == 1
    assert hit_rank([wrong], label) is None


def test_report_counts_hits_misses_and_unlabelled_samples():
    label = _label(SN2)
    right = _completion(*SN2)
    wrong = _completion([(0, 5, 1), (0, 2, -1)], [(5, -2)])
    found = {"a": [right], "b": [wrong, right], "c": [wrong], "d": []}
    samples = [SimpleNamespace(step_id=k, edit_label=label) for k in found]
    samples.append(SimpleNamespace(step_id="e", edit_label=None))

    def decode(sample, skips: Counter):
        return found[sample.step_id]

    r = decode_report(samples, decode)
    assert (r["n"], r["top1"], r["top3"], r["not_found"]) == (4, 0.25, 0.5, 2)
    assert r["skips"] == {"DECODE_NOTHING_FOUND": 1, "NO_EDIT_LABEL": 1}
