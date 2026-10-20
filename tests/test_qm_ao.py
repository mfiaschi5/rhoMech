"""AO layout from tblite's own GFN2 shell list (transition metals list d first)."""

import dataclasses

import numpy as np

from ksmech.qm.ao import ao_layout, aos_per_element, element_ao_l, gfn2_shells
from ksmech.qm.records import load_record, save_record


def test_shell_order_comes_from_tblite():
    assert element_ao_l(1) == (0,)
    assert element_ao_l(6) == (0, 1, 1, 1)
    assert element_ao_l(17) == (0, 1, 1, 1, 2, 2, 2, 2, 2)
    assert element_ao_l(22) == (2, 2, 2, 2, 2, 0, 1, 1, 1)  # Ti: 3d, 4s, 4p
    assert element_ao_l(29) == (2, 2, 2, 2, 2, 0, 1, 1, 1)  # Cu
    n, ell, zeta, ng = gfn2_shells()[1][0]
    assert (n, ell, ng) == (1, 0, 3) and zeta > 1.0
    for z in (1, 6, 8, 12, 17, 22, 29, 35, 53):
        assert len(element_ao_l(z)) == aos_per_element(z)


def test_layout_is_atom_by_atom():
    ao_atom, ao_l = ao_layout(np.array([6, 22, 1]))
    assert list(ao_atom) == [0] * 4 + [1] * 9 + [2]
    assert list(ao_l) == [0, 1, 1, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 0]


def test_stale_layout_is_recomputed_on_load(sp_record, tmp_path):
    rec = sp_record("Cl[Ti](Cl)(Cl)Cl")
    good = (rec.ao_atom.copy(), rec.ao_l.copy())
    stale_l = np.concatenate([(0, 1, 1, 1, 2, 2, 2, 2, 2) for _ in rec.atomic_numbers])
    stale = dataclasses.replace(rec, ao_l=stale_l.astype(np.int32))
    back = load_record(save_record(stale, tmp_path / "ticl4.npz"))
    assert np.array_equal(back.ao_l, good[1]) and np.array_equal(back.ao_atom, good[0])
