"""tblite's GFN2 basis as contracted Gaussians: analytic overlaps against tblite's own S."""

import numpy as np
import pytest

from ksmech.physics.gto import (
    ao_values,
    atom_ao_slices,
    element_shells,
    overlap_matrix,
    shell_overlap,
    slater_to_gauss,
)
from ksmech.qm.ao import gfn2_shells
from ksmech.qm.geometry import BOHR_ANG


def test_stewart_sto3g_1s():
    sh = slater_to_gauss(3, 1, 0, 1.0)
    assert np.allclose(sh.alpha, [2.227660584, 0.4057711562, 0.1098175104])
    raw = sh.coeff / (2 * sh.alpha / np.pi) ** 0.75
    assert np.allclose(raw, [0.1543289673, 0.5353281423, 0.4446345422])
    zeta = gfn2_shells()[1][0][2]
    assert np.allclose(element_shells(1)[0].alpha, sh.alpha * zeta**2)


def test_h1s_c2pz_at_1_09_matches_the_methane_record(sp_record):
    """The brief's check: the analytic H 1s / C 2pz overlap at 1.09 Angstrom along z equals the
    corresponding element of a methane record (its C-H bond projected on the bond axis)."""
    rec = sp_record("C")
    c = int(np.flatnonzero(rec.atomic_numbers == 6)[0])
    h = int(np.flatnonzero(rec.atomic_numbers == 1)[0])
    sl = atom_ao_slices(rec.atomic_numbers)
    bond = rec.coords_ang[h] - rec.coords_ang[c]
    u = bond / np.linalg.norm(bond)
    p_block = rec.S[sl[c].start + 1 : sl[c].start + 4, sl[h].start]  # (py, pz, px) x H 1s
    record_sigma = float(p_block @ u[[1, 2, 0]])
    c2p, h1s = element_shells(6)[1], element_shells(1)[0]
    analytic = shell_overlap(c2p, h1s, np.array([[0.0, 0.0, 1.09 / BOHR_ANG]]))[0, 1, 0]  # pz
    print(f"C-H {np.linalg.norm(bond):.4f} A: record {record_sigma:.5f}, analytic {analytic:.5f}")
    assert abs(abs(analytic) - abs(record_sigma)) < 1e-3
    exact = shell_overlap(c2p, h1s, np.array([[0.0, 0.0, np.linalg.norm(bond) / BOHR_ANG]]))
    assert abs(exact[0, 1, 0] - record_sigma) < 1e-8


@pytest.mark.parametrize("smiles", ["C", "CCl", "CC(C)=O", "C[Mg]Br", "Cl[Ti](Cl)(Cl)Cl", "CI"])
def test_full_overlap_matrix_matches_tblite(sp_record, smiles):
    rec = sp_record(smiles)
    S = overlap_matrix(rec.atomic_numbers, rec.coords_ang)
    assert S.shape == rec.S.shape and np.abs(S - rec.S).max() < 1e-8


def test_ao_values_follow_tblite_component_order():
    z, origin = np.array([17]), np.zeros((1, 3))
    on_z = ao_values(z, origin, np.array([[0.0, 0.0, 0.8]]))[0]
    on_x = ao_values(z, origin, np.array([[0.8, 0.0, 0.0]]))[0]
    p, d = on_z[1:4], on_z[4:9]  # Cl: s, p (y, z, x), d (xy, yz, z2, xz, x2-y2)
    assert p[1] > 0 and abs(p[0]) < 1e-14 and abs(p[2]) < 1e-14
    assert on_x[3] > 0 and abs(on_x[1]) < 1e-14
    assert d[2] > 0 and np.allclose(np.delete(d, 2), 0.0)
    assert on_x[4 + 4] > 0 and on_x[4 + 2] < 0  # x2-y2 positive on x, z2 negative there
