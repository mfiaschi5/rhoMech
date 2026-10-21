"""The frozen fixed-distance (2.3 Angstrom) cap-and-roll kernel, `contact/legacy_fixed23.py`:
invariant to a rigid rotation of either record, and pinned to stored outputs."""

import copy
from pathlib import Path

import numpy as np
import pytest

from ksmech.orbitals.tokens import build_molecule_tokens
from ksmech.physics.contact import legacy_fixed23 as K
from ksmech.physics.gto import element_shells

PINNED = Path(__file__).parent / "fixtures" / "legacy_fixed23_pinned.npz"
PAIRS = {  # name -> (donor SMILES, acceptor SMILES, intra)
    "methoxide_chloromethane": ("C[O-]", "CCl", False),
    "ethene_formaldehyde": ("C=C", "C=O", False),
    "acrolein_intra": ("C=CC=O", "C=CC=O", True),
}


def rotated(t, R: np.ndarray, tag: str):
    """The tokens of a record rigidly rotated by R: positions, every LMO (shell by shell), S, F."""
    u = copy.deepcopy(t)
    u.smiles = t.smiles + tag  # a new identity: the kernel's memory caches key on it
    for a in u.atoms:
        a.position = np.asarray(a.position) @ R.T
    nao = len(t.ao_atom)
    D, k = np.zeros((nao, nao)), 0
    for a in t.atoms:
        for sh in element_shells(a.atomic_number):
            D[k : k + sh.n_sph, k : k + sh.n_sph] = K.shell_rotation(sh.l, R)
            k += sh.n_sph
    for o in u.orbitals:
        o.coeff_vector = D @ o.coeff_vector
    u.S, u.F = D @ t.S @ D.T, D @ t.F @ D.T
    return u


def _random_rotation(rng) -> np.ndarray:
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    return R if np.linalg.det(R) > 0 else -R


def _block(t_F, t_G, intra: bool):
    rows = [k for k, o in enumerate(t_F.orbitals) if o.kind == "occupied"]
    cols = [k for k, o in enumerate(t_G.orbitals) if o.kind == "virtual"]
    return K.block_contacts(t_F, rows, t_G, cols, K.OrientedConfig(distance=2.3), intra=intra)


def _tokens(sp_record, name):
    donor, acceptor, intra = PAIRS[name]
    t_F = build_molecule_tokens(sp_record(donor))
    t_G = t_F if intra else build_molecule_tokens(sp_record(acceptor))
    return t_F, t_G, intra


@pytest.mark.parametrize("name", ["methoxide_chloromethane", "ethene_formaldehyde"])
def test_rotating_either_record_changes_nothing(sp_record, name):
    """Each fragment rotated by its own random R (the old lab-frame grid moved |S| by up to 4.8%
    and beta by 13%, and changed 25% of the argmax sites)."""
    t_F, t_G, intra = _tokens(sp_record, name)
    ref = _block(t_F, t_G, intra)
    rng = np.random.default_rng(7)
    for trial in range(2):
        R1, R2 = _random_rotation(rng), _random_rotation(rng)
        got = _block(rotated(t_F, R1, f"|F{trial}"), rotated(t_G, R2, f"|G{trial}"), intra)
        assert np.allclose(got.S, ref.S, rtol=1e-7, atol=1e-10)
        assert np.allclose(got.beta, ref.beta, rtol=1e-7, atol=1e-9)
        assert np.array_equal(got.sites, ref.sites) and np.array_equal(got.clash, ref.clash)


def test_rotating_one_molecule_changes_nothing_intramolecular(sp_record):
    t, _, _ = _tokens(sp_record, "acrolein_intra")
    ref = _block(t, t, True)
    u = rotated(t, _random_rotation(np.random.default_rng(3)), "|rot")
    got = _block(u, u, True)
    assert np.allclose(got.S, ref.S, rtol=1e-7, atol=1e-10)
    assert np.array_equal(got.sites, ref.sites) and np.array_equal(got.clash, ref.clash)


def test_outputs_are_pinned(sp_record):
    """The frozen baseline: any change to these numbers needs a new ORIENTED_VERSION and a new
    pinned file (regenerate with `python tests/test_legacy_fixed23.py`)."""
    pinned = np.load(PINNED)
    assert int(pinned["version"]) == K.ORIENTED_VERSION
    for name in PAIRS:
        c = _block(*_tokens(sp_record, name))
        assert np.allclose(c.S, pinned[f"{name}/S"], rtol=1e-9, atol=1e-12), name
        assert np.allclose(c.beta, pinned[f"{name}/beta"], rtol=1e-9, atol=1e-11), name
        assert np.array_equal(c.sites, pinned[f"{name}/sites"]), name
        assert np.array_equal(c.clash, pinned[f"{name}/clash"]), name
