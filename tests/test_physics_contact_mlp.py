"""The descriptor contact kernel (`kernel=mlp`): fit, persistence, use in the KS terms."""

import dataclasses

import numpy as np
import pytest

from ksmech.orbitals.classify import OrbitalType
from ksmech.orbitals.tokens import build_molecule_tokens
from ksmech.physics.contact_kernel import ContactConfig, ContactKernel
from ksmech.physics.contact_mlp import (
    KINDS,
    descriptor_matrix,
    fit_contact_mlp,
    load_contact_mlp,
    samples_from_tokens,
    save_contact_mlp,
)
from ksmech.physics.contact_table import contact_rows
from ksmech.physics.ks_terms import ks3_contact
from tests.token_select import highest_lone_pair, sigma_star

SMILES = ["CCO", "CC(=O)C", "CCCl", "OCCCCCl", "CC(=O)OC", "C=CC=O", "CCN", "CCCC"]


@pytest.fixture(scope="module")
def fitted(sp_record):
    tokens = [build_molecule_tokens(sp_record(s)) for s in SMILES]
    samples = samples_from_tokens(tokens)
    return tokens, samples, fit_contact_mlp(samples, hidden=(16,), epochs=5, val_fraction=0.25)


def test_samples_are_the_table_samples(fitted):
    tokens, samples, _ = fitted
    for kind in KINDS:
        X, y, mol = samples[kind].arrays()
        n = sum(sum(k == kind for k, *_ in contact_rows(t)) for t in tokens)
        assert len(y) == n == len(samples[kind].kappa) == len(mol) and X.shape == (n, 2 * 23)
        assert (y >= 0).all() and set(mol) <= set(range(len(tokens)))


def test_fit_reports_held_out_scores_for_mlp_and_table(fitted):
    *_, model = fitted
    for kind in KINDS:
        meta = model.meta[kind]
        n_with_samples = len(set(fitted[1][kind].arrays()[2]))  # molecules with such contacts
        assert meta["n_train"] > 0 and meta["n_molecules"] == n_with_samples >= 2
        for which in ("mlp_held_out", "table_held_out"):
            assert meta[which]["n"] > 0 and np.isfinite(meta[which]["median_abs_log10_error"])


def test_round_trip_and_positive_predictions(fitted, tmp_path):
    tokens, samples, model = fitted
    back = load_contact_mlp(save_contact_mlp(model, tmp_path / "mlp.npz"))
    X = samples["occ_virt"].arrays()[0][:50]
    assert np.allclose(back.predict("occ_virt", X), model.predict("occ_virt", X), rtol=1e-12)
    assert (model.predict("occ_virt", X) > 0).all()
    assert back.meta == model.meta
    with pytest.raises(FileNotFoundError, match="contact-mlp"):
        load_contact_mlp(tmp_path / "none.npz")


def test_mlp_kernel_amplitude_and_ks3(fitted, sp_record):
    *_, model = fitted
    kernel = ContactKernel(None, ContactConfig(kernel="mlp"), mlp=model)
    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    occ, virt = methanol.occupied(), chloromethane.virtual()
    A = kernel.amplitude_matrix(occ, methanol, virt, chloromethane)
    S = model.pair_matrix("occ_virt", descriptor_matrix(occ), descriptor_matrix(virt))
    w_r = np.array([o.descriptors.center_weights[0] for o in occ])
    w_s = np.array([o.descriptors.center_weights[0] for o in virt])
    assert np.allclose(A, S * w_r[:, None] * w_s[None, :]) and (A > 0).all()
    lone, c_cl = highest_lone_pair(methanol, "O"), sigma_star(chloromethane, {"C", "Cl"})
    assert ks3_contact(lone, methanol, c_cl, chloromethane, kernel) < 0


def test_mlp_kernel_reads_no_orbital_type(fitted, sp_record):
    *_, model = fitted
    t = build_molecule_tokens(sp_record("CC(=O)C"))
    relabelled = [dataclasses.replace(o, type=OrbitalType.UNKNOWN) for o in t.orbitals]
    V, V2 = descriptor_matrix(t.orbitals), descriptor_matrix(relabelled)
    assert np.array_equal(V, V2)
    assert np.array_equal(model.pair_matrix("occ_occ", V, V), model.pair_matrix("occ_occ", V2, V2))
