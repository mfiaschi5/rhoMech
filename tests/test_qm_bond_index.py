"""Wiberg bond indices of the Lewis bonds (`qm.bond_index`) on the fixture molecules."""

import numpy as np
import pytest
from rdkit import Chem

from ksmech.data import provenance as prov
from ksmech.qm.bond_index import (
    bond_index_path,
    lewis_bond_indices,
    load_bond_indices,
    save_bond_indices,
    wiberg_matrix,
)
from ksmech.qm.records import save_record
from tests.fixtures.molecules import MOLECULES


def test_minimal_basis_h2_has_index_one():
    P = np.ones((2, 2))  # two electrons in the bonding combination of an orthonormal basis
    W = wiberg_matrix(np.eye(2), P, np.array([0, 1]), 2)
    assert W[0, 1] == pytest.approx(1.0) and W[0, 0] == 0.0


def _by_elements(rec, pairs, wbi):
    z = rec.atomic_numbers
    sym = Chem.GetPeriodicTable().GetElementSymbol
    out = {}
    for (i, j), w in zip(pairs, wbi, strict=True):
        out.setdefault(tuple(sorted((sym(int(z[i])), sym(int(z[j]))))), []).append(float(w))
    return out


@pytest.mark.parametrize("name", list(MOLECULES))
def test_every_lewis_bond_has_an_index_near_its_order(name, sp_record):
    rec = sp_record(MOLECULES[name])
    pairs, wbi = lewis_bond_indices(rec)
    mol = Chem.AddHs(Chem.MolFromSmiles(MOLECULES[name]))
    assert len(pairs) == mol.GetNumBonds() and (pairs[:, 0] < pairs[:, 1]).all()
    assert (wbi > 0.5).all() and (wbi < 3.2).all()
    by = _by_elements(rec, pairs, wbi)
    if name == "acetonitrile":
        assert max(by[("C", "N")]) > 2.4  # the triple bond
    if name == "benzene":
        assert np.allclose(by[("C", "C")], np.mean(by[("C", "C")]), atol=0.02)
        assert 1.3 < np.mean(by[("C", "C")]) < 1.6  # aromatic, between single and double
    if name in ("acetone", "acetaldehyde"):
        assert max(by[("C", "O")]) > 1.6  # the carbonyl


def test_the_sidecar_round_trips_and_goes_stale_with_its_record(sp_record, tmp_path):
    rec = sp_record("CCl")
    record_path = save_record(rec, tmp_path / "rec.npz")
    out = save_bond_indices(rec, record_path, tmp_path / "wbi")
    assert out == bond_index_path(tmp_path / "wbi", rec.smiles, rec.conformer_id, rec.method)
    pairs, wbi = lewis_bond_indices(rec)
    loaded = load_bond_indices(out)
    assert loaded == {(int(i), int(j)): float(w) for (i, j), w in zip(pairs, wbi, strict=True)}
    rec.energy_ha += 1.0  # the record is recomputed: the indices built from it are stale
    save_record(rec, record_path)
    with pytest.raises(prov.StaleArtifact):
        load_bond_indices(out)


def test_tokens_without_their_sidecar_count_as_missing(token_loader, sp_record, tmp_path):
    from ksmech.arrows.reactants import MissingTokens, ReactantTokens, TokenCache
    from ksmech.orbitals.build import tokens_path
    from ksmech.orbitals.tokens import save_tokens

    rec = sp_record("CCl")
    cache = TokenCache(tmp_path / "tokens", bond_index_dir=tmp_path / "wbi")
    save_tokens(
        token_loader(rec.smiles), tokens_path(cache.tokens_dir, rec.smiles, 0, cache.method)
    )
    assert cache(rec.smiles) is None and "bond-index" in cache.stale[rec.smiles]
    assert cache.missing_reason(rec.smiles) == "NO_BOND_INDEX"
    assert cache.missing_reason("O") == "MISSING_TOKENS"  # never served: no file
    with pytest.raises(MissingTokens) as exc:
        ReactantTokens.from_smiles("[CH3:1][Cl:2].[OH2:3]", cache)
    assert exc.value.reasons == {rec.smiles: "NO_BOND_INDEX", "O": "MISSING_TOKENS"}
    assert exc.value.reason == "NO_BOND_INDEX"  # what a step with both is counted as
    save_bond_indices(rec, save_record(rec, tmp_path / "rec.npz"), tmp_path / "wbi")
    fresh = TokenCache(tmp_path / "tokens", bond_index_dir=tmp_path / "wbi")
    tokens = fresh(rec.smiles)
    assert tokens is not None and tokens.bond_index == load_bond_indices(
        bond_index_path(tmp_path / "wbi", rec.smiles, 0, rec.method)
    )
    rec.energy_ha += 1.0  # the record is recomputed: its sidecar is stale, never "missing"
    save_record(rec, tmp_path / "rec.npz")
    stale = TokenCache(tmp_path / "tokens", bond_index_dir=tmp_path / "wbi")
    assert stale(rec.smiles) is None and stale.missing_reason(rec.smiles) == prov.STALE_ARTIFACT
    with pytest.raises(MissingTokens) as exc:
        ReactantTokens.from_smiles("[CH3:1][Cl:2].[OH2:3]", stale)
    assert exc.value.reason == prov.STALE_ARTIFACT
