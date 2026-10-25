"""keys.py and pools.py: species, pool, state and edge keys; root states; persistence;
preflight codes. RDKit only (the featurizer preflight reads the tblite parameter table)."""

import json
import re

import pytest
from rdkit import Chem

from ksmech.serve import keys
from ksmech.serve.contracts import Limits, PoolState
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.pools import (
    SpeciesInput,
    initial_state,
    load_state,
    preflight_inputs,
    save_state,
    state_from_mapped,
    state_from_mol,
    state_path,
)
from tests.fixtures.molecules import MOLECULES
from tests.test_serve_fixtures import no_xtb, serve_settings  # noqa: F401 (fixtures)

IONS = {"methoxide": "C[O-]", "chloromethane": "CCl", "hydroxide": "[OH-]"}
ALL = {**MOLECULES, **IONS}
LIMITS = Limits(6, 4, 150, 10, 12, 200)
URL_SAFE = re.compile(r"^[ps][0-9a-f]{20}$")


@pytest.mark.parametrize("name", sorted(ALL))
def test_species_key_is_map_free_and_stable(name):
    smi = ALL[name]
    key, flag = keys.species_key_of(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))
    assert flag == "OK"  # the sigma key is exact: never truncated
    assert ":" not in key  # no atom maps
    skeleton, charge = key.rsplit("|", 1)  # the sigma skeleton, then the total charge
    assert Chem.MolFromSmiles(skeleton, sanitize=False) is not None
    assert int(charge) == Chem.GetFormalCharge(Chem.MolFromSmiles(smi))
    mapped = Chem.MolFromSmiles(smi)
    for a in mapped.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)
    (frag,) = keys.fragment_keys(mapped)
    assert frag.species_key == key  # maps on the input do not change the key
    assert keys.species_key_of(frag.token_key) == (key, flag)


def test_resonance_invariance():
    a = keys.side_keys("C=C[O-]")
    b = keys.side_keys("[CH2-]C=O")
    assert a.token_keys != b.token_keys  # two token (QM cache) keys
    assert a.species_keys == b.species_keys  # one species key
    assert a.pool_key == b.pool_key


def test_side_keys_mapped_and_unmapped():
    mapped = "[CH3:1][O-:2].[CH3:3][Cl:4]"
    s = keys.side_keys(mapped)
    assert s.token_keys == ["C[O-]", "CCl"]
    assert s.pool_key == "[CH3][Cl]|0.[CH3][O]|-1"  # sigma keys
    assert keys.side_keys("CCl.C[O-]").pool_key == s.pool_key
    assert s.flags == ["OK", "OK"]
    with pytest.raises(ServeError) as e:
        keys.side_keys("C1CC")
    assert e.value.code == Code.SPECIES_KEY_FAILED
    with pytest.raises(ServeError) as e:
        keys.species_key_of("not a smiles")
    assert e.value.code == Code.SPECIES_KEY_FAILED


def test_pool_key_and_id():
    k1 = keys.pool_key_of(["C[O-]", "CCl", "C[O-]"])
    k2 = keys.pool_key_of(["CCl", "C[O-]", "C[O-]"])
    assert k1 == k2 == "CCl.C[O-].C[O-]"
    assert keys.split_pool_key(k1) == {"C[O-]": 2, "CCl": 1}
    assert keys.split_pool_key("") == {}
    pid = keys.pool_id(k1)
    assert pid == keys.pool_id(k2) and URL_SAFE.match(pid)
    assert keys.pool_id("CCl") != pid


def test_state_id_sensitive_to_maps():
    a = keys.state_id("[CH3:1][Cl:2]")
    b = keys.state_id("[CH3:2][Cl:1]")
    assert a != b and URL_SAFE.match(a) and URL_SAFE.match(b)


def test_edge_id():
    matrix = {"bonds": [[1, 5, 1], [5, 6, -1]], "lone": [[1, -2], [6, 2]]}
    e = keys.edge_id("predicted", "s" + "0" * 20, matrix)
    assert re.match(r"^e[0-9a-f]{16}$", e)
    assert e == keys.edge_id("predicted", "s" + "0" * 20, [matrix["bonds"], matrix["lone"]])
    c = keys.edge_id("corrected", "s" + "0" * 20, matrix)
    assert c.startswith("c") and c[1:] != e[1:]
    assert keys.edge_id("predicted", "s" + "1" * 20, matrix) != e
    other = {"bonds": [[1, 5, 1], [5, 7, -1]], "lone": [[1, -2], [7, 2]]}
    assert keys.edge_id("predicted", "s" + "0" * 20, other) != e
    with pytest.raises(ValueError):
        keys.edge_id("other", "s" + "0" * 20, matrix)


# ---- root states


def _check_explicit_and_mapped(state: PoolState):
    from ksmech.data import chem

    mol = chem.parse_mapped(state.mapped_smiles)
    assert mol.GetNumAtoms() == state.n_atoms
    assert sorted(a.GetAtomMapNum() for a in mol.GetAtoms()) == list(range(1, state.n_atoms + 1))
    assert all(a.GetTotalNumHs(includeNeighbors=False) == 0 for a in mol.GetAtoms())
    assert not any(a.GetIsAromatic() for a in mol.GetAtoms())  # Kekule


@pytest.mark.parametrize("name", sorted(ALL))
def test_initial_state_every_molecule(name):
    state = initial_state([SpeciesInput(ALL[name], 1)], LIMITS)
    _check_explicit_and_mapped(state)
    assert URL_SAFE.match(state.state_id) and URL_SAFE.match(state.pool_id)
    assert state.state_id == keys.state_id(state.mapped_smiles)
    assert state.pool_id == keys.pool_id(state.pool_key)
    assert state_from_mapped(state.mapped_smiles) == state  # canonical form is a fixed point
    mol = Chem.AddHs(Chem.MolFromSmiles(ALL[name]))
    assert state.n_atoms == mol.GetNumAtoms()


def test_initial_state_copies_and_species():
    state = initial_state([SpeciesInput("C[O-]", 2), SpeciesInput("CCl", 1)], LIMITS)
    _check_explicit_and_mapped(state)
    assert state.pool_key == "[CH3][Cl]|0.[CH3][O]|-1.[CH3][O]|-1"
    assert [(s.species_key, s.count, s.charge) for s in state.species] == [
        ("[CH3][Cl]|0", 1, 0),
        ("[CH3][O]|-1", 2, -1),
    ]
    assert sorted(state.fragment_token_keys) == ["CCl", "C[O-]", "C[O-]"]
    assert state.n_atoms == 15
    multi = initial_state([SpeciesInput("[Na+].[Cl-]", 1)], LIMITS)
    assert multi.pool_key == "[Cl]|-1.[Na]|1"


def test_initial_state_order_invariant_pool_key():
    a = initial_state([SpeciesInput("C[O-]", 1), SpeciesInput("CCl", 1)], LIMITS)
    b = initial_state([SpeciesInput("CCl", 1), SpeciesInput("C[O-]", 1)], LIMITS)
    assert a.pool_key == b.pool_key and a.pool_id == b.pool_id


def test_initial_state_drops_user_maps():
    a = initial_state([SpeciesInput("[CH3:7][O-:9]", 1)], LIMITS)
    b = initial_state([SpeciesInput("C[O-]", 1)], LIMITS)
    assert a == b


def _code(fn) -> Code:
    with pytest.raises(ServeError) as e:
        fn()
    return e.value.code


def test_initial_state_errors(monkeypatch):
    assert _code(lambda: initial_state([], LIMITS)) == Code.EMPTY_INPUT
    assert _code(lambda: initial_state([SpeciesInput("", 1)], LIMITS)) == Code.EMPTY_INPUT
    assert _code(lambda: initial_state([SpeciesInput("C1CC", 1)], LIMITS)) == Code.INVALID_SMILES
    assert (
        _code(lambda: initial_state([SpeciesInput("[CH3]", 1)], LIMITS))
        == Code.RADICAL_NOT_SUPPORTED
    )
    small = Limits(6, 4, 9, 10, 12, 200)
    pair = [SpeciesInput("C[O-]", 1), SpeciesInput("CCl", 1)]
    assert _code(lambda: initial_state(pair, small)) == Code.LIMIT_EXCEEDED
    assert _code(lambda: initial_state([SpeciesInput("C", 5)], LIMITS)) == Code.LIMIT_EXCEEDED
    many = [SpeciesInput("C", 1)] * 7
    assert _code(lambda: initial_state(many, LIMITS)) == Code.LIMIT_EXCEEDED
    e = pytest.raises(
        ServeError, initial_state, [SpeciesInput("CC", 1), SpeciesInput("C1", 1)], LIMITS
    )
    assert e.value.detail == {"field": 1} and e.value.status == 422


def test_initial_state_spin_parity(monkeypatch):
    """RDKit assigns radical electrons to every odd-electron SMILES, so the parity error is
    reached through resolve_charge_spin itself."""
    import ksmech.qm.xtb_calc as xtb_calc
    from ksmech.qm.records import QMError, ReasonCode

    def odd(mol):
        raise QMError(ReasonCode.SPIN_PARITY, "9 electrons cannot have 0 unpaired")

    monkeypatch.setattr(xtb_calc, "resolve_charge_spin", odd)
    assert _code(lambda: initial_state([SpeciesInput("C", 1)], LIMITS)) == Code.SPIN_PARITY


def test_state_from_mapped_requires_full_maps():
    assert _code(lambda: state_from_mapped("[CH3:1][Cl:2]")) == Code.IMPLICIT_H_IN_POOL
    assert _code(lambda: state_from_mapped("[C:1]([H])([H])([H])[Cl:2]")) == Code.INVALID_SMILES
    assert _code(lambda: state_from_mapped("C1CC")) == Code.INVALID_SMILES


def test_state_from_mol_matches_mapped():
    from ksmech.data import chem

    state = initial_state([SpeciesInput("CC=O", 1)], LIMITS)
    assert state_from_mol(chem.parse_mapped(state.mapped_smiles)) == state


# ---- persistence


def test_save_and_load_state(serve_settings):  # noqa: F811
    s = serve_settings()
    state = initial_state([SpeciesInput("C[O-]", 1), SpeciesInput("CCl", 1)], LIMITS)
    path = save_state(state, s)
    assert path == state_path(state.state_id, s) == s.pools / state.state_id / "state.json"
    assert json.loads(path.read_text())["state_id"] == state.state_id
    mtime = path.stat().st_mtime_ns
    assert save_state(state, s) == path and path.stat().st_mtime_ns == mtime  # idempotent
    assert load_state(state.state_id, s) == state
    assert not list(path.parent.glob(".*tmp"))


def test_load_state_not_found(serve_settings):  # noqa: F811
    s = serve_settings()
    for bad in ("s" + "0" * 20, "../etc", "p" + "0" * 20, "s" + "0" * 19 + "g"):
        with pytest.raises(ServeError) as e:
            load_state(bad, s)
        assert e.value.code == Code.POOL_NOT_FOUND and e.value.status == 404
    with pytest.raises(ServeError):
        state_path("../../x", s)


def test_load_state_corrupt(serve_settings):  # noqa: F811
    s = serve_settings()
    sid = "s" + "1" * 20
    path = s.pools / sid / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    with pytest.raises(ServeError) as e:
        load_state(sid, s)
    assert e.value.code == Code.POOL_NOT_FOUND


# ---- preflight


def _featurizer(make, chemistry="synthetic"):
    from ksmech.serve.featurize import make_featurizer

    return make_featurizer(make(chemistry=chemistry))


def test_preflight_ok(serve_settings):  # noqa: F811
    feat = _featurizer(serve_settings)
    pf = preflight_inputs([SpeciesInput("C[O-]", 1), SpeciesInput("CCl.[Na+]", 1)], feat, LIMITS)
    assert pf.error is None and pf.state is not None
    assert pf.state.pool_key == "[CH3][Cl]|0.[CH3][O]|-1.[Na]|1"
    rows = pf.species
    assert [r["field"] for r in rows] == [0, 1, 1]
    assert [r["token_key"] for r in rows] == ["C[O-]", "CCl", "[Na+]"]
    first = rows[0]
    assert first["charge"] == -1 and first["multiplicity"] == 1 and first["n_atoms"] == 5
    assert first["tokens_cached"] is False and first["qm_cached"] is False and first["errors"] == []


def test_preflight_codes(serve_settings, monkeypatch):  # noqa: F811
    feat = _featurizer(serve_settings)

    def code_of(inputs, limits=LIMITS):
        pf = preflight_inputs(inputs, feat, limits)
        assert pf.state is None
        return pf.error.code, (pf.error.detail or {}).get("field")

    assert code_of([]) == (Code.EMPTY_INPUT, None)
    assert code_of([SpeciesInput("CC", 1), SpeciesInput("  ", 1)]) == (Code.EMPTY_INPUT, 1)
    assert code_of([SpeciesInput("C1CC", 1)]) == (Code.INVALID_SMILES, 0)
    assert code_of([SpeciesInput("CC", 1), SpeciesInput("[CH3]", 1)]) == (
        Code.RADICAL_NOT_SUPPORTED,
        1,
    )
    assert code_of([SpeciesInput("[SeH2]", 1)]) == (Code.UNSUPPORTED_ELEMENT, 0)  # no EHT levels
    assert code_of([SpeciesInput("C", 9)]) == (Code.LIMIT_EXCEEDED, 0)
    assert code_of([SpeciesInput("C", 1)] * 7) == (Code.LIMIT_EXCEEDED, None)
    big = Limits(6, 4, 9, 10, 12, 200)
    assert code_of([SpeciesInput("C[O-]", 1), SpeciesInput("CCl", 1)], big) == (
        Code.LIMIT_EXCEEDED,
        None,
    )
    import ksmech.qm.xtb_calc as xtb_calc
    from ksmech.qm.records import QMError, ReasonCode

    def odd(mol):
        raise QMError(ReasonCode.SPIN_PARITY, "odd")

    monkeypatch.setattr(xtb_calc, "resolve_charge_spin", odd)
    assert code_of([SpeciesInput("C", 1)]) == (Code.SPIN_PARITY, 0)


def test_preflight_rows_carry_errors(serve_settings):  # noqa: F811
    feat = _featurizer(serve_settings)
    pf = preflight_inputs([SpeciesInput("CC", 1), SpeciesInput("C1CC", 2)], feat, LIMITS)
    assert pf.error.code == Code.INVALID_SMILES
    bad = [r for r in pf.species if r["field"] == 1]
    assert len(bad) == 1 and bad[0]["token_key"] is None and bad[0]["count"] == 2
    assert bad[0]["errors"][0]["code"] == "INVALID_SMILES"
    assert bad[0]["errors"][0]["detail"] == {"field": 1}


def test_preflight_notes_dropped_maps(serve_settings):  # noqa: F811
    feat = _featurizer(serve_settings)
    pf = preflight_inputs([SpeciesInput("[CH3:1][O-:2]", 1)], feat, LIMITS)
    assert pf.error is None and pf.notes and "atom maps" in pf.notes[0]


def test_preflight_real_mode_elements(serve_settings):  # noqa: F811
    """Real and cached modes accept every element with GFN2 parameters."""
    feat = _featurizer(serve_settings, "cached")
    pf = preflight_inputs([SpeciesInput("[SeH2]", 1)], feat, LIMITS)
    assert pf.error is None and pf.species[0]["errors"] == []
