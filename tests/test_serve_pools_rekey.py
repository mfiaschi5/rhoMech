"""`ksmech-serve rekey-pools` (open-shell plan P6 step 5): the stored pool states under the
identity key v2, and `PoolState.from_dict` refusing a key of another version."""

import json

import pytest
from rdkit import Chem

from ksmech.data import chem
from ksmech.data.resonance import sigma_key
from ksmech.serve.chem.pools import load_state, rekey_states, save_state, state_from_mapped
from ksmech.serve.core import keys
from ksmech.serve.core.contracts import KEY_VERSION, PoolState
from ksmech.serve.core.errors import ServeError
from tests.test_serve_fixtures import serve_settings  # noqa: F401 (fixture)

ATOMIC_O = "[O].CC"  # the O atom: N = 6, a Hund row of the species table
ETHANOL = "CCO"


def _mapped(smiles: str) -> str:
    """Every atom an explicit, mapped atom (a stored state's form)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    for a in mol.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)
    return Chem.MolToSmiles(mol)


def _v1(state: PoolState) -> dict:
    """The dict serve wrote before P6: v1 species keys, no key version."""
    mol = chem.parse_mapped(state.mapped_smiles)
    species = sorted(sigma_key(f) for f in chem.fragments(mol))
    pool_key = keys.pool_key_of(species)
    d = state.to_dict()
    d.pop("key_version")
    d["pool_key"], d["pool_id"] = pool_key, keys.pool_id(pool_key)
    for row, key in zip(d["species"], sorted(set(species)), strict=True):
        row["species_key"] = key
    return d


def _store(settings, smiles: str) -> tuple[PoolState, dict]:
    state = state_from_mapped(_mapped(smiles))
    path = save_state(state, settings)
    old = _v1(state)
    path.write_text(json.dumps(old, sort_keys=True, indent=1) + "\n")
    return state, old


def test_an_open_shell_state_is_rekeyed_and_a_closed_shell_keeps_its_pool_id(serve_settings):  # noqa: F811
    s = serve_settings()
    o_state, o_old = _store(s, ATOMIC_O)
    e_state, e_old = _store(s, ETHANOL)
    assert o_old["pool_key"] != o_state.pool_key and e_old["pool_key"] == e_state.pool_key
    out = rekey_states(s, dry_run=False)
    assert {k: out[k] for k in ("states", "unchanged", "rekeyed", "purged", "unreadable")} == {
        "states": 2, "unchanged": 1, "rekeyed": 1, "purged": 0, "unreadable": 0}  # fmt: skip
    assert out["changed"] == [(o_state.state_id, o_old["pool_id"], o_state.pool_id)]
    assert load_state(o_state.state_id, s) == o_state
    e_new = load_state(e_state.state_id, s)
    assert e_new.pool_id == e_old["pool_id"] and e_new.key_version == KEY_VERSION


def test_a_dry_run_writes_nothing_and_counts(serve_settings):  # noqa: F811
    s = serve_settings()
    _store(s, ATOMIC_O)
    _store(s, ETHANOL)
    before = {p: p.read_bytes() for p in s.pools.glob("*/state.json")}
    out = rekey_states(s, dry_run=True)
    assert out["dry_run"] and out["rekeyed"] == 1 and out["unchanged"] == 1
    assert {p: p.read_bytes() for p in s.pools.glob("*/state.json")} == before


def test_purge_removes_a_rekeyed_state(serve_settings):  # noqa: F811
    s = serve_settings()
    o_state, _ = _store(s, ATOMIC_O)
    out = rekey_states(s, dry_run=False, purge=True)
    assert out["purged"] == 1 and not (s.pools / o_state.state_id).exists()


def test_a_state_without_the_key_version_is_refused(serve_settings):  # noqa: F811
    s = serve_settings()
    e_state, _ = _store(s, ETHANOL)
    with pytest.raises(ValueError, match="rekey-pools"):
        PoolState.from_dict(json.loads((s.pools / e_state.state_id / "state.json").read_text()))
    with pytest.raises(ServeError):
        load_state(e_state.state_id, s)


def test_the_rekey_command_reports_a_dry_run(serve_settings, capsys, monkeypatch):  # noqa: F811
    from ksmech.serve.app import cli

    s = serve_settings()
    _store(s, ATOMIC_O)
    monkeypatch.setattr(cli, "_settings", lambda **kw: s)
    assert cli.main(["rekey-pools", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert '"rekeyed": 1' in out and '"dry_run": true' in out


def test_a_closed_shell_species_key_is_unchanged():
    for smiles in ("CCO", "CC(C)=O", "C[O-]", "[Br+]", "[OH+]"):
        assert keys.species_key_of(Chem.CanonSmiles(smiles))[0] == sigma_key(
            Chem.MolFromSmiles(smiles)
        )
