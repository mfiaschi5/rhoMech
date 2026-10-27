"""serve/workbench/molecule.py: input formats, the cache key and stereo, one-fragment and query-atom
checks, preflight (radical, atom limit), the molecule id and registry, the sketch <-> record
mapping on permuted and symmetric drawings, and the status of a registered molecule."""

from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from ksmech.serve.core import schemas as S
from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.workbench import molecule
from ksmech.serve.workbench import service as workbench_service
from tests.test_serve_fixtures_api import NeverScheduler
from tests.test_serve_fixtures_wb import (  # noqa: F401 (fixtures)
    no_xtb,
    synthetic_root,
    wb_runtime,
    wb_settings,
)

pytestmark = pytest.mark.serve


def molblock(smiles: str, *, explicit_h: bool = False, order: list[int] | None = None) -> str:
    """A 2D molblock of `smiles` (wedges from the stereo), atoms renumbered by `order`."""
    mol = Chem.MolFromSmiles(smiles)
    if explicit_h:
        mol = Chem.AddHs(mol)
    if order is not None:
        mol = Chem.RenumberAtoms(mol, order)
    AllChem.Compute2DCoords(mol)
    return Chem.MolToMolBlock(mol)


def code_of(fn, *args, **kwargs) -> str:
    with pytest.raises(ServeError) as info:
        fn(*args, **kwargs)
    return str(info.value.code)


@pytest.fixture
def ctx(wb_settings):  # noqa: F811
    from ksmech.serve.runtime.context import build_context

    return build_context(wb_settings)


# ---- input


def test_smiles_and_molblock_give_one_key_and_id():
    a = molecule.parse_input(smiles="OCC")
    b = molecule.parse_input(molblock=molblock("CCO"))
    c = molecule.parse_input(molblock=molblock("CCO", explicit_h=True))
    assert a.key == b.key == c.key == "CCO"
    assert a.source == "smiles" and b.source == c.source == "molblock"
    assert c.mol.GetNumAtoms() == 9  # the drawn H atoms are kept
    mid = molecule.molecule_id(a.key)
    assert mid.startswith("m") and len(mid) == 21 and int(mid[1:], 16) >= 0
    assert mid == molecule.molecule_id(c.key) != molecule.molecule_id("CCC")


def test_stereo_comes_from_the_wedges():
    r = molecule.parse_input(molblock=molblock("C[C@H](O)CC"))
    s = molecule.parse_input(molblock=molblock("C[C@@H](O)CC"))
    assert "@" in r.key and "@" in s.key and r.key != s.key
    assert r.key == molecule.parse_input(smiles="C[C@H](O)CC").key
    assert molecule.molecule_id(r.key) != molecule.molecule_id(s.key)
    e = molecule.parse_input(molblock=molblock("C/C=C/C"))
    assert e.key == "C/C=C/C"


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"smiles": "CCO.O"}, Code.MULTIPLE_FRAGMENTS),
        ({"molblock": None, "smiles": None}, Code.REQUEST_INVALID),
        ({"molblock": "x", "smiles": "C"}, Code.REQUEST_INVALID),
        ({"smiles": "  "}, Code.EMPTY_INPUT),
        ({"smiles": "C1CC"}, Code.INVALID_SMILES),
        ({"molblock": "not a molfile"}, Code.INVALID_MOLFILE),
    ],
)
def test_input_errors(kwargs, code):
    assert code_of(molecule.parse_input, **kwargs) == code


def test_two_fragment_and_query_molblocks():
    assert code_of(molecule.parse_input, molblock=molblock("CCO.O")) == Code.MULTIPLE_FRAGMENTS
    query = molblock("CCO").replace(" O   ", " A   ")
    assert " A   " in query
    assert code_of(molecule.parse_input, molblock=query) == Code.INVALID_MOLFILE


def test_preflight_errors(ctx, wb_settings):  # noqa: F811
    from dataclasses import replace

    body = S.MoleculeIn(smiles="[CH3]")
    assert code_of(molecule.create_molecule, ctx, body) == Code.RADICAL_NOT_SUPPORTED
    ctx.settings = replace(wb_settings, max_atoms=5)
    with pytest.raises(ServeError) as info:
        molecule.create_molecule(ctx, S.MoleculeIn(smiles="CCO"))
    assert info.value.code == Code.LIMIT_EXCEEDED and info.value.status == 422
    assert info.value.detail == {"key": "CCO", "n_atoms": 9, "max_atoms": 5}


# ---- mapping


def _check_mapping(drawn: Chem.Mol, key: str, m: dict) -> None:
    from ksmech.qm.embed import prepare_mol

    record = prepare_mol(key)
    s2r, r2s, parent = m["sketch_to_record"], m["record_to_sketch"], m["record_parent_sketch"]
    assert m["n_sketch_atoms"] == drawn.GetNumAtoms() == len(s2r)
    assert m["n_record_atoms"] == record.GetNumAtoms() == len(r2s) == len(parent)
    assert sorted(s2r) == sorted(set(s2r))
    for i, r in enumerate(s2r):
        assert drawn.GetAtomWithIdx(i).GetAtomicNum() == record.GetAtomWithIdx(r).GetAtomicNum()
        assert r2s[r] == i and parent[r] == i
    for b in drawn.GetBonds():  # the mapping is an isomorphism
        assert record.GetBondBetweenAtoms(s2r[b.GetBeginAtomIdx()], s2r[b.GetEndAtomIdx()])
    for r, sk in enumerate(r2s):
        if sk is None:  # an implicit H of the drawing goes to its drawn heavy atom
            assert record.GetAtomWithIdx(r).GetAtomicNum() == 1
            assert record.GetBondBetweenAtoms(r, s2r[parent[r]]) is not None


@pytest.mark.parametrize(
    ("smiles", "order", "explicit_h"),
    [
        ("CCO", [2, 0, 1], False),  # permuted heavy atoms
        ("CC(=O)C", [1, 3, 0, 2], False),  # symmetric: two equivalent methyls
        ("CCO", None, True),  # every H drawn
        ("C[C@H](O)CC", [3, 1, 4, 0, 2], False),  # a stereocentre, permuted
    ],
)
def test_sketch_mapping(smiles, order, explicit_h):
    p = molecule.parse_input(molblock=molblock(smiles, explicit_h=explicit_h, order=order))
    m = molecule.sketch_mapping(p.mol, p.key)
    assert m is not None and m["chirality_matched"]
    _check_mapping(p.mol, p.key, m)
    S.SketchMappingOut(**m)


def test_mapping_failure_is_none():
    drawn = molecule.parse_input(smiles="CCO").mol
    assert molecule.sketch_mapping(drawn, "CCC") is None


# ---- registry and status


def test_create_molecule_ready_with_mapping(ctx):
    out = molecule.create_molecule(ctx, S.MoleculeIn(molblock=molblock("CCO", order=[2, 1, 0])))
    assert out.status == "ready" and out.key == "CCO" and out.tokens_ready and out.qm_ready
    assert out.mapping is not None and out.mapping.sketch_to_record == [2, 1, 0]
    assert out.mapping_error is None and out.n_atoms == 9 and out.charge == 0
    entry = molecule.load_entry(ctx.settings, out.molecule_id)
    assert entry["key"] == "CCO" and entry["failure"] is None
    again = molecule.molecule_out(ctx, out.molecule_id)
    assert again.status == "ready" and again.mapping is None and again.mapping_error is None


def test_unknown_molecule_is_404(ctx):
    with pytest.raises(ServeError) as info:
        molecule.molecule_out(ctx, "m" + "0" * 20)
    assert info.value.code == Code.MOLECULE_NOT_FOUND and info.value.status == 404


def test_status_missing_qm_only_and_running(ctx, wb_settings):  # noqa: F811
    from dataclasses import replace

    from ksmech.orbitals.build import tokens_path
    from ksmech.serve.runtime.context import build_context

    entry = molecule.register(ctx.settings, "CCCCO", ctx.featurizer.method)
    assert molecule.molecule_out(ctx, entry["molecule_id"]).status == "missing"
    path = tokens_path(wb_settings.serve_tokens, "COC", 0, ctx.featurizer.method)
    moved = path.with_suffix(".bak")
    path.rename(moved)
    try:
        e = molecule.register(ctx.settings, "COC", ctx.featurizer.method)
        out = molecule.molecule_out(ctx, e["molecule_id"])
        assert out.status == "qm_only" and out.qm_ready and not out.tokens_ready
    finally:
        moved.rename(path)
    quick = replace(wb_settings, interactive_wait_s=0.01)
    never = build_context(quick, scheduler=NeverScheduler())
    out = molecule.create_molecule(never, S.MoleculeIn(smiles="CCCCO"))
    assert out.status == "running"


def test_failed_featurize_is_recorded_and_retried_only_on_request(
    ctx,
    wb_runtime,  # noqa: F811
    monkeypatch,
):
    from ksmech.serve.chem.synthetic import SyntheticError

    def broken(self, key):
        raise SyntheticError("EMBED_FAILED", f"no conformer for {key}")

    monkeypatch.setattr(type(wb_runtime.featurizer), "_build", broken)
    out = molecule.create_molecule(ctx, S.MoleculeIn(smiles="CCCC"))
    assert out.status == "failed" and out.error is not None
    assert out.error.code == Code.SYNTHETIC_FAILED and out.error.upstream == "EMBED_FAILED"
    n = ctx.scheduler.n_done
    assert molecule.create_molecule(ctx, S.MoleculeIn(smiles="CCCC")).status == "failed"
    assert ctx.scheduler.n_done == n  # a recorded failure is not run again
    with pytest.raises(ServeError) as info:  # nor by a GET that needs the QM record
        workbench_service.geometry_out(ctx, out.molecule_id)
    assert info.value.code == Code.SYNTHETIC_FAILED and info.value.upstream == "EMBED_FAILED"
    assert ctx.scheduler.n_done == n
    molecule.create_molecule(ctx, S.MoleculeIn(smiles="CCCC", retry=True))
    assert ctx.scheduler.n_done == n + 1
    monkeypatch.undo()
    fixed = molecule.create_molecule(ctx, S.MoleculeIn(smiles="CCCC", retry=True))
    assert fixed.status == "ready" and fixed.error is None
    assert molecule.load_entry(ctx.settings, fixed.molecule_id)["failure"] is None
