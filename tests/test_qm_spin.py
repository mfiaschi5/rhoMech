"""qm.spin: the tagged method of a species data.spin reassigns (open-shell plan P4 steps 2-4)."""

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from ksmech.data import chem
from ksmech.qm import spin as qm_spin
from ksmech.qm.embed import embed, prepare_mol
from ksmech.qm.geometry import BOHR_ANG, formed_bonds, kick
from ksmech.qm.records import QMError, ReasonCode, cache_key
from ksmech.qm.xtb_calc import GFN2Calculator, QMConfig

BASE = QMConfig().method
SRC = Path(__file__).resolve().parents[1] / "src" / "ksmech"


@pytest.mark.parametrize(
    ("smiles", "tag"),
    [
        ("Cl[C]Cl", "spin-v1-u0"),  # the species table pairs it
        ("[C]=O", "spin-v1-u0"),
        ("[Cl][Cu][Cl]", "spin-v1-u1"),  # RDKit counts 0 on an odd Cu(II) fragment
        ("[Br][Fe-]([Br])([Br])[Br]", "spin-v1-u5"),
        ("[O][O]", None),  # the table's 2 is RDKit's 2: the key is unchanged
        ("CC(C)=O", None),
        ("[CH3]", None),
        ("[I][Cu][I]", None),  # ambiguous: no count, so RDKit's and SPIN_PARITY
    ],
)
def test_a_species_is_tagged_only_where_data_spin_reassigns_it(smiles, tag):
    assert qm_spin.spin_tag(smiles) == tag
    want = BASE if tag is None else f"{BASE}+{tag}"
    assert qm_spin.method_for(smiles, BASE) == want == qm_spin.config_for(smiles, QMConfig()).method


def test_method_for_is_idempotent_and_keeps_every_untagged_key():
    tagged = qm_spin.method_for("Cl[C]Cl", BASE)
    assert qm_spin.method_for("Cl[C]Cl", tagged) == tagged
    for name in ("CC", "CCO", "c1ccccc1", "CC#N"):
        assert cache_key(name, 0, qm_spin.method_for(name, BASE)) == cache_key(name, 0, BASE)


def test_the_tag_sits_before_serve_s_metal_suffix():
    from ksmech.serve.chem.metal_qm import MetalQMConfig

    metal = MetalQMConfig()
    composed = qm_spin.config_for("[Cl][Cu][Cl]", metal)
    assert isinstance(composed, MetalQMConfig)
    assert composed.method == f"{BASE}+spin-v1-u1+metal-geom-v1"
    assert qm_spin.method_for("[Cl][Cu][Cl]", metal.method) == composed.method


def test_a_closed_shell_metal_complex_keeps_serve_s_method_and_key():
    from ksmech.serve.chem.metal_qm import MetalQMConfig

    metal, key = MetalQMConfig(), "Cl[Pd]Cl"  # Pd(II) d8, even, RDKit 0
    assert qm_spin.config_for(key, metal) is metal
    tagged = qm_spin.method_for(key, metal.method)
    assert cache_key(key, 0, tagged) == cache_key(key, 0, metal.method)


def _mol(smiles: str) -> Chem.Mol:
    mol = prepare_mol(chem.unmapped_canonical(Chem.MolFromSmiles(smiles)))
    embed(mol, 1)
    return mol


def test_spin_for_qm_takes_the_assigned_count_with_its_tag():
    assert qm_spin.spin_for_qm(_mol("Cl[C]Cl"), "Cl[C]Cl") == (0, 0, "spin-v1-u0")
    assert qm_spin.spin_for_qm(_mol("[Cl][Cu][Cl]"), "[Cl][Cu][Cl]") == (0, 1, "spin-v1-u1")
    assert qm_spin.spin_for_qm(_mol("CC"), "CC") == (0, 0, None)
    with pytest.raises(QMError) as exc:
        qm_spin.spin_for_qm(_mol("[I][Cu][I]"), "[I][Cu][I]")
    assert exc.value.reason is ReasonCode.SPIN_PARITY


def test_the_kick_is_fixed():
    a, b = kick((3, 3)), kick((3, 3))
    assert np.array_equal(a, b) and 0 < np.abs(a).max() * BOHR_ANG < 0.1


def test_formed_bonds_finds_a_new_contact_and_skips_drawn_bonds():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCC"))
    xyz = np.array([[0.0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]] + [[0, 0, 9.0 + i] for i in range(8)])
    assert formed_bonds(mol, xyz) == []
    xyz[2] = [0.0, 1.6, 0.0]  # C3 next to C1: a cyclopropane-like contact the drawing lacks
    assert [(i, j) for i, j, _ in formed_bonds(mol, xyz)] == [(0, 2)]


def test_a_tagged_singlet_carbene_is_bent_and_stamped_with_its_tag(qm_config):
    """Cl[C]Cl: ETKDG draws it linear, a saddle of the singlet; the kicked start finds the bent
    minimum, and the record carries the assigned count and the tagged method."""
    rec = GFN2Calculator(qm_config).run(_mol("Cl[C]Cl"), 0)
    assert rec.unpaired == 0 and rec.method == f"{qm_config.method}+spin-v1-u0"
    c, cl1, cl2 = (rec.coords_ang[i] for i in (1, 0, 2))
    u, v = cl1 - c, cl2 - c
    angle = np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
    assert angle < 150.0


# the functions that turn a configured method into a cache path: every call in the engine must
# pass a method that went through `method_for`, or one a stored record or token file carries
PATH_BUILDERS = re.compile(
    r"\b(cache_path|tokens_path|bond_index_path|tombstone_path|thermo_path)\("
)
RAW_METHOD = re.compile(r"(config|qm|self|cfg)\.method\b")


def test_every_engine_cache_path_goes_through_method_for():
    offenders = []
    for path in SRC.rglob("*.py"):
        if "serve" in path.parts or path.name in ("records.py", "spin.py"):
            continue  # serve keeps RDKit spin until S1; records.py defines the builders
        text = path.read_text()
        calls = [ln for ln in text.splitlines() if PATH_BUILDERS.search(ln) and "def " not in ln]
        offenders += [f"{path.name}: {ln.strip()}" for ln in calls if RAW_METHOD.search(ln)]
        if (
            calls
            and "method_for(" not in text
            and not all(re.search(r"(record|rec|tokens|hit)\.method", ln) for ln in calls)
        ):
            offenders.append(f"{path.name}: builds cache paths without method_for")
    assert not offenders, offenders


def test_the_index_reads_the_spin_tag():
    from ksmech.qm.index import spin_columns

    assert spin_columns("CC", BASE, 0) == dict(
        unpaired=0, spin_source="rdkit", spin_rule_version=None
    )
    tagged = qm_spin.method_for("[Cl][Cu][Cl]", BASE)
    assert spin_columns("[Cl][Cu][Cl]", tagged) == dict(
        unpaired=1, spin_source="metal", spin_rule_version="spin-v1"
    )


def test_a_tagged_config_differs_only_in_its_tag():
    tagged = qm_spin.config_for("Cl[C]Cl", QMConfig())
    assert replace(tagged, spin_tag=None) == QMConfig()
