"""Chemistry sanity checks on the QM layer, run on the fixed fixture molecules.

Shared by `scripts/validate_qm.py` and the slow tests. Every check raises AssertionError with
the numbers that decided it, and returns a short description of what it measured.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

from ksmech.data import chem
from ksmech.data.provenance import StaleArtifact
from ksmech.qm.ao import per_atom_sum
from ksmech.qm.embed import embed, prepare_mol
from ksmech.qm.records import (
    QMError,
    QMRecord,
    cache_path,
    is_valid_cache_file,
    load_record,
    save_record,
)
from ksmech.qm.xtb_calc import GFN2Calculator, QMConfig

log = logging.getLogger(__name__)

Records = Mapping[str, QMRecord]
ORTHO_TOL = 1e-6
UNIT_DIAG_TOL = 1e-8  # tblite's S diagonal is off by up to ~2e-10 for Cl (normalization noise)


def compute_fixture_records(
    molecules: Mapping[str, str], config: QMConfig | None = None
) -> dict[str, QMRecord]:
    """Featurize every fixture molecule (1 conformer), reading and filling the on-disk cache."""
    config = config or QMConfig()
    calc = GFN2Calculator(config)
    out = {}
    for name, smi in molecules.items():
        canonical = chem.unmapped_canonical(Chem.MolFromSmiles(smi))
        path = cache_path(config.cache_dir, canonical, 0, config.method)
        if is_valid_cache_file(path):
            try:
                out[name] = load_record(path, (canonical, 0, config.method))
                continue
            except StaleArtifact as exc:
                log.info("recomputing %s: %s", name, exc)
        mol = prepare_mol(canonical)
        embed(mol, n_conformers=1, seed=0)
        res = calc.try_run(mol, 0)
        if res.record is None:
            raise QMError(res.reason, f"{name} ({smi}): {res.detail}")
        save_record(res.record, Path(path))
        out[name] = res.record
    return out


def gap_ev(rec: QMRecord) -> float:
    return rec.lumo_ev - rec.homo_ev


def _symbols(rec: QMRecord) -> list[str]:
    pt = Chem.GetPeriodicTable()
    return [pt.GetElementSymbol(int(z)) for z in rec.atomic_numbers]


def check_gap_benzene_vs_acrolein(r: Records) -> str:
    gb, ga = gap_ev(r["benzene"]), gap_ev(r["acrolein"])
    assert gb > ga, f"HOMO-LUMO gap benzene {gb:.3f} eV <= acrolein {ga:.3f} eV"
    return f"gap benzene {gb:.3f} eV > acrolein {ga:.3f} eV"


def check_acetone_charge_on_oxygen(r: Records) -> str:
    rec = r["acetone"]
    i = int(np.argmin(rec.charges))
    assert _symbols(rec)[i] == "O", f"most negative charge on {_symbols(rec)[i]}{i}"
    return f"most negative Mulliken charge O ({rec.charges[i]:+.3f})"


def check_methylamine_fukui_minus_on_n(r: Records) -> str:
    rec = r["methylamine"]
    i = int(np.argmax(rec.fukui_fmo_minus))
    assert _symbols(rec)[i] == "N", f"largest f- on {_symbols(rec)[i]}{i}"
    return f"frozen-orbital f- largest on N ({rec.fukui_fmo_minus[i]:.3f})"


def check_acetaldehyde_fukui_plus_on_carbonyl_c(r: Records) -> str:
    rec = r["acetaldehyde"]
    mol = prepare_mol(rec.atom_order_smiles)
    carbonyl = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and any(n.GetAtomicNum() == 8 for n in a.GetNeighbors())
    ]
    assert len(carbonyl) == 1, f"expected one carbonyl C, found {carbonyl}"
    i = int(np.argmax(rec.fukui_fmo_plus))
    assert i == carbonyl[0], f"largest f+ on {_symbols(rec)[i]}{i}, carbonyl C is {carbonyl[0]}"
    return f"frozen-orbital f+ largest on carbonyl C ({rec.fukui_fmo_plus[i]:.3f})"


def check_overlap_symmetric_unit_diagonal(r: Records) -> str:
    worst = 0.0
    for name, rec in r.items():
        assert np.allclose(rec.S, rec.S.T, atol=1e-12), f"{name}: S is not symmetric"
        err = float(np.abs(np.diag(rec.S) - 1.0).max())
        assert err < UNIT_DIAG_TOL, f"{name}: S diagonal differs from 1 by {err:.2e}"
        worst = max(worst, err)
    return f"S symmetric, max |S_ii - 1| = {worst:.1e} over {len(r)} molecules"


def check_mo_orthonormality(r: Records) -> str:
    worst = 0.0
    for name, rec in r.items():
        err = float(np.abs(rec.C.T @ rec.S @ rec.C - np.eye(rec.C.shape[1])).max())
        assert err < ORTHO_TOL, f"{name}: max |C^T S C - 1| = {err:.2e}"
        worst = max(worst, err)
    return f"max |C^T S C - 1| = {worst:.1e} (limit {ORTHO_TOL:.0e})"


def check_ao_map_valence_electrons(r: Records) -> str:
    """Gross AO population per atom plus the Mulliken charge must be a whole number."""
    worst = 0.0
    for name, rec in r.items():
        density = (rec.C * rec.occupations) @ rec.C.T
        pop = per_atom_sum(np.diag(density @ rec.S), rec.ao_atom, rec.n_atoms)
        z_val = pop + rec.charges
        err = float(np.abs(z_val - np.round(z_val)).max())
        assert err < 1e-6, f"{name}: AO map gives non-integer valence electrons ({err:.2e})"
        worst = max(worst, err)
    return f"populations + charges are whole numbers (max dev {worst:.1e})"


CHECKS: list[Callable[[Records], str]] = [
    check_gap_benzene_vs_acrolein,
    check_acetone_charge_on_oxygen,
    check_methylamine_fukui_minus_on_n,
    check_acetaldehyde_fukui_plus_on_carbonyl_c,
    check_overlap_symmetric_unit_diagonal,
    check_mo_orthonormality,
    check_ao_map_valence_electrons,
]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def run_checks(records: Records) -> list[CheckResult]:
    results = []
    for check in CHECKS:
        try:
            results.append(CheckResult(check.__name__, True, check(records)))
        except AssertionError as exc:
            results.append(CheckResult(check.__name__, False, str(exc)))
    return results
