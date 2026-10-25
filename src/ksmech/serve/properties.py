"""Molecular properties of the molecule workbench (worker task `wb_properties`), display only.

Everything comes from the QM record (and the model token file's localization statistics when
there is one):

- **Global**: energy, HOMO, LUMO, gap, mu, eta, omega (the record's Koopmans-style
  descriptors), charge, multiplicity, AO / MO counts, N_valence = sum of the occupations,
  `fermi_smeared` (`localize.is_fermi_smeared`), the record's warnings.
- **Atoms**: upstream `orbitals.tokens.atom_tokens` on the record (exactly the token fields:
  Mulliken charge, finite-difference and frozen-orbital Fukui, dual, %Vbur, the frontier
  weights, chirality, formal charge), the Wiberg valence, and the diagnostics hybridization,
  aromatic flag and functional-group label (`model_input=False`, shown greyed).
- **Bonds**: Wiberg indices `qm.bond_index.wiberg_matrix(S, P, ao_atom, n)` with
  P = C diag(n) C^T, and lengths, for every Lewis bond of the record plus every non-bonded pair
  with W >= `WIBERG_NONBONDED_MIN`.

The JSON is cached under `data/serve/workbench/properties/<sha>.json`, keyed by the key, the
QM method and content sha, the model token file's digest, `WORKBENCH_VERSION` and the oriented
fingerprint. NaN is written as null.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ksmech.serve.contracts import Runtime
from ksmech.serve.errors import jsonable
from ksmech.serve.molecule import WORKBENCH_VERSION, cache_sha

if TYPE_CHECKING:
    from ksmech.orbitals.tokens import MoleculeTokens
    from ksmech.qm.records import QMRecord
    from ksmech.serve.settings import ServeSettings

WIBERG_NONBONDED_MIN = 0.15


def properties_path(
    s: ServeSettings, key: str, method: str, qm_digest: str, token_digest: str | None
) -> Path:
    from ksmech.serve.versions import token_fingerprint

    parts = ["wb_properties", key, method, qm_digest, token_digest, WORKBENCH_VERSION]
    return s.workbench_properties / f"{cache_sha([*parts, token_fingerprint()])}.json"


def _finite(x: float) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def global_props(rec: QMRecord, t: MoleculeTokens | None) -> dict:
    from ksmech.orbitals.localize import is_fermi_smeared
    from ksmech.qm.fukui import HARTREE_TO_EV, OCC_THRESHOLD

    occ = np.asarray(rec.occupations, dtype=float)
    homo, lumo = _finite(rec.homo_ev), _finite(rec.lumo_ev)
    return {
        "energy_ha": float(rec.energy_ha),
        "energy_ev": float(rec.energy_ha) * HARTREE_TO_EV,
        "homo_ev": homo,
        "lumo_ev": lumo,
        "gap_ev": None if homo is None or lumo is None else lumo - homo,
        "mu_ev": _finite(rec.chemical_potential_ev),
        "eta_ev": _finite(rec.hardness_ev),
        "omega_ev": _finite(rec.electrophilicity_ev),
        "charge": int(rec.charge),
        "multiplicity": int(rec.unpaired) + 1,
        "n_atoms": int(rec.n_atoms),
        "n_ao": int(rec.S.shape[0]),
        "n_mo": int(rec.C.shape[1]),
        "n_occupied": int(np.sum(occ >= OCC_THRESHOLD)),
        "n_valence_electrons": float(occ.sum()),
        "fermi_smeared": bool(is_fermi_smeared(rec)),
        "pm_converged": None if t is None else bool(t.pm_converged),
        "pm_sweeps_occ": None if t is None else int(t.pm_sweeps_occ),
        "pm_sweeps_virt": None if t is None else int(t.pm_sweeps_virt),
        "warnings": [w for w in str(rec.warnings or "").split(",") if w],
    }


def wiberg(rec: QMRecord) -> np.ndarray:
    """(n_atoms, n_atoms) Wiberg indices of the record's total density."""
    from ksmech.qm.bond_index import wiberg_matrix

    P = (rec.C * np.asarray(rec.occupations, dtype=float)) @ rec.C.T
    return wiberg_matrix(rec.S, P, np.asarray(rec.ao_atom), rec.n_atoms)


def _atom_row(a, W: np.ndarray) -> dict:
    from ksmech.serve.token_rows import fg_label

    i = int(a.atom_index)
    return {
        "index": i,
        "Z": int(a.atomic_number),
        "formal_charge": int(a.formal_charge),
        "mulliken_charge": float(a.mulliken_charge),
        "f_plus_fd": a.f_plus,
        "f_minus_fd": a.f_minus,
        "dual_fd": a.dual,
        "f_plus_fmo": a.f_plus_fmo,
        "f_minus_fmo": a.f_minus_fmo,
        "fmo_weights": [float(x) for x in np.asarray(a.fmo_weights).ravel()],
        "vbur": a.vbur,
        "chirality": float(a.chirality),
        "wiberg_valence": float(W[i].sum()),
        "diagnostic": {
            "diagnostic": True,
            "hybridization": str(a.hybridization),
            "aromatic": bool(a.aromatic),
            "fg_label": fg_label(int(a.fg_id)),
        },
    }


def atom_props(rec: QMRecord, mol, W: np.ndarray) -> list[dict]:
    """`AtomPropsOut` fields of every record atom (upstream `atom_tokens`)."""
    from rdkit import Chem

    from ksmech.orbitals.fgroups import default_vocab
    from ksmech.orbitals.tokens import atom_tokens

    fg_ids = default_vocab().atom_ids(Chem.MolFromSmiles(rec.atom_order_smiles), rec.n_atoms)
    pt = Chem.GetPeriodicTable()
    rows = []
    for a in atom_tokens(rec, mol, fg_ids):
        row = _atom_row(a, W)
        row["element"] = pt.GetElementSymbol(row["Z"])
        row["f_zero_fd"] = float(rec.fukui_zero[row["index"]]) if len(rec.fukui_zero) else None
        rows.append(row)
    return rows


def bond_props(rec: QMRecord, mol, W: np.ndarray) -> list[dict]:
    """Every Lewis bond (by record index), then every other pair with W >= the threshold."""
    from ksmech.serve.geometry import bond_rows

    xyz = np.asarray(rec.coords_ang, dtype=float)

    def row(i: int, j: int, order: int | None) -> dict:
        return {
            "i": i,
            "j": j,
            "bonded": order is not None,
            "order": order,
            "wiberg": float(W[i, j]),
            "length_ang": float(np.linalg.norm(xyz[i] - xyz[j])),
        }

    out, bonded = [], set()
    for i, j, order in bond_rows(mol):
        i, j = min(i, j), max(i, j)
        bonded.add((i, j))
        out.append(row(i, j, order))
    ii, jj = np.nonzero(np.triu(W >= WIBERG_NONBONDED_MIN, k=1))
    out += [row(i, j, None) for i, j in zip(ii.tolist(), jj.tolist(), strict=True)]
    return [r for r in out if r["bonded"] or (r["i"], r["j"]) not in bonded]


def properties_body(rt: Runtime, key: str) -> dict:
    """`MoleculePropertiesOut` fields except `molecule_id` and `geometry`."""
    from ksmech.orbitals.frontier import LEVELS
    from ksmech.qm.embed import prepare_mol
    from ksmech.serve.orbital_sets import load_qm, provenance_of

    rec, _, qm_digest = load_qm(rt, key)
    t = rt.loader(key)
    mol = prepare_mol(rec.atom_order_smiles)
    W = wiberg(rec)
    glob = global_props(rec, t)
    if rec.atom_order_smiles != key:
        note = f"ATOM_ORDER: the record uses the atom order of {rec.atom_order_smiles}"
        glob["warnings"].append(note)
    token_digest = rt.loader.digest(key) if t is not None else None
    return {
        "key": key,
        "global_props": glob,
        "atoms": atom_props(rec, mol, W),
        "bonds": bond_props(rec, mol, W),
        "fmo_levels": list(LEVELS),
        "wiberg_threshold": WIBERG_NONBONDED_MIN,
        "provenance": provenance_of(rt, key, qm_digest, token_digest),
    }


def t_wb_properties(rt: Runtime, payload: dict) -> dict:
    """{"key"} -> {"path"}: the properties JSON of one molecule (cached)."""
    from ksmech.serve.orbital_sets import load_qm
    from ksmech.serve.pools import atomic_write_text

    key = str(payload["key"])
    _, _, qm_digest = load_qm(rt, key)
    path = properties_path(rt.settings, key, rt.featurizer.method, qm_digest, rt.loader.digest(key))
    if not path.is_file():
        body = properties_body(rt, key)
        atomic_write_text(path, json.dumps(jsonable(body), sort_keys=True) + "\n", rt.settings)
    return {"path": str(path)}
