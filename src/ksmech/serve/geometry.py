"""Per-species 3D geometry of a pool state (adapter A16), for the 3D panel and the contact view.

Every fragment keeps its own QM record frame (Angstrom): the atoms are `AtomToken.position` in
record order, the bonds and charges come from `prepare_mol(token_key)` (the record's atom order
and connectivity). Fragments come from separate QM runs, so their frames are unrelated;
`display_offsets` lays them out along +x with a 3 Angstrom gap, for display only.

Only numpy, RDKit and other serve modules are imported at the top; upstream helpers are imported
inside the functions, so the API process can use this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem, rdBase

from ksmech.serve.errors import Code, ServeError
from ksmech.serve.keys import species_key_of

if TYPE_CHECKING:
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.orbitals.tokens import MoleculeTokens

FRAGMENT_GAP_ANG = 3.0
AROMATIC_BOND = 4  # MDL bond code, used only when a record cannot be kekulized
FRAME_NOTE = (
    "the record frame of this species' own QM run (Angstrom); frames of separate runs are "
    "unrelated, so display_offset only lays the fragments out side by side"
)


def positions(t: MoleculeTokens) -> np.ndarray:
    """(n_atoms, 3) record positions (Angstrom)."""
    return np.array([a.position for a in t.atoms], dtype=float).reshape(-1, 3)


def atomic_numbers(t: MoleculeTokens) -> np.ndarray:
    return np.array([a.atomic_number for a in t.atoms], dtype=int)


def orbital_position(rt: ReactantTokens, token_id: int) -> tuple[int, int]:
    """(fragment f, position k in `rt.fragments[f].tokens.orbitals`) of a global orbital token
    id; ServeError(NOT_AN_ORBITAL_TOKEN) for an atom token or an id out of range."""
    j = int(token_id) - rt.n_atoms
    if not 0 <= j < rt.n_orbitals:
        span = f"{rt.n_atoms}..{rt.n_tokens - 1}"
        msg = f"token {token_id} is not an orbital token (orbital ids are {span})"
        raise ServeError(Code.NOT_AN_ORBITAL_TOKEN, msg)
    f = int(rt.orbital_fragment[j])
    return f, j - int(rt.orbital_offsets[f])


def map_of(rt: ReactantTokens, global_site: int) -> int | None:
    """The atom-map number of a global atom, or None (an implicit H, or an unmapped atom)."""
    m = rt.mol_atom.get(int(global_site))
    if m is None:
        return None
    return int(rt.mol.GetAtomWithIdx(m).GetAtomMapNum()) or None


def record_mol(token_key: str) -> Chem.Mol:
    """`prepare_mol(token_key)`: the record's atom order, bonds and charges, H explicit."""
    from ksmech.qm.embed import prepare_mol

    return prepare_mol(token_key)


def kekule_copy(mol: Chem.Mol) -> Chem.Mol:
    """A kekulized copy (aromatic flags cleared); the aromatic form when that fails."""
    m = Chem.Mol(mol)
    try:
        with rdBase.BlockLogs():
            Chem.Kekulize(m, clearAromaticFlags=True)
    except Chem.rdchem.KekulizeException:
        return Chem.Mol(mol)
    return m


def bond_rows(mol: Chem.Mol, atoms: Iterable[int] | None = None) -> list[tuple[int, int, int]]:
    """(record i, record j, order) of every bond of the kekulized record (only bonds between
    `atoms` when given); order 4 marks an aromatic bond that could not be kekulized."""
    keep = None if atoms is None else {int(a) for a in atoms}
    out = []
    for bond in kekule_copy(mol).GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if keep is not None and not (i in keep and j in keep):
            continue
        order = AROMATIC_BOND if bond.GetIsAromatic() else int(round(bond.GetBondTypeAsDouble()))
        out.append((int(i), int(j), max(order, 1)))
    return out


def _with_coords(mol: Chem.Mol, xyz_ang: np.ndarray) -> Chem.Mol:
    xyz = np.asarray(xyz_ang, dtype=float).reshape(-1, 3)
    if len(xyz) != mol.GetNumAtoms():
        raise ValueError(f"{len(xyz)} positions for {mol.GetNumAtoms()} atoms")
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, p.tolist())
    conf.Set3D(True)
    m.AddConformer(conf, assignId=True)
    return m


def molblock_with_coords(token_key: str, xyz_ang: np.ndarray) -> str:
    """`prepare_mol(token_key)` with one conformer at the given positions, as a molblock (bonds
    and charges kept)."""
    return Chem.MolToMolBlock(_with_coords(record_mol(token_key), xyz_ang))


def molblock_subset(record: Chem.Mol, atoms: Sequence[int], xyz_ang: np.ndarray) -> str:
    """A molblock of the record atoms `atoms` (ascending) at the positions `xyz_ang` (one row
    per atom, same order), with the bonds among them; no H is added to cut valences."""
    keep = sorted(int(a) for a in atoms)
    if keep != list(atoms):
        raise ValueError("molblock_subset needs ascending atom indices")
    m = Chem.RWMol(kekule_copy(record))
    for idx in sorted(set(range(m.GetNumAtoms())) - set(keep), reverse=True):
        m.RemoveAtom(idx)
    for a in m.GetAtoms():
        a.SetNoImplicit(True)
        a.SetAtomMapNum(0)
    m.UpdatePropertyCache(strict=False)
    return Chem.MolToMolBlock(_with_coords(m.GetMol(), xyz_ang), kekulize=False)


def xyz_text(z: Sequence[int], xyz_ang: np.ndarray, comment: str) -> str:
    """An XYZ file: the atom count, a one-line comment, then `Sym x y z` in %.6f."""
    z = np.asarray(z, dtype=int).ravel()
    xyz = np.asarray(xyz_ang, dtype=float).reshape(-1, 3)
    if len(z) != len(xyz):
        raise ValueError(f"{len(z)} elements for {len(xyz)} positions")
    pt = Chem.GetPeriodicTable()
    lines = [str(len(z)), " ".join(str(comment).split())]
    lines += [
        f"{pt.GetElementSymbol(int(a))} {x:.6f} {y:.6f} {w:.6f}"
        for a, (x, y, w) in zip(z, xyz, strict=True)
    ]
    return "\n".join(lines) + "\n"


def display_offsets(rt: ReactantTokens) -> list[tuple[float, float, float]]:
    """Per fragment, a translation that lays the fragments out along +x by bounding box with a
    3 Angstrom gap (the first fragment stays in place). Display only."""
    out: list[tuple[float, float, float]] = []
    x_next = 0.0
    for f, frag in enumerate(rt.fragments):
        x = positions(frag.tokens)[:, 0]
        dx = 0.0 if f == 0 else x_next - float(x.min())
        out.append((dx, 0.0, 0.0))
        x_next = float(x.max()) + dx + FRAGMENT_GAP_ANG
    return out


def atom_rows(
    rt: ReactantTokens, f: int, indices: Sequence[int], xyz_ang: np.ndarray
) -> list[dict]:
    """`AtomXYZOut` fields of record atoms `indices` of fragment f at the positions `xyz_ang`."""
    t = rt.fragments[f].tokens
    off = int(rt.atom_offsets[f])
    pt = Chem.GetPeriodicTable()
    xyz = np.asarray(xyz_ang, dtype=float).reshape(-1, 3)
    rows = []
    for i, p in zip(indices, xyz, strict=True):
        z = int(t.atoms[int(i)].atomic_number)
        rows.append(
            {
                "global_site": off + int(i),
                "record_index": int(i),
                "map": map_of(rt, off + int(i)),
                "Z": z,
                "symbol": pt.GetElementSymbol(z),
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
            }
        )
    return rows


def _xyz_comment(key: str, record: Chem.Mol, t: MoleculeTokens) -> str:
    charge = Chem.GetFormalCharge(record)
    mult = 1 + sum(a.GetNumRadicalElectrons() for a in record.GetAtoms())
    return f"{key} charge={charge} mult={mult} method={t.method} warnings={t.qm_warnings or 'none'}"


def fragment_geometry(
    rt: ReactantTokens, f: int, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> dict:
    """`SpeciesGeometryOut` fields of fragment f (record frame; `offset` is display only)."""
    frag = rt.fragments[f]
    t, record = frag.tokens, frag.record
    xyz = positions(t)
    if record.GetNumAtoms() != len(xyz):
        msg = f"record of {frag.key} has {record.GetNumAtoms()} atoms, tokens {len(xyz)}"
        raise ServeError(Code.CONSISTENCY_FAILED, msg)
    return {
        "fragment": int(f),
        "token_key": frag.key,
        "species_key": species_key_of(frag.key)[0],
        "atoms": atom_rows(rt, f, range(len(xyz)), xyz),
        "bonds": bond_rows(record),
        "xyz": xyz_text(atomic_numbers(t), xyz, _xyz_comment(frag.key, record, t)),
        "molblock": Chem.MolToMolBlock(_with_coords(record, xyz)),
        "display_offset": tuple(float(v) for v in offset),
        "frame_note": FRAME_NOTE,
    }


def species_geometry(rt: ReactantTokens) -> list[dict]:
    """`SpeciesGeometryOut` fields of every fragment, in `ReactantTokens` fragment order."""
    offsets = display_offsets(rt)
    return [fragment_geometry(rt, f, offsets[f]) for f in range(len(rt.fragments))]
