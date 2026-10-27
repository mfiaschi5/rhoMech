"""Server-side 2D depiction of pool states and dataset steps (design 11.4 step 0 and tier 3).

API side, RDKit only, for every saved state with or without tokens. The molblock carries 2D
coordinates (`rdDepictor.SetPreferCoordGen(True)`, then `Compute2DCoords`), no atom maps, and
no H atom except those whose map number is in `keep_maps` (a removed H raises its parent's
explicit H count, so labels still read CH3). The atom table maps each `depiction_idx` (the
molblock's atom order) to its pool atom (`pool_atom`: the atom index of
`parse_mapped(mapped_smiles)`, the index space of every reaction matrix and arrow reading),
the pool atoms of the H atoms hidden on it (`hidden_h_atoms`), the map number, element,
fragment and, with tokens, the global site (the token index space).

With `server_svg`, the same molecule is drawn by `rdMolDraw2D.MolDraw2DSVG` and
`GetDrawCoords(i)` gives the exact drawing position of every atom (`server_coords`, indexed by
`depiction_idx`, in the SVG's viewBox units): the client's last tier, for depictions whose atom
positions cannot be fitted from the SVG (CCl, [OH-] + CCl, ...).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from ksmech.serve.core.errors import Code, ServeError

if TYPE_CHECKING:
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.serve.core.contracts import PoolState

SVG_WIDTH = 320
SVG_HEIGHT = 220


def _parse(smiles: str) -> Chem.Mol:
    from ksmech.data import chem

    try:
        return chem.parse_mapped(smiles)
    except (chem.SmilesParseError, chem.SanitizeError) as exc:
        raise ServeError(Code.INVALID_SMILES, f"cannot depict {smiles!r}: {exc}") from exc


def _removable_h(atom: Chem.Atom, keep_maps: set[int]) -> int | None:
    """The parent atom index of an H that the depiction hides, else None. Kept: H atoms whose
    map is in keep_maps, isotopes, and H without exactly one non-H neighbour (H-, H+, H2)."""
    if atom.GetAtomicNum() != 1 or atom.GetIsotope():
        return None
    if atom.GetAtomMapNum() and atom.GetAtomMapNum() in keep_maps:
        return None
    nbrs = atom.GetNeighbors()
    if len(nbrs) != 1 or nbrs[0].GetAtomicNum() == 1:
        return None
    return nbrs[0].GetIdx()


def _hide_hydrogens(
    mol: Chem.Mol, keep_maps: set[int]
) -> tuple[Chem.Mol, list[int], dict[int, list[int]]]:
    """(the depiction mol, original atom index of each depiction atom, original indices of
    the H atoms hidden on each original atom). Maps are cleared."""
    hidden: Counter[int] = Counter()
    hidden_on: dict[int, list[int]] = {}
    drop = []
    for atom in mol.GetAtoms():
        parent = _removable_h(atom, keep_maps)
        if parent is not None:
            drop.append(atom.GetIdx())
            hidden[parent] += 1
            hidden_on.setdefault(parent, []).append(atom.GetIdx())
    rw = Chem.RWMol(mol)
    for parent, n in hidden.items():
        p = rw.GetAtomWithIdx(parent)
        p.SetNumExplicitHs(p.GetNumExplicitHs() + n)
        p.SetNoImplicit(True)
    for idx in sorted(drop, reverse=True):
        rw.RemoveAtom(idx)
    dep = rw.GetMol()
    for a in dep.GetAtoms():
        a.SetAtomMapNum(0)
    with rdBase.BlockLogs():
        dep.UpdatePropertyCache(strict=False)
    dropped = set(drop)
    return dep, [i for i in range(mol.GetNumAtoms()) if i not in dropped], hidden_on


def _fragment_of(mol: Chem.Mol) -> dict[int, int]:
    return {i: f for f, atoms in enumerate(Chem.GetMolFrags(mol)) for i in atoms}


def _server_drawing(dep: Chem.Mol) -> tuple[str, list[tuple[float, float]]]:
    """The SVG of `dep` and `GetDrawCoords(i)` of every atom (viewBox units)."""
    drawer = rdMolDraw2D.MolDraw2DSVG(SVG_WIDTH, SVG_HEIGHT)
    drawer.drawOptions().prepareMolsBeforeDrawing = False
    with rdBase.BlockLogs():
        prepared = rdMolDraw2D.PrepareMolForDrawing(dep, kekulize=False, addChiralHs=False)
    drawer.DrawMolecule(prepared)
    drawer.FinishDrawing()
    coords = [drawer.GetDrawCoords(i) for i in range(dep.GetNumAtoms())]
    return drawer.GetDrawingText(), [(float(p.x), float(p.y)) for p in coords]


def depiction(
    mol: Chem.Mol,
    ref: str,
    keep_maps: Iterable[int],
    server_svg: bool,
    global_of: Mapping[int, int] | None = None,
) -> dict:
    """`DepictionOut` fields of a mapped, explicit-H mol (`chem.parse_mapped`). `global_of`
    maps the mol's atom indices to global sites (`ReactantTokens.global_atom`)."""
    dep, origin, hidden_on = _hide_hydrogens(mol, {int(m) for m in keep_maps})
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(dep)
    frag = _fragment_of(mol)
    atoms = []
    for k, i in enumerate(origin):
        a = mol.GetAtomWithIdx(i)
        atoms.append(
            {
                "depiction_idx": k,
                "pool_atom": int(i),
                "hidden_h_atoms": sorted(hidden_on.get(i, [])),
                "global_site": None if global_of is None else global_of.get(i),
                "map": int(a.GetAtomMapNum()),
                "element": a.GetSymbol(),
                "fragment": int(frag[i]),
                "hidden_h": int(dep.GetAtomWithIdx(k).GetTotalNumHs()),
            }
        )
    svg, coords = _server_drawing(dep) if server_svg else (None, None)
    return {
        "ref": ref,
        "molblock": Chem.MolToMolBlock(dep),
        "atoms": atoms,
        "server_svg": svg,
        "server_coords": coords,
    }


def pool_depiction(
    state: PoolState, rt: ReactantTokens | None, keep_maps: set[int], server_svg: bool
) -> dict:
    """`DepictionOut` fields of a saved state (ref = state_id). `rt` (optional) gives each atom
    its global site; `rt.mol` is the same `parse_mapped(state.mapped_smiles)`."""
    mol = _parse(state.mapped_smiles)
    if rt is not None and rt.mol.GetNumAtoms() != mol.GetNumAtoms():
        raise ServeError(Code.CONSISTENCY_FAILED, f"tokens of {state.state_id} disagree with it")
    global_of = None if rt is None else rt.global_atom
    return depiction(mol, state.state_id, keep_maps, server_svg, global_of)


def step_depiction(
    side_smiles: str, keep_maps: set[int], server_svg: bool, *, ref: str | None = None
) -> dict:
    """`DepictionOut` fields of one side of a dataset step (mapped or not; unmapped atoms have
    map 0, global sites are None). `ref` defaults to "smiles:" + sha256(side)[:12]."""
    mol = _parse(side_smiles)
    ref = ref or "smiles:" + hashlib.sha256(side_smiles.encode()).hexdigest()[:12]
    return depiction(mol, ref, keep_maps, server_svg, None)
