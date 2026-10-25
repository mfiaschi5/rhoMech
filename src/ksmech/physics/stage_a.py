"""The pair features: geometry-free, computed from the tokens alone (spec, pair feature vector;
docs/decisions.md 2026-10-22). The module keeps its stage-A name from when a contact stage B
existed (now on branch `archive/stage-b`).

For an ordered (occupied r, virtual s) pair of the reactant set:
- `g`, `log_abs_g`: the spectral gap factor G(r, s) and log|G| (always defined);
- `inter`: 1 when r and s sit on different fragments;
- `bonded`, `bond_order`, `graph_distance`: between the two tokens' centre sets, on the
  record's Lewis structure (`prepare_mol`): bonded when some centre of r is bonded to some
  centre of s, the largest such bond order, and the shortest graph distance between the two
  sets (0 when they share an atom), capped at 6; across fragments 0, 0 and 6;
- `wbi`: the largest Wiberg bond index (`qm.bond_index`, from the record's density) over the
  Lewis bonds joining the two centre sets; 0 when they are not bonded;
- `centroid_distance`, `cross_coupling`: the distance between the two centroids and |l_r^T B
  l_s| over the AO blocks between their centres, in the reactant pose (within a fragment);
- `align_cos`: |cos| of the angle between the p vectors of r and s on their main centres (the
  lobe axes; undefined when either has no p character there);
- `dihedral_cos`: when the two orbitals sit on adjacent bonds (a centre of r bonded to a centre
  of s, each with another centre), the cosine of the site-partner-partner-site dihedral (-1 is
  antiperiplanar); undefined otherwise.
No contact geometry and no Klopman-Salem energy: the model has neither.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from ksmech.orbitals.descriptors import DescriptorConfig
from ksmech.orbitals.localize import P_AXIS_ORDER
from ksmech.orbitals.tokens import MoleculeTokens, OrbitalToken
from ksmech.physics.ks_terms import cross_center_coupling_matrix, gap_kernel_matrix, wh_matrix
from ksmech.physics.pairs import LOG_FLOOR, PairBlock
from ksmech.physics.params import REDUCED, KSParams
from ksmech.qm.embed import prepare_mol

STAGE_A_FEATURES = (
    "g",
    "log_abs_g",
    "inter",
    "bonded",
    "bond_order",
    "graph_distance",
    "wbi",
    "centroid_distance",
    "cross_coupling",
    "align_cos",
    "dihedral_cos",
)
A = {name: k for k, name in enumerate(STAGE_A_FEATURES)}
GRAPH_CAP = 6


def _graph(t: MoleculeTokens) -> tuple[np.ndarray, np.ndarray]:
    """(graph distances, bond orders) of the record's Lewis structure, atoms in token order."""
    mol = prepare_mol(t.smiles)
    if mol.GetNumAtoms() != len(t.atoms):
        raise ValueError(f"{t.smiles}: Lewis structure has {mol.GetNumAtoms()} atoms")
    Chem.Kekulize(mol, clearAromaticFlags=False)
    dist = Chem.GetDistanceMatrix(mol)
    order = np.zeros((mol.GetNumAtoms(),) * 2)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        order[i, j] = order[j, i] = b.GetBondTypeAsDouble()
    return np.minimum(dist, GRAPH_CAP), order


def _wbi_between(t: MoleculeTokens, cr: list[int], cs: list[int], order: np.ndarray) -> float:
    if t.bond_index is None:
        raise ValueError(f"{t.smiles}: tokens without bond indices (`ksmech-qm bond-index`)")
    pairs = [(min(i, j), max(i, j)) for i in cr for j in cs if i != j and order[i, j] > 0]
    return max((t.bond_index.get(p, 0.0) for p in pairs), default=0.0)


def _centres(o: OrbitalToken) -> list[int]:
    return [int(a) for a in o.descriptors.all_center_atoms if a >= 0]


def _p_vector(o: OrbitalToken, t: MoleculeTokens, atom: int) -> np.ndarray | None:
    aos = np.flatnonzero((t.ao_atom == atom) & (np.asarray(t.ao_l) == 1))
    if len(aos) != 3:
        return None
    v = np.zeros(3)
    v[list(P_AXIS_ORDER)] = o.coeff_vector[aos]
    n = np.linalg.norm(v)
    return v / n if n >= DescriptorConfig().min_p_norm else None


def _dihedral_cos(t: MoleculeTokens, cr: list[int], cs: list[int], order: np.ndarray):
    """cos of the dihedral o_r - c_r - c_s - o_s for a bond c_r-c_s between the two centre
    sets, with o_r / o_s the leading other centre of each token; None when not adjacent."""
    pos = np.array([a.position for a in t.atoms], dtype=float)
    for c_r in cr:
        for c_s in cs:
            if c_r == c_s or order[c_r, c_s] == 0:
                continue
            o_r = next((x for x in cr if x != c_r and x != c_s), None)
            o_s = next((x for x in cs if x != c_s and x != c_r), None)
            if o_r is None or o_s is None:
                continue
            b1, b2, b3 = pos[c_r] - pos[o_r], pos[c_s] - pos[c_r], pos[o_s] - pos[c_s]
            n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
            norm = np.linalg.norm(n1) * np.linalg.norm(n2)
            return float(n1 @ n2 / norm) if norm > 1e-9 else None
    return None


def _intra(t: MoleculeTokens, occ, virt, z: np.ndarray, d: np.ndarray, params) -> None:
    """Fill the within-fragment entries of one (occ x virt) block in place."""
    dist, order = _graph(t)
    cen_o = np.array([o.descriptors.centroid for o in occ]).reshape(-1, 3)
    cen_v = np.array([o.descriptors.centroid for o in virt]).reshape(-1, 3)
    z[..., A["centroid_distance"]] = np.linalg.norm(cen_o[:, None] - cen_v[None], axis=-1)
    B = wh_matrix(t, params)
    z[..., A["cross_coupling"]] = np.abs(cross_center_coupling_matrix(occ, virt, t.ao_atom, B))
    d[..., [A["centroid_distance"], A["cross_coupling"]]] = True
    for i, r in enumerate(occ):
        cr, pr = _centres(r), _p_vector(r, t, int(r.descriptors.center_atoms[0]))
        for j, s in enumerate(virt):
            cs = _centres(s)
            sub = dist[np.ix_(cr, cs)]
            bonds = order[np.ix_(cr, cs)]
            z[i, j, A["graph_distance"]] = sub.min()
            z[i, j, A["bonded"]] = float((bonds > 0).any())
            z[i, j, A["bond_order"]] = bonds.max()
            z[i, j, A["wbi"]] = _wbi_between(t, cr, cs, order)
            ps = _p_vector(s, t, int(s.descriptors.center_atoms[0]))
            if pr is not None and ps is not None:
                z[i, j, A["align_cos"]], d[i, j, A["align_cos"]] = abs(pr @ ps), True
            cos = _dihedral_cos(t, cr, cs, order)
            if cos is not None:
                z[i, j, A["dihedral_cos"]], d[i, j, A["dihedral_cos"]] = cos, True
    d[..., [A["graph_distance"], A["bonded"], A["bond_order"], A["wbi"]]] = True


def stage_a_block(fragments: list[MoleculeTokens], params: KSParams = REDUCED) -> PairBlock:
    """The geometry-free features of every (occupied, virtual) pair of the reactant set, in the
    sample's orbital order, at `params` (samples store them at `REDUCED`; `to_params` scales
    the cross-centre coupling). `sites` stays -1 (schema 15 only)."""
    n_occ = [len(t.occupied()) for t in fragments]
    n_virt = [len(t.virtual()) for t in fragments]
    shape, n_feat = (sum(n_occ), sum(n_virt)), len(STAGE_A_FEATURES)
    block = PairBlock(
        np.zeros((*shape, n_feat)),
        np.zeros((*shape, n_feat), dtype=bool),
        np.zeros(shape, dtype=np.int8),
        np.full((*shape, 2), -1, dtype=np.int64),
    )
    r0, c0 = np.cumsum([0, *n_occ]), np.cumsum([0, *n_virt])
    for f, t_f in enumerate(fragments):
        for g, t_g in enumerate(fragments):
            if not n_occ[f] or not n_virt[g]:
                continue
            cell = np.s_[r0[f] : r0[f + 1], c0[g] : c0[g + 1]]
            occ, virt = t_f.occupied(), t_g.virtual()
            G = gap_kernel_matrix(occ, virt)
            z, d = block.z[cell], block.defined[cell]
            z[..., A["g"]], z[..., A["log_abs_g"]] = G, np.log(np.maximum(np.abs(G), LOG_FLOOR))
            z[..., A["inter"]] = float(f != g)
            d[..., [A["g"], A["log_abs_g"], A["inter"]]] = True
            block.gap_sign[cell] = np.sign(G).astype(np.int8)
            if f == g:
                _intra(t_f, occ, virt, z, d, params)
            else:
                z[..., A["graph_distance"]] = GRAPH_CAP
                d[..., [A["graph_distance"], A["bonded"], A["bond_order"], A["wbi"]]] = True
    bad = ~np.isfinite(block.z)
    block.n_non_finite = int(bad.sum())
    block.defined[bad] = False
    block.z[~block.defined] = 0.0
    return block
