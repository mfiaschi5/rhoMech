"""The tokens of a step's reactant side: every fragment, one global index space.

A token file belongs to a map-free molecule (`chem.unmapped_canonical` of the fragment) whose
atoms follow `prepare_mol` order. `record_correspondence` maps each atom of the drawn,
atom-mapped fragment onto that record order:

- heavy atoms (and H atoms RDKit keeps, such as `[H+]`) by the canonical SMILES output order:
  the fragment with maps removed and aromaticity perceived writes exactly the cache key, and
  record atom i is the i-th atom written. When the strings differ (tokens of another
  resonance form of the same skeleton) the skeleton substructure match that agrees best on
  charges and H counts is used instead;
- explicit H atoms (mapped `[H:20]`) onto the record hydrogens of the same heavy atom.

Global indices: atom tokens first, fragment by fragment (so an atom's token id equals its
global atom index), then orbital tokens fragment by fragment in each file's own order
(occupied, then virtual).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

from ksmech.data import chem
from ksmech.data import provenance as prov
from ksmech.data.provenance import StaleArtifact
from ksmech.orbitals.build import DEFAULT_TOKENS_DIR, tokens_path
from ksmech.orbitals.tokens import (
    CandidateConfig,
    MoleculeTokens,
    OrbitalKind,
    OrbitalToken,
    load_tokens,
)
from ksmech.qm.embed import prepare_mol
from ksmech.qm.xtb_calc import QMConfig

_IDX = "_ksmech_frag_idx"
_MAX_MATCHES = 256

TokenLoader = Callable[[str], MoleculeTokens | None]


class CorrespondenceError(ValueError):
    """The drawn fragment cannot be matched onto its token record."""


class MissingTokens(LookupError):
    """At least one reactant fragment has no token file (yet)."""

    def __init__(self, keys: list[str]):
        super().__init__(f"no tokens for {keys}")
        self.keys = keys


def _heavy_by_canonical_order(frag: Chem.Mol, key: str) -> dict[int, int] | None:
    m = Chem.RWMol(frag)
    for a in m.GetAtoms():
        a.SetIntProp(_IDX, a.GetIdx())
        a.SetAtomMapNum(0)
    try:
        with rdBase.BlockLogs():
            m = Chem.RemoveHs(m, sanitize=False)
            Chem.SanitizeMol(m)
    except Chem.rdchem.MolSanitizeException:
        return None
    if Chem.MolToSmiles(m) != key:
        return None
    order = m.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]
    return {m.GetAtomWithIdx(int(j)).GetIntProp(_IDX): i for i, j in enumerate(order)}


def _skeleton(mol: Chem.Mol) -> Chem.Mol:
    """Heavy-atom graph, every bond single, charges and aromatic flags cleared; each atom keeps
    its original index, charge and H count as properties."""
    m = Chem.RWMol(mol)
    for a in m.GetAtoms():
        a.SetIntProp(_IDX, a.GetIdx())
        a.SetIntProp("_q", a.GetFormalCharge())
        a.SetIntProp("_h", a.GetTotalNumHs(includeNeighbors=True))
    with rdBase.BlockLogs():
        m = Chem.RWMol(Chem.RemoveHs(m, sanitize=False))
    for b in m.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in m.GetAtoms():
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
    m.UpdatePropertyCache(strict=False)
    return m.GetMol()


def _heavy_by_skeleton(frag: Chem.Mol, record: Chem.Mol) -> dict[int, int]:
    q, t = _skeleton(frag), _skeleton(record)
    if q.GetNumAtoms() != t.GetNumAtoms():
        raise CorrespondenceError("heavy-atom counts differ")
    matches = t.GetSubstructMatches(q, uniquify=False, maxMatches=_MAX_MATCHES)
    if not matches:
        raise CorrespondenceError("skeletons do not match")

    def score(match) -> int:
        pairs = [(q.GetAtomWithIdx(i), t.GetAtomWithIdx(j)) for i, j in enumerate(match)]
        return sum(
            (a.GetIntProp("_q") == b.GetIntProp("_q")) + (a.GetIntProp("_h") == b.GetIntProp("_h"))
            for a, b in pairs
        )

    best = max(matches, key=score)
    return {
        q.GetAtomWithIdx(i).GetIntProp(_IDX): t.GetAtomWithIdx(j).GetIntProp(_IDX)
        for i, j in enumerate(best)
    }


def _hydrogens(frag: Chem.Mol, record: Chem.Mol, heavy: dict[int, int]) -> dict[int, int]:
    """Explicit H atoms of `frag` onto unused record H atoms of the same heavy atom."""
    used = set(heavy.values())
    out = {}
    for a in frag.GetAtoms():
        if a.GetIdx() in heavy:
            continue
        if a.GetAtomicNum() != 1 or a.GetDegree() != 1:
            raise CorrespondenceError(f"atom {a.GetIdx()} left unmatched")
        parent = record.GetAtomWithIdx(heavy[a.GetNeighbors()[0].GetIdx()])
        free = [
            n.GetIdx()
            for n in parent.GetNeighbors()
            if n.GetAtomicNum() == 1 and n.GetIdx() not in used
        ]
        if not free:
            raise CorrespondenceError(f"no record H left for atom {a.GetIdx()}")
        out[a.GetIdx()] = min(free)
        used.add(min(free))
    return out


def record_correspondence(frag: Chem.Mol, record: Chem.Mol, key: str) -> dict[int, int]:
    """Fragment atom index -> record atom index for every atom of the drawn fragment."""
    heavy = _heavy_by_canonical_order(frag, key)
    if heavy is None or any(
        frag.GetAtomWithIdx(i).GetAtomicNum() != record.GetAtomWithIdx(j).GetAtomicNum()
        for i, j in heavy.items()
    ):
        heavy = _heavy_by_skeleton(frag, record)
    return heavy | _hydrogens(frag, record, heavy)


@dataclass
class FragmentTokens:
    key: str  # map-free canonical SMILES: the token-cache key
    tokens: MoleculeTokens
    record: Chem.Mol  # prepare_mol(key): the record's atom order and bonds
    mol_atoms: tuple[int, ...]  # reactant-mol atom indices of this fragment
    record_of: dict[int, int]  # reactant-mol atom index -> record atom index


class TokenCache:
    """Token files of one QM method and conformer, loaded on demand. The `maxsize` most
    recently used molecules stay in memory (a file holds S and F, up to ~1 MB each).

    A file whose provenance header no longer matches (schema, config or its QM record) is
    treated as absent and its reason kept in `stale` (STALE_ARTIFACT), unless `allow_stale`
    (diagnostics only). Every token file served is reported to `provenance.note_input`."""

    def __init__(
        self,
        tokens_dir: Path = DEFAULT_TOKENS_DIR,
        method: str | None = None,
        conformer_id: int = 0,
        maxsize: int = 1024,
        allow_stale: bool = False,
    ):
        self.tokens_dir = Path(tokens_dir)
        self.method = method or QMConfig().method
        self.conformer_id = conformer_id
        self.maxsize = maxsize
        self.allow_stale = allow_stale
        self.stale: dict[str, str] = {}
        self._memo: OrderedDict[str, MoleculeTokens | None] = OrderedDict()

    def _load(self, key: str) -> MoleculeTokens | None:
        path = tokens_path(self.tokens_dir, key, self.conformer_id, self.method)
        if not path.exists():
            return None
        try:
            return load_tokens(path, verify=not self.allow_stale)
        except StaleArtifact as exc:
            self.stale[key] = str(exc)
            return None

    def __call__(self, key: str) -> MoleculeTokens | None:
        if key in self._memo:
            self._memo.move_to_end(key)
            value = self._memo[key]
        else:
            value = self._load(key)
            self._memo[key] = value
            if len(self._memo) > self.maxsize:
                self._memo.popitem(last=False)
        if value is not None and value.source_path:
            prov.note_input(prov.file_input_name(value.source_path), value.digest)
        return value


def _fragment_keys(mol: Chem.Mol) -> list[tuple[tuple[int, ...], Chem.Mol, str | None]]:
    out = []
    mapping: list = []
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping)
    for atoms, frag in zip(mapping, frags, strict=True):
        try:
            key = chem.unmapped_canonical(frag)
        except (chem.SanitizeError, ValueError):
            key = None
        out.append((tuple(atoms), frag, key))
    return out


def load_fragments(mol: Chem.Mol, loader: TokenLoader) -> list[FragmentTokens]:
    """Tokens and atom correspondence of every fragment. Raises MissingTokens."""
    parts = _fragment_keys(mol)
    found = {key: loader(key) for _, _, key in parts if key is not None}
    missing = sorted(
        {key or "<uncanonicalizable>" for _, _, key in parts if key is None or found[key] is None}
    )
    if missing:
        raise MissingTokens(missing)
    out = []
    for atoms, frag, key in parts:
        record = prepare_mol(key)
        local = record_correspondence(frag, record, key)
        out.append(
            FragmentTokens(key, found[key], record, atoms, {atoms[i]: j for i, j in local.items()})
        )
    return out


@dataclass
class ReactantTokens:
    """Every fragment of one reactant side in one global token index space."""

    mol: Chem.Mol  # the side as drawn (atom-mapped, Kekulé)
    fragments: list[FragmentTokens]
    candidate: CandidateConfig = field(default_factory=CandidateConfig)

    @classmethod
    def from_smiles(
        cls, smiles: str, loader: TokenLoader, candidate: CandidateConfig | None = None
    ) -> ReactantTokens:
        mol = chem.parse_mapped(smiles)
        return cls(mol, load_fragments(mol, loader), candidate or CandidateConfig())

    @cached_property
    def atom_offsets(self) -> list[int]:
        return list(np.cumsum([0] + [len(f.tokens.atoms) for f in self.fragments]))

    @cached_property
    def orbital_offsets(self) -> list[int]:
        return list(np.cumsum([0] + [len(f.tokens.orbitals) for f in self.fragments]))

    @property
    def n_atoms(self) -> int:
        return int(self.atom_offsets[-1])

    @property
    def n_orbitals(self) -> int:
        return int(self.orbital_offsets[-1])

    @property
    def n_tokens(self) -> int:
        return self.n_atoms + self.n_orbitals

    @cached_property
    def global_atom(self) -> dict[int, int]:
        """Reactant-mol atom index -> global atom index."""
        return {
            m: int(self.atom_offsets[f]) + r
            for f, frag in enumerate(self.fragments)
            for m, r in frag.record_of.items()
        }

    @cached_property
    def mol_atom(self) -> dict[int, int]:
        """Global atom index -> reactant-mol atom index (implicit H atoms have none)."""
        return {g: m for m, g in self.global_atom.items()}

    def site_of_map(self, map_number: int) -> int:
        return self.global_atom[chem.map_to_index(self.mol)[map_number]]

    @cached_property
    def atom_fragment(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.fragments)), np.diff(self.atom_offsets))

    @cached_property
    def orbitals(self) -> list[OrbitalToken]:
        return [o for f in self.fragments for o in f.tokens.orbitals]

    @cached_property
    def orbital_fragment(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.fragments)), np.diff(self.orbital_offsets))

    def orbital(self, token_id: int) -> OrbitalToken:
        return self.orbitals[token_id - self.n_atoms]

    def centres(self, token_id: int) -> dict[int, float]:
        """Global atom index -> population for every centre (weight >= tau_atom) of an
        orbital token."""
        j = token_id - self.n_atoms
        off = int(self.atom_offsets[self.orbital_fragment[j]])
        d = self.orbitals[j].descriptors
        return {
            off + int(a): float(w)
            for a, w in zip(d.all_center_atoms, d.all_center_weights, strict=True)
        }

    @cached_property
    def tokens_on_atom(self) -> dict[int, list[int]]:
        """Global atom index -> orbital token ids with that atom among their centres."""
        out: dict[int, list[int]] = {}
        for j in range(self.n_orbitals):
            for a in self.centres(self.n_atoms + j):
                out.setdefault(a, []).append(self.n_atoms + j)
        return out

    @cached_property
    def occupied(self) -> np.ndarray:
        """(n_orbitals,) True for occupied-block tokens."""
        return np.array([o.kind == OrbitalKind.OCCUPIED for o in self.orbitals], dtype=bool)

    @cached_property
    def d_type(self) -> np.ndarray:
        """(n_orbitals,) d polarization functions (main centre mostly d, no valence d shell)."""
        return np.array(
            [f.tokens.is_d_type(o) for f in self.fragments for o in f.tokens.orbitals], dtype=bool
        )

    @cached_property
    def donor_ok(self) -> np.ndarray:
        """(n_orbitals,) the donor candidate mask: doubly occupied, not fractional."""
        c, occ = self.candidate, self._occupations
        frac = np.array([c.is_fractional(x) for x in occ], dtype=bool)
        return self.occupied & (occ > c.full) & ~frac

    @cached_property
    def acceptor_ok(self) -> np.ndarray:
        """(n_orbitals,) the acceptor candidate mask before the donor-partner exclusion:
        empty, not fractional, not d-type unless `allow_d_acceptors`."""
        c, occ = self.candidate, self._occupations
        frac = np.array([c.is_fractional(x) for x in occ], dtype=bool)
        ok = ~self.occupied & (occ < c.empty) & ~frac
        return ok if c.allow_d_acceptors else ok & ~self.d_type

    @cached_property
    def _occupations(self) -> np.ndarray:
        return np.array([o.descriptors.occupation for o in self.orbitals], dtype=float)

    @cached_property
    def atom_positions(self) -> np.ndarray:
        """(n_atoms, 3) Angstrom, record geometries (bookkeeping; never embedded)."""
        rows = [a.position for f in self.fragments for a in f.tokens.atoms]
        return np.array(rows, dtype=float).reshape(-1, 3)

    @cached_property
    def centroids(self) -> np.ndarray:
        return np.array([o.centroid for o in self.orbitals], dtype=float).reshape(-1, 3)

    def record_bonded(self, a: int, b: int) -> bool:
        """Global atoms a and b are bonded in their (shared) record."""
        fa, fb = int(self.atom_fragment[a]), int(self.atom_fragment[b])
        if fa != fb:
            return False
        off = int(self.atom_offsets[fa])
        rec = self.fragments[fa].record
        return rec.GetBondBetweenAtoms(a - off, b - off) is not None

    def drawn_order(self, a: int, b: int) -> int:
        """Bond order a-b in the drawn reactants; a record H with no drawn atom is an implicit
        H of its neighbour (order 1)."""
        ma, mb = self.mol_atom.get(a), self.mol_atom.get(b)
        if ma is not None and mb is not None:
            bond = self.mol.GetBondBetweenAtoms(ma, mb)
            return 0 if bond is None else int(bond.GetBondTypeAsDouble())
        return int(self.record_bonded(a, b))
