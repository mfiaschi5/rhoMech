"""Atom and orbital tokens of one molecule, built from a `QMRecord`.

An orbital token is a localized orbital described by continuous physical descriptors
(`OrbitalDescriptors`: occupation, centres and their weights and s/p/d composition, the
geometry of the two main centres, spatial extent, spectral decomposition over the canonical
MOs). Those descriptors are the model input: `MoleculeTokens.orbital_features()` (continuous),
`orbital_spectra()` (the spectral set) and `orbital_centres()` (the token-to-atom weights through
which an orbital inherits its atoms' embeddings). The classifier's `type` (sigma, pi, lone
pair, ...) and its parents are kept on the token for tests, alignment diagnostics and the
interface; they never enter the model input and never decide which arrows are legal
(`CandidateConfig` masks on descriptors).

An atom token's model input is `MoleculeTokens.atom_categorical()` (atomic number, formal
charge: embedded) and `atom_features()` (charge, frozen-orbital Fukui, dual descriptor,
frontier weights, buried volume, chirality). Hybridization, aromaticity and the
functional-group id are stored, tagged `model_input=False`, and never enter either matrix.

Atoms and orbitals share one token-id sequence: atoms first (`token_id == atom_index`), then
occupied orbitals by ascending diagnostic energy, then virtual orbitals likewise. Orbital
parents are record atom indices (the order of `prepare_mol(atom_order_smiles)`); atom-map
numbers live on `AtomToken.map_number` and are 0 until the arrow-alignment phase assigns them.

Units: energies in eV, positions and centroids in Angstrom, `F` (the converged Fock matrix) in
Hartree. `C_loc` holds the S-orthonormal AO coefficients of every orbital token as columns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path

import numpy as np
from rdkit import Chem
from rich.table import Table

from ksmech.data import provenance as prov
from ksmech.orbitals.chirality import chirality_signs
from ksmech.orbitals.classify import Classification, ClassifyConfig, OrbitalType, classify_block
from ksmech.orbitals.descriptors import (
    MODEL_INPUT,
    DescriptorConfig,
    OrbitalDescriptors,
    build_descriptors,
    feature_names,
)
from ksmech.orbitals.fgroups import FGVocab, default_vocab
from ksmech.orbitals.frontier import LEVELS, frontier_weights, frozen_orbital_fukui_blocks
from ksmech.orbitals.localize import (
    LocalizedOrbitals,
    PMConfig,
    is_fermi_smeared,
    localize_record,
)
from ksmech.qm.embed import prepare_mol
from ksmech.qm.records import QMRecord

SCHEMA_VERSION = 8


class OrbitalKind(StrEnum):
    OCCUPIED = "occupied"
    VIRTUAL = "virtual"


@dataclass
class OrbitalToken:
    token_id: int
    kind: OrbitalKind
    descriptors: OrbitalDescriptors  # the model input (`continuous_vector`, `spectral_set`)
    coeff_vector: np.ndarray  # (nao,) AO coefficients, for the KS coupling term
    # diagnostics from the classifier: never model input, never a mask; tests and tables
    type: OrbitalType = field(metadata={MODEL_INPUT: False})
    parent_atoms: tuple[int, ...] = ()  # record atom indices, by descending population
    parent_weights: tuple[float, ...] = ()
    s_share: float = float("nan")
    p_share: float = float("nan")
    d_share: float = float("nan")
    overlap_population: float = float("nan")
    metal_bound: bool = False
    metal_atom: int = -1

    @property
    def energy_ev(self) -> float:
        """Diagnostic E_r = sum_k p_rk E_k (the diagonal of U^T F U)."""
        return self.descriptors.energy_ev

    @property
    def centroid(self) -> np.ndarray:
        return self.descriptors.centroid


def _atom_in():
    return field(metadata={MODEL_INPUT: True})


def _atom_diag():
    return field(metadata={MODEL_INPUT: False})


# model-input atom fields that are embedded as categories, not fed to the continuous MLP
ATOM_CATEGORICAL = ("atomic_number", "formal_charge")


@dataclass
class AtomToken:
    """Model-input fields carry `model_input=True` metadata, in the order of the spec's u_a.

    Hybridization, the aromatic flag and the functional-group id are chemists' taxonomies:
    stored for tests, the interface and ablations, never model input. So are the
    finite-difference f+ / f- (relaxation-dominated) and the absolute position.
    """

    token_id: int = _atom_diag()
    atom_index: int = _atom_diag()
    map_number: int = _atom_diag()  # 0 = not assigned
    atomic_number: int = _atom_in()
    formal_charge: int = _atom_in()
    mulliken_charge: float = _atom_in()
    f_plus_fmo: float = _atom_in()  # frozen-orbital f+ (LUMO population)
    f_minus_fmo: float = _atom_in()  # frozen-orbital f- (HOMO population)
    dual: float = _atom_in()  # finite-difference dual descriptor f+ - f- (NaN when unavailable)
    fmo_weights: np.ndarray = _atom_in()  # (4,) HOMO-1, HOMO, LUMO, LUMO+1 block weights
    vbur: float = _atom_in()  # %Vbur, 3.5 Angstrom sphere
    chirality: float = _atom_in()  # +1 / -1 on a specified stereocentre, else 0 (`chirality`)
    f_plus: float = _atom_diag()  # finite-difference condensed Fukui (NaN when unavailable)
    f_minus: float = _atom_diag()
    hybridization: str = _atom_diag()
    aromatic: bool = _atom_diag()
    fg_id: int = _atom_diag()  # functional-group vocabulary index (0 none, 1 unk)
    position: np.ndarray = _atom_diag()  # (3,) Angstrom; geometry enters through distances


def atom_feature_names() -> list[str]:
    """Names of the continuous atom inputs, the columns of `MoleculeTokens.atom_features()`."""
    names: list[str] = []
    for f in fields(AtomToken):
        if not f.metadata[MODEL_INPUT] or f.name in ATOM_CATEGORICAL:
            continue
        names += [f"fmo_{lv}" for lv in LEVELS] if f.name == "fmo_weights" else [f.name]
    return names


def _atom_row(a: AtomToken) -> np.ndarray:
    parts = [
        np.asarray(getattr(a, f.name), dtype=float).ravel()
        for f in fields(AtomToken)
        if f.metadata[MODEL_INPUT] and f.name not in ATOM_CATEGORICAL
    ]
    return np.concatenate(parts)


@dataclass(frozen=True)
class CandidateConfig:
    """Which orbital tokens may be arrow donors / acceptors, decided on descriptors alone.

    Donor: a doubly occupied token. Acceptor: an empty token that is not a d-type polarization
    function (unless `allow_d_acceptors`) and, for a given donor of the same molecule, not a
    token excluded with it (`excluded_pair`, threshold `site_weight`). Until the radical
    extension, a token
    whose occupation lies in `fractional_band` (a SOMO at 1, a Fermi-smeared level of a
    small-gap molecule) is neither (`exclude_fractional`); its occupation stays the continuous
    model input either way.
    """

    allow_d_acceptors: bool = False  # GFN2 d polarization functions are basis artefacts
    full: float = 1.5  # occupation above this: doubly occupied
    empty: float = 0.5  # occupation below this: empty
    exclude_fractional: bool = True
    fractional_band: tuple[float, float] = (0.2, 1.8)  # open interval
    site_weight: float = 0.20  # `site_pairs_excluded`: weight that makes a near site a self-pair

    def is_fractional(self, occupation: float) -> bool:
        lo, hi = self.fractional_band
        return self.exclude_fractional and lo < occupation < hi


# groups 3-11: the d shell is valence (Ti(IV) accepts into it), not a polarization function
VALENCE_D = frozenset([*range(21, 30), *range(39, 48), *range(57, 80)])


def is_d_type(d: OrbitalDescriptors, z_main: int) -> bool:
    """A d polarization function: the main centre is mostly d and its element (atomic number
    `z_main`) has no valence d shell in GFN2 (Zn, Cd, Hg keep d10 in the core)."""
    return int(np.argmax(d.center_spd[0])) == 2 and int(z_main) not in VALENCE_D


def is_antibonding_partner(donor: OrbitalToken, acceptor: OrbitalToken) -> bool:
    """`acceptor` is an antibonding orbital of the donor's own bond, read from descriptors:
    both have at least two centres, the same two leading centres, a bonding (positive) overlap
    population on the donor and an antibonding (negative) one on the acceptor."""
    d, a = donor.descriptors, acceptor.descriptors
    shared = (
        len(d.all_center_atoms) >= 2
        and len(a.all_center_atoms) >= 2
        and set(d.all_center_atoms[:2]) == set(a.all_center_atoms[:2])
    )
    return shared and d.overlap_population > 0 and a.overlap_population < 0


BONDED = 1.3  # x covalent-radii sum, as in the QM layer and the contact table


def bonded_matrix(t: MoleculeTokens) -> np.ndarray:
    """(n_atoms, n_atoms) True for atom pairs within `BONDED` x their covalent-radii sum, False
    on the diagonal (record geometry)."""
    pt = Chem.GetPeriodicTable()
    z = [a.atomic_number for a in t.atoms]
    pos = np.array([a.position for a in t.atoms], dtype=float).reshape(-1, 3)
    radii = np.array([pt.GetRcovalent(x) for x in z])
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    return (d <= BONDED * (radii[:, None] + radii[None, :])) & ~np.eye(len(z), dtype=bool)


def site_pairs_excluded(
    t: MoleculeTokens, donors, a, acceptors, b, weight: float, bonded: np.ndarray | None = None
) -> np.ndarray:
    """Elementwise over site pairs of one molecule (donor token `donors[k]` at its site atom
    `a[k]`, acceptor `acceptors[k]` at `b[k]`): True where the two sites are the same atom or
    bonded and the acceptor carries at least `weight` of its population on a or the donor at
    least `weight` on b. Such a site pair is an interaction of a bond with itself (n(O) ->
    sigma*(O-H) of one O-H), not an arrow. A weight below the centre cut (0.15) reads as 0.
    `bonded`: `bonded_matrix(t)`, if already computed."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    near = (bonded_matrix(t) if bonded is None else bonded)[a, b] | (a == b)
    w_acc = np.array([_weight_on(o, x) for o, x in zip(acceptors, a, strict=True)])
    w_don = np.array([_weight_on(o, x) for o, x in zip(donors, b, strict=True)])
    return near & ((w_acc >= weight) | (w_don >= weight))


def _weight_on(o: OrbitalToken, atom: int) -> float:
    d = o.descriptors
    hit = np.flatnonzero(np.asarray(d.all_center_atoms) == atom)
    return float(np.asarray(d.all_center_weights)[hit[0]]) if len(hit) else 0.0


def site_excluded_matrix(t: MoleculeTokens, donors, acceptors, weight: float) -> np.ndarray:
    """(len(donors), len(acceptors)) True where every site pair of the two tokens (each centre
    of the donor x each centre of the acceptor) is excluded by `site_pairs_excluded`."""
    out = np.ones((len(donors), len(acceptors)), dtype=bool)
    cen_a = [np.asarray(o.descriptors.all_center_atoms, dtype=int) for o in acceptors]
    j = np.repeat(np.arange(len(acceptors)), [len(c) for c in cen_a])
    b = np.concatenate(cen_a) if cen_a else np.zeros(0, dtype=int)
    acc = [acceptors[k] for k in j]
    bonded = bonded_matrix(t)
    for i, d in enumerate(donors):
        for site in np.asarray(d.descriptors.all_center_atoms, dtype=int):
            a = np.full(len(j), site)
            hit = site_pairs_excluded(t, [d] * len(j), a, acc, b, weight, bonded)
            kept = np.zeros(len(acceptors), dtype=bool)
            np.logical_or.at(kept, j, ~hit)
            out[i] &= ~kept
    return out


def excluded_pair(
    donor: OrbitalToken,
    acceptor: OrbitalToken,
    t: MoleculeTokens,
    weight: float = CandidateConfig.site_weight,
) -> bool:
    """A donor-acceptor pair of molecule `t` that is never an arrow candidate: the donor's own
    antibond (the spec's rule), or a pair whose every site pair is a bond interacting with
    itself (`site_excluded_matrix`). A centre shared at a small weight does not exclude: a
    tetrahedral intermediate's C-Cl sigma* carries 0.20 on its O(-), and the collapse
    O -> C stays a candidate (docs/decisions.md). Only compare tokens of one molecule."""
    if is_antibonding_partner(donor, acceptor):
        return True
    return bool(site_excluded_matrix(t, [donor], [acceptor], weight)[0, 0])


@dataclass
class MoleculeTokens:
    smiles: str
    method: str
    conformer_id: int
    atoms: list[AtomToken]
    orbitals: list[OrbitalToken]
    S: np.ndarray
    F: np.ndarray  # Hartree
    ao_atom: np.ndarray
    ao_l: np.ndarray
    pm_sweeps_occ: int = 0
    pm_sweeps_virt: int = 0
    pm_converged: bool = True
    qm_warnings: str = ""
    schema_version: int = SCHEMA_VERSION
    fg_vocab_size: int = 0
    fermi_smeared: bool = False  # an MO is > 0.05 from its aufbau occupation (`is_fermi_smeared`)
    digest: str = field(default="", repr=False, compare=False)  # content sha256 (provenance)
    source_path: str = field(default="", repr=False, compare=False)  # file it was loaded from

    @property
    def C_loc(self) -> np.ndarray:
        """(nao, n_orbitals) localized coefficients, orbital tokens as columns."""
        nao = self.S.shape[0]
        return np.column_stack([o.coeff_vector for o in self.orbitals]).reshape(nao, -1)

    def occupied(self) -> list[OrbitalToken]:
        return [o for o in self.orbitals if o.kind == OrbitalKind.OCCUPIED]

    def virtual(self) -> list[OrbitalToken]:
        return [o for o in self.orbitals if o.kind == OrbitalKind.VIRTUAL]

    def symbol(self, atom_index: int) -> str:
        return Chem.GetPeriodicTable().GetElementSymbol(self.atoms[atom_index].atomic_number)

    def assign_map_numbers(self, index_to_map: dict[int, int]) -> None:
        for i, m in index_to_map.items():
            self.atoms[i].map_number = int(m)

    def orbital_features(self) -> np.ndarray:
        """(n_orbitals, n_features) continuous model input; see `feature_names()`."""
        n = len(feature_names())
        return np.array([o.descriptors.continuous_vector() for o in self.orbitals]).reshape(-1, n)

    def orbital_spectra(self) -> np.ndarray:
        """(n_orbitals, spectral_max, 2) spectral sets (E_k eV, p_rk); padding rows have p = 0."""
        rows = [o.descriptors.spectral_set() for o in self.orbitals]
        return np.array(rows, dtype=float).reshape(-1, DescriptorConfig().spectral_max, 2)

    def orbital_centres(self) -> np.ndarray:
        """(n_orbitals, n_atoms) population w_a of each orbital on each of its centres, else 0."""
        out = np.zeros((len(self.orbitals), len(self.atoms)))
        for k, o in enumerate(self.orbitals):
            out[k, o.descriptors.all_center_atoms] = o.descriptors.all_center_weights
        return out

    @staticmethod
    def feature_names() -> list[str]:
        return feature_names()

    def _atom_raw(self) -> np.ndarray:
        n = len(atom_feature_names())
        return np.array([_atom_row(a) for a in self.atoms], dtype=float).reshape(-1, n)

    def atom_features(self) -> np.ndarray:
        """(n_atoms, n) continuous atom inputs (`atom_feature_names()`), missing values as 0."""
        return np.nan_to_num(self._atom_raw(), nan=0.0)

    def atom_missing(self) -> np.ndarray:
        """(n_atoms, n) True where `atom_features()` holds a 0 for an unavailable value
        (finite-difference dual after FUKUI_SCF_FAIL, a frontier level that does not exist)."""
        return np.isnan(self._atom_raw())

    def atom_categorical(self) -> np.ndarray:
        """(n_atoms, 2) integer inputs that are embedded: atomic number, formal charge."""
        rows = [[getattr(a, f) for f in ATOM_CATEGORICAL] for a in self.atoms]
        return np.array(rows, dtype=np.int64).reshape(-1, len(ATOM_CATEGORICAL))

    def is_d_type(self, o: OrbitalToken) -> bool:
        """`o` is a d polarization function (see the module-level `is_d_type`)."""
        return is_d_type(
            o.descriptors, self.atoms[int(o.descriptors.center_atoms[0])].atomic_number
        )

    def donor_candidates(self, config: CandidateConfig | None = None) -> list[OrbitalToken]:
        """Doubly occupied tokens: where an arrow may start."""
        config = config or CandidateConfig()
        return [
            o
            for o in self.occupied()
            if o.descriptors.occupation > config.full
            and not config.is_fractional(o.descriptors.occupation)
        ]

    def acceptor_candidates(
        self, config: CandidateConfig | None = None, donor: OrbitalToken | None = None
    ) -> list[OrbitalToken]:
        """Empty tokens an arrow may end on: d-type tokens only with `allow_d_acceptors`, and
        never a token excluded with `donor` (of this molecule) when one is given."""
        config = config or CandidateConfig()
        return [
            o
            for o in self.virtual()
            if o.descriptors.occupation < config.empty
            and not config.is_fractional(o.descriptors.occupation)
            and (config.allow_d_acceptors or not self.is_d_type(o))
            and (donor is None or not excluded_pair(donor, o, self, config.site_weight))
        ]

    def atoms_with_only_d_virtuals(self) -> list[int]:
        """Atoms that are a centre of at least one virtual token, all of which are d-type.

        An arrow sink on such an atom has no acceptor candidate by default; the alignment phase
        reports how many dataset sinks land there.
        """
        d_type: dict[int, set[bool]] = {}
        for o in self.virtual():
            for a in o.descriptors.all_center_atoms:
                d_type.setdefault(int(a), set()).add(self.is_d_type(o))
        return sorted(a for a, flags in d_type.items() if flags == {True})


def _check_mol_matches(mol: Chem.Mol, rec: QMRecord) -> None:
    z_mol = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if z_mol != [int(z) for z in rec.atomic_numbers]:
        raise ValueError("mol atom order differs from the QM record; build it with prepare_mol")


def _value(arr: np.ndarray, i: int) -> float:
    return float(arr[i]) if len(arr) > i else float("nan")


def atom_tokens(rec: QMRecord, mol: Chem.Mol, fg_ids: np.ndarray) -> list[AtomToken]:
    out = []
    fmo = frontier_weights(rec)
    f_plus_fmo, f_minus_fmo = frozen_orbital_fukui_blocks(rec)
    chirality = chirality_signs(mol, rec.coords_ang)
    for i, atom in enumerate(mol.GetAtoms()):
        out.append(
            AtomToken(
                token_id=i,
                atom_index=i,
                map_number=atom.GetAtomMapNum(),
                atomic_number=int(rec.atomic_numbers[i]),
                formal_charge=atom.GetFormalCharge(),
                hybridization=str(atom.GetHybridization()),
                aromatic=bool(atom.GetIsAromatic()),
                mulliken_charge=float(rec.charges[i]),
                f_plus=_value(rec.fukui_plus, i),
                f_minus=_value(rec.fukui_minus, i),
                dual=_value(rec.fukui_dual, i),
                f_plus_fmo=float(f_plus_fmo[i]),
                f_minus_fmo=float(f_minus_fmo[i]),
                vbur=_value(rec.buried_volume, i),
                chirality=float(chirality[i]),
                position=rec.coords_ang[i].copy(),
                fg_id=int(fg_ids[i]),
                fmo_weights=fmo[i].copy(),
            )
        )
    return out


def _orbital_token(
    token_id: int, kind: OrbitalKind, c: np.ndarray, cls: Classification, desc: OrbitalDescriptors
) -> OrbitalToken:
    return OrbitalToken(
        token_id=token_id,
        kind=kind,
        descriptors=desc,
        coeff_vector=c.copy(),
        type=cls.type,
        parent_atoms=cls.parent_atoms,
        parent_weights=cls.parent_weights,
        s_share=cls.s_share,
        p_share=cls.p_share,
        d_share=cls.d_share,
        overlap_population=cls.overlap_population,
        metal_bound=cls.metal_bound,
        metal_atom=cls.metal_atom,
    )


def _block_tokens(rec, kind, C, U, idx, first_id, classify_config, descriptor_config):
    """Tokens of one localized block, ordered by ascending diagnostic energy."""
    occupied = kind == OrbitalKind.OCCUPIED
    classes = classify_block(
        C,
        rec.S,
        rec.ao_atom,
        rec.ao_l,
        rec.coords_ang,
        occupied,
        classify_config,
        rec.atomic_numbers,
    )
    e_block, n_block = rec.orbital_energies_ev[idx], rec.occupations[idx]
    descs = [
        build_descriptors(
            C[:, k],
            classes[k].populations,
            rec.S,
            rec.ao_atom,
            rec.ao_l,
            rec.coords_ang,
            U[:, k],
            e_block,
            n_block,
            idx,
            descriptor_config,
        )
        for k in range(C.shape[1])
    ]
    order = np.argsort([d.energy_ev for d in descs], kind="stable")
    return [
        _orbital_token(first_id + i, kind, C[:, k], classes[k], descs[k])
        for i, k in enumerate(order)
    ]


def orbital_tokens(
    rec: QMRecord,
    loc: LocalizedOrbitals,
    first_id: int,
    classify_config: ClassifyConfig | None = None,
    descriptor_config: DescriptorConfig | None = None,
) -> list[OrbitalToken]:
    """Occupied then virtual tokens, each block by ascending diagnostic energy."""
    out: list[OrbitalToken] = []
    blocks = (
        (OrbitalKind.OCCUPIED, loc.C_occ, loc.U_occ, loc.occ_index),
        (OrbitalKind.VIRTUAL, loc.C_virt, loc.U_virt, loc.virt_index),
    )
    for kind, C, U, idx in blocks:
        if C.shape[1] == 0:
            continue
        out += _block_tokens(
            rec, kind, C, U, idx, first_id + len(out), classify_config, descriptor_config
        )
    return out


def build_molecule_tokens(
    rec: QMRecord,
    mol: Chem.Mol | None = None,
    pm_config: PMConfig | None = None,
    classify_config: ClassifyConfig | None = None,
    vocab: FGVocab | None = None,
    descriptor_config: DescriptorConfig | None = None,
) -> MoleculeTokens:
    """Localize, describe, classify and tokenize one QM record. Deterministic: PM starts from
    the canonical MOs with a fixed pair schedule, so the same record gives the same tokens.

    `mol` defaults to `prepare_mol(rec.atom_order_smiles)`; pass it when atom maps matter.
    """
    mol = mol if mol is not None else prepare_mol(rec.atom_order_smiles)
    _check_mol_matches(mol, rec)
    vocab = vocab or default_vocab()
    fg_ids = vocab.atom_ids(Chem.MolFromSmiles(rec.atom_order_smiles), rec.n_atoms)
    loc = localize_record(rec, pm_config)
    atoms = atom_tokens(rec, mol, fg_ids)
    orbitals = orbital_tokens(rec, loc, len(atoms), classify_config, descriptor_config)
    return MoleculeTokens(
        smiles=rec.smiles,
        method=rec.method,
        conformer_id=rec.conformer_id,
        atoms=atoms,
        orbitals=orbitals,
        S=rec.S,
        F=rec.fock,
        ao_atom=rec.ao_atom,
        ao_l=rec.ao_l,
        pm_sweeps_occ=loc.stats_occ.n_sweeps,
        pm_sweeps_virt=loc.stats_virt.n_sweeps,
        pm_converged=loc.converged,
        qm_warnings=rec.warnings,
        fg_vocab_size=len(vocab),
        fermi_smeared=is_fermi_smeared(rec),
    )


_ATOM_FIELDS = (
    "atom_index",
    "map_number",
    "atomic_number",
    "formal_charge",
    "hybridization",
    "aromatic",
    "mulliken_charge",
    "f_plus",
    "f_minus",
    "dual",
    "f_plus_fmo",
    "f_minus_fmo",
    "vbur",
    "chirality",
    "position",
    "fg_id",
    "fmo_weights",
)
_ORB_DIAG_SCALARS = ("s_share", "p_share", "d_share", "overlap_population")
_DESC_FIELDS = tuple(f.name for f in fields(OrbitalDescriptors))
_RAGGED_DESC = ("full_energies", "full_weights")  # length = block size, padded on disk
_RAGGED_CENTRES = {"all_center_atoms": (-1, np.int32), "all_center_weights": (np.nan, float)}
_META = (
    "smiles",
    "method",
    "conformer_id",
    "pm_sweeps_occ",
    "pm_sweeps_virt",
    "pm_converged",
    "qm_warnings",
    "schema_version",
    "fg_vocab_size",
    "fermi_smeared",
)


def _padded(rows: list[tuple], fill, dtype) -> np.ndarray:
    width = max((len(r) for r in rows), default=0)
    out = np.full((len(rows), width), fill, dtype=dtype)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def _payload(t: MoleculeTokens) -> dict[str, np.ndarray]:
    pay = {k: np.asarray(getattr(t, k)) for k in _META}
    for f in _ATOM_FIELDS:
        pay[f"atom_{f}"] = np.asarray([getattr(a, f) for a in t.atoms])
    pay["orb_kind"] = np.asarray([str(o.kind) for o in t.orbitals])
    pay["orb_type"] = np.asarray([str(o.type) for o in t.orbitals])
    for f in _ORB_DIAG_SCALARS:
        pay[f"orb_{f}"] = np.asarray([getattr(o, f) for o in t.orbitals], dtype=float)
    pay["orb_parent_atoms"] = _padded([o.parent_atoms for o in t.orbitals], -1, np.int32)
    pay["orb_parent_weights"] = _padded([o.parent_weights for o in t.orbitals], np.nan, float)
    pay["orb_metal_bound"] = np.asarray([o.metal_bound for o in t.orbitals], dtype=bool)
    pay["orb_metal_atom"] = np.asarray([o.metal_atom for o in t.orbitals], dtype=np.int32)
    for f in _DESC_FIELDS:
        values = [getattr(o.descriptors, f) for o in t.orbitals]
        if f in _RAGGED_DESC:
            values = [tuple(np.asarray(v, dtype=float)) for v in values]
            pay[f"desc_{f}"] = _padded(values, 0.0, float)
        elif f in _RAGGED_CENTRES:
            fill, dtype = _RAGGED_CENTRES[f]
            pay[f"desc_{f}"] = _padded([tuple(v) for v in values], fill, dtype)
        else:
            pay[f"desc_{f}"] = np.asarray(values)
    pay["desc_n_block"] = np.asarray([len(o.descriptors.full_weights) for o in t.orbitals])
    pay["C_loc"] = t.C_loc
    for k in ("S", "F", "ao_atom", "ao_l"):
        pay[k] = np.asarray(getattr(t, k))
    return pay


def token_config(
    pm_config: PMConfig | None = None, descriptor_config: DescriptorConfig | None = None
) -> dict:
    """The configuration a token file is stamped with and checked against: the descriptor
    thresholds (tau_atom, spectral cutoff) and the PM convergence settings."""
    pm = pm_config or PMConfig()
    return {
        **asdict(descriptor_config or DescriptorConfig()),
        "pm_tol": pm.tol,
        "pm_max_sweeps": pm.max_sweeps,
    }


def token_digest(tokens: MoleculeTokens) -> str:
    """The content hash of the tokens (the one their file header carries); computed once."""
    if not tokens.digest:
        tokens.digest = prov.array_digest(_payload(tokens))
    return tokens.digest


def save_tokens(
    tokens: MoleculeTokens,
    path: Path,
    inputs: dict[str, str] | None = None,
    config: dict | None = None,
) -> Path:
    """Write as .npz (no pickle) with its provenance header, atomically. `inputs`: the QM
    record it was built from (`prov.file_input(qm_path)`); `config`: `token_config(...)`."""
    arrays = _payload(tokens)
    tokens.digest = prov.array_digest(arrays)
    header = prov.stamp(
        "tokens", SCHEMA_VERSION, inputs or {}, config or token_config(), tokens.digest
    )
    out = prov.save_npz(path, arrays, header)
    tokens.source_path = str(out)
    return out


def _load_atoms(z: dict) -> list[AtomToken]:
    n = len(z["atom_atom_index"])
    cols = {f: z[f"atom_{f}"] for f in _ATOM_FIELDS}
    atoms = []
    for i in range(n):
        kw = {f: cols[f][i] for f in _ATOM_FIELDS}
        kw["position"] = np.asarray(kw["position"], dtype=float)
        kw["fmo_weights"] = np.asarray(kw["fmo_weights"], dtype=float)
        kw["hybridization"] = str(kw["hybridization"])
        for f in ("atom_index", "map_number", "atomic_number", "formal_charge", "fg_id"):
            kw[f] = int(kw[f])
        kw["aromatic"] = bool(kw["aromatic"])
        for f in (
            "mulliken_charge",
            "f_plus",
            "f_minus",
            "dual",
            "f_plus_fmo",
            "f_minus_fmo",
            "vbur",
            "chirality",
        ):
            kw[f] = float(kw[f])
        atoms.append(AtomToken(token_id=i, **kw))
    return atoms


def _load_descriptors(z: dict, k: int) -> OrbitalDescriptors:
    kw = {}
    for f in fields(OrbitalDescriptors):
        value = z[f"desc_{f.name}"][k]
        if f.name in _RAGGED_DESC:
            value = value[: int(z["desc_n_block"][k])]
        elif f.name in _RAGGED_CENTRES:
            value = value[: int((z["desc_all_center_atoms"][k] >= 0).sum())]
        if f.type == "int":
            kw[f.name] = int(value)
        elif f.type == "float":
            kw[f.name] = float(value)
        else:
            kw[f.name] = np.asarray(value)
    return OrbitalDescriptors(**kw)


def _load_orbitals(z: dict, first_id: int) -> list[OrbitalToken]:
    out = []
    for k in range(len(z["orb_kind"])):
        parents = z["orb_parent_atoms"][k]
        keep = parents >= 0
        out.append(
            OrbitalToken(
                token_id=first_id + k,
                kind=OrbitalKind(str(z["orb_kind"][k])),
                descriptors=_load_descriptors(z, k),
                coeff_vector=np.asarray(z["C_loc"][:, k], dtype=float),
                type=OrbitalType(str(z["orb_type"][k])),
                parent_atoms=tuple(int(p) for p in parents[keep]),
                parent_weights=tuple(float(w) for w in z["orb_parent_weights"][k][keep]),
                s_share=float(z["orb_s_share"][k]),
                p_share=float(z["orb_p_share"][k]),
                d_share=float(z["orb_d_share"][k]),
                overlap_population=float(z["orb_overlap_population"][k]),
                metal_bound=bool(z["orb_metal_bound"][k]),
                metal_atom=int(z["orb_metal_atom"][k]),
            )
        )
    return out


TokenSchemaError = prov.StaleArtifact  # the old name: a stale token file; rebuild it


def load_tokens(path: Path, config: dict | None = None, verify: bool = True) -> MoleculeTokens:
    """Read a token file. Its header is verified against the current schema, `config`
    (default `token_config()`) and the QM record it was built from; raises StaleArtifact.
    `verify=False` is for diagnostics only."""
    header = (
        prov.verify(path, "tokens", SCHEMA_VERSION, config or token_config())
        if verify
        else prov.read_header(path)
    )
    with np.load(Path(path), allow_pickle=False) as npz:
        z = {k: npz[k] for k in npz.files if k != prov.HEADER_KEY}
    atoms = _load_atoms(z)
    meta = {k: z[k].item() for k in _META}
    for k in ("conformer_id", "pm_sweeps_occ", "pm_sweeps_virt", "schema_version", "fg_vocab_size"):
        meta[k] = int(meta[k])
    meta["pm_converged"] = bool(meta["pm_converged"])
    meta["fermi_smeared"] = bool(meta["fermi_smeared"])
    return MoleculeTokens(
        atoms=atoms,
        orbitals=_load_orbitals(z, len(atoms)),
        S=z["S"],
        F=z["F"],
        ao_atom=z["ao_atom"],
        ao_l=z["ao_l"],
        digest=header.content_sha256 if header else "",
        source_path=str(path),
        **meta,
    )


def describe_parents(tokens: MoleculeTokens, orb: OrbitalToken) -> str:
    text = " ".join(
        f"{tokens.symbol(a)}{a}:{w:.2f}"
        for a, w in zip(orb.parent_atoms, orb.parent_weights, strict=True)
    )
    if orb.metal_bound:
        text += f" [{tokens.symbol(orb.metal_atom)}{orb.metal_atom}-bound]"
    return text


def token_table(tokens: MoleculeTokens, title: str | None = None) -> Table:
    """One row per orbital token: descriptors first, then the diagnostic type and parents."""
    table = Table(title=title or tokens.smiles)
    cols = (
        "id",
        "kind",
        "occ",
        "centres",
        "cos",
        "OP",
        "Rg",
        "n_spec",
        "E (eV)",
        "type",
        "parents",
    )
    for col in cols:
        table.add_column(col, justify="left" if col in ("kind", "type", "parents") else "right")
    for o in tokens.orbitals:
        d = o.descriptors
        table.add_row(
            str(o.token_id),
            "occ" if o.kind == OrbitalKind.OCCUPIED else "virt",
            f"{d.occupation:.1f}",
            str(d.n_centers),
            f"{d.cos_character():.2f}",
            f"{d.overlap_population:+.2f}",
            f"{d.radius_gyration:.2f}",
            str(d.n_spectral()),
            f"{d.energy_ev:8.2f}",
            str(o.type),
            describe_parents(tokens, o),
        )
    return table
