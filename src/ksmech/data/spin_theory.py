"""The theory rules of open-shell plan P2 step 3 (B): a species' lowest spin state predicted
from principles, independently of the spGFN2 scan (A), for the combination rule (C) that writes
the species table (spin_table_v2 from spin_theory_v2). Read by that step and the tests, never by
`data.spin`.

`theory_verdict(mol, rec)` reads the drawing (`mol`, H-explicit, `qm.embed.prepare_mol` atom
order) and one restricted GFN2 record of it (`rec`: ALPB water, the production geometry off any
symmetric saddle (the scan re-optimizes it from a displaced start, v2), the lowest spin the
electron count allows, so a degenerate open shell holds its electrons as fractional
occupations of degenerate MOs). It returns the predicted number of unpaired electrons (2S) or None
('undetermined'), the rule chain that gave it and the citations. No rule names a species: every
rule reads an element, a count, a geometry or the MOs.

The rules, first applicable wins:

1. Transition metals (`data.domain.GROUP`): d^n from `domain.oxidation_state` / `d_count`
   (the dative law's bookkeeping). By count, in any geometry: d0 and d10 are closed, d1 and d9
   have one unpaired electron. Otherwise ligand field theory on the coordination polyhedron
   (the drawn neighbours, the nearest ideal polyhedron by the RMS of the sorted L-M-L angles,
   from `rec.coords_ang`): a first-row tetrahedron is high spin (Delta_t ~ 4/9 Delta_o); a
   square plane holding d8 is low spin; an octahedron holds d2, d3 and d8 with one outcome and
   d4-d7 by the field: every donor below water in the spectrochemical series (the halides) is
   high spin, every donor among its strong-field end (C and P donors) low spin, a 4d/5d metal
   low spin, anything else (O, N, S donors, a mixed set) undetermined. A Jahn-Teller-active
   configuration is annotated, never decided by. A polyhedron with no rule is undetermined, and
   so is a species with two open-shell metal centres (their exchange coupling is not given by
   ligand field theory) or with drawn radicals beside its metal.
2. A single atom: Hund's rules on its open subshell (term ^{2S+1}L).
3. A degenerate open shell: one partially filled group of degenerate MOs (energies within
   1e-4 eV) that spans one irreducible representation of the point group (the characters of
   the group operations on the MO coefficients); Hund's first rule gives the most unpaired
   electrons the group allows, and the ground term's spatial part is the exterior power of the
   irrep (O2's pi_g*^2 gives 3Sigma_g-). Finite groups are named by Mulliken letter with g/u
   and primes; the numeric subscripts are not assigned (the characters are recorded).
4. A carbene centre (a neutral C with two bonds' worth of valence), first applicable wins
   (v2 adds a and b):
   a. an unsaturated carbene, the centre's one neighbour a carbon doubly bonded to it
      (R2C=C:, a vinylidene): a singlet, the sigma pair in an sp orbital far below p_pi
      (Stang);
   b. two pi-donor neighbours (a halogen, or N, O, P, S with a lone pair): a singlet, their
      lone pairs donate into the empty p_pi and raise it (Bourissou, the dihalocarbenes and
      diaminocarbenes);
   c. Gleiter-Hoffmann on the sigma-p_pi gap, read as the restricted GFN2 HOMO-LUMO gap (in
      place of their extended-Hueckel separation) when both MOs sit mostly on the centre: a
      singlet above 2 eV, a triplet below 1.5 eV, undetermined between. The GFN2 gap is a
      proxy: at its own singlet geometry dibromocarbene reads 1.40 eV, which rule b decides.
5. Otherwise undetermined.

The AO operators: an operation maps atom i to perm[i] and rotates each shell by the real
spherical harmonics' representation of R (tblite's order: p as y, z, x; d as xy, yz, z2, xz,
x2-y2), fitted on sample points; every operator is checked to leave the overlap matrix
invariant (to OPERATOR_TOL: production geometries are symmetric only approximately) before its
characters are used, so a convention error makes the irreps unknown, never wrong.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
from rdkit import Chem

from ksmech.data import domain
from ksmech.orbitals.frontier import EXACT_EV
from ksmech.qm import point_group as pg
from ksmech.qm.records import QMRecord

VERSION = "spin_theory_v2"
THEORY_TABLE_PATH = Path(__file__).resolve().parent / "spin_tables" / "theory_v2.json"
SYMMETRY_TOLERANCE_ANG = pg.DEFAULT_TOLERANCE  # the point group at 0.1 Angstrom
DEGENERACY_EV = EXACT_EV  # equal MO energies (1e-4 eV), as the exact-degeneracy frontier columns
# |T^T S T - S| above this: the structure is too far from its group (or an AO convention is
# wrong, which gives O(0.1)) for the characters to be read; exact symmetry gives 1e-15
OPERATOR_TOL = 1e-3
CHARACTER_TOL = 1e-2  # a character within this of its exact value (approximate symmetry)
GH_SINGLET_EV = 2.0  # Gleiter-Hoffmann: a sigma-p_pi gap above this, a singlet
GH_TRIPLET_EV = 1.5  # below this, a triplet
LINEAR_PROBE_RAD = 1.0  # a C_phi rotation at an angle no finite axis has, for lambda
PI_DONORS = frozenset([7, 8, 9, 15, 16, 17, 35, 53])  # N, O, F, P, S, Cl, Br, I (with a pair)

CITATIONS = {
    "gleiter_hoffmann": "R. Gleiter and R. Hoffmann, J. Am. Chem. Soc. 90, 5457 (1968)",
    "stang": "P. J. Stang, Chem. Rev. 78, 383 (1978) (unsaturated carbenes)",
    "bourissou": (
        "D. Bourissou, O. Guerret, F. P. Gabbai and G. Bertrand, Chem. Rev. 100, 39 (2000)"
    ),
    "tsuchida": "R. Tsuchida, Bull. Chem. Soc. Jpn. 13, 388 (1938) (the spectrochemical series)",
    # the textbook statements of ligand field theory (no page numbers are claimed): the
    # spectrochemical series in Tsuchida's order, high and low spin in Oh, Delta_t ~ 4/9 Delta_o
    # and high-spin tetrahedra, low-spin square-planar d8, larger Delta for 4d/5d, Jahn-Teller
    "ligand_field": (
        "G. L. Miessler, P. J. Fischer and D. A. Tarr, Inorganic Chemistry, 5th ed. (Pearson, "
        "2014), ch. 10, Coordination Chemistry II: Bonding"
    ),
    "hund": "F. Hund, Z. Phys. 33, 345 (1925)",
    "herzberg": (
        "G. Herzberg, Molecular Spectra and Molecular Structure I. Spectra of Diatomic "
        "Molecules, 2nd ed. (Van Nostrand, 1950)"
    ),
    "jahn_teller": "H. A. Jahn and E. Teller, Proc. R. Soc. Lond. A 161, 220 (1937)",
}

# donor elements by their place in Tsuchida's series: the halides sit below water, the C and P
# donors (CN-, CO, PR3) at the strong-field end; O, N and S donors span the middle
WEAK_DONORS = frozenset([9, 17, 35, 53])
STRONG_DONORS = frozenset([6, 15])
_LETTERS_L = "SPDFGHIK"


@dataclass
class TheoryVerdict:
    smiles: str
    unpaired: int | None  # 2S of the predicted lowest state; None = undetermined
    rule_chain: list[str]
    citations: list[str]
    point_group: str | None
    term: str | None = None
    n_electrons: int = 0
    record_unpaired: int = 0
    metals: list[dict] = field(default_factory=list)
    open_shell: dict | None = None
    frontier: list[dict] = field(default_factory=list)
    carbene: dict | None = None
    notes: list[str] = field(default_factory=list)
    version: str = VERSION

    @property
    def undetermined(self) -> bool:
        return self.unpaired is None

    def as_dict(self) -> dict:
        return asdict(self)


# ---- the AO operators and the characters


def _harmonics(ell: int, r: np.ndarray) -> np.ndarray:
    """Real solid harmonics in tblite's AO order, each with the same norm on the sphere."""
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    if ell == 0:
        return np.ones((len(r), 1))
    if ell == 1:
        return np.c_[y, z, x]
    s3 = math.sqrt(3.0)
    return np.c_[
        s3 * x * y,
        s3 * y * z,
        (3 * z * z - (r * r).sum(1)) / 2,
        s3 * x * z,
        s3 / 2 * (x * x - y * y),
    ]


@cache
def _sample_points() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(40, 3))


def shell_rotation(ell: int, R: np.ndarray) -> np.ndarray:
    """D with Y_m(R^-1 r) = sum_m' D[m', m] Y_m'(r): the shell's functions under R."""
    pts = _sample_points()
    return np.linalg.lstsq(_harmonics(ell, pts), _harmonics(ell, pts @ R), rcond=None)[0]


def ao_operator(ao_atom: np.ndarray, ao_l: np.ndarray, R: np.ndarray, perm) -> np.ndarray:
    """The operation's matrix on the AO basis: atom i's shells go to atom perm[i], rotated."""
    T = np.zeros((len(ao_atom), len(ao_atom)))
    for i in range(int(ao_atom.max()) + 1):
        for ell in sorted(set(ao_l[ao_atom == i].tolist())):
            cols = np.flatnonzero((ao_atom == i) & (ao_l == ell))
            rows = np.flatnonzero((ao_atom == int(perm[i])) & (ao_l == ell))
            T[np.ix_(rows, cols)] = shell_rotation(ell, R)
    return T


def _axis_rotation(u: np.ndarray, phi: float) -> np.ndarray:
    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(phi) * k + (1 - math.cos(phi)) * k @ k


def _ops_of(s: pg.Search) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Named operations (R, perm) of the group: the finite group's own, keyed op0, op1, ...;
    for a linear structure C_phi, a sigma_v and, in D∞h, the inversion; None for an atom."""
    if s.name == "Kh":
        return None
    if s.operations:
        return {f"op{k}": (o.R, o.perm) for k, o in enumerate(s.operations)}
    u = next(e.vector for e in s.elements if e.kind == "C" and e.order is None)
    n = len(s.frame.z)
    plane = next(e.vector for e in s.elements if e.label == "∞σv")
    out = {
        "E": (np.eye(3), np.arange(n)),
        "C_phi": (_axis_rotation(u, LINEAR_PROBE_RAD), np.arange(n)),
        "sigma_v": (np.eye(3) - 2 * np.outer(plane, plane), np.arange(n)),
    }
    inv = [e for e in s.elements if e.kind == "i"]
    if inv:
        out["i"] = (-np.eye(3), inv[0].ops[0].perm)
    return out


def representation(rec: QMRecord, s: pg.Search) -> dict[str, np.ndarray] | None:
    """Every operation's AO matrix T, or None when the group has none (an atom) or one T does
    not leave S invariant (then no character is read: unknown, never wrong)."""
    ops = _ops_of(s)
    if ops is None:
        return None
    out = {}
    for name, (R, perm) in ops.items():
        T = ao_operator(rec.ao_atom, rec.ao_l, R, perm)
        if np.abs(T.T @ rec.S @ T - rec.S).max() > OPERATOR_TOL:
            return None
        out[name] = T
    return out


def block_matrices(rec: QMRecord, reps: dict[str, np.ndarray], mos) -> dict[str, np.ndarray]:
    """rho(g) = X^T S T(g) X on the MOs' span (X the S-orthonormal coefficient columns)."""
    X = rec.C[:, list(mos)]
    return {g: X.T @ rec.S @ T @ X for g, T in reps.items()}


def exterior_characters(rho: dict[str, np.ndarray], k: int) -> dict[str, float]:
    """Characters of the k-th exterior power: the k-th elementary symmetric polynomial of each
    rho(g)'s eigenvalues, (-1)^k times the characteristic polynomial's k-th coefficient."""
    return {g: float(np.real((-1) ** k * np.poly(m)[k])) for g, m in rho.items()}


def _identity_like(R: np.ndarray, det: int) -> bool:
    return det == 1 and np.allclose(R, np.eye(3), atol=1e-6)


def _op_name(op: pg.Op, elements: list[pg.Element]) -> str:
    """'E', 'i', a mirror's label, or C_k / S_k with the element's primes."""
    if _identity_like(op.R, op.det):
        return "E"
    for e in elements:
        if any(o.key == op.key for o in e.ops):
            if e.kind in ("sigma", "i"):
                return e.label
            primes = "".join(ch for ch in e.label if ch in "′″")
            return f"{e.kind}{pg.element_order(pg.op_angle(op))}{primes}"
    return "?"


def class_characters(s: pg.Search, chars: dict[str, float]) -> list[dict]:
    """A finite group's characters per conjugacy class: name, size, character."""
    group = {o.key: o for o in s.operations}
    cls = pg.conjugacy_classes(group)
    out: dict[int, dict] = {}
    for k, o in enumerate(s.operations):
        c = cls[o.key]
        if c not in out:
            out[c] = {"class": _op_name(o, s.elements), "size": 0, "character": chars[f"op{k}"]}
        out[c]["size"] += 1
    return [{**v, "character": round(v["character"], 6)} for v in out.values()]


def _finite_label(s: pg.Search, chars: dict[str, float]) -> tuple[bool, str | None]:
    """(irreducible, the Mulliken letter with g/u and primes; numeric subscripts are not
    assigned) of one character set of a finite group."""
    ops = s.operations
    vals = np.array([chars[f"op{k}"] for k in range(len(ops))])
    if abs((vals**2).sum() / len(ops) - 1.0) > CHARACTER_TOL:
        return False, None
    dim = round(next(v for o, v in zip(ops, vals, strict=True) if _identity_like(o.R, o.det)))
    letter = {2: "E", 3: "T", 4: "G", 5: "H"}.get(dim, "A")
    principal = [e for e in s.elements if e.principal and e.kind == "C"]
    if dim == 1 and principal:
        gen = min(principal[0].ops, key=pg.op_angle, default=None)
        k = next(i for i, o in enumerate(ops) if gen is not None and o.key == gen.key)
        letter = "A" if vals[k] > 0 else "B"
    inv = [i for i, o in enumerate(ops) if o.det == -1 and np.allclose(o.R, -np.eye(3), atol=1e-6)]
    if inv:
        return True, letter + ("g" if vals[inv[0]] > 0 else "u")
    mirror_h = [e for e in s.elements if e.kind == "sigma" and e.label in ("σh", "σ")]
    if mirror_h:
        k = next(i for i, o in enumerate(ops) if o.key == mirror_h[0].ops[0].key)
        return True, letter + ("′" if vals[k] > 0 else "″")
    return True, letter


def _linear_label(chars: dict[str, float]) -> tuple[bool, str | None]:
    """Sigma+/-, Pi, Delta, Phi with g/u, from the characters of C_phi, sigma_v and i."""
    dim = round(chars["E"])
    if dim == 1:
        if abs(chars["C_phi"] - 1.0) > CHARACTER_TOL:
            return False, None
        name = "Sigma" + ("+" if chars["sigma_v"] > 0 else "-")
    elif dim == 2:
        lam = [
            k
            for k in range(1, 5)
            if abs(2 * math.cos(k * LINEAR_PROBE_RAD) - chars["C_phi"]) < CHARACTER_TOL
        ]
        if len(lam) != 1:
            return False, None
        name = ("Pi", "Delta", "Phi", "Gamma")[lam[0] - 1]
    else:
        return False, None
    if "i" in chars:
        parity = "_g" if chars["i"] > 0 else "_u"
        name = name[:5] + parity + name[5:] if name.startswith("Sigma") else name + parity
    return True, name


def label_of(s: pg.Search, chars: dict[str, float]) -> tuple[bool, str | None]:
    """(irreducible, label) of one set of characters of the structure's group."""
    if s.operations:
        return _finite_label(s, chars)
    return _linear_label(chars)


# ---- MO groups, the frontier and the open shell


def degenerate_groups(rec: QMRecord) -> list[list[int]]:
    """MOs in energy order, neighbours within DEGENERACY_EV in one group."""
    eps = rec.orbital_energies_ev
    groups: list[list[int]] = [[0]]
    for k in range(1, len(eps)):
        if eps[k] - eps[groups[-1][-1]] <= DEGENERACY_EV:
            groups[-1].append(k)
        else:
            groups.append([k])
    return groups


def open_groups(rec: QMRecord) -> list[dict]:
    """The degenerate groups holding a non-integral share of their 2d places: 0 < round(n) < 2d
    with n their summed occupation. Degeneracy is read from the energies: the SCF leaves a
    symmetric pair's occupations unequal by up to 1e-4 e (Fermi smearing amplifies its 1e-6 eV
    residue), so their spread is only recorded."""
    out = []
    for g in degenerate_groups(rec):
        occ = rec.occupations[g]
        n = float(occ.sum())
        if 0 < round(n) < 2 * len(g):
            spread = float(np.ptp(occ))
            out.append({"mos": g, "d": len(g), "n_electrons": n, "occupation_spread": spread})
    return out


def _group_entry(rec, s, reps, name: str, mos: list[int]) -> dict:
    entry = {
        "set": name,
        "mos": mos,
        "energy_ev": round(float(rec.orbital_energies_ev[mos].mean()), 6),
        "occupation": round(float(rec.occupations[mos].sum()), 6),
        "dim": len(mos),
        "irreducible": None,
        "label": None,
    }
    if s.name == "Kh":
        ell = int(np.argmax([sum(_l_share(rec, k, ell) for k in mos) for ell in range(3)]))
        entry["irreducible"], entry["label"] = len(mos) == 2 * ell + 1, "spd"[ell]
    if reps is None:
        return entry
    chars = {g: float(np.trace(m)) for g, m in block_matrices(rec, reps, mos).items()}
    entry["irreducible"], entry["label"] = label_of(s, chars)
    if s.operations:
        entry["characters"] = class_characters(s, chars)
    else:
        entry["characters"] = [{"class": g, "character": round(v, 6)} for g, v in chars.items()]
    return entry


def frontier_sets(rec: QMRecord, s: pg.Search, reps) -> list[dict]:
    """The degenerate groups holding the aufbau HOMO and LUMO, with their irreps (characters
    of the group operations on the MO coefficients). In a degenerate open shell both are the
    open group."""
    homo = (valence_electrons(rec) + 1) // 2 - 1
    groups = degenerate_groups(rec)
    out = []
    for name, k in (("HOMO", homo), ("LUMO", homo + 1)):
        g = next((g for g in groups if k in g), None)
        if g is not None and k >= 0:
            out.append(_group_entry(rec, s, reps, name, g))
    return out


def electron_total(rec: QMRecord) -> int:
    """All electrons, sum Z - charge (its parity is the valence count's)."""
    return int(rec.atomic_numbers.sum()) - int(rec.charge)


def valence_electrons(rec: QMRecord) -> int:
    """The electrons in GFN2's (valence-only) MOs: the summed occupations."""
    return round(float(rec.occupations.sum()))


def mo_atom_populations(rec: QMRecord, k: int) -> np.ndarray:
    """Mulliken gross population of MO k on every atom."""
    c = rec.C[:, k]
    return np.bincount(rec.ao_atom, weights=c * (rec.S @ c), minlength=rec.n_atoms)


# ---- rule 2: a single atom; rule 3: a degenerate open shell (Hund)


def _atom_rule(rec: QMRecord, v: TheoryVerdict) -> None:
    """Hund's rules on the atom's open subshell: the most unpaired electrons, then the largest L."""
    v.citations.append("hund")
    opens = open_groups(rec)
    if not opens:
        v.unpaired, v.term = 0, "1S"
        v.rule_chain.append("atom: closed subshells")
        return
    g = opens[0]
    weights = [sum(_l_share(rec, k, ell) for k in g["mos"]) for ell in range(3)]
    ell = int(np.argmax(weights))
    if len(opens) > 1 or g["d"] != 2 * ell + 1:
        v.rule_chain.append(f"atom: open groups {[o['d'] for o in opens]} are not one subshell")
        return
    n = round(g["n_electrons"])
    n_eff = min(n, 2 * g["d"] - n)
    big_l = sum(ell - k for k in range(n_eff))
    v.unpaired, v.term = n_eff, f"{n_eff + 1}{_LETTERS_L[big_l]}"
    v.open_shell = {"mos": g["mos"], "d": g["d"], "n_electrons": n, "label": "spd"[ell]}
    v.rule_chain.append(f"Hund (atom): {'spd'[ell]}^{n} gives {n_eff} unpaired, {v.term}")


def _l_share(rec: QMRecord, k: int, ell: int) -> float:
    c = rec.C[:, k]
    return float((c * (rec.S @ c))[rec.ao_l == ell].sum())


def _hund_rule(rec: QMRecord, s: pg.Search, reps, v: TheoryVerdict) -> bool:
    """One open group of d >= 2 equal-occupation MOs spanning one irrep: Hund's first rule."""
    opens = open_groups(rec)
    if len(opens) != 1 or opens[0]["d"] < 2:
        return False
    g = opens[0]
    if reps is None:
        v.notes.append(f"open group of {g['d']} MOs: no AO operator, no irreps")
        return False
    rho = block_matrices(rec, reps, g["mos"])
    irreducible, label = label_of(s, {k: float(np.trace(m)) for k, m in rho.items()})
    if not irreducible:
        v.notes.append(f"open group of {g['d']} MOs spans no single irrep (accidental)")
        return False
    n = round(g["n_electrons"])
    n_eff = min(n, 2 * g["d"] - n)
    term_irr, term = label_of(s, exterior_characters(rho, n_eff))
    v.unpaired, v.term = n_eff, (f"{n_eff + 1}{term}" if term_irr else None)
    v.open_shell = {"mos": g["mos"], "d": g["d"], "n_electrons": n, "label": label}
    v.citations += ["hund", "herzberg"]
    v.rule_chain += [
        f"degenerate open shell: {label}^{n} in {g['d']} MOs (point group {s.name})",
        f"Hund's first rule: {n_eff} unpaired",
    ]
    v.rule_chain.append(f"ground term {v.term}" if v.term else "ground term: reducible, not named")
    return True


# ---- rule 4: a carbene centre (Gleiter-Hoffmann)


def carbene_centres(mol: Chem.Mol) -> list[int]:
    """Neutral carbons with two bonds' worth of valence (two non-bonding electrons)."""
    return [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and a.GetFormalCharge() == 0 and a.GetTotalValence() == 2
    ]


def _lone_pairs(atom: Chem.Atom) -> int:
    """Non-bonding electron pairs of an atom as drawn (valence - charge - bonds - H) // 2."""
    outer = Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum())
    bonds = sum(round(b.GetBondTypeAsDouble()) for b in atom.GetBonds())
    return (outer - atom.GetFormalCharge() - bonds - atom.GetTotalNumHs()) // 2


def structural_carbene(atom: Chem.Atom) -> str | None:
    """Rules 4a and 4b on the centre's neighbours: the rule chain line of a singlet, or None."""
    nbrs = atom.GetNeighbors()
    if len(nbrs) == 1 and nbrs[0].GetAtomicNum() == 6:
        bond = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), nbrs[0].GetIdx())
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            return "unsaturated carbene (R2C=C:, a vinylidene): singlet"
    donors = [n for n in nbrs if n.GetAtomicNum() in PI_DONORS and _lone_pairs(n) >= 1]
    if len(nbrs) == 2 and len(donors) == 2:
        return "two pi-donor neighbours with lone pairs: singlet"
    return None


def _carbene_rule(mol: Chem.Mol, rec: QMRecord, v: TheoryVerdict) -> bool:
    centres = carbene_centres(mol)
    if not centres:
        return False
    v.citations += ["gleiter_hoffmann", "bourissou"]
    if len(centres) > 1 or valence_electrons(rec) % 2:
        v.rule_chain.append("carbene: more than one centre or an odd electron count, no rule")
        return True
    c, homo = centres[0], valence_electrons(rec) // 2 - 1
    share = [float(mo_atom_populations(rec, k)[c]) for k in (homo, homo + 1)]
    on_centre = all(int(np.argmax(mo_atom_populations(rec, k))) == c for k in (homo, homo + 1))
    gap = float(rec.orbital_energies_ev[homo + 1] - rec.orbital_energies_ev[homo])
    v.carbene = {"centre": c, "homo": homo, "gap_ev": round(gap, 4), "centre_share": share}
    if (line := structural_carbene(mol.GetAtomWithIdx(c))) is not None:
        v.unpaired = 0
        v.citations += ["stang"] if line.startswith("unsaturated") else []
        v.rule_chain.append(line)
        return True
    if not on_centre:
        v.rule_chain.append("carbene: the HOMO and LUMO are not both mostly on the centre")
        return True
    rule = f"Gleiter-Hoffmann: sigma-p_pi gap {gap:.2f} eV"
    if gap > GH_SINGLET_EV:
        v.unpaired = 0
        v.rule_chain.append(f"{rule} > {GH_SINGLET_EV} eV: singlet")
    elif gap < GH_TRIPLET_EV:
        v.unpaired = 2
        v.rule_chain.append(f"{rule} < {GH_TRIPLET_EV} eV: triplet")
    else:
        v.rule_chain.append(f"{rule} between {GH_TRIPLET_EV} and {GH_SINGLET_EV} eV: undetermined")
    return True


# ---- rule 1: transition metals (ligand field theory)


def _unit_rows(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _sorted_angles(vecs: np.ndarray) -> np.ndarray:
    u = _unit_rows(vecs)
    i, j = np.triu_indices(len(u), 1)
    return np.sort(np.degrees(np.arccos(np.clip((u[i] * u[j]).sum(1), -1.0, 1.0))))


@cache
def ideal_polyhedra() -> dict[int, dict[str, np.ndarray]]:
    """Sorted L-M-L angles of the ideal polyhedra, by coordination number."""
    t = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    o = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    tri = np.array([[math.cos(a), math.sin(a), 0.0] for a in (0, 2 * math.pi / 3, 4 * math.pi / 3)])
    h = math.sqrt(3.0) / 2  # a regular trigonal prism: triangle edges = vertical edges
    prism = np.r_[tri + [0, 0, h], tri - [0, 0, h]]
    shapes = {
        3: {"trigonal planar": tri, "trigonal pyramidal": t[:3], "T-shaped": o[[0, 1, 2]]},
        4: {"tetrahedral": t, "square planar": o[:4], "see-saw": np.r_[o[4:], tri[:2]]},
        5: {"trigonal bipyramidal": np.r_[o[4:], tri], "square pyramidal": o[:5]},
        6: {"octahedral": o, "trigonal prismatic": prism},
    }
    return {cn: {k: _sorted_angles(v) for k, v in d.items()} for cn, d in shapes.items()}


def polyhedron(rec: QMRecord, centre: int, ligands: list[int]) -> tuple[str, float | None]:
    """The nearest ideal polyhedron by the RMS (degrees) of the sorted L-M-L angles; for a
    coordination number without ideals, 'CN<n>' and None."""
    shapes = ideal_polyhedra().get(len(ligands))
    if shapes is None:
        return f"CN{len(ligands)}", None
    angles = _sorted_angles(rec.coords_ang[ligands] - rec.coords_ang[centre])
    rms = {k: float(np.sqrt(((angles - a) ** 2).mean())) for k, a in shapes.items()}
    best = min(rms, key=rms.get)
    return best, round(rms[best], 2)


def ligand_field(donors: list[int]) -> str:
    """'weak' (every donor a halide, below water in Tsuchida's series), 'strong' (every donor
    C or P, its strong-field end) or 'middle or mixed'."""
    if donors and all(z in WEAK_DONORS for z in donors):
        return "weak"
    if donors and all(z in STRONG_DONORS for z in donors):
        return "strong"
    return "middle or mixed"


def _spin_high(d: int) -> int:
    return d if d <= 5 else 10 - d


def _spin_low_oh(d: int) -> int:
    return d if d <= 3 else 6 - d if d <= 6 else d - 6


def _oh_rule(z: int, d: int, field: str) -> tuple[int | None, str]:
    if d in (2, 3, 8):
        return _spin_high(d), f"Oh d{d}: one outcome"
    if z not in domain.FIRST_ROW_TM:
        return _spin_low_oh(d), f"Oh d{d}, 4d/5d metal (larger Delta_o): low spin"
    if field == "weak":
        return _spin_high(d), f"Oh d{d}, weak field (halide donors): high spin"
    if field == "strong":
        return _spin_low_oh(d), f"Oh d{d}, strong field (C/P donors): low spin"
    return None, f"Oh d{d}, {field} field: undetermined"


def centre_rule(z: int, d: int | None, poly: str, field: str) -> tuple[int | None, str]:
    """(unpaired electrons of one metal centre or None, the rule that gave it)."""
    if d is None or not 0 <= d <= 10:
        return None, f"d count {d} outside 0..10: no rule"
    if d in (0, 10):
        return 0, f"d{d}: closed d shell, by count"
    if d in (1, 9):
        return 1, f"d{d}: one {'electron' if d == 1 else 'hole'}, by count, any geometry"
    if poly == "tetrahedral":
        if z in domain.FIRST_ROW_TM:
            return _spin_high(d), f"Td high spin (Delta_t ~ 4/9 Delta_o): d{d}"
        return None, f"Td d{d} beyond the first row: no rule"
    if poly == "square planar":
        if d == 8:
            return 0, "square-planar d8: low spin"
        return None, f"square-planar d{d}: no rule"
    if poly == "octahedral":
        return _oh_rule(z, d, field)
    return None, f"{poly} d{d}: no ligand-field rule"


def _jahn_teller(poly: str, d: int | None, unpaired: int | None) -> bool:
    """An Oh configuration with an odd number of e_g electrons (high-spin d4, low-spin d7, d9)."""
    if poly != "octahedral" or d is None:
        return False
    return d == 9 or (d == 4 and unpaired == 4) or (d == 7 and unpaired == 1)


def metal_centre(mol: Chem.Mol, rec: QMRecord, idx: int) -> dict:
    atom = mol.GetAtomWithIdx(idx)
    ligands = [n.GetIdx() for n in atom.GetNeighbors()]
    poly, rms = polyhedron(rec, idx, ligands)
    donors = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in ligands]
    d = domain.d_count(atom)
    field = ligand_field(donors)
    unpaired, rule = centre_rule(atom.GetAtomicNum(), d, poly, field)
    return {
        "atom": idx,
        "element": atom.GetSymbol(),
        "oxidation_state": int(domain.oxidation_state(atom)),
        "d": None if d is None else int(d),
        "coordination_number": len(ligands),
        "polyhedron": poly,
        "polyhedron_rms_deg": rms,
        "donors": sorted(Chem.GetPeriodicTable().GetElementSymbol(z) for z in donors),
        "field": field,
        "unpaired": unpaired,
        "rule": rule,
        "jahn_teller": _jahn_teller(poly, d, unpaired),
    }


def _metal_rule(mol: Chem.Mol, rec: QMRecord, v: TheoryVerdict) -> bool:
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in domain.GROUP]
    if not metals:
        return False
    v.citations += ["ligand_field", "tsuchida"]
    v.metals = [metal_centre(mol, rec, i) for i in metals]
    for c in v.metals:
        head = f"{c['element']}({c['oxidation_state']:+d}) d{c['d']} {c['polyhedron']}"
        v.rule_chain.append(f"{head}: {c['rule']}")
        if c["jahn_teller"]:
            v.citations.append("jahn_teller")
            v.notes.append(f"{c['element']} atom {c['atom']}: Jahn-Teller active, annotated only")
    ligand_radicals = sum(
        a.GetNumRadicalElectrons() for a in mol.GetAtoms() if a.GetIdx() not in metals
    )
    open_centres = [c for c in v.metals if c["unpaired"] != 0]
    if ligand_radicals:
        v.rule_chain.append("drawn radicals beside the metal: no rule")
    elif len(open_centres) > 1:
        v.rule_chain.append("two open-shell metal centres: their coupling is not ligand-field")
    elif open_centres and open_centres[0]["unpaired"] is None:
        v.rule_chain.append("metal centre undetermined")
    else:
        total = open_centres[0]["unpaired"] if open_centres else 0
        if total % 2 == electron_total(rec) % 2:
            v.unpaired = total
        else:
            v.rule_chain.append(f"{total} unpaired contradicts the electron count's parity")
    return True


# ---- the verdict and the table


def _check_order(mol: Chem.Mol, rec: QMRecord) -> None:
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if z != [int(x) for x in rec.atomic_numbers]:
        raise ValueError(f"mol and record of {rec.smiles!r} differ in atom order or elements")


def theory_verdict(mol: Chem.Mol, rec: QMRecord) -> TheoryVerdict:
    """The predicted lowest spin state of one species by the rules of the module docstring."""
    _check_order(mol, rec)
    s = pg.search(rec.atomic_numbers, rec.coords_ang, SYMMETRY_TOLERANCE_ANG)
    n_e = electron_total(rec)
    v = TheoryVerdict(rec.smiles, None, [], [], s.name, None, n_e, int(rec.unpaired))
    reps = representation(rec, s)
    if reps is None and s.name != "Kh":
        v.notes.append("an AO operator does not leave S invariant: no irreps read")
    v.frontier = frontier_sets(rec, s, reps)
    if not _metal_rule(mol, rec, v):
        if rec.unpaired != n_e % 2:
            v.rule_chain.append(f"record at u = {rec.unpaired}, not the lowest spin: no MO rule")
        elif rec.n_atoms == 1:
            _atom_rule(rec, v)
        elif not (_hund_rule(rec, s, reps, v) or _carbene_rule(mol, rec, v)):
            v.rule_chain.append("no rule applies")
    if v.unpaired is None:
        v.rule_chain.append("undetermined")
    v.citations = sorted(set(v.citations))
    return v


def theory_digest(table: dict) -> str:
    """sha256 of the canonical JSON of the table without its digest (version, constants,
    citations and entries)."""
    body = {k: val for k, val in table.items() if k != "digest"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def theory_table(entries: dict[str, TheoryVerdict | dict]) -> dict:
    """The theory table (`VERSION`): the verdicts by canonical SMILES, the rule constants and
    citations."""
    rows = {
        smi: e.as_dict() if isinstance(e, TheoryVerdict) else e
        for smi, e in sorted(entries.items())
    }
    constants = {
        "symmetry_tolerance_ang": SYMMETRY_TOLERANCE_ANG,
        "degeneracy_ev": DEGENERACY_EV,
        "operator_tol": OPERATOR_TOL,
        "gleiter_hoffmann_ev": [GH_TRIPLET_EV, GH_SINGLET_EV],
        "pi_donors": sorted(PI_DONORS),
        "weak_donors": sorted(WEAK_DONORS),
        "strong_donors": sorted(STRONG_DONORS),
    }
    table = {"version": VERSION, "constants": constants, "citations": CITATIONS, "entries": rows}
    return table | {"digest": theory_digest(table)}


def write_theory_table(entries: dict, path: Path = THEORY_TABLE_PATH) -> Path:
    table = theory_table(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def load_theory_table(path: Path = THEORY_TABLE_PATH) -> dict:
    """The table, refused when its version or digest does not match."""
    table = json.loads(path.read_text())
    if table.get("version") != VERSION or table.get("digest") != theory_digest(table):
        raise ValueError(f"{path}: not a valid {VERSION} table (version or digest)")
    return table
