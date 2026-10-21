"""Textbook selectivity checks: hand-built reactant sets and the first arrow chemists expect.

Each case is an atom-mapped reactant SMILES and an expected pair (donor site atom, acceptor site
atom), given by map numbers; some cases also name a competing pair (E2 next to SN2, 1,2 next to
1,4 addition) whose score is printed beside it. Every valid (occupied, virtual) pair of the
reactant set, within and across the molecules, is scored. The site of a token is its main
centre (largest population): sigma*(C-Cl) sits on C, sigma*(C-H) on H, a carbonyl pi* on C. A
predicted pair matches an expectation when its two sites are the expected atoms up to graph
symmetry (any of the three hydronium protons). A case passes when its expected pair is among
the top 3.

Molecules without a token file get a GFN2-xTB record (`compute_fixture_records`, the same
method and cache as the dataset) and tokens, both written to the caches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

from ksmech.arrows.reactants import ReactantTokens, TokenCache
from ksmech.arrows.steps import MechanismSample, reactant_sample
from ksmech.data import chem
from ksmech.data import provenance as prov
from ksmech.data.provenance import StaleArtifact
from ksmech.orbitals.build import tokens_path
from ksmech.orbitals.tokens import build_molecule_tokens, load_tokens, save_tokens
from ksmech.physics.ranking import token_class
from ksmech.qm.embed import prepare_mol
from ksmech.qm.records import cache_path
from ksmech.qm.xtb_calc import QMConfig


@dataclass(frozen=True)
class Expectation:
    label: str
    donor: int  # map number of the donor site atom
    acceptor: int  # map number of the acceptor site atom


@dataclass(frozen=True)
class TextbookCase:
    name: str
    smiles: str
    expected: Expectation
    alternatives: tuple[Expectation, ...] = ()


CASES = (
    TextbookCase("MeO- + MeCl", "C[O-:1].[CH3:2][Cl:3]", Expectation("SN2: O -> C", 1, 2)),
    TextbookCase(
        "MeO- + EtBr",
        "C[O-:1].[H:5][CH2:4][CH2:2][Br:3]",
        Expectation("SN2: O -> C-Br carbon", 1, 2),
        (Expectation("E2: O -> beta H", 1, 5),),
    ),
    TextbookCase(
        "CN- + acrolein",
        "N#[C-:1].[CH2:2]=[CH:3][CH:4]=[O:5]",
        Expectation("1,4: C -> beta carbon", 1, 2),
        (Expectation("1,2: C -> carbonyl carbon", 1, 4),),
    ),
    TextbookCase("MeNH2 + AcCl", "C[NH2:1].C[C:2](=[O:3])Cl", Expectation("N -> carbonyl C", 1, 2)),
    TextbookCase(
        "H3O+ + acetone", "CC(C)=[O:1].[H:3][OH2+:2]", Expectation("acetone O -> H3O+ H", 1, 3)
    ),
)


def fragment_keys(smiles: str) -> list[str]:
    mol = chem.parse_mapped(smiles)
    return [chem.unmapped_canonical(f) for f in Chem.GetMolFrags(mol, asMols=True)]


def ensure_tokens(keys: list[str], tokens_dir: Path, qm: QMConfig | None = None) -> list[str]:
    """Build the token files that are missing (QM record from the cache, else computed).
    Returns the keys that were built."""
    from ksmech.qm.validation import compute_fixture_records

    qm = qm or QMConfig()
    built = []
    for key in keys:
        path = tokens_path(tokens_dir, key, 0, qm.method)
        if path.exists():
            try:
                load_tokens(path)
                continue
            except StaleArtifact:
                pass  # rebuilt below
        record = compute_fixture_records({key: key}, qm)[key]
        qm_path = cache_path(qm.cache_dir, key, 0, qm.method)
        save_tokens(build_molecule_tokens(record), path, prov.file_input(qm_path))
        built.append(key)
    return built


def symmetry_classes(rt: ReactantTokens) -> np.ndarray:
    """(n_atoms,) class id per global atom: same fragment and same canonical rank, no ties
    broken, so symmetry-equivalent atoms share a class."""
    out = np.empty(rt.n_atoms, dtype=int)
    for f, frag in enumerate(rt.fragments):
        ranks = Chem.CanonicalRankAtoms(prepare_mol(frag.key), breakTies=False)
        off = int(rt.atom_offsets[f])
        out[off : off + len(ranks)] = [f * 100000 + r for r in ranks]
    return out


@dataclass
class CaseResult:
    case: TextbookCase
    top: list[dict]  # the 3 best pairs: donor, site, acceptor, site, score
    expected_rank: int | None
    expected_score: float
    alternatives: list[tuple[Expectation, int | None, float]]

    @property
    def passed(self) -> bool:
        return self.expected_rank is not None and self.expected_rank < 3


def _match_rank(scores, valid, sites, classes, exp, site_of) -> tuple[int | None, float]:
    """Best rank (0-based, ties against) and score of the pairs matching an expectation."""
    d, a = classes[site_of(exp.donor)], classes[site_of(exp.acceptor)]
    match = valid & (classes[sites[0]][:, None] == d) & (classes[sites[1]][None, :] == a)
    if not match.any():
        return None, float("nan")
    best = scores[match].max()
    return int(((scores >= best) & valid & ~match).sum()), float(best)


def run_case(case: TextbookCase, sample: MechanismSample, rt: ReactantTokens, scores) -> CaseResult:
    from ksmech.physics.baseline import main_centres

    main = main_centres(sample)
    occ_idx = np.flatnonzero(sample.orbital_occupied)
    virt_idx = np.flatnonzero(~sample.orbital_occupied)
    sites = (main[occ_idx], main[virt_idx])
    classes = symmetry_classes(rt)
    valid = sample.pair_valid
    ranked = np.argsort(np.where(valid, scores, -np.inf), axis=None)[::-1][:3]
    top = []
    for flat in ranked:
        i, j = np.unravel_index(flat, scores.shape)
        r, s = rt.orbitals[occ_idx[i]], rt.orbitals[virt_idx[j]]
        t_r = rt.fragments[rt.orbital_fragment[occ_idx[i]]].tokens
        t_s = rt.fragments[rt.orbital_fragment[virt_idx[j]]].tokens
        top.append(
            {
                "donor": token_class(r, t_r),
                "donor_site": _atom_label(rt, sites[0][i]),
                "acceptor": token_class(s, t_s),
                "acceptor_site": _atom_label(rt, sites[1][j]),
                "score": float(scores[i, j]),
            }
        )
    site_of = rt.site_of_map
    rank, score = _match_rank(scores, valid, sites, classes, case.expected, site_of)
    alts = [
        (alt, *_match_rank(scores, valid, sites, classes, alt, site_of))
        for alt in case.alternatives
    ]
    return CaseResult(case, top, rank, score, alts)


def _atom_label(rt: ReactantTokens, g: int) -> str:
    m = rt.mol_atom.get(int(g))
    mapnum = rt.mol.GetAtomWithIdx(m).GetAtomMapNum() if m is not None else 0
    f = int(rt.atom_fragment[g])
    t = rt.fragments[f].tokens
    symbol = t.symbol(int(g) - int(rt.atom_offsets[f]))
    return f"{symbol}{g}" + (f"(:{mapnum})" if mapnum else "")


def case_sample(case: TextbookCase, loader, kernels) -> tuple[MechanismSample, ReactantTokens]:
    rt = ReactantTokens.from_smiles(case.smiles, loader)
    return reactant_sample(f"textbook:{case.name}", "textbook", rt, kernels), rt


def run_suite(scorer, loader: TokenCache, kernels) -> list[CaseResult]:
    """`scorer(sample, loader) -> (n_occ, n_virt) scores` for every case (`kernels`: the
    `physics.pairs.PairKernels` the samples' pair features are computed with)."""
    out = []
    for case in CASES:
        sample, rt = case_sample(case, loader, kernels)
        out.append(run_case(case, sample, rt, scorer(sample, loader)))
    return out
