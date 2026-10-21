"""First-arrow ranking: candidates, labels, untrained Klopman-Salem scorers, failure classes.

Candidates of a step: every (occupied, virtual) token pair of its reactant side that passes
`MechanismSample.pair_valid` (doubly occupied donor, empty acceptor, no d polarization function,
within a fragment not the donor's own antibond nor a pair whose every site pair is a bond
interacting with itself; read from descriptors and the record geometry). Target: the
first labeled arrow of the step (smallest `arrow_index`), whose labeled pairs are its donor set x
acceptor set. Metric: top-k accuracy under multi-label marginalization: a hit when any labeled
pair ranks among the k best candidates. Ties count against the label (the rank is the number of
other candidates scoring at least as well as the best labeled pair). A step whose labeled pairs
are all filtered out is a miss, counted as `n_no_valid_label`.

Untrained scorers work on per-sample component matrices (`pair_components`, physical K = 1.75,
eps = 4; energies in eV, more negative = more stabilizing). With beta the kernel's resonance
integral (AO-level for the oriented kernel), F = (E_r + E_s) / 2 the mean of the two LMO Fock
diagonals and S the contact amplitude:

- `ks3+ks2` = -(KS3 + KS2): **the untrained physics floor** (oriented kernel, d = 2.3);
- `frozen_head` = -(KS3 + KS2 + KS1) / tau: the model's frozen KS head untrained (tau = 1 eV;
  KS1 >= 0 the closed-shell repulsion), the floor with the repulsion subtracted;
- `ks3` = -KS3 = -beta^2 G, the full product alone;
- `fbar2_g` = -F^2 G, the orbital energies without the amplitude;
- `fbar2_g+ks2` = -(F^2 G + KS2), the same with the Coulomb term between the main centres;
- diagnostics: `amp2` = S^2 alone, `amp2_g` = -S^2 G, `gap_kernel` = -G, `ks2` = -KS2.
"""

from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from collections.abc import Callable

import numpy as np

from ksmech.arrows.align import ArrowLabel
from ksmech.arrows.reactants import TokenLoader
from ksmech.arrows.steps import MechanismSample
from ksmech.orbitals.tokens import MoleculeTokens, OrbitalToken
from ksmech.physics.contact_kernel import ContactKernel
from ksmech.physics.pairs import PAIR_FEATURES, valid_pairs
from ksmech.physics.params import KSParams, to_params

TOP_K = (1, 3, 5)
_F = {name: k for k, name in enumerate(PAIR_FEATURES)}

Components = dict[str, np.ndarray]  # name -> (n_occ, n_virt) physical matrix
Scorer = Callable[[Components], np.ndarray]

SCORERS: dict[str, Scorer] = {
    "ks3+ks2": lambda c: -(c["ks3"] + c["ks2"]),
    "frozen_head": lambda c: -(c["ks3"] + c["ks2"] + c["ks1"]) / KSParams().tau,
    "ks3": lambda c: -c["ks3"],
    "fbar2_g": lambda c: -(c["fbar"] ** 2) * c["gap_kernel"],
    "fbar2_g+ks2": lambda c: -((c["fbar"] ** 2) * c["gap_kernel"] + c["ks2"]),
}
DIAGNOSTIC_SCORERS: dict[str, Scorer] = {
    "amp2": lambda c: c["amp"] ** 2,
    "amp2_g": lambda c: -(c["amp"] ** 2) * c["gap_kernel"],
    "gap_kernel": lambda c: -c["gap_kernel"],
    "ks2": lambda c: -c["ks2"],
}
ALL_SCORERS = {**SCORERS, **DIAGNOSTIC_SCORERS}


def physical_pair_z(sample: MechanismSample, params: KSParams | None = None) -> np.ndarray:
    """The stored z (computed at K = eps = 1) at `params` (default: the physical values);
    undefined entries stay 0."""
    z = to_params(sample.pair_z, params or KSParams(), PAIR_FEATURES)
    return np.where(sample.pair_defined, z, 0.0)


def _by_fragment(tokens: list[MoleculeTokens], kind: str) -> list[np.ndarray]:
    sizes = [len(t.occupied() if kind == "occ" else t.virtual()) for t in tokens]
    starts = np.cumsum([0, *sizes])
    return [np.arange(starts[f], starts[f + 1]) for f in range(len(tokens))]


def pair_components(
    sample: MechanismSample, loader: TokenLoader, kernel: ContactKernel, params=None
) -> Components:
    """Physical (n_occ, n_virt) KS3 (= beta^2 G with `kernel`), KS2, KS1 (stored, >= 0), G, F,
    the amplitude, and the fragment flag and (oriented kernel) the CLASH flag of the
    amplitude."""
    params = params or KSParams()
    tokens = sample.tokens(loader)
    z = physical_pair_z(sample, params)
    occ_rows, virt_cols = _by_fragment(tokens, "occ"), _by_fragment(tokens, "virt")
    amp = np.zeros(z.shape[:2])
    beta = np.zeros(z.shape[:2])
    clash = np.zeros(z.shape[:2], dtype=bool)
    for f, t_f in enumerate(tokens):
        for g, t_g in enumerate(tokens):
            cell = np.ix_(occ_rows[f], virt_cols[g])
            occ, virt, same = t_f.occupied(), t_g.virtual(), f == g
            amp[cell] = kernel.amplitude_matrix(occ, t_f, virt, t_g, same)
            beta[cell] = kernel.beta_matrix(occ, t_f, virt, t_g, params, same)
            if kernel.config.kernel == "oriented":
                clash[cell] = kernel.oriented_details(occ, t_f, virt, t_g, same)[1]
    e_occ = np.array([o.energy_ev for t in tokens for o in t.occupied()])
    e_virt = np.array([o.energy_ev for t in tokens for o in t.virtual()])
    G = sample.pair_gap_sign * np.exp(z[..., _F["log_abs_g"]])
    return {
        "ks3": beta**2 * G,
        "ks2": z[..., _F["ks2"]],
        "ks2_ok": sample.pair_defined[..., _F["ks2"]],
        "log_abs_ks1": z[..., _F["log_abs_ks1"]],
        "ks1_ok": sample.pair_defined[..., _F["log_abs_ks1"]],
        "ks1": np.where(
            sample.pair_defined[..., _F["log_abs_ks1"]], np.exp(z[..., _F["log_abs_ks1"]]), 0.0
        ),
        "gap_kernel": G,
        "fbar": 0.5 * (e_occ[:, None] + e_virt[None, :]),
        "amp": amp,
        "inter": z[..., _F["inter"]] > 0.5,
        "clash": clash,
    }


def refresh_validity(sample: MechanismSample, loader: TokenLoader) -> MechanismSample:
    """The sample with `pair_valid` recomputed from its tokens (the current filter), never
    widened: the store keeps pair features for the stored valid cells only."""
    now = valid_pairs(sample.tokens(loader))
    return dataclasses.replace(sample, pair_valid=sample.pair_valid & now)


def first_label(sample: MechanismSample) -> ArrowLabel:
    return min(sample.labels, key=lambda lab: lab.arrow_index)


def orbital_positions(sample: MechanismSample) -> tuple[np.ndarray, np.ndarray]:
    """Sample orbital index -> row (occupied) or column (virtual) of the pair block, -1 else."""
    occ = sample.orbital_occupied
    row = np.full(sample.n_orbitals, -1)
    col = np.full(sample.n_orbitals, -1)
    row[occ] = np.arange(occ.sum())
    col[~occ] = np.arange((~occ).sum())
    return row, col


def label_pairs(sample: MechanismSample, label: ArrowLabel) -> np.ndarray:
    """(n_occ, n_virt) True on the labeled pairs (before the validity filter)."""
    row, col = orbital_positions(sample)
    out = np.zeros(sample.pair_valid.shape, dtype=bool)
    rows = [row[t - sample.n_atoms] for t in label.donor_ids if t >= sample.n_atoms]
    cols = [col[t - sample.n_atoms] for t in label.acceptor_ids if t >= sample.n_atoms]
    rows, cols = [r for r in rows if r >= 0], [c for c in cols if c >= 0]
    if rows and cols:
        out[np.ix_(rows, cols)] = True
    return out


def label_rank(scores: np.ndarray, valid: np.ndarray, positive: np.ndarray) -> int | None:
    """Number of valid non-labeled candidates scoring at least as well as the best valid
    labeled pair (0 = top-1); None when no labeled pair is valid."""
    pos = positive & valid
    if not pos.any():
        return None
    best = scores[pos].max()
    return int(((scores >= best) & valid & ~pos).sum())


def summarize_ranks(ranks: list[int | None], ks=TOP_K) -> dict:
    """Top-k accuracy (a missing label is a miss), median rank of the valid ones, counts."""
    valid = np.array([r for r in ranks if r is not None])
    nan = float("nan")
    out = {
        f"top{k}": float(np.mean([r is not None and r < k for r in ranks])) if ranks else nan
        for k in ks
    }
    out["median_rank"] = float(np.median(valid)) if len(valid) else float("nan")
    out["n"] = len(ranks)
    out["n_no_valid_label"] = sum(r is None for r in ranks)
    return out


def score_ranks(samples: list[MechanismSample], scores: list[np.ndarray]) -> list[int | None]:
    """Rank of the first labeled pair of each sample under precomputed score matrices."""
    return [
        label_rank(sc, s.pair_valid, label_pairs(s, first_label(s)))
        for s, sc in zip(samples, scores, strict=True)
    ]


def evaluate(samples, components: list[Components], scorers: dict[str, Scorer]) -> dict:
    """{scorer: {"ranks": [...], **summary}} over `samples`."""
    out = {}
    for name, scorer in scorers.items():
        ranks = score_ranks(samples, [scorer(c) for c in components])
        out[name] = {"ranks": ranks, **summarize_ranks(ranks)}
    return out


# ---- failure classes (diagnostic: classifier labels describe what was ranked, never input)


def token_class(o: OrbitalToken, t: MoleculeTokens) -> str:
    """Diagnostic description: classifier type and the elements of the two leading centres."""
    centres = [a for a in o.descriptors.center_atoms[:2] if a >= 0]
    return f"{o.type}({'-'.join(t.symbol(int(a)) for a in centres)})"


def pair_class(tokens: list[MoleculeTokens], i: int, j: int) -> dict:
    """Diagnostic and descriptor facts of pair (occupied row i, virtual column j)."""
    occ = [(t, o) for t in tokens for o in t.occupied()]
    virt = [(t, o) for t in tokens for o in t.virtual()]
    (t_d, d), (t_a, a) = occ[i], virt[j]
    z_a = t_a.atoms[int(a.descriptors.center_atoms[0])].atomic_number
    return {
        "class": f"{token_class(d, t_d)} -> {token_class(a, t_a)}",
        "acceptor_on_h": bool(z_a == 1),
        "donor_s_rich": bool(d.descriptors.center_spd[0][0] > 0.5),
        "intermolecular": t_d is not t_a,
        "donor_energy_ev": float(d.energy_ev),
    }


def _site_weight(sample: MechanismSample, token: int, site: int) -> float:
    rows = sample.incidence[sample.incidence[:, 0] == token - sample.n_atoms]
    return float(rows[rows[:, 1] == site, 2].sum())


def teacher_cell(sample: MechanismSample) -> tuple[int, int] | None:
    """The first arrow's representative pair, independent of any scorer: the donor-set and
    acceptor-set members with the largest weight on their sites (the alignment's teacher
    representatives), when that pair is valid; else the first valid labeled pair."""
    lab = first_label(sample)
    row, col = orbital_positions(sample)
    if lab.teacher_donor >= 0 and lab.teacher_acceptor >= 0:  # chosen among valid cells
        n = sample.n_atoms
        return int(row[lab.teacher_donor - n]), int(col[lab.teacher_acceptor - n])
    donors = [t for t in lab.donor_ids if t >= sample.n_atoms]
    acceptors = [t for t in lab.acceptor_ids if t >= sample.n_atoms]
    if not donors or not acceptors:
        return None
    d = max(sorted(donors), key=lambda t: _site_weight(sample, t, lab.donor_site))
    a = max(sorted(acceptors), key=lambda t: _site_weight(sample, t, lab.acceptor_site))
    cell = (int(row[d - sample.n_atoms]), int(col[a - sample.n_atoms]))
    if sample.pair_valid[cell]:
        return cell
    valid = label_pairs(sample, lab) & sample.pair_valid
    return tuple(int(x) for x in np.argwhere(valid)[0]) if valid.any() else None


def describe_sample(sample, tokens, scores: np.ndarray, G: np.ndarray, rank, k: int) -> dict:
    """Label class (its teacher pair), top-ranked class and hit flag under one score matrix."""
    valid_scores = np.where(sample.pair_valid, scores, -np.inf)
    top = np.unravel_index(np.argmax(valid_scores), scores.shape)
    cell = teacher_cell(sample)
    label = pair_class(tokens, *cell) if cell else {"class": "no valid labeled pair"}
    return {
        "step_id": sample.step_id,
        "rank": rank,
        "hit": rank is not None and rank < k,
        "label": label,
        "top": pair_class(tokens, int(top[0]), int(top[1])),
        "label_G": float(G[cell]) if cell else None,
    }


def failure_classes(samples, components, ranks, loader: TokenLoader, scorer, k: int = 5) -> dict:
    """Group the misses of one scorer: by the labeled pair's class, by the class of the pair
    ranked first instead, and by descriptor facts of both."""
    rows = [
        describe_sample(s, s.tokens(loader), scorer(c), c["gap_kernel"], r, k)
        for s, c, r in zip(samples, components, ranks, strict=True)
    ]
    misses = [r for r in rows if not r["hit"]]
    by_label: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]["class"]].append(r["hit"])
    return {
        "n": len(rows),
        "n_miss": len(misses),
        "label_class_hit_rate": {
            c: {"n": len(h), "hit_rate": float(np.mean(h))}
            for c, h in sorted(by_label.items(), key=lambda x: -len(x[1]))
        },
        "miss_label_classes": Counter(r["label"]["class"] for r in misses).most_common(15),
        "miss_top_classes": Counter(r["top"]["class"] for r in misses).most_common(15),
        "miss_facts": _facts(misses),
        "rows": rows,
    }


def class_table(rows: list[dict], ranks_by_scorer: dict[str, list], k: int = 5) -> dict:
    """Top-k hit rate per labeled-pair class for several scorers (classes from `rows`)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["label"]["class"]].append(i)
    return {
        cls: {"n": len(idx)}
        | {
            name: float(np.mean([ranks[i] is not None and ranks[i] < k for i in idx]))
            for name, ranks in ranks_by_scorer.items()
        }
        for cls, idx in sorted(groups.items(), key=lambda x: -len(x[1]))
    }


def _facts(misses: list[dict]) -> dict[str, float]:
    if not misses:
        return {}

    def share(get) -> float:
        return float(np.mean([bool(get(r)) for r in misses]))

    return {
        "top_acceptor_on_h": share(lambda r: r["top"]["acceptor_on_h"]),
        "top_donor_s_rich": share(lambda r: r["top"]["donor_s_rich"]),
        "top_donor_below_minus_15_ev": share(lambda r: r["top"]["donor_energy_ev"] < -15),
        "label_acceptor_on_h": share(lambda r: r["label"].get("acceptor_on_h")),
        "label_intramolecular": share(lambda r: r["label"].get("intermolecular") is False),
        "label_G_positive": share(lambda r: (r["label_G"] or 0.0) > 0),
    }
