"""Full-step decoding of stored samples (spec v9, "Evaluation protocol"): the beam decoder
(`model.decode`) on each sample of a part, B + R checked on the sample's own Lewis structure
(`lewis_apply`), and a hit at k when one of the k best completions is a variant of the label
class (`EditLabel.variants`, on the global atoms, as the completions are).

`lewis_apply` tracks the non-bonding electrons N of every atom (open-shell plan P6: an atom
entry of +-1 moves one electron) and refuses a completion with the first of: BOND_ORDER (an
order leaves [0, 3]), LONE_PAIRS (a negative N), OCTET_EXCEEDED (an H-Ne atom the step touches
over 8 valence electrons, N + 2 x its bond orders; H and He over 2), FORMAL_CHARGE_RANGE (a
non-metal it touches left at |formal charge| >= 2 that it did not carry), PRODUCT_INVALID
(RDKit cannot write the product), RESONANCE_ONLY_STEP (the identity keys with the atom maps
agree: no atom changes its sigma bonds or H, and no fragment its charge or spin). The product
key is the pool key (`data.resonance.pool_key`: the map-free identity key v2 of every fragment,
u from the tracked N), so completions that draw the same products differently merge, as serve's
pool key merges them, while homolysis and heterolysis of one bond, and the two heterolyses
(CH3+ + Br- against CH3- + Br+), stay apart. Decode version 4 keyed by the identity key of the
whole pool, whose charges are zeroed over every fragment, and merged the two heterolyses.

The caps of the search (`decode_caps`): `legacy` keeps DecodeConfig's fixed 8 bond and 2 atom
changes; `labels` takes the label maxima of the training part of the source decoded (the
largest bond and atom counts of a label's smallest variant, `delta_be.label_counts`, written
by the sample builder into each part's report), never of val or test. max_candidates stays 48
unless overridden (spec v9.1, "fixed decoder caps" against caps per source).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

from ksmech.arrows.delta_be import EditLabel
from ksmech.arrows.lewis import MAX_ORDER
from ksmech.arrows.symmetry import lewis_mol
from ksmech.data import provenance as prov
from ksmech.data.resonance import identity_key, pool_key
from ksmech.model.decode import (
    NONMETALS,
    SCREEN_CODES,
    Apply,
    Completion,
    DecodeConfig,
    decode_pool,
)

Decode = Callable[[object, Counter], list[Completion]]
CAPS = ("legacy", "labels")
CAP_FIELDS = ("max_bonds", "max_atoms", "max_candidates")


def caps_from_report(report: Mapping, where: str = "the report") -> dict[str, int]:
    """max_bonds and max_atoms from a part's report (`arrows.steps.split_report`): its label
    maxima. ValueError for a report written before them (a v17 store) or with no label."""
    maxima = report.get("label_maxima")
    if not maxima:
        raise ValueError(f"{where} has no label maxima: rebuild the samples (ksmech-arrows)")
    caps = {"max_bonds": int(maxima["bonds"]), "max_atoms": int(maxima["atoms"])}
    if caps["max_bonds"] < 1:
        raise ValueError(f"{where}: no labelled sample (label maxima {dict(maxima)})")
    return caps


def training_caps(view: Path, sample_version: int) -> dict[str, int]:
    """The label maxima of a training view (`samples_<split>_train.json`), verified as every
    view is read: kind, sample version, inputs and a clean build."""
    prov.verify_clean(view, "samples", sample_version)
    return caps_from_report(json.loads(Path(view).read_text())["report"], str(view))


def decode_caps(
    mode: str, train_view: Path, sample_version: int, overrides: Mapping[str, int | None]
) -> dict[str, int]:
    """max_bonds, max_atoms and max_candidates of a decode: DecodeConfig's defaults (legacy)
    or the training part's label maxima (labels), then every override that is given."""
    if mode not in CAPS:
        raise ValueError(f"caps {mode!r}: one of {CAPS}")
    legacy = DecodeConfig()
    caps = {name: getattr(legacy, name) for name in CAP_FIELDS}
    if mode == "labels":
        caps |= training_caps(train_view, sample_version)
    caps |= {k: int(v) for k, v in overrides.items() if k in CAP_FIELDS and v is not None}
    if min(caps.values()) < 0 or caps["max_bonds"] < 1:
        raise ValueError(f"caps {caps}: at least the anchor, no negative count")
    return caps


def _valence_limit(z: int) -> int | None:
    """Valence electrons an atom may carry: 2 for H and He, 8 for Li-Ne, None beyond."""
    return 2 if z <= 2 else 8 if z <= 10 else None


def _pool_mol(z: np.ndarray, q: np.ndarray, order: dict[tuple[int, int], int], n) -> Chem.Mol:
    """The pool on the token atoms with its radical electrons from N, atom k mapped k + 1."""
    rows = [(i, j, o) for (i, j), o in sorted(order.items()) if o > 0]
    mol = lewis_mol(z, q, np.asarray(rows, dtype=np.int64).reshape(-1, 3), n)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return mol


def _atom_issue(z, q0, q, nonbonding, order, touched) -> str | None:
    """OCTET_EXCEEDED or FORMAL_CHARGE_RANGE on the touched atoms (module docstring), or None."""
    electrons = nonbonding.astype(np.int64)
    for (i, j), o in order.items():
        electrons[i] += 2 * o
        electrons[j] += 2 * o
    for i in sorted(touched):
        limit = _valence_limit(int(z[i]))
        if limit is not None and electrons[i] > limit:
            return "OCTET_EXCEEDED"
    for i in sorted(touched):
        if int(z[i]) in NONMETALS and abs(q[i]) >= 2 and abs(q0[i]) < 2:
            return "FORMAL_CHARGE_RANGE"
    return None


def lewis_apply(sample) -> Apply:
    """B + R of a completion on the sample's Lewis structure: (product key, None) or (None, the
    first issue code) (module docstring)."""
    z = sample.atom_categorical[:, 0].astype(np.int64)
    q0 = sample.atom_categorical[:, 1].astype(np.int64)
    order0 = {(int(i), int(j)): int(o) for i, j, o in sample.bonds}
    n0 = np.asarray(sample.nonbonding, dtype=np.int64)
    if len(n0) != len(z):
        who = getattr(sample, "step_id", "a sample")
        raise ValueError(f"{who}: no N for every atom (a sample older than v19?)")
    with rdBase.BlockLogs():
        before = identity_key(_pool_mol(z, q0, order0, n0), keep_maps=True, nonbonding=n0)

    def apply(bonds, lone):
        order, n, q = dict(order0), n0.copy(), q0.copy()
        for i, j, d in bonds:
            o = order.get((i, j), 0) + d
            if not 0 <= o <= MAX_ORDER:
                return None, "BOND_ORDER"
            order[(i, j)] = o
            q[i] -= d
            q[j] -= d
        for i, d in lone:
            n[i] += d
            q[i] -= d
        if (n < 0).any():
            return None, "LONE_PAIRS"
        touched = {a for i, j, _ in bonds for a in (i, j)} | {i for i, _ in lone}
        why = _atom_issue(z, q0, q, n, order, touched)
        if why is not None:
            return None, why
        try:
            with rdBase.BlockLogs():
                mol = _pool_mol(z, q, order, n)
                mapped = identity_key(mol, keep_maps=True, nonbonding=n)
                key = pool_key(mol, nonbonding=n)
        except Exception:  # noqa: BLE001 - counted as PRODUCT_INVALID
            return None, "PRODUCT_INVALID"
        return (None, "RESONANCE_ONLY_STEP") if mapped == before else (key, None)

    return apply


def _matrix_key(bonds, lone) -> tuple[frozenset, frozenset]:
    pairs = frozenset((min(i, j), max(i, j), int(d)) for i, j, d in bonds)
    return pairs, frozenset((int(i), int(d)) for i, d in lone)


def hit_rank(found: list[Completion], label: EditLabel) -> int | None:
    """The 0-based rank of the first completion that is a variant of the label class, None
    when no completion is."""
    wanted = {_matrix_key(v.bonds, v.lone) for v in label.variants}
    for rank, c in enumerate(found):
        if _matrix_key(c.bonds, c.lone) in wanted:
            return rank
    return None


def sample_decoder(model, cfg: DecodeConfig, scope: str = "all") -> Decode:
    """`decode(sample, skips)`: the ranked completions of one stored sample."""

    def decode(sample, skips: Counter) -> list[Completion]:
        return decode_pool(model, sample, cfg, lewis_apply(sample), scope, skips)

    return decode


def _odd_top1(found: list[Completion]) -> bool:
    """Whether the best completion has an odd atom entry."""
    return bool(found) and any(d % 2 for _, d in found[0].lone)


def any_rank(found: list[Completion], rank: int | None, products: frozenset[str]) -> int | None:
    """The 0-based rank of the first completion that is the own label (`rank`) or whose product
    key is one of `products`, the outcomes the source records for the pool; None when none is."""
    hits = [r for r, c in enumerate(found) if c.key is not None and c.key in products]
    ranks = [r for r in (rank, hits[0] if hits else None) if r is not None]
    return min(ranks) if ranks else None


def _row(sample, found: list[Completion], outcomes: Mapping, skips: Counter) -> dict:
    """One sample's row: the own rank and `rank_any` against its pool's recorded outcomes (the
    own label alone when the step has no entry: `rank_any` = `rank`, `n_outcomes` 0), and how
    many outcomes the pool records. An own hit whose product key is not the recorded one is
    counted as OWN_KEY_MISMATCH."""
    step_id = getattr(sample, "step_id", None)
    rank = hit_rank(found, sample.edit_label)
    pool = outcomes.get(step_id)
    products = pool.products if pool is not None else frozenset()
    if pool is not None and rank is not None and found[rank].key not in (None, pool.own):
        skips["OWN_KEY_MISMATCH"] += 1
    row = {"step_id": step_id, "rank": rank, "rank_any": any_rank(found, rank, products)}
    return row | {"n_outcomes": len(products), "n_found": len(found), "odd_top1": _odd_top1(found)}


def _tops(rows: list[dict], field: str, ks: tuple[int, ...], suffix: str = "") -> dict:
    n = len(rows)
    hit = [r[field] for r in rows]
    return {
        f"top{k}{suffix}": sum(h is not None and h < k for h in hit) / n if n else 0.0 for k in ks
    }


def _multi_outcome(rows: list[dict], ks: tuple[int, ...]) -> dict:
    """Own and any top-k over the samples whose pool records more than one product."""
    multi = [r for r in rows if r["n_outcomes"] > 1]
    return {"n": len(multi)} | _tops(multi, "rank", ks) | _tops(multi, "rank_any", ks, "_any")


def decode_report(
    samples: Iterable,
    decode: Decode,
    ks: tuple[int, ...] = (1, 3, 5),
    outcomes: Mapping | None = None,
) -> dict:
    """Top-k exact-R hits over `samples`, and top-k against any outcome the source records for
    the same reactant pool (`outcomes`: step id -> `eval.siblings.Outcomes`), the misses, the
    decoder's drop counts and the time per sample. A sample without a label class is counted as
    NO_EDIT_LABEL and left out. `per_sample` holds one row per scored sample (step id, the
    0-based rank of the first own hit or None, `rank_any`, the outcomes its pool records, the
    completions found, whether the best completion has an odd atom entry, and with any the
    completions the closed-shell screen and filter dropped, by code), so two decodes can be
    compared sample by sample. `odd_top1` is the share of the samples with a completion whose
    best one has an odd entry; `multi_outcome` the own and any top-k over the samples whose pool
    records several products."""
    rows: list[dict] = []
    skips: Counter = Counter()
    seconds: list[float] = []
    for sample in samples:
        if sample.edit_label is None:
            skips["NO_EDIT_LABEL"] += 1
            continue
        before = {c: skips[f"DECODE_DROPPED:{c}"] for c in SCREEN_CODES}
        t0 = time.perf_counter()
        found = decode(sample, skips)
        seconds.append(time.perf_counter() - t0)
        rows.append(_row(sample, found, outcomes or {}, skips))
        dropped = {c: skips[f"DECODE_DROPPED:{c}"] - before[c] for c in SCREEN_CODES}
        if any(dropped.values()):
            rows[-1]["screened"] = {c: k for c, k in dropped.items() if k}
        if not found:
            skips["DECODE_NOTHING_FOUND"] += 1
    with_found = [r for r in rows if r["n_found"]]
    return (
        _tops(rows, "rank", ks)
        | _tops(rows, "rank_any", ks, "_any")
        | {
            "n": len(rows),
            "not_found": sum(r["rank"] is None for r in rows),
            "odd_top1": sum(r["odd_top1"] for r in with_found) / max(len(with_found), 1),
            "multi_outcome": _multi_outcome(rows, ks),
            "mean_seconds": float(np.mean(seconds)) if seconds else 0.0,
            "skips": dict(sorted(skips.items())),
            "per_sample": rows,
        }
    )
