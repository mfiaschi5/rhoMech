"""Full-step decoding of stored samples (spec v9, "Evaluation protocol"): the beam decoder
(`model.decode`) on each sample of a part, B + R checked on the sample's own Lewis structure
(`lewis_apply`), and a hit at k when one of the k best completions is a variant of the label
class (`EditLabel.variants`, on the global atoms, as the completions are).

`lewis_apply` refuses a completion with the first of: BOND_ORDER (an order leaves [0, 3]),
LONE_PAIRS (a negative lone-pair count), OCTET_EXCEEDED (an H-Ne atom the step touches over 8
valence electrons, H and He over 2), FORMAL_CHARGE_RANGE (a non-metal it touches left at
|formal charge| >= 2 that it did not carry), PRODUCT_INVALID (RDKit cannot write the product),
RESONANCE_ONLY_STEP (no atom changes its sigma bonds or H: the sigma keys with the atom maps
agree). The product key is the map-free sigma key of the pool (`data.resonance.sigma_key`), so
completions that draw the same products differently merge, as serve's pool key merges them.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable

import numpy as np
from rdkit import Chem, rdBase

from ksmech.arrows.delta_be import EditLabel
from ksmech.arrows.lewis import MAX_ORDER
from ksmech.arrows.symmetry import lewis_mol
from ksmech.data.resonance import sigma_key
from ksmech.model.decode import NONMETALS, Apply, Completion, DecodeConfig, decode_pool

Decode = Callable[[object, Counter], list[Completion]]


def _valence_limit(z: int) -> int | None:
    """Valence electrons an atom may carry: 2 for H and He, 8 for Li-Ne, None beyond."""
    return 2 if z <= 2 else 8 if z <= 10 else None


def _pool_mol(z: np.ndarray, q: np.ndarray, order: dict[tuple[int, int], int]) -> Chem.Mol:
    """The pool on the token atoms, atom k mapped k + 1."""
    rows = [(i, j, o) for (i, j), o in sorted(order.items()) if o > 0]
    mol = lewis_mol(z, q, np.asarray(rows, dtype=np.int64).reshape(-1, 3))
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return mol


def _atom_issue(z, q0, q, lone_pairs, order, touched) -> str | None:
    """OCTET_EXCEEDED or FORMAL_CHARGE_RANGE on the touched atoms (module docstring), or None."""
    electrons = 2 * lone_pairs.astype(np.int64)
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
    lp0 = np.asarray(sample.lone_pairs, dtype=np.int64)
    with rdBase.BlockLogs():
        before = sigma_key(_pool_mol(z, q0, order0), keep_maps=True)

    def apply(bonds, lone):
        order, lp, q = dict(order0), lp0.copy(), q0.copy()
        for i, j, d in bonds:
            o = order.get((i, j), 0) + d
            if not 0 <= o <= MAX_ORDER:
                return None, "BOND_ORDER"
            order[(i, j)] = o
            q[i] -= d
            q[j] -= d
        for i, d in lone:
            lp[i] += d // 2
            q[i] -= d
        if (lp < 0).any():
            return None, "LONE_PAIRS"
        touched = {a for i, j, _ in bonds for a in (i, j)} | {i for i, _ in lone}
        why = _atom_issue(z, q0, q, lp, order, touched)
        if why is not None:
            return None, why
        try:
            with rdBase.BlockLogs():
                mol = _pool_mol(z, q, order)
                mapped, key = sigma_key(mol, keep_maps=True), sigma_key(mol)
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


def decode_report(samples: Iterable, decode: Decode, ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    """Top-k exact-R hits over `samples`, the misses, the decoder's drop counts and the time
    per sample. A sample without a label class is counted as NO_EDIT_LABEL and left out."""
    ranks: list[int | None] = []
    skips: Counter = Counter()
    seconds: list[float] = []
    for sample in samples:
        if sample.edit_label is None:
            skips["NO_EDIT_LABEL"] += 1
            continue
        t0 = time.perf_counter()
        found = decode(sample, skips)
        seconds.append(time.perf_counter() - t0)
        ranks.append(hit_rank(found, sample.edit_label))
        if not found:
            skips["DECODE_NOTHING_FOUND"] += 1
    n = len(ranks)
    out = {f"top{k}": sum(r is not None and r < k for r in ranks) / n if n else 0.0 for k in ks}
    return out | {
        "n": n,
        "not_found": sum(r is None for r in ranks),
        "mean_seconds": float(np.mean(seconds)) if seconds else 0.0,
        "skips": dict(sorted(skips.items())),
    }
