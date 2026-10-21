"""Contact overlap table S[kappa] for orbital pairs that share no geometry (spec section 5.4).

    kappa = (Z_m(r), l_r, Z_m(s), l_s)

the element and dominant shell of each token's main centre m(.), with l the largest of the
centre's s / p / d shares (per shell, so the key does not depend on how a p orbital points).
No orbital label enters the key. The table is built once from intramolecular statistics of the
token cache: for every non-bonded atom pair (a, b) whose separation lies within `window` of the
sum of van der Waals radii, every occupied token r with m(r) = a and every token s with
m(s) = b, record |l_r,a^T S_ab l_s,b| (the AOs of r on a against the AOs of s on b). S[kappa]
is the median of each group. Two tables: occupied-virtual (the orbital term) and
occupied-occupied (the repulsion term). A lookup backs off (Z, l, Z, l) -> (Z, Z) -> (l, l) ->
the global median, using a level only when it holds at least `min_count` samples.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
from rdkit import Chem

from ksmech.orbitals.tokens import MoleculeTokens, OrbitalKind, OrbitalToken

KINDS = ("occ_virt", "occ_occ")
LEVELS = ("element_shell", "element", "shell", "global")
DEFAULT_TABLE_PATH = Path("data/cache/contact_table.json")
_BONDED = 1.3  # x covalent-radii sum: the QM layer's bond-stretch criterion


def contact_key(o: OrbitalToken, atomic_numbers: np.ndarray) -> tuple[int, int]:
    """(Z, l) of a token's main centre and its dominant shell."""
    d = o.descriptors
    return int(atomic_numbers[d.center_atoms[0]]), int(np.argmax(d.center_spd[0]))


@cache
def vdw_sum(za: int, zb: int) -> float:
    table = Chem.GetPeriodicTable()
    return table.GetRvdw(int(za)) + table.GetRvdw(int(zb))


def _contact_pairs(t: MoleculeTokens, window: float) -> list[tuple[int, int]]:
    """Ordered non-bonded atom pairs (a, b) at van der Waals contact, both orientations."""
    table = Chem.GetPeriodicTable()
    z = [a.atomic_number for a in t.atoms]
    pos = np.array([a.position for a in t.atoms]).reshape(-1, 3)
    out = []
    for a in range(len(z)):
        for b in range(a + 1, len(z)):
            r = float(np.linalg.norm(pos[a] - pos[b]))
            bonded = r <= _BONDED * (table.GetRcovalent(z[a]) + table.GetRcovalent(z[b]))
            if not bonded and abs(r - vdw_sum(z[a], z[b])) <= window:
                out += [(a, b), (b, a)]
    return out


def contact_rows(t: MoleculeTokens, window: float = 0.3) -> list[tuple[str, int, int, float]]:
    """(kind, r, s, |l_r,a^T S_ab l_s,b|) for every contact sample of one molecule, with r and s
    positions in `t.orbitals`: the samples of the table."""
    main = np.array([int(o.descriptors.center_atoms[0]) for o in t.orbitals])
    occ = np.array([o.kind == OrbitalKind.OCCUPIED for o in t.orbitals])
    L = t.C_loc
    out = []
    for a, b in _contact_pairs(t, window):
        rs = np.flatnonzero((main == a) & occ)
        ss = np.flatnonzero(main == b)
        if not len(rs) or not len(ss):
            continue
        ia, ib = t.ao_atom == a, t.ao_atom == b
        block = np.abs(L[ia][:, rs].T @ t.S[np.ix_(ia, ib)] @ L[ib][:, ss])
        for i, r in enumerate(rs):
            for j, s in enumerate(ss):
                kind = "occ_occ" if occ[s] else "occ_virt"
                out.append((kind, int(r), int(s), float(block[i, j])))
    return out


def collect_contacts(t: MoleculeTokens, window: float = 0.3) -> list[tuple[str, tuple, float]]:
    """(kind, kappa, |l_r,a^T S_ab l_s,b|) for every contact sample of one molecule."""
    z = np.array([a.atomic_number for a in t.atoms])
    keys = [contact_key(o, z) for o in t.orbitals]
    return [(kind, (*keys[r], *keys[s]), v) for kind, r, s, v in contact_rows(t, window)]


def _level_keys(kappa: tuple[int, int, int, int]) -> tuple[tuple, ...]:
    za, la, zb, lb = kappa
    return (kappa, (za, zb), (la, lb), ())


@dataclass
class ContactTable:
    """Medians of one kind of contact sample at every backoff level, with sample counts."""

    kind: str
    min_count: int
    medians: dict[str, dict[tuple, float]]
    counts: dict[str, dict[tuple, int]]
    n_molecules: int = 0
    method: str = ""
    lookups: dict[str, int] = field(default_factory=lambda: dict.fromkeys(LEVELS, 0))

    def lookup(self, kappa: tuple[int, int, int, int], n: int = 1) -> tuple[float, str]:
        """(S[kappa], the backoff level that supplied it); counts `n` lookups at that level
        (a vectorized caller resolves each distinct key once for `n` token pairs)."""
        for level, key in zip(LEVELS, _level_keys(kappa), strict=True):
            if self.counts[level].get(key, 0) >= self.min_count or level == "global":
                self.lookups[level] += n
                return self.medians[level].get(key, float("nan")), level
        raise AssertionError("unreachable")


def _accumulate(samples) -> dict[str, dict[str, dict[tuple, list[float]]]]:
    groups = {kind: {lv: defaultdict(list) for lv in LEVELS} for kind in KINDS}
    for kind, kappa, value in samples:
        for level, key in zip(LEVELS, _level_keys(kappa), strict=True):
            groups[kind][level][key].append(value)
    return groups


def tables_from_samples(
    samples, min_count: int = 5, n_molecules: int = 0, method: str = ""
) -> dict[str, ContactTable]:
    """Both tables from (kind, kappa, value) samples of many molecules, in one pass."""
    out = {}
    for kind, by_level in _accumulate(samples).items():
        medians = {lv: {k: float(np.median(v)) for k, v in g.items()} for lv, g in by_level.items()}
        counts = {lv: {k: len(v) for k, v in g.items()} for lv, g in by_level.items()}
        out[kind] = ContactTable(kind, min_count, medians, counts, n_molecules, method)
    return out


def _encode(d: dict[tuple, float]) -> dict[str, float]:
    return {",".join(map(str, k)): v for k, v in d.items()}


def _decode(d: dict[str, float]) -> dict[tuple, float]:
    return {tuple(int(x) for x in k.split(",")) if k else (): v for k, v in d.items()}


def save_tables(tables: dict[str, ContactTable], path: Path = DEFAULT_TABLE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        kind: dict(
            min_count=t.min_count,
            n_molecules=t.n_molecules,
            method=t.method,
            medians={lv: _encode(m) for lv, m in t.medians.items()},
            counts={lv: _encode(c) for lv, c in t.counts.items()},
        )
        for kind, t in tables.items()
    }
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


def load_tables(path: Path = DEFAULT_TABLE_PATH) -> dict[str, ContactTable]:
    """The occupied-virtual and occupied-occupied tables written by `save_tables`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: no contact table; build it with `ksmech-orbitals contact-table`"
        )
    data = json.loads(path.read_text())
    return {
        kind: ContactTable(
            kind,
            int(d["min_count"]),
            {lv: _decode(m) for lv, m in d["medians"].items()},
            {lv: {k: int(v) for k, v in _decode(c).items()} for lv, c in d["counts"].items()},
            int(d["n_molecules"]),
            d["method"],
        )
        for kind, d in data.items()
    }


def tables_from_tokens(
    tokens: list[MoleculeTokens], min_count: int = 5, window: float = 0.3
) -> dict[str, ContactTable]:
    samples = (smp for t in tokens for smp in collect_contacts(t, window))
    method = tokens[0].method if tokens else ""
    return tables_from_samples(samples, min_count, len(tokens), method)


def tables_from_paths(
    paths: list[Path], method: str, min_count: int = 5, window: float = 0.3
) -> tuple[dict[str, ContactTable], dict[str, int]]:
    """Both tables from token files, read one at a time; also the skip counts by reason."""
    from ksmech.orbitals.tokens import TokenSchemaError, load_tokens

    skipped: dict[str, int] = defaultdict(int)
    used = 0

    def samples():
        nonlocal used
        for path in paths:
            if not Path(path).is_file():
                skipped["missing_tokens"] += 1
                continue
            try:
                t = load_tokens(path)
            except TokenSchemaError:
                skipped["old_schema"] += 1
                continue
            if t.method != method:
                skipped["other_method"] += 1
                continue
            used += 1
            yield from collect_contacts(t, window)

    tables = tables_from_samples(samples(), min_count, 0, method)
    for t in tables.values():
        t.n_molecules = used
    return tables, dict(skipped)
