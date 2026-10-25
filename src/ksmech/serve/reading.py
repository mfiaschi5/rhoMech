"""The arrow reading of a reaction matrix (spec v9, "Orbital interpretation").

R is decomposed into electron-pair moves, the curved arrows of the step:

- `lone_pair_to_bond`: a lone pair on a forms the bond a-b (source [a], target [a, b]);
- `bond_to_bond`: the pair of the bond a-c moves to the new bond a-b, which forms at a
  (source [a, c], target [a, b]; a is the shared atom);
- `bond_to_lone_pair`: the bond a-c breaks and its pair stays on c (source [a, c], target
  [c]; the receiving atom is the last of the source).

Every forming bond (a +1 of R, counted once per unit) takes its pair from a lone pair of one of
its atoms (a -2 of R there) or from a breaking bond at one of its atoms; every breaking bond
(a -1) ends as a lone pair of one of its atoms (a +2 there) or in a forming bond. A
decomposition is a perfect matching of the sources (lone-pair losses and breaking bonds) onto
the sinks (forming bonds and lone-pair gains); charge conservation makes the two counts
equal. When several exist, the one whose forming bonds have the largest pooled gap factors is
read: each forming bond a-b fed from atom a scores |G_dir[a, b]|, the directed pooled gap
with the donor on a (`serve.gatom.PooledGap.directed`; G_atom itself is symmetric, so it
cannot tell the two readings of one bond apart). Ties keep the first matching in a fixed
order (sources and sinks sorted), so the reading is deterministic. An R without any
decomposition (a lone pair that moves to no bond at its atom, an odd lone-pair change) has no
reading: `decompose` returns None.

Atoms are pool atoms (`parse_mapped(state.mapped_smiles)` indices). The PMechDB arrow code of
a reading (`pmech_code`) uses the map numbers of those atoms.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

LONE_TO_BOND = "lone_pair_to_bond"
BOND_TO_BOND = "bond_to_bond"
BOND_TO_LONE = "bond_to_lone_pair"
MAX_NODES = 200_000  # search nodes before the reading gives up (never reached on PMechDB)

Directed = Callable[[int, int], float]  # |pooled gap| with the donor on the first atom


@dataclass(frozen=True)
class Move:
    kind: str
    source: tuple[int, ...]
    target: tuple[int, ...]

    @property
    def forms(self) -> tuple[int, int] | None:
        """(donor end, other end) of the bond the move forms, None for a heterolysis."""
        return None if self.kind == BOND_TO_LONE else (self.target[0], self.target[1])

    def to_dict(self) -> dict:
        return {"kind": self.kind, "source": list(self.source), "target": list(self.target)}

    @classmethod
    def from_dict(cls, d: dict) -> Move:
        return cls(
            str(d["kind"]), tuple(int(x) for x in d["source"]), tuple(int(x) for x in d["target"])
        )


def _units(R) -> tuple[list, list, list, list] | None:
    """(forming bonds, breaking bonds, lone-pair losses, lone-pair gains), one per unit."""
    forming = [(i, j) for i, j, d in R.bonds if d > 0 for _ in range(d)]
    breaking = [(i, j) for i, j, d in R.bonds if d < 0 for _ in range(-d)]
    if any(d % 2 for _, d in R.lone):
        return None
    loss = [i for i, d in R.lone if d < 0 for _ in range(-d // 2)]
    gain = [i for i, d in R.lone if d > 0 for _ in range(d // 2)]
    return sorted(forming), sorted(breaking), sorted(loss), sorted(gain)


def _options(src, forming: list, gain: list) -> list[tuple[str, int, Move]]:
    """(sink kind, sink index, move) for one source: ("lone", a) or ("bond", (a, c))."""
    kind, x = src
    out = []
    if kind == "lone":
        for k, (i, j) in enumerate(forming):
            if x in (i, j):
                b = j if x == i else i
                out.append(("f", k, Move(LONE_TO_BOND, (x,), (x, b))))
        return out
    a0, c0 = x
    for k, (i, j) in enumerate(forming):
        for a, c in ((a0, c0), (c0, a0)):
            if a in (i, j) and c not in (i, j):
                b = j if a == i else i
                out.append(("f", k, Move(BOND_TO_BOND, (a, c), (a, b))))
    for k, g in enumerate(gain):
        if g in x:
            other = x[0] if g == x[1] else x[1]
            out.append(("g", k, Move(BOND_TO_LONE, (other, g), (g,))))
    return out


def _score(moves: Sequence[Move], directed: Directed | None) -> float:
    if directed is None:
        return 0.0
    return float(sum(abs(directed(*m.forms)) for m in moves if m.forms is not None))


def decompose(R, directed: Directed | None = None) -> list[Move] | None:
    """The arrow reading of R (see the module docstring), or None when R has none."""
    units = _units(R)
    if units is None:
        return None
    forming, breaking, loss, gain = units
    if len(loss) + len(breaking) != len(forming) + len(gain):
        return None
    sources = [("lone", a) for a in loss] + [("bond", b) for b in breaking]
    options = [_options(s, forming, gain) for s in sources]
    best: list = [None, -1.0]
    nodes = [0]

    def search(k: int, used_f: set, used_g: set, moves: list) -> None:
        nodes[0] += 1
        if nodes[0] > MAX_NODES:
            return
        if k == len(sources):
            score = _score(moves, directed)
            if score > best[1] + 1e-12:
                best[0], best[1] = list(moves), score
            return
        for sink, idx, move in options[k]:
            used = used_f if sink == "f" else used_g
            if idx in used:
                continue
            used.add(idx)
            moves.append(move)
            search(k + 1, used_f, used_g, moves)
            moves.pop()
            used.discard(idx)

    search(0, set(), set(), [])
    return best[0]


def pmech_code(moves: Sequence[Move], map_of: Callable[[int], int]) -> str:
    """The PMechDB arrow code of a reading (`a=b`, `a,c=a,b`, `a,c=c`) on map numbers."""
    out = []
    for m in moves:
        src = ",".join(str(map_of(a)) for a in m.source)
        if m.kind == LONE_TO_BOND:
            out.append(f"{src}={map_of(m.target[1])}")
        else:
            out.append(f"{src}={','.join(str(map_of(a)) for a in m.target)}")
    return ";".join(out)
