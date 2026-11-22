"""Extract the GFN2-xTB parameter tables of tblite (and mctc-lib's element data) into JSON.

    python native/ksxtb/tools/extract_gfn2.py --tblite SRC/tblite --mctc SRC/mctc-lib \
        --out src/ksmech/qm/gpu/gfn2_params.json

The Fortran array literals are parsed, not retyped: `reshape([...], shape(x)) * evtoau`,
`1.0_wp + reshape(...)`, `0.01_wp * [...]`, `spread(v, 1, n)`. Arrays are stored with
their Fortran dimensions, flattened in Fortran (column-major) order, so `np.reshape(a, dims,
order="F")` restores them. Units are tblite's after its own scalings (Hartree, Bohr).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

# CODATA 2018 Bohr radius in Angstrom, as mctc-lib derives it (mctc_io_convert)
AUTOAA = 0.529177210903
CONSTANTS = {"evtoau": 1.0 / 27.21138505, "aatoau": 1.0 / AUTOAA}

GFN2_ARRAYS = (
    "nshell", "ang_shell", "principal_quantum_number", "number_of_primitives",
    "slater_exponent", "p_selfenergy", "p_kcn", "p_shpoly", "reference_occ",
    "rep_alpha", "rep_zeff", "hubbard_parameter", "shell_hubbard", "p_hubbard_derivs",
    "shell_hubbard_derivs", "p_dkernel", "p_qkernel", "p_vcn", "p_rad", "kdiag",
)  # fmt: skip
GFN2_SCALARS = (
    "s6", "s8", "a1", "a2", "s9", "rep_kexp", "rep_kexp_light", "rep_rexp", "wexp",
    "enscale", "gexp", "mp_dmp3", "mp_dmp5", "mp_shift", "mp_kexp", "mp_rmax",
)  # fmt: skip
SLATER_ARRAYS = tuple(
    [f"pAlpha{n}" for n in range(1, 7)] + [f"pCoeff{n}" for n in range(2, 7)]
    + ["pAlpha6s", "pCoeff6s", "pAlpha6p", "pCoeff6p"]
)  # fmt: skip


def _statements(text: str) -> list[str]:
    """Fortran free-form statements with comments removed and continuations joined."""
    out, cur = [], ""
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].rstrip()
        stripped = line.lstrip()
        if not stripped and cur:  # a comment line inside a continued statement
            continue
        if stripped.startswith("&"):
            stripped = stripped[1:]
        if stripped.endswith("&"):
            cur += stripped[:-1] + " "
            continue
        out.append(cur + stripped)
        cur = ""
    return out


def _arr(*items):
    flat = []
    for it in items:
        flat.extend(np.ravel(np.asarray(it, dtype=float), order="F").tolist())
    return np.array(flat)


def _reshape(values, _shape=None):
    return np.asarray(values, dtype=float)


def _spread(value, _dim, n):
    return np.full(int(n), float(value))


_NUM = re.compile(r"(?<![\w.])(\d+\.?\d*|\.\d+)(?:[eEdD]([+-]?\d+))?_wp")


def _to_python(expr: str) -> str:
    expr = _NUM.sub(lambda m: f"{m.group(1)}e{m.group(2) or 0}", expr)
    expr = re.sub(r"shape\s*\(\s*\w+\s*\)", "None", expr)
    expr = expr.replace("[", "_arr(").replace("]", ")")
    expr = re.sub(r"\breshape\s*\(", "_reshape(", expr)
    return re.sub(r"\bspread\s*\(", "_spread(", expr)


def _dims(spec: str | None, consts: dict) -> list[tuple[int, int]]:
    if not spec:
        return []
    dims = []
    for part in spec.split(","):
        part = part.strip()
        lo, hi = part.split(":") if ":" in part else ("1", part)
        dims.append((int(_eval(lo, consts)), int(_eval(hi, consts)) if hi != "*" else -1))
    return dims


def _eval(expr: str, consts: dict):
    ns = {"_arr": _arr, "_reshape": _reshape, "_spread": _spread, "sqrt": math.sqrt}
    return eval(_to_python(expr), ns, dict(consts))  # noqa: S307 - our own source tables


_INT = re.compile(r"integer\s*,\s*parameter\s*::\s*(\w+)\s*=\s*(\d+)\s*$", re.IGNORECASE)


def _file_constants(stmts: list[str]) -> dict:
    """CONSTANTS plus the file's own scalar integer parameters (max_elem is 86 in tblite's
    gfn2.f90 but 118 in mctc-lib's tables)."""
    consts = dict(CONSTANTS)
    for stmt in stmts:
        m = _INT.search(stmt.strip())
        if m:
            consts[m.group(1)] = int(m.group(2))
    return consts


_DECL = re.compile(r"parameter\s*::\s*(.+)$", re.IGNORECASE)
_ITEM = re.compile(r"^\s*(\w+)\s*(?:\(([^)]*)\))?\s*=\s*(.+)$", re.DOTALL)


def _top_level_split(text: str) -> list[str]:
    """Split at the commas outside () and [] (`s6 = 1.0_wp, s8 = 2.7_wp`)."""
    parts, depth, cur = [], 0, ""
    for ch in text:
        depth += ch in "([" and 1 or (ch in ")]" and -1 or 0)
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return [*parts, cur]


def _declarations(stmt: str):
    m = _DECL.search(stmt)
    if not m:
        return
    for piece in _top_level_split(m.group(1)):
        item = _ITEM.match(piece)
        if item:
            yield item.groups()


def _value(name, spec, expr, consts, path) -> dict:
    value = np.atleast_1d(np.asarray(_eval(expr, consts), dtype=float))
    dims = _dims(spec, consts)
    shape = [hi - lo + 1 if hi >= 0 else value.size for lo, hi in dims] or [value.size]
    if int(np.prod(shape)) != value.size:
        raise ValueError(f"{path.name}:{name}: {value.size} values for dims {shape}")
    return {"dims": shape, "lbounds": [lo for lo, _ in dims] or [1], "values": value.tolist()}


def parse(path: Path, names: tuple[str, ...]) -> dict[str, dict]:
    """The named parameters of one Fortran file: {name: {dims, lbounds, values}}."""
    stmts = _statements(path.read_text())
    consts = _file_constants(stmts)
    found = {}
    for stmt in stmts:
        for name, spec, expr in _declarations(stmt):
            if name in names:
                found[name] = _value(name, spec, expr, consts, path)
    missing = set(names) - set(found)
    if missing:
        raise ValueError(f"{path.name}: parameters not found: {sorted(missing)}")
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tblite", type=Path, required=True, help="tblite checkout (v0.7.0)")
    ap.add_argument("--mctc", type=Path, required=True, help="mctc-lib checkout (v0.5.x)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    src = args.tblite / "src" / "tblite"
    data = {
        "source": "tblite v0.7.0 src/tblite/xtb/gfn2.f90, basis/slater.f90; mctc-lib data",
        "gfn2": parse(src / "xtb" / "gfn2.f90", GFN2_ARRAYS + GFN2_SCALARS),
        "slater": parse(src / "basis" / "slater.f90", SLATER_ARRAYS),
        "atomic_rad": parse(args.mctc / "src/mctc/data/atomicrad.f90", ("atomic_rad",)),
        "pauling_en": parse(args.mctc / "src/mctc/data/paulingen.f90", ("pauling_en",)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=0) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
