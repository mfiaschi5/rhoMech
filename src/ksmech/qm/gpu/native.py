"""The ksxtb CUDA library: build, load, and a batch of molecules on the device.

The library is compiled on first use (or when a source is newer) with the nvcc of
`KSXTB_NVCC`, default the CUDA 13.3 wheel toolchain under ~/.local/share/ksxtb (CUDA 13.0's
headers clash with glibc 2.42, docs/decisions.md 2026-11-22). Device memory is held in torch
tensors; the C entry points take their raw pointers and torch's current stream.
"""

from __future__ import annotations

import ctypes as C
import os
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import torch

from ksmech.qm.gpu.basis import element_shells, mol_basis, pair_cutoff, params
from ksmech.qm.gpu.reference import h0_params

ROOT = Path(__file__).resolve().parents[4] / "native" / "ksxtb"
LIB = ROOT / "build" / "libksxtb.so"
DEFAULT_NVCC = (
    Path.home() / ".local/share/ksxtb/tc13x/lib/python3.11/site-packages/nvidia/cu13/bin/nvcc"
)
NVCC_FLAGS = (
    "-O3", "-arch=sm_89", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-lineinfo",
    "-allow-unsupported-compiler",  # gcc 16 on this host; the checks against tblite cover it
)  # fmt: skip


def _sources() -> list[Path]:
    return sorted([*ROOT.glob("src/*.cu"), *ROOT.glob("src/*.cuh"), *ROOT.glob("include/*.h")])


def build(force: bool = False) -> Path:
    """Compile libksxtb.so when it is missing or older than a source."""
    srcs = _sources()
    if not force and LIB.exists() and LIB.stat().st_mtime >= max(p.stat().st_mtime for p in srcs):
        return LIB
    nvcc = os.environ.get("KSXTB_NVCC", str(DEFAULT_NVCC))
    LIB.parent.mkdir(parents=True, exist_ok=True)
    cmd = [nvcc, *NVCC_FLAGS, f"-I{ROOT / 'include'}", "-o", str(LIB)]
    cmd += [str(p) for p in srcs if p.suffix == ".cu"]
    subprocess.run(cmd, check=True)
    return LIB


P = C.c_void_p


class KsBatch(C.Structure):
    _fields_ = [
        (n, P)
        for n in (
            "xyz", "atom_rad", "atom_mol", "sh_atom", "sh_cg", "sh_ao", "sh_shpoly", "sh_se",
            "sh_dsedcn", "cg_nprim", "cg_alpha", "cg_coeff", "mol_mat", "mol_nao", "mol_ao",
            "mol_cut2",
        )
    ] + [("nat", C.c_int), ("nsh", C.c_int), ("nmol", C.c_int)]  # fmt: skip


class KsPairs(C.Structure):
    _fields_ = [("si", P), ("sj", P), ("hscale", P)] + [
        (n, C.c_int) for n in ("n", "li", "lj", "onsite")
    ]


class KsFwdOut(C.Structure):
    _fields_ = [(n, P) for n in ("S", "H0", "D", "Q")]


class KsGradIn(C.Structure):
    _fields_ = [(n, P) for n in ("P", "X", "vao", "vdp", "vqp")]


class KsGradOut(C.Structure):
    _fields_ = [(n, P) for n in ("grad", "dedcn")]


@cache
def lib() -> C.CDLL:
    so = C.CDLL(str(build()))
    for name, args in (
        ("ksxtb_forward", [C.POINTER(KsBatch), C.POINTER(KsPairs), C.c_int,
                           C.POINTER(KsFwdOut), P]),
        ("ksxtb_h0_gradient", [C.POINTER(KsBatch), C.POINTER(KsPairs), C.c_int,
                               C.POINTER(KsGradIn), C.POINTER(KsGradOut), P]),
    ):  # fmt: skip
        fn = getattr(so, name)
        fn.argtypes, fn.restype = args, C.c_int
    return so


def _check(code: int, what: str) -> None:
    if code != 0:
        raise RuntimeError(f"{what} failed with cudaError {code}")


def _ptr(t: torch.Tensor | None) -> int | None:
    return None if t is None else t.data_ptr()


@dataclass
class HostBatch:
    """Host-side arrays of a batch; `device()` moves them to the GPU."""

    arrays: dict[str, np.ndarray]
    pairs: list[tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]]  # li lj onsite si sj h
    nat: int
    nsh: int
    nmol: int
    mat_total: int
    ao_total: int


def _cg_table(zs: set[int]):
    rows, index = [], {}
    for z in sorted(zs):
        for ish, cg in enumerate(element_shells(z)):
            index[z, ish] = len(rows)
            rows.append(cg)
    alpha = np.zeros((len(rows), 6))
    coeff = np.zeros((len(rows), 6))
    for r, cg in enumerate(rows):
        alpha[r, : cg.nprim], coeff[r, : cg.nprim] = cg.alpha, cg.coeff
    return index, np.array([cg.nprim for cg in rows], np.int32), alpha, coeff


def _mol_pairs(basis, sh0: int, hscale: np.ndarray, ls: np.ndarray, out: dict) -> None:
    """Append one molecule's shell pairs, global shell indices, to out[(li, lj, onsite)]."""
    atom = basis.sh_atom
    si, sj = np.nonzero(atom[:, None] >= atom[None, :])
    keep = (atom[si] > atom[sj]) | (atom[si] == atom[sj])
    si, sj = si[keep], sj[keep]
    onsite = atom[si] == atom[sj]
    for li in range(3):
        for lj in range(3):
            for on in (0, 1):
                sel = (ls[si] == li) & (ls[sj] == lj) & (onsite == bool(on))
                if sel.any():
                    a, b = si[sel], sj[sel]
                    out.setdefault((li, lj, on), []).append((a + sh0, b + sh0, hscale[b, a]))


def host_batch(mols: list[tuple[np.ndarray, np.ndarray]]) -> HostBatch:
    """Pack molecules (z, xyz in Bohr) into a batch."""
    zs = {int(z) for zz, _ in mols for z in zz}
    cg_index, cg_nprim, cg_alpha, cg_coeff = _cg_table(zs)
    rad_tab = params()["atomic_rad"]["atomic_rad"]
    shp = params()["gfn2"]["p_shpoly"]
    cols = {k: [] for k in ("xyz", "atom_rad", "atom_mol", "sh_atom", "sh_cg", "sh_ao",
                            "sh_shpoly", "mol_mat", "mol_nao", "mol_ao", "mol_cut2")}  # fmt: skip
    pairs: dict = {}
    nat = nsh = mat = nao_tot = 0
    for m, (z, xyz) in enumerate(mols):
        z = np.asarray(z, dtype=int)
        basis = mol_basis(z)
        ls = np.array([basis.shell(s).l for s in range(len(basis.sh_atom))])
        zsh = z[basis.sh_atom]
        cols["xyz"].append(np.asarray(xyz, dtype=float).reshape(-1))
        cols["atom_rad"].append(rad_tab[z - 1])
        cols["atom_mol"].append(np.full(len(z), m))
        cols["sh_atom"].append(basis.sh_atom + nat)
        cols["sh_cg"].append(
            [cg_index[int(a), int(b)] for a, b in zip(zsh, basis.sh_elem, strict=True)]
        )
        cols["sh_ao"].append(basis.sh_ao)
        cols["sh_shpoly"].append(shp[ls, zsh - 1])
        cols["mol_mat"].append([mat])
        cols["mol_nao"].append([basis.nao])
        cols["mol_ao"].append([nao_tot])
        cols["mol_cut2"].append([pair_cutoff(z) ** 2])
        _mol_pairs(basis, nsh, h0_params(basis).hscale, ls, pairs)
        nat, nsh, mat, nao_tot = (
            nat + len(z),
            nsh + len(ls),
            mat + basis.nao**2,
            nao_tot + basis.nao,
        )
    ints = {"atom_mol", "sh_atom", "sh_cg", "sh_ao", "mol_nao", "mol_ao"}
    arrays = {k: np.concatenate([np.asarray(x) for x in v]) for k, v in cols.items()}
    arrays = {k: v.astype(np.int32 if k in ints else (np.int64 if k == "mol_mat" else float))
              for k, v in arrays.items()}  # fmt: skip
    arrays.update(cg_nprim=cg_nprim, cg_alpha=cg_alpha.ravel(), cg_coeff=cg_coeff.ravel())
    plist = []
    for (li, lj, on), parts in sorted(pairs.items()):
        si, sj, h = (np.concatenate([p[k] for p in parts]) for k in range(3))
        plist.append((li, lj, on, si.astype(np.int32), sj.astype(np.int32), h.astype(float)))
    return HostBatch(arrays, plist, nat, nsh, len(mols), mat, nao_tot)


class DeviceBatch:
    """A batch on the GPU: geometry-independent tables, the pair lists and the outputs."""

    def __init__(self, hb: HostBatch, device: str = "cuda") -> None:
        self.hb = hb
        dev = torch.device(device)
        self.t = {
            k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in hb.arrays.items()
        }
        self.t["sh_se"] = torch.zeros(hb.nsh, dtype=torch.float64, device=dev)
        self.t["sh_dsedcn"] = torch.zeros(hb.nsh, dtype=torch.float64, device=dev)
        self.pairs_t = [
            (li, lj, on, *(torch.from_numpy(a).to(dev) for a in (si, sj, h)))
            for li, lj, on, si, sj, h in hb.pairs
        ]
        self.pairs_c = (KsPairs * len(self.pairs_t))(
            *[
                KsPairs(si.data_ptr(), sj.data_ptr(), h.data_ptr(), len(si), li, lj, on)
                for li, lj, on, si, sj, h in self.pairs_t
            ]  # fmt: skip
        )
        mat = hb.mat_total
        f64 = {"dtype": torch.float64, "device": dev}
        self.S, self.H0 = torch.empty(mat, **f64), torch.empty(mat, **f64)
        self.D, self.Q = torch.empty(3 * mat, **f64), torch.empty(6 * mat, **f64)
        self.grad = torch.zeros(3 * hb.nat, **f64)
        self.dedcn = torch.zeros(hb.nat, **f64)

    def c_batch(self) -> KsBatch:
        t = self.t
        b = KsBatch(*[_ptr(t[name]) for name, _ in KsBatch._fields_[:16]])
        b.nat, b.nsh, b.nmol = self.hb.nat, self.hb.nsh, self.hb.nmol
        return b

    def set_xyz(self, xyz: torch.Tensor | np.ndarray) -> None:
        src = xyz if torch.is_tensor(xyz) else torch.as_tensor(np.asarray(xyz, dtype=float))
        self.t["xyz"].copy_(src.reshape(-1))

    def forward(self) -> None:
        out = KsFwdOut(*map(_ptr, (self.S, self.H0, self.D, self.Q)))
        stream = torch.cuda.current_stream().cuda_stream
        b = self.c_batch()
        code = lib().ksxtb_forward(
            C.byref(b), self.pairs_c, len(self.pairs_t), C.byref(out), stream
        )
        _check(code, "ksxtb_forward")

    def h0_gradient(self, P, X, vao, vdp, vqp) -> None:
        """Accumulate into self.grad / self.dedcn (zeroed here)."""
        self.grad.zero_()
        self.dedcn.zero_()
        gin = KsGradIn(*map(_ptr, (P, X, vao, vdp, vqp)))
        gout = KsGradOut(_ptr(self.grad), _ptr(self.dedcn))
        stream = torch.cuda.current_stream().cuda_stream
        b = self.c_batch()
        code = lib().ksxtb_h0_gradient(
            C.byref(b), self.pairs_c, len(self.pairs_t), C.byref(gin), C.byref(gout), stream
        )
        _check(code, "ksxtb_h0_gradient")

    def matrix(self, flat: torch.Tensor, m: int, ncomp: int = 1) -> np.ndarray:
        """Molecule m's block of a packed matrix, (ncomp, nao, nao) on the host."""
        off, nao = int(self.hb.arrays["mol_mat"][m]), int(self.hb.arrays["mol_nao"][m])
        blk = flat[ncomp * off : ncomp * off + ncomp * nao * nao].reshape(ncomp, nao, nao)
        return blk.cpu().numpy()
