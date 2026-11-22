"""Stage 2 on the GPU: the GFN2 terms beyond the one-electron integrals for a batch.

`ClassicalBatch` packs the per-atom parameter tables, the pair lists and the D4 reference C6 of
a batch (`native/ksxtb/src/classical.cu`, layouts in `ksxtb.h`); `geometry()` computes the
coordination numbers, multipole radii and Born radii of the current geometry, `evaluate()` the
energies, potentials and gradient at given SCF variables. The values follow
`ksmech.qm.gpu.terms`, which is exact against tblite.
"""

from __future__ import annotations

import ctypes as C
import re
from functools import cache

import numpy as np
import torch

from ksmech.qm.gpu import native
from ksmech.qm.gpu import terms as T
from ksmech.qm.gpu.basis import mol_basis, params, scalar

HEADER = native.ROOT / "include" / "ksxtb.h"
P = C.c_void_p


@cache
def enums() -> dict[str, int]:
    """Every enumerator of ksxtb.h, so the Python side reads the C layouts, not a copy."""
    out: dict[str, int] = {}
    text = re.sub(r"/\*.*?\*/", "", HEADER.read_text(), flags=re.DOTALL)
    for body in re.findall(r"enum\s*\{(.*?)\}", text, flags=re.DOTALL):
        nxt = 0
        for item in (s.strip() for s in body.split(",")):
            if not item:
                continue
            name, _, expr = item.partition("=")
            nxt = eval(expr, {}, dict(out)) if expr.strip() else nxt  # noqa: S307 - our header
            out[name.strip()] = int(nxt)
            nxt += 1
    return out


def _e(name: str) -> int:
    return enums()[name]


class KsClassical(C.Structure):
    pass


KsClassical._fields_ = [
    (n, P)
    for n in ("xyz", "atom_mol", "atab", "itab", "sh_atom", "sh_eta", "sh_hd", "pi", "pj",
              "mol_at0", "mol_nat", "mol_pair0", "c6ref", "grid", "gweight")
] + [("sc", C.c_double * _e("KS_NSC"))] + [
    (n, C.c_int) for n in ("nat", "nsh", "nmol", "npair", "nsp", "solvent")
]  # fmt: skip

WORK = ("cn", "cn4", "mrad", "dmrdcn", "psi", "brad", "dbrdpsi", "gw", "dgwdcn", "dgwdq", "gw0",
        "dgw0dcn", "c60", "dedcn", "dedcn4", "dedbr", "surface", "eatom", "egeo")  # fmt: skip


class KsWork(C.Structure):
    _fields_ = [(n, P) for n in WORK]


class KsScf(C.Structure):
    _fields_ = [(n, P) for n in ("qsh", "qat", "dp", "qp")]


class KsClsOut(C.Structure):
    _fields_ = [(n, P) for n in ("energy", "vsh", "vat", "vdp", "vqp", "grad")]


TERMS = ("repulsion", "es2", "es3", "aes", "d4_two", "d4_three", "alpb", "cds")


@cache
def _lib() -> C.CDLL:
    so = native.lib()
    so.ksxtb_geometry.argtypes = [C.POINTER(KsClassical), C.POINTER(KsWork), P]
    so.ksxtb_classical.argtypes = [C.POINTER(KsClassical), C.POINTER(KsWork), C.POINTER(KsScf),
                                   C.POINTER(KsClsOut), P, C.c_int, P]  # fmt: skip
    so.ksxtb_geometry.restype = so.ksxtb_classical.restype = C.c_int
    return so


def _scalars(solvent: bool) -> np.ndarray:
    assert scalar("gexp") == 2.0, "the ES2 kernel assumes gexp = 2"
    keps, alpbet = T._alpb_consts()
    vals = {
        "KS_REP_KEXP": scalar("rep_kexp"), "KS_REP_KEXP_LIGHT": scalar("rep_kexp_light"),
        "KS_REP_REXP": scalar("rep_rexp"), "KS_MP_DMP3": scalar("mp_dmp3"),
        "KS_MP_DMP5": scalar("mp_dmp5"), "KS_MP_SHIFT": scalar("mp_shift"),
        "KS_MP_KEXP": scalar("mp_kexp"), "KS_MP_RMAX": scalar("mp_rmax"),
        "KS_D4_S6": scalar("s6"), "KS_D4_S8": scalar("s8"), "KS_D4_A1": scalar("a1"),
        "KS_D4_A2": scalar("a2"), "KS_D4_S9": scalar("s9"), "KS_D4_GA": T.D4_GA,
        "KS_D4_WF": T.D4_WF, "KS_KEPS": keps, "KS_ALPBET": alpbet,
        "KS_BORN_C1": float(T._solv("c1")[0]), "KS_SASA_W": T.SASA_SMOOTH,
    }  # fmt: skip
    sc = np.zeros(_e("KS_NSC"))
    for k, v in vals.items():
        sc[_e(k)] = v
    return sc


def _atom_tables(z: np.ndarray, species: dict[int, int], sh0: np.ndarray, nsh: np.ndarray):
    g, d4 = params()["gfn2"], params()["dftd4"]
    a = np.zeros((len(z), _e("KA_COLS")))
    i = np.zeros((len(z), _e("KI_COLS")), dtype=np.int32)
    rvdw = T._solv("vdw_rad_d3")[z - 1]
    vdwsa, wrp, _, _ = T._sasa_consts(z)
    probe = float(T._solv("rprobe")[0]) * T.AATOAU
    cols = {
        "KA_RCOV": params()["covalent_rad"]["covalent_rad_d3"][z - 1],
        "KA_RCOV4": d4["covalent_rad_d3"][z - 1], "KA_EN4": d4["pauling_en"][z - 1],
        "KA_REP_ALPHA": g["rep_alpha"][z - 1], "KA_REP_ZEFF": g["rep_zeff"][z - 1],
        "KA_MP_RAD": g["p_rad"][z - 1], "KA_MP_VCN": g["p_vcn"][z - 1],
        "KA_DKERNEL": g["p_dkernel"][z - 1], "KA_QKERNEL": g["p_qkernel"][z - 1],
        "KA_R4R2": T._r4r2(z).numpy(), "KA_D4_ZI": d4["effective_nuclear_charge"][z - 1],
        "KA_D4_GI": d4["chemical_hardness"][z - 1] * T.D4_GC, "KA_RVDW": rvdw,
        "KA_RHO": rvdw * T._solv("sx")[z - 1],
        "KA_SVDW": rvdw - float(T._solv("soset")[0]) * 0.1 * T.AATOAU,
        "KA_VDWSA": vdwsa, "KA_WRP": wrp, "KA_TENSION": T._solv("gamscale")[z - 1] * 1e-5,
        "KA_HBOND": -T.KCALTOAU * T._solv("sqrtGhbond")[z - 1] ** 2
        / (4 * np.pi * (rvdw + probe) ** 2),
    }  # fmt: skip
    for k, v in cols.items():
        a[:, _e(k)] = v
    for row, zz in enumerate(z.tolist()):
        el = T.d4_element(zz)
        a[row, _e("KA_REFCN") : _e("KA_REFCN") + el.nref] = el.refcn
        a[row, _e("KA_REFQ") : _e("KA_REFQ") + el.nref] = el.refq
        i[row, _e("KI_NGW") : _e("KI_NGW") + el.nref] = el.ngw
        i[row, _e("KI_NREF")] = el.nref
    i[:, _e("KI_Z")], i[:, _e("KI_SP")] = z, [species[zz] for zz in z.tolist()]
    i[:, _e("KI_SH0")], i[:, _e("KI_NSH")] = sh0, nsh
    return a, i


def _shell_tables(basis) -> tuple[np.ndarray, np.ndarray]:
    g = params()["gfn2"]
    ls, zs = T.shell_info(basis)
    return T._shell_hardness(basis).numpy(), g["p_hubbard_derivs"][zs - 1] * g[
        "shell_hubbard_derivs"
    ][ls]


def _c6_table(zs: list[int]) -> np.ndarray:
    out = np.zeros((len(zs), len(zs), 7, 7))
    for a, za in enumerate(zs):
        for b, zb in enumerate(zs):
            c = T.d4_c6ref(za, zb)
            out[a, b, : c.shape[0], : c.shape[1]] = c
    return out


def host_arrays(mols: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    """Pack molecules (z, xyz in Bohr) into the arrays of KsClassical."""
    zs = sorted({int(z) for zz, _ in mols for z in zz})
    species = {z: k for k, z in enumerate(zs)}
    cols: dict[str, list] = {k: [] for k in ("xyz", "atom_mol", "atab", "itab", "sh_atom",
                                             "sh_eta", "sh_hd", "pi", "pj", "mol_at0",
                                             "mol_nat", "mol_pair0")}  # fmt: skip
    nat = nsh = npair = 0
    for m, (z, xyz) in enumerate(mols):
        z = np.asarray(z, dtype=int)
        basis = mol_basis(z)
        nsh_at = np.bincount(basis.sh_atom, minlength=len(z))
        sh0 = nsh + np.concatenate([[0], np.cumsum(nsh_at)[:-1]])
        atab, itab = _atom_tables(z, species, sh0, nsh_at)
        eta, hd = _shell_tables(basis)
        il, jl = np.tril_indices(len(z), -1)  # row-major: (1,0), (2,0), (2,1), ...
        part = {"xyz": np.asarray(xyz, float).reshape(-1), "atom_mol": [m] * len(z),
                "atab": atab, "itab": itab, "sh_atom": basis.sh_atom + nat, "sh_eta": eta,
                "sh_hd": hd, "pi": il + nat, "pj": jl + nat, "mol_at0": [nat],
                "mol_nat": [len(z)], "mol_pair0": [npair]}  # fmt: skip
        for k, v in part.items():
            cols[k].append(np.asarray(v))
        nat, nsh, npair = nat + len(z), nsh + len(basis.sh_atom), npair + len(il)
    ints = {"atom_mol", "itab", "sh_atom", "pi", "pj", "mol_at0", "mol_nat"}
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out = {k: v.astype(np.int32 if k in ints else (np.int64 if k == "mol_pair0" else float))
           for k, v in out.items()}  # fmt: skip
    grid, weight = T._lebedev230()
    out.update(c6ref=_c6_table(zs).ravel(), grid=grid.ravel(), gweight=weight)
    return out


class ClassicalBatch:
    """A batch's stage-2 tables, workspace and outputs on the GPU."""

    def __init__(self, mols, solvent: bool, device: str = "cuda") -> None:
        arr = host_arrays(mols)
        dev = torch.device(device)
        self.t = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in arr.items()}
        self.nat, self.nsh, self.nmol = len(arr["atom_mol"]), len(arr["sh_atom"]), len(mols)
        self.npair, self.nsp = len(arr["pi"]), round(np.sqrt(len(arr["c6ref"]) / 49))
        self.solvent = solvent
        f64 = {"dtype": torch.float64, "device": dev}
        n, nr = self.nat, _e("KS_NREF")
        sizes = {"gw": n * nr, "dgwdcn": n * nr, "dgwdq": n * nr, "gw0": n * nr,
                 "dgw0dcn": n * nr, "c60": 3 * self.npair, "eatom": n * _e("KE_N"),
                 "egeo": n * _e("KE_N")}  # fmt: skip
        self.w = {k: torch.zeros(max(sizes.get(k, n), 1), **f64) for k in WORK}
        self.sc = _scalars(solvent)
        self.out = self.new_outputs()
        self._c = self._build_struct()
        self._w = KsWork(*[self.w[k].data_ptr() for k in WORK])

    def new_outputs(self) -> dict[str, torch.Tensor]:
        """A set of output buffers, flat as the kernels write them."""
        f64 = {"dtype": torch.float64, "device": self.t["xyz"].device}
        n = self.nat
        return {"energy": torch.zeros(self.nmol * _e("KE_N"), **f64),
                "vsh": torch.zeros(self.nsh, **f64), "vat": torch.zeros(n, **f64),
                "vdp": torch.zeros(3 * n, **f64), "vqp": torch.zeros(6 * n, **f64),
                "grad": torch.zeros(3 * n, **f64)}  # fmt: skip

    def _build_struct(self) -> KsClassical:
        names = [n for n, _ in KsClassical._fields_[:15]]
        c = KsClassical(*[self.t[k].data_ptr() for k in names])
        c.sc[:] = self.sc.tolist()
        c.nat, c.nsh, c.nmol, c.npair, c.nsp = self.nat, self.nsh, self.nmol, self.npair, self.nsp
        c.solvent = int(self.solvent)
        return c

    def set_xyz(self, xyz) -> None:
        self.t["xyz"].copy_(torch.as_tensor(np.asarray(xyz).reshape(-1)))

    def geometry(self) -> None:
        """Both CNs, multipole and Born radii, zero-charge D4 C6, ATM energy, surface areas."""
        stream = torch.cuda.current_stream().cuda_stream
        native._check(_lib().ksxtb_geometry(C.byref(self._c), C.byref(self._w), stream),
                      "ksxtb_geometry")  # fmt: skip

    def evaluate(
        self,
        qsh,
        qat,
        dp,
        qp,
        dedcn_in: torch.Tensor | None = None,
        gradient: bool = True,
        out: dict | None = None,
    ) -> dict:
        """Energies (nmol, terms), vsh, vat, vdp (nat, 3), vqp (nat, 6) and, with `gradient`,
        grad (nat, 3), into `out` (from `new_outputs`, default the batch's own buffers)."""
        out = self.out if out is None else out
        keep = [x.contiguous() for x in (qsh, qat, dp, qp)]
        s = KsScf(*[x.data_ptr() for x in keep])
        o = KsClsOut(*[out[k].data_ptr() for k, _ in KsClsOut._fields_])
        stream = torch.cuda.current_stream().cuda_stream
        ext = None if dedcn_in is None else dedcn_in.contiguous()
        ext_ptr = None if ext is None else ext.data_ptr()
        code = _lib().ksxtb_classical(C.byref(self._c), C.byref(self._w), C.byref(s), C.byref(o),
                                      ext_ptr, int(gradient), stream)  # fmt: skip
        native._check(code, "ksxtb_classical")
        return {"energy": out["energy"].view(self.nmol, -1), "vsh": out["vsh"],
                "vat": out["vat"], "vdp": out["vdp"].view(-1, 3), "vqp": out["vqp"].view(-1, 6),
                "grad": out["grad"].view(-1, 3)}  # fmt: skip


def _shell_selfenergy_tables(mols) -> tuple[np.ndarray, np.ndarray]:
    """se0 and kcn of every shell of a batch, se = se0 - kcn * CN (tblite get_selfenergy)."""
    g = params()["gfn2"]
    se0, kcn = [], []
    for z, _ in mols:
        basis = mol_basis(np.asarray(z, dtype=int))
        ls, zs = T.shell_info(basis)
        se0.append(g["p_selfenergy"][basis.sh_elem, zs - 1])
        kcn.append(g["p_kcn"][ls, zs - 1])
    return np.concatenate(se0), np.concatenate(kcn)


def _mulliken(db, mols, density: torch.Tensor):
    """Shell charges and atomic multipoles of a packed density (stage 1's matrix layout)."""
    qsh, dp, qp = [], [], []
    for m, (z, _) in enumerate(mols):
        basis = mol_basis(np.asarray(z, dtype=int))
        n, off = basis.nao, int(db.hb.arrays["mol_mat"][m])

        def blk(flat, k, off=off, n=n):
            return flat[k * off : k * off + k * n * n].view(k, n, n)

        p = density[off : off + n * n].view(n, n)
        ao_atom = torch.as_tensor(basis.ao_atom, device=p.device)
        nsh_ao = np.diff(np.append(basis.sh_ao, basis.nao))
        ao_sh = torch.as_tensor(np.repeat(np.arange(len(basis.sh_atom)), nsh_ao), device=p.device)
        pop = (p * blk(db.S, 1)[0]).sum(0)
        ref = torch.as_tensor(T.reference_occupations(basis), device=p.device)
        qsh.append(ref - torch.zeros_like(ref).index_add(0, ao_sh, pop))
        for out, mat, k in ((dp, db.D, 3), (qp, db.Q, 6)):
            per = -(p[None] * blk(mat, k)).sum(1)  # (k, nao)
            out.append(torch.zeros(k, len(z), **_like(p)).index_add(1, ao_atom, per).T)
    return torch.cat(qsh), torch.cat(dp).contiguous(), torch.cat(qp).contiguous()


def _like(t: torch.Tensor) -> dict:
    return {"dtype": t.dtype, "device": t.device}


def total_at_density(mols, densities, wdensities, solvent: bool) -> tuple[np.ndarray, np.ndarray]:
    """GFN2 energy (without the electronic entropy) and gradient of each molecule at a given
    density and energy-weighted density, everything on the GPU: the CN self-energies, stage 1's
    integrals, Mulliken charges and multipoles, the stage-2 terms with their potentials, stage
    1's H0 gradient and the CN chain of its dE/dCN."""
    db = native.DeviceBatch(native.host_batch(mols))
    cb = ClassicalBatch(mols, solvent)
    cb.geometry()
    se0, kcn = (torch.from_numpy(a).cuda() for a in _shell_selfenergy_tables(mols))
    sh_atom = cb.t["sh_atom"].long()
    db.t["sh_se"].copy_(se0 - kcn * cb.w["cn"][sh_atom])
    db.t["sh_dsedcn"].copy_(-kcn)
    db.forward()
    pack = [torch.from_numpy(np.concatenate([np.ravel(x) for x in xs])).cuda()
            for xs in (densities, wdensities)]  # fmt: skip
    qsh, dp, qp = _mulliken(db, mols, pack[0])
    qat = torch.zeros(cb.nat, **_like(qsh)).index_add(0, sh_atom, qsh)
    first = cb.evaluate(qsh, qat, dp, qp)
    vsh = first["vsh"] + first["vat"][sh_atom]
    vao = _shell_to_ao(db, vsh)
    db.h0_gradient(pack[0], pack[1], vao, first["vdp"].clone().reshape(-1),
                   first["vqp"].clone().reshape(-1))  # fmt: skip
    energy = first["energy"].sum(1).clone()
    out = cb.evaluate(qsh, qat, dp, qp, db.dedcn)
    grad = (out["grad"] + db.grad.view(-1, 3)).cpu().numpy()
    band = torch.stack([
        (pack[0][o : o + n * n] * db.H0[o : o + n * n]).sum()
        for o, n in zip(db.hb.arrays["mol_mat"].tolist(), db.hb.arrays["mol_nao"].tolist(),
                        strict=True)])  # fmt: skip
    e = (energy + band).cpu().numpy() + (T.solvent_shift() if solvent else 0.0)
    return e, grad


def _shell_to_ao(db, vsh: torch.Tensor) -> torch.Tensor:
    """Broadcast a shell vector to the AOs of each shell (stage 1's AO layout)."""
    sh_ao = db.hb.arrays["sh_ao"].astype(np.int64)
    mol_ao = db.hb.arrays["mol_ao"].astype(np.int64)
    atom_mol = db.hb.arrays["atom_mol"]
    sh_mol = atom_mol[db.hb.arrays["sh_atom"]]
    first = mol_ao[sh_mol] + sh_ao
    ends = np.append(first[1:], db.hb.ao_total)
    idx = np.repeat(np.arange(len(first)), ends - first)
    return vsh[torch.as_tensor(idx, device=vsh.device)].contiguous()
