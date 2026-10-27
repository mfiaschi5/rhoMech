"""The page of an embedding trace (`ksmech-eval embedding`): one self-contained HTML file
(`embedding_page.html`, no external script) that draws the pool's tokens on a 2D depiction of
its Lewis structure and colours them by any stage of the trace (`eval.embedding.trace`).

Orbital types (sigma, pi, lone pair, ...) name the orbital tokens on the page. They are the
classifier's diagnostics, read from the token files, and never a model input; the page says so.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor

from ksmech.arrows.steps import MechanismSample
from ksmech.arrows.symmetry import lewis_mol
from ksmech.eval.embedding import PARTS
from ksmech.orbitals.tokens import MoleculeTokens, atom_feature_names, feature_names

TEMPLATE = Path(__file__).with_name("embedding_page.html")
TAU_ATOM = 0.15  # centre weight of an atom that names an orbital (spec, tau_atom)
ONE_CENTRE = 0.8  # an orbital with a centre this heavy is drawn beside that atom
ATTENTION_BUDGET = 8_000_000  # attention cells kept per head; above, only the head mean
TYPE_SYMBOL = {
    "lone_pair": "n",
    "sigma": "σ",
    "pi": "π",
    "delocalized_pi": "π(deloc)",
    "sigma_star": "σ*",
    "pi_star": "π*",
    "empty_p": "p(empty)",
    "empty_s": "s(empty)",
    "empty_d": "d(empty)",
    "delocalized_pi_star": "π*(deloc)",
    "unknown": "?",
}


def orbital_types(tokens: list[MoleculeTokens]) -> list[str]:
    """The diagnostic type of every orbital token, in sample order (fragment by fragment)."""
    return [str(o.type) for t in tokens for o in t.orbitals]


def orbital_centres(sample: MechanismSample) -> list[list[tuple[int, float]]]:
    """Per orbital token, its centre atoms (global index, weight), heaviest first."""
    out: list[list[tuple[int, float]]] = [[] for _ in range(sample.n_orbitals)]
    for j, g, w in sample.incidence:
        out[int(j)].append((int(g), round(float(w), 3)))
    return [sorted(c, key=lambda x: -x[1]) for c in out]


SUPERSCRIPT = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


def _charge(q: int) -> str:
    """A formal charge as a superscript: "", ⁺, ²⁻, ..."""
    if q == 0:
        return ""
    return f"{abs(q) if abs(q) > 1 else ''}{'+' if q > 0 else '-'}".translate(SUPERSCRIPT)


def token_rows(sample: MechanismSample, types: list[str] | None) -> list[dict]:
    """One row per token on the MECH layout: name, kind (mech, atom, occupied, virtual), the
    element of an atom, the diagnostic type and the centres of an orbital, the fragment."""
    pt = Chem.GetPeriodicTable()
    z, q = sample.atom_categorical[:, 0], sample.atom_categorical[:, 1]
    symbols = [pt.GetElementSymbol(int(x)) for x in z]
    rows = [{"name": "MECH", "kind": "mech", "frag": -1}]
    for g, sym in enumerate(symbols):
        name = f"{sym}{g}{_charge(int(q[g]))}"
        frag = int(sample.atom_fragment[g])
        rows.append({"name": name, "kind": "atom", "el": sym, "q": int(q[g]), "frag": frag})
    for j, centres in enumerate(orbital_centres(sample)):
        lead = [g for g, w in centres if w >= TAU_ATOM][:2] or [centres[0][0]]
        occupied = bool(sample.orbital_occupied[j])
        kind = "occupied" if occupied else "virtual"
        label = TYPE_SYMBOL.get(types[j], types[j]) if types else ("occ" if occupied else "virt")
        rows.append({
            "name": f"{label} {'-'.join(f'{symbols[g]}{g}' for g in lead)}",
            "kind": kind,
            "type": types[j] if types else None,
            "frag": int(sample.orbital_fragment[j]),
            "centres": centres,
        })  # fmt: skip
    return rows


def _spread(points: np.ndarray, grid: float = 0.15, radius: float = 0.24) -> np.ndarray:
    """Points that fall on one spot (within `grid`) moved apart on a small ring."""
    out = points.copy()
    spots: dict[tuple, list[int]] = {}
    for k, p in enumerate(points):
        spots.setdefault(tuple(np.round(p / grid).astype(int)), []).append(k)
    for members in spots.values():
        for t, k in enumerate(members if len(members) > 1 else []):
            angle = 2 * np.pi * t / len(members) + np.pi / 4
            out[k] = points[k] + radius * np.array([np.cos(angle), np.sin(angle)])
    return out


def layout(sample: MechanismSample, centres: list) -> np.ndarray:
    """(A + O, 2) drawing positions: RDKit 2D coordinates of the Lewis structure for the atoms;
    an orbital at the weighted mean of its centres, a one-centre orbital pushed out of its atom
    away from its neighbours, orbitals on one spot spread on a small ring."""
    z, q = sample.atom_categorical[:, 0], sample.atom_categorical[:, 1]
    mol = lewis_mol(z, q, sample.bonds)
    rdDepictor.Compute2DCoords(mol)
    xy = mol.GetConformer().GetPositions()[:, :2]
    nbrs: dict[int, list[int]] = {a: [] for a in range(len(z))}
    for i, j, _ in sample.bonds:
        nbrs[int(i)].append(int(j))
        nbrs[int(j)].append(int(i))
    orb = []
    for c in centres:
        g = np.array([a for a, _ in c])
        w = np.array([x for _, x in c])
        if w[0] >= ONE_CENTRE * w.sum():
            a = int(g[0])
            out = xy[a] - xy[nbrs[a]].mean(0) if nbrs[a] else np.array([0.0, 1.0])
            orb.append(xy[a] + 0.55 * out / max(np.linalg.norm(out), 1e-6))
        else:
            orb.append((w[:, None] * xy[g]).sum(0) / w.sum())
    return np.vstack([xy, _spread(np.array(orb).reshape(-1, 2))])


def pca_path(h: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """(L + 1, N + 1, 2) the first two principal components of the token vectors after every
    layer, each layer's signs turned to agree with the layer before; the share of the variance
    they explain per layer."""
    coords, shares = [], []
    for x in h.astype(np.float64):
        xc = x - x.mean(0)
        _, s, vt = np.linalg.svd(xc, full_matrices=False)
        c = xc @ vt[:2].T
        if coords:
            c *= np.where((c * coords[-1]).sum(0) < 0, -1.0, 1.0)
        coords.append(c)
        shares.append(float((s[:2] ** 2).sum() / max((s**2).sum(), 1e-12)))
    return np.stack(coords), shares


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _r(a, n: int = 4):
    return np.round(np.asarray(a, dtype=np.float64), n).tolist()


def _attention(attn: np.ndarray) -> dict:
    """Attention as bytes, sqrt-scaled (p = (b / 255)^2, so small weights keep resolution);
    every head when it fits ATTENTION_BUDGET, else only the head mean."""
    per_head = attn.size <= ATTENTION_BUDGET
    a = attn if per_head else attn.mean(1, keepdims=True)
    q = np.round(np.sqrt(np.clip(a, 0.0, 1.0)) * 255).astype(np.uint8)
    return {"shape": list(q.shape), "per_head": per_head, "data": _b64(q)}


def _features(sample: MechanismSample, tr: dict) -> dict:
    """The raw and the standardized model inputs of every atom and orbital token, the missing
    flags of the atoms and the spectral sets (energy in eV, weight)."""
    n_atom = len(atom_feature_names())
    return {
        "atom_names": atom_feature_names(),
        "orbital_names": feature_names(),
        "atom_raw": _r(sample.atom_features),
        "atom_std": _r(tr["inputs"]["atom_x"][:, :n_atom], 3),
        "atom_missing": sample.atom_missing.astype(int).tolist(),
        "orbital_raw": _r(sample.orbital_features),
        "orbital_std": _r(tr["inputs"]["orbital_x"], 3),
        "spectra": [_r(s[s[:, 1] > 0], 3) for s in sample.orbital_spectra],
    }


def _parts(tr: dict) -> dict:
    """Per token, the norm of every part of h^(0) and its cosine with h^(0)."""
    h0 = tr["h"][0]
    norm0 = np.linalg.norm(h0, axis=-1)
    norms, cos = [], []
    for k in PARTS:
        v = tr["parts"][k]
        n = np.linalg.norm(v, axis=-1)
        norms.append(n)
        cos.append((v * h0).sum(-1) / np.maximum(n * norm0, 1e-12))
    return {"names": list(PARTS), "norm": _r(np.stack(norms, 1), 3), "cos": _r(np.stack(cos, 1), 3)}


def _label(sample: MechanismSample) -> list[list]:
    """The variants of the sample's label class as R entries, [] without one."""
    if sample.edit_label is None:
        return []
    out = []
    for v in sample.edit_label.variants:
        bonds = [["bond", int(i), int(j), int(d)] for i, j, d in v.bonds]
        out.append(bonds + [["lone", int(i), int(d)] for i, d in v.lone])
    return out


def page_payload(tr: dict, sample: MechanismSample, types: list[str] | None, meta: dict) -> dict:
    """Everything the page draws, as JSON-ready values."""
    rows = token_rows(sample, types)
    xy = layout(sample, [r["centres"] for r in rows if "centres" in r])
    for row, p in zip(rows[1:], xy, strict=True):
        row["x"], row["y"] = round(float(p[0]), 3), round(float(-p[1]), 3)
    coords, shares = pca_path(tr["h"])
    n_layers, n, d = tr["h"].shape
    return {
        "meta": meta | {"errors": tr["errors"], "n_atoms": sample.n_atoms, "d": d},
        "tokens": rows,
        "bonds": sample.bonds.astype(int).tolist(),
        "lone_pairs": sample.lone_pairs.astype(int).tolist(),
        "features": _features(sample, tr),
        "parts": _parts(tr),
        "layers": {
            "h": {"shape": [n_layers, n, d], "data": _b64(tr["h"].astype(np.float32))},
            "pca": _r(coords, 3),
            "pca_share": _r(shares, 3),
            "d_attn": _r(tr["d_attn"], 3),
            "d_ffn": _r(tr["d_ffn"], 3),
            "spread": _r(tr["spread"], 3),
            "attention": _attention(tr["attn"]),
        },
        "readout": tr["readout"] | {"label": _label(sample)},
    }


def write_page(payload: dict, out: Path) -> Path:
    """The page with `payload` inlined (no file beside it)."""
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*PAYLOAD*/null", data)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
