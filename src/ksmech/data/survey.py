"""The open-shell survey (open-shell plan P0): a read-only subclass for every step a domain rule
excludes and for every kept step with a suspicious metal, plus the SPIN_PARITY QM failures by
kind. `ksmech-data open-shell-survey --source S` writes `open_shell_survey.json`; nothing else
is touched.

The subclasses are survey labels, never domain codes: the classifier functions move into
`data.domain` and `data.spin` in later phases (P0a, P1, P3, P4), and this report is their test
oracle. Radical drawings, first match wins, drawing errors before genuine radicals:

- C3_METHYL: a C with 3 radical electrons and one neighbour (a methyl written `[C]`);
- CDI_IMIDAZOLYL: a C in a 5-ring, degree 2, both neighbours N, one radical electron (the
  imidazol-2-yl of CDI missing its H; tested by topology: RDKit does not flag it aromatic);
- SN_PLACEHOLDER, AL_DRAWING: Sn or Al drawn with radical electrons;
- BARE_ALKALI: a neutral alkali atom with no neighbour;
- BETA_LACTAM_N: an N radical in a 4-ring; NITRONE_DRAWING: a cationic O radical;
  P_FOUR_BONDS: a P radical with four neighbours;
- AMBIGUOUS_KETYL: a C radical bonded to O in a step with an alkali metal (the 'isopropoxide'
  drawings: possibly sodium ketyls, possibly a missing H);
- CO_DRAWN: [C]=O; CARBENE: any other C with two radical electrons;
- RADICAL_REACTIVE / RADICAL_SPECTATOR: the rest, by whether an arrow touches a radical atom.

A radical drawing gets its subclass whatever P3's open-shell laws made of it: a step coded
RADICAL_DRAWN or with one of the laws' codes (`domain.OPEN_SHELL_CODES`), and an admitted one
(code KEPT, class RADICAL_SPECTATOR or RADICAL_REACTIVE), so the subclasses do not move when
the admission does.

A step with an odd metal fragment gets its metal subclass whatever P4's admission made of it:
coded OPEN_SHELL_METAL, one of the laws' codes, or admitted (KEPT, class OSM_CU2 or OSM_FE3).

Metals: OSM_FE3, OSM_CU2, CUH2_LIGAND, CU_SUSPECT (Cu=C, CuI2), PD_DATIVE (an odd Pd fragment
with an uncompensated onium), PD_ALREADY_NEG (a step holding an odd Pd already at -1), L1PD
([Pd] bonded only to an onium P, on a METAL_PLACEHOLDER step), PLACEHOLDER; on kept steps
PLACEHOLDER_EVEN (a shape the placeholder law `domain.is_placeholder` excludes: none after P0a;
P0 counted its own shape here, a non-s-block metal with at most one neighbour of either charge,
24 PMechDB steps), ZNCL_ION, ML1 (a metal whose one neighbour is an onium donor it compensates,
the law's exemption: L1Pd(0)), ONIUM_PD_NI_EVEN and CR_IV. Each metal's oxidation state and d count
are `data.domain.oxidation_state` and `d_count` (moved there in P0a). SPIN_PARITY failures are
named by what they hold (a placeholder shape, else the first open-shell-capable metal).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, rdBase

from ksmech.data import chem
from ksmech.data.domain import (
    _S_BLOCK,
    ADMITTED_METAL,
    ADMITTED_OPEN_SHELL,
    METALS,
    OPEN_SHELL_CODES,
    d_count,
    domain_classes,
    is_odd,
    is_onium_donor,
    is_placeholder,
    oxidation_state,
)

ALKALI = frozenset([3, 11, 19, 37, 55])


def placeholder_shape(frag: Chem.Mol) -> bool:
    """The P0 survey's placeholder shape: a non-s-block metal with at most one neighbour (H
    included), whatever its charge. A survey label; the domain law is `domain.is_placeholder`."""
    return any(
        a.GetAtomicNum() in METALS
        and a.GetAtomicNum() not in _S_BLOCK
        and a.GetDegree() + a.GetTotalNumHs() <= 1
        for a in frag.GetAtoms()
    )


def _radical(atom: Chem.Atom) -> int:
    return atom.GetNumRadicalElectrons()


def _is_cdi(atom: Chem.Atom) -> bool:
    ring = atom.IsInRingSize(5) and atom.GetDegree() == 2
    return ring and all(n.GetSymbol() == "N" for n in atom.GetNeighbors())


def _drawing_error(a: Chem.Atom, alkali: bool) -> str | None:
    """The drawing-error subclass one radical atom shows, None for a plausible radical."""
    sym, r, deg = a.GetSymbol(), _radical(a), a.GetDegree()
    rules = (
        (sym == "C" and r == 3 and deg == 1, "C3_METHYL"),
        (sym == "C" and r == 1 and _is_cdi(a), "CDI_IMIDAZOLYL"),
        (sym == "Sn", "SN_PLACEHOLDER"),
        (sym == "Al", "AL_DRAWING"),
        (a.GetAtomicNum() in ALKALI and deg == 0 and a.GetFormalCharge() == 0, "BARE_ALKALI"),
        (sym == "N" and a.IsInRingSize(4), "BETA_LACTAM_N"),
        (sym == "O" and a.GetFormalCharge() > 0 and a.GetTotalNumHs() > 0, "NITRONE_DRAWING"),
        (sym == "P" and deg == 4, "P_FOUR_BONDS"),
        (sym == "C" and r == 1 and alkali and any(n.GetSymbol() == "O" for n in a.GetNeighbors()),
         "AMBIGUOUS_KETYL"),
    )  # fmt: skip
    return next((name for hit, name in rules if hit), None)


def _carbene(a: Chem.Atom) -> str | None:
    if a.GetSymbol() != "C" or _radical(a) != 2:
        return None
    co = a.GetDegree() == 1 and a.GetNeighbors()[0].GetSymbol() == "O"
    return "CO_DRAWN" if co else "CARBENE"


def classify_radical(sides: list[Chem.Mol], arrow_maps: set[int]) -> tuple[str, list[int]]:
    """(subclass, map numbers of the radical atoms the arrows touch) of a RADICAL_DRAWN step."""
    atoms = [a for mol in sides for a in mol.GetAtoms() if _radical(a)]
    alkali = any(a.GetAtomicNum() in ALKALI for mol in sides for a in mol.GetAtoms())
    touched = sorted({a.GetAtomMapNum() for a in atoms} & arrow_maps)
    for rule in (lambda a: _drawing_error(a, alkali), _carbene):
        hits = [name for a in atoms if (name := rule(a))]
        if hits:
            return hits[0], touched
    return ("RADICAL_REACTIVE" if touched else "RADICAL_SPECTATOR"), touched


def _odd_metal_class(frag: Chem.Mol) -> str | None:
    for a in frag.GetAtoms():
        sym = a.GetSymbol()
        if sym == "Fe":
            return "OSM_FE3"
        if sym == "Cu":
            iodides = sum(n.GetSymbol() == "I" for n in a.GetNeighbors())
            double_c = any(b.GetBondTypeAsDouble() == 2 for b in a.GetBonds())
            if iodides >= 2 or double_c:
                return "CU_SUSPECT"
            return "CUH2_LIGAND" if a.GetTotalNumHs() >= 2 else "OSM_CU2"
        if sym == "Pd":
            if a.GetDegree() == 1 and a.GetNeighbors()[0].GetFormalCharge() > 0:
                return "L1PD"
            return "PD_ALREADY_NEG" if a.GetFormalCharge() < 0 else "PD_DATIVE"
    return None


METAL_ORDER = ("PD_ALREADY_NEG", "CU_SUSPECT", "CUH2_LIGAND", "OSM_CU2", "OSM_FE3", "PD_DATIVE")


def _odd_metal(sides: list[Chem.Mol]) -> bool:
    """Some fragment of either side is odd and holds a metal `_odd_metal_class` names."""
    frags = (f for m in sides for f in Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False))
    return any(is_odd(f) and _odd_metal_class(f) for f in frags)


def _survey_code(step, sides: list[Chem.Mol]) -> str | None:
    """'metal', 'radical' or 'kept': which classifier reads the step (module docstring)."""
    code, classes = step.domain_exclusion, domain_classes(step.domain_class)
    metal = classes & {str(c) for c in ADMITTED_METAL}
    if code in ("OPEN_SHELL_METAL", "METAL_PLACEHOLDER") or (code is None and metal):
        return "metal"
    if code in OPEN_SHELL_CODES and _odd_metal(sides):
        return "metal"
    radical = classes & {str(c) for c in ADMITTED_OPEN_SHELL}
    if code == "RADICAL_DRAWN" or code in OPEN_SHELL_CODES or (code is None and radical):
        return "radical"
    return "kept" if code is None else None


def classify_metal(sides: list[Chem.Mol], code: str) -> str:
    """Subclass of an OPEN_SHELL_METAL or METAL_PLACEHOLDER step from the subclasses of all its
    odd fragments, on both sides: a step that holds a Pd already at -1 anywhere is
    PD_ALREADY_NEG, and a placeholder step is L1PD when it holds the [Pd]-P+ drawing."""
    found = set()
    for mol in sides:
        for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
            if is_odd(frag) and (name := _odd_metal_class(frag)):
                found.add(name)
    if code == "METAL_PLACEHOLDER":
        return "L1PD" if "L1PD" in found else "PLACEHOLDER"
    return next((name for name in METAL_ORDER if name in found), "OTHER")


def _is_ml1(atom: Chem.Atom) -> bool:
    """The placeholder law's exemption: one neighbour (H counted), an onium donor the metal
    compensates with its negative charge."""
    one = atom.GetDegree() == 1 and atom.GetTotalNumHs() == 0
    return one and atom.GetFormalCharge() < 0 and is_onium_donor(atom.GetNeighbors()[0])


def _kept_metal_class(frag: Chem.Mol) -> str | None:
    for a in frag.GetAtoms():
        if a.GetAtomicNum() not in METALS or a.GetAtomicNum() in _S_BLOCK:
            continue
        if a.GetSymbol() == "Zn" and a.GetFormalCharge() == 1 and a.GetDegree() == 1:
            return "ZNCL_ION"
        if is_placeholder(frag):
            return "PLACEHOLDER_EVEN"
        if _is_ml1(a):
            return "ML1"
        oniums = [n for n in a.GetNeighbors() if n.GetFormalCharge() > 0 and n.GetSymbol() != "C"]
        if a.GetSymbol() in ("Pd", "Ni") and oniums:
            return "ONIUM_PD_NI_EVEN"
        if a.GetSymbol() == "Cr" and oxidation_state(a) == 4:
            return "CR_IV"
    return None


def classify_kept(sides: list[Chem.Mol]) -> str | None:
    """Subclass of a kept step with a suspicious metal in an even fragment, None otherwise."""
    for mol in sides:
        for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
            if not is_odd(frag) and (name := _kept_metal_class(frag)):
                return name
    return None


def spin_parity_kind(smiles: str) -> str:
    """What a SPIN_PARITY QM failure holds: a placeholder metal, else its first open-shell
    capable metal, else P (an odd P drawing), else OTHER."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "UNPARSED"
    if placeholder_shape(mol):
        return "placeholder"
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    for sym in ("Cu", "Fe", "Pd", "Ni", "Cr", "Ti", "Sc", "Zn", "P"):
        if sym in symbols:
            return sym
    return "OTHER"


def _metals(sides: list[Chem.Mol]) -> list[dict]:
    seen = []
    for mol in sides:
        for a in mol.GetAtoms():
            if a.GetAtomicNum() in METALS and a.GetAtomicNum() not in _S_BLOCK:
                seen.append({"metal": a.GetSymbol(), "os": oxidation_state(a), "d": d_count(a)})
    return seen


def survey_row(step) -> dict | None:
    """The survey row of one step, None for a kept step with nothing to report."""
    sides = [chem.parse_mapped(step.reactants_smiles), chem.parse_mapped(step.products_smiles)]
    code = step.domain_exclusion
    arrow_maps = {m for arrow in step.arrows for m in arrow.atoms}
    touched: list[int] = []
    reader = _survey_code(step, sides)
    if reader == "radical":
        sub, touched = classify_radical(sides, arrow_maps)
    elif reader == "metal":
        sub = classify_metal(sides, code or "KEPT")
    elif reader == "kept":
        sub = classify_kept(sides)
        if sub is None:
            return None
    else:
        sub = "OTHER"
    row = {"step_id": step.step_id, "code": code or "KEPT", "subclass": sub}
    row |= {"resonance_step": step.resonance_step, "radical_maps_touched": touched}
    return row | {
        "metals": _metals(sides),
        "reaction": f"{step.reactants_smiles}>>{step.products_smiles}",
    }


def _own_method(smiles: str) -> str:
    """The species' current QM method (`qm.spin.method_for`): a manifest row under another one
    (the untagged SPIN_PARITY row of a Cu(II) species P4 computes tagged) is history."""
    from ksmech.qm.spin import method_for
    from ksmech.qm.xtb_calc import QMConfig

    return method_for(smiles, QMConfig().method)


def survey(steps, manifest=None) -> dict:
    """Counts by code and subclass, the rows, and the SPIN_PARITY kinds of the QM manifest
    (each species under its current method)."""
    rows = [r for s in steps if (r := survey_row(s)) is not None]
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        counts[r["code"]][r["subclass"]] += 1
    out = {"counts": {code: dict(c.most_common()) for code, c in sorted(counts.items())}}
    out["resonance_steps"] = dict(Counter(r["code"] for r in rows if r["resonance_step"]))
    if manifest is not None:
        failed = manifest[(manifest["status"] == "failed") & (manifest["reason"] == "SPIN_PARITY")]
        rows = zip(failed["smiles"], failed["method"], strict=True)
        failed = failed[[m == _own_method(smiles) for smiles, m in rows]]
        kinds = Counter(spin_parity_kind(s) for s in failed["smiles"].unique())
        out["spin_parity_molecules"] = dict(kinds.most_common())
    radicals = [r for r in rows if r["subclass"] in ("RADICAL_SPECTATOR", "RADICAL_REACTIVE")]
    out["genuine_radical_candidates"] = [r["step_id"] for r in radicals]
    return out | {"rows": rows}


def write_survey(source: str, processed_dir: Path) -> tuple[Path, dict]:
    """`open_shell_survey.json` beside the source's steps, stamped with its inputs."""
    import pandas as pd

    from ksmech.data import provenance as prov
    from ksmech.data.io import read_steps, source_dir, steps_path

    parquet = steps_path(source, processed_dir)
    manifest_path = source_dir(source, processed_dir) / "qm_manifest.parquet"
    manifest = pd.read_parquet(manifest_path) if manifest_path.exists() else None
    with rdBase.BlockLogs():
        result = survey(read_steps(parquet), manifest)
    inputs = prov.file_input(parquet)
    if manifest is not None:
        inputs |= prov.file_input(manifest_path)
    out = source_dir(source, processed_dir) / "open_shell_survey.json"
    prov.write_json(out, result, prov.stamp("open_shell_survey", 1, inputs, {"source": source}))
    return out, result
