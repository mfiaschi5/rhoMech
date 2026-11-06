"""Open-shell plan P8 step 1: what FlowER's files hold, measured (read-only, streamed).

    uv run python scripts/probes/flower_inspect.py [--out FILE]

Over data/raw/flower/flower_new_dataset/{train,val,test}.txt: the line count, the lines whose
'|'-suffix is numeric, the count of each non-numeric suffix per file, and for the RS/RC lines:
unmapped atoms, heavy atoms with implicit H, the bath gases ([Ar], [He]: their lines by tag and
whether each is one unbonded mapped atom on both sides), the drawings of O2, CO, ozone and N2
(map-free canonical SMILES of a fragment, by tag), and the lines that carry each metal, by tag.
Nothing is written but the report.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

RAW = Path("data/raw/flower/flower_new_dataset")
FILES = ("train", "val", "test")
TAGS = ("RS", "RC")
NOBLE = ("Ar", "He")
METALS = ("Li", "Na", "K", "Mg", "Ca", "Sn", "Si", "B", "Al", "Zn", "Cu", "Fe", "Ti", "Pd", "Ni")
GASES = {  # map-free canonical SMILES of a fragment -> what it is
    "[O][O]": "O2",
    "O=O": "O2",
    "[C]=O": "CO",
    "[C-]#[O+]": "CO",
    "[O]O[O]": "ozone",
    "[O-][O+]=O": "ozone",
    "[O][O+][O-]": "ozone",
    "[N]=[N]": "N2",
    "N#N": "N2",
}


def _parse(side: str) -> Chem.Mol | None:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    return Chem.MolFromSmiles(side, params)


def _line_facts(rxn: str, tag: str, out: dict) -> None:
    sides = [_parse(s) for s in rxn.split(">>")]
    if any(m is None for m in sides):
        out["unparsable"][tag] += 1
        return
    for mol in sides:
        out["unmapped_atoms"] += sum(a.GetAtomMapNum() == 0 for a in mol.GetAtoms())
        out["heavy_with_implicit_h"] += sum(
            a.GetAtomicNum() > 1 and a.GetNumImplicitHs() > 0 for a in mol.GetAtoms()
        )
    symbols = {a.GetSymbol() for a in sides[0].GetAtoms()}
    for el in NOBLE:
        if el in symbols:
            out["noble"][f"{el}/{tag}"] += 1
            alone = all(
                all(a.GetDegree() == 0 for a in m.GetAtoms() if a.GetSymbol() == el)
                and sum(a.GetSymbol() == el for a in m.GetAtoms()) == 1
                for m in sides
            )
            out["noble_unbonded_one_atom"] += int(alone)
    for el in METALS:
        if el in symbols:
            out["metal_lines"][f"{el}/{tag}"] += 1
    seen = set()
    for mol in sides:
        for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
            m = Chem.Mol(frag)
            for a in m.GetAtoms():
                a.SetAtomMapNum(0)
            smi = Chem.MolToSmiles(Chem.RemoveHs(m, sanitize=False))
            smi = Chem.CanonSmiles(smi) if Chem.MolFromSmiles(smi) else smi
            if smi in GASES and smi not in seen:
                seen.add(smi)
                out["gases"][f"{GASES[smi]} {smi} /{tag}"] += 1


def inspect(raw: Path = RAW) -> dict:
    RDLogger.DisableLog("rdApp.*")
    out: dict = {
        "lines": Counter(),
        "numeric_suffix": Counter(),
        "suffix": defaultdict(Counter),
        "unparsable": Counter(),
        "unmapped_atoms": 0,
        "heavy_with_implicit_h": 0,
        "noble": Counter(),
        "noble_unbonded_one_atom": 0,
        "metal_lines": Counter(),
        "gases": Counter(),
    }
    for name in FILES:
        with open(raw / f"{name}.txt") as fh:
            for line in fh:
                rxn, _, tag = line.rstrip("\n").rpartition("|")
                out["lines"][name] += 1
                if re.fullmatch(r"\d+", tag):
                    out["numeric_suffix"][name] += 1
                    continue
                out["suffix"][name][tag] += 1
                if tag in TAGS:
                    _line_facts(rxn, tag, out)
    return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in out.items()} | {
        "suffix": {k: dict(v) for k, v in out["suffix"].items()}
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = inspect()
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
