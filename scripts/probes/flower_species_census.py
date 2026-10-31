"""The RDKit census of FlowER's radical lines (open-shell plan P2 step 6).

    uv run python scripts/probes/flower_species_census.py [--out FILE]

Read-only over data/raw/flower/flower_new_dataset/{train,val,test}.txt: the lines tagged RS or RC
(RMechDB's radical steps). Every fragment of either side becomes a species, keyed by its map-free
canonical SMILES (`chem.unmapped_canonical`), with RDKit's radical electrons u as drawn. The
report gives the distribution of u over unique species and the lines that hold each u; the atoms
and topologically linear species with u > 0 (whose open shell may be a symmetry-degenerate set,
fermi_smeared under today's occupation rule: the atoms' restricted occupations are read from
their shell counts here, the linear ones are left to the QM probe); the CO, N2, O2 and ozone
drawings with their lines by tag; and every u >= 2 species, which is the organic species list of
P2's spin scan. Fragments RDKit cannot read are counted by reason, never dropped silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

from ksmech.data import chem, domain

RAW = Path("data/raw/flower/flower_new_dataset")
FILES = ("train", "val", "test")
TAGS = ("RS", "RC")
NAMED = {"CO": "[C]=O", "N2": "[N]=[N]", "O2": "[O][O]", "ozone_chain": "[O]O[O]"}
NAMED |= {"ozone_charge_separated": "[O][O+][O-]"}


def radical_lines(raw: Path = RAW) -> list[tuple[str, str, str]]:
    """(file, tag, reaction) of every RS/RC line, in file order."""
    out = []
    for name in FILES:
        path = raw / f"{name}.txt"
        for line in path.read_text().splitlines():
            rxn, _, tag = line.rpartition("|")
            if tag in TAGS:
                out.append((name, tag, rxn))
    return out


def side_species(rxn: str) -> list[tuple[str, int, Chem.Mol]]:
    """(canonical SMILES, u, fragment) of every fragment on either side."""
    out = []
    for side in rxn.split(">>"):
        for frag in chem.fragments(chem.parse_mapped(side)):
            u = sum(a.GetNumRadicalElectrons() for a in frag.GetAtoms())
            out.append((chem.unmapped_canonical(frag), u, frag))
    return out


def is_linear(frag: Chem.Mol) -> bool:
    """One or two atoms, or a chain whose inner atoms are all sp (RDKit's hybridization)."""
    if frag.GetNumAtoms() <= 2:
        return True
    degrees = [a.GetDegree() for a in frag.GetAtoms()]
    inner = [a for a in frag.GetAtoms() if a.GetDegree() == 2]
    return max(degrees) <= 2 and all(
        a.GetHybridization() == Chem.HybridizationType.SP for a in inner
    )


def atom_open_shell(frag: Chem.Mol, u: int) -> dict | None:
    """The restricted occupations of a bare main-group atom: alpha and beta electrons are
    smeared separately over the valence p set, so a p count not divisible by 3 in either spin
    gives equal fractional occupations (Br.: 3 x 5/3). None for a molecule or a d/f element."""
    if frag.GetNumAtoms() != 1:
        return None
    z = frag.GetAtomWithIdx(0).GetAtomicNum()
    if z in domain.GROUP or 57 <= z <= 71 or 89 <= z <= 103:
        return None
    n = Chem.GetPeriodicTable().GetNOuterElecs(z) - frag.GetAtomWithIdx(0).GetFormalCharge()
    alpha, beta = (n + u) // 2, (n - u) // 2
    # the s orbital takes one electron per spin before the p set (H and He have no p set)
    p_alpha, p_beta = (0, 0) if z <= 2 else (max(alpha - 1, 0), max(beta - 1, 0))
    frac = p_alpha % 3 != 0 or p_beta % 3 != 0
    return {"valence": n, "p_alpha": p_alpha, "p_beta": p_beta, "degenerate_fractional": frac}


def _new_species(smi: str, u: int, frag: Chem.Mol) -> dict:
    return {
        "smiles": smi,
        "u": u,
        "n_atoms": frag.GetNumAtoms(),
        "heavy_atoms": chem.heavy_atom_count(frag),
        "charge": chem.formal_charge(frag),
        "linear": is_linear(frag),
        "atom_shell": atom_open_shell(frag, u),
        "lines": Counter(),
    }


def collect(lines) -> tuple[dict, Counter, Counter]:
    """Species by SMILES (with the lines that hold each, per tag), unread fragments by reason,
    and lines per tag."""
    species: dict[str, dict] = {}
    failures: Counter = Counter()
    per_tag: Counter = Counter()
    for _, tag, rxn in lines:
        per_tag[tag] += 1
        try:
            found = side_species(rxn)
        except (chem.SmilesParseError, chem.SanitizeError) as exc:
            failures[type(exc).__name__] += 1
            continue
        for smi in {s for s, _, _ in found}:
            entry = next(x for x in found if x[0] == smi)
            species.setdefault(smi, _new_species(*entry))["lines"][tag] += 1
    return species, failures, per_tag


def _lines(rows: list[dict], tag_lines: dict) -> dict:
    """Lines holding any of `rows`, per tag (a line counted once however many it holds)."""
    smis = {r["smiles"] for r in rows}
    out = Counter()
    for tag, held in tag_lines.items():
        out[tag] = sum(1 for line in held if line & smis)
    return dict(out)


def line_species(lines) -> dict[str, list[set[str]]]:
    """Per tag, the set of species SMILES of each line."""
    out = defaultdict(list)
    for _, tag, rxn in lines:
        try:
            out[tag].append({s for s, _, _ in side_species(rxn)})
        except (chem.SmilesParseError, chem.SanitizeError):
            out[tag].append(set())
    return out


def _row(r: dict) -> dict:
    return {k: (dict(v) if k == "lines" else v) for k, v in r.items()}


def report(lines) -> dict:
    species, failures, per_tag = collect(lines)
    tag_lines = line_species(lines)
    rows = sorted(species.values(), key=lambda r: (-r["u"], r["smiles"]))
    by_u = defaultdict(list)
    for r in rows:
        by_u[r["u"]].append(r)
    named = {}
    for key, drawn in NAMED.items():
        smi = chem.unmapped_canonical(chem.parse_mapped(drawn))
        hit = [r for r in rows if r["smiles"] == smi]
        named[key] = {"smiles": smi, "lines": _lines(hit, tag_lines)}
    open_small = [r for r in rows if r["u"] > 0 and (r["n_atoms"] == 1 or r["linear"])]
    multi = [r for r in rows if r["u"] >= 2]
    return {
        "files": list(FILES),
        "lines": dict(per_tag),
        "unread_lines": dict(failures),
        "n_species": len(rows),
        "u_species": {u: len(v) for u, v in sorted(by_u.items())},
        "u_lines": {u: _lines(v, tag_lines) for u, v in sorted(by_u.items())},
        "named": named,
        "atoms_and_linear_open": {
            "n": len(open_small),
            "atoms_degenerate_fractional": sum(
                1 for r in open_small if (r["atom_shell"] or {}).get("degenerate_fractional")
            ),
            "lines": _lines(open_small, tag_lines),
            "species": [_row(r) for r in open_small],
        },
        "multi_radical": {
            "n": len(multi),
            "lines": _lines(multi, tag_lines),
            "species": [_row(r) for r in multi],
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=None, help="JSON report (default: stdout)")
    args = parser.parse_args(argv)
    RDLogger.DisableLog("rdApp.*")
    rep = report(radical_lines(args.raw))
    text = json.dumps(rep, indent=1, default=str)
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
