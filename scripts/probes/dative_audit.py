"""The oxidation-state audit of the dative law (open-shell plan P0a step 6; re-run in P5).

    uv run python scripts/probes/dative_audit.py [--source reactmech] [--n 20] [--seed 0]

Read-only: nothing is written. Every step of the source is screened again from its drawn sides
(the stored SMILES, or the `charge_repair` record's sides before the repair for a step already
repaired), so the same cases come back before and after the P0a build. The steps the dative law
repairs (`data.domain`, rule 'D dative onium->M') fall into three strata, read on the fragments
it re-charges: L1PD (a Pd whose only neighbour is an onium donor: the monoligated Pd(0)
returns), ODD (any other odd fragment: the odd-Pd returns) and EVEN (only even fragments: the
two-onium Pd/Ni re-charges of kept steps). `--n` cases are drawn at `--seed` over the sorted
step ids of each stratum, shared as evenly as the strata allow. For each case and each repaired
fragment the probe prints the fragment before and after, each acceptor's oxidation state
(`domain.oxidation_state`) and d count before and after, the counterions (the other charged
fragments of that side) and the pool balance (each side's net charge before and after). The
verdict, whether the repaired oxidation state is the one the chemistry implies, is read by hand.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from ksmech.data import chem, domain
from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_steps, source_dir, steps_path
from ksmech.data.validation import screened

STRATA = ("EVEN", "ODD", "L1PD")


def drawn_sides(step) -> tuple[Chem.Mol, Chem.Mol]:
    """The step's sides as the source drew them."""
    if step.charge_repair:
        note = json.loads(step.charge_repair)
        sides = note["reactants_before"], note["products_before"]
    else:
        sides = step.reactants_smiles, step.products_smiles
    return chem.parse_mapped(sides[0]), chem.parse_mapped(sides[1])


def stratum(rows: list[dict]) -> str:
    """The stratum of a step from its dative repairs' fragments as drawn."""
    befores = [chem.parse_mapped(r["before"]) for r in rows]
    for frag in befores:
        for a in frag.GetAtoms():
            nbrs = list(a.GetNeighbors())
            if a.GetSymbol() == "Pd" and len(nbrs) == 1 and domain.is_onium_donor(nbrs[0]):
                return "L1PD"
    return "ODD" if any(domain.is_odd(f) for f in befores) else "EVEN"


def repaired_cases(steps) -> dict[str, list[tuple[str, object, Chem.Mol, Chem.Mol, list]]]:
    """Stratum -> (step id, step, drawn reactants, drawn products, dative repair rows)."""
    out: dict[str, list] = {s: [] for s in STRATA}
    for step in steps:
        r, p = drawn_sides(step)
        screen = screened(r, p, step.arrows, require_arrows=bool(step.arrows))
        rows = [
            {"side": s, "before": b, "after": a}
            for s, rule, b, a in screen.repairs
            if rule == domain.DATIVE_RULE
        ]
        if rows:
            out[stratum(rows)].append((step.step_id, step, r, p, rows))
    return {k: sorted(v, key=lambda c: c[0]) for k, v in out.items()}


def draw(cases: dict[str, list], n: int, seed: int) -> list[tuple[str, tuple]]:
    """`n` cases shared over the strata as evenly as their sizes allow, drawn at `seed`."""
    rng = np.random.default_rng(seed)
    quota, left = dict.fromkeys(STRATA, 0), n
    while left and any(quota[s] < len(cases[s]) for s in STRATA):
        for s in sorted(STRATA, key=lambda s: -len(cases[s])):
            if left and quota[s] < len(cases[s]):
                quota[s], left = quota[s] + 1, left - 1
    picked = []
    for s in STRATA:
        idx = sorted(rng.choice(len(cases[s]), size=quota[s], replace=False).tolist())
        picked += [(s, cases[s][i]) for i in idx]
    return picked


def acceptor_states(smiles: str) -> str:
    """'Pd +2 d8 u0' for every transition metal of a fragment and every other acceptor that
    carries an onium donor: oxidation state, d count, uncompensated donors."""
    out = []
    for a in chem.parse_mapped(smiles).GetAtoms():
        d = domain.d_count(a)
        donor = any(domain.is_onium_donor(n) for n in a.GetNeighbors())
        if not domain.is_acceptor(a) or (d is None and not donor):
            continue
        state = f"{a.GetSymbol()} {domain.oxidation_state(a):+d}" + ("" if d is None else f" d{d}")
        out.append(f"{state} u{domain.uncompensated(a)}")
    return ", ".join(out) or "-"


def counterions(side: Chem.Mol, repaired: list[str]) -> str:
    """The charged fragments of a side other than the re-charged ones, with their charges."""
    out = Counter()
    for frag in chem.fragments(side):
        smi = chem.unmapped_canonical(frag)
        if Chem.GetFormalCharge(frag) and smi not in repaired:
            out[smi] += 1
    return ", ".join(f"{k} x{v}" if v > 1 else k for k, v in sorted(out.items())) or "none"


def print_case(k: int, name: str, case: tuple, parts: dict[str, str]) -> None:
    step_id, step, r, p, rows = case
    code = step.domain_exclusion or "kept"
    print(f"[{k}] {name}  {step_id}  stored code {code}, frozen part {parts.get(step_id, '-')}")
    for side, mol in (("reactants", r), ("products", p)):
        mine = [x for x in rows if x["side"] == side]
        lowered = sum(_charge(x["before"]) - _charge(x["after"]) for x in mine)
        q = chem.formal_charge(mol)
        print(
            f"  {side}: pool {q:+d} -> {q - lowered:+d}; counterions: "
            + counterions(mol, [x["before"] for x in mine])
        )
        for x in mine:
            print(f"    before {x['before']}")
            print(f"           {acceptor_states(x['before'])}")
            print(f"    after  {x['after']}")
            print(f"           {acceptor_states(x['after'])}")


def census(cases: dict[str, list]) -> Counter:
    """(stratum, side, metal, oxidation state before -> after, d after, pool after) over every
    re-charged fragment of every repaired step, not only the drawn ones."""
    out: Counter = Counter()
    for name, rows in cases.items():
        for _, _, r, p, repairs in rows:
            pools = {"reactants": chem.formal_charge(r), "products": chem.formal_charge(p)}
            lowered = Counter()
            for x in repairs:
                lowered[x["side"]] += _charge(x["before"]) - _charge(x["after"])
            for x in repairs:
                pair = zip(_metals(x["before"]), _metals(x["after"]), strict=True)
                pool = pools[x["side"]] - lowered[x["side"]]
                for (sym, os0, _), (_, os1, d1) in pair:
                    out[(name, x["side"], sym, f"{os0:+d} -> {os1:+d}", d1, f"{pool:+d}")] += 1
    return out


def _charge(smiles: str) -> int:
    return Chem.GetFormalCharge(chem.parse_mapped(smiles))


def _metals(smiles: str) -> list[tuple[str, int, int | None]]:
    """(symbol, oxidation state, d count) of the acceptors bonded to an onium donor, by symbol
    (no repaired fragment holds two acceptors with donors: the census would pair them by
    symbol)."""
    mol = chem.parse_mapped(smiles)
    return sorted(
        (a.GetSymbol(), domain.oxidation_state(a), domain.d_count(a))
        for a in mol.GetAtoms()
        if domain.is_acceptor(a) and any(domain.is_onium_donor(n) for n in a.GetNeighbors())
    )


def frozen_parts(source: str, processed: Path) -> dict[str, str]:
    path = source_dir(source, processed) / "split_frozen_molecule.json"
    if not path.exists():
        return {}
    splits = json.loads(path.read_text())["splits"]
    return {i: name.rsplit("_", 1)[-1] for name, ids in splits.items() for i in ids}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", default="reactmech")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    steps = read_steps(steps_path(args.source, args.processed_dir))
    cases = repaired_cases(steps)
    sizes = ", ".join(f"{s} {len(cases[s])}" for s in STRATA)
    print(
        f"{args.source}: {sum(map(len, cases.values()))} steps repaired by the dative law ({sizes})"
    )
    parts = frozen_parts(args.source, args.processed_dir)
    for k, (name, case) in enumerate(draw(cases, args.n, args.seed), start=1):
        print_case(k, name, case, parts)
    print("census of every re-charged fragment: stratum, side, metal, OS, d after, pool after")
    for key, count in sorted(census(cases).items(), key=lambda kv: (kv[0][0], -kv[1])):
        print(f"  {count:4d}  " + "  ".join(str(k) for k in key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
