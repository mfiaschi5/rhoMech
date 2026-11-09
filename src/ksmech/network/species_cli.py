"""`ksmech-network species`: the species graph of a pool and its hyperpaths (reaction-network
plan, item J).

    ksmech-network species --run pm_control_s0 --start 'C[O-].CC(C)=O' \
        [--target 'CC(=O)CC(C)(C)O'] [--reservoir CO] [--tolerance 0.05] [--max-species 10] \
        [--max-expansions 60] [--partner-weight 1] [--k 3] [--levels exact skeleton] \
        [--beam 5] [--build-missing] [--out FILE]

The graph grows species by species from the start pool's species (`network.species`, every pool
one model call on CPU through `network.lift.ModelPoolReactor`), the reservoir added to every
pool, until no frontier species has `--tolerance` of mass or a budget (`--max-species` species
in the core, `--max-expansions` pools) ends it. Reported: every species with its mass and role,
the end species with their shares of what stays (the fixed-concentration approximation,
`network.flux`), the unexplored mass, every cut and skip code; with `--target`, the k shortest
hyperpaths (`network.hyperpaths`: additive cost, arrival level, the events of a linearization,
classes up to the order of independent steps) and, for every target species formed, the
atom-flux analysis (each heavy atom's dominant backward path and its share, merged). `--out`
writes it all as JSON with a provenance header.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from rdkit import Chem, rdBase
from rich.console import Console
from rich.table import Table

from ksmech.data import provenance as prov

SPECIES_VERSION = 1


def add_parser(sub) -> None:
    from ksmech.orbitals.build import DEFAULT_TOKENS_DIR
    from ksmech.train.loop import DEFAULT_MODELS_DIR

    s = sub.add_parser("species", help="the species graph of a pool and its hyperpaths")
    s.add_argument("--run", required=True, help="training run under <models-dir>/<run-source>/")
    s.add_argument("--run-source", default="pmechdb", help="the run's models directory")
    s.add_argument("--checkpoint", default="best", choices=["best", "last"])
    s.add_argument("--start", required=True, help="the starting pool (SMILES)")
    s.add_argument("--target", help="the target product(s); without it, what forms")
    s.add_argument("--reservoir", default="", help="species in every pool, never consumed")
    s.add_argument("--tolerance", type=float, default=0.05, help="mass to join the core")
    s.add_argument("--max-species", type=int, default=10, help="species in the core")
    s.add_argument("--max-expansions", type=int, default=60, help="pools expanded (model calls)")
    s.add_argument("--partner-weight", type=float, default=1.0,
                   help="a partner pool's weight against the species alone")  # fmt: skip
    s.add_argument("--k", type=int, default=3, help="hyperpaths to report")
    s.add_argument("--levels", nargs="+", default=["EXACT", "SKELETON"], type=str.upper,
                   choices=["EXACT", "SKELETON"], help="arrival levels that count")  # fmt: skip
    s.add_argument("--beam", type=int, default=5, help="anchors completed and products kept each")
    s.add_argument("--build-missing", action="store_true", help="compute missing tokens (GFN2)")
    s.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    s.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    s.add_argument("--out", type=Path, help="the result (JSON)")


def plain(sp) -> str:
    """A species' drawing as a SMILES without map numbers or explicit H atoms."""
    if not sp.smiles:
        return sp.key
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(sp.smiles)
        if mol is None:
            return sp.key
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False))


def _role(g, key: str) -> str:
    if key in g.sources:
        return "start"
    if key in g.reservoir:
        return "reservoir"
    return "core" if key in g.core else "frontier"


def species_rows(g, mass) -> list[dict]:
    """Every species by mass: SMILES, key, role, mass, what stays of it (terminal: for a core
    species the share no channel took, which is unexplored) and, for a frontier species, its
    share of the mass the frontier holds (the end species where the expansion stopped)."""
    from ksmech.network.flux import balances

    bal = balances(g, mass)
    frontier = set(g.frontier())
    held = sum(mass.get(k, 0.0) for k in frontier) or 1.0
    rows = []
    for k in sorted(g.species, key=lambda k: (-mass.get(k, 0.0), k)):
        terminal = max(bal[k].terminal, 0.0) if k in bal else mass.get(k, 0.0)
        share = mass.get(k, 0.0) / held if k in frontier else 0.0
        rows.append({"smiles": plain(g.species[k]), "key": k, "role": _role(g, k),
                     "mass": mass.get(k, 0.0), "terminal": terminal,
                     "end_share": share})  # fmt: skip
    return rows


def step_text(g, step) -> str:
    side = lambda keys: " + ".join(plain(g.species[k]) for k in keys)  # noqa: E731
    return f"{side(step.reactants)} >> {side(step.products)}"


def hyperpath_rows(g, slots, levels, k: int) -> tuple[list[dict], Counter]:
    """The k shortest hyperpaths to the target slots, one per species class."""
    from ksmech.network.hyperpaths import hypergraph_of, k_shortest_hyperpaths, linearize, unfold

    hg = hypergraph_of(g, slots, tuple(levels))
    found, counts = k_shortest_hyperpaths(hg, k, distinct=lambda h: unfold(h).species_class)
    rows = []
    for r, hp in enumerate(found):
        u = unfold(hp)
        lin = linearize(u)
        rows.append({"rank": r + 1, "cost": hp.cost, "events_logp": u.logp,
                     "arrival": hp.arrival, "n_events": len(u.events),
                     "events": [step_text(g, u.events[e.proposal.info["event"]].step)
                                for e in lin],
                     "signatures": [e.proposal.signature for e in lin]})  # fmt: skip
    return rows, counts


def flux_rows(g, mass, slots, levels=("EXACT", "SKELETON")) -> list[dict]:
    """The atom-flux analysis of every species the graph forms that is a target slot (at the
    levels allowed); a slot no species formed is a row with `formed` False."""
    from ksmech.network.flux import atom_pathways, merged_pathway
    from ksmech.network.hyperpaths import slot_matches

    out = []
    for slot in sorted(set(slots)):
        hits = [(k, lv) for k, lv in slot_matches(g, slot, levels) if mass.get(k, 0.0) > 0]
        if not hits:
            out.append({"target": slot[0], "formed": False})
        for key, level in hits:
            paths = atom_pathways(g, mass, key)
            merged = merged_pathway(paths)
            out.append({"target": slot[0], "key": key, "level": level, "formed": True,
                        "smiles": plain(g.species[key]), "mass": mass[key],
                        "share": merged.share,
                        "steps": [step_text(g, g.steps[s]) for s in merged.steps],
                        "atoms": [{"atom": p.atom, "share": p.share, "origin": p.origin,
                                   "steps": [step_text(g, g.steps[s]) for s in p.steps]}
                                  for p in paths]})  # fmt: skip
    return out


def species_report(g, slots=None, levels=("EXACT", "SKELETON"), k: int = 3) -> dict:
    """The JSON body of `ksmech-network species` for a grown graph (module docstring)."""
    from ksmech.network.flux import masses

    mass, diverged = masses(g)
    unexplored = sum(mass.get(x, 0.0) * g.leftover.get(x, 0.0) for x in g.core)
    frontier = sum(mass.get(x, 0.0) for x in g.frontier())
    body = {"reason": g.reason, "core": list(g.core), "pools": g.pools,
            "n_species": len(g.species), "n_steps": len(g.steps), "mass_diverged": diverged,
            "unexplored_mass": unexplored, "frontier_mass": frontier,
            "species": species_rows(g, mass), "cuts": dict(g.cuts),
            "skips": dict(g.skips)}  # fmt: skip
    if slots:
        rows, counts = hyperpath_rows(g, slots, levels, k)
        body.update(hyperpaths=rows, hyperpath_counts=dict(counts),
                    atom_flux=flux_rows(g, mass, slots, levels))  # fmt: skip
    return body


def _print(console: Console, body: dict) -> None:
    table = Table(title=f"species ({body['reason']}, {body['pools']} pools, "
                        f"{len(body['core'])} in the core)")  # fmt: skip
    for col in ("smiles", "role", "mass", "end share"):
        table.add_column(col)
    for r in body["species"][:25]:
        table.add_row(r["smiles"], r["role"], f"{r['mass']:.3g}", f"{r['end_share']:.3f}")
    console.print(table)
    console.print(f"unexplored mass {body['unexplored_mass']:.3g}, "
                  f"frontier mass {body['frontier_mass']:.3g}")  # fmt: skip
    for r in body.get("hyperpaths", []):
        console.print(f"hyperpath {r['rank']}: cost {r['cost']:.2f}, {r['n_events']} event(s), "
                      f"{r['arrival']}")  # fmt: skip
        for e in r["events"]:
            console.print(f"    {e}")
    for f in body.get("atom_flux", []):
        if f["formed"]:
            console.print(f"atom flux of {f['smiles']}: merged share {f['share']:.3f}, "
                          f"{len(f['steps'])} step(s)")  # fmt: skip
    for code, n in sorted({**body["cuts"], **body["skips"]}.items()):
        console.print(f"  {code}: {n}")


def cmd_species(args, console: Console) -> int:
    from ksmech.arrows.reactants import TokenCache
    from ksmech.model.decode import DecodeConfig
    from ksmech.network.lift import ModelPoolReactor, start_of, target_slots
    from ksmech.network.pools import ModelExpander, pool_state
    from ksmech.network.species import ExpansionParams, expand_species
    from ksmech.train.loop import load_model

    ckpt = args.models_dir / args.run_source / args.run / f"{args.checkpoint}.pt"
    model, _ = load_model(ckpt, "cpu")
    cfg = DecodeConfig(anchors=args.beam, per_anchor=args.beam)
    expander = ModelExpander(model, TokenCache(args.tokens_dir), cfg,
                             build_missing=args.build_missing)  # fmt: skip
    records, amounts = start_of(pool_state(args.start))
    reservoir: Counter = Counter()
    if args.reservoir:
        extra, reservoir = start_of(pool_state(args.reservoir))
        records.update(extra)
    params = ExpansionParams(args.tolerance, args.max_species, args.max_expansions,
                             args.partner_weight)  # fmt: skip
    g = expand_species(ModelPoolReactor(expander), records, amounts, sorted(reservoir), params)
    slots = target_slots(args.target) if args.target else None
    body = species_report(g, slots, args.levels, args.k)
    _print(console, body)
    if args.out:
        keys = ("run", "run_source", "checkpoint", "start", "target", "reservoir", "tolerance",
                "max_species", "max_expansions", "partner_weight", "k", "levels", "beam",
                "build_missing")  # fmt: skip
        header = prov.stamp("network_species", SPECIES_VERSION, prov.file_input(ckpt),
                            {k: getattr(args, k) for k in keys})  # fmt: skip
        prov.write_json(args.out, body, header)
        console.print(f"wrote {args.out}")
    return 0
