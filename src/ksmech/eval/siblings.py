"""The recorded outcomes of a reactant pool (reaction-network plan, item A: multi-reference
evaluation).

The same reactants can take several correct elementary steps, and a source records some of them
as separate steps (PMechDB: ethyl acetate + ethoxide both as the enolate and as the tetrahedral
adduct). A decode then scores a completion against the sample's own label (`rank`) and against
every product the source records for the same reactant pool (`rank_any`).

Pools and products are compared by the map-free identity key v2 (`data.resonance.identity_key`)
of each side's drawing, never by matrix: sibling steps number their atoms differently. The
product key is the one the sample store writes (`MechanismSample.product_key`), which equals
the key `eval.decode.lewis_apply` gives B + R of the label (checked on 400 PMechDB val samples,
decisions.md 2026-11-09). Siblings come from the whole source, i.e. its training steps
(`data.io.read_training_steps`: resonance steps and excluded steps left out), whatever part of
the split they fall in; the report counts the siblings of another part. A step whose side RDKit
cannot read is counted as SIBLING_KEY_FAILED and has no siblings.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from rdkit import rdBase

from ksmech.data.chem import parse_mapped
from ksmech.data.resonance import identity_key

SIBLING_KEY_FAILED = "SIBLING_KEY_FAILED"


@dataclass(frozen=True)
class Outcomes:
    """The recorded outcomes of one step's reactant pool: every product key the source records
    for it (the step's own included) and the parts of the steps that record another product."""

    own: str
    products: frozenset[str]
    other_parts: tuple[str, ...] = ()  # one entry per sibling step with another product

    @property
    def multi(self) -> bool:
        return len(self.products) > 1


def side_keys(steps: Iterable, skips: Counter) -> dict[str, tuple[str, str]]:
    """step id -> (map-free identity key of the reactants, of the products)."""
    out: dict[str, tuple[str, str]] = {}
    for step in steps:
        try:
            with rdBase.BlockLogs():
                pool = identity_key(parse_mapped(step.reactants_smiles))
                product = identity_key(parse_mapped(step.products_smiles))
        except Exception:  # noqa: BLE001 - counted, the step gets no siblings
            skips[SIBLING_KEY_FAILED] += 1
            continue
        out[step.step_id] = (pool, product)
    return out


def outcome_table(
    keys: Mapping[str, tuple[str, str]], parts: Mapping[str, str] | None = None
) -> dict[str, Outcomes]:
    """step id -> its pool's `Outcomes`, the pools grouped by reactant key. `parts` maps a step
    id to its part of the split (a step in none counts as 'none')."""
    parts = parts or {}
    by_pool: dict[str, list[str]] = defaultdict(list)
    for step_id, (pool, _) in keys.items():
        by_pool[pool].append(step_id)
    out: dict[str, Outcomes] = {}
    for members in by_pool.values():
        products = frozenset(keys[m][1] for m in members)
        for m in members:
            own = keys[m][1]
            others = sorted(parts.get(o, "none") for o in members if keys[o][1] != own)
            out[m] = Outcomes(own, products, tuple(others))
    return out


def parts_of(split: Mapping[str, list[str]]) -> dict[str, str]:
    """step id -> part from a split file's {part: [step ids]}."""
    return {step_id: part for part, ids in split.items() for step_id in ids}


def outcome_summary(table: Mapping[str, Outcomes], ids: Iterable[str], part: str) -> dict:
    """Over the decoded ids: how many lie in a pool with several recorded products, how many
    outcomes such pools have, and where their siblings come from (`part` is the decoded part)."""
    ids = [i for i in ids if i in table]
    multi = [table[i] for i in ids if table[i].multi]
    where = Counter(p for o in multi for p in o.other_parts)
    return {
        "n_with_key": len(ids),
        "n_multi": len(multi),
        "max_outcomes": max((len(o.products) for o in multi), default=1),
        "siblings_same_part": where.get(part, 0),
        "siblings_other_part": sum(v for k, v in where.items() if k != part),
        "siblings_by_part": dict(sorted(where.items())),
    }
