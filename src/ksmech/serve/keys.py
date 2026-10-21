"""Identity of species, pools, states and edges (adapter A1).

- token / QM key: `chem.unmapped_canonical(fragment)`, the cache file name (not resonance
  invariant: an enolate drawn as a carbanion is another key);
- species key: `canonical_resonance_key(Chem.MolFromSmiles(token_key)).key`, map-free and
  resonance invariant, memoized by token key;
- pool key: the sorted species keys joined by "." with repeats (each key is one fragment, so the
  join is unambiguous and is itself a SMILES); pool id "p" + sha256("pool-v1|" + key)[:20];
- state id "s" + sha256(mapped_smiles)[:20] (the mapped Lewis form addresses `/pool/{id}`);
- edge id "e" (predicted) or "c" (corrected) + sha256(source state, arrows, kind)[:16].

Upstream helpers (`ksmech.data.chem`, `ksmech.data.resonance`) are imported inside the
functions, so importing this module loads only RDKit.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem, rdBase

from ksmech.serve.contracts import ArrowSpec
from ksmech.serve.errors import Code, ServeError

POOL_KEY_VERSION = 1
EDGE_PREFIX = {"predicted": "e", "corrected": "c"}


@dataclass(frozen=True)
class FragmentKey:
    atoms: tuple[int, ...]  # atom indices of the fragment in the parent mol
    token_key: str
    species_key: str
    flag: str  # data.resonance.ResonanceFlag value
    charge: int


@dataclass(frozen=True)
class SideKeys:
    token_keys: list[str]  # per fragment, GetMolFrags order
    species_keys: list[str]
    pool_key: str
    flags: list[str]


@lru_cache(maxsize=65536)
def species_key_of(token_key: str) -> tuple[str, str]:
    """(species key, ResonanceFlag) of a token key. Raises ServeError(SPECIES_KEY_FAILED)."""
    from ksmech.data.resonance import canonical_resonance_key

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(token_key)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ServeError(Code.SPECIES_KEY_FAILED, f"RDKit cannot parse token key {token_key!r}")
    rk = canonical_resonance_key(mol)
    return rk.key, str(rk.flag)


def fragment_keys(mol: Chem.Mol) -> list[FragmentKey]:
    """Keys of every fragment in `GetMolFrags` order (the order of `ReactantTokens`)."""
    from ksmech.data import chem

    mapping: list = []
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping)
    out = []
    for atoms, frag in zip(mapping, frags, strict=True):
        try:
            token_key = chem.unmapped_canonical(frag)
        except (chem.SanitizeError, ValueError) as exc:
            raise ServeError(
                Code.SPECIES_KEY_FAILED, f"fragment {Chem.MolToSmiles(frag)}: {exc}"
            ) from exc
        species, flag = species_key_of(token_key)
        charge = sum(a.GetFormalCharge() for a in frag.GetAtoms())
        out.append(FragmentKey(tuple(int(a) for a in atoms), token_key, species, flag, charge))
    return out


def pool_key_of(species_keys: Iterable[str]) -> str:
    return ".".join(sorted(species_keys))


def split_pool_key(pool_key: str) -> Counter[str]:
    """The multiset of species keys of a pool key."""
    return Counter(k for k in pool_key.split(".") if k)


def pool_id(pool_key: str) -> str:
    return "p" + hashlib.sha256(f"pool-v{POOL_KEY_VERSION}|{pool_key}".encode()).hexdigest()[:20]


def state_id(mapped_smiles: str) -> str:
    return "s" + hashlib.sha256(mapped_smiles.encode()).hexdigest()[:20]


def _arrow_list(a: ArrowSpec | Mapping | Sequence) -> list[int]:
    if isinstance(a, ArrowSpec):
        return a.as_list()
    if isinstance(a, Mapping):
        return ArrowSpec.from_dict(a).as_list()
    return [int(x) for x in a]


def edge_id(kind: str, source_state_id: str, arrows: Sequence[ArrowSpec]) -> str:
    """Idempotent edge identity: the same arrows from the same state give the same id."""
    if kind not in EDGE_PREFIX:
        raise ValueError(f"edge kind must be one of {sorted(EDGE_PREFIX)}, got {kind!r}")
    body = json.dumps([_arrow_list(a) for a in arrows])
    raw = f"{source_state_id}|{body}|{kind}"
    return EDGE_PREFIX[kind] + hashlib.sha256(raw.encode()).hexdigest()[:16]


def side_keys(smiles: str) -> SideKeys:
    """Token keys, species keys and the pool key of one reaction side, mapped or not.
    Raises ServeError(SPECIES_KEY_FAILED)."""
    from ksmech.data import chem

    try:
        mol = chem.parse_mapped(smiles)
    except (chem.SmilesParseError, chem.SanitizeError) as exc:
        raise ServeError(Code.SPECIES_KEY_FAILED, str(exc)) from exc
    frags = fragment_keys(mol)
    species = [f.species_key for f in frags]
    return SideKeys(
        [f.token_key for f in frags], species, pool_key_of(species), [f.flag for f in frags]
    )
