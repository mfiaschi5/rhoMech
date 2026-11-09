"""The QM free energies of pool states (reaction-network plan, item B wired into items E and F).

`qm_thermo_model` is the provider `network.free_energy.load_thermo_model('qm')` returns: each
fragment of a pool state is looked up by its record key (the map-free canonical SMILES the QM
cache and the thermochemistry sidecars use, `data.chem.unmapped_canonical`) in
`data/cache/thermo/` (`qm.thermo_cache.thermo_of`), evaluated at T (`qm.thermo.thermo_at`: GFN2
with ALPB, qRRHO, the 1 M standard state), and summed:

    G(pool) = sum_i G_i(T) + RT ln(c_i / c0),

with c_i = c0 = 1 M for every species but the reservoir's, which sit at their `molar`
concentration (water as solvent: 55.3 M), so a reservoir copy is priced at its chemical
potential, as `network.reservoir` asks. H and S come with G, so the chain can give T*:
S holds the standard-state shift as an entropy (-RT ln(V_m / V0) / T), H does not.

A fragment without a sidecar (or with a stale one) leaves its pool without free energy (None,
counted by the chain as NO_FREE_ENERGY); the misses are counted here by code, per record key
once: THERMO_MISSING, THERMO_STALE. Run `ksmech-qm thermo` over the species a network reaches.
Energies in kcal/mol, entropies in kcal/(mol K).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping

from ksmech.network.free_energy import Thermo, ThermoModel, combine, rt

THERMO_MISSING, THERMO_STALE = "THERMO_MISSING", "THERMO_STALE"


def fragment_records(smiles: str) -> list[tuple[str, str]]:
    """(identity key v2, record key) of every fragment of a pool's SMILES."""
    from rdkit import Chem, rdBase

    from ksmech.data.chem import parse_mapped, unmapped_canonical
    from ksmech.data.resonance import identity_key

    with rdBase.BlockLogs():
        mol = parse_mapped(smiles)
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        return [(identity_key(f), unmapped_canonical(f)) for f in frags]


class QMThermo:
    """Per-species thermochemistry from the sidecars, memoized by (record key, T)."""

    def __init__(self, settings=None) -> None:
        from ksmech.qm.thermo import DEFAULTS

        self.settings = settings or DEFAULTS
        self.memo: dict[tuple[str, float], Thermo | None] = {}
        self.counts: Counter = Counter()

    def __call__(self, record_key: str, temperature_k: float) -> Thermo | None:
        key = (record_key, float(temperature_k))
        if key not in self.memo:
            self.memo[key] = self._compute(record_key, float(temperature_k))
        return self.memo[key]

    def _compute(self, record_key: str, t: float) -> Thermo | None:
        from ksmech.data.provenance import StaleArtifact
        from ksmech.qm.thermo import HARTREE_KCAL, thermo_at
        from ksmech.qm.thermo_cache import thermo_of

        try:
            r = thermo_at(thermo_of(record_key), t, self.settings)
        except FileNotFoundError:
            self.counts[THERMO_MISSING] += 1
            return None
        except StaleArtifact:
            self.counts[THERMO_STALE] += 1
            return None
        s = (r.entropy - r.state_shift / t) * HARTREE_KCAL
        return Thermo(r.free_energy * HARTREE_KCAL, r.enthalpy * HARTREE_KCAL, s)


def at_concentration(base: Thermo, molar: float, temperature_k: float) -> Thermo:
    """mu = G + RT ln(c / 1 M): the shift is entropic (S - R ln c)."""
    shift = rt(temperature_k) * math.log(molar)
    return Thermo(
        base.g + shift, base.h, None if base.s is None else base.s - shift / temperature_k
    )


def qm_thermo_model(
    reservoir_smiles: Mapping[str, str] | None = None,
    molar: Mapping[str, float] | None = None, species: QMThermo | None = None,
) -> ThermoModel:  # fmt: skip
    """The `ThermoModel` of the module docstring. `reservoir_smiles`: identity key -> SMILES of
    each reservoir species (`network.reservoir.Reservoir.smiles`); `molar`: identity key -> its
    concentration in mol/L (1 M when absent); `species` the per-species lookup, whose `counts`
    hold the misses (pass one in to read them)."""
    reservoir_smiles, molar = dict(reservoir_smiles or {}), dict(molar or {})
    species = species or QMThermo()

    def free_energy(state, temperature_k: float) -> Thermo | None:
        terms = []
        for key, record in fragment_records(state.smiles):
            one = species(record, temperature_k)
            if one is None:
                return None
            if key in reservoir_smiles:
                one = at_concentration(one, molar.get(key, 1.0), temperature_k)
            terms.append((one, 1.0))
        return combine(Thermo(0.0, 0.0, 0.0), terms)

    def chemical_potential(key: str, temperature_k: float) -> Thermo | None:
        if key not in reservoir_smiles:
            return None
        (_, record), *_ = fragment_records(reservoir_smiles[key])
        one = species(record, temperature_k)
        return None if one is None else at_concentration(one, molar.get(key, 1.0), temperature_k)

    return ThermoModel(free_energy, chemical_potential)
