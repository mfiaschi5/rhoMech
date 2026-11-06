"""Hyper-parameters of the model (spec v9, "Complexity, invariances, and defaults")."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    d: int = 256  # token width
    d_pair: int = 64  # pair width
    layers: int = 8  # encoder layers
    heads: int = 8  # all learned
    rbf: int = 32  # Gaussians of the distance expansion, centred on [0, rbf_max]
    rbf_max: float = 8.0  # Angstrom
    fragments: int = 8  # fragment embedding rows (fragment index mod 8)
    decoder_hidden: int = 128  # width of MLP_R / MLP_N
    physics_hidden: int = 32  # width of MLP_Y over the atom-pair physics y_ij
    K: float = 1.75  # Wolfsberg-Helmholz constant of the cross-centre coupling (fixed)
    drop_pair_features: tuple[str, ...] = ()  # ablation: these z entries are zero and masked
    pair_cells: str = "valid"  # valid (v17: z and pooling on pair_valid only) | all (every cell,
    #   the exclusion flags and the legacy mask as channels; open-shell plan P0a, item 1)
    orbital_extra: tuple[str, ...] = ()  # orbital inputs beyond v17: retained_fraction, energy_ev
    frontier_blocks: str = "window"  # window (0.05 eV, v17) | exact (1e-4 eV) frontier columns
    atom_fukui: str = "frozen"  # frozen (v17: frozen-orbital f+ / f-) | fd (the finite-difference
    #   f+ / f- in their place) | both (FD appended); open-shell plan P2b, ablated in P9
    checkpoint: bool = True  # recompute each encoder layer in the backward pass (memory)
    encoder: str = "dense"  # dense (pair track, spec v9) | flex | sparse
    flex_kernel: bool = True  # flex: FlexAttention on the GPU (else the dense reference)
    topk: int = 64  # sparse: keys per query, besides MECH and itself
    indexer_heads: int = 4  # sparse: heads of the lightweight indexer
    indexer_dim: int = 32  # sparse: width of an indexer head

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["drop_pair_features"] = list(self.drop_pair_features)
        d["orbital_extra"] = list(self.orbital_extra)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        kw = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        kw["drop_pair_features"] = tuple(kw.get("drop_pair_features", ()))
        kw["orbital_extra"] = tuple(kw.get("orbital_extra", ()))
        return cls(**kw)


# The two configurations of the open-shell plan's P0b comparison (U2): GENERAL, the reference,
# reads every pair cell with the exclusion flags as channels and the two orbital extras; LEGACY
# turns every P0a flag off (the v17 inputs). frontier_blocks and the FD columns stay at their
# legacy settings in both: P9 ablates them. Explicit options override a preset.
PRESETS: dict[str, dict[str, Any]] = {
    "general": {"pair_cells": "all", "orbital_extra": ("retained_fraction", "energy_ev")},
    "legacy": {"pair_cells": "valid", "orbital_extra": ()},
}
