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
    checkpoint: bool = True  # recompute each encoder layer in the backward pass (memory)
    encoder: str = "dense"  # dense (pair track, spec v9) | flex | sparse
    flex_kernel: bool = True  # flex: FlexAttention on the GPU (else the dense reference)
    topk: int = 64  # sparse: keys per query, besides MECH and itself
    indexer_heads: int = 4  # sparse: heads of the lightweight indexer
    indexer_dim: int = 32  # sparse: width of an indexer head

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["drop_pair_features"] = list(self.drop_pair_features)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        kw = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        kw["drop_pair_features"] = tuple(kw.get("drop_pair_features", ()))
        return cls(**kw)
