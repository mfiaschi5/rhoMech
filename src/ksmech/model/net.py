"""The ksmech model: input preparation, the two-track encoder and the edit decoder in one
module, with the statistics it was trained with (spec v9).

`KSMech.encode(batch)` runs on a collated batch already on the model's device and returns the
final token vectors, the atom-pair vectors, the atom-pair physics y_ij and the encoder's
auxiliary losses; `step_loss`, `anchor_ranks` and `greedy_edit` (`ksmech.model.loss`) read
them. The encoder is `cfg.encoder`: `dense` (spec v9, the pair track), `flex` (FlexAttention
with a relative bias, no pair state) or `sparse` (top-k keys from a learned indexer).
"""

from __future__ import annotations

from torch import nn

from ksmech.model.config import ModelConfig
from ksmech.model.decoder import EditDecoder
from ksmech.model.encoder import DenseEncoder
from ksmech.model.flex import FlexEncoder
from ksmech.model.inputs import (
    Y_EXTRA,
    FeatureStats,
    atom_pair_physics,
    frozen,
    standardized_pair_z,
    token_inputs,
)
from ksmech.model.sparse import SparseEncoder
from ksmech.physics.pairs import PairStats

ENCODERS = {"dense": DenseEncoder, "flex": FlexEncoder, "sparse": SparseEncoder}


class KSMech(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        feature_stats: FeatureStats,
        pair_stats: PairStats,
    ):
        super().__init__()
        self.cfg, self.feature_stats, self.pair_stats = cfg, feature_stats, pair_stats
        n_feat = len(pair_stats.names)
        unknown = set(cfg.drop_pair_features) - set(pair_stats.names)
        if unknown:
            raise ValueError(f"drop_pair_features names unknown features: {sorted(unknown)}")
        if cfg.encoder not in ENCODERS:
            raise ValueError(f"encoder {cfg.encoder!r}, expected one of {sorted(ENCODERS)}")
        n_atom_x = 2 * len(feature_stats.atom_mean)
        n_orbital_x = len(feature_stats.orbital_mean)
        self.encoder = ENCODERS[cfg.encoder](cfg, n_atom_x, n_orbital_x, n_feat)
        self.decoder = EditDecoder(cfg, n_feat + Y_EXTRA)

    def prepare(self, batch: dict):
        """(token inputs, z_hat, defined, y_ij): what every encoder and the decoder read that is
        not learned, in float32 without gradient."""
        with frozen(batch["token_mask"].device.type):
            z_hat, defined = standardized_pair_z(
                batch, self.pair_stats, self.cfg.drop_pair_features, self.cfg.K
            )
            inp = token_inputs(batch, self.feature_stats)
            y, _ = atom_pair_physics(batch, z_hat, defined)
        return inp, z_hat, defined, y

    def encode(self, batch: dict, dense_warmup: bool = False):
        """(h, p, y, aux): token vectors (B, N + 1, d) with MECH at 0, the atom-pair vectors
        (B, A, A, d_p), the atom-pair physics (B, A, A, F + 3), and the encoder's auxiliary
        losses (the sparse indexer's KL; `dense_warmup` is the sparse encoder's warm-up)."""
        inp, z_hat, defined, y = self.prepare(batch)
        h, p, aux = self.encoder(batch, inp, z_hat, defined, dense_warmup)
        return h, p, y, aux
