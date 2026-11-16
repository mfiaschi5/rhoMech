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

from ksmech.model.config import ModelConfig, check_local
from ksmech.model.decoder import EditDecoder
from ksmech.model.encoder import DenseEncoder
from ksmech.model.flex import FlexEncoder
from ksmech.model.inputs import (
    Y_EXTRA,
    FeatureStats,
    atom_pair_physics,
    charge_product,
    dropped_features,
    frozen,
    standardized_pair_z,
    token_inputs,
    without_open_cells,
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
        check_local(cfg)
        n_atom_x = 2 * len(feature_stats.atom_mean)
        n_orbital_x = len(feature_stats.orbital_mean)
        self.encoder = ENCODERS[cfg.encoder](cfg, n_atom_x, n_orbital_x, n_feat)
        self.decoder = EditDecoder(cfg, n_feat + Y_EXTRA)

    def prepare(self, batch: dict):
        """(token inputs, z_hat, defined, y_ij): what every encoder and the decoder read that is
        not learned, in float32 without gradient. y_ij is the pooled atom-pair physics, or with
        `anchor_head='orbital_pairs'` that head's charge product (`inputs.charge_product`)."""
        with frozen(batch["token_mask"].device.type):
            cfg, cells = self.cfg, self.cfg.pair_cells
            batch = batch if cfg.somo_channels else without_open_cells(batch)
            dropped = dropped_features(cfg, self.pair_stats.names)
            z_hat, defined = standardized_pair_z(batch, self.pair_stats, dropped, cfg.K, cells)
            inp = token_inputs(batch, self.feature_stats)
            if cfg.anchor_head == "orbital_pairs":
                y = charge_product(batch)
            else:
                y, _ = atom_pair_physics(batch, z_hat, defined, cells)
        return inp, z_hat, defined, y

    def encode(self, batch: dict, dense_warmup: bool = False):
        """(h, p, y, aux): token vectors (B, N + 1, d) with MECH at 0, the atom-pair vectors
        (B, A, A, d_p) (`encoder.PairVectors` for the orbital-pair anchor head), the atom-pair
        physics (B, A, A, F + 3) (that head's charge product, (B, A, A, 1)), and the encoder's
        auxiliary losses (the sparse indexer's KL; `dense_warmup` is the sparse encoder's
        warm-up)."""
        inp, z_hat, defined, y = self.prepare(batch)
        h, p, aux = self.encoder(batch, inp, z_hat, defined, dense_warmup)
        return h, p, y, aux
