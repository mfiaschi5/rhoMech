"""Phase 3 of the out-of-distribution plan: the encoder pretrained on QMugs molecules with the
donor-acceptor energy of every orbital pair as target (`physics.donor_acceptor`; docs/decisions.md,
2026-11-22). `pretrain.samples` builds the store, `train.pretrain` trains on it, and
`ksmech-train fit --init-from` starts a mechanism run from the pretrained encoder."""
