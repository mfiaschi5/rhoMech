"""The embedding trace of one sample (`ksmech-eval embedding`): what a trained model does with
its input, stage by stage, for reading the model rather than scoring it.

1. The token embedding h^(0) (`encoder.TokenEmbedding`) split into its additive parts
   (`embedding_parts`): on an atom token E_Z + E_charge + MLP_A(atom inputs); on an orbital
   token MLP_O(orbital inputs), the set encoder's sum over its spectral entries and the
   centre-weighted atom embeddings it inherits; the fragment embedding on both; the learned
   MECH token at position 0.
2. Every encoder layer (`layer_trace`): its attention rebuilt from the layer's own inputs and
   weights as content (q.k / sqrt(d_h)) plus the learned pair bias (the pair track of the dense
   encoder, the relative bias of flex, MLP_P over the pair channels of sparse, on the keys its
   indexer selects), and the token updates of the attention and of the feed-forward block. The
   layer inputs are recorded by forward hooks on the model's own forward pass, and the rebuilt
   update is checked against the layer's output (`errors`), so the trace is what the model ran.
3. The decoder (`anchor_readout`): every anchor logit split into the site terms (the two atoms'
   token vectors against the step query read from h_MECH), the pair term (the atom-pair vector)
   and the physics term MLP_Y(y_ij); the top anchors, the greedy R and, with a label class,
   the label's rank.

Token rows follow the MECH layout: 0 is MECH, then the atoms, then the orbitals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ksmech.arrows.batches import ANCHOR_SIDE, collate_samples
from ksmech.arrows.lewis import reactant_lewis
from ksmech.arrows.reactants import ReactantTokens, TokenCache
from ksmech.arrows.steps import MechanismSample, reactant_sample
from ksmech.model.encoder import TokenEmbedding
from ksmech.model.inputs import CHARGE_MAX, MAX_Z, pair_channels, to_device
from ksmech.model.loss import anchor_log_probs, anchor_ranks, greedy_edit
from ksmech.model.net import KSMech
from ksmech.model.sparse import select_keys
from ksmech.qm.xtb_calc import QMConfig

PARTS = (
    "mech",
    "element",
    "charge",
    "atom_inputs",
    "orbital_inputs",
    "spectral",
    "inherited",
    "fragment",
)
SIDE = {v: k for k, v in ANCHOR_SIDE.items()}
TERMS = ("site", "pair", "physics")


def embedding_parts(embed: TokenEmbedding, batch: dict, inp: dict) -> dict[str, torch.Tensor]:
    """The additive parts of h^(0) (PARTS), each (B, N + 1, d) on the MECH layout and zero
    where it does not act: their sum is `embed(batch, inp)`."""
    cat = batch["atom_categorical"]
    n_atoms = cat.shape[1]
    on_atom = batch["is_atom"][:, :n_atoms, None]
    charge = cat[..., 1].clamp(-CHARGE_MAX, CHARGE_MAX) + CHARGE_MAX
    atom = {
        "element": embed.z(cat[..., 0].clamp(0, MAX_Z)) * on_atom,
        "charge": embed.charge(charge) * on_atom,
        "atom_inputs": embed.atom(inp["atom_x"]) * on_atom,
    }
    h_atom = atom["element"] + atom["charge"] + atom["atom_inputs"]
    spectral = embed.spectral(inp["spectra"]) * inp["spectra_mask"][..., None]
    orbital = {
        "orbital_inputs": embed.orbital(inp["orbital_x"]),
        "spectral": spectral.sum(-2),
        "inherited": batch["incidence"][:, n_atoms:, :].to(h_atom.dtype) @ h_atom,
    }
    no_atom, no_orbital = torch.zeros_like(h_atom), torch.zeros_like(orbital["spectral"])
    tokens = {k: torch.cat([v, no_orbital], 1) for k, v in atom.items()}
    tokens |= {k: torch.cat([no_atom, v], 1) for k, v in orbital.items()}
    tokens["fragment"] = embed.fragment(batch["fragment"].clamp(min=0) % embed.n_fragments)
    mask = batch["token_mask"][..., None]
    first = torch.zeros_like(tokens["fragment"][:, :1])
    mech = embed.mech.expand(first.shape[0], -1, -1).to(first.dtype)
    out = {"mech": torch.cat([mech, torch.zeros_like(tokens["fragment"])], 1)}
    out |= {k: torch.cat([first, v * mask], 1) for k, v in tokens.items()}
    return {k: out[k] for k in PARTS}


@dataclass
class LayerCall:
    layer: torch.nn.Module
    args: tuple
    output: object

    @property
    def h_out(self) -> torch.Tensor:
        return self.output[0] if isinstance(self.output, tuple) else self.output


def encode_with_calls(model: KSMech, batch: dict):
    """(h, p, y, calls): `model.encode(batch)` with the arguments and the output of every
    encoder layer recorded by forward hooks."""
    calls: list[LayerCall] = []

    def hook(module, args, output):
        calls.append(LayerCall(module, args, output))

    hooks = [layer.register_forward_hook(hook) for layer in model.encoder.layers]
    try:
        h, p, y, _ = model.encode(batch)
    finally:
        for handle in hooks:
            handle.remove()
    return h, p, y, calls


def _selected_keys(layer, h: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """(B, N + 1, N + 1) the keys a sparse layer's indexer selects for each query."""
    idx, ok = select_keys(layer.indexer(h), keys, layer.k)
    out = torch.zeros(idx.shape[0], idx.shape[1], keys.shape[1], device=h.device)
    return out.scatter_reduce(-1, idx, ok.float(), "amax") > 0


def attention_terms(model: KSMech, call: LayerCall, batch: dict, z_hat, defined):
    """(content, bias, read, v) of one layer call: content q.k / sqrt(d_h) and the learned bias,
    both (B, H, N + 1, N + 1) float32; `read` the keys each query attends to, of the same shape;
    v (B, H, N + 1, d_h)."""
    layer, h = call.layer, call.args[0]
    b, n, _ = h.shape
    q, k, v = layer.qkv(layer.norm_h(h)).view(b, n, 3, layer.heads, -1).permute(2, 0, 3, 1, 4)
    content = (q @ k.transpose(-1, -2)).float() / math.sqrt(q.shape[-1])
    kind = model.cfg.encoder
    if kind == "dense":
        _, p, keys = call.args
        bias = layer.bias(layer.norm_p(p)).permute(0, 3, 1, 2).float()
        read = keys[:, None, None, :]
    elif kind == "flex":
        x = call.args[1]
        bias = layer.bias.dense(x, layer.bias.z(x.ctx.z)).float()
        read = x.keys[:, None, None, :]
    else:
        _, _, _, keys, _, dense = call.args
        cfg = model.cfg
        bias = layer._bias(pair_channels(batch, z_hat, defined, cfg.rbf, cfg.rbf_max))
        read = keys[:, None, None, :] if dense else _selected_keys(layer, h, keys)[:, None]
    return content, bias, read.expand_as(content), v


def _masked_std(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """(B, H) standard deviation of x over the cells m of each head."""
    n = m.sum((-1, -2)).clamp(min=1)
    mean = torch.where(m, x, 0.0).sum((-1, -2)) / n
    var = torch.where(m, (x - mean[..., None, None]) ** 2, 0.0).sum((-1, -2)) / n
    return var.sqrt()


def layer_trace(model: KSMech, call: LayerCall, batch: dict, z_hat, defined) -> dict:
    """One encoder layer: the attention (B, H, N + 1, N + 1); the spread (standard deviation)
    of the content and of the bias logits per head over the cells read, (B, H, 2); the norms
    of the attention and the feed-forward updates per token, (B, N + 1) each; the largest
    deviation of h + both updates from the layer's output."""
    layer, h = call.layer, call.args[0]
    content, bias, read, v = attention_terms(model, call, batch, z_hat, defined)
    attn = torch.softmax((content + bias).masked_fill(~read, float("-inf")), -1)
    b, n, d = h.shape
    d_attn = layer.out((attn.to(v.dtype) @ v).transpose(1, 2).reshape(b, n, d))
    d_ffn = layer.ffn(layer.norm_f(h + d_attn))
    return {
        "attn": attn,
        "spread": torch.stack([_masked_std(content, read), _masked_std(bias, read)], -1),
        "d_attn": d_attn.norm(dim=-1),
        "d_ffn": d_ffn.norm(dim=-1),
        "error": float((h + d_attn + d_ffn - call.h_out).abs().max()),
    }


def anchor_terms(model: KSMech, h, p, y) -> torch.Tensor:
    """(B, A, A, 2, 3) the terms of `EditDecoder.anchor_logits` (TERMS): site, the two atoms'
    token vectors against the step query of h_MECH; pair, u_sign of the atom-pair vector;
    physics, MLP_Y(y_ij). Their sum is the logit on every candidate."""
    dec = model.decoder
    n_atoms = y.shape[1]
    u = dec.w_sign(dec.step_query(h[:, 0])).view(h.shape[0], 2, -1)
    site = torch.einsum("bad,bsd->bas", h[:, 1 : n_atoms + 1], u)
    site = site[:, :, None] + site[:, None]
    return torch.stack([site, dec.u_sign(p), dec.physics(y.to(p.dtype))], -1).float()


def r_changes(r_pairs: np.ndarray, r_atoms: np.ndarray) -> list[list]:
    """The non-zero entries of R: ["bond", i, j, +-1] for i < j, then ["lone", i, +-2]."""
    i, j = np.nonzero(np.triu(r_pairs, 1))
    bonds = [["bond", int(a), int(c), int(r_pairs[a, c])] for a, c in zip(i, j, strict=True)]
    return bonds + [["lone", int(a), int(r_atoms[a])] for a in np.flatnonzero(r_atoms)]


def _label_anchors(batch: dict) -> torch.Tensor:
    """(A, A, 2) the admissible anchors i < j of sample 0's label class."""
    anchors = batch["r_anchors"][0] & batch["variant_mask"][0, :, None, None, None]
    return anchors.any(0) & torch.ones_like(anchors[0, ..., 0]).triu(1)[..., None]


def anchor_readout(model: KSMech, h, p, y, batch: dict, top: int = 10) -> dict:
    """The decoder on sample 0: the `top` anchors by probability with their logit terms, per
    atom the probability that the anchor involves it, the greedy R of the top anchor and,
    with a label class, the label's rank and the probability of its anchors."""
    logits = model.decoder.anchor_logits(h, p, y, batch)
    lp = anchor_log_probs(logits)
    terms = anchor_terms(model, h, p, y)[0]
    prob = lp[0].exp()
    labelled = bool(batch["variant_mask"][0].any())
    label = _label_anchors(batch) if labelled else torch.zeros_like(prob, dtype=torch.bool)
    flat = lp[0].flatten()
    vals, idx = flat.topk(min(top, int(torch.isfinite(flat).sum())))
    n = lp.shape[1]
    rows = []
    for f in idx.tolist():
        i, j, s = f // (2 * n), (f % (2 * n)) // 2, f % 2
        row = {"i": i, "j": j, "side": SIDE[s], "p": float(prob[i, j, s])}
        row |= {"logit": float(logits[0, i, j, s]), "in_label": bool(label[i, j, s])}
        rows.append(row | dict(zip(TERMS, terms[i, j, s].tolist(), strict=True)))
    r_pairs, r_atoms = greedy_edit(model.decoder, h, p, lp, batch)
    out = {
        "top": rows,
        "atom_probability": (prob.sum((1, 2)) + prob.sum((0, 2))).tolist(),
        "greedy": r_changes(r_pairs[0].cpu().numpy(), r_atoms[0].cpu().numpy()),
        "n_candidates": int(torch.isfinite(flat).sum()),
    }
    if labelled:
        out["label_rank"] = int(anchor_ranks(lp, batch)[0])
        out["label_probability"] = float(prob[label].sum())
    return out


@torch.no_grad()
def trace(model: KSMech, sample: MechanismSample, device: str = "cpu", top: int = 10) -> dict:
    """Every stage of the model on one sample, in the drawing it carries, as numpy arrays on
    the MECH layout (see the module docstring); `errors` holds the largest deviation of the
    rebuilt h^(0) and of every rebuilt layer from what the model computed."""
    model.eval()
    batch = to_device(collate_samples([sample]), torch.device(device))
    inp, z_hat, defined, _ = model.prepare(batch)
    parts = embedding_parts(model.encoder.embed, batch, inp)
    h, p, y, calls = encode_with_calls(model, batch)
    layers = [layer_trace(model, c, batch, z_hat, defined) for c in calls]
    h0 = sum(parts.values())
    hs = [h0] + [c.h_out for c in calls]

    def stack(key: str) -> np.ndarray:
        return torch.stack([x[key][0] for x in layers]).float().cpu().numpy()

    return {
        "parts": {k: v[0].float().cpu().numpy() for k, v in parts.items()},
        "h": torch.stack([x[0] for x in hs]).float().cpu().numpy(),
        "attn": stack("attn"),
        "spread": stack("spread"),
        "d_attn": stack("d_attn"),
        "d_ffn": stack("d_ffn"),
        "inputs": {k: inp[k][0].float().cpu().numpy() for k in ("atom_x", "orbital_x")},
        "readout": anchor_readout(model, h, p, y, batch, top),
        "errors": {
            "embedding": float((h0 - calls[0].args[0]).abs().max()),
            "layers": [x["error"] for x in layers],
        },
    }


def ensure_pool(keys: list[str], loader: TokenCache, qm: QMConfig | None = None) -> list[str]:
    """Token files and Wiberg sidecars of every molecule in `keys`, each computed (GFN2-xTB at
    a force-field conformer, into the QM cache) when missing. Returns the keys built."""
    from ksmech.physics.textbook import ensure_tokens
    from ksmech.qm.bond_index import bond_index_path, save_bond_indices
    from ksmech.qm.records import cache_path
    from ksmech.qm.validation import compute_fixture_records

    qm = qm or QMConfig()
    built = set(ensure_tokens(keys, loader.tokens_dir, qm))
    for key in keys:
        if bond_index_path(loader.bond_index_dir, key, 0, qm.method).exists():
            continue
        record = compute_fixture_records({key: key}, qm)[key]
        record_path = cache_path(qm.cache_dir, key, 0, qm.method)
        save_bond_indices(record, record_path, loader.bond_index_dir)
        built.add(key)
    return sorted(built)


def sample_from_smiles(smiles: str, loader: TokenCache) -> MechanismSample:
    """The model inputs of a reactant pool given as SMILES (one molecule or several, '.'
    separated), without a label: tokens and the Lewis structure with its Wiberg indices."""
    rt = ReactantTokens.from_smiles(smiles, loader)
    sample = reactant_sample("smiles", "smiles", rt)
    sample.bonds, sample.lone_pairs, sample.wiberg = reactant_lewis(rt)
    return sample
