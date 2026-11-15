"""The out-of-distribution report of a checkpoint on one part of a split (`ksmech-eval ood`;
docs/decisions.md, 2026-11-15): what the anchor ranking does away from the run's training
distribution, and which inputs it leans on.

Five sections, all on the anchor (NLL, top-1/3/5 with set marginalization) and the greedy R:

- `baseline`: the part as stored.
- `symmetry`: a probe that changes nothing of the chemistry. `fragment_order` relabels the
  fragments of every pool (the order the source writes its reactants in). Any change of the
  metrics, or of the top anchor (`top_changed`), is a defect of the model, not of the data.
- `reads`: permutation importance of the pair-level inputs, which `eval.importance` leaves
  untouched: the distances within a molecule, the Lewis-structure channels of the atom pairs,
  and the orbital-pair features by group (PAIR_GROUPS). A probe permutes its input between
  like positions (the atoms or orbitals of one molecule; the intra- or the intermolecular
  cells of a batch), so every value stays inside its training distribution.
- `terms`: the three terms of the anchor logit (`eval.embedding.anchor_terms`: site, pair,
  physics), each alone and each left out: which of them ranks the anchor.
- `coverage`: the samples by how often their edit signature (`signature`: the elements and
  the entries of R, no atom indices) occurs in the run's training part: >= 20 times, 1-19
  times, never. A model that fails on the seen signatures picks the wrong site of a reaction
  it knows.

A probe changes only what the encoder reads: the decoder's masks and the labels stay those of
the stored sample, so a probe never makes the label unreachable.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch

from ksmech.data import provenance as prov
from ksmech.model.inputs import to_device
from ksmech.model.loss import (
    anchor_log_probs,
    anchor_ranks,
    edit_matches,
    greedy_edit,
    step_loss,
)

OOD_VERSION = 1
SIGNATURE_KIND = "edit_signatures"
SEEN_OFTEN = 20  # training occurrences from which a signature counts as well covered
BUCKETS = ("seen_often", "seen_rarely", "unseen")
PAIR_GROUPS = {
    "pair:gap": ("g", "log_abs_g", "g2", "log_abs_g2", "own_g", "own_g2"),
    "pair:geometry": ("centroid_distance", "cross_coupling", "align_cos", "dihedral_cos"),
    "pair:connectivity": ("bonded", "bond_order", "graph_distance", "wbi"),
}
Probe = Callable[[dict, torch.Generator], dict]


# ---- the probes


def _rand(shape, gen: torch.Generator, like: torch.Tensor) -> torch.Tensor:
    """Uniform numbers from the host generator (a pass is reproducible on any device), on
    `like`'s device."""
    return torch.rand(shape, generator=gen).to(like.device)


def permute_fragments(batch: dict, gen: torch.Generator) -> dict:
    """The fragment labels of every pool shifted cyclically by a random non-zero offset: the
    same molecules written in another order. `same_fragment` is of the stored labels and stays."""
    frag = batch["fragment"]
    n = frag.max(1).values + 1  # fragments per pool
    shift = 1 + (_rand(len(frag), gen, frag) * (n - 1)).long().clamp(max=(n - 2).clamp(min=0))
    moved = (frag + shift[:, None]) % n[:, None]
    return batch | {"fragment": torch.where((frag >= 0) & (n[:, None] > 1), moved, frag)}


def _within(group: torch.Tensor, real: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """(B, T) for each row a permutation of its positions that moves a real token only to the
    place of a real token with the same `group` label; the padding stays. Two sorts by label,
    one stable and one with random ties, list every group's positions in the same slots."""
    t = group.shape[1]
    label = torch.where(real, group, group.max() + 1 + torch.arange(t, device=group.device))
    plain = torch.argsort(label, dim=1, stable=True)
    mixed = torch.argsort(label + _rand(label.shape, gen, label), dim=1)
    return torch.empty_like(plain).scatter_(1, plain, mixed)


def _take(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return x[torch.arange(len(x), device=x.device)[:, None], perm]


def permute_positions(batch: dict, gen: torch.Generator) -> dict:
    """The positions of the atoms, and the centroids of the orbitals, permuted within each
    molecule: the model's distances no longer belong to their pairs."""
    a = batch["atom_positions"].shape[1]
    frag, real = batch["fragment"], batch["token_mask"]
    atoms = _within(frag[:, :a], real[:, :a], gen)
    orbitals = _within(frag[:, a:], real[:, a:], gen)
    return batch | {
        "atom_positions": _take(batch["atom_positions"], atoms),
        "orbital_centroids": _take(batch["orbital_centroids"], orbitals),
    }


def permute_lewis(batch: dict, gen: torch.Generator) -> dict:
    """The atom-pair channels of the Lewis structure (bond order, graph distance, Wiberg
    index) under a relabelling of the atoms within each molecule: a real molecular graph, on
    the wrong atoms."""
    a = batch["bond_order"].shape[1]
    perm = _within(batch["fragment"][:, :a], batch["token_mask"][:, :a], gen)
    rows = torch.arange(len(perm), device=perm.device)[:, None, None]
    out = dict(batch)
    for name in ("bond_order", "graph_distance", "wiberg"):
        out[name] = batch[name][rows, perm[:, :, None], perm[:, None, :]]
    return out


def permute_pair_features(batch: dict, names: Iterable[str], gen: torch.Generator) -> dict:
    """The orbital-pair features `names` (those the batch stores), with their defined flags,
    permuted between the real cells of the batch: the intramolecular cells among themselves
    and the intermolecular ones among themselves."""
    stored = tuple(batch["pair_features"])
    cols = [stored.index(n) for n in names if n in stored]
    if not cols:
        return batch
    b, r, s = batch["pair_cells"].unbind(1)
    frag = batch["fragment"]
    inter = frag[b, batch["pair_row_pos"][b, r]] != frag[b, batch["pair_col_pos"][b, s]]
    src = _within(inter[None].long(), torch.ones_like(inter[None]), gen)[0]
    z, d = batch["pair_z"][b, r, s], batch["pair_defined"][b, r, s]  # (K, F) of the real cells
    z[:, cols], d[:, cols] = z[src][:, cols], d[src][:, cols]
    out_z, out_d = batch["pair_z"].clone(), batch["pair_defined"].clone()
    out_z[b, r, s], out_d[b, r, s] = z, d
    return batch | {"pair_z": out_z, "pair_defined": out_d}


def _chain(*probes: Probe) -> Probe:
    def run(batch: dict, gen: torch.Generator) -> dict:
        for p in probes:
            batch = p(batch, gen)
        return batch

    return run


def _pair_probe(names: Iterable[str]) -> Probe:
    names = tuple(names)
    return lambda batch, gen: permute_pair_features(batch, names, gen)


def read_probes(pair_names: Iterable[str]) -> dict[str, Probe]:
    """The probes of the `reads` section for a model that reads the pair features
    `pair_names`: a pair group is left out when the model reads none of its features."""
    pair_names = set(pair_names)
    groups = {k: tuple(n for n in v if n in pair_names) for k, v in PAIR_GROUPS.items()}
    out: dict[str, Probe] = {"distances": permute_positions, "lewis": permute_lewis}
    out |= {k: _pair_probe(v) for k, v in groups.items() if v}
    if groups["pair:geometry"]:
        out["geometry"] = _chain(permute_positions, out["pair:geometry"])
    if groups["pair:connectivity"]:
        out["connectivity"] = _chain(permute_lewis, out["pair:connectivity"])
    return out


SYMMETRY_PROBES: dict[str, Probe] = {"fragment_order": permute_fragments}


# ---- scoring


@torch.no_grad()
def score(model, loader, device, dtype, probe: Probe | None = None, seed: int = 0) -> dict:
    """Per sample, in the loader's order: `step_id`, the 0-based `rank` of the best admissible
    anchor, the anchor `nll`, the flat index of the `top` anchor in its batch's table and
    whether the greedy R is `exact`. With a `probe` the encoder reads the probed batch; the
    decoder's masks and the labels are the stored ones."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, list] = {k: [] for k in ("step_id", "rank", "nll", "top", "exact")}
    for batch in loader:
        batch = to_device(batch, device)
        seen = probe(batch, gen) if probe else None  # on the device: its gathers are large
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            h, p, y, _ = model.encode(batch if seen is None else seen)
            _, parts = step_loss(model.decoder, h, p, y, batch)
            lp = parts["lp_anchor"]
            exact = edit_matches(*greedy_edit(model.decoder, h, p, lp, batch), batch)
        out["step_id"] += list(batch["step_id"])
        out["rank"] += anchor_ranks(lp, batch).tolist()
        out["nll"] += parts["anchor_nll"].double().tolist()
        out["top"] += lp.flatten(1).argmax(1).tolist()
        out["exact"] += exact.tolist()
    return out


@torch.no_grad()
def term_ranks(model, loader, device, dtype) -> dict[str, list[int]]:
    """Per sample, the 0-based rank of the best admissible anchor when the logit is one term
    of TERMS alone (`site`, `pair`, `physics`) and when that term is left out (`no_site`, ...)."""
    from ksmech.eval.embedding import TERMS, anchor_terms

    model.eval()
    out: dict[str, list[int]] = {f"{pre}{t}": [] for t in TERMS for pre in ("", "no_")}
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            h, p, y, _ = model.encode(batch)
            terms = anchor_terms(model, h, p, y)  # (B, A, A, 2, 3)
        a = terms.shape[1]
        upper = torch.ones(a, a, dtype=torch.bool, device=terms.device).triu(1)
        ok = torch.stack([batch["can_form"], batch["can_break"]], -1) & upper[None, :, :, None]
        for k, name in enumerate(TERMS):
            alone, rest = terms[..., k], terms.sum(-1) - terms[..., k]
            for key, logits in ((name, alone), (f"no_{name}", rest)):
                lp = anchor_log_probs(logits.masked_fill(~ok, float("-inf")))
                out[key] += anchor_ranks(lp, batch).tolist()
    return out


def term_summary(ranks: dict[str, list[int]]) -> dict:
    """top-1/3/5 of each entry of `term_ranks`."""
    return {
        name: {f"top{k}": float((np.asarray(r) < k).mean()) for k in (1, 3, 5)} | {"n": len(r)}
        for name, r in ranks.items()
    }


def summary(scores: dict, keep: Iterable[bool] | None = None) -> dict:
    """n, anchor NLL, top-1/3/5 and the greedy-R rate of `score`'s rows (those of `keep`)."""
    mask = np.ones(len(scores["rank"]), dtype=bool) if keep is None else np.asarray(list(keep))
    rank = np.asarray(scores["rank"])[mask]
    n = len(rank)
    if not n:
        return {"n": 0}
    out = {"n": n, "anchor_nll": float(np.asarray(scores["nll"])[mask].mean())}
    out |= {f"top{k}": float((rank < k).mean()) for k in (1, 3, 5)}
    return out | {"r_exact": float(np.asarray(scores["exact"])[mask].mean())}


def per_sample(scores: dict) -> list[dict]:
    """`score`'s rows as records (step id, rank, anchor NLL, greedy R), for the report."""
    keys = ("step_id", "rank", "nll", "exact")
    rows = zip(*(scores[k] for k in keys), strict=True)
    return [dict(zip(keys, row, strict=True)) for row in rows]


def against(base: dict, probed: list[dict]) -> dict:
    """The mean over the repeats of a probe's metrics, their change against the baseline
    (`d_*`), and `top_changed`: the share of samples whose top anchor is another one."""
    rows = [summary(p) for p in probed]
    ref = summary(base)
    keys = [k for k in ref if k != "n"]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out |= {f"d_{k}": out[k] - ref[k] for k in keys}
    top = np.asarray(base["top"])
    out["top_changed"] = float(np.mean([(np.asarray(p["top"]) != top).mean() for p in probed]))
    return out | {"n": ref["n"], "repeats": len(probed)}


# ---- coverage by edit signature


def signature(sample) -> str | None:
    """The edit of a sample's label without its atoms: per variant of the class, the sorted
    bond entries (the two elements, the change) and atom entries (element, change); the
    smallest of the variants' texts, so the drawing does not matter. None without a label."""
    if sample.edit_label is None:
        return None
    z = sample.atom_categorical[:, 0]
    texts = []
    for v in sample.edit_label.variants:
        bonds = sorted((*sorted((int(z[i]), int(z[j]))), int(d)) for i, j, d in v.bonds)
        lone = sorted((int(z[i]), int(d)) for i, d in v.lone)
        texts.append(f"{bonds}|{lone}")
    return min(texts)


def training_signatures(view: Path, allow_stale: bool = False, subset: int = 0) -> Counter:
    """How often each signature occurs in a training view, or with `subset` in the run's
    random training subset of that size (`train.loop.train_subset`); cached beside the view as
    `edit_signatures_<view stem>[_sub<n>].json` and rebuilt when the view changed or the cache
    was built from a dirty tree."""
    from ksmech.arrows.steps import iter_samples, sample_ids
    from ksmech.train.loop import train_subset

    view = Path(view)
    stem = view.stem.removeprefix("samples_") + (f"_sub{subset}" if subset else "")
    cache = view.with_name(f"edit_signatures_{stem}.json")
    config = {"subset": int(subset)}
    try:
        header = prov.verify(cache, SIGNATURE_KIND, OOD_VERSION, config, prov.file_input(view))
        prov.require_clean(header, cache)
        return Counter(json.loads(cache.read_text())["counts"])
    except (prov.StaleArtifact, FileNotFoundError):
        pass
    ids = train_subset(sample_ids(view, allow_stale), subset) if subset else None
    counts: Counter = Counter()
    for s in iter_samples(view, allow_stale=allow_stale, ids=ids):
        if (sig := signature(s)) is not None:
            counts[sig] += 1
    header = prov.stamp(SIGNATURE_KIND, OOD_VERSION, prov.file_input(view), config)
    prov.write_json(cache, {"counts": dict(counts)}, header)
    return counts


def bucket(count: int) -> str:
    return BUCKETS[0] if count >= SEEN_OFTEN else BUCKETS[1] if count else BUCKETS[2]


def coverage(scores: dict, signatures: dict[str, str | None], train: Counter) -> dict:
    """`summary` of the samples of each bucket of BUCKETS, with the bucket's share of the
    part; `signatures`: step id -> the sample's signature."""
    names = [bucket(train.get(signatures.get(i) or "", 0)) for i in scores["step_id"]]
    n = max(len(names), 1)
    out = {}
    for b in BUCKETS:
        keep = [x == b for x in names]
        out[b] = summary(scores, keep) | {"share": sum(keep) / n}
    return out


# ---- the report


def ood_report(model, loader, device, dtype, repeats: int = 1, seed: int = 0, log=None) -> dict:
    """(report, baseline scores): `baseline`, `terms`, `symmetry` and `reads` of a model on a loader
    (the coverage needs the training part: `coverage`). `repeats` passes per probe, the
    seeds `seed + 1000 r`."""
    base = score(model, loader, device, dtype)
    report: dict = {"baseline": summary(base), "symmetry": {}, "reads": {}}
    report["terms"] = term_summary(term_ranks(model, loader, device, dtype))
    sections = (("symmetry", SYMMETRY_PROBES), ("reads", read_probes(model.pair_stats.names)))
    for section, probes in sections:
        for name, probe in probes.items():
            runs = [
                score(model, loader, device, dtype, probe, seed + 1000 * r) for r in range(repeats)
            ]
            report[section][name] = against(base, runs)
            if log:
                log(section, name, report[section][name])
    return report, base
