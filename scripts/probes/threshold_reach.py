"""Reach of the thresholds no reviser resolved (open-shell plan P0a, step 9), read-only.

    PYTHONPATH=src python scripts/probes/threshold_reach.py [--source pmechdb reactmech]
        [--split molecule] [--processed-dir DIR] [--tokens-dir DIR] [--qm-dir DIR]
        [--out-dir DIR] [--workers 4] [--limit N]

Five thresholds sit in the token path with no ablation behind them: `tau_atom` (an atom is a
centre of an orbital at >= 0.15 of its Mulliken population, `DescriptorConfig`), the fractional
band 0.2 < n < 1.8 that keeps a token neither donor nor acceptor (`CandidateConfig`), the
0.05 eV merge of degenerate canonical levels in the spectral set (`DescriptorConfig.
degeneracy_ev`), `SMEAR_TOL` = 0.05 (the fermi_smeared test, `localize.is_fermi_smeared`) and
the 0.5 / 1.5 occupations that split localization into the virtual, singly and doubly occupied
blocks (`qm.fukui.OCC_THRESHOLD`, `localize.DOUBLE_THRESHOLD`; the candidate rule's `empty` and
`full` are the same two numbers). For each alternative value it counts, per part of the split
and per source, the orbital tokens, molecules, samples and gold anchors that would change.

Recomputation reuses the package's functions with another config, on what the token files
store: `build_descriptors` on each LMO column (`C_loc`, `S`, `ao_atom`, the Mulliken populations
of `classify.atom_populations`, the stored block spectrum) for tau_atom, then
`physics.pairs.pair_flags` / `valid_pairs` for the within-molecule cell flags it moves;
`spectral_entries` on `full_energies` / `full_weights` for the merge; `donor_candidates` /
`acceptor_candidates` for the band; `is_fermi_smeared` and the block predicates on the canonical
MO occupations of the QM record (`qm.records`: path and header verification, then only the
occupation, energy and spin members are read, never C, S or F). The current values are run as
variants too: every count must be 0 there (`self_check`).

A token is changed by a variant (`tokens`, with the reason in `tokens_<category>`) when its
inputs move (`input`; tau: the centre set or any descriptor derived from it; merge: the spectral
set, compared as a set of (E, p) rows within ATOL, or the retained fraction), its donor or
acceptor mask flips (`donor`, `acceptor`), its block's MO membership moves (`block`; block
split: every token of both blocks when an MO crosses the occupied / virtual line, the occupied
tokens when it only crosses the doubly / singly line; the frontier atom inputs move with it), or
its molecule's fermi_smeared flag flips (`flag`; SMEAR_TOL: the flag is molecule-level, so every
token and atom of the molecule counts). Apart from these, tau also flips within-molecule pair
cells' own_antibond / site_self flags (`cells_flags`, of which `cells_valid` flip the legacy
pair_valid mask; `tokens_cell` are the tokens on them), and the merge moves each changed
spectral set by a 1-D Wasserstein distance counted in `shift_*` bins. A molecule is affected
when a token is; a sample when a reactant fragment is. A sample's gold anchor is affected when
an anchor (i, j) of any variant of its edit-label class has i or j among the centres (before or
after) of a changed token (`anchor_samples`; `anchor_samples_cells` adds the tokens on flipped
cells); `anchor_samples_strict` asks for an atom-specific change: i or j enters or leaves a
centre set or its weight moves, or a token centred on it changes mask, block or flag (a
spectral-set change alone does not count). Of the samples with ArrowLabels (`labels` and, v18,
`labels_all`), `label_mask_samples` have a labeled donor whose donor mask or a labeled acceptor
whose acceptor mask flips, `label_token_samples` a labeled token changed in any way.

Samples are read as their stored JSON part only (fragment keys, edit label, ArrowLabels, flag)
from verified shards: the full unpack decompresses every pair cell, which this report never
reads, and is tied to one store layout. Writes `<out-dir>/<source>/threshold_reach.json`
(default out-dir: the processed dir) and nothing else.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
from rich.console import Console
from rich.table import Table

from ksmech.arrows import steps, store
from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, source_dir
from ksmech.orbitals.build import DEFAULT_TOKENS_DIR, tokens_path
from ksmech.orbitals.classify import atom_populations
from ksmech.orbitals.descriptors import (
    DescriptorConfig,
    OrbitalDescriptors,
    build_descriptors,
    spectral_entries,
)
from ksmech.orbitals.localize import DOUBLE_THRESHOLD, SMEAR_TOL, is_fermi_smeared
from ksmech.orbitals.tokens import CandidateConfig, MoleculeTokens, OrbitalKind, load_tokens
from ksmech.physics.pairs import pair_flags, valid_pairs
from ksmech.processes import in_processes
from ksmech.qm import records
from ksmech.qm.fukui import OCC_THRESHOLD
from ksmech.qm.xtb_calc import QMConfig

log = logging.getLogger("threshold_reach")

REACH_VERSION = 1
REPORT_NAME = "threshold_reach.json"
# float descriptors that agree within this are unchanged: the samples keep them as float32, so
# a smaller move (a 1e-8 MO folding into a kept spectral level) never reaches the model
ATOL = 1e-6
CONFORMER_ID = 0  # the token cache's conformer (`arrows.reactants.TokenCache`)
# 2026-10-27 ReactMech token build: 19,543 molecules in 42 min 56 s at 8 workers (plan, P0a)
TOKEN_CPU_SECONDS = (42 * 60 + 56) * 8 / 19_543
QM_MEMBERS = ("occupations", "orbital_energies_ev", "unpaired")  # all the flag tests read

CURRENT = {
    "tau_atom": DescriptorConfig().tau_atom,
    "fractional_band": tuple(CandidateConfig().fractional_band),
    "spectral_merge_ev": DescriptorConfig().degeneracy_ev,
    "smear_tol": SMEAR_TOL,
    "block_split": (OCC_THRESHOLD, DOUBLE_THRESHOLD),
}


def scaled_band(scale: float) -> tuple[float, float]:
    """The fractional band with both edges' distances to the integer occupations 0 and 2
    scaled: 0.5 halves them, 2 doubles them (the band stays symmetric under n -> 2 - n)."""
    lo, hi = CURRENT["fractional_band"]
    return round(lo * scale, 9), round(2.0 - (2.0 - hi) * scale, 9)


ALTERNATIVES = {
    "tau_atom": (0.10, 0.20),
    "fractional_band": (scaled_band(0.5), scaled_band(2.0)),
    "spectral_merge_ev": (1e-4, 0.01, 0.1),
    "smear_tol": (SMEAR_TOL / 2, SMEAR_TOL * 2),
    # halfway from (0.5, 1.5) to the empty / doubly occupied levels, and to the SOMO's n = 1;
    # both keep the n -> 2 - n symmetry of the current pair
    "block_split": ((0.25, 1.75), (0.75, 1.25)),
}
NEEDS = {
    "tau_atom": "token rebuild",
    "fractional_band": "mask channel",
    "spectral_merge_ev": "token rebuild",
    "smear_tol": "mask channel",
    "block_split": "token rebuild",
}
# the descriptors tau_atom decides (the centre list and everything computed from it)
TAU_FIELDS = (
    "n_centers",
    "center_weights",
    "center_spd",
    "cos_a",
    "cos_b",
    "n_axis_centers",
    "overlap_population",
    "distance_ab",
    "radius_gyration",
    "center_atoms",
    "all_center_atoms",
    "all_center_weights",
    "center_s_coeff",
    "center_p_norm",
)
CATEGORIES = ("input", "donor", "acceptor", "block", "flag")  # why a token changed
# 1-D Wasserstein distance between a token's stored and rebuilt spectral sets: bin edges, eV
SHIFT_EDGES_EV = (1e-3, 1e-2, 1e-1)
SHIFT_BINS = ("shift_lt_1meV", "shift_1_10meV", "shift_10_100meV", "shift_ge_100meV")
SAMPLE_COUNTS = (
    "samples",
    "anchor_samples",
    "anchor_samples_strict",
    "anchor_samples_cells",
    "label_mask_samples",
    "label_token_samples",
    "samples_flag_flipped",
)
MOLECULE_COUNTS = (
    "tokens",
    *(f"tokens_{c}" for c in CATEGORIES),
    "tokens_cell",
    "cells_flags",
    "cells_valid",
    *SHIFT_BINS,
    "molecules",
    "molecules_flag",
)


@dataclass(frozen=True)
class Variant:
    threshold: str
    value: float | tuple[float, float]
    current: bool = False

    @property
    def name(self) -> str:
        v = self.value
        text = f"{v[0]:g}-{v[1]:g}" if isinstance(v, tuple) else f"{v:g}"
        return f"{self.threshold}={text}"


def all_variants() -> list[Variant]:
    """Each threshold at its current value (the self-check), then its alternatives."""
    out = []
    for name, value in CURRENT.items():
        out += [Variant(name, value, True), *(Variant(name, v) for v in ALTERNATIVES[name])]
    return out


VARIANTS = all_variants()


# ---- one molecule


class MoleculeView:
    """The stored tokens of one molecule (and its QM occupations), with what every variant
    compares against: the centre sets, the default masks and the default cell flags."""

    def __init__(self, t: MoleculeTokens, occupations: SimpleNamespace | None):
        self.t, self.rec = t, occupations
        # (nao, n_orbitals), C order: column k is a strided view, as the block matrix column
        # the build handed `classify_block` and `build_descriptors` (a contiguous copy changes
        # BLAS's S @ c in the last bit)
        self.C = np.ascontiguousarray(t.C_loc)
        self.coords = np.array([a.position for a in t.atoms], dtype=float).reshape(-1, 3)
        self.occ = [k for k, o in enumerate(t.orbitals) if o.kind == OrbitalKind.OCCUPIED]
        self.virt = [k for k, o in enumerate(t.orbitals) if o.kind == OrbitalKind.VIRTUAL]
        self.centres = [_centre_set(o.descriptors) for o in t.orbitals]
        self.donors, self.acceptors = mask_sets(t, CandidateConfig())
        self._pops: list[np.ndarray] | None = None
        self._flags: tuple[dict, np.ndarray] | None = None

    @property
    def populations(self) -> list[np.ndarray]:
        """Mulliken population of every orbital on every atom, as the classifier computes it."""
        if self._pops is None:
            t, n = self.t, len(self.t.atoms)
            self._pops = [atom_populations(c, t.S, t.ao_atom, n) for c in self.C.T]
        return self._pops

    @property
    def flags(self) -> tuple[dict, np.ndarray]:
        """(pair_flags, valid_pairs) of the stored tokens."""
        if self._flags is None:
            self._flags = pair_flags([self.t]), valid_pairs([self.t])
        return self._flags


def _centre_set(d: OrbitalDescriptors) -> set[int]:
    return {int(a) for a in d.all_center_atoms}


def mask_sets(t: MoleculeTokens, config: CandidateConfig) -> tuple[set[int], set[int]]:
    """Orbital indices of the donor and acceptor candidates (`MoleculeTokens` methods)."""
    pos = {id(o): k for k, o in enumerate(t.orbitals)}
    donors = {pos[id(o)] for o in t.donor_candidates(config)}
    return donors, {pos[id(o)] for o in t.acceptor_candidates(config)}


def _hit() -> dict:
    """An empty hit: the changed orbital indices by category, the touched atoms (strict and
    broad anchor readings), the tokens on flipped cells and their atoms, the flipped cells,
    the spectral shifts by bin, and whether a molecule-level flag moved."""
    out: dict = {c: set() for c in (*CATEGORIES, "cell", "cell_atoms", "strict", "broad")}
    return out | {"n_cells": 0, "n_cells_valid": 0, "shift": [0] * len(SHIFT_BINS)}


def _same(a, b) -> bool:
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape:
        return False
    if np.issubdtype(x.dtype, np.integer) and np.issubdtype(y.dtype, np.integer):
        return bool(np.array_equal(x, y))
    return bool(np.allclose(x, y, rtol=0.0, atol=ATOL, equal_nan=True))


def rebuilt_centres(m: MoleculeView, k: int, config: DescriptorConfig) -> OrbitalDescriptors:
    """Orbital k's stored descriptors with the centre-derived fields rebuilt under `config`
    (`build_descriptors` on the stored column; the occupation it is handed is a dummy, so only
    `TAU_FIELDS` are taken from it)."""
    t, o = m.t, m.t.orbitals[k]
    d = o.descriptors
    n = len(d.full_weights)
    new = build_descriptors(
        m.C[:, k],
        m.populations[k],
        t.S,
        t.ao_atom,
        t.ao_l,
        m.coords,
        np.sqrt(d.full_weights),
        d.full_energies,
        np.zeros(n),
        np.arange(n),
        config,
    )
    return replace(d, **{f: getattr(new, f) for f in TAU_FIELDS})


def _moved_atoms(old: OrbitalDescriptors, new: OrbitalDescriptors) -> set[int]:
    """Atoms that enter or leave the centre set, or whose centre weight moves."""
    w_old = dict(zip(map(int, old.all_center_atoms), old.all_center_weights, strict=True))
    w_new = dict(zip(map(int, new.all_center_atoms), new.all_center_weights, strict=True))
    moved = set(w_old) ^ set(w_new)
    return moved | {a for a in set(w_old) & set(w_new) if abs(w_old[a] - w_new[a]) > ATOL}


def tau_hit(m: MoleculeView, tau: float) -> dict:
    """Tokens whose centres move at `tau_atom`, then the cell flags those moves flip."""
    hit, config = _hit(), DescriptorConfig(tau_atom=tau)
    new = [rebuilt_centres(m, k, config) for k in range(len(m.t.orbitals))]
    for k, (o, d) in enumerate(zip(m.t.orbitals, new, strict=True)):
        if not all(_same(getattr(o.descriptors, f), getattr(d, f)) for f in TAU_FIELDS):
            hit["input"].add(k)
            hit["strict"] |= _moved_atoms(o.descriptors, d)
            hit["broad"] |= m.centres[k] | _centre_set(d)
    if hit["input"]:
        _cell_hits(m, new, hit)
    return hit


def _cell_hits(m: MoleculeView, new: list[OrbitalDescriptors], hit: dict) -> None:
    """Within-molecule cells whose own_antibond / site_self flag flips with the new centres,
    the tokens on them, and the cells whose legacy pair_valid flips."""
    orbitals = [replace(o, descriptors=d) for o, d in zip(m.t.orbitals, new, strict=True)]
    t_new = replace(m.t, orbitals=orbitals)
    (old, valid), flags = m.flags, pair_flags([t_new])
    flip = np.zeros(valid.shape, dtype=bool)
    for name in ("own_antibond", "site_self"):
        flip |= old[name] != flags[name]
    rows, cols = np.nonzero(flip)
    tokens = {m.occ[r] for r in rows} | {m.virt[c] for c in cols}
    hit["cell"] |= tokens
    hit["cell_atoms"] |= set().union(*(m.centres[k] | _centre_set(new[k]) for k in tokens))
    hit["n_cells"] = int(flip.sum())
    hit["n_cells_valid"] = int((valid != valid_pairs([t_new])).sum())


def rebuilt_spectrum(d: OrbitalDescriptors, config: DescriptorConfig) -> tuple:
    """(energies, weights, retained fraction) of the spectral set `spectral_entries` gives the
    stored block spectrum under `config` (p_k = full_weights; the MO indices are not read)."""
    n = len(d.full_weights)
    _, e, w, retained = spectral_entries(
        np.sqrt(d.full_weights), d.full_energies, np.arange(n), config
    )
    return e, w, retained


def _set_rows(e, w) -> np.ndarray:
    """The non-padding (E, p) rows of a spectral set, by energy."""
    e, w = np.asarray(e, dtype=float), np.asarray(w, dtype=float)
    keep = w > 0
    order = np.argsort(e[keep], kind="stable")
    return np.column_stack([e[keep][order], w[keep][order]])


def same_spectrum(d: OrbitalDescriptors, rebuilt: tuple) -> bool:
    """The rebuilt spectral set equals the stored one as a set of (E, p) rows (the set encoder
    is permutation invariant), and so does the retained fraction."""
    e, w, retained = rebuilt
    same = _same(_set_rows(e, w), _set_rows(d.spectral_energies, d.spectral_weights))
    return same and abs(retained - d.retained_fraction) <= ATOL


def spectral_shift(d: OrbitalDescriptors, rebuilt: tuple) -> float:
    """1-D Wasserstein distance (eV) between the stored and the rebuilt spectral sets, each a
    distribution of weight over energy: how far the set encoder's input moves."""
    a, b = _set_rows(d.spectral_energies, d.spectral_weights), _set_rows(*rebuilt[:2])
    e = np.concatenate([a[:, 0], b[:, 0]])
    w = np.concatenate([a[:, 1], -b[:, 1]])
    order = np.argsort(e, kind="stable")
    return float(np.sum(np.abs(np.cumsum(w[order])[:-1]) * np.diff(e[order])))


def merge_hit(m: MoleculeView, tol: float) -> dict:
    hit, config = _hit(), DescriptorConfig(degeneracy_ev=tol)
    for k, o in enumerate(m.t.orbitals):
        rebuilt = rebuilt_spectrum(o.descriptors, config)
        if not same_spectrum(o.descriptors, rebuilt):
            hit["input"].add(k)
            hit["broad"] |= m.centres[k]
            shift = spectral_shift(o.descriptors, rebuilt)
            hit["shift"][int(np.searchsorted(SHIFT_EDGES_EV, shift, side="right"))] += 1
    return hit


def _mask_hit(m: MoleculeView, config: CandidateConfig, hit: dict | None = None) -> dict:
    hit = hit if hit is not None else _hit()
    donors, acceptors = mask_sets(m.t, config)
    hit["donor"] |= donors ^ m.donors
    hit["acceptor"] |= acceptors ^ m.acceptors
    for k in hit["donor"] | hit["acceptor"]:
        hit["strict"] |= m.centres[k]
        hit["broad"] |= m.centres[k]
    return hit


def band_hit(m: MoleculeView, band: tuple[float, float]) -> dict:
    return _mask_hit(m, replace(CandidateConfig(), fractional_band=tuple(band)))


def _whole_block(m: MoleculeView, hit: dict, category: str, tokens, atoms=()) -> dict:
    """A molecule-level change: `tokens` (and their centres, plus `atoms`) all count."""
    touched = set(atoms).union(*(m.centres[k] for k in tokens))
    hit[category] |= set(tokens)
    hit["strict"] |= touched
    hit["broad"] |= touched
    hit["molecule"] = True
    return hit


def smear_hit(m: MoleculeView, tol: float) -> dict:
    """The molecule's fermi_smeared flag at `tol` differs from the stored one: every token and
    atom of the molecule enters or leaves the slice."""
    hit = _hit()
    if is_fermi_smeared(m.rec, tol) == m.t.fermi_smeared:
        return hit
    return _whole_block(m, hit, "flag", range(len(m.t.orbitals)), range(len(m.t.atoms)))


def blocks(occupations: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Per canonical MO: 0 virtual (n < lo), 1 singly (lo <= n < hi), 2 doubly occupied block,
    the predicates of `localize.localize_record` and `_localize_occupied`."""
    return np.where(occupations < lo, 0, np.where(occupations >= hi, 2, 1))


def block_hit(m: MoleculeView, split: tuple[float, float]) -> dict:
    """MOs that change localization block at (lo, hi), and the candidate rule's empty / full
    at the same two numbers."""
    lo, hi = split
    hit = _mask_hit(m, replace(CandidateConfig(), empty=lo, full=hi))
    occ = np.asarray(m.rec.occupations, dtype=float)
    old, new = blocks(occ, OCC_THRESHOLD, DOUBLE_THRESHOLD), blocks(occ, lo, hi)
    if np.array_equal(old, new):
        return hit
    crossed = bool(np.any((old == 0) != (new == 0)))
    return _whole_block(m, hit, "block", range(len(m.t.orbitals)) if crossed else m.occ)


HANDLERS = {
    "tau_atom": tau_hit,
    "fractional_band": band_hit,
    "spectral_merge_ev": merge_hit,
    "smear_tol": smear_hit,
    "block_split": block_hit,
}


def compact(hit: dict) -> dict:
    """The hit as plain sorted tuples, empty fields dropped; `tokens` is their union."""
    tokens = set().union(*(hit[c] for c in CATEGORIES))
    sets = (*CATEGORIES, "cell", "cell_atoms", "strict", "broad")
    out = {c: tuple(sorted(hit[c])) for c in sets if hit[c]}
    out |= {k: hit[k] for k in ("n_cells", "n_cells_valid") if hit[k]}
    if any(hit["shift"]):
        out["shift"] = tuple(hit["shift"])
    if tokens:
        out["tokens"] = tuple(sorted(tokens))
    if hit.get("molecule"):
        out["molecule"] = True
    return out


def molecule_reach(t: MoleculeTokens, occupations: SimpleNamespace | None) -> dict:
    """Every variant on one molecule: {variant name: compact hit}; the current values are
    always kept (the self-check), the alternatives only when they change something."""
    m = MoleculeView(t, occupations)
    hits = {}
    for v in VARIANTS:
        hit = compact(HANDLERS[v.threshold](m, v.value))
        if v.current or hit:
            hits[v.name] = hit
    return {
        "n_atoms": len(t.atoms),
        "n_orbitals": len(t.orbitals),
        "smeared": bool(t.fermi_smeared),
        "hits": hits,
    }


def occupation_record(t: MoleculeTokens, qm_dir: Path) -> SimpleNamespace:
    """The canonical MO occupations, energies and spin of the QM record the tokens were built
    from, verified by `records.verify_record`; raises FileNotFoundError when absent."""
    path = records.cache_path(qm_dir, t.smiles, t.conformer_id, t.method)
    if not path.is_file():
        raise FileNotFoundError(path)
    records.verify_record(path, t.smiles, t.conformer_id, t.method)
    with np.load(path, allow_pickle=False) as z:
        occ, energies, unpaired = (z[name] for name in QM_MEMBERS)
        return SimpleNamespace(
            occupations=occ, orbital_energies_ev=energies, unpaired=int(unpaired)
        )


def molecule_task(task: tuple[str, str, str, str]) -> dict:
    """(key, tokens dir, QM dir, method) -> {key, skip reason or '', molecule_reach ...}."""
    key, tokens_dir, qm_dir, method = task
    path = tokens_path(Path(tokens_dir), key, CONFORMER_ID, method)
    if not path.exists():
        return {"key": key, "skip": steps.SkipReason.MISSING_TOKENS.value}
    try:
        t = load_tokens(path)
        occupations = occupation_record(t, Path(qm_dir))
    except prov.StaleArtifact as exc:
        return {"key": key, "skip": prov.STALE_ARTIFACT, "detail": str(exc)}
    except FileNotFoundError as exc:
        return {"key": key, "skip": "MISSING_QM_RECORD", "detail": str(exc)}
    return {"key": key, "skip": "", **molecule_reach(t, occupations)}


# ---- samples


class SampleRef(NamedTuple):
    step_id: str
    keys: tuple[str, ...]  # fragment token-cache keys, in global order
    anchor_atoms: tuple[int, ...]  # global atoms of every anchor of every class variant
    donors: tuple[int, ...]  # global token ids of the ArrowLabels' donor sets
    acceptors: tuple[int, ...]
    labeled: bool  # the sample has ArrowLabels
    smeared: bool  # the stored fermi_smeared flag


def sample_ref(meta: dict) -> SampleRef:
    """The part of a stored sample's JSON this report reads (present in v17 and v18)."""
    label = meta.get("edit_label") or {}
    atoms = {int(a) for anchors in label.get("anchors", []) for i, j, _ in anchors for a in (i, j)}
    arrows = [*meta.get("labels", []), *meta.get("labels_all", [])]
    return SampleRef(
        str(meta["step_id"]),
        tuple(meta["fragment_keys"]),
        tuple(sorted(atoms)),
        tuple(sorted({int(x) for lab in arrows for x in lab["donor_ids"]})),
        tuple(sorted({int(x) for lab in arrows for x in lab["acceptor_ids"]})),
        bool(arrows),
        bool(meta.get("fermi_smeared", False)),
    )


def iter_sample_refs(view: Path, limit: int = 0) -> Iterator[SampleRef]:
    """The SampleRefs of one split view, shard by shard. The view, its split file and the index
    are verified by `steps.sample_ids`, every shard read by `provenance.verify_clean`, as the
    sample loaders do; only each shard's `meta` member is decompressed."""
    ids = steps.sample_ids(view)
    wanted = set(ids[:limit] if limit else ids)
    root = store.store_dir(view.parent)
    shards = json.loads((root / "index.json").read_text())["shards"]
    for name, members in sorted(shards.items()):
        rows = [k for k, sid in enumerate(members) if sid in wanted]
        if not rows:
            continue
        prov.verify_clean(root / name, "sample_shard", steps.SAMPLE_VERSION)
        with np.load(root / name, allow_pickle=False) as z:
            metas = json.loads(str(z["meta"]))
        yield from (sample_ref(metas[k]) for k in rows)


def _locate(index: int, offsets: np.ndarray) -> tuple[int, int]:
    f = int(np.searchsorted(offsets, index, side="right")) - 1
    return f, int(index - offsets[f])


def sample_hits(ref: SampleRef, mols: dict[str, dict], name: str) -> dict[str, bool]:
    """Which of `SAMPLE_COUNTS` the variant `name` reaches on one sample."""
    infos = [mols[k] for k in ref.keys]
    hits = [info["hits"].get(name, {}) for info in infos]
    atom_off = np.cumsum([0, *(i["n_atoms"] for i in infos)])
    orb_off = np.cumsum([0, *(i["n_orbitals"] for i in infos)])

    anchors = [_locate(a, atom_off) for a in ref.anchor_atoms]

    def atoms_in(*fields: str) -> bool:
        return any(loc in hits[f].get(x, ()) for f, loc in anchors for x in fields)

    def tokens_in(ids, field: str) -> bool:
        local = [_locate(x - int(atom_off[-1]), orb_off) for x in ids]
        return any(k in hits[f].get(field, ()) for f, k in local)

    old_flag = any(i["smeared"] for i in infos)
    new_flag = any(i["smeared"] != bool(h.get("flag")) for i, h in zip(infos, hits, strict=True))
    labels = ref.labeled
    return {
        "samples": any(hits),
        "anchor_samples": atoms_in("broad"),
        "anchor_samples_strict": atoms_in("strict"),
        "anchor_samples_cells": atoms_in("broad", "cell_atoms"),
        "label_mask_samples": labels
        and (tokens_in(ref.donors, "donor") or tokens_in(ref.acceptors, "acceptor")),
        "label_token_samples": labels and tokens_in(ref.donors + ref.acceptors, "tokens"),
        "samples_flag_flipped": old_flag != new_flag,
    }


# ---- counting


def part_reach(refs: list[SampleRef], mols: dict[str, dict], name: str) -> dict[str, int]:
    """Every count of `MOLECULE_COUNTS` (over the part's distinct molecules) and
    `SAMPLE_COUNTS` (over its samples) for the variant `name`."""
    counts = Counter(dict.fromkeys((*MOLECULE_COUNTS, *SAMPLE_COUNTS), 0))
    for key in sorted({k for r in refs for k in r.keys}):
        hit = mols[key]["hits"].get(name, {})
        if not hit:
            continue
        counts["molecules"] += 1
        counts["molecules_flag"] += int(hit.get("molecule", False))
        counts["tokens"] += len(hit.get("tokens", ()))
        for c in (*CATEGORIES, "cell"):
            counts[f"tokens_{c}"] += len(hit.get(c, ()))
        counts["cells_flags"] += hit.get("n_cells", 0)
        counts["cells_valid"] += hit.get("n_cells_valid", 0)
        for b, n in zip(SHIFT_BINS, hit.get("shift", (0,) * len(SHIFT_BINS)), strict=True):
            counts[b] += n
    for ref in refs:
        counts.update({k: int(v) for k, v in sample_hits(ref, mols, name).items()})
    return dict(counts)


def part_totals(refs: list[SampleRef], mols: dict[str, dict]) -> dict[str, int]:
    keys = {k for r in refs for k in r.keys}
    return {
        "samples": len(refs),
        "samples_with_labels": sum(r.labeled for r in refs),
        "samples_fermi_smeared": sum(r.smeared for r in refs),
        "molecules": len(keys),
        "tokens": sum(mols[k]["n_orbitals"] for k in keys),
        "atoms": sum(mols[k]["n_atoms"] for k in keys),
    }


def examples(mols: dict[str, dict], name: str, n: int = 5) -> list[dict]:
    """The molecules with the most changed tokens under the variant `name`."""
    rows = [
        {"key": k, "tokens": len(m["hits"][name].get("tokens", ())), "of": m["n_orbitals"]}
        for k, m in mols.items()
        if m["hits"].get(name)
    ]
    return sorted(rows, key=lambda r: (-r["tokens"], r["key"]))[:n]


def rebuild_cost(n_molecules: int) -> dict:
    return {
        "token_files": n_molecules,
        "cpu_hours_estimate": round(n_molecules * TOKEN_CPU_SECONDS / 3600, 2),
        "basis": "2026-10-27 ReactMech token build, 19,543 molecules in 42 min 56 s at 8 "
        "workers (open-shell plan P0a); the sample store and its views are rebuilt after it",
    }


def reach_report(parts: dict[str, list[SampleRef]], mols: dict[str, dict]) -> dict:
    """The report body of one source: totals, the self-check and the reach of every
    alternative, per part and over all parts."""
    parts = {**parts, "all": [r for p in parts.values() for r in p]}
    body: dict = {"totals": {p: part_totals(refs, mols) for p, refs in parts.items()}}
    body["self_check"] = {
        v.name: {p: part_reach(refs, mols, v.name) for p, refs in parts.items()}
        for v in VARIANTS
        if v.current
    }
    n_all = body["totals"]["all"]["molecules"]
    body["reach"] = [
        {
            "threshold": v.threshold,
            "value": v.value,
            "current": CURRENT[v.threshold],
            "needs": NEEDS[v.threshold],
            **({"cost": rebuild_cost(n_all)} if NEEDS[v.threshold] == "token rebuild" else {}),
            "parts": {p: part_reach(refs, mols, v.name) for p, refs in parts.items()},
            "examples": examples(mols, v.name),
        }
        for v in VARIANTS
        if not v.current
    ]
    return body


def self_check_failures(body: dict) -> list[str]:
    """Current-value variants that change anything (must be none)."""
    return [
        f"{name} {part}: {k}={n}"
        for name, by_part in body["self_check"].items()
        for part, counts in by_part.items()
        for k, n in counts.items()
        if n
    ]


# ---- the run


def read_parts(source: str, split: str, processed_dir: Path, limit: int):
    """{part: SampleRefs} and the views read."""
    parts, views = {}, []
    for part in steps.PARTS:
        view = steps.samples_path(source, f"{split}_{part}", processed_dir)
        parts[part] = list(iter_sample_refs(view, limit))
        views.append(view)
    return parts, views


def run_molecules(keys: list[str], args) -> tuple[dict[str, dict], Counter]:
    method = QMConfig().method
    tasks = [(k, str(args.tokens_dir), str(args.qm_dir), method) for k in keys]
    mols, skipped = {}, Counter()
    for k, res in enumerate(in_processes(molecule_task, tasks, args.workers, chunksize=8), 1):
        if res["skip"]:
            skipped[res["skip"]] += 1
            log.warning("%s: %s %s", res["key"], res["skip"], res.get("detail", ""))
        else:
            mols[res["key"]] = res
        if k % 2000 == 0:
            log.info("%d / %d molecules", k, len(keys))
    return mols, skipped


def drop_skipped(parts, mols) -> tuple[dict[str, list[SampleRef]], Counter]:
    """Samples whose every fragment was analysed; the others counted by part."""
    kept, skipped = {}, Counter()
    for part, refs in parts.items():
        kept[part] = [r for r in refs if all(k in mols for k in r.keys)]
        skipped[part] = len(refs) - len(kept[part])
    return kept, skipped


def flag_mismatches(parts, mols) -> int:
    """Samples whose stored fermi_smeared flag is not the OR of their token files' flags."""
    return sum(
        r.smeared != any(mols[k]["smeared"] for k in r.keys) for p in parts.values() for r in p
    )


def run_source(source: str, args, console: Console) -> tuple[Path, dict]:
    start = time.monotonic()
    parts, views = read_parts(source, args.split, args.processed_dir, args.limit)
    keys = sorted({k for refs in parts.values() for r in refs for k in r.keys})
    mols, skipped_mols = run_molecules(keys, args)
    parts, skipped_samples = drop_skipped(parts, mols)
    body = {"source": source, "split": args.split, "current": CURRENT, "needs": NEEDS}
    body |= reach_report(parts, mols)
    body["skipped"] = {"molecules": dict(skipped_mols), "samples": dict(skipped_samples)}
    body["sample_flag_mismatches"] = flag_mismatches(parts, mols)
    body["seconds"] = round(time.monotonic() - start, 1)
    inputs: dict[str, str] = {}
    for view in views:
        inputs |= prov.file_input(view)
    inputs |= prov.file_input(store.store_dir(views[0].parent) / "index.json")
    out = write_report(source_dir(source, args.out_dir) / REPORT_NAME, body, inputs, args)
    print_report(body, console)
    console.print(f"wrote {out} in {body['seconds']} s")
    return out, body


def write_report(path: Path, body: dict, inputs: dict[str, str], args) -> Path:
    config = {
        "source": body["source"],
        "split": args.split,
        "limit": args.limit,
        "current": CURRENT,
        "alternatives": ALTERNATIVES,
        "atol": ATOL,
    }
    content = prov.sha256_text(json.dumps(body, sort_keys=True, default=str))
    header = prov.stamp("threshold_reach", REACH_VERSION, inputs, config, content)
    return prov.write_json(path, body, header)


def _pct(n: int, total: int) -> str:
    return f"{n} ({n / total:.1%})" if total else str(n)


def _detail(threshold: str, c: dict) -> str:
    """The variant-specific count beside the common columns."""
    if threshold == "tau_atom":
        return f"cells {c['cells_flags']} ({c['cells_valid']} pair_valid)"
    if threshold == "spectral_merge_ev":
        return f"shift >= 10 meV: {c['shift_10_100meV'] + c['shift_ge_100meV']}"
    if threshold == "fractional_band":
        return f"donor {c['tokens_donor']}, acceptor {c['tokens_acceptor']}"
    if threshold == "smear_tol":
        return f"sample flag flips {c['samples_flag_flipped']}"
    return f"molecules re-blocked {c['molecules_flag']}"


def print_report(body: dict, console: Console) -> None:
    table = Table(title=f"threshold reach, {body['source']} {body['split']} split")
    cols = ("threshold", "part", "tokens", "molecules", "samples", "anchor", "strict", "label")
    for col in (*cols, "detail"):
        table.add_column(col, justify="left" if col in ("threshold", "part", "detail") else "right")
    for row in body["reach"]:
        name = Variant(row["threshold"], row["value"]).name
        for part in steps.PARTS:
            c, tot = row["parts"][part], body["totals"][part]
            table.add_row(
                name if part == steps.PARTS[0] else "",
                part,
                _pct(c["tokens"], tot["tokens"]),
                _pct(c["molecules"], tot["molecules"]),
                _pct(c["samples"], tot["samples"]),
                _pct(c["anchor_samples"], tot["samples"]),
                _pct(c["anchor_samples_strict"], tot["samples"]),
                _pct(c["label_mask_samples"], tot["samples_with_labels"]),
                _detail(row["threshold"], c),
            )
    console.print(table)
    console.print(f"totals: {body['totals']}")
    failures = self_check_failures(body)
    console.print("self-check at the current values: " + ("; ".join(failures) or "0 changed"))
    console.print(f"skipped: {body['skipped']}, flag mismatches: {body['sample_flag_mismatches']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", nargs="+", default=["pmechdb", "reactmech"])
    parser.add_argument("--split", default="molecule", help="split strategy of the views")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    parser.add_argument("--qm-dir", type=Path, default=records.DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: --processed-dir")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="first N samples per part (0: all)")
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir or args.processed_dir
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    console = Console(width=170)
    failed = False
    for source in args.source:
        _, body = run_source(source, args, console)
        failed |= bool(self_check_failures(body))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
