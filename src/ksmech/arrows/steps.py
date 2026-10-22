"""Training samples: one aligned elementary step with the tokens of its whole reactant side.

`MechanismSample` holds the model inputs of every reactant fragment in one global index space
(atoms first, then orbitals; `ksmech.arrows.reactants`), the fragment of every token, the
candidate masks, the token-to-centre incidence (orbital, atom, weight), the `ArrowLabel`s with
their sites, and the products' resonance key. It also stores the pair-track features z_rs of
every (occupied r, virtual s) pair (`physics.pairs.PAIR_FEATURES`), computed at K = eps = 1
(`physics.params.REDUCED`; `params.to_params` applies other values), their `defined` mask, the
sign of G (bookkeeping: the logs drop it), the validity filter, the contact amplitude's argmax
site pairs, and the symmetric atom-atom Coulomb matrix (also at eps = 1). The full
`MoleculeTokens` (coefficients, S, F) are not copied into every sample: `fragment_keys` names
the token files, and `MechanismSample.tokens(loader)` reloads them.

A step becomes a sample only when every reactant fragment has tokens and every arrow is
aligned or absorbed; every other step is skipped with a reason code and counted.

`build_all_samples(source, kernels)` writes the sample store (`arrows.store`) and one
`data/processed/<source>/samples_<split>.json` view per split, with
`split = "<strategy>_<part>"`: strategy `molecule` (the default), `scaffold`, `loco-<class>`
(leave one mechanism class out) or, only when named, `random_leaky` (per step), part `train`,
`val` or `test`.
`collate_samples` pads a batch and builds the masks the model needs.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console

from ksmech.arrows import store
from ksmech.arrows.align import AlignFlag, ArrowLabel, align_step, representatives
from ksmech.arrows.apply_predicted import bond_break_label
from ksmech.arrows.reactants import (
    CorrespondenceError,
    MissingTokens,
    ReactantTokens,
    TokenCache,
    TokenLoader,
)
from ksmech.data import chem, splits
from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_training_steps, source_dir, steps_path
from ksmech.data.resonance import canonical_resonance_key
from ksmech.data.schema import ElementaryStep
from ksmech.orbitals.tokens import MoleculeTokens
from ksmech.physics.pairs import (
    PAIR_FEATURES,
    PairKernels,
    PairStats,
    atom_coulomb_matrix,
    fit_pair_stats,
    pair_block,
    valid_pairs,
)
from ksmech.physics.params import REDUCED, KSParams, to_params

log = logging.getLogger(__name__)

SAMPLE_VERSION = 10
PARTS = ("train", "val", "test")
STRATEGIES = ("molecule", "scaffold", "loco", "random_leaky")
# what `ksmech-arrows samples` builds unless told otherwise: never the per-step random split,
# which leaks molecules across sets on PMechDB (no pathway ids; docs/review_2026-10-21.md, D1)
DEFAULT_STRATEGIES = ("molecule", "scaffold", "loco")


class SkipReason(StrEnum):
    MISSING_TOKENS = "MISSING_TOKENS"  # a reactant fragment has no token file (yet)
    CORRESPONDENCE = "CORRESPONDENCE"  # drawn fragment cannot be matched onto its record
    UNALIGNED_ARROW = "UNALIGNED_ARROW"  # an arrow has no label (a lone heterolysis)
    LABEL_FILTERED = "LABEL_FILTERED"  # a label has no pair left that pair_valid allows


@dataclass
class MechanismSample:
    step_id: str
    source: str
    fragment_keys: list[str]  # token-cache key (map-free canonical SMILES) per fragment
    atom_categorical: np.ndarray  # (A, 2) int64: atomic number, formal charge
    atom_features: np.ndarray  # (A, Fa) float32, missing values 0
    atom_missing: np.ndarray  # (A, Fa) bool
    atom_fragment: np.ndarray  # (A,) int64
    atom_map: np.ndarray  # (A,) int64 atom-map number, 0 for unmapped atoms
    atom_positions: np.ndarray  # (A, 3) float32 Angstrom, for distances only
    orbital_features: np.ndarray  # (O, Fo) float32
    orbital_spectra: np.ndarray  # (O, 8, 2) float32
    orbital_fragment: np.ndarray  # (O,) int64
    orbital_occupied: np.ndarray  # (O,) bool
    donor_mask: np.ndarray  # (O,) bool: donor candidates
    acceptor_mask: np.ndarray  # (O,) bool: acceptor candidates (before the partner rule)
    orbital_centroids: np.ndarray  # (O, 3) float32
    incidence: np.ndarray  # (E, 3) float32 rows (orbital j, atom g, weight w_g of token j)
    pair_z: np.ndarray  # (n_occ, n_virt, 11) float32 z_rs at K = eps = 1, PAIR_FEATURES order
    pair_defined: np.ndarray  # (n_occ, n_virt, 11) bool
    pair_gap_sign: np.ndarray  # (n_occ, n_virt) int8 sign of G (bookkeeping, not a feature)
    pair_valid: np.ndarray  # (n_occ, n_virt) bool: donor, acceptor, not an excluded pair
    pair_contact_sites: np.ndarray  # (n_occ, n_virt, 2) int64 global atoms of the contact
    #   amplitude's argmax (donor site, acceptor site), oriented kernel; -1 for other kernels
    atom_coulomb: np.ndarray  # (A, A) float32 KS2 between atoms at eps = 1, symmetric
    labels: list[ArrowLabel] = field(default_factory=list)
    product_key: str = ""
    product_key_flag: str = ""
    fermi_smeared: bool = False  # a reactant fragment is outside the closed-shell domain
    label_filter: dict[str, int] = field(default_factory=dict)  # labeled (donor, acceptor)
    #   cells that pair_valid rejects, by group (`_filter_group`); never dropped silently
    inputs: dict[str, str] = field(default_factory=dict)  # provenance: every token file and
    #   contact block this sample was built from (`file:<path>` -> content sha256)

    @property
    def n_atoms(self) -> int:
        return len(self.atom_categorical)

    @property
    def n_orbitals(self) -> int:
        return len(self.orbital_features)

    def tokens(self, loader: TokenLoader) -> list[MoleculeTokens]:
        """The full token objects of every fragment, in fragment order."""
        return [loader(key) for key in self.fragment_keys]

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(np.ascontiguousarray(value))
            elif f.name == "labels":
                value = [lab.to_dict() for lab in value]
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, d: dict) -> MechanismSample:
        kw = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
        kw["labels"] = [ArrowLabel.from_dict(x) for x in d["labels"]]
        return cls(**kw)


def _incidence(rt: ReactantTokens) -> np.ndarray:
    rows = [(j, g, w) for j in range(rt.n_orbitals) for g, w in rt.centres(rt.n_atoms + j).items()]
    return np.array(rows, dtype=np.float32).reshape(-1, 3)


def _atom_maps(rt: ReactantTokens) -> np.ndarray:
    out = np.zeros(rt.n_atoms, dtype=np.int64)
    for g, m in rt.mol_atom.items():
        out[g] = rt.mol.GetAtomWithIdx(m).GetAtomMapNum()
    return out


def make_sample(
    step: ElementaryStep, rt: ReactantTokens, labels: list[ArrowLabel], kernels: PairKernels
) -> MechanismSample:
    """The sample of an aligned step (the product key and pair features are computed here)."""
    key = canonical_resonance_key(chem.parse_mapped(step.products_smiles))
    sample = reactant_sample(step.step_id, str(step.source), rt, kernels)
    sample.labels, sample.product_key, sample.product_key_flag = labels, key.key, str(key.flag)
    return sample


def reactant_sample(
    step_id: str, source: str, rt: ReactantTokens, kernels: PairKernels
) -> MechanismSample:
    """The model inputs of a reactant side, without labels or products (also used for
    hand-built reactant sets, e.g. the textbook selectivity checks)."""
    tok = [f.tokens for f in rt.fragments]
    block = pair_block(tok, kernels, REDUCED)
    return MechanismSample(
        step_id=step_id,
        source=source,
        fragment_keys=[f.key for f in rt.fragments],
        atom_categorical=np.concatenate([t.atom_categorical() for t in tok]),
        atom_features=np.concatenate([t.atom_features() for t in tok]).astype(np.float32),
        atom_missing=np.concatenate([t.atom_missing() for t in tok]),
        atom_fragment=rt.atom_fragment.astype(np.int64),
        atom_map=_atom_maps(rt),
        atom_positions=rt.atom_positions.astype(np.float32),
        orbital_features=np.concatenate([t.orbital_features() for t in tok]).astype(np.float32),
        orbital_spectra=np.concatenate([t.orbital_spectra() for t in tok]).astype(np.float32),
        orbital_fragment=rt.orbital_fragment.astype(np.int64),
        orbital_occupied=rt.occupied.copy(),
        donor_mask=rt.donor_ok.copy(),
        acceptor_mask=rt.acceptor_ok.copy(),
        orbital_centroids=rt.centroids.astype(np.float32),
        incidence=_incidence(rt),
        pair_z=block.z.astype(np.float32),
        pair_defined=block.defined,
        pair_gap_sign=block.gap_sign,
        pair_valid=valid_pairs(tok, rt.candidate),
        pair_contact_sites=block.sites,
        atom_coulomb=atom_coulomb_matrix(tok, REDUCED).astype(np.float32),
        fermi_smeared=any(t.fermi_smeared for t in tok),
    )


def _pair_index(sample: MechanismSample) -> tuple[dict[int, int], dict[int, int]]:
    """Global token id -> row (occupied) / column (virtual) of the pair arrays."""
    n = sample.n_atoms
    occ = np.flatnonzero(sample.orbital_occupied)
    virt = np.flatnonzero(~sample.orbital_occupied)
    return {n + int(j): i for i, j in enumerate(occ)}, {n + int(j): i for i, j in enumerate(virt)}


def _symbol(rt: ReactantTokens, atom: int) -> str:
    f = int(np.searchsorted(rt.atom_offsets, atom, side="right")) - 1
    return rt.fragments[f].tokens.symbol(atom - rt.atom_offsets[f])


def _filter_group(rt: ReactantTokens, lab: ArrowLabel, r: int, s: int, w: float) -> str:
    """Why a labeled cell is invalid: which side carries >= w on the other's site."""
    sym = f"{_symbol(rt, lab.donor_site)}->{_symbol(rt, lab.acceptor_site)}"
    if rt.centres(s).get(lab.donor_site, 0.0) >= w:
        return f"{sym}: acceptor on the donor site"
    if rt.centres(r).get(lab.acceptor_site, 0.0) >= w:
        return f"{sym}: donor on the acceptor site"
    return f"{sym}: other"


def filter_labels(sample: MechanismSample, rt: ReactantTokens) -> list[ArrowLabel] | None:
    """The labels restricted to the pairs `pair_valid` allows (the bonded-site filter and the
    antibonding-partner rule). A donor (acceptor) with no valid partner in the other set leaves
    its set and the label is flagged LABEL_FILTERED; every rejected cell is counted by group in
    `sample.label_filter`. The teacher pair (r*, s*) is chosen among valid cells. None when a
    label has no valid cell left (the step is skipped as LABEL_FILTERED)."""
    row, col = _pair_index(sample)

    def valid(r: int, s: int) -> bool:
        return r in row and s in col and bool(sample.pair_valid[row[r], col[s]])

    out = []
    w = rt.candidate.site_weight
    for lab in sample.labels:
        for r in lab.donor_ids:
            for s in lab.acceptor_ids:
                if not valid(r, s):
                    key = _filter_group(rt, lab, r, s, w)
                    sample.label_filter[key] = sample.label_filter.get(key, 0) + 1
        donors = frozenset(r for r in lab.donor_ids if any(valid(r, s) for s in lab.acceptor_ids))
        acceptors = frozenset(s for s in lab.acceptor_ids if any(valid(r, s) for r in donors))
        if not donors or not acceptors:
            return None
        trimmed = donors != lab.donor_ids or acceptors != lab.acceptor_ids
        lab = replace(lab, donor_ids=donors, acceptor_ids=acceptors)
        if trimmed:
            lab = replace(lab, flags=lab.flags | {AlignFlag.LABEL_FILTERED})
        r, s = representatives(rt, lab, valid)
        breaks, forced = bond_break_label(rt, lab, s)
        out.append(
            replace(
                lab, teacher_donor=r, teacher_acceptor=s, breaks_bd=breaks, breaks_bd_forced=forced
            )
        )
    return out


def sample_for_step(
    step: ElementaryStep, loader: TokenLoader, kernels: PairKernels
) -> MechanismSample | SkipReason:
    with prov.collecting() as used:
        try:
            rt = ReactantTokens.from_smiles(step.reactants_smiles, loader)
        except MissingTokens:
            return SkipReason.MISSING_TOKENS
        except CorrespondenceError:
            return SkipReason.CORRESPONDENCE
        alignment = align_step(step, rt)
        if not alignment.complete:
            return SkipReason.UNALIGNED_ARROW
        sample = make_sample(step, rt, alignment.labels, kernels)
    labels = filter_labels(sample, rt)
    if labels is None:
        log.info("%s: a label has no valid pair: %s", step.step_id, sample.label_filter)
        return SkipReason.LABEL_FILTERED
    sample.labels = labels
    sample.inputs = dict(sorted(used.items()))
    return sample


def build_step_samples(
    steps: Iterable[ElementaryStep], loader: TokenLoader, kernels: PairKernels
) -> tuple[dict[str, MechanismSample], dict[str, SkipReason]]:
    """Samples of every step that has one, and the skip reason of every other step."""
    samples, skipped = {}, {}
    for step in steps:
        out = sample_for_step(step, loader, kernels)
        if isinstance(out, MechanismSample):
            samples[step.step_id] = out
        else:
            skipped[step.step_id] = out
    return samples, skipped


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", name).strip("-")


def split_ids(steps: list[ElementaryStep], strategy: str, seed: int = 0) -> dict[str, list[str]]:
    """Split name (`<strategy>_<part>`) -> step ids. Raises ValueError when the strategy
    cannot run on these steps (leave-one-class-out without class labels)."""
    if strategy == "random":
        raise ValueError("the per-step split is now `random_leaky` and must be asked for by name")
    if strategy == "molecule":
        per = {"molecule": splits.split_by_molecule(steps, seed=seed)}
    elif strategy == "random_leaky":
        per = {"random_leaky": splits.split_by_pathway(steps, seed=seed)}
    elif strategy == "scaffold":
        per = {"scaffold": splits.split_by_scaffold(steps, seed=seed)}
    elif strategy == "loco":
        if not any(s.mechanism_class is not None for s in steps):
            raise ValueError("no step has a mechanism_class; leave-one-class-out is impossible")
        per = {
            f"loco-{_safe(c)}": sp
            for c, sp in splits.leave_one_class_out_splits(steps, seed=seed).items()
        }
    else:
        raise ValueError(f"unknown split strategy {strategy!r}; one of {STRATEGIES}")
    return {f"{name}_{part}": getattr(sp, part) for name, sp in per.items() for part in PARTS}


def samples_path(source: str, split: str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    """`samples_<split>.json`: the split's view over the sample store (`arrows.store`)."""
    return source_dir(source, processed_dir) / f"samples_{split}.json"


def splits_path(source: str, strategy: str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    return source_dir(source, processed_dir) / f"splits_{strategy}.json"


def write_split_file(
    path: Path, strategy: str, seed: int, ids: dict[str, list[str]], parquet: Path
) -> Path:
    """The step ids of every part of one strategy, stamped with the parquet they came from."""
    header = prov.stamp(
        "split",
        SAMPLE_VERSION,
        prov.file_input(parquet),
        {"strategy": strategy, "seed": seed},
        prov.sha256_text(json.dumps(ids, sort_keys=True)),
    )
    return prov.write_json(path, {"splits": ids}, header)


def load_split_file(path: Path) -> dict[str, list[str]]:
    """The ids of a split file; raises StaleArtifact when its parquet changed."""
    prov.verify(path, "split", SAMPLE_VERSION)
    return json.loads(Path(path).read_text())["splits"]


def split_report(split: str, ids: list[str], samples: dict, skipped: dict) -> dict:
    """The counts of one split: steps, samples, skip reasons, labels, filtered labels."""
    kept = [samples[i] for i in ids if i in samples]
    reasons = Counter(str(skipped[i]) for i in ids if i in skipped)
    return {
        "split": split,
        "n_steps": len(ids),
        "n_samples": len(kept),
        "skipped": dict(reasons.most_common()),
        "n_labels": sum(len(s.labels) for s in kept),
        "n_labels_filtered": sum(
            AlignFlag.LABEL_FILTERED in lab.flags for s in kept for lab in s.labels
        ),
        "label_filter": dict(sum((Counter(s.label_filter) for s in kept), Counter()).most_common()),
        "n_fermi_smeared": sum(s.fermi_smeared for s in kept),
    }


def _config(split: str, kernels: PairKernels, seed: int) -> dict:
    return {"split": split, "seed": seed, "sample_version": SAMPLE_VERSION, **kernels.describe()}


def build_all_samples(
    source: str,
    kernels: PairKernels,
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    loader: TokenLoader | None = None,
    seed: int = 0,
    console: Console | None = None,
) -> dict:
    """Build every step's sample once into the store, then one split file per strategy and one
    view per split. A strategy that cannot run on the source is reported, not raised."""
    console = console or Console()
    parquet = steps_path(source, processed_dir)
    steps = read_training_steps(parquet)
    samples, skipped = build_step_samples(steps, loader or TokenCache(), kernels)
    out_dir = source_dir(source, processed_dir)
    index = store.write_store(out_dir, samples, skipped, parquet, SAMPLE_VERSION)
    report = {
        "n_training_steps": len(steps),
        "n_samples": len(samples),
        "skipped": dict(Counter(str(r) for r in skipped.values()).most_common()),
        "label_filter": split_report("all", sorted(samples), samples, {})["label_filter"],
        "splits": {},
        "strategies_not_applicable": {},
    }
    for strategy in strategies:
        try:
            ids = split_ids(steps, strategy, seed)
        except ValueError as exc:
            report["strategies_not_applicable"][strategy] = str(exc)
            continue
        split_file = write_split_file(
            splits_path(source, strategy, processed_dir), strategy, seed, ids, parquet
        )
        for split, members in ids.items():
            counts = split_report(split, members, samples, skipped)
            view = samples_path(source, split, processed_dir)
            config = _config(split, kernels, seed)
            store.write_view(view, members, counts, index, split_file, config)
            report["splits"][split] = counts
            if split.endswith("_train"):
                train = [samples[i] for i in members if i in samples]
                stats_path = pair_stats_path(source, split[: -len("_train")], processed_dir)
                stats_path.write_text(pair_stats(train).to_json() + "\n")
    return report


def pair_stats_path(source: str, name: str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    """`pair_feature_stats_<name>.json`: the standardization statistics of one split
    strategy, fitted on its training part."""
    return source_dir(source, processed_dir) / f"pair_feature_stats_{name}.json"


def pair_stats(samples: list[MechanismSample], params: KSParams | None = None) -> PairStats:
    """Statistics of the defined pair features of `samples` at the physical K and eps."""
    params = params or KSParams()
    return fit_pair_stats(
        (to_params(s.pair_z, params, PAIR_FEATURES), s.pair_defined) for s in samples
    )


def load_samples(path: Path, allow_stale: bool = False) -> list[MechanismSample]:
    """The samples of one split view (`samples_<split>.json`). The view, the store index and
    every shard read are verified first (parquet, split file, token files, contact blocks);
    StaleArtifact names what changed. `allow_stale` is for diagnostic CLIs only: the mismatch
    is logged and the store is read anyway."""
    return store.load_view(Path(path), MechanismSample, SAMPLE_VERSION, allow_stale)


class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path):
        self.samples = load_samples(path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> MechanismSample:
        return self.samples[i]


def _pad(arrays: list[np.ndarray], length: int, fill=0) -> torch.Tensor:
    """Stack arrays along a new batch axis, padding axis 0 to `length` with `fill`."""
    out = np.full((len(arrays), length, *arrays[0].shape[1:]), fill, dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return torch.from_numpy(out)


def _token_masks(batch: list[MechanismSample], n_atoms: int, n_orb: int) -> dict:
    """Per-token masks in the batch layout: atoms at [0, A), orbitals at [A, A + O)."""
    b, n = len(batch), n_atoms + n_orb
    m = {
        k: torch.zeros(b, n, dtype=torch.bool)
        for k in (
            "token_mask",
            "is_atom",
            "is_orbital",
            "occupied",
            "virtual",
            "donor_mask",
            "acceptor_mask",
        )
    }
    fragment = torch.full((b, n), -1, dtype=torch.long)
    for i, s in enumerate(batch):
        na, no = s.n_atoms, s.n_orbitals
        orb = slice(n_atoms, n_atoms + no)
        occ = torch.from_numpy(s.orbital_occupied)
        m["token_mask"][i, :na] = m["token_mask"][i, orb] = True
        m["is_atom"][i, :na] = True
        m["is_orbital"][i, orb] = True
        m["occupied"][i, orb], m["virtual"][i, orb] = occ, ~occ
        m["donor_mask"][i, orb] = torch.from_numpy(s.donor_mask)
        m["acceptor_mask"][i, orb] = torch.from_numpy(s.acceptor_mask)
        fragment[i, :na] = torch.from_numpy(s.atom_fragment)
        fragment[i, orb] = torch.from_numpy(s.orbital_fragment)
    real = m["token_mask"]
    m["fragment"] = fragment
    m["same_fragment"] = (
        (fragment[:, :, None] == fragment[:, None, :]) & real[:, :, None] & real[:, None, :]
    )
    return m


def _incidence_matrix(batch: list[MechanismSample], n_atoms: int, n_orb: int) -> torch.Tensor:
    """(B, N, A) token-to-centre weights: an orbital row holds w_a over its centres, an atom
    row is one-hot on itself (the site pointer's support)."""
    out = torch.zeros(len(batch), n_atoms + n_orb, n_atoms)
    for i, s in enumerate(batch):
        idx = torch.arange(s.n_atoms)
        out[i, idx, idx] = 1.0
        if len(s.incidence):
            j = torch.from_numpy(s.incidence[:, 0]).long() + n_atoms
            g = torch.from_numpy(s.incidence[:, 1]).long()
            out[i, j, g] = torch.from_numpy(s.incidence[:, 2])
    return out


def _label_tensors(batch: list[MechanismSample], n_atoms: int, n_orb: int) -> dict:
    """Multi-hot donor / acceptor sets over batch token positions and the site atoms."""
    b, n = len(batch), n_atoms + n_orb
    k = max(1, max(len(s.labels) for s in batch))
    out = {
        "donor_sets": torch.zeros(b, k, n, dtype=torch.bool),
        "acceptor_sets": torch.zeros(b, k, n, dtype=torch.bool),
        "donor_site": torch.full((b, k), -1, dtype=torch.long),
        "acceptor_site": torch.full((b, k), -1, dtype=torch.long),
        "label_mask": torch.zeros(b, k, dtype=torch.bool),
    }
    for i, s in enumerate(batch):

        def pos(t: int, s=s) -> int:
            return t if t < s.n_atoms else n_atoms + t - s.n_atoms

        for j, lab in enumerate(s.labels):
            out["donor_sets"][i, j, [pos(t) for t in lab.donor_ids]] = True
            out["acceptor_sets"][i, j, [pos(t) for t in lab.acceptor_ids]] = True
            out["donor_site"][i, j] = lab.donor_site
            out["acceptor_site"][i, j] = lab.acceptor_site
            out["label_mask"][i, j] = True
    return out


def collate_samples(batch: list[MechanismSample]) -> dict:
    """Pad a batch. Token positions: atoms at [0, A), orbitals at [A, A + O), where A and O
    are the largest atom and orbital counts in the batch; sites index atoms directly."""
    n_atoms = max(s.n_atoms for s in batch)
    n_orb = max(s.n_orbitals for s in batch)
    out = {
        "step_id": [s.step_id for s in batch],
        "product_key": [s.product_key for s in batch],
        "atom_categorical": _pad([s.atom_categorical for s in batch], n_atoms),
        "atom_features": _pad([s.atom_features for s in batch], n_atoms),
        "atom_missing": _pad([s.atom_missing for s in batch], n_atoms, False),
        "atom_positions": _pad([s.atom_positions for s in batch], n_atoms),
        "orbital_features": _pad([s.orbital_features for s in batch], n_orb),
        "orbital_spectra": _pad([s.orbital_spectra for s in batch], n_orb),
        "orbital_centroids": _pad([s.orbital_centroids for s in batch], n_orb),
        "incidence": _incidence_matrix(batch, n_atoms, n_orb),
    }
    out.update(_token_masks(batch, n_atoms, n_orb))
    out.update(_label_tensors(batch, n_atoms, n_orb))
    return out


def make_loader(
    path: Path, batch_size: int = 16, shuffle: bool = True, **kwargs: Any
) -> torch.utils.data.DataLoader:
    """DataLoader over one samples file with `collate_samples`."""
    return torch.utils.data.DataLoader(
        SampleDataset(path),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_samples,
        **kwargs,
    )
