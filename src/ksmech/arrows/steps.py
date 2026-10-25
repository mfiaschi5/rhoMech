"""Training samples: one elementary step with the tokens of its whole reactant side.

`MechanismSample` holds the model inputs of every reactant fragment in one global index space
(atoms first, then orbitals; `ksmech.arrows.reactants`), the fragment of every token, the
candidate masks, the token-to-centre incidence (orbital, atom, weight), the target and the
products' sigma key. The target is the reaction matrix (spec v8): the step's edit-label class
with its anchors on the token atoms (`arrows.delta_be`), with the reactant's Lewis structure on
the same atoms (bonds, lone pairs, Wiberg indices: `arrows.lewis`). The `ArrowLabel`s, the
orbital reading of the arrows, are kept where every arrow aligns; `orbital_reading` says
whether they do. It also stores the pair-track features z_rs of
every (occupied r, virtual s) pair, computed at K = eps = 1 (`physics.params.REDUCED`;
`params.to_params` applies other values), their `defined` mask, the sign of G (bookkeeping: the
logs drop it), the validity filter, and the symmetric atom-atom Coulomb matrix (also at eps =
1). `pair_features` names the columns: the geometry-free stage-A features
(`physics.stage_a.STAGE_A_FEATURES`) in the sample store, or the legacy contact features
(`physics.pairs.PAIR_FEATURES`, with the contact amplitude's argmax sites) when a sample is
built with contact kernels (textbook cases, tests) or joined with the legacy contact sidecar
(`arrows.contacts`, for the physics floor and the GBDT). The full
`MoleculeTokens` (coefficients, S, F) are not copied into every sample: `fragment_keys` names
the token files, and `MechanismSample.tokens(loader)` reloads them.

A step becomes a sample when every reactant fragment has tokens and the step has an edit
label that is a valid edit of its reactant (`lewis.check_edit`); every other step is skipped with
a reason code and counted. A step whose arrows do not all align (a lone heterolysis has no
donor-acceptor pair) is a sample without ArrowLabels.

`build_all_samples(source)` writes the sample store (`arrows.store`) and one
`data/processed/<source>/samples_<split>.json` view per split, with
`split = "<strategy>_<part>"`: strategy `molecule` (the default), `scaffold`, `loco-<class>`
(leave one mechanism class out) or, only when named, `random_leaky` (per step), part `train`,
`val` or `test`.
`arrows.batches.collate_samples` pads a batch and builds the masks the model needs; it lives
apart so that this module does not import torch.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import numpy as np
from rich.console import Console

from ksmech.arrows import store
from ksmech.arrows.align import AlignFlag, ArrowLabel, align_step, representatives
from ksmech.arrows.apply_predicted import PredictedArrow, leaving_labels, with_leaving
from ksmech.arrows.delta_be import EditLabel, EditSkip, class_shape, edit_label, to_global
from ksmech.arrows.lewis import MissingBondIndex, check_edit, reactant_lewis
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
from ksmech.data.resonance import sigma_key
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
from ksmech.physics.stage_a import STAGE_A_FEATURES, stage_a_block
from ksmech.processes import chunked, in_processes

log = logging.getLogger(__name__)

SAMPLE_VERSION = 15  # 15: the reaction-matrix target and the reactant's Lewis structure
PARTS = ("train", "val", "test")
STRATEGIES = ("molecule", "scaffold", "loco", "random_leaky")
# what `ksmech-arrows samples` builds unless told otherwise: never the per-step random split,
# which leaks molecules across sets on PMechDB (no pathway ids; docs/review_2026-10-21.md, D1)
DEFAULT_STRATEGIES = ("molecule", "scaffold", "loco")


class SkipReason(StrEnum):
    MISSING_TOKENS = "MISSING_TOKENS"  # a reactant fragment has no token file (yet)
    STALE_ARTIFACT = "STALE_ARTIFACT"  # a fragment's token file or sidecar no longer matches
    CORRESPONDENCE = "CORRESPONDENCE"  # drawn fragment cannot be matched onto its record
    EDIT_FAILED = "EDIT_FAILED"  # the edit-label codes (`delta_be.EditSkip`), passed through
    PRODUCT_MISMATCH = "PRODUCT_MISMATCH"
    RESONANCE_STEP = "RESONANCE_STEP"
    NO_ANCHOR = "NO_ANCHOR"
    ATOM_WITHOUT_TOKEN = "ATOM_WITHOUT_TOKEN"
    NO_BOND_INDEX = "NO_BOND_INDEX"  # a reactant bond (or a fragment) has no Wiberg index
    R_CHECK_FAILED = "R_CHECK_FAILED"  # the label is no valid edit of the reactant (`check_edit`)


READING_ALIGNED = "aligned"  # every arrow has its ArrowLabel
READING_FILTERED = "LABEL_FILTERED"  # a label had no pair left that pair_valid allows


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
    pair_z: np.ndarray  # (n_occ, n_virt, F) float32 z_rs at K = eps = 1, `pair_features` order
    pair_defined: np.ndarray  # (n_occ, n_virt, F) bool
    pair_gap_sign: np.ndarray  # (n_occ, n_virt) int8 sign of G (bookkeeping, not a feature)
    pair_valid: np.ndarray  # (n_occ, n_virt) bool: donor, acceptor, not an excluded pair
    pair_contact_sites: np.ndarray  # (n_occ, n_virt, 2) int64 global atoms of the contact
    #   amplitude's argmax (donor site, acceptor site), oriented kernel; -1 for other kernels
    #   and for stage-A samples
    atom_coulomb: np.ndarray  # (A, A) float32 KS2 between atoms at eps = 1, symmetric
    labels: list[ArrowLabel] = field(default_factory=list)  # empty unless `orbital_reading`
    #   is READING_ALIGNED
    product_key: str = ""  # sigma key of the products (`data.resonance.sigma_key`)
    edit_label: EditLabel | None = None  # the R class and its anchors on the global atoms
    bonds: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int16))  # (E, 3)
    #   int16 (i, j, order), i < j: the reactant as drawn (Kekulé), implicit H included
    lone_pairs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))  # (A,)
    wiberg: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float16))  # (E,)
    orbital_reading: str = ""  # READING_ALIGNED, READING_FILTERED or the first unaligned reason
    fermi_smeared: bool = False  # a reactant fragment is outside the closed-shell domain
    label_filter: dict[str, int] = field(default_factory=dict)  # labeled (donor, acceptor)
    #   cells that pair_valid rejects, by group (`_filter_group`); never dropped silently
    inputs: dict[str, str] = field(default_factory=dict)  # provenance: every token file and
    #   contact block this sample was built from (`file:<path>` -> content sha256)
    pair_features: tuple[str, ...] = STAGE_A_FEATURES  # the columns of pair_z / pair_defined

    @property
    def n_atoms(self) -> int:
        return len(self.atom_categorical)

    @property
    def n_orbitals(self) -> int:
        return len(self.orbital_features)

    def tokens(self, loader: TokenLoader) -> list[MoleculeTokens]:
        """The full token objects of every fragment, in fragment order."""
        return [loader(key) for key in self.fragment_keys]


def _incidence(rt: ReactantTokens) -> np.ndarray:
    rows = [(j, g, w) for j in range(rt.n_orbitals) for g, w in rt.centres(rt.n_atoms + j).items()]
    return np.array(rows, dtype=np.float32).reshape(-1, 3)


def _atom_maps(rt: ReactantTokens) -> np.ndarray:
    out = np.zeros(rt.n_atoms, dtype=np.int64)
    for g, m in rt.mol_atom.items():
        out[g] = rt.mol.GetAtomWithIdx(m).GetAtomMapNum()
    return out


def make_sample(
    step: ElementaryStep,
    rt: ReactantTokens,
    labels: list[ArrowLabel],
    kernels: PairKernels | None = None,
) -> MechanismSample:
    """The sample of a step with these ArrowLabels (the product key and pair features are
    computed here)."""
    sample = reactant_sample(step.step_id, str(step.source), rt, kernels)
    sample.labels = labels
    sample.product_key = sigma_key(chem.parse_mapped(step.products_smiles))
    return sample


def reactant_sample(
    step_id: str, source: str, rt: ReactantTokens, kernels: PairKernels | None = None
) -> MechanismSample:
    """The model inputs of a reactant side, without labels or products (also used for
    hand-built reactant sets, e.g. the textbook selectivity checks). Pair features: stage A,
    or the legacy contact features with `kernels`."""
    tok = [f.tokens for f in rt.fragments]
    if kernels is None:
        block, names = stage_a_block(tok, REDUCED), STAGE_A_FEATURES
    else:
        block, names = pair_block(tok, kernels, REDUCED), PAIR_FEATURES
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
        pair_features=names,
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
    `sample.label_filter`. The teacher pair (r*, s*) is chosen among valid cells, and the
    leaving bonds are read on the teacher arrows in step order (`leaving_labels`). None when a
    label has no valid cell left (the sample then keeps no ArrowLabels: READING_FILTERED)."""
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
        out.append(replace(lab, teacher_donor=r, teacher_acceptor=s))
    arrows = [
        PredictedArrow(x.teacher_donor, x.donor_site, x.teacher_acceptor, x.acceptor_site)
        for x in out
    ]
    leaving = leaving_labels(rt, arrows, out)
    return [with_leaving(x, lab) for x, lab in zip(out, leaving, strict=True)]


def edit_target(step: ElementaryStep, rt: ReactantTokens) -> tuple | SkipReason:
    """(the edit label on the global atoms, bonds, lone pairs, Wiberg indices) of a step, or
    why it has none: an edit-label code, a bond without a Wiberg index, or a label that is no
    valid edit of the reactant (`lewis.check_edit`)."""
    label = edit_label(step)
    if not isinstance(label, EditSkip):
        label = to_global(label, rt.global_atom)
    if isinstance(label, EditSkip):
        return SkipReason(str(label))
    try:
        bonds, lone_pairs, wiberg = reactant_lewis(rt)
    except MissingBondIndex as exc:
        log.info("%s: %s", step.step_id, exc)
        return SkipReason.NO_BOND_INDEX
    problem = check_edit(label, bonds, lone_pairs)
    if problem is not None:
        log.info("%s: the edit label is no valid edit of the reactant (%s)", step.step_id, problem)
        return SkipReason.R_CHECK_FAILED
    return label, bonds, lone_pairs, wiberg


def _set_orbital_reading(sample: MechanismSample, rt: ReactantTokens, alignment) -> None:
    """The ArrowLabels restricted to valid pairs when every arrow aligns, and the reading."""
    if not alignment.complete:
        sample.labels, sample.orbital_reading = [], str(alignment.unaligned[0].reason)
        return
    labels = filter_labels(sample, rt)
    if labels is None:
        log.info("%s: a label has no valid pair: %s", sample.step_id, sample.label_filter)
        sample.labels, sample.orbital_reading = [], READING_FILTERED
    else:
        sample.labels, sample.orbital_reading = labels, READING_ALIGNED


def sample_for_step(
    step: ElementaryStep, loader: TokenLoader, kernels: PairKernels | None = None
) -> MechanismSample | SkipReason:
    """The sample of a step, or why it has none: the reactant's tokens and Lewis structure,
    the edit label, and the ArrowLabels when every arrow aligns (`orbital_reading`)."""
    with prov.collecting() as used:
        try:
            rt = ReactantTokens.from_smiles(step.reactants_smiles, loader)
        except MissingTokens as exc:
            log.info("%s: %s", step.step_id, exc)
            return SkipReason(exc.reason)
        except CorrespondenceError:
            return SkipReason.CORRESPONDENCE
        target = edit_target(step, rt)
        if isinstance(target, SkipReason):
            return target
        alignment = align_step(step, rt)
        sample = make_sample(step, rt, alignment.labels if alignment.complete else [], kernels)
    sample.edit_label, sample.bonds, sample.lone_pairs, sample.wiberg = target
    _set_orbital_reading(sample, rt, alignment)
    sample.inputs = dict(sorted(used.items()))
    return sample


def build_step_samples(
    steps: Iterable[ElementaryStep], loader: TokenLoader, kernels: PairKernels | None = None
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


CHUNK = 32  # steps per task of the parallel build


def _chunk_worker(task: tuple) -> tuple[dict, dict]:
    steps, loader, kernels = task
    return build_step_samples(steps, loader, kernels)


def build_step_samples_parallel(
    steps: list[ElementaryStep],
    loader: TokenLoader,
    kernels: PairKernels | None = None,
    n_workers: int = 1,
) -> tuple[dict[str, MechanismSample], dict[str, SkipReason]]:
    """`build_step_samples` in chunks of `CHUNK` consecutive steps, over `n_workers` spawned
    processes when above 1 (`processes.in_processes`). Each process keeps its own token and
    block memory; block files are written atomically, so two processes computing one block
    cannot corrupt it."""
    tasks = [(chunk, loader, kernels) for chunk in chunked(steps, CHUNK)]
    samples, skipped = {}, {}
    for done, (s, k) in enumerate(in_processes(_chunk_worker, tasks, n_workers), 1):
        samples.update(s)
        skipped.update(k)
        if done % 20 == 0 or done == len(tasks):
            log.info("samples: %d / %d chunks, %d samples", done, len(tasks), len(samples))
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
    """The counts of one split: steps, samples, skip reasons, edit-label class sizes and
    anchor kinds, orbital readings, ArrowLabels and filtered ArrowLabels."""
    kept = [samples[i] for i in ids if i in samples]
    reasons = Counter(str(skipped[i]) for i in ids if i in skipped)
    return {
        "split": split,
        "n_steps": len(ids),
        "n_samples": len(kept),
        "skipped": dict(reasons.most_common()),
        **class_shape([s.edit_label for s in kept if s.edit_label]),
        "orbital_reading": dict(Counter(s.orbital_reading for s in kept).most_common()),
        "n_labels": sum(len(s.labels) for s in kept),
        "n_labels_filtered": sum(
            AlignFlag.LABEL_FILTERED in lab.flags for s in kept for lab in s.labels
        ),
        "label_filter": dict(sum((Counter(s.label_filter) for s in kept), Counter()).most_common()),
        "n_fermi_smeared": sum(s.fermi_smeared for s in kept),
    }


def _config(split: str, seed: int) -> dict:
    return {
        "split": split,
        "seed": seed,
        "sample_version": SAMPLE_VERSION,
        "pair_features": list(STAGE_A_FEATURES),
    }


def build_all_samples(
    source: str,
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    loader: TokenLoader | None = None,
    seed: int = 0,
    console: Console | None = None,
    n_workers: int = 1,
) -> dict:
    """Build every step's sample once into the store, then one split file per strategy and one
    view per split. A strategy that cannot run on the source is reported, not raised. Pair
    features: stage A (the legacy contact features go to `arrows.contacts`)."""
    console = console or Console()
    parquet = steps_path(source, processed_dir)
    steps = read_training_steps(parquet)
    samples, skipped = build_step_samples_parallel(steps, loader or TokenCache(), None, n_workers)
    out_dir = source_dir(source, processed_dir)
    index = store.write_store(out_dir, samples, skipped, parquet, SAMPLE_VERSION)
    every = split_report("all", sorted(samples), samples, {})
    report = {
        "n_training_steps": len(steps),
        "n_samples": len(samples),
        "skipped": dict(Counter(str(r) for r in skipped.values()).most_common()),
        **{k: every[k] for k in ("class_size", "anchor_kind", "orbital_reading", "label_filter")},
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
            config = _config(split, seed)
            store.write_view(view, members, counts, index, split_file, config)
            report["splits"][split] = counts
            if split.endswith("_train"):
                train = [samples[i] for i in members if i in samples]
                stats_path = pair_stats_path(source, split[: -len("_train")], processed_dir)
                write_pair_stats(stats_path, pair_stats(train), view, index)
    return report


def pair_stats_path(source: str, name: str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    """`pair_feature_stats_<name>.json`: the standardization statistics of one split
    strategy, fitted on its training part."""
    return source_dir(source, processed_dir) / f"pair_feature_stats_{name}.json"


def pair_stats(samples: list[MechanismSample], params: KSParams | None = None) -> PairStats:
    """Statistics of the pair features of `samples` at the physical K and eps, over what a
    loaded sample holds: the defined entries of the valid cells (`pair_valid`; the store keeps
    no others), with the values rounded to float16 as stored. The statistics of samples just
    built therefore equal those refitted from the store."""
    params = params or KSParams()
    names = samples[0].pair_features if samples else STAGE_A_FEATURES
    if any(s.pair_features != names for s in samples):
        raise ValueError("pair statistics over samples with different pair features")
    blocks = (
        (
            to_params(s.pair_z.astype(np.float16), params, names),
            s.pair_defined & s.pair_valid[..., None],
        )
        for s in samples
    )
    return fit_pair_stats(blocks, names)


def write_pair_stats(path: Path, stats: PairStats, view: Path, index: Path) -> Path:
    """The statistics with a provenance header: the training view and the store index they were
    fitted on are its inputs, the physical K and eps and the features its configuration."""
    params = KSParams()
    config = {
        "sample_version": SAMPLE_VERSION,
        "pair_features": list(stats.names),
        "cells": "valid",
        "params": {"K": params.K, "eps": params.eps},
    }
    body = {"features": stats.to_dict()}
    header = prov.stamp(
        "pair_stats",
        SAMPLE_VERSION,
        {**prov.file_input(view), **prov.file_input(index)},
        config,
        prov.sha256_text(json.dumps(body, sort_keys=True)),
    )
    return prov.write_json(path, body, header)


def load_pair_stats(path: Path, allow_stale: bool = False) -> PairStats:
    """The statistics of `pair_feature_stats_<strategy>.json`, verified like the samples (its
    training view and the store index unchanged, built from a clean tree); StaleArtifact
    otherwise, unless `allow_stale` (diagnostics)."""
    prov.verify_clean(path, "pair_stats", SAMPLE_VERSION, allow_stale=allow_stale)
    d = json.loads(Path(path).read_text())["features"]
    return PairStats.from_dict(d, tuple(d))


def load_samples(
    path: Path, allow_stale: bool = False, ids: Iterable[str] | None = None
) -> list[MechanismSample]:
    """The samples of one split view (`samples_<split>.json`), only `ids` when given. The view,
    the store index and every shard read are verified first (parquet, split file, token files,
    contact blocks); StaleArtifact names what changed. `allow_stale` is for diagnostic CLIs
    only: the mismatch is logged and the store is read anyway."""
    return store.load_view(Path(path), MechanismSample, SAMPLE_VERSION, allow_stale, ids)


def sample_ids(path: Path, allow_stale: bool = False) -> list[str]:
    """The step ids of one split view that have a stored sample (the view is verified)."""
    return store.view_ids(Path(path), SAMPLE_VERSION, allow_stale)


def iter_samples(
    path: Path, allow_stale: bool = False, ids: Iterable[str] | None = None
) -> Iterator[MechanismSample]:
    """As `load_samples`, one shard at a time (shard order), for callers that reduce each
    sample and drop it."""
    return store.iter_view(Path(path), MechanismSample, SAMPLE_VERSION, allow_stale, ids)
