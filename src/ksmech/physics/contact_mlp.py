"""Orbital-descriptor contact kernel (`ContactConfig(kernel="mlp")`, spec section 5.4).

    S_contact(r, s) = exp(MLP([v_r, v_s]))

v_r is the token's continuous model-input vector (`OrbitalDescriptors.continuous_vector`: no
element, no type), so the kernel asks how strongly two orbitals of this shape overlap at van der
Waals contact, not which elements they sit on. It is fitted by least squares on
log |l_r,a^T S_ab l_s,b| over exactly the samples of the contact table
(`contact_table.contact_rows`), one network per kind (occupied-virtual, occupied-occupied), and
takes the place of the table value S[kappa] in the amplitude a(r, s) = S_contact w_m(r) w_m(s).

Training uses torch; the weights are stored as npz and evaluated in numpy. Molecules are split
90 / 10, and the held-out samples score both this network and a table built from the training
molecules (log-space R^2, median |log10 error|): the starting point of the Prompt 7 comparison.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ksmech.orbitals.tokens import MoleculeTokens, OrbitalToken
from ksmech.physics.contact_table import (
    KINDS,
    contact_key,
    contact_rows,
    tables_from_samples,
)

DEFAULT_MLP_PATH = Path("data/cache/contact_mlp.npz")
FLOOR = 1e-7  # overlaps below this are taken as this value in log space


def descriptor_matrix(orbitals: list[OrbitalToken]) -> np.ndarray:
    """(n, d) continuous model-input descriptors of `orbitals`."""
    rows = [o.descriptors.continuous_vector() for o in orbitals]
    return np.array(rows, dtype=np.float32).reshape(len(orbitals), -1)


def _silu(x: np.ndarray) -> np.ndarray:
    """x * sigmoid(x), written with tanh so that no exponential overflows."""
    return 0.5 * x * (1.0 + np.tanh(0.5 * x))


@dataclass
class ContactMLP:
    """One fitted network per kind: [(W, b), ...] with W of shape (in, out)."""

    layers: dict[str, list[tuple[np.ndarray, np.ndarray]]]
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)

    def predict(self, kind: str, X: np.ndarray) -> np.ndarray:
        """S_contact for rows [v_r, v_s] of `X` (positive)."""
        h = (np.asarray(X, dtype=np.float64) - self.mean[kind]) / self.std[kind]
        layers = self.layers[kind]
        for W, b in layers[:-1]:
            h = _silu(h @ W + b)
        W, b = layers[-1]
        return np.exp((h @ W + b)[:, 0])

    def pair_matrix(self, kind: str, V_r: np.ndarray, V_s: np.ndarray) -> np.ndarray:
        """(len(V_r), len(V_s)) S_contact of every row-column pair of descriptor vectors."""
        n_r, n_s = len(V_r), len(V_s)
        if not n_r or not n_s:
            return np.zeros((n_r, n_s))
        X = np.concatenate([np.repeat(V_r, n_s, axis=0), np.tile(V_s, (n_r, 1))], axis=1)
        return self.predict(kind, X).reshape(n_r, n_s)


@dataclass
class MLPSamples:
    """Contact samples of one kind: descriptor pairs, overlaps, table keys, molecule ids."""

    X: list[np.ndarray] = field(default_factory=list)
    y: list[np.ndarray] = field(default_factory=list)
    kappa: list[tuple] = field(default_factory=list)
    mol: list[np.ndarray] = field(default_factory=list)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.concatenate(self.X) if self.X else np.zeros((0, 0), dtype=np.float32)
        y = np.concatenate(self.y) if self.y else np.zeros(0)
        mol = np.concatenate(self.mol) if self.mol else np.zeros(0, dtype=int)
        return X, y, mol


def add_molecule(samples: dict[str, MLPSamples], t: MoleculeTokens, mol_id: int, window=0.3):
    """Append the contact samples of one molecule to `samples` (one entry per kind)."""
    V = descriptor_matrix(t.orbitals)
    z = np.array([a.atomic_number for a in t.atoms])
    keys = [contact_key(o, z) for o in t.orbitals]
    rows = defaultdict(list)
    for kind, r, s, value in contact_rows(t, window):
        rows[kind].append((r, s, value))
    for kind, entries in rows.items():
        r, s, v = (np.array(x) for x in zip(*entries, strict=True))
        smp = samples[kind]
        smp.X.append(np.concatenate([V[r], V[s]], axis=1))
        smp.y.append(v.astype(float))
        smp.kappa += [(*keys[i], *keys[j]) for i, j in zip(r, s, strict=True)]
        smp.mol.append(np.full(len(v), mol_id))


def samples_from_tokens(tokens: list[MoleculeTokens], window: float = 0.3):
    samples = {kind: MLPSamples() for kind in KINDS}
    for i, t in enumerate(tokens):
        add_molecule(samples, t, i, window)
    return samples


def samples_from_paths(paths: list[Path], method: str, window: float = 0.3):
    """Samples of every readable token file of `method`, and the skip counts by reason."""
    from ksmech.orbitals.tokens import TokenSchemaError, load_tokens

    samples = {kind: MLPSamples() for kind in KINDS}
    skipped: dict[str, int] = defaultdict(int)
    for i, path in enumerate(paths):
        if not Path(path).is_file():
            skipped["missing_tokens"] += 1
            continue
        try:
            t = load_tokens(path)
        except TokenSchemaError:
            skipped["old_schema"] += 1
            continue
        if t.method != method:
            skipped["other_method"] += 1
            continue
        add_molecule(samples, t, i, window)
    return samples, dict(skipped)


def _train_net(X: np.ndarray, y: np.ndarray, hidden, epochs: int, seed: int):
    """Least squares on log y; returns the numpy layers."""
    import torch

    torch.manual_seed(seed)
    dims = [X.shape[1], *hidden, 1]
    mods = []
    for i in range(len(dims) - 1):
        mods.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(torch.nn.SiLU())
    net = torch.nn.Sequential(*mods).double()
    with torch.no_grad():
        mods[-1].bias.fill_(float(y.mean()))
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(yt), generator=gen).split(4096):
            opt.zero_grad()
            loss = torch.mean((net(Xt[idx])[:, 0] - yt[idx]) ** 2)
            loss.backward()
            opt.step()
        sched.step()
    linear = [m for m in mods if isinstance(m, torch.nn.Linear)]
    return [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()) for m in linear]


def _scores(log_pred: np.ndarray, log_true: np.ndarray) -> dict[str, float]:
    resid = log_pred - log_true
    total = np.sum((log_true - log_true.mean()) ** 2)
    return {
        "r2_log": float(1.0 - np.sum(resid**2) / total) if total > 0 else float("nan"),
        "median_abs_log10_error": float(np.median(np.abs(resid)) / np.log(10.0)),
        "n": len(log_true),
    }


def _held_out_table(samples: MLPSamples, train: np.ndarray, test: np.ndarray, kind: str):
    """Log-space predictions of a table built from the training molecules only."""
    _, y, _ = samples.arrays()
    rows = [(kind, samples.kappa[i], float(y[i])) for i in np.flatnonzero(train)]
    table = tables_from_samples(rows, min_count=5)[kind]
    pred = np.array([table.lookup(samples.kappa[i])[0] for i in np.flatnonzero(test)])
    return np.log(np.maximum(pred, FLOOR))


def fit_kind(samples: MLPSamples, kind: str, hidden, epochs: int, seed: int, val_fraction: float):
    """(layers, mean, std, metrics) for one kind; the metrics compare MLP and table held out."""
    X, y, mol = samples.arrays()
    rng = np.random.default_rng(seed)
    molecules = np.unique(mol)
    held = set(rng.choice(molecules, max(1, int(round(val_fraction * len(molecules)))), False))
    test = np.isin(mol, list(held)) if len(molecules) > 1 else np.zeros(len(mol), bool)
    train = ~test
    logy = np.log(np.maximum(y, FLOOR))
    mean, std = X[train].mean(axis=0), X[train].std(axis=0)
    std = np.where(std > 0, std, 1.0)
    Z = ((X - mean) / std).astype(np.float64)
    layers = _train_net(Z[train], logy[train], hidden, epochs, seed)
    model = ContactMLP({kind: layers}, {kind: mean}, {kind: std})
    metrics = {"n_train": int(train.sum()), "n_molecules": len(molecules)}
    if test.any():
        metrics["mlp_held_out"] = _scores(np.log(model.predict(kind, X[test])), logy[test])
        metrics["table_held_out"] = _scores(_held_out_table(samples, train, test, kind), logy[test])
    return layers, mean, std, metrics


def fit_contact_mlp(
    samples: dict[str, MLPSamples],
    hidden: tuple[int, ...] = (64, 64),
    epochs: int = 20,
    seed: int = 0,
    val_fraction: float = 0.1,
    method: str = "",
) -> ContactMLP:
    """One network per kind, fitted on the contact samples of many molecules."""
    out = ContactMLP({}, {}, {}, {"method": method, "hidden": list(hidden), "epochs": epochs})
    for kind in KINDS:
        if not samples[kind].y:
            raise ValueError(f"no {kind} contact samples to fit the MLP kernel on")
        layers, mean, std, metrics = fit_kind(
            samples[kind], kind, hidden, epochs, seed, val_fraction
        )
        out.layers[kind], out.mean[kind], out.std[kind] = layers, mean, std
        out.meta[kind] = metrics
    return out


def save_contact_mlp(model: ContactMLP, path: Path = DEFAULT_MLP_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"meta": np.array(json.dumps(model.meta))}
    for kind, layers in model.layers.items():
        arrays[f"{kind}.mean"], arrays[f"{kind}.std"] = model.mean[kind], model.std[kind]
        for i, (W, b) in enumerate(layers):
            arrays[f"{kind}.W{i}"], arrays[f"{kind}.b{i}"] = W, b
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return path


def load_contact_mlp(path: Path = DEFAULT_MLP_PATH) -> ContactMLP:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no contact MLP; fit it with `ksmech-physics contact-mlp`")
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    model = ContactMLP({}, {}, {}, json.loads(str(data["meta"])))
    for kind in KINDS:
        n = sum(1 for k in data if k.startswith(f"{kind}.W"))
        model.layers[kind] = [(data[f"{kind}.W{i}"], data[f"{kind}.b{i}"]) for i in range(n)]
        model.mean[kind], model.std[kind] = data[f"{kind}.mean"], data[f"{kind}.std"]
    return model
