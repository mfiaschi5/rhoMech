# ksmech

## Project goal

ksmech is a deep-learning tool that predicts organic reaction mechanisms as sequences of
elementary steps. Each step is a set of arrows, and each arrow runs from a donor localized
orbital to an acceptor localized orbital. A transformer scores candidate arrows, and its
attention is biased by the Klopman-Salem equation, so the model starts from a physical
picture of orbital interactions instead of learning it from scratch.

The prediction target is an autoregressive sequence of elementary steps. A step is a list of
(donor orbital token, acceptor orbital token) arrows. After each step the arrows are applied
as an edit to the bond-electron matrix, giving an intermediate. The intermediate is
re-featurized (3D embedding, xTB, localization, new tokens) and fed back in to predict the
next step, until the model emits a stop token.

## Representation

Two token levels share one sequence: atom tokens and localized orbital tokens.

- Occupied orbitals: lone pair, sigma bond, pi bond.
- Virtual orbitals: sigma*, pi*, empty p.

Each orbital token carries its type, energy, 3D centroid, parent atoms, and AO composition.
Functional-group labels are only an auxiliary embedding added to tokens. They are never the
primary input and never decide which arrows are legal.

## Physics

The three Klopman-Salem terms are computed as pair features between orbital tokens:

1. filled-filled (closed-shell) repulsion,
2. Coulomb interaction between the charges on the two sites,
3. filled-empty orbital interaction (overlap squared over the energy gap).

These features enter the transformer as per-head additive attention biases. One head is
hard-wired to the pure KS kernel (no learned parameters) so the ablation "physics only vs.
learned" is a flag away.

## Data sources

| Dataset | Role |
|---|---|
| PMechDB | Curated polar mechanistic steps. Development set. |
| ReactMech | 67 reaction classes. Out-of-distribution splits. |
| FlowER dataset | About 1.4M steps. Scale-up. |
| mech-USPTO-31K | Held-out benchmark. Never trained on. |
| QMugs | QM pretraining. |

## Engineering rules

- Never fabricate a dataset column. Open and inspect the real file first, then document the
  schema in `docs/data_schema.md`.
- Every featurization function has a unit test on the fixed molecule set in
  `tests/fixtures/molecules.py`: ethane, ethanol, acetone, acetaldehyde, methyl acetate,
  chloromethane, methylamine, acetonitrile, benzene, acrolein.
- No silent failures. Every skipped molecule is logged with a reason code and counted, and
  the counts are reported at the end of the run.
- All QM results are cached on disk, keyed by canonical SMILES plus conformer id.
- Run `pytest` before saying any task is done, and paste the summary line.
- Record every non-obvious design choice in `docs/decisions.md`, with a date.
- Prefer plain numpy over adding dependencies.
- Keep functions under 60 lines.

## Commands

```bash
uv sync                       # install everything into .venv
uv run pytest                 # run tests (use -x to stop at first failure)
uv run ruff check .           # lint
uv run ruff format .          # format
uv run pre-commit install     # enable the ruff + pytest -x commit hook
```

Main CLI entry points are added phase by phase and listed here as they land:

- Data: _not yet_
- QM featurization: _not yet_
- Orbital tokens: _not yet_
- Arrow alignment: _not yet_
- Physics baseline: _not yet_
- Training: _not yet_
- Evaluation: _not yet_
