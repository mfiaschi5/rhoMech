# ksmech

Predict organic reaction mechanisms as sequences of elementary steps, where each step is a set of
arrows from a donor localized orbital to an acceptor localized orbital, scored by a transformer
whose attention is biased by the Klopman-Salem equation.

See [CLAUDE.md](CLAUDE.md) for the design and engineering rules, and
[docs/decisions.md](docs/decisions.md) for the running decision log.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is managed by uv.

```bash
uv sync
uv run pre-commit install
uv run pytest
uv run ruff check .
```

### tblite / xtb

`tblite` (Python bindings for GFN2-xTB) installs from PyPI on most Linux/macOS setups. Check with:

```bash
uv run python -c "import tblite"
```

If that fails (no wheel for your platform), install the bindings and the `xtb` binary with conda
instead, and make them visible to your environment:

```bash
conda install -c conda-forge tblite-python xtb
```

`ksmech.qm` is planned to keep a shell fallback that calls the `xtb` binary when the Python
bindings are unavailable (not yet implemented; arrives with the QM phase).

## Phases

1. **Data**: dataset download, parsing, canonical schema, splits
2. **QM featurization**: 3D embedding, xTB/tblite calculations, caching
3. **Orbital tokens**: localization, atom and orbital token construction
4. **Arrow alignment**: arrow parsing, arrow-to-orbital alignment, step application
5. **Physics baseline**: Klopman-Salem terms and pair features, scored without learning
6. **Model**: transformer with KS attention bias, decoder, losses
7. **Evaluation**: metrics, benchmarks, reports
8. **Scale-up**: FlowER-size training, QMugs pretraining
