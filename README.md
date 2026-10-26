# ksmech

Predict organic reaction mechanisms as sequences of elementary steps, where each step is the
change it makes to the Lewis structure (a reaction matrix), predicted by a transformer over atom
and localized-orbital tokens whose pair track carries geometry-free Klopman-Salem quantities (the
spectral gap factor and the cross-centre coupling).

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

torch comes from one of two dependency groups. `rocm` (the default) is the ROCm 7.2 build
for AMD cards. Run with `HSA_OVERRIDE_GFX_VERSION=10.3.0` on an RX 6600 / 6600 XT (gfx1032),
which runs the gfx1030 kernels. On an NVIDIA host, install the CUDA build instead:
`uv sync --no-group rocm --group cuda`, then `export UV_NO_SYNC=1`. Without it every `uv run`
syncs back to the default groups and reinstalls the ROCm build.

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
5. **Physics baseline**: geometry-free pair features, the untrained gap-factor ranking and a
   LightGBM reference
6. **Model**: transformer with a pair track, edit decoder, losses
7. **Evaluation**: metrics, benchmarks, reports
8. **Scale-up**: FlowER-size training, QMugs pretraining
