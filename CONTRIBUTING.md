# Contributing to CellGraphFM

Thanks for your interest in improving CellGraphFM! This project aims to be a
clean, well-tested foundation for cellular-graph research in computational
pathology, so we care a lot about readability, tests, and reproducibility.

## Development setup

```bash
git clone https://github.com/Aaqibbbb/Biocell.git
cd Biocell
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -e ".[torch,viz,dev]"
```

If you only want to work on the dependency-light core (graph construction,
explanations, data structures), `pip install -e ".[dev]"` is enough — the
`torch` layer is optional and its tests are skipped when it is not installed.

## Before you open a PR

Run the same checks CI runs:

```bash
ruff check .
ruff format --check .
pytest
```

To auto-fix formatting and lint issues:

```bash
ruff format .
ruff check . --fix
```

## Guidelines

- **Keep the core light.** Anything that needs `torch`/`torch-geometric` must be
  behind a lazy import so that `import cellgraphfm` works with only `numpy` and
  `scipy` installed.
- **Test what you add.** New functionality needs unit tests. Tests that need
  `torch` should use `pytest.importorskip("torch")`.
- **Type hints + docstrings.** Public functions should have type hints and a
  short docstring describing shapes and units.
- **Explainability first.** Prefer APIs that produce pathologist-interpretable
  outputs (cell types, interactions) over opaque tensors where practical.

## Reporting issues

Please include the CellGraphFM version (`cellgraphfm --version`), your Python
version, and a minimal reproduction.
