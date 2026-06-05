# Installation and Performance

SCCircuits is intended to be usable from a normal Python environment. Pixi is
kept for repository development, not as a requirement for researchers using the
package.

## Install from PyPI

```bash
python -m pip install sccircuits
```

The base install includes the numerical and plotting dependencies used by the
public API: NumPy, SciPy, Matplotlib, SymPy, and PyYAML.

For notebook widgets, development tools, or documentation builds:

```bash
python -m pip install "sccircuits[interactive]"
python -m pip install "sccircuits[dev]"
python -m pip install "sccircuits[docs]"
```

## Development Setup

Pixi remains the recommended contributor environment because it keeps the
NumPy/SciPy stack reproducible while running tests, linting, type checking, and
building the documentation.

```bash
git clone https://github.com/joanjcaceres/sccircuits.git
cd sccircuits
pixi run -e sccircuits install-dev
pixi run -e sccircuits test
```

Editable development without Pixi is also supported:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,interactive,docs]"
```

## Performance Model

The first public package is pure Python and relies on NumPy and SciPy wheels for
cross-platform BLAS/LAPACK-backed diagonalization. This keeps installation
simple on Windows, macOS, and Linux while preserving the SciPy dense Hermitian
diagonalization path used by `BBQ` and `Circuit`.

Run the small benchmark script to record a local baseline:

```bash
python benchmarks/diagonalization_smoke.py
```

For machines where the BLAS/LAPACK backend matters, Pixi or conda can still be
useful for controlled development environments. They are optional for package
users.

## Release Checklist

Before publishing a release:

- CI passes the Pixi tests, lint, typecheck, and docs build.
- Pip smoke CI passes on Windows, macOS, and Linux for Python 3.11, 3.12, and
  3.13.
- The package builds with `python -m build` and passes `python -m twine check`.
- PyPI Trusted Publishing is configured for the GitHub `pypi` environment.
- The published package can be installed with `python -m pip install sccircuits`.
