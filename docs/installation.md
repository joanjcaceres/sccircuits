# Installation

## Recommended: Pixi

Pixi is the default environment manager for this repository. It keeps the NumPy
and SciPy stack pinned to a tested configuration and is the easiest path for
running tests and building docs locally.

```bash
git clone https://github.com/joanjcaceres/sccircuits.git
cd sccircuits
pixi run -e sccircuits install-dev
pixi run -e sccircuits test
```

Useful development commands:

```bash
pixi run -e sccircuits docs-html
pixi run -e sccircuits docs-doctest
pixi run -e sccircuits example
```

### Apple Silicon note

On `osx-arm64`, the Pixi environment pins BLAS and LAPACK to the Accelerate
backend. That keeps the SciPy linear algebra stack aligned with the setup used
to develop this repository.

## Alternative: `pip` and `venv`

If you prefer a plain Python environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev,interactive,docs]"
pytest
```

The `docs` extra installs Sphinx, MyST Markdown support, and the Read the Docs
theme used by this site.

## What gets installed

Core runtime dependencies:

- NumPy
- SciPy
- Matplotlib
- SymPy

Optional groups:

- `interactive`: Jupyter and IPython support for widget-heavy workflows
- `docs`: Sphinx, MyST, and the RTD theme
- `dev`: test and formatting tools
