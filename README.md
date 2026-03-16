# SCCircuits

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17751124.svg)](https://doi.org/10.5281/zenodo.17751124)

SCCircuits is a Python package for superconducting circuit modeling, black-box
quantization workflows, and spectroscopy fitting.

The project documentation is published at
[sccircuits.readthedocs.io](https://sccircuits.readthedocs.io/). The docs cover
installation, API reference, the BBQ circuit designer app, and the theoretical
background behind the star and chain representations used by the package.

## Install

Pixi is the recommended development workflow for this repository:

```bash
git clone https://github.com/joanjcaceres/sccircuits.git
cd sccircuits
pixi run -e sccircuits install-dev
pixi run -e sccircuits test
```

For a plain virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev,interactive,docs]"
```

## Quick Start

Use harmonic-mode data and let `Circuit` build the chain representation:

```python
frequencies = [5.0, 6.2, 7.8]
phase_zpf = [0.12, 0.08, 0.03]
dimensions = [18, 10, 6]

from sccircuits import Circuit

circuit = Circuit.from_harmonic_modes(
    frequencies=frequencies,
    phase_zpf=phase_zpf,
    dimensions=dimensions,
    Ej=0.95,
)

evals, _ = circuit.eigensystem(truncation=[12, 8, 6])
print(evals[:3])
```

The full documentation includes guides for:

- chain- and star-basis circuit modeling
- `TransitionFitter` and `FitAnalysis`
- `PointPicker` YAML export and fitting workflows
- the BBQ designer app in [`apps/bbq_circuit_designer.py`](apps/bbq_circuit_designer.py)
- theory notes adapted from the companion thesis chapter

## Documentation Locally

Build the docs locally with:

```bash
pixi run -e sccircuits docs-html
pixi run -e sccircuits docs-doctest
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, testing, and
coding conventions.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
