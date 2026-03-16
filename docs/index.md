# SCCircuits Documentation

SCCircuits is a Python package for superconducting circuit analysis built around
three connected workflows:

- circuit models expressed in chain or star coordinates
- black-box quantization from capacitance and inductance matrices
- spectroscopy fitting using generic transition models

This documentation is organized around how those workflows fit together in
practice. The user guides explain how to build models, fit data, and export
matrices from the BBQ designer app. The theory section summarizes the modeling
ideas behind the package and is adapted from the companion thesis chapter
`3-sccircuits.tex`.

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
guides/index
theory/index
api/index
```

## Start Here

- [Installation](installation.md) for Pixi and `pip` setups.
- [Quickstart](quickstart.md) for a minimal circuit calculation and a small
  fitting example.
- [Guides](guides/index.md) for package workflows and the BBQ designer app.
- [Theory](theory/index.md) for the star/chain modeling background.
- [API Reference](api/index.md) for class and function documentation.

## Package Scope

SCCircuits currently centers on:

- `Circuit` for chain-based Hamiltonians and conversions from harmonic-mode data
- `BBQ` for turning circuit matrices into linear modes and phase zero-point
  fluctuations
- `TransitionFitter` for least-squares or differential-evolution fits of
  transition data
- `FitAnalysis` for post-fit diagnostics
- `PointPicker` for collecting spectroscopy points and exporting YAML

## Project Links

- GitHub: <https://github.com/joanjcaceres/sccircuits>
- Documentation source: `docs/`
- Issue tracker: <https://github.com/joanjcaceres/sccircuits/issues>
