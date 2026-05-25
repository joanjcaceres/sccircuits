# SCCircuits

SCCircuits is a Python package for numerical superconducting-circuit analysis.
It includes tools for circuit Hamiltonians, spectroscopy fitting, point picking,
and black-box quantization from circuit matrices.

## Documentation Map

- [Circuit Matrix Quantization](theory/circuit-matrix-quantization.md) explains
  the generalized eigenproblem, mode normalization, and branch phase
  zero-point fluctuations used by `BBQ`.
- [Graph-To-Matrix Quantization](theory/graph-to-matrix-quantization.md)
  documents the planned upstream graph layer: incidence matrices, external flux
  mapping, variable reductions, and the output contract into `BBQ`.
- [BBQ](api/bbq.md) shows the practical workflow from capacitance and inverse
  inductance matrices to frequencies, phase ZPF values, and Hamiltonians.

## Local Development

Install the Pixi environment and run the docs build:

```bash
pixi run -e sccircuits docs-build
```

Run the numerical test suite:

```bash
pixi run -e sccircuits test
```
