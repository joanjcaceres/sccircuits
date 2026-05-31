# SCCircuits

SCCircuits is a Python package for numerical superconducting-circuit analysis.
It focuses on matrix-based black-box quantization, dense Hamiltonian
construction, spectroscopy workflows, and small research utilities for
superconducting quantum circuits.

## Start Here

- [BBQ Researcher Guide](getting-started/bbq-researcher-guide.md) is the best
  first page for students and researchers using the cQEDraw-to-BBQ workflow. It
  explains the path from circuit matrices to frequencies, phase ZPFs, and
  Hamiltonians.
- [BBQ API](api/bbq.md) shows the practical Python calls for `sccircuits.BBQ`.
- [Circuit Matrix Quantization](theory/circuit-matrix-quantization.md) is the
  detailed reference for the matrix reductions and normal-mode calculation.

## Documentation Coverage

This website is the public documentation surface for SCCircuits. It is designed
to grow module by module rather than hide important details in repository
folders.

The first complete documentation path is `BBQ`, because it is currently the
most mature workflow and the one used directly with cQEDraw. SCCircuits also
contains Hamiltonian construction, spectroscopy-fitting utilities, point
selection tools, and numerical helpers. Those pieces should be documented here
as their supported workflows become clearer.

## Typical Workflow

1. Draw or assemble a superconducting circuit.
2. Export or construct the capacitance matrix, inverse-inductance matrix, and
   nonlinear branch records.
3. Pass those numerical objects to `sccircuits.BBQ`.
4. Inspect mode frequencies, branch phase zero-point fluctuations, and
   Hamiltonian spectra.

[`cQEDraw`](https://cqedraw.org/) is the companion web editor for drawing and
analyzing superconducting circuit graphs before passing matrices to `BBQ`.

## Trust and Scope

`BBQ` starts from matrices. It does not infer a circuit graph, choose loop
fluxes, or derive a symbolic Lagrangian. Those decisions belong to the drawing
or circuit-assembly layer. The theory reference documents how `BBQ` validates
the matrices, removes frozen or free coordinates, solves the physical oscillator
subspace, and reconstructs branch phase zero-point fluctuations in the original
node basis.

## Local Development

Build the documentation site:

```bash
pixi run -e sccircuits docs-build
```

Run the numerical test suite:

```bash
pixi run -e sccircuits test
```
