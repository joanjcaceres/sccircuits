# SCCircuits Documentation

SCCircuits is a Python package for numerical superconducting-circuit analysis.
The public documentation site is currently focused on one complete workflow:
using `sccircuits.BBQ` for black-box quantization from circuit matrices.

SCCircuits contains more than `BBQ`, including Hamiltonian construction,
spectroscopy-fitting utilities, point selection tools, and numerical helpers.
Those areas should be documented here as their recommended workflows become
stable. For now, the goal is to make the `BBQ` workflow clear enough that a
researcher can use it, audit it, and connect it to cQEDraw with confidence.

## BBQ Documentation Path

Start with these pages in order:

1. [BBQ Overview](bbq/index.md): what the class does, where it fits, and what
   assumptions it makes.
2. [BBQ Quickstart](bbq/quickstart.md): a minimal working matrix example.
3. [BBQ Examples](bbq/examples.md): worked examples and the planned cQEDraw
   project-download example.
4. [BBQ API Reference](api/bbq.md): constructor arguments, public attributes,
   branch conventions, and Hamiltonian methods.
5. [Mathematical Reference](theory/circuit-matrix-quantization.md): the
   matrix-reduction workflow, singular-coordinate handling, units, and phase
   zero-point fluctuation formulas.

## cQEDraw Workflow

[`cQEDraw`](https://cqedraw.org/) is the companion web editor for drawing and
analyzing superconducting circuit graphs before passing matrices to `BBQ`.

The intended workflow is:

1. Draw or load a superconducting circuit in cQEDraw.
2. Export the capacitance matrix, inverse-inductance matrix, and Josephson
   junction records.
3. Pass those numerical objects to `sccircuits.BBQ`.
4. Inspect mode frequencies, branch phase zero-point fluctuations, Josephson
   energies, and Hamiltonian spectra.

## Trust Boundary

`BBQ` starts from matrices. It does not parse a circuit graph, choose loop
fluxes, choose a gauge, or derive a symbolic Lagrangian. Those decisions belong
to the drawing or circuit-assembly layer.

The documented responsibility of `BBQ` is narrower and easier to audit: it
validates the supplied matrices, removes frozen or free coordinates, solves the
physical oscillator subspace, reconstructs mode vectors in the original node
basis, computes branch phase ZPFs, and builds dense Hamiltonians when requested.

## Local Development

Build the documentation site:

```bash
pixi run -e sccircuits docs-build
```

Serve it locally:

```bash
pixi run -e sccircuits mkdocs serve
```
