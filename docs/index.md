# SCCircuits Documentation

SCCircuits helps researchers turn superconducting-circuit matrices into the
quantities they usually need next: mode frequencies, branch phase zero-point
fluctuations, and Hamiltonians.

The site is currently focused on `sccircuits.BBQ`, the most mature workflow and
the one that connects directly to cQEDraw exports.

## Why Start with BBQ?

- You can draw or assemble a circuit elsewhere and pass explicit matrices to
  Python.
- You can inspect frequencies and phase ZPFs without manually reducing large
  matrix problems.
- You can keep the connection to original cQEDraw junction records while `BBQ`
  handles frozen, free, or zero-mode directions internally.

## BBQ Documentation Path

For first use, read these pages:

1. [Installation and Performance](install.md): how to install SCCircuits with
   pip and when Pixi is useful for development.
2. [BBQ Overview](bbq/index.md): what the class does, where it fits, and what
   assumptions it makes.
3. [BBQ Quickstart](bbq/quickstart.md): a minimal working matrix example.

Then continue as needed:

- [BBQ Examples](bbq/examples.md): worked examples and the planned cQEDraw
  project-download example.
- [BBQ API Reference](api/bbq.md): constructor arguments, public attributes,
  branch conventions, and Hamiltonian methods.
- [Mathematical Reference](theory/circuit-matrix-quantization.md): the
  underlying matrix reductions and formulas.

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

## Local Development

Install the released package:

```bash
python -m pip install sccircuits
```

For repository development, use Pixi.

Build the documentation site:

```bash
pixi run -e sccircuits docs-build
```

Serve it locally:

```bash
pixi run -e sccircuits mkdocs serve
```
