# SCCircuits

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17751124.svg)](https://doi.org/10.5281/zenodo.17751124)

SCCircuits is a Python package for numerical superconducting-circuit analysis.
It focuses on matrix-based black-box quantization, dense Hamiltonian
construction, spectroscopy workflows, and small research utilities for
superconducting quantum circuits.

## Documentation

The user-facing documentation is published at:

[https://joanjcaceres.github.io/sccircuits/](https://joanjcaceres.github.io/sccircuits/)

Recommended starting points:

- [BBQ Overview](https://joanjcaceres.github.io/sccircuits/bbq/) explains the practical workflow from cQEDraw or circuit matrices to `BBQ` results.
- [BBQ Quickstart](https://joanjcaceres.github.io/sccircuits/bbq/quickstart/) gives a runnable matrix example with frequencies and phase ZPFs.
- [BBQ Examples](https://joanjcaceres.github.io/sccircuits/bbq/examples/) collects worked examples and the planned cQEDraw project-download example.
- [BBQ API](https://joanjcaceres.github.io/sccircuits/api/bbq/) shows how to call `sccircuits.BBQ`, read frequencies and phase ZPFs, and build Hamiltonians.
- [Circuit Matrix Quantization](https://joanjcaceres.github.io/sccircuits/theory/circuit-matrix-quantization/) documents the matrix reductions, mode normalization, units, and branch phase zero-point fluctuations.

The site is intentionally incremental. `BBQ` is the first fully documented
research workflow because it is currently the most mature path between cQEDraw
and SCCircuits. The rest of the package should be documented through the same
site as those workflows stabilize.

The Markdown source for the website lives in `docs/`. Build it locally with:

```bash
pixi run -e sccircuits docs-build
```

## Companion GUI

[`cQEDraw`](https://cqedraw.org/) is the companion web app for drawing and
analyzing superconducting circuit graphs. A typical workflow is:

1. Draw the circuit in cQEDraw.
2. Export or copy the generated capacitance matrix, inverse-inductance matrix,
   and Josephson junction records.
3. Pass those matrices and records to `sccircuits.BBQ`.
4. Inspect mode frequencies, branch phase ZPFs, and Hamiltonian spectra in
   Python.

## Installation

### Recommended: Pixi

Pixi is the default environment manager for this repository. It keeps NumPy and
SciPy binary dependencies reproducible across local development machines.

```bash
git clone https://github.com/joanjcaceres/sccircuits.git
cd sccircuits
pixi run -e sccircuits install-dev
pixi run -e sccircuits test
```

### Alternative: pip and venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,interactive]"
```

For Apple Silicon machines, Pixi is recommended because the repository pins a
known-good BLAS/LAPACK setup.

## Quick Example

```python
import numpy as np
from sccircuits import BBQ

capacitance_matrix = np.array(
    [
        [40e-15, -32.9e-15],
        [-32.9e-15, 40e-15],
    ]
)

inverse_inductance_matrix = np.array(
    [
        [1 / 1.23e-9, 0.0],
        [0.0, 1 / 1.23e-9],
    ]
)

bbq = BBQ(
    capacitance_matrix,
    inverse_inductance_matrix,
    nonlinear_branches=(0, 1),
)

print("Mode frequencies in GHz:", bbq.frequencies_ghz)
print("Branch phase ZPFs:", bbq.branch_phase_zpfs)
```

For cQEDraw exports with Josephson junction records, pass the records directly:

```python
bbq = BBQ(
    capacitance_matrix,
    inverse_inductance_matrix,
    junctions=junctions,
)
```

`BBQ` keeps the junction row order when reporting `branch_phase_zpfs` and
`josephson_energies_ghz`, so drawing tools can map results back to the original
branches.

## Core Modules

- `sccircuits.BBQ`: black-box quantization from capacitance and inverse-inductance matrices.
- `sccircuits.Circuit`: dense superconducting-circuit Hamiltonian construction and diagonalization.
- `sccircuits.TransitionFitter`: transition-frequency fitting utilities.
- `sccircuits.PointPicker`: interactive point selection for data analysis.

## Development

Run the main checks with:

```bash
pixi run -e sccircuits test
pixi run -e sccircuits lint
pixi run -e sccircuits docs-build
```

If dependency resolution becomes inconsistent, recreate the Pixi environment:

```bash
pixi clean
pixi run -e sccircuits install-dev
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Contact

Joan Caceres - contact@joanjcaceres.com

Project link: [https://github.com/joanjcaceres/sccircuits](https://github.com/joanjcaceres/sccircuits)
