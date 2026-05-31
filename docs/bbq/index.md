# BBQ Overview

`sccircuits.BBQ` is the matrix-to-modes backend for SCCircuits. It starts from a
capacitance matrix and an inverse-inductance matrix, then computes the linear
normal modes and branch phase zero-point fluctuations needed for
superconducting-circuit Hamiltonians.

Use `BBQ` when the circuit has already been converted into numerical matrices,
for example by cQEDraw or by your own lumped-element code.

## Why It Is Useful

Manual matrix reduction becomes fragile for larger circuits, especially when
some coordinates have no capacitance, no restoring potential, or are removed by
constraints. `BBQ` keeps that reduction inside one auditable class and returns
the quantities needed for the next modeling step.

Given numerical circuit matrices, `BBQ` computes:

- normal-mode frequencies;
- branch phase zero-point fluctuations;
- Josephson energies when they are supplied by the matrix-export workflow;
- dense Hamiltonians for selected modes and nonlinear branches.

The implementation follows the black-box quantization idea of Nigg et al. for
using linearized modes as the basis for weakly nonlinear circuits, and follows
the free/frozen-variable reduction language of Chitta et al. for handling
non-dynamical directions. See the [Mathematical Reference](../theory/circuit-matrix-quantization.md#references)
for full citations.

## What You Need to Provide

- `capacitance_matrix`: symmetric capacitance matrix in Farads.
- `inverse_inductance_matrix`: symmetric inverse-inductance matrix in inverse
  Henries.
- `nonlinear_branches` or cQEDraw `junctions`: branch definitions used to
  compute phase ZPFs.

## What You Get Back

- `frequencies_ghz`
- `angular_frequencies`
- `normal_mode_vectors`
- `branch_phase_nodes`
- `branch_phase_zpfs`
- `josephson_energies_ghz`

## Minimal Workflow

```python
from sccircuits import BBQ

capacitance_matrix, inverse_inductance_matrix = circuit_matrices(params)
junctions = josephson_branches(params)

bbq = BBQ(
    capacitance_matrix,
    inverse_inductance_matrix,
    junctions=junctions,
)

print(bbq.frequencies_ghz)
print(bbq.branch_phase_zpfs)
print(bbq.josephson_energies_ghz)
```

The matrix-export function names above are placeholders. In a cQEDraw workflow,
they correspond to the generated Python snippet that defines the matrices and
the Josephson branch records.

For a runnable example with explicit matrices, continue to the
[BBQ Quickstart](quickstart.md).

## Trust Boundary

`BBQ` starts from matrices. It does not parse a circuit drawing, choose a gauge,
decide which external fluxes are independent, or derive a symbolic Lagrangian.
Those choices belong to the drawing or circuit-assembly layer.

Its job is narrower: validate the supplied matrices, reduce non-oscillatory
directions, solve the positive-frequency oscillator subspace, reconstruct mode
vectors in the original node basis, and compute branch phase ZPFs.

## Reduction Summary

`BBQ` follows a numerical matrix-reduction workflow before it solves for
oscillator modes:

1. It checks that the capacitance matrix and inverse-inductance matrix are
   finite, square, symmetric, and have compatible sizes.
2. It identifies coordinates with no capacitance row or column. These frozen
   coordinates have no kinetic energy, so they are eliminated by minimizing the
   quadratic potential.
3. It projects out remaining null-capacitance directions.
4. It separates zero-potential or DC directions from positive oscillator
   directions.
5. It solves only the finite positive-frequency oscillator problem.
6. It reconstructs the mode vectors back to the original node basis.
7. It computes branch phase ZPFs using the original branch incidence rows.

The reconstruction step is important for cQEDraw and similar tools. Even when a
node coordinate is eliminated internally, branch quantities are still reported
using the original node and junction records supplied by the caller.

The detailed derivation and tolerance rules are in the
[Mathematical Reference](../theory/circuit-matrix-quantization.md).

## Next Pages

- [BBQ Quickstart](quickstart.md) for a runnable matrix example.
- [BBQ Examples](examples.md) for worked examples and the planned cQEDraw
  project-download example.
- [BBQ API Reference](../api/bbq.md) for constructor arguments, public attributes, and
  Hamiltonian methods.
- [Circuit Matrix Quantization](../theory/circuit-matrix-quantization.md) for
  the reduction workflow and mathematical reference.
- [cQEDraw](https://cqedraw.org/) for the companion circuit-drawing workflow.
