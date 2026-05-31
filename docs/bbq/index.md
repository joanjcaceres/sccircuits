# BBQ Overview

`sccircuits.BBQ` is the matrix-to-modes backend for SCCircuits. It starts from a
capacitance matrix and an inverse-inductance matrix, then computes the linear
normal modes and branch phase zero-point fluctuations needed for
superconducting-circuit Hamiltonians.

This is currently the first SCCircuits workflow documented end to end. The
documentation is intentionally focused on making `BBQ` usable and auditable
before expanding the same level of coverage to the rest of the package.

## What `BBQ` Computes

Given numerical circuit matrices, `BBQ` computes:

- normal-mode frequencies;
- branch phase zero-point fluctuations;
- Josephson energies when they are supplied by the matrix-export workflow;
- dense Hamiltonians for selected modes and nonlinear branches.

The most common public quantities are:

- `frequencies_ghz`
- `angular_frequencies`
- `normal_mode_vectors`
- `branch_phase_nodes`
- `branch_phase_zpfs`
- `josephson_energies_ghz`

## Where `BBQ` Fits

A typical workflow has three layers:

1. **Circuit drawing or assembly**: a tool such as
   [cQEDraw](https://cqedraw.org/) stores the graph, elements, node labels, and
   branch records.
2. **Matrix export**: the drawing or assembly layer builds the capacitance
   matrix, inverse-inductance matrix, and Josephson junction records.
3. **Matrix-to-modes calculation**: `sccircuits.BBQ` validates the matrices,
   removes singular directions that do not represent finite-frequency
   oscillators, solves the remaining modal problem, and reports the results.

This boundary is intentional. `BBQ` should be easy to audit because it starts
from explicit numerical matrices. Topology decisions such as loop-flux choices,
graph parsing, and independent external flux variables belong upstream of
`BBQ`.

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

## Internal Calculation

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

For the detailed derivation and tolerance rules, read
[Circuit Matrix Quantization](../theory/circuit-matrix-quantization.md).

For a branch between two nodes, the sign of `branch_phase_zpfs` follows the
branch direction. Reversing a branch reverses the sign. This is expected and is
usually less important than keeping a consistent convention when comparing
branches or adding external phase offsets.

## Building a Hamiltonian

After inspecting the linear modes, choose which modes to keep and set their
Hilbert-space dimensions:

```python
bbq.selected_mode_indices = [0, 1]
bbq.truncation_dimensions = [20, 12]

H_linear = bbq.hamiltonian_linear()
H_nonlinear = bbq.hamiltonian_nonlinear(
    josephson_energies=bbq.josephson_energies_ghz,
    external_phases=0.0,
)

H = H_linear + H_nonlinear
```

Hamiltonian matrices are returned in GHz. When comparing transition
frequencies, subtract the ground-state energy from the eigenvalue spectrum.

## What `BBQ` Does Not Decide

`BBQ` does not:

- parse a circuit drawing;
- decide which external fluxes are independent;
- choose a gauge;
- derive a symbolic Lagrangian from graph elements;
- decide which modes are physically relevant for a specific experiment.

Those choices must be made by the researcher or by the circuit-assembly layer
before calling `BBQ`.

## Next Pages

- [BBQ Quickstart](quickstart.md) for a runnable matrix example.
- [BBQ Examples](examples.md) for worked examples and the planned cQEDraw
  project-download example.
- [BBQ API Reference](../api/bbq.md) for constructor arguments, public attributes, and
  Hamiltonian methods.
- [Circuit Matrix Quantization](../theory/circuit-matrix-quantization.md) for
  the reduction workflow and mathematical reference.
- [cQEDraw](https://cqedraw.org/) for the companion circuit-drawing workflow.
