# BBQ API Reference

`sccircuits.BBQ` implements black-box quantization from a capacitance matrix and
an inverse inductance matrix. It is intended for workflows where the linearized
circuit is already available, either from a circuit-drawing tool or from another
classical circuit solver.

`BBQ` is the reduced matrix-to-modes backend. It assumes that graph parsing,
element assembly, loop-flux choices, variable classification, and physical
reductions have already happened before the matrices are passed in. The
companion cQEDraw workflow currently owns drawing and matrix export; future
graph-layer functionality in SCCircuits should prepare the same matrix and
branch-offset data before calling `BBQ`.

## Constructor

```python
BBQ(
    capacitance_matrix,
    inverse_inductance_matrix,
    nonlinear_branches=None,
    *,
    junctions=None,
)
```

Arguments:

- `capacitance_matrix`: symmetric capacitance matrix in Farads.
- `inverse_inductance_matrix`: symmetric inverse-inductance matrix in inverse
  Henries.
- `nonlinear_branches`: one branch tuple or a list of branch tuples. Use this
  when the caller knows the nonlinear branch direction but does not have
  cQEDraw junction records.
- `junctions`: cQEDraw-style Josephson junction records. Use this when the
  matrix export includes branch direction, matrix node indices, and Josephson
  parameters.

Pass either `nonlinear_branches` or `junctions`, not both.

## Basic Workflow

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

bbq = BBQ(capacitance_matrix, inverse_inductance_matrix)

print(bbq.angular_frequencies)   # angular frequencies in rad/s
print(bbq.frequencies_ghz)       # ordinary frequencies in GHz
print(bbq.branch_phase_zpfs)     # empty when no nonlinear branches exist
```

If the circuit has a nonlinear branch and you want its phase zero-point
fluctuations, pass the branch direction explicitly:

```python
bbq = BBQ(
    capacitance_matrix,
    inverse_inductance_matrix,
    nonlinear_branches=(0, 1),
)

print(bbq.branch_phase_nodes)    # (positive_node, negative_node), None means ground
print(bbq.branch_phase_zpfs)     # branch-by-mode phase ZPF matrix
```

The nonlinear branch convention is `nonlinear_branches=(node_a, node_b)`, with
phase $\Phi_b - \Phi_a$. Reversing the tuple flips the sign of
`branch_phase_zpfs`. The normalized phase direction is also available as
`branch_phase_nodes`, where each row is `(positive_node, negative_node)` and
the phase is $\Phi_\mathrm{positive} - \Phi_\mathrm{negative}$.

For multiple nonlinear branches, pass a list of branch tuples:

```python
bbq = BBQ(
    capacitance_matrix,
    inverse_inductance_matrix,
    nonlinear_branches=[(0, 1), (1, 0)],
)

print(bbq.branch_phase_zpfs.shape)  # (number_of_branches, number_of_modes)
```

For one nonlinear branch, `branch_phase_zpfs` still has two axes; its shape is
`(1, number_of_modes)`.

## cQEDraw Josephson Junction Records

cQEDraw snippets can export Josephson junction records with the matrix indices,
phase direction, Josephson inductance, and Josephson energy for each junction.
Pass those records directly with `junctions=`:

```python
capacitance_matrix, inverse_inductance_matrix = circuit_matrices(params)
junctions = josephson_branches(params)

bbq = BBQ(
    capacitance_matrix,
    inverse_inductance_matrix,
    junctions=junctions,
)

print(bbq.frequencies_ghz)
print(bbq.branch_phase_nodes)      # one row per input junction record
print(bbq.branch_phase_zpfs)       # shape: (junctions, modes)
print(bbq.josephson_energies_ghz)  # one value per junction, if exported
```

`BBQ` treats these records as numerical branch definitions. Web-specific
identifiers such as `edge_id` and `project_nodes` are not retained; callers that
need to map results back to a drawing should keep the original `junctions` list
and use row order. The rows of `branch_phase_nodes`, `branch_incidence_matrix`,
`branch_phase_zpfs`, and `josephson_energies_ghz` correspond to the rows of
`junctions`.

The only required fields are `phase_positive_index` and
`phase_negative_index`; `None` means the grounded side. `matrix_nodes` is
validated when present. If every record includes `E_j_GHz`, or every record
includes `L_j` from which `E_j_GHz` can be computed,
`bbq.josephson_energies_ghz` is populated. Hamiltonian construction remains
explicit: pass Josephson energies to `hamiltonian_nonlinear(...)`.

## Hamiltonians

After inspecting the modes, choose the modes to retain and the Hilbert-space
dimension for each one:

```python
bbq.selected_mode_indices = [0]
bbq.truncation_dimensions = 40

H_linear = bbq.hamiltonian_linear()
Hnl = bbq.hamiltonian_nonlinear(
    josephson_energies=7.5,
    external_phases=0.0,
)
H = H_linear + Hnl
```

With multiple nonlinear branches, pass one Josephson energy and one external
phase per branch:

```python
Hnl = bbq.hamiltonian_nonlinear(
    josephson_energies=[7.5, 3.2],
    external_phases=[0.0, 0.1],
)
```

`external_phases` are gauge-fixed phase offsets in radians, ordered by
nonlinear branch row. They are not necessarily independent physical loop
fluxes; independence is determined by the circuit graph topology before the
problem reaches `BBQ`.

Hamiltonian energies are in GHz. `hamiltonian_linear()` includes the harmonic
zero-point contribution $f_k(n_k + 1/2)$. When analyzing transition
frequencies, subtract the ground-state energy from each spectrum.
`hamiltonian_nonlinear()` uses the convention
$-E_J(s\cos(\varphi+\varphi_\mathrm{ext})+\varphi^2/2)$ per branch. The
quadratic term avoids double-counting the linearized Josephson inductance
already included in `inverse_inductance_matrix`.

## Validation Rules

`BBQ` validates that:

- `capacitance_matrix` and `inverse_inductance_matrix` are finite square matrices.
- Both matrices are symmetric and have the same shape.
- `capacitance_matrix` is positive on the retained physical capacitance subspace.
- Frozen coordinates have a positive definite inverse-inductance block, giving
  a unique Schur reduction.
- `nonlinear_branches`, when provided, is either one valid branch tuple or an
  iterable of valid branch tuples.
- Only positive finite normal modes are retained.

`BBQ` handles several singular-matrix cases directly. Coordinates with zero
capacitance rows and columns are eliminated by minimizing the quadratic
potential, null capacitance eigen-directions are projected out, and
zero-potential modes are removed before solving the positive oscillator
subspace. Branch phase ZPFs are computed after reconstructing the modes back to
the original node basis.

## Public Attributes

`BBQ` exposes the physical quantities used in the matrix-to-modes workflow.

| Attribute | Meaning |
| --- | --- |
| `capacitance_matrix` | Input capacitance matrix in Farads. |
| `inverse_inductance_matrix` | Input inverse-inductance matrix in inverse Henries. |
| `nonlinear_branches` | Normalized nonlinear branch tuples. |
| `branch_phase_nodes` | One `(positive_node, negative_node)` pair per nonlinear branch. |
| `branch_incidence_matrix` | Branch-by-node incidence matrix used for branch phase differences. |
| `angular_frequencies_squared` | Positive oscillator eigenvalues in `(rad/s)^2`. |
| `angular_frequencies` | Positive angular frequencies in rad/s. |
| `frequencies_ghz` | Ordinary frequencies in GHz. |
| `normal_mode_vectors` | Capacitance-normalized normal-mode vectors in the original node basis. |
| `branch_phase_zpfs` | Branch-by-mode dimensionless phase zero-point fluctuation matrix. |
| `josephson_energies_ghz` | Josephson energies in GHz when every junction record provides enough data. |
| `selected_mode_indices` | Mode indices retained for Hamiltonian construction. |
| `truncation_dimensions` | Hilbert-space dimension for each selected mode. |
| `branch_phase_operators` | Branch phase operators for the selected truncated Hilbert space. |
| `josephson_suppression_factors` | Suppression factors from modes omitted from the truncated Hilbert space. |

## Public Methods

| Method | Purpose |
| --- | --- |
| `hamiltonian_linear()` | Build the selected-mode harmonic Hamiltonian in GHz. |
| `hamiltonian_nonlinear(josephson_energies, external_phases)` | Build the Josephson nonlinear Hamiltonian in GHz. |
| `plot_modes(which=0)` | Plot selected capacitance-normalized linear mode shapes. |

Hamiltonian methods require `selected_mode_indices` and
`truncation_dimensions` to be set first.
