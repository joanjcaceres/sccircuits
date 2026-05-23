# BBQ

`sccircuits.BBQ` implements black-box quantization from a capacitance matrix and
an inverse inductance matrix. It is intended for workflows where the linearized
circuit is already available, either from a circuit-drawing tool or from another
classical circuit solver.

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

bbq = BBQ(
    capacitance_matrix,
    inverse_inductance_matrix,
    nonlinear_branches=(0, 1),
)

print(bbq.angular_frequencies)   # angular frequencies in rad/s
print(bbq.frequencies_ghz)       # ordinary frequencies in GHz
print(bbq.branch_phase_zpfs)     # branch-by-mode phase ZPF matrix
```

The nonlinear branch convention is `nonlinear_branches=(node_a, node_b)`, with
phase $\Phi_b - \Phi_a$. Reversing the tuple flips the sign of
`branch_phase_zpfs`.

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

Hamiltonian energies are in GHz. `hamiltonian_linear()` includes the harmonic
zero-point contribution $f_k(n_k + 1/2)$. When analyzing transition
frequencies, subtract the ground-state energy from each spectrum.

## Validation Rules

`BBQ` validates that:

- `capacitance_matrix` and `inverse_inductance_matrix` are finite square matrices.
- Both matrices are symmetric and have the same shape.
- `capacitance_matrix` is positive on the retained physical capacitance subspace.
- `nonlinear_branches` is either one valid branch tuple or a non-empty iterable
  of valid branch tuples.
- Only positive finite normal modes are retained.

Null or numerically tiny capacitance directions are projected out before the
generalized eigenproblem is solved. Here "tiny" means a capacitance eigenvalue
no larger than `1e-12 * max(abs(eigenvalues(capacitance_matrix)))`.

## Main Public Quantities

`BBQ` exposes the physical quantities used in the generalized-eigenproblem
workflow:

- `capacitance_matrix`
- `inverse_inductance_matrix`
- `nonlinear_branches`
- `branch_incidence_matrix`
- `angular_frequencies_squared`
- `angular_frequencies`
- `frequencies_ghz`
- `normal_mode_vectors`
- `branch_phase_zpfs`
- `selected_mode_indices`
- `truncation_dimensions`
- `branch_phase_operators`
- `josephson_suppression_factors`
