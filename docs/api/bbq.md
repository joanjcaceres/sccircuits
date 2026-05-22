# BBQ

`sccircuits.BBQ` implements black-box quantization from a capacitance matrix and
an inverse inductance matrix. It is intended for workflows where the linearized
circuit is already available, either from a circuit-drawing tool or from another
classical circuit solver.

## Basic Workflow

```python
import numpy as np
from sccircuits import BBQ

C_matrix = np.array(
    [
        [40e-15, -32.9e-15],
        [-32.9e-15, 40e-15],
    ]
)
L_inv_matrix = np.array(
    [
        [1 / 1.23e-9, 0.0],
        [0.0, 1 / 1.23e-9],
    ]
)

bbq = BBQ(C_matrix, L_inv_matrix, non_linear_nodes=(0, 1))

print(bbq.linear_modes)      # angular frequencies in rad/s
print(bbq.linear_modes_GHz)  # ordinary frequencies in GHz
print(bbq.phase_zpf_list)    # dimensionless branch phase ZPF values
```

The nonlinear branch convention is `non_linear_nodes=(node_a, node_b)`, with
phase \( \Phi_b - \Phi_a \). Reversing the tuple flips the sign of
`phase_zpf_list`.

## Hamiltonians

After inspecting the modes, choose the modes to retain and the Hilbert-space
dimension for each one:

```python
bbq.selected_modes = [0]
bbq.dimensions = 40

H0 = bbq.hamiltonian_0()
Hnl = bbq.hamiltonian_nl(Ej=7.5, phi_ext=0.0)
H = H0 + Hnl
```

Hamiltonian energies are in GHz. `hamiltonian_0()` includes the harmonic
zero-point contribution \( f_k(n_k + 1/2) \). When analyzing transition
frequencies, subtract the ground-state energy from each spectrum.

## Validation Rules

`BBQ` validates that:

- `C_matrix` and `L_inv_matrix` are finite square matrices.
- Both matrices are symmetric and have the same shape.
- `C_matrix` is positive on the retained physical capacitance subspace.
- `non_linear_nodes` contains one or two valid node indices.
- Only positive finite normal modes are retained.

Null or numerically tiny capacitance directions are projected out before the
generalized eigenproblem is solved. Here "tiny" means a capacitance eigenvalue
no larger than `1e-12 * max(abs(eigenvalues(C_matrix)))`.

## Compatibility Notes

The refactored implementation keeps the older public attributes where practical:

- `C_inv_sqrt`: the inverse square root used by older code to form a
  transformed matrix.
- `dynamical_matrix`: the older transformed matrix
  `C_inv_sqrt @ L_inv_matrix @ C_inv_sqrt`.
- `eigensys_dynamical_matrix`: eigenvalues and vectors of that transformed
  matrix.
- `linear_modes`
- `linear_modes_GHz`
- `phase_zpf_list`

The first three compatibility attributes are computed only when accessed. The
frequencies, C-normalized mode vectors, phase ZPF values, and Hamiltonians are
computed from the generalized eigenproblem.

The correctly spelled `Ej_suppression_factor` should be preferred in new code.
The older misspelled `Ej_supression_factor` remains as an alias for existing
notebooks and scripts.
