# BBQ Workflow

`BBQ` turns capacitance and inverse-inductance matrices into the linear modes
needed to build a superconducting circuit model.

## Minimal Workflow

```python
import numpy as np
from sccircuits import BBQ

C_matrix = np.array([[40.0e-15]], dtype=float)
L_inv_matrix = np.array([[1.0 / 1.23e-9]], dtype=float)

bbq = BBQ(C_matrix, L_inv_matrix, non_linear_nodes=(0,))
print(bbq.linear_modes_GHz)
print(bbq.phase_zpf_list)
```

## Selecting Modes for a Hamiltonian

After solving the linear problem, pick the modes you want to keep and set a
Hilbert-space dimension for each one:

```python
bbq.selected_modes = [0]
bbq.dimensions = (10,)
H0 = bbq.hamiltonian_0()
Hnl = bbq.hamiltonian_nl(Ej=0.95, phi_ext=0.0)
```

That pattern is useful when you want to inspect the linear sector or build a
small nonlinear Hamiltonian directly from the BBQ object.

## Bridging into `Circuit`

When you want the chain-based model used throughout SCCircuits, feed the BBQ
outputs into `Circuit.from_harmonic_modes()`:

```python
from sccircuits import Circuit

circuit = Circuit.from_harmonic_modes(
    frequencies=bbq.linear_modes_GHz,
    phase_zpf=bbq.phase_zpf_list,
    dimensions=[18],
    Ej=0.95,
)
```

This is the typical workflow when:

- a GUI or symbolic tool gives you `C` and `L^{-1}`
- you want to fit spectroscopy in the chain basis
- you still want to retain a direct connection to the underlying circuit data

## Getting the Matrices

If you do not already have `C_matrix` and `L_inv_matrix`, use the
[BBQ Circuit Designer](bbq-circuit-designer.md) app. It exports NumPy/SymPy
snippets with `C_matrix_func()` and `L_inv_matrix_func()` helpers that can be
passed straight into `BBQ`.
