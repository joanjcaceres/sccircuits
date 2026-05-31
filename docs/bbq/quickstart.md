# BBQ Quickstart

This page shows a minimal `BBQ` calculation with explicit matrices. It is meant
to be small enough to copy into a Python session and large enough to show the
main public quantities.

## Install

For local development in this repository:

```bash
pixi run -e sccircuits install-dev
```

For a plain Python environment:

```bash
pip install -e ".[dev,interactive]"
```

## Define the Matrices

The example uses a two-node linearized lumped circuit. The capacitance matrix is
given in Farads and the inverse-inductance matrix is given in inverse Henries.

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
```

## Solve the Modes

Pass the matrices to `BBQ`. The nonlinear branch `(0, 1)` means that the branch
phase is the flux at node `1` minus the flux at node `0`.

```python
bbq = BBQ(
    capacitance_matrix,
    inverse_inductance_matrix,
    nonlinear_branches=(0, 1),
)

print("Frequencies in GHz:")
print(bbq.frequencies_ghz)

print("Branch phase ZPFs:")
print(bbq.branch_phase_zpfs)
```

Expected output:

```text
Frequencies in GHz:
[16.80752677 53.85653414]
Branch phase ZPFs:
[[0.35562862 0.        ]]
```

The first mode has a finite phase zero-point fluctuation across the selected
branch. The second mode is common to both branch endpoints in this example, so
the differential phase across `(0, 1)` is zero.

## Read the Main Results

- `frequencies_ghz`: ordinary frequencies in GHz.
- `angular_frequencies`: angular frequencies in rad/s.
- `normal_mode_vectors`: capacitance-normalized mode vectors in the original
  node basis.
- `branch_phase_zpfs`: one row per nonlinear branch and one column per mode.

## Build a Hamiltonian

After inspecting the linear modes, choose the modes and Hilbert-space
dimensions to keep:

```python
bbq.selected_mode_indices = [0]
bbq.truncation_dimensions = 40

H_linear = bbq.hamiltonian_linear()
H_nonlinear = bbq.hamiltonian_nonlinear(
    josephson_energies=7.5,
    external_phases=0.0,
)

H = H_linear + H_nonlinear
```

Hamiltonian matrices are returned in GHz. When comparing transition
frequencies, subtract the ground-state energy from the eigenvalue spectrum.

## Next Step

For cQEDraw-style Josephson junction records and larger examples, continue to
[BBQ Examples](examples.md).
