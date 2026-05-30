# Matrix Reduction

SCCircuits provides small matrix-level reductions that can be applied before
constructing `BBQ`. These helpers are not a full graph quantization layer; they
operate on already assembled capacitance and inverse-inductance matrices.

## Frozen Coordinates

Use `reduce_frozen_coordinates` when a coordinate appears in the quadratic
potential but has no kinetic energy. In matrix terms, the coordinate has zero
rows and columns in `capacitance_matrix`, while its block in
`inverse_inductance_matrix` is invertible.

For coordinates split into dynamic `d` and frozen `f`,

```text
C = [[C_dd, 0],
     [0,    0]]
```

and

```text
L_inv = [[L_dd, L_df],
         [L_fd, L_ff]]
```

the helper minimizes the quadratic potential with respect to `f` and returns:

```text
C_eff = C_dd
L_inv_eff = L_dd - L_df L_ff^-1 L_fd
```

Example:

```python
import numpy as np
from sccircuits import BBQ, reduce_frozen_coordinates

C = np.diag([2.0, 0.0, 3.0])
L_inv = np.array(
    [
        [6.0, 2.0, 1.0],
        [2.0, 4.0, -1.0],
        [1.0, -1.0, 5.0],
    ]
)

reduction = reduce_frozen_coordinates(C, L_inv)
bbq = BBQ(reduction.capacitance_matrix, reduction.inverse_inductance_matrix)
```

The returned `CoordinateReduction` also records which original coordinates were
kept or eliminated and provides `reduced_index(...)` for remapping endpoints
that survive the reduction.

When a branch phase used an eliminated coordinate, simple endpoint remapping is
not enough. The eliminated coordinate is now a linear combination of the dynamic
coordinates. Use the stored transform to reduce branch-incidence rows:

```python
B_reduced = B_original @ reduction.reduced_to_original
```

or call:

```python
B_reduced = reduction.transform_branch_incidence(B_original)
```

## What This Does Not Do

This helper is distinct from:

- `BBQ`'s numerical projection of null or tiny capacitance directions, which is
  a subspace selection inside an already supplied matrix problem.
- Free or cyclic coordinate elimination, where a coordinate appears in the
  kinetic energy but not the potential. That later case uses the dual Schur
  complement of the capacitance matrix.
- Branch orientation and external flux handling. Those are graph-topology
  responsibilities and should be represented in graph/export metadata before
  matrix reduction.
