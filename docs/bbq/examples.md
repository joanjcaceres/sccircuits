# BBQ Examples

This page collects examples that help researchers check the `BBQ` workflow from
matrix input to physical output.

For examples tied directly to automated tests, see
[BBQ Validation](validation.md).

## Example 1: Two-Node Lumped Circuit

This example is the same calculation used in the quickstart. It is useful as a
small regression check because the branch phase ZPF has a simple interpretation:
the differential mode across the selected branch has finite phase fluctuation,
while the common mode does not.

```python
import numpy as np
from sccircuits import BBQ

C = np.array(
    [
        [40e-15, -32.9e-15],
        [-32.9e-15, 40e-15],
    ]
)

L_inv = np.array(
    [
        [1 / 1.23e-9, 0.0],
        [0.0, 1 / 1.23e-9],
    ]
)

bbq = BBQ(C, L_inv, nonlinear_branches=(0, 1))

for mode, frequency in enumerate(bbq.frequencies_ghz):
    phase_zpf = bbq.branch_phase_zpfs[0, mode]
    print(f"mode {mode}: frequency={frequency:.8f} GHz, zpf={phase_zpf:.8f}")
```

Output:

```text
mode 0: frequency=16.80752677 GHz, zpf=0.35562862
mode 1: frequency=53.85653414 GHz, zpf=0.00000000
```

## Example 2: cQEDraw Export

The intended larger example is a cQEDraw project where the matrices are too
large or too tedious to write by hand, but easy to generate from the drawing.
That example should include:

- a downloadable cQEDraw project JSON file;
- the generated Python matrix snippet;
- the `BBQ(...)` call using the exported `junctions`;
- the resulting frequencies and branch phase ZPFs;
- a short explanation of which branch row corresponds to the Josephson junction
  of interest.

When the project file is ready, place it under a documentation asset directory
such as `docs/assets/cqedraw/` and link it from this page. The example should
keep the original cQEDraw junction records so readers can map each
`branch_phase_zpfs` row back to the corresponding drawn edge.

## What Good Examples Should Show

Each BBQ example should make these items explicit:

- the units of `capacitance_matrix` and `inverse_inductance_matrix`;
- the branch direction used for phase ZPFs;
- whether `junctions` or `nonlinear_branches` is used;
- the number of retained positive-frequency modes;
- the interpretation of the largest phase ZPF values;
- any zero modes, frozen coordinates, or removed singular directions.

The [Implementation Notes](implementation-notes.md) explain how those singular
directions are handled in the current code.
