# Iterative Diagonalization

`Circuit.eigensystem()` is implemented on top of
`IterativeHamiltonianDiagonalizer`. Most users will not instantiate the
diagonalizer directly, but understanding its role helps when choosing
truncations.

## What It Does

The diagonalizer builds the multimode Hamiltonian one subsystem at a time:

1. diagonalize the current truncated problem
2. keep only the lowest-energy states
3. attach the next mode
4. repeat

That matches the chain structure used by `Circuit`.

## Truncation Strategy

You can pass either:

- one integer used at every step
- a list with one truncation per step
- a callable that returns the truncation for a given mode index

Examples:

```python
evals, _ = circuit.eigensystem(truncation=20)
evals, _ = circuit.eigensystem(truncation=[24, 12, 8, 6])
```

## Standalone Use

The diagonalizer is public and can be used outside `Circuit` when you already
have mode Hamiltonians and coupling operators:

```python
import numpy as np
from scipy.sparse import diags

from sccircuits import IterativeHamiltonianDiagonalizer

diag = IterativeHamiltonianDiagonalizer([6, 4])
H0 = diags([5.0 * (np.arange(8) + 0.5)], [0]).toarray()
X0 = diags([np.sqrt(np.arange(1, 8))], [-1]).toarray()
diag.add_initial_mode(H0, X0)
```

For most package workflows, though, the higher-level `Circuit` wrapper is the
right interface.
