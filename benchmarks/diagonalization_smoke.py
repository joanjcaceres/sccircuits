"""Small SCCircuits diagonalization benchmark.

Run from a source checkout or installed environment:

    python benchmarks/diagonalization_smoke.py

The numbers are not a formal benchmark suite. They are a compact sanity check
for the NumPy/SciPy diagonalization path used by BBQ and Circuit.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import scipy
from scipy.linalg import eigh

from sccircuits import BBQ, Circuit


def _time_call(label: str, callback, *, repeats: int = 3) -> None:
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        callback()
        durations.append(perf_counter() - start)

    best = min(durations)
    mean = sum(durations) / len(durations)
    print(f"{label}: best={best:.4f}s mean={mean:.4f}s repeats={repeats}")


def _chain_laplacian(size: int, diagonal: float, coupling: float) -> np.ndarray:
    matrix = np.diag(np.full(size, diagonal))
    off_diagonal = np.diag(np.full(size - 1, -coupling), k=1)
    return matrix + off_diagonal + off_diagonal.T


def run_dense_eigh() -> None:
    size = 96
    matrix = _chain_laplacian(size, diagonal=4.0, coupling=1.0)
    eigh(matrix, subset_by_index=(0, 15), check_finite=False)


def run_bbq() -> None:
    size = 24
    capacitance = _chain_laplacian(size, diagonal=50.0e-15, coupling=12.0e-15)
    inverse_inductance = _chain_laplacian(size, diagonal=8.0e8, coupling=2.0e8)
    BBQ(
        capacitance_matrix=capacitance,
        inverse_inductance_matrix=inverse_inductance,
        nonlinear_branches=(0, size - 1),
    )


def run_circuit_eigensystem() -> None:
    circuit = Circuit(
        non_linear_frequency=5.5,
        non_linear_phase_zpf=0.18,
        dimensions=[28, 18],
        Ej=9.0,
        linear_frequencies=[7.0],
        linear_couplings=[0.08],
    )
    circuit.eigensystem(truncation=12)


def main() -> None:
    print(f"NumPy {np.__version__}")
    print(f"SciPy {scipy.__version__}")
    _time_call("scipy.linalg.eigh subset", run_dense_eigh)
    _time_call("BBQ linear modes", run_bbq)
    _time_call("Circuit.eigensystem", run_circuit_eigensystem)


if __name__ == "__main__":
    main()
