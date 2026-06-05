"""Smoke checks for an installed SCCircuits package.

This script is meant to run outside the repository root after
``python -m pip install .``. It intentionally avoids pytest so base install CI
does not need development dependencies.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(tempfile.gettempdir(), "sccircuits-cache"),
)
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "sccircuits-matplotlib"),
)

import sccircuits
from sccircuits import BBQ, Circuit


def run_bbq_smoke() -> None:
    capacitance = 2.0e-15
    inductance = 7.0e-9
    bbq = BBQ(
        capacitance_matrix=np.array([[capacitance]]),
        inverse_inductance_matrix=np.array([[1.0 / inductance]]),
        nonlinear_branches=(0,),
    )

    expected_omega = 1.0 / np.sqrt(inductance * capacitance)
    if not np.allclose(bbq.angular_frequencies, [expected_omega]):
        raise AssertionError("BBQ angular frequency smoke check failed.")
    if bbq.branch_phase_zpfs.shape != (1, 1):
        raise AssertionError("BBQ branch ZPF shape smoke check failed.")


def run_circuit_smoke() -> None:
    circuit = Circuit(
        non_linear_frequency=5.0,
        non_linear_phase_zpf=0.2,
        dimensions=[6],
        Ej=8.0,
    )
    energies, basis_vectors = circuit.eigensystem(truncation=4)

    if energies.shape != (4,):
        raise AssertionError("Circuit eigensystem energy shape smoke check failed.")
    if basis_vectors.shape != (6, 4):
        raise AssertionError("Circuit eigensystem basis shape smoke check failed.")
    if not np.all(np.diff(energies) >= 0.0):
        raise AssertionError("Circuit eigensystem energies are not sorted.")


def main() -> None:
    if sccircuits.get_version() != sccircuits.__version__:
        raise AssertionError("Package version helpers disagree.")

    run_bbq_smoke()
    run_circuit_smoke()


if __name__ == "__main__":
    main()
