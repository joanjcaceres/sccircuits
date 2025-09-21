"""Tests for the Circuit class second-harmonic implementation."""

import numpy as np
from scipy.linalg import cosm
from scipy.sparse import diags

from sccircuits import Circuit


def _phi_operator(dimension: int, phi_zpf: float) -> np.ndarray:
    """Construct the phase operator used in the nonlinear Hamiltonian."""
    data = np.sqrt(np.arange(1, dimension))
    return phi_zpf * diags([data, data], [1, -1]).toarray()


def test_single_mode_initialization_defaults():
    circuit = Circuit(
        non_linear_frequency=5.0,
        non_linear_phase_zpf=0.15,
        linear_frequencies=None,
        linear_couplings=None,
        dimensions=[8],
        Ej=1.2,
    )

    assert circuit.modes == 1
    assert circuit.linear_frequencies.size == 0
    assert circuit.linear_coupling.size == 0
    assert circuit.Ej_second == 0.0


def test_second_harmonic_hamiltonian_contribution():
    dimension = 6
    phi_zpf = 0.2
    phase_ext = 0.3
    Ej = 1.1
    Ej_second = 0.35
    freq = 5.4

    circuit = Circuit(
        non_linear_frequency=freq,
        non_linear_phase_zpf=phi_zpf,
        linear_frequencies=None,
        linear_couplings=None,
        dimensions=[dimension],
        Ej=Ej,
        Ej_second=Ej_second,
        phase_ext=phase_ext,
    )

    h_nl = circuit.hamiltonian_nl().toarray()

    diag = freq * (np.arange(dimension) + 0.5)
    phi_op = _phi_operator(dimension, phi_zpf)
    phi_shift = phi_op + phase_ext * np.eye(dimension)
    expected = np.diag(diag)
    expected -= Ej * cosm(phi_shift)
    expected -= Ej_second * cosm(2.0 * phi_shift)

    assert np.allclose(h_nl, expected)


def test_second_harmonic_default_matches_explicit_zero():
    base = Circuit(
        non_linear_frequency=4.8,
        non_linear_phase_zpf=0.12,
        linear_frequencies=None,
        linear_couplings=None,
        dimensions=[7],
        Ej=0.9,
    )

    explicit_zero = Circuit(
        non_linear_frequency=4.8,
        non_linear_phase_zpf=0.12,
        linear_frequencies=None,
        linear_couplings=None,
        dimensions=[7],
        Ej=0.9,
        Ej_second=0.0,
    )

    h_base = base.hamiltonian_nl().toarray()
    h_zero = explicit_zero.hamiltonian_nl().toarray()

    assert np.allclose(h_base, h_zero)


def test_gradient_names_include_second_harmonic():
    circuit = Circuit(
        non_linear_frequency=4.5,
        non_linear_phase_zpf=0.1,
        linear_frequencies=None,
        linear_couplings=None,
        dimensions=[6],
        Ej=0.8,
        Ej_second=0.2,
    )

    circuit.eigensystem(truncation=4)
    _, _, gradients, names = circuit.eigensystem_with_gradients(truncation=4)

    assert "Ej_second" in names
    idx = names.index("Ej_second")
    assert gradients.shape[1] > idx

