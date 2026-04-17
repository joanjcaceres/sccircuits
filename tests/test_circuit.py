"""Unit tests for the Circuit class, including nonlinear and second-harmonic features."""

import numpy as np
from scipy.linalg import cosm
from scipy.sparse import diags

from sccircuits import Circuit


def _phi_operator(dimension: int, phi_zpf: float) -> np.ndarray:
    """Construct the phase operator used in the nonlinear Hamiltonian."""
    data = np.sqrt(np.arange(1, dimension))
    return phi_zpf * diags([data, data], [1, -1]).toarray()


def _dense(matrix_like) -> np.ndarray:
    """Convert sparse or dense inputs into a plain ndarray."""
    if hasattr(matrix_like, "toarray"):
        return matrix_like.toarray()
    if hasattr(matrix_like, "A"):
        return matrix_like.A
    return np.asarray(matrix_like)


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

    h_nl = _dense(circuit.hamiltonian_nl())

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

    h_base = _dense(base.hamiltonian_nl())
    h_zero = _dense(explicit_zero.hamiltonian_nl())

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


def test_harmonic_inputs_retrievable_after_factory_construction():
    frequencies = [5.1, 6.2, 8.3]
    phase_zpf = [0.12, 0.07, 0.03]
    dimensions = [20, 10, 5]

    circuit = Circuit.from_harmonic_modes(
        frequencies=frequencies,
        phase_zpf=phase_zpf,
        dimensions=dimensions,
        Ej=0.9,
        Ej_second=0.15,
    )

    assert np.allclose(circuit.frequencies, frequencies)
    assert np.allclose(circuit.phase_zpf, phase_zpf)

    modes = circuit.harmonic_modes()
    assert np.allclose(modes["frequencies"], frequencies)
    assert np.allclose(modes["phase_zpf"], phase_zpf)


def test_harmonic_modes_raises_when_not_available():
    circuit = Circuit(
        non_linear_frequency=4.0,
        non_linear_phase_zpf=0.1,
        linear_frequencies=None,
        linear_couplings=None,
        dimensions=[6],
        Ej=0.7,
    )

    with np.testing.assert_raises(AttributeError):
        circuit.harmonic_modes()


def test_dynamic_truncation_schedule_applies_per_mode():
    circuit = Circuit(
        non_linear_frequency=5.0,
        non_linear_phase_zpf=0.12,
        dimensions=[12, 6, 4],
        Ej=1.1,
        linear_frequencies=[6.5, 8.0],
        linear_couplings=[0.25, 0.1],
    )

    truncation_schedule = [8, 5, 3]
    energies, _ = circuit.eigensystem(truncation=truncation_schedule)

    assert energies.shape[0] == truncation_schedule[-1]


def test_bare_modes_reconstruct_harmonic_inputs_from_chain_matrix():
    frequencies = [5.1, 6.2, 8.3]
    phase_zpf = [0.12, 0.07, 0.03]

    circuit = Circuit.from_harmonic_modes(
        frequencies=frequencies,
        phase_zpf=phase_zpf,
        dimensions=[20, 10, 5],
        Ej=0.9,
    )

    bare = circuit.bare_modes

    assert np.allclose(bare["frequencies"], frequencies)
    assert np.allclose(bare["phase_zpf"], phase_zpf)


def test_from_star_basis_round_trips_star_representation():
    base = Circuit.from_harmonic_modes(
        frequencies=[5.1, 6.2, 8.3],
        phase_zpf=[0.12, 0.07, 0.03],
        dimensions=[20, 10, 5],
        Ej=0.9,
        Ej_second=0.1,
    )
    star = base.star_representation()

    reconstructed = Circuit.from_star_basis(
        non_linear_frequency=base.non_linear_frequency,
        non_linear_phase_zpf=base.non_linear_phase_zpf,
        linear_frequencies=star["linear_frequencies"],
        linear_couplings=star["linear_couplings"],
        dimensions=[20, 10, 5],
        Ej=0.9,
        Ej_second=0.1,
    )

    reconstructed_star = reconstructed.star_representation()

    assert np.allclose(reconstructed.linear_frequencies, base.linear_frequencies)
    assert np.allclose(reconstructed.linear_coupling, base.linear_coupling)
    assert np.allclose(
        reconstructed_star["linear_frequencies"], star["linear_frequencies"]
    )
    assert np.allclose(reconstructed_star["linear_couplings"], star["linear_couplings"])
    assert np.allclose(reconstructed.harmonic_modes()["frequencies"], base.frequencies)
    assert np.allclose(reconstructed.harmonic_modes()["phase_zpf"], base.phase_zpf)


def test_from_star_representation_alias_matches_from_star_basis():
    kwargs = {
        "non_linear_frequency": 5.0,
        "non_linear_phase_zpf": 0.15,
        "linear_frequencies": [6.0, 7.0],
        "linear_couplings": [0.25, 0.1],
        "dimensions": [12, 6, 4],
        "Ej": 0.8,
    }

    from_basis = Circuit.from_star_basis(**kwargs)
    from_representation = Circuit.from_star_representation(**kwargs)

    assert np.allclose(
        from_basis.linear_frequencies, from_representation.linear_frequencies
    )
    assert np.allclose(from_basis.linear_coupling, from_representation.linear_coupling)
