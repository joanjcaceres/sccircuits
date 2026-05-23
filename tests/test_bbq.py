"""Numerical characterization tests for the BBQ class."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import e, hbar
from scipy.linalg import cosm, eigh
from scipy.sparse import diags

from sccircuits import BBQ


def test_single_mode_lc_frequency_and_units():
    capacitance = 2.0e-15
    inductance = 7.0e-9

    bbq = BBQ(
        capacitance_matrix=np.array([[capacitance]]),
        inverse_inductance_matrix=np.array([[1.0 / inductance]]),
        nonlinear_branches=(0,),
    )

    expected_omega = 1.0 / np.sqrt(inductance * capacitance)

    assert np.allclose(bbq.angular_frequencies, [expected_omega])
    assert np.allclose(
        bbq.frequencies_ghz,
        [expected_omega / (2.0 * np.pi * 1e9)],
    )

    phi_0 = hbar / (2.0 * e)
    expected_zpf = np.sqrt(hbar / (2.0 * expected_omega)) / (
        phi_0 * np.sqrt(capacitance)
    )
    assert bbq.branch_phase_zpfs.shape == (1, 1)
    assert np.allclose(np.abs(bbq.branch_phase_zpfs[0]), [expected_zpf])


def test_generalized_solver_matches_scipy_reference():
    capacitance_matrix = np.array([[2.0, 0.2], [0.2, 1.5]]) * 1e-15
    inverse_inductance_matrix = np.array([[4.0, -1.0], [-1.0, 3.0]]) * 1e9

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(0, 1),
    )
    expected_evals, _ = eigh(inverse_inductance_matrix, capacitance_matrix)

    assert np.allclose(bbq.angular_frequencies_squared, expected_evals)
    assert np.allclose(bbq.angular_frequencies, np.sqrt(expected_evals))
    assert np.allclose(
        (
            bbq.normal_mode_vectors.T
            @ capacitance_matrix
            @ bbq.normal_mode_vectors
        ),
        np.eye(2),
    )
    assert np.allclose(
        (
            bbq.normal_mode_vectors.T
            @ inverse_inductance_matrix
            @ bbq.normal_mode_vectors
        ),
        np.diag(bbq.angular_frequencies_squared),
        rtol=1e-12,
        atol=1e10,
    )


def test_tiny_capacitance_direction_is_excluded():
    capacitance_matrix = np.diag([2.0e-15, 1.0e-30])
    inverse_inductance_matrix = np.diag([1.0 / 7.0e-9, 1.0 / 5.0e-9])

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(0, 1),
    )

    assert bbq.angular_frequencies.shape == (1,)
    assert bbq.normal_mode_vectors.shape == (2, 1)


def test_positive_modes_do_not_require_legacy_sign_crossing():
    capacitance_matrix = np.eye(2)
    inverse_inductance_matrix = np.array([[3.0, 1.0], [1.0, 3.0]])

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(0, 1),
    )

    assert bbq.angular_frequencies.shape == (2,)
    assert np.allclose(bbq.angular_frequencies_squared, [2.0, 4.0])


def test_branch_reversal_flips_phase_zpf_only():
    capacitance_matrix = np.array([[2.0, 0.2], [0.2, 1.5]]) * 1e-15
    inverse_inductance_matrix = np.array([[4.0, -1.0], [-1.0, 3.0]]) * 1e9

    forward = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(0, 1),
    )
    reverse = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(1, 0),
    )

    assert np.allclose(
        forward.angular_frequencies,
        reverse.angular_frequencies,
    )
    assert np.allclose(forward.branch_phase_zpfs, -reverse.branch_phase_zpfs)


def test_multiple_nonlinear_branches_return_branch_by_mode_zpfs():
    capacitance_matrix = np.eye(3)
    inverse_inductance_matrix = np.diag([2.0, 3.0, 5.0])

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=[(0, 1), (2,), (1, 0)],
    )

    phi_0 = hbar / (2.0 * e)
    expected = (
        bbq.branch_incidence_matrix
        @ bbq.normal_mode_vectors
        * np.sqrt(hbar / (2.0 * bbq.angular_frequencies))[np.newaxis, :]
        / phi_0
    )

    assert bbq.nonlinear_branches == ((0, 1), (2,), (1, 0))
    assert bbq.branch_phase_zpfs.shape == (3, 3)
    assert np.allclose(bbq.branch_phase_zpfs, expected)
    assert np.allclose(
        bbq.branch_phase_zpfs[0, :],
        -bbq.branch_phase_zpfs[2, :],
    )


def test_plot_modes_uses_normalized_mode_indices(monkeypatch):
    capacitance_matrix = np.eye(3)
    inverse_inductance_matrix = np.diag([2.0, 3.0, 5.0])
    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(0, 1),
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    try:
        bbq.plot_modes(which=-1)
        legend_labels = [
            label.get_text()
            for label in plt.gca().get_legend().get_texts()
        ]
    finally:
        plt.close("all")

    assert legend_labels[0].startswith("$f_2")


def test_hamiltonian_linear_matches_harmonic_diagonal():
    capacitance_matrix = np.array([[2.0e-15]])
    inverse_inductance_matrix = np.array([[1.0 / 7.0e-9]])
    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(0,),
    )
    bbq.selected_mode_indices = [0]
    bbq.truncation_dimensions = 4

    expected = np.diag(bbq.frequencies_ghz[0] * (np.arange(4) + 0.5))

    assert np.allclose(bbq.hamiltonian_linear(), expected)


def test_branch_phase_operators_are_nested_by_branch_then_mode():
    capacitance_matrix = np.eye(2)
    inverse_inductance_matrix = np.diag([2.0, 3.0])

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(0, 1),
    )
    bbq.selected_mode_indices = [0]
    bbq.truncation_dimensions = 4

    phase_operators = bbq.branch_phase_operators

    assert len(phase_operators) == 1
    assert len(phase_operators[0]) == 1
    assert phase_operators[0][0].shape == (4, 4)


def test_hamiltonian_nonlinear_sums_multiple_nonlinear_branches():
    capacitance_matrix = np.eye(2)
    inverse_inductance_matrix = np.diag([2.0, 3.0])
    josephson_energies = np.array([1.3, 0.7])
    external_phases = np.array([0.27, -0.12])
    dimension = 4

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=[(0,), (1,)],
    )
    bbq.selected_mode_indices = [0]
    bbq.truncation_dimensions = dimension

    data = np.sqrt(np.arange(1, dimension))
    identity = np.eye(dimension)
    expected = np.zeros((dimension, dimension))
    suppression_factors = np.exp(
        -0.5 * np.sum(bbq.branch_phase_zpfs[:, [1]] ** 2, axis=1)
    )
    for branch_index, josephson_energy in enumerate(josephson_energies):
        phi_operator = (
            bbq.branch_phase_zpfs[branch_index, 0]
            * diags([data, data], [1, -1]).toarray()
        )
        expected += -josephson_energy * (
            suppression_factors[branch_index]
            * cosm(phi_operator + external_phases[branch_index] * identity)
            + 0.5 * phi_operator @ phi_operator
        )

    assert np.allclose(
        bbq.hamiltonian_nonlinear(
            josephson_energies=josephson_energies,
            external_phases=external_phases,
        ),
        expected,
    )


def test_unconfigured_hamiltonian_settings_raise_clear_errors():
    capacitance_matrix = np.array([[2.0e-15]])
    inverse_inductance_matrix = np.array([[1.0 / 7.0e-9]])
    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(0,),
    )

    with pytest.raises(ValueError, match="Set selected_mode_indices"):
        _ = bbq.selected_mode_indices

    with pytest.raises(ValueError, match="Set truncation_dimensions"):
        _ = bbq.truncation_dimensions


def test_selected_mode_indices_accepts_common_integer_sequences():
    capacitance_matrix = np.eye(2)
    inverse_inductance_matrix = np.diag([2.0, 3.0])
    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(0, 1),
    )

    bbq.selected_mode_indices = (0, 1)
    assert bbq.selected_mode_indices == [0, 1]

    bbq.selected_mode_indices = np.array([1])
    assert bbq.selected_mode_indices == [1]


def test_hamiltonian_nonlinear_matches_manual_matrix_cosine():
    capacitance_matrix = np.array([[2.0, 0.2], [0.2, 1.5]]) * 1e-15
    inverse_inductance_matrix = np.array([[4.0, -1.0], [-1.0, 3.0]]) * 1e9
    josephson_energy = 1.3
    external_phase = 0.27
    dimension = 5

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(0, 1),
    )
    bbq.selected_mode_indices = [0]
    bbq.truncation_dimensions = dimension

    data = np.sqrt(np.arange(1, dimension))
    phi_operator = (
        bbq.branch_phase_zpfs[0, 0]
        * diags([data, data], [1, -1]).toarray()
    )
    identity = np.eye(dimension)
    expected = -josephson_energy * (
        bbq.josephson_suppression_factors[0]
        * cosm(phi_operator + external_phase * identity)
        + 0.5 * phi_operator @ phi_operator
    )

    assert np.allclose(
        bbq.hamiltonian_nonlinear(
            josephson_energies=josephson_energy,
            external_phases=external_phase,
        ),
        expected,
    )


@pytest.mark.parametrize(
    (
        "capacitance_matrix,inverse_inductance_matrix,"
        "nonlinear_branches,error_match"
    ),
    [
        (
            np.ones((2, 3)),
            np.eye(2),
            (0, 1),
            "capacitance_matrix must be a square 2D matrix",
        ),
        (
            np.eye(2),
            np.eye(3),
            (0, 1),
            "same shape",
        ),
        (
            np.array([[1.0, 0.1], [0.2, 1.0]]),
            np.eye(2),
            (0, 1),
            "capacitance_matrix must be symmetric",
        ),
        (
            np.array([[1.0, np.inf], [np.inf, 1.0]]),
            np.eye(2),
            (0, 1),
            "finite",
        ),
        (
            np.eye(2),
            np.eye(2),
            (0, 2),
            "outside the circuit",
        ),
        (
            np.eye(2),
            np.eye(2),
            [(0, 1), (0, 1, 2)],
            "one or two node indices",
        ),
    ],
)
def test_invalid_inputs_raise_clear_value_errors(
    capacitance_matrix,
    inverse_inductance_matrix,
    nonlinear_branches,
    error_match,
):
    with pytest.raises(ValueError, match=error_match):
        BBQ(
            capacitance_matrix,
            inverse_inductance_matrix,
            nonlinear_branches=nonlinear_branches,
        )


def test_josephson_suppression_factors_returns_one_value_per_branch():
    capacitance_matrix = np.array([[2.0, 0.2], [0.2, 1.5]]) * 1e-15
    inverse_inductance_matrix = np.array([[4.0, -1.0], [-1.0, 3.0]]) * 1e9

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=[(0, 1), (1, 0)],
    )
    bbq.selected_mode_indices = [0]

    assert bbq.josephson_suppression_factors.shape == (2,)
    assert np.allclose(
        bbq.josephson_suppression_factors[0],
        bbq.josephson_suppression_factors[1],
    )
