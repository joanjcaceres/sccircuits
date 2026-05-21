"""Numerical characterization tests for the BBQ class."""

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
        C_matrix=np.array([[capacitance]]),
        L_inv_matrix=np.array([[1.0 / inductance]]),
        non_linear_nodes=(0,),
    )

    expected_omega = 1.0 / np.sqrt(inductance * capacitance)

    assert np.allclose(bbq.linear_modes, [expected_omega])
    assert np.allclose(
        bbq.linear_modes_GHz,
        [expected_omega / (2.0 * np.pi * 1e9)],
    )

    phi_0 = hbar / (2.0 * e)
    expected_zpf = np.sqrt(hbar / (2.0 * expected_omega)) / (
        phi_0 * np.sqrt(capacitance)
    )
    assert np.allclose(np.abs(bbq.phase_zpf_list), [expected_zpf])


def test_generalized_solver_matches_scipy_reference():
    C_matrix = np.array([[2.0, 0.2], [0.2, 1.5]]) * 1e-15
    L_inv_matrix = np.array([[4.0, -1.0], [-1.0, 3.0]]) * 1e9

    bbq = BBQ(C_matrix, L_inv_matrix, non_linear_nodes=(0, 1))
    expected_evals, _ = eigh(L_inv_matrix, C_matrix)

    assert np.allclose(bbq.mode_eigenvalues, expected_evals)
    assert np.allclose(bbq.linear_modes, np.sqrt(expected_evals))
    assert np.allclose(
        bbq.mode_vectors.T @ C_matrix @ bbq.mode_vectors,
        np.eye(2),
    )
    assert np.allclose(
        bbq.mode_vectors.T @ L_inv_matrix @ bbq.mode_vectors,
        np.diag(bbq.mode_eigenvalues),
        rtol=1e-12,
        atol=1e10,
    )


def test_tiny_capacitance_direction_is_excluded():
    C_matrix = np.diag([2.0e-15, 1.0e-30])
    L_inv_matrix = np.diag([1.0 / 7.0e-9, 1.0 / 5.0e-9])

    bbq = BBQ(C_matrix, L_inv_matrix, non_linear_nodes=(0, 1))

    assert bbq.linear_modes.shape == (1,)
    assert bbq.mode_vectors.shape == (2, 1)


def test_positive_modes_do_not_require_legacy_sign_crossing():
    C_matrix = np.eye(2)
    L_inv_matrix = np.array([[3.0, 1.0], [1.0, 3.0]])

    bbq = BBQ(C_matrix, L_inv_matrix, non_linear_nodes=(0, 1))

    assert bbq.linear_modes.shape == (2,)
    assert np.allclose(bbq.mode_eigenvalues, [2.0, 4.0])


def test_branch_reversal_flips_phase_zpf_only():
    C_matrix = np.array([[2.0, 0.2], [0.2, 1.5]]) * 1e-15
    L_inv_matrix = np.array([[4.0, -1.0], [-1.0, 3.0]]) * 1e9

    forward = BBQ(C_matrix, L_inv_matrix, non_linear_nodes=(0, 1))
    reverse = BBQ(C_matrix, L_inv_matrix, non_linear_nodes=(1, 0))

    assert np.allclose(forward.linear_modes, reverse.linear_modes)
    assert np.allclose(forward.phase_zpf_list, -reverse.phase_zpf_list)


def test_hamiltonian_0_matches_harmonic_diagonal():
    C_matrix = np.array([[2.0e-15]])
    L_inv_matrix = np.array([[1.0 / 7.0e-9]])
    bbq = BBQ(C_matrix, L_inv_matrix, non_linear_nodes=(0,))
    bbq.selected_modes = [0]
    bbq.dimensions = 4

    expected = np.diag(bbq.linear_modes_GHz[0] * (np.arange(4) + 0.5))

    assert np.allclose(bbq.hamiltonian_0(), expected)


def test_hamiltonian_nl_matches_manual_matrix_cosine():
    C_matrix = np.array([[2.0, 0.2], [0.2, 1.5]]) * 1e-15
    L_inv_matrix = np.array([[4.0, -1.0], [-1.0, 3.0]]) * 1e9
    Ej = 1.3
    phi_ext = 0.27
    dimension = 5

    bbq = BBQ(C_matrix, L_inv_matrix, non_linear_nodes=(0, 1))
    bbq.selected_modes = [0]
    bbq.dimensions = dimension

    data = np.sqrt(np.arange(1, dimension))
    phi_operator = (
        bbq.phase_zpf_list[0] * diags([data, data], [1, -1]).toarray()
    )
    identity = np.eye(dimension)
    expected = -Ej * (
        bbq.Ej_suppression_factor * cosm(phi_operator + phi_ext * identity)
        + 0.5 * phi_operator @ phi_operator
    )

    assert np.allclose(bbq.hamiltonian_nl(Ej=Ej, phi_ext=phi_ext), expected)


@pytest.mark.parametrize(
    "C_matrix,L_inv_matrix,non_linear_nodes,error_match",
    [
        (
            np.ones((2, 3)),
            np.eye(2),
            (0, 1),
            "C_matrix must be a square 2D matrix",
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
            "C_matrix must be symmetric",
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
    ],
)
def test_invalid_inputs_raise_clear_value_errors(
    C_matrix, L_inv_matrix, non_linear_nodes, error_match
):
    with pytest.raises(ValueError, match=error_match):
        BBQ(C_matrix, L_inv_matrix, non_linear_nodes=non_linear_nodes)


def test_suppression_factor_alias_matches_correct_spelling():
    C_matrix = np.array([[2.0, 0.2], [0.2, 1.5]]) * 1e-15
    L_inv_matrix = np.array([[4.0, -1.0], [-1.0, 3.0]]) * 1e9

    bbq = BBQ(C_matrix, L_inv_matrix, non_linear_nodes=(0, 1))
    bbq.selected_modes = [0]

    assert bbq.Ej_suppression_factor == bbq.Ej_supression_factor
