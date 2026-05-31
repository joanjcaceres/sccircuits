"""Numerical characterization tests for the BBQ class."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import e, hbar
from scipy.linalg import cosm, eigh
from scipy.sparse import diags

from sccircuits import BBQ


def _josephson_branch_record(
    *,
    edge_id=7,
    project_nodes=(101, 102),
    matrix_nodes=(0, None),
    phase_positive_index=0,
    phase_negative_index=None,
    phase_sign=1,
    L_j=7.0e-9,
    E_j_GHz=3.1,
):
    return {
        "edge_id": edge_id,
        "project_nodes": project_nodes,
        "matrix_nodes": matrix_nodes,
        "phase_positive_index": phase_positive_index,
        "phase_negative_index": phase_negative_index,
        "phase_sign": phase_sign,
        "inductance_expr": "Lj",
        "L_j": L_j,
        "E_j_GHz": E_j_GHz,
    }


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


def test_linear_circuit_can_omit_nonlinear_branches_and_junctions():
    capacitance_matrix = np.eye(2)
    inverse_inductance_matrix = np.diag([2.0, 3.0])

    bbq = BBQ(capacitance_matrix, inverse_inductance_matrix)

    assert bbq.nonlinear_branches == ()
    assert bbq.branch_phase_nodes == ()
    assert bbq.branch_incidence_matrix.shape == (0, 2)
    assert bbq.branch_phase_zpfs.shape == (0, 2)
    assert bbq.josephson_energies_ghz is None
    assert np.allclose(bbq.angular_frequencies_squared, [2.0, 3.0])


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


def test_si_scale_frozen_coordinate_uses_schur_reduction_and_reconstruction():
    capacitance = 2.0e-15
    capacitance_matrix = np.diag([capacitance, 0.0])
    inverse_inductance_matrix = np.array(
        [
            [6.0e9, 2.0e9],
            [2.0e9, 4.0e9],
        ]
    )

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=[(0,), (1,)],
    )

    expected_stiffness = 6.0e9 - (2.0e9 * 2.0e9 / 4.0e9)
    expected_omega_squared = expected_stiffness / capacitance
    expected_omega = np.sqrt(expected_omega_squared)

    assert np.allclose(bbq.angular_frequencies_squared, [expected_omega_squared])
    assert np.allclose(bbq.angular_frequencies, [expected_omega])
    assert bbq.normal_mode_vectors.shape == (2, 1)
    assert np.allclose(
        bbq.normal_mode_vectors[1, 0] / bbq.normal_mode_vectors[0, 0],
        -0.5,
    )
    assert np.allclose(
        np.abs(bbq.normal_mode_vectors[0, 0]),
        1.0 / np.sqrt(capacitance),
    )
    assert np.allclose(
        bbq.branch_phase_zpfs[1],
        -0.5 * bbq.branch_phase_zpfs[0],
    )


def test_frozen_coordinate_requires_positive_definite_stiffness_block():
    capacitance_matrix = np.diag([2.0e-15, 0.0])
    inverse_inductance_matrix = np.array(
        [
            [6.0e9, 0.0],
            [0.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="positive definite"):
        BBQ(
            capacitance_matrix,
            inverse_inductance_matrix,
            nonlinear_branches=(1,),
        )


def test_singular_inverse_inductance_drops_dc_mode():
    c0 = 2.0e-15
    c1 = 3.0e-15
    stiffness = 4.0e9
    capacitance_matrix = np.diag([c0, c1])
    inverse_inductance_matrix = stiffness * np.array(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
        ]
    )

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(0, 1),
    )

    expected_omega_squared = stiffness * (1.0 / c0 + 1.0 / c1)
    expected_omega = np.sqrt(expected_omega_squared)
    phi_0 = hbar / (2.0 * e)
    expected_branch_amplitude = np.sqrt((c0 + c1) / (c0 * c1))
    expected_zpf = (
        expected_branch_amplitude * np.sqrt(hbar / (2.0 * expected_omega)) / phi_0
    )

    assert bbq.angular_frequencies.shape == (1,)
    assert np.allclose(bbq.angular_frequencies_squared, [expected_omega_squared])
    assert np.allclose(
        bbq.normal_mode_vectors.T @ capacitance_matrix @ bbq.normal_mode_vectors,
        np.eye(1),
    )
    assert np.allclose(
        bbq.normal_mode_vectors.T
        @ inverse_inductance_matrix
        @ bbq.normal_mode_vectors,
        [[expected_omega_squared]],
    )
    assert np.allclose(np.abs(bbq.branch_phase_zpfs), [[expected_zpf]])


def test_exact_singular_capacitance_matrix_uses_positive_subspace():
    capacitance = 2.0e-15
    stiffness = 1.0 / 7.0e-9
    capacitance_matrix = capacitance * np.array([[1.0, -1.0], [-1.0, 1.0]])
    inverse_inductance_matrix = stiffness * np.eye(2)

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(0, 1),
    )

    positive_capacitance_basis = np.array([[1.0], [-1.0]]) / np.sqrt(2.0)
    c_reduced = (
        positive_capacitance_basis.T @ capacitance_matrix @ (positive_capacitance_basis)
    )
    l_inv_reduced = (
        positive_capacitance_basis.T
        @ inverse_inductance_matrix
        @ (positive_capacitance_basis)
    )
    expected_omega_squared = l_inv_reduced[0, 0] / c_reduced[0, 0]
    expected_omega = np.sqrt(expected_omega_squared)

    assert np.allclose(c_reduced, [[2.0 * capacitance]])
    assert np.allclose(l_inv_reduced, [[stiffness]])
    assert np.allclose(bbq.angular_frequencies_squared, [expected_omega_squared])
    assert np.allclose(bbq.angular_frequencies, [expected_omega])
    assert bbq.normal_mode_vectors.shape == (2, 1)
    assert np.allclose(bbq.normal_mode_vectors.sum(axis=0), [0.0])
    assert np.allclose(
        bbq.normal_mode_vectors.T @ capacitance_matrix @ bbq.normal_mode_vectors,
        np.eye(1),
    )

    phi_0 = hbar / (2.0 * e)
    expected_branch_mode_amplitude = 1.0 / np.sqrt(capacitance)
    expected_zpf = (
        expected_branch_mode_amplitude * np.sqrt(hbar / (2.0 * expected_omega)) / phi_0
    )

    assert bbq.branch_phase_zpfs.shape == (1, 1)
    assert np.allclose(np.abs(bbq.branch_phase_zpfs), [[expected_zpf]])


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
    assert bbq.branch_phase_nodes == ((1, 0), (2, None), (0, 1))
    assert bbq.branch_phase_zpfs.shape == (3, 3)
    assert np.allclose(bbq.branch_phase_zpfs, expected)
    assert np.allclose(
        bbq.branch_phase_zpfs[0, :],
        -bbq.branch_phase_zpfs[2, :],
    )


def test_junction_records_ground_jj_matches_single_branch_api():
    capacitance_matrix = np.array([[2.0e-15]])
    inverse_inductance_matrix = np.array([[1.0 / 7.0e-9]])
    branch_record = _josephson_branch_record(
        edge_id=11,
        matrix_nodes=[0, None],
        phase_positive_index=0,
        phase_negative_index=None,
        phase_sign=1,
        E_j_GHz=4.2,
    )

    direct = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(0,),
    )
    from_junctions = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        junctions=[branch_record],
    )

    assert np.allclose(
        from_junctions.angular_frequencies,
        direct.angular_frequencies,
    )
    assert np.allclose(
        from_junctions.branch_phase_zpfs,
        direct.branch_phase_zpfs,
    )
    assert from_junctions.branch_phase_nodes == ((0, None),)
    assert np.allclose(from_junctions.josephson_energies_ghz, [4.2])


def test_junction_records_ground_phase_reversal_flips_zpf_row():
    capacitance_matrix = np.array([[2.0e-15]])
    inverse_inductance_matrix = np.array([[1.0 / 7.0e-9]])

    forward = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        junctions=[
            _josephson_branch_record(
                matrix_nodes=(0, None),
                phase_positive_index=0,
                phase_negative_index=None,
                phase_sign=1,
            )
        ],
    )
    reverse = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        junctions=[
            _josephson_branch_record(
                matrix_nodes=(0, None),
                phase_positive_index=None,
                phase_negative_index=0,
                phase_sign=-1,
            )
        ],
    )

    assert np.allclose(forward.angular_frequencies, reverse.angular_frequencies)
    assert np.allclose(forward.branch_phase_zpfs, -reverse.branch_phase_zpfs)
    assert forward.branch_phase_nodes == ((0, None),)
    assert reverse.branch_phase_nodes == ((None, 0),)
    assert np.allclose(reverse.branch_incidence_matrix, [[-1.0]])


def test_junction_records_floating_jj_matches_branch_tuple_api():
    capacitance_matrix = np.array([[2.0, 0.2], [0.2, 1.5]]) * 1e-15
    inverse_inductance_matrix = np.array([[4.0, -1.0], [-1.0, 3.0]]) * 1e9
    branch_record = _josephson_branch_record(
        matrix_nodes=(0, 1),
        phase_positive_index=1,
        phase_negative_index=0,
        phase_sign=1,
    )

    direct = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(0, 1),
    )
    from_junctions = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        junctions=[branch_record],
    )

    assert np.allclose(
        from_junctions.angular_frequencies,
        direct.angular_frequencies,
    )
    assert np.allclose(
        from_junctions.branch_phase_zpfs,
        direct.branch_phase_zpfs,
    )


def test_junction_records_multiple_jjs_return_branch_rows_and_energies():
    capacitance_matrix = np.eye(3)
    inverse_inductance_matrix = np.diag([2.0, 3.0, 5.0])
    junction_records = [
        _josephson_branch_record(
            edge_id=7,
            project_nodes=(101, 102),
            matrix_nodes=(0, 1),
            phase_positive_index=1,
            phase_negative_index=0,
            phase_sign=1,
            E_j_GHz=1.3,
        ),
        _josephson_branch_record(
            edge_id=8,
            project_nodes=(103, None),
            matrix_nodes=(2, None),
            phase_positive_index=2,
            phase_negative_index=None,
            phase_sign=1,
            E_j_GHz=0.7,
        ),
    ]

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        junctions=junction_records,
    )

    assert bbq.branch_phase_zpfs.shape == (2, 3)
    assert bbq.branch_incidence_matrix.shape == (2, 3)
    assert bbq.branch_phase_nodes == ((1, 0), (2, None))
    assert bbq.josephson_energies_ghz is not None
    assert np.allclose(bbq.josephson_energies_ghz, [1.3, 0.7])


def test_junction_records_outputs_follow_input_row_order():
    capacitance_matrix = np.eye(3)
    inverse_inductance_matrix = np.diag([2.0, 3.0, 5.0])
    junction_records = [
        _josephson_branch_record(
            edge_id=8,
            matrix_nodes=(2, None),
            phase_positive_index=None,
            phase_negative_index=2,
            phase_sign=-1,
            E_j_GHz=0.7,
        ),
        _josephson_branch_record(
            edge_id=7,
            matrix_nodes=(0, 1),
            phase_positive_index=1,
            phase_negative_index=0,
            phase_sign=1,
            E_j_GHz=1.3,
        ),
    ]

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        junctions=junction_records,
    )

    assert bbq.branch_phase_nodes == ((None, 2), (1, 0))
    assert np.allclose(
        bbq.branch_incidence_matrix,
        [[0.0, 0.0, -1.0], [-1.0, 1.0, 0.0]],
    )
    assert np.allclose(bbq.josephson_energies_ghz, [0.7, 1.3])


def test_junction_records_phase_reversal_flips_only_selected_junction():
    capacitance_matrix = np.eye(2)
    inverse_inductance_matrix = np.diag([2.0, 3.0])
    forward = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        junctions=[
            _josephson_branch_record(
                matrix_nodes=(0, 1),
                phase_positive_index=1,
                phase_negative_index=0,
                phase_sign=1,
            ),
            _josephson_branch_record(
                edge_id=8,
                matrix_nodes=(1, None),
                phase_positive_index=1,
                phase_negative_index=None,
                phase_sign=1,
            ),
        ],
    )
    reverse_first = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        junctions=[
            _josephson_branch_record(
                matrix_nodes=(0, 1),
                phase_positive_index=0,
                phase_negative_index=1,
                phase_sign=-1,
            ),
            _josephson_branch_record(
                edge_id=8,
                matrix_nodes=(1, None),
                phase_positive_index=1,
                phase_negative_index=None,
                phase_sign=1,
            ),
        ],
    )

    assert np.allclose(
        forward.branch_phase_zpfs[0],
        -reverse_first.branch_phase_zpfs[0],
    )
    assert np.allclose(
        forward.branch_phase_zpfs[1],
        reverse_first.branch_phase_zpfs[1],
    )


def test_junctions_conflict_with_custom_nonlinear_branches():
    capacitance_matrix = np.eye(1)
    inverse_inductance_matrix = np.eye(1)

    with pytest.raises(
        ValueError,
        match="either junctions or nonlinear_branches",
    ):
        BBQ(
            capacitance_matrix,
            inverse_inductance_matrix,
            nonlinear_branches=(0,),
            junctions=[_josephson_branch_record()],
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


def test_hamiltonian_nonlinear_scalar_external_phase_matches_one_value_list():
    capacitance_matrix = np.array([[2.0, 0.2], [0.2, 1.5]]) * 1e-15
    inverse_inductance_matrix = np.array([[4.0, -1.0], [-1.0, 3.0]]) * 1e9

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=(0, 1),
    )
    bbq.selected_mode_indices = [0]
    bbq.truncation_dimensions = 5

    from_scalar = bbq.hamiltonian_nonlinear(
        josephson_energies=1.3,
        external_phases=0.27,
    )
    from_one_value_list = bbq.hamiltonian_nonlinear(
        josephson_energies=[1.3],
        external_phases=[0.27],
    )

    assert np.allclose(from_scalar, from_one_value_list)


def test_hamiltonian_nonlinear_rejects_wrong_length_external_phases():
    capacitance_matrix = np.eye(2)
    inverse_inductance_matrix = np.diag([2.0, 3.0])

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        nonlinear_branches=[(0,), (1,)],
    )
    bbq.selected_mode_indices = [0]
    bbq.truncation_dimensions = 4

    with pytest.raises(
        ValueError,
        match="external_phases must contain one value per nonlinear branch",
    ):
        bbq.hamiltonian_nonlinear(
            josephson_energies=[1.3, 0.7],
            external_phases=[0.27],
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


@pytest.mark.parametrize(
    "record_update,error_match",
    [
        ({"phase_positive_index": 2}, "phase_positive_index.*outside"),
        (
            {"phase_positive_index": None, "phase_negative_index": None},
            "one grounded side",
        ),
        ({"matrix_nodes": (0, 2)}, "matrix_nodes.*outside"),
        ({"matrix_nodes": (0, 0)}, "grounded side"),
    ],
)
def test_invalid_junction_records_raise_clear_value_errors(
    record_update,
    error_match,
):
    capacitance_matrix = np.eye(1)
    inverse_inductance_matrix = np.eye(1)
    branch_record = _josephson_branch_record()
    branch_record.update(record_update)

    with pytest.raises(ValueError, match=error_match):
        BBQ(
            capacitance_matrix,
            inverse_inductance_matrix,
            junctions=[branch_record],
        )


def test_junction_records_compute_josephson_energy_from_inductance():
    capacitance_matrix = np.eye(1)
    inverse_inductance_matrix = np.eye(1)
    branch_record = _josephson_branch_record()
    del branch_record["E_j_GHz"]

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        junctions=[branch_record],
    )
    josephson_inductance = float(branch_record["L_j"])
    expected = (hbar / (2.0 * e)) ** 2 / (
        josephson_inductance * 2.0 * np.pi * hbar * 1e9
    )

    assert np.allclose(bbq.josephson_energies_ghz, [expected])


def test_junction_records_can_omit_josephson_energies():
    capacitance_matrix = np.eye(1)
    inverse_inductance_matrix = np.eye(1)
    branch_record = _josephson_branch_record()
    del branch_record["E_j_GHz"]
    del branch_record["L_j"]

    bbq = BBQ(
        capacitance_matrix,
        inverse_inductance_matrix,
        junctions=[branch_record],
    )

    assert bbq.josephson_energies_ghz is None


def test_junction_records_reject_mixed_josephson_energy_availability():
    capacitance_matrix = np.eye(2)
    inverse_inductance_matrix = np.eye(2)
    with_energy = _josephson_branch_record(
        matrix_nodes=(0, None),
        phase_positive_index=0,
        phase_negative_index=None,
    )
    without_energy = _josephson_branch_record(
        matrix_nodes=(1, None),
        phase_positive_index=1,
        phase_negative_index=None,
    )
    del without_energy["E_j_GHz"]
    del without_energy["L_j"]

    with pytest.raises(ValueError, match="every Josephson junction record"):
        BBQ(
            capacitance_matrix,
            inverse_inductance_matrix,
            junctions=[with_energy, without_energy],
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
