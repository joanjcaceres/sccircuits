"""Tests for pre-BBQ circuit matrix reductions."""

from __future__ import annotations

import numpy as np
import pytest

from sccircuits.reduction import (
    dynamic_coordinates_from_capacitance,
    reduce_frozen_coordinates,
)


def test_reduce_frozen_coordinates_matches_potential_schur_complement():
    capacitance_matrix = np.diag([2.0, 0.0, 3.0])
    inverse_inductance_matrix = np.array(
        [
            [6.0, 2.0, 1.0],
            [2.0, 4.0, -1.0],
            [1.0, -1.0, 5.0],
        ]
    )

    reduction = reduce_frozen_coordinates(
        capacitance_matrix,
        inverse_inductance_matrix,
    )

    expected_inverse_inductance = np.array(
        [
            [5.0, 1.5],
            [1.5, 4.75],
        ]
    )

    assert np.array_equal(
        dynamic_coordinates_from_capacitance(capacitance_matrix),
        [True, False, True],
    )
    assert np.array_equal(reduction.kept_indices, [0, 2])
    assert np.array_equal(reduction.eliminated_indices, [1])
    assert reduction.original_to_reduced == {0: 0, 2: 1}
    assert reduction.reduced_index(None) is None
    assert reduction.reduced_index(2) == 1
    with pytest.raises(ValueError, match="was eliminated"):
        reduction.reduced_index(1)

    assert np.allclose(reduction.capacitance_matrix, np.diag([2.0, 3.0]))
    assert np.allclose(
        reduction.inverse_inductance_matrix,
        expected_inverse_inductance,
    )
    assert np.allclose(
        reduction.reduced_to_original,
        [
            [1.0, 0.0],
            [-0.5, 0.25],
            [0.0, 1.0],
        ],
    )
    assert np.allclose(
        reduction.transform_branch_incidence([[-1.0, 1.0, 0.0]]),
        [[-1.5, 0.25]],
    )


def test_reduce_frozen_coordinates_rejects_frozen_capacitance_coupling():
    capacitance_matrix = np.array([[1.0, 0.2], [0.2, 0.0]])
    inverse_inductance_matrix = np.eye(2)

    with pytest.raises(ValueError, match="zero capacitance rows and columns"):
        reduce_frozen_coordinates(
            capacitance_matrix,
            inverse_inductance_matrix,
            dynamic_mask=np.array([True, False]),
        )


def test_reduce_frozen_coordinates_rejects_singular_frozen_potential_block():
    capacitance_matrix = np.diag([1.0, 0.0])
    inverse_inductance_matrix = np.zeros((2, 2))

    with pytest.raises(ValueError, match="positive definite"):
        reduce_frozen_coordinates(
            capacitance_matrix,
            inverse_inductance_matrix,
            dynamic_mask=np.array([True, False]),
        )


def test_reduce_frozen_coordinates_rejects_missing_dynamic_subspace():
    capacitance_matrix = np.zeros((1, 1))
    inverse_inductance_matrix = np.eye(1)

    with pytest.raises(ValueError, match="No dynamic coordinates"):
        reduce_frozen_coordinates(capacitance_matrix, inverse_inductance_matrix)
