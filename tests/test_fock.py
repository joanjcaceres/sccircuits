"""Tests for analytical operators projected onto Fock bases."""

import numpy as np
import pytest
from scipy.linalg import cosm
from scipy.sparse import diags

from sccircuits import (
    cosine_fock_matrix,
    cosine_fock_matrix_derivative,
    cosine_fock_product_matrix,
)


def _phase_operator(dimension: int, coefficient: float) -> np.ndarray:
    data = np.sqrt(np.arange(1, dimension))
    return coefficient * diags([data, data], [1, -1]).toarray()


@pytest.mark.parametrize("phase_offset", [0.0, 0.37, -0.81])
def test_cosine_fock_matrix_matches_large_basis_projection(phase_offset: float):
    dimension = 7
    reference_dimension = 48
    phase_zpf = 0.43
    reference = cosm(
        _phase_operator(reference_dimension, phase_zpf)
        + phase_offset * np.eye(reference_dimension)
    )[:dimension, :dimension]

    actual = cosine_fock_matrix(dimension, phase_zpf, phase_offset)

    assert actual.dtype == np.float64
    assert np.allclose(actual, actual.T)
    assert np.allclose(actual, reference, atol=1e-13)


def test_cosine_fock_matrix_handles_zero_phase_zpf():
    actual = cosine_fock_matrix(5, 0.0, 0.42)

    assert np.allclose(actual, np.cos(0.42) * np.eye(5))


def test_cosine_fock_matrix_derivative_matches_finite_difference():
    dimension = 8
    phase_zpf = 0.31
    phase_offset = -0.27
    step = 1e-6
    finite_difference = (
        cosine_fock_matrix(dimension, phase_zpf + step, phase_offset)
        - cosine_fock_matrix(dimension, phase_zpf - step, phase_offset)
    ) / (2.0 * step)

    actual = cosine_fock_matrix_derivative(
        dimension,
        phase_zpf,
        phase_offset,
    )

    assert np.allclose(actual, finite_difference, atol=2e-9)


@pytest.mark.parametrize("dimension", [0, -2, 3.5])
def test_cosine_fock_matrix_rejects_invalid_dimensions(dimension):
    with pytest.raises(ValueError, match="positive integer"):
        cosine_fock_matrix(dimension, 0.2)


def test_cosine_fock_matrix_rejects_negative_phase_zpf():
    with pytest.raises(ValueError, match="non-negative"):
        cosine_fock_matrix(5, -0.2)


def test_cosine_fock_product_matrix_matches_large_tensor_basis_projection():
    dimensions = (3, 4)
    reference_dimensions = (11, 12)
    coefficients = (-0.27, 0.19)
    phase_offset = 0.31

    reference_phase = np.kron(
        _phase_operator(reference_dimensions[0], coefficients[0]),
        np.eye(reference_dimensions[1]),
    ) + np.kron(
        np.eye(reference_dimensions[0]),
        _phase_operator(reference_dimensions[1], coefficients[1]),
    )
    reference_cosine = cosm(
        reference_phase
        + phase_offset * np.eye(np.prod(reference_dimensions, dtype=int))
    )
    retained_indices = np.array(
        [
            first * reference_dimensions[1] + second
            for first in range(dimensions[0])
            for second in range(dimensions[1])
        ]
    )
    projected_reference = reference_cosine[np.ix_(retained_indices, retained_indices)]

    actual = cosine_fock_product_matrix(
        dimensions,
        coefficients,
        phase_offset,
    )

    assert np.allclose(actual, actual.T)
    assert np.allclose(actual, projected_reference, atol=1e-13)
